"""
Plotting helpers for 02_irrigation_detection.ipynb and 03_field_soil_moisture_map.ipynb.

Public API
----------
plot_field_buffer_map – basemap showing the field polygon and its buffer ring
plot_cluster_map      – scatter map of buffer pixels by reference/irrigated cluster
plot_sm_events        – field/reference SM chart with rain & irrigation markers
plot_pixel_sm_map     – per-pixel SM map for a single date, continuous color scale
build_sm_calendar     – interactive month calendar for picking an acquisition date
"""
from __future__ import annotations

import calendar as _calendar
import datetime as dt

import branca.colormap
import folium
import ipywidgets as widgets
import pandas as pd
import plotly.graph_objects as go

_ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Sequential ramp for soil moisture (a magnitude, not a polarity): one hue, light→dark,
# rather than a multi-hue "rainbow" scale — keeps the encoding monotone and
# colorblind-safe. Light = low SM, dark = high SM, matching "more water → deeper blue".
_SM_COLOR_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# branca's colormap legend renders on top of the raw satellite tiles with no background
# of its own, and its tick/caption text is black by default — both get lost over the
# darker parts of the imagery. Give the legend a translucent dark panel and white text
# so it stays legible regardless of what's underneath.
#
# The whole legend is scaled up (rather than bumping .legend text's font-size alone)
# because branca lays out tick labels and the caption at fixed pixel coordinates sized
# for its own default (smaller) font — enlarging just the font leaves the text too big
# for the gaps branca left for it, so ticks/caption overlap. A uniform transform grows
# text and spacing together, preserving the proportions branca computed.
_LEGEND_STYLE = """
<style>
.legend {
    background: rgba(13, 13, 12, 0.72);
    padding: 6px 12px 8px;
    border-radius: 6px;
    transform: scale(1.3);
    transform-origin: bottom right;
}
.legend text {
    fill: #ffffff;
}
</style>
"""


def _esri_map(center: list[float], zoom_start: int) -> folium.Map:
    return folium.Map(
        location=center, zoom_start=zoom_start,
        tiles=_ESRI_WORLD_IMAGERY, attr="Esri WorldImagery",
    )


def plot_field_buffer_map(
    field_geom,
    ring_geom,
    center: list[float],
    zoom_start: int = 14,
) -> folium.Map:
    """Basemap showing the field polygon and its surrounding buffer ring."""
    m = _esri_map(center, zoom_start)

    folium.GeoJson(
        field_geom.__geo_interface__,
        name="Field",
        style_function=lambda _: {"color": "#1565C0", "weight": 3, "fillOpacity": 0.40},
        tooltip="Field polygon",
    ).add_to(m)

    folium.GeoJson(
        ring_geom.__geo_interface__,
        name="Buffer ring",
        style_function=lambda _: {"color": "#BF360C", "weight": 2, "fillOpacity": 0.06},
        tooltip="Buffer ring (reference area)",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def plot_cluster_map(
    field_geom,
    features: pd.DataFrame,
    center: list[float],
    zoom_start: int = 14,
    sample_size: int = 2000,
    random_state: int = 42,
) -> folium.Map:
    """
    Scatter map of buffer pixels coloured by reference (non-irrigated) vs.
    potentially-irrigated cluster. Samples up to sample_size pixels for
    rendering performance.
    """
    sample = features.sample(min(sample_size, len(features)), random_state=random_state)
    ref_px = sample[sample["is_reference"]]
    irr_px = sample[~sample["is_reference"]]

    m = _esri_map(center, zoom_start)

    folium.GeoJson(
        field_geom.__geo_interface__,
        style_function=lambda _: {"color": "#1565C0", "weight": 3, "fillOpacity": 0.3},
    ).add_to(m)

    for _, row in ref_px.iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]], radius=3,
            color="#2E7D32", fill=True, fill_opacity=0.8,
            tooltip=f'reference  {row["mean_sm"]:.1f} vol%  {row["jump_count"]} jumps',
        ).add_to(m)

    for _, row in irr_px.iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]], radius=3,
            color="#E65100", fill=True, fill_opacity=0.8,
            tooltip=f'irrigated  {row["mean_sm"]:.1f} vol%  {row["jump_count"]} jumps',
        ).add_to(m)

    return m


def plot_pixel_sm_map(
    field_geom,
    pixel_df: pd.DataFrame,
    center: list[float],
    zoom_start: int = 17,
) -> folium.Map:
    """
    Per-pixel soil moisture map for a single acquisition date, zoomed to the field.

    The color scale auto-stretches to *this date's* own min/max rather than a fixed
    season-wide range: the field-wide spread on any given day is often just a couple
    of vol%, and pinning the scale to the full season would compress that into a
    near-uniform shade, hiding the spatial pattern the map exists to show. The
    trade-off is that color is no longer comparable across dates — the same shade of
    blue can mean a different absolute vol% on different days. This date's actual
    range is shown in the legend caption, and hovering a pixel gives its exact value.

    Each pixel is drawn as a circle on a single-hue sequential scale (light = low SM,
    dark blue = high SM) — soil moisture is a magnitude, not a two-sided quantity, so
    one hue keeps the encoding monotone and unambiguous rather than cycling through
    unrelated colors.

    Parameters
    ----------
    field_geom : Shapely geometry — field boundary, drawn as an outline
    pixel_df   : rows for a single datetime only (columns: lon, lat, sm)
    center     : [lat, lon] map center
    """
    lo, hi = float(pixel_df["sm"].min()), float(pixel_df["sm"].max())
    if lo == hi:
        vmin, vmax = lo - 0.5, hi + 0.5
    else:
        # Small padding on each end so the most extreme pixel isn't pinned to the very
        # tip of the ramp, which reads as clipped/saturated.
        pad = (hi - lo) * 0.08
        vmin, vmax = lo - pad, hi + pad

    colormap = branca.colormap.LinearColormap(colors=_SM_COLOR_RAMP, vmin=vmin, vmax=vmax)
    colormap.caption = f"Surface Soil Moisture (vol%) — this date: {lo:.1f}–{hi:.1f}"

    m = _esri_map(center, zoom_start)
    m.get_root().header.add_child(folium.Element(_LEGEND_STYLE))

    folium.GeoJson(
        field_geom.__geo_interface__,
        style_function=lambda _: {"color": "#FFFFFF", "weight": 2, "fillOpacity": 0},
        tooltip="Field boundary",
    ).add_to(m)

    # A thin white ring (rather than a dark one) keeps each pixel legible against the
    # satellite basemap regardless of the pixel's own color — the lightest ramp steps
    # would otherwise disappear into bright soil/imagery with a dark outline instead.
    for _, row in pixel_df.iterrows():
        folium.CircleMarker(
            [row["lat"], row["lon"]], radius=7,
            color="#ffffff", weight=1, opacity=0.9,
            fill=True, fill_color=colormap(row["sm"]), fill_opacity=0.95,
            tooltip=f'{row["sm"]:.1f} vol%',
        ).add_to(m)

    colormap.add_to(m)
    return m


_WEEKDAY_LABELS = {"mon": ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"],
                    "sun": ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]}

_DAY_CELL_PX = 34

# ipywidgets' default button padding (comfortably sized for normal button labels) eats
# most of a ~34px-wide day cell, so a 2-digit day number gets ellipsis-truncated to
# "1…" etc. Day buttons get this scoped class so their padding can be stripped without
# touching button styling anywhere else in the notebook.
_CALENDAR_STYLE = """
<style>
.sm-cal-day {
    padding: 0 !important;
    min-width: 0 !important;
    font-size: 12px !important;
    line-height: 1 !important;
}
</style>
"""


def _readable_text_color(hex_color: str) -> str:
    """Black or white text, whichever reads better on the given hex fill color."""
    h = hex_color.lstrip("#")[:6]
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1a1a1a" if luminance > 140 else "#ffffff"


def _add_months(month_start: dt.date, delta: int) -> dt.date:
    total = (month_start.year * 12 + (month_start.month - 1)) + delta
    year, month = divmod(total, 12)
    return dt.date(year, month + 1, 1)


def build_sm_calendar(
    field_pixels: pd.DataFrame,
    sm_min: float,
    sm_max: float,
    initial_date,
    on_select,
    week_start: str = "mon",
    n_months: int = 3,
) -> widgets.VBox:
    """
    Interactive month calendar for picking a soil-moisture acquisition date.

    Shows n_months side by side, centered on the currently selected month (one month
    of context on each side by default) — enough screen space to browse without
    paging for every neighboring month. Each day is a button colored by that date's
    field-mean SM, on the same single-hue ramp as plot_pixel_sm_map — but here the
    color scale is pinned to a fixed (sm_min, sm_max) rather than auto-stretching per
    date, so a shade means the same vol% on every month; a season overview needs
    colors comparable across months, unlike the per-pixel map's single-date view.
    Days with no acquisition are grayed out and disabled.

    Prev/Next slide the whole window by one month (rather than jumping by n_months),
    clamped so the *center* month never leaves the range of months that actually
    contain data — the outer two months can still render as all-gray context past
    that edge. Clicking an available day calls on_select(acquisition_datetime) and
    only updates which day is highlighted, not which months are in view.

    Parameters
    ----------
    field_pixels : long DataFrame with 'datetime' and 'sm' columns (as produced by
                   load_pixel_timeseries) — grouped here by calendar day, using each
                   day's mean sm for color and its latest acquisition datetime as the
                   representative timestamp passed to on_select
    sm_min, sm_max : fixed color-scale bounds (e.g. the season's overall sm range)
    initial_date   : datetime already selected when the calendar is first built —
                      determines the centered month and which day is highlighted
    on_select      : callback invoked with the acquisition datetime (pd.Timestamp)
                      when the user clicks an available day
    week_start     : 'mon' or 'sun'
    n_months       : how many consecutive months to show side by side
    """
    daily = field_pixels.assign(day=field_pixels["datetime"].dt.date).groupby("day")
    day_data: dict[dt.date, tuple[pd.Timestamp, float]] = {
        day: (grp["datetime"].max(), float(grp["sm"].mean())) for day, grp in daily
    }
    months_with_data = sorted({d.replace(day=1) for d in day_data})

    colormap = branca.colormap.LinearColormap(colors=_SM_COLOR_RAMP, vmin=sm_min, vmax=sm_max)
    weekday_labels = _WEEKDAY_LABELS[week_start]

    initial_date = pd.Timestamp(initial_date)
    state = {"center": initial_date.date().replace(day=1), "selected": initial_date.date()}

    left_offset = n_months // 2
    title_labels = [widgets.HTML() for _ in range(n_months)]
    grids = [
        widgets.GridBox(layout=widgets.Layout(
            grid_template_columns=f"repeat(7, {_DAY_CELL_PX}px)", grid_gap="2px",
        ))
        for _ in range(n_months)
    ]
    month_blocks = [
        widgets.VBox([title_labels[i], grids[i]],
                      layout=widgets.Layout(
                          width=f"{7 * _DAY_CELL_PX + 6 * 2}px",
                          margin="0 14px 0 0" if i < n_months - 1 else "0",
                      ))
        for i in range(n_months)
    ]
    prev_btn = widgets.Button(description="◀", layout=widgets.Layout(width="32px"))
    next_btn = widgets.Button(description="▶", layout=widgets.Layout(width="32px"))

    def _make_day_button(cell_date: dt.date) -> widgets.Button:
        btn = widgets.Button(
            description=str(cell_date.day),
            layout=widgets.Layout(width=f"{_DAY_CELL_PX}px", height="28px", margin="0"),
        )
        btn.add_class("sm-cal-day")
        info = day_data.get(cell_date)
        if info is None:
            btn.disabled = True
            btn.style.button_color = "#3a3a3a"
            btn.style.text_color = "#8a8a8a"
            return btn

        acquisition_dt, mean_sm = info
        fill = colormap(mean_sm)
        btn.style.button_color = fill
        btn.style.text_color = _readable_text_color(fill)
        btn.tooltip = f"{cell_date.isoformat()} — {mean_sm:.1f} vol%"
        btn.layout.border = (
            "3px solid #FFD54F" if cell_date == state["selected"] else "1px solid rgba(0,0,0,0.15)"
        )

        def _on_click(_btn, d=cell_date, acq=acquisition_dt):
            state["selected"] = d
            _render_all()
            on_select(acq)

        btn.on_click(_on_click)
        return btn

    def _render_one_month(idx: int, month_start: dt.date):
        n_days = _calendar.monthrange(month_start.year, month_start.month)[1]
        first_weekday = dt.date(month_start.year, month_start.month, 1).weekday()
        if week_start == "sun":
            first_weekday = (first_weekday + 1) % 7

        cells = [
            widgets.HTML(f"<div style='width:{_DAY_CELL_PX}px;text-align:center;"
                         f"font-size:11px;color:#9aa0a6;'>{lbl}</div>")
            for lbl in weekday_labels
        ]
        cells += [widgets.HTML(f"<div style='width:{_DAY_CELL_PX}px;height:28px;'></div>")
                  for _ in range(first_weekday)]
        cells += [_make_day_button(dt.date(month_start.year, month_start.month, d))
                  for d in range(1, n_days + 1)]

        grids[idx].children = tuple(cells)
        title_labels[idx].value = (
            f"<div style='text-align:center;font-weight:600;font-size:13px;"
            f"padding:2px 0;'>{month_start.strftime('%B %Y')}</div>"
        )

    def _render_all():
        for i in range(n_months):
            _render_one_month(i, _add_months(state["center"], i - left_offset))
        prev_btn.disabled = state["center"] <= months_with_data[0]
        next_btn.disabled = state["center"] >= months_with_data[-1]

    def _go_prev(_):
        state["center"] = _add_months(state["center"], -1)
        _render_all()

    def _go_next(_):
        state["center"] = _add_months(state["center"], 1)
        _render_all()

    prev_btn.on_click(_go_prev)
    next_btn.on_click(_go_next)
    _render_all()

    months_row = widgets.HBox(month_blocks)
    nav = widgets.HBox(
        [prev_btn, months_row, next_btn],
        layout=widgets.Layout(align_items="center"),
    )
    return widgets.VBox([widgets.HTML(_CALENDAR_STYLE), nav])


def plot_sm_events(
    field_ts: pd.Series,
    field_sm_std: pd.Series,
    ref_ts: pd.Series,
    ref_sm_std: pd.Series,
    rain_ev: pd.DataFrame,
    irr_ev: pd.DataFrame,
    title: str = "Irrigation & Rain Event Detection",
) -> go.Figure:
    """
    Build the field/reference soil-moisture chart with detected events overlaid.

    - Field SM: markers only (irregular satellite overpasses, not daily samples —
      a connecting line would imply unobserved data), with ±1 std error bars
      across field pixels.
    - Reference SM: a smoothed trend line only (no raw markers), since it's a
      derived rolling-mean series rather than raw observations, plus a shaded
      ±1 std band.
    - Rain and irrigation events: markers at the field SM value, sized and made
      more opaque with higher detection confidence.

    Parameters
    ----------
    field_ts, field_sm_std : Series(datetime → value) — field mean SM and its std
    ref_ts, ref_sm_std      : Series(datetime → value) — smoothed reference SM and its std
    rain_ev, irr_ev         : subsets of the events DataFrame for each event type
                              (columns: datetime, field_sm, field_delta, confidence,
                              buffer_fraction)
    title                   : chart title

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=field_ts.index, y=field_ts,
        name="Field mean SM",
        mode="markers",
        marker=dict(size=6, color="#1565C0"),
        error_y=dict(type="data", array=field_sm_std, visible=True,
                     color="rgba(21,101,192,0.4)", thickness=1, width=3),
    ))

    # Reference SM ±1 std band — a derived, continuous statistic (unlike raw
    # per-date values), so a filled area between the smoothed line's bounds is
    # appropriate even though the underlying acquisitions are irregular.
    fig.add_trace(go.Scatter(
        x=ref_ts.index, y=ref_ts + ref_sm_std,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ref_ts.index, y=ref_ts - ref_sm_std,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(46,125,50,0.15)",
        name="Reference SM ±1 std", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ref_ts.index, y=ref_ts,
        name="Reference SM (smoothed)",
        mode="lines",
        line=dict(color="#2E7D32", width=2),
    ))

    if not rain_ev.empty:
        fig.add_trace(go.Scatter(
            x=rain_ev["datetime"], y=rain_ev["field_sm"],
            name="Rain event",
            mode="markers",
            marker=dict(
                symbol="circle-open", color="#1976D2", line=dict(width=2),
                size=8 + 14 * rain_ev["confidence"],
                opacity=0.3 + 0.7 * rain_ev["confidence"],
            ),
            customdata=rain_ev[["confidence", "buffer_fraction"]].values,
            hovertemplate=(
                "Rain<br>confidence: %{customdata[0]:.2f}"
                "<br>buffer fraction: %{customdata[1]:.0%}<extra></extra>"
            ),
        ))

    if not irr_ev.empty:
        fig.add_trace(go.Scatter(
            x=irr_ev["datetime"], y=irr_ev["field_sm"],
            name="Irrigation event",
            mode="markers",
            marker=dict(
                symbol="triangle-up", color="#E65100",
                size=8 + 14 * irr_ev["confidence"],
                opacity=0.3 + 0.7 * irr_ev["confidence"],
            ),
            customdata=irr_ev[["confidence", "field_delta"]].values,
            hovertemplate=(
                "Irrigation<br>confidence: %{customdata[0]:.2f}"
                "<br>ΔSM: %{customdata[1]:+.1f} vol%<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=f"{title}"
                 "<br><sup>Marker size/opacity scale with confidence; error bars/band show ±1 std</sup>",
        ),
        xaxis_title="Date",
        yaxis_title="Surface Soil Moisture (vol%)",
        template="plotly_white",
        height=550,
        hovermode="x unified",
    )
    return fig
