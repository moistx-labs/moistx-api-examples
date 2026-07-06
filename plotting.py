"""
Plotting helpers for 02_irrigation_detection.ipynb.

Public API
----------
plot_field_buffer_map – basemap showing the field polygon and its buffer ring
plot_cluster_map      – scatter map of buffer pixels by reference/irrigated cluster
plot_sm_events        – field/reference SM chart with rain & irrigation markers
"""
from __future__ import annotations

import folium
import pandas as pd
import plotly.graph_objects as go

_ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


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
