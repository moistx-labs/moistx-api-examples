"""
MoistX irrigation detection helpers.

Public API
----------
compute_ring_buffer   – auto-generate a ring buffer polygon around a field
download_sm_files     – download & cache GeoTIFFs from the MoistX file endpoint
load_pixel_timeseries – extract per-pixel SM values from a list of GeoTIFFs
compute_pixel_features – compute temporal features (mean, std, jump count) per pixel
find_reference_pixels  – K-means to identify non-irrigated reference pixels
compute_jump_fractions – per-acquisition fraction of buffer pixels that jumped
time_centered_smooth   – rolling mean centered in calendar time, not acquisition count
detect_events          – classify SM rises as rain or irrigation
summarise_events       – print a plain-English summary and return the event table
"""
from __future__ import annotations

import hashlib
import os
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio
import rasterio.transform
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_geom
from shapely.geometry import mapping
from shapely.wkt import loads as load_wkt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

_NODATA   = 65535
_SM_SCALE = 10.0


# ── Internal ──────────────────────────────────────────────────────────────────

def _parse_filename_meta(filepath: str) -> tuple[pd.Timestamp | None, str | None]:
    """Return (datetime, source) parsed from a MoistX GeoTIFF filename."""
    stem  = Path(filepath).stem
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if len(p) == 16 and "T" in p and p.endswith("Z"):
            try:
                dt = pd.to_datetime(p, format="%Y%m%dT%H%M%SZ")
            except Exception:
                return None, None
            source_parts = [x for x in parts[i + 1:] if x not in ("SM", "5cm", "cropped")]
            return dt, "_".join(source_parts) if source_parts else None
    return None, None


# ── Public API ────────────────────────────────────────────────────────────────

def compute_ring_buffer(
    field_wkt: str,
    buffer_m: float = 500.0,
) -> tuple:
    """
    Expand the field polygon outward by buffer_m metres, then subtract the field
    to produce a ring-shaped buffer.

    Returns (field_geom, ring_geom, ring_wkt).
    Warns if the ring area approaches the 1,000 ha API limit.
    """
    field_geom = load_wkt(field_wkt)
    gdf        = gpd.GeoDataFrame(geometry=[field_geom], crs="EPSG:4326")
    utm        = gdf.estimate_utm_crs()
    gdf_utm    = gdf.to_crs(utm)

    outer_utm = gdf_utm.geometry.iloc[0].buffer(buffer_m)
    ring_utm  = outer_utm.difference(gdf_utm.geometry.iloc[0])

    ring_gdf  = gpd.GeoDataFrame(geometry=[ring_utm], crs=utm).to_crs("EPSG:4326")
    ring_geom = ring_gdf.geometry.iloc[0]
    ring_ha   = gpd.GeoDataFrame(geometry=[ring_utm], crs=utm).area.iloc[0] / 10_000

    if ring_ha > 900:
        print(
            f"Warning: buffer ring is {ring_ha:.0f} ha — close to the 1,000 ha API limit. "
            "Consider reducing BUFFER_M."
        )
    else:
        print(f"Buffer ring: {ring_ha:.0f} ha")

    return field_geom, ring_geom, ring_geom.wkt


def download_sm_files(
    wkt: str,
    start: str,
    end: str,
    cache_dir: str,
    api_key: str,
    base_url: str = "https://moistx.com/api",
    label: str = "",
) -> list[str]:
    """
    Download a ZIP of clipped GeoTIFFs from the MoistX file endpoint and
    unpack them into a local cache directory.

    The cache key is an MD5 of (wkt + start + end), so re-running with identical
    parameters reads from disk without hitting the API.

    Returns a sorted list of .tif file paths.
    """
    cache_key  = hashlib.md5(f"{wkt}{start}{end}".encode()).hexdigest()[:12]
    cache_path = Path(cache_dir) / cache_key
    existing   = sorted(cache_path.glob("*.tif"))

    if existing:
        print(f"[{label or 'cache'}] {len(existing)} files already cached → {cache_path}")
        return [str(f) for f in existing]

    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"[{label or 'download'}] Fetching {start[:10]} → {end[:10]} …")

    resp = requests.post(
        f"{base_url}/get/file/soil-moisture",
        headers={"X-Api-Key": api_key},
        params={
            "coordinate_reference_system": "EPSG:4326",
            "start_datetime":              start,
            "end_datetime":                end,
        },
        json={"wkt": wkt},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    zip_path = cache_path / "data.zip"
    with open(zip_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_path)
    zip_path.unlink()

    tif_files = sorted(cache_path.glob("*.tif"))
    print(f"[{label}] {len(tif_files)} GeoTIFFs downloaded and cached")
    return [str(f) for f in tif_files]


def load_pixel_timeseries(
    tif_paths: list[str],
    geom,
) -> pd.DataFrame:
    """
    Extract per-pixel SM values from a list of (already clipped) GeoTIFFs.

    Parameters
    ----------
    tif_paths : list of local .tif file paths
    geom      : Shapely geometry — pixels outside this geometry are skipped

    Returns DataFrame: pixel_id, lon, lat, datetime, sm (vol%), source
    """
    rows = []
    n    = len(tif_paths)

    for idx, fpath in enumerate(tif_paths):
        dt, source = _parse_filename_meta(fpath)
        if dt is None:
            continue

        try:
            with rasterio.open(fpath) as src:
                geom_in_crs = transform_geom("EPSG:4326", src.crs, mapping(geom))
                nodata_val  = src.nodata if src.nodata is not None else _NODATA

                out_image, out_transform = rasterio_mask(
                    src, [geom_in_crs], crop=True, nodata=nodata_val, all_touched=False
                )
                data  = out_image[0]
                valid = data != _NODATA
                if src.nodata is not None:
                    valid &= data != src.nodata

                rs, cs = np.where(valid)
                if len(rs) == 0:
                    continue

                # Batch coordinate transform: raster CRS → WGS84
                xs, ys   = rasterio.transform.xy(out_transform, rs, cs)
                lons, lats = warp_transform(src.crs, "EPSG:4326", list(xs), list(ys))
                sm_vals  = data[rs, cs] / _SM_SCALE

                for lon, lat, sm in zip(lons, lats, sm_vals):
                    rows.append({
                        "pixel_id": f"{lon:.5f},{lat:.5f}",
                        "lon":      lon,
                        "lat":      lat,
                        "datetime": dt,
                        "sm":       sm,
                        "source":   source,
                    })

        except Exception as e:
            print(f"  Warning — skipping {Path(fpath).name}: {e}")

        if (idx + 1) % 10 == 0 or idx == n - 1:
            print(f"  Loaded {idx + 1}/{n} files …", end="\r")

    print()
    if not rows:
        return pd.DataFrame(columns=["pixel_id", "lon", "lat", "datetime", "sm", "source"])
    return pd.DataFrame(rows)


def compute_pixel_features(
    pixel_df: pd.DataFrame,
    jump_thr: float = 3.0,
    min_obs:  int   = 3,
    jump_fractions: pd.Series | None = None,
    rain_fraction:  float = 0.90,
) -> pd.DataFrame:
    """
    Compute temporal features per pixel for clustering.

    Features: mean_sm, sm_std, jump_count, n_obs.
    jump_count only counts *local* jumps (SM rises > jump_thr) on dates that are not
    flagged as buffer-wide rain (jump_fractions >= rain_fraction). Excluding rain dates
    keeps a pixel's jump_count reflecting irrigation-like activity rather than being
    inflated by rain events shared across the whole buffer.

    Pixels with fewer than min_obs observations are excluded.

    Returns DataFrame: pixel_id, lon, lat, mean_sm, sm_std, jump_count, n_obs
    """
    rain_dates = set()
    if jump_fractions is not None and not jump_fractions.empty:
        rain_dates = set(jump_fractions.index[jump_fractions >= rain_fraction])

    def _features(grp: pd.DataFrame) -> pd.Series:
        ts         = grp.sort_values("datetime")
        sm         = ts["sm"]
        deltas     = sm.diff()
        local_jump = (deltas > jump_thr) & (~ts["datetime"].isin(rain_dates))
        return pd.Series({
            "mean_sm":    round(float(sm.mean()), 2),
            "sm_std":     round(float(sm.std(ddof=1)) if len(sm) > 1 else 0.0, 2),
            "jump_count": int(local_jump.sum()),
            "n_obs":      len(sm),
            "lon":        float(grp["lon"].iloc[0]),
            "lat":        float(grp["lat"].iloc[0]),
        })

    feats = (
        pixel_df.groupby("pixel_id", group_keys=False)
        .apply(_features, include_groups=False)
        .reset_index()
    )
    feats = feats[feats["n_obs"] >= min_obs].reset_index(drop=True)
    print(f"Pixel features computed: {len(feats)} pixels with ≥{min_obs} observations")
    return feats


def find_reference_pixels(
    features_df: pd.DataFrame,
    k: int = 2,
    ref_jump_percentile: float = 0.25,
) -> tuple[pd.DataFrame, frozenset]:
    """
    K-means clustering on (mean_sm, jump_count, sm_std) splits buffer pixels into a
    low-SM and a high-SM group. The low-SM group is the reference candidate, but only
    its calmest pixels are kept as reference: those whose local jump_count falls at or
    below the ref_jump_percentile of that cluster's own jump_count distribution.

    A percentile (rather than an absolute jump count) is used because even non-irrigated
    pixels commonly show a handful of jumps over a season from retrieval noise or
    patchy sub-threshold wetting — an absolute cutoff like "zero jumps ever" can be
    unsatisfiable and would silently empty the reference set. Taking the bottom
    percentile still excludes the jumpiest, most plausibly-irrigated pixels from the
    low-SM cluster while adapting to the data's actual noise floor.

    Returns (features_df_with_labels, frozenset of reference pixel_ids).
    """
    feat_cols = ["mean_sm", "jump_count", "sm_std"]
    X         = features_df[feat_cols].fillna(0).values

    X_scaled = StandardScaler().fit_transform(X)
    labels   = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)

    out = features_df.copy()
    out["cluster"] = labels

    cluster_mean = out.groupby("cluster")["mean_sm"].mean()
    ref_cluster  = int(cluster_mean.idxmin())
    low_sm_mask  = out["cluster"] == ref_cluster

    jump_cutoff  = float(out.loc[low_sm_mask, "jump_count"].quantile(ref_jump_percentile))
    strict_mask  = low_sm_mask & (out["jump_count"] <= jump_cutoff)
    out["is_reference"] = strict_mask

    ref_ids = frozenset(out.loc[strict_mask, "pixel_id"])
    n_low_sm, n_ref, n_tot = int(low_sm_mask.sum()), len(ref_ids), len(out)
    ref_sm  = cluster_mean[ref_cluster]
    irr_sm  = cluster_mean.drop(ref_cluster).mean()

    print(
        f"Low-SM cluster      : {n_low_sm}/{n_tot} pixels  "
        f"(mean SM {ref_sm:.1f} vol%  vs.  {irr_sm:.1f} vol% other cluster)"
    )
    print(
        f"Reference (strict)  : {n_ref}/{n_low_sm} pixels with jump_count <= {jump_cutoff:.0f}  "
        f"(calmest {ref_jump_percentile:.0%} of the low-SM cluster by local jump activity) "
        f"kept as non-irrigated reference"
    )
    return out, ref_ids


def compute_jump_fractions(
    buffer_pixel_df: pd.DataFrame,
    jump_thr: float = 3.0,
) -> pd.Series:
    """
    For each acquisition date, compute the fraction of buffer pixels
    that show a positive SM rise > jump_thr relative to their own previous observation.

    A high fraction (≥ RAIN_FRACTION) indicates a widespread wetting event (rain).

    Returns Series(datetime → fraction).
    """
    df = buffer_pixel_df.sort_values(["pixel_id", "datetime"]).copy()
    df["prev_sm"] = df.groupby("pixel_id")["sm"].shift(1)
    df = df.dropna(subset=["prev_sm"])
    df["jumped"] = (df["sm"] - df["prev_sm"]) > jump_thr

    return (
        df.groupby("datetime")["jumped"]
        .mean()
        .rename("jump_fraction")
        .sort_index()
    )


def time_centered_smooth(series: pd.Series, window_days: float) -> pd.Series:
    """
    Rolling mean centered in calendar time rather than acquisition count.

    A plain `series.rolling(N, center=True)` centers on the Nth-nearest
    acquisition by position, which is only centered in time if acquisitions are
    evenly spaced. Satellite revisits are irregular (gaps of 1-5+ days), so a
    count-based window can end up skewed toward whichever side happens to be
    denser at a given date, shifting peaks in time. This instead averages every
    value within ±window_days/2 of each date, which stays symmetric in time
    regardless of the local sampling pattern. (pandas' offset-based rolling
    windows don't support center=True, hence the manual implementation.)
    """
    half = pd.Timedelta(days=window_days / 2)
    idx = series.index
    return pd.Series(
        [series[(idx >= t - half) & (idx <= t + half)].mean() for t in idx],
        index=idx,
    ).rename(series.name)


def _merge_close_events(irr_events: pd.DataFrame, min_gap_days: int) -> pd.DataFrame:
    """
    Cluster consecutive irrigation events spaced < min_gap_days apart and keep
    only the highest-confidence event from each cluster.
    Comparison is made against the *last* kept event in the growing cluster,
    so a chain A→B→C where each consecutive gap < min_gap_days collapses to one event.
    """
    if len(irr_events) <= 1:
        return irr_events

    sorted_ev = irr_events.sort_values("datetime").reset_index(drop=True)
    clusters: list[list[int]] = []
    current: list[int] = [0]

    for i in range(1, len(sorted_ev)):
        gap = (sorted_ev.loc[i, "datetime"] - sorted_ev.loc[current[-1], "datetime"]).days
        if gap < min_gap_days:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)

    kept = []
    for cluster in clusters:
        cluster_df = sorted_ev.loc[cluster]
        kept.append(cluster_df.loc[cluster_df["confidence"].idxmax()])

    return pd.DataFrame(kept).reset_index(drop=True)


def detect_events(
    field_ts:                pd.Series,
    ref_ts:                  pd.Series,
    jump_fractions:          pd.Series,
    jump_thr:                float = 3.0,
    rain_fraction:           float = 0.90,
    max_gap_days:            int   = 5,
    min_irrigation_gap_days: int   = 4,
) -> pd.DataFrame:
    """
    Detect and classify SM rise events in the field time series.

    For each field acquisition where SM rises by more than jump_thr vol%:
    - Look up the nearest buffer acquisition within ±max_gap_days.
    - If ≥ rain_fraction of buffer pixels also jumped → **rain**.
    - Otherwise → **irrigation** (localized rise in the field, buffer stable).
    - If no buffer data within the window → **unclassified**.

    Confidence combines jump magnitude (vs. field baseline variability), reference
    pixel stability, and spatial coverage of the buffer signal. It is further scaled
    down the closer buffer_fraction sits to rain_fraction, softening the hard cutoff
    between rain and irrigation instead of trusting a single threshold at face value.

    Parameters
    ----------
    field_ts       : Series(datetime → sm) — field mean
    ref_ts         : Series(datetime → sm) — reference pixel mean
    jump_fractions : Series(datetime → fraction) — buffer jump fractions
    jump_thr                : vol% rise threshold to consider a potential event
    rain_fraction           : fraction of buffer pixels jumping that classifies an event as rain
    max_gap_days            : max days to match a field event to a buffer acquisition date
    min_irrigation_gap_days : consecutive irrigation events closer than this are merged;
                              the highest-confidence one in the cluster is kept

    Returns DataFrame: datetime, event_type, field_sm, field_delta,
                       ref_delta, buffer_fraction, confidence
    """
    field_std = float(field_ts.std()) or 1.0
    ref_delta_series = ref_ts.sort_index().diff()
    sorted_dates     = field_ts.sort_index().index

    events = []
    for i in range(1, len(sorted_dates)):
        dt          = sorted_dates[i]
        prev_dt     = sorted_dates[i - 1]
        field_delta = float(field_ts[dt]) - float(field_ts[prev_dt])

        if field_delta <= jump_thr:
            continue

        # Match to nearest buffer acquisition
        if jump_fractions.empty:
            buf_frac  = np.nan
            ref_delta = np.nan
        else:
            abs_secs     = np.abs((jump_fractions.index - dt).total_seconds().values)
            min_diff_pos = int(np.argmin(abs_secs))
            if abs_secs[min_diff_pos] > max_gap_days * 86400:
                buf_frac  = np.nan
                ref_delta = np.nan
            else:
                nearest_dt = jump_fractions.index[min_diff_pos]
                buf_frac   = float(jump_fractions.iloc[min_diff_pos])
                rd = ref_delta_series.get(nearest_dt)
                ref_delta = 0.0 if (rd is None or pd.isna(rd)) else float(rd)

        # Classify and score
        if np.isnan(buf_frac):
            event_type = "unclassified"
            confidence = None
        else:
            # Distance of buf_frac from rain_fraction, normalised against the room
            # available on its side of the threshold (0 = right at the threshold,
            # fully ambiguous; 1 = at the extreme, fully certain).
            if buf_frac >= rain_fraction:
                boundary_certainty = (buf_frac - rain_fraction) / max(1.0 - rain_fraction, 1e-6)
            else:
                boundary_certainty = (rain_fraction - buf_frac) / max(rain_fraction, 1e-6)
            boundary_certainty = float(np.clip(boundary_certainty, 0.0, 1.0))
            boundary_factor    = 0.5 + 0.5 * boundary_certainty

            mag_score = min(abs(field_delta) / (2 * field_std), 1.0)
            if buf_frac >= rain_fraction:
                event_type    = "rain"
                spatial_score = min(buf_frac, 1.0)
                base_conf     = 0.5 * mag_score + 0.5 * spatial_score
            else:
                event_type    = "irrigation"
                ref_stable    = 1.0 - min(abs(ref_delta) / jump_thr, 1.0) if not np.isnan(ref_delta) else 0.5
                spatial_score = 1.0 - min(buf_frac, 1.0)
                base_conf     = (mag_score + ref_stable + spatial_score) / 3.0
            confidence = round(base_conf * boundary_factor, 2)

        events.append({
            "datetime":        dt,
            "event_type":      event_type,
            "field_sm":        round(float(field_ts[dt]), 1),
            "field_delta":     round(field_delta, 1),
            "ref_delta":       round(ref_delta, 1) if not np.isnan(ref_delta) else None,
            "buffer_fraction": round(buf_frac, 2) if not np.isnan(buf_frac) else None,
            "confidence":      confidence,
        })

    cols = ["datetime", "event_type", "field_sm", "field_delta",
            "ref_delta", "buffer_fraction", "confidence"]
    result = pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)

    # Merge irrigation events that are too close together
    if min_irrigation_gap_days > 0 and not result.empty:
        irr_mask = result["event_type"] == "irrigation"
        if irr_mask.sum() > 1:
            merged = _merge_close_events(result[irr_mask], min_irrigation_gap_days)
            result = (
                pd.concat([result[~irr_mask], merged])
                .sort_values("datetime")
                .reset_index(drop=True)
            )

    return result


def summarise_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a plain-English summary and return the classified events table
    (rain + irrigation only; unclassified rows excluded).
    """
    if events_df.empty:
        print("No significant SM rise events detected in the season.")
        return events_df

    irr = events_df[events_df["event_type"] == "irrigation"]
    rain = events_df[events_df["event_type"] == "rain"]
    unc  = events_df[events_df["event_type"] == "unclassified"]

    print("─" * 50)
    print(f"Field irrigated      : {'Yes' if len(irr) > 0 else 'No (or no events detected)'}")
    print(f"Irrigation events    : {len(irr)}")
    if not irr.empty:
        for _, row in irr.iterrows():
            print(f"  {row['datetime'].date()}  Δ={row['field_delta']:+.1f} vol%  confidence={row['confidence']:.2f}")
    print(f"Rain events          : {len(rain)}")
    if not rain.empty:
        for _, row in rain.iterrows():
            print(f"  {row['datetime'].date()}  Δ={row['field_delta']:+.1f} vol%  confidence={row['confidence']:.2f}")
    if not unc.empty:
        print(f"Unclassified events : {len(unc)}  (no buffer data within ±5 days)")
    print("─" * 50)

    return (
        events_df[events_df["event_type"].isin(["rain", "irrigation"])]
        .sort_values("datetime")
        .reset_index(drop=True)
    )
