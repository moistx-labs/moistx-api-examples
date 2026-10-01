"""
MoistX irrigation detection helpers.

Public API
----------
load_field_geometry   – load a field boundary from any OGR-supported vector file
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

# Clipped files from the file endpoint. With include_metadata=True, each SM file
# comes with a sensors and a flag file on the same grid.
_SM_SUFFIX      = "_SM_5cm_cropped.tif"
_SENSORS_SUFFIX = "_sensors_cropped.tif"
_FLAG_SUFFIX    = "_flag_cropped.tif"

# Sensors bitmask (sensors file) → names, in the same order the API lists them
_SENSOR_NAMES = {1: "Sentinel-1 ascending", 2: "Sentinel-1 descending", 4: "Sentinel-2", 8: "Landsat-8/9"}
# Single-sensor files carry their sensor in the filename instead
_SENSOR_BIT_BY_SOURCE = {"S1_asc": 1, "S1_desc": 2, "S2": 4, "L89": 8}
# NDVI quality flag (flag file): good = NDVI < 0.75, poor = NDVI ≥ 0.75
_FLAG_LABELS = {0: "good", 1: "poor", 2: "unknown"}


# ── Internal ──────────────────────────────────────────────────────────────────

def _sensor_label(bits: int | None) -> str | None:
    """'Sentinel-2 + Landsat-8/9' style label for a sensors bitmask; None if unknown."""
    if not bits:
        return None
    return " + ".join(name for bit, name in _SENSOR_NAMES.items() if bits & bit)


def _read_companion(sm_path: str, suffix: str, geom_in_crs) -> tuple[np.ndarray, float] | None:
    """Masked read of the sensors/flag file that belongs to an SM file; None if it wasn't downloaded."""
    path = sm_path[: -len(_SM_SUFFIX)] + suffix
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        out_image, _ = rasterio_mask(src, [geom_in_crs], crop=True, nodata=src.nodata, all_touched=False)
        return out_image[0], src.nodata

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

def load_field_geometry(filepath: str) -> tuple:
    """
    Load a field boundary from any OGR-supported vector file (GeoJSON, Shapefile,
    GeoPackage, KML, ...) via geopandas.read_file.

    Multiple features/parts are unioned into a single geometry so the rest of the
    pipeline (buffering, clipping, WKT for the API) can treat the field as one shape.
    Reprojects to WGS84 (EPSG:4326) if the source file uses a different CRS.

    Returns (field_geom, field_wkt).
    """
    gdf = gpd.read_file(filepath)
    if gdf.empty:
        raise ValueError(f"{filepath} contains no features")
    if gdf.crs is None:
        raise ValueError(f"{filepath} has no CRS defined")
    gdf = gdf.to_crs("EPSG:4326")

    try:
        field_geom = gdf.geometry.union_all()
    except AttributeError:
        field_geom = gdf.geometry.unary_union

    return field_geom, field_geom.wkt


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
    include_metadata: bool = False,
) -> list[str]:
    """
    Download a ZIP of clipped GeoTIFFs from the MoistX file endpoint and
    unpack them into a local cache directory.

    With include_metadata=True the ZIP also contains, per acquisition, a sensors
    file (contributing sensors per pixel) and a flag file (NDVI quality flag per
    pixel), where the region provides them. load_pixel_timeseries picks these up
    automatically.

    The cache key is an MD5 of (wkt + start + end [+ include_metadata]), so
    re-running with identical parameters reads from disk without hitting the API.

    Returns a sorted list of the soil moisture .tif file paths (sensors/flag files
    sit next to them in the same folder).
    """
    key_src    = f"{wkt}{start}{end}" + ("|metadata" if include_metadata else "")
    cache_key  = hashlib.md5(key_src.encode()).hexdigest()[:12]
    cache_path = Path(cache_dir) / cache_key
    existing   = sorted(cache_path.glob(f"*{_SM_SUFFIX}"))

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
            "include_metadata":            include_metadata,
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

    tif_files = sorted(cache_path.glob(f"*{_SM_SUFFIX}"))
    n_meta    = len(list(cache_path.glob(f"*{_FLAG_SUFFIX}")))
    print(f"[{label}] {len(tif_files)} GeoTIFFs downloaded and cached"
          + (f" (+ sensors/quality files for {n_meta})" if include_metadata else ""))
    return [str(f) for f in tif_files]


def load_pixel_timeseries(
    tif_paths: list[str],
    geom,
) -> pd.DataFrame:
    """
    Extract per-pixel SM values from a list of (already clipped) GeoTIFFs.

    If the matching sensors/flag files were downloaded (include_metadata=True),
    each pixel also gets the sensors that contributed to it and its NDVI quality
    flag. Otherwise `quality` is None, and `sensors` is only known for
    single-sensor files (from the filename).

    Parameters
    ----------
    tif_paths : list of local .tif file paths
    geom      : Shapely geometry — pixels outside this geometry are skipped

    Returns DataFrame: pixel_id, lon, lat, datetime, sm (vol%), source, sensors, quality
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

            # Sensors/flag files share the SM grid, so the same (row, col) is the same pixel
            sensors_read = _read_companion(fpath, _SENSORS_SUFFIX, geom_in_crs)
            flag_read    = _read_companion(fpath, _FLAG_SUFFIX, geom_in_crs)
            if sensors_read is not None:
                sensors = [_sensor_label(int(b)) for b in sensors_read[0][rs, cs]]
            else:
                sensors = [_sensor_label(_SENSOR_BIT_BY_SOURCE.get(source))] * len(rs)
            if flag_read is not None:
                quality = [_FLAG_LABELS.get(int(f)) for f in flag_read[0][rs, cs]]
            else:
                quality = [None] * len(rs)

            for lon, lat, sm, sens, qual in zip(lons, lats, sm_vals, sensors, quality):
                rows.append({
                    "pixel_id": f"{lon:.5f},{lat:.5f}",
                    "lon":      lon,
                    "lat":      lat,
                    "datetime": dt,
                    "sm":       sm,
                    "source":   source,
                    "sensors":  sens,
                    "quality":  qual,
                })

        except Exception as e:
            print(f"  Warning — skipping {Path(fpath).name}: {e}")

        if (idx + 1) % 10 == 0 or idx == n - 1:
            print(f"  Loaded {idx + 1}/{n} files …", end="\r")

    print()
    if not rows:
        return pd.DataFrame(columns=["pixel_id", "lon", "lat", "datetime", "sm", "source", "sensors", "quality"])
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
    jump_thr:                float = 2.5,
    rain_fraction:           float = 0.70,
    min_irrigation_gap_days: int   = 4,
    field_coverage:          pd.Series | None = None,
    min_field_coverage:      float = 0.5,
) -> pd.DataFrame:
    """
    Detect and classify SM rise events in the field time series.

    Works on the field-minus-reference difference (anomaly) per acquisition rather
    than on raw field rises. Field and reference come from the same acquisition, so
    whatever is common to both cancels out: weather-driven wetting/drying and, in
    harmonized data, the offset between sensors (e.g. Sentinel-1 reading wetter than
    the optical sensors) that otherwise shows up as a "rise" every time the sensor
    changes between consecutive dates. What is left is wetting specific to the field.

    Between each pair of consecutive usable dates (reference available on that date,
    and field_coverage ≥ min_field_coverage if given):
    - anomaly rises by more than jump_thr, field SM itself rises by more than
      jump_thr / 2, and < rain_fraction of buffer pixels jumped → **irrigation**
      (the field wetted more than its surroundings). The field-rise condition is
      needed because the field-reference offset still varies somewhat by sensor
      (the reference can read relatively wetter on Sentinel-1 dates), so the
      reference drying back alone can lift the anomaly while the field stays flat
    - field SM rises by more than jump_thr, and ≥ rain_fraction of buffer pixels
      jumped → **rain** (widespread wetting)
    - buffer jump fraction unknown for the date → **unclassified**

    Note that buffer jump fractions compare each pixel with its own previous
    observation, so a sensor switch can raise them too — rain labels are less
    certain than irrigation labels on dates where the sensor changes.

    Confidence combines jump magnitude (vs. the series' own variability), how little
    of the buffer jumped (irrigation) or how much (rain), and field pixel coverage.
    It is further scaled down the closer buffer_fraction sits to rain_fraction,
    softening the hard cutoff between rain and irrigation.

    Parameters
    ----------
    field_ts       : Series(datetime → sm) — field mean
    ref_ts         : Series(datetime → sm) — reference pixel mean, *unsmoothed*, on
                     the same acquisition datetimes as field_ts (smoothing it in time
                     would break the same-date comparison)
    jump_fractions : Series(datetime → fraction) — buffer jump fractions
    jump_thr                : vol% rise (of the anomaly for irrigation, of field SM
                              for rain) needed to flag an event
    rain_fraction           : fraction of buffer pixels jumping that classifies an event as rain
    min_irrigation_gap_days : consecutive irrigation events closer than this are merged;
                              the highest-confidence one in the cluster is kept
    field_coverage          : Series(datetime → fraction of field pixels with a valid
                              value); dates below min_field_coverage are skipped, since a
                              field mean from a handful of pixels is unreliable
    min_field_coverage      : minimum field_coverage for a date to be used

    Returns DataFrame: datetime, event_type, field_sm, field_delta, ref_delta,
                       anomaly_delta, buffer_fraction, field_coverage, confidence
    """
    df = pd.concat({"field": field_ts, "ref": ref_ts}, axis=1).dropna().sort_index()
    if field_coverage is not None:
        df["coverage"] = field_coverage.reindex(df.index)
        df = df[df["coverage"] >= min_field_coverage]
    else:
        df["coverage"] = np.nan
    df["anomaly"] = df["field"] - df["ref"]

    field_std   = float(df["field"].diff().std()) or 1.0
    anomaly_std = float(df["anomaly"].diff().std()) or 1.0

    events = []
    for i in range(1, len(df)):
        dt, cur, prev = df.index[i], df.iloc[i], df.iloc[i - 1]
        field_delta   = float(cur["field"] - prev["field"])
        ref_delta     = float(cur["ref"] - prev["ref"])
        anomaly_delta = float(cur["anomaly"] - prev["anomaly"])
        coverage      = float(cur["coverage"])

        bf       = jump_fractions.get(dt)
        buf_frac = np.nan if bf is None or pd.isna(bf) else float(bf)

        if np.isnan(buf_frac):
            if anomaly_delta <= jump_thr and field_delta <= jump_thr:
                continue
            event_type = "unclassified"
            confidence = None
        else:
            is_rain = buf_frac >= rain_fraction and field_delta > jump_thr
            # The field itself must also get noticeably wetter: an anomaly rise where
            # the field stays flat is the reference drying, not the field wetting.
            is_irr  = (buf_frac < rain_fraction and anomaly_delta > jump_thr
                       and field_delta > jump_thr / 2)
            if not (is_rain or is_irr):
                continue

            # Distance of buf_frac from rain_fraction, normalised against the room
            # available on its side of the threshold (0 = right at the threshold,
            # fully ambiguous; 1 = at the extreme, fully certain).
            if is_rain:
                boundary_certainty = (buf_frac - rain_fraction) / max(1.0 - rain_fraction, 1e-6)
            else:
                boundary_certainty = (rain_fraction - buf_frac) / max(rain_fraction, 1e-6)
            boundary_factor = 0.5 + 0.5 * float(np.clip(boundary_certainty, 0.0, 1.0))

            if is_rain:
                event_type    = "rain"
                mag_score     = min(field_delta / (2 * field_std), 1.0)
                spatial_score = min(buf_frac, 1.0)
            else:
                event_type    = "irrigation"
                mag_score     = min(anomaly_delta / (2 * anomaly_std), 1.0)
                spatial_score = 1.0 - min(buf_frac, 1.0)
            scores = [mag_score, spatial_score] + ([coverage] if not np.isnan(coverage) else [])
            confidence = round(float(np.mean(scores)) * boundary_factor, 2)

        events.append({
            "datetime":        dt,
            "event_type":      event_type,
            "field_sm":        round(float(cur["field"]), 1),
            "field_delta":     round(field_delta, 1),
            "ref_delta":       round(ref_delta, 1),
            "anomaly_delta":   round(anomaly_delta, 1),
            "buffer_fraction": round(buf_frac, 2) if not np.isnan(buf_frac) else None,
            "field_coverage":  round(coverage, 2) if not np.isnan(coverage) else None,
            "confidence":      confidence,
        })

    cols = ["datetime", "event_type", "field_sm", "field_delta", "ref_delta",
            "anomaly_delta", "buffer_fraction", "field_coverage", "confidence"]
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
            print(f"  {row['datetime'].date()}  Δ vs reference={row['anomaly_delta']:+.1f} vol%  "
                  f"(field Δ={row['field_delta']:+.1f})  confidence={row['confidence']:.2f}")
    print(f"Rain events          : {len(rain)}")
    if not rain.empty:
        for _, row in rain.iterrows():
            print(f"  {row['datetime'].date()}  Δ={row['field_delta']:+.1f} vol%  confidence={row['confidence']:.2f}")
    if not unc.empty:
        print(f"Unclassified events : {len(unc)}  (no buffer jump fraction for the date)")
    print("─" * 50)

    return (
        events_df[events_df["event_type"].isin(["rain", "irrigation"])]
        .sort_values("datetime")
        .reset_index(drop=True)
    )
