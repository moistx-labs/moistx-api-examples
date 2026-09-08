<img src="https://moistx.com/assets/moistx_logo-BNq5fFti.png" alt="MoistX logo" width="200"/>

# MoistX API Examples

Example notebooks showing how to use the [MoistX API](https://moistx.com/api/) to fetch and
analyse satellite-derived surface soil moisture (SSM, 0–5 cm) data.

## The MoistX API

- **Docs:** [Swagger UI](https://moistx.com/api/docs) · [ReDoc](https://moistx.com/api/redoc)
- **Auth:** pass your key as an `X-Api-Key` header. Get one from your
  [MoistX Dashboard](https://moistx.com/dashboard).
- **Key endpoints used by these notebooks:**

  | Endpoint | Used by |
  |----------|---------|
  | `GET /get/point/soil-moisture` | `01_getting_started.ipynb` — time series for a single lat/lon |
  | `POST /get/file/soil-moisture` | `02_irrigation_detection.ipynb`, `03_field_soil_moisture_map.ipynb` — clipped GeoTIFFs for a polygon |

  Both accept a maximum 365-day date range per request; the file endpoint additionally
  caps the request polygon at 1,000 hectares.

### Data license & attribution

Data is free to use, share, and modify (including commercial use), provided you attribute
MoistX: *"Data provided by MoistX (https://moistx.com)"*. The underlying satellite/model
sources carry their own attribution requirements:

- **Sentinel-1 / Sentinel-2:** Copernicus Sentinel data, processed by MoistX, used in
  accordance with the Copernicus Sentinel Data Terms and Conditions.
- **ERA5-Land:** Generated using Copernicus Climate Change Service (C3S) information from
  the Copernicus Climate Data Store.
- **Landsat 8/9:** Courtesy of the U.S. Geological Survey and NASA.

## Setup

**pip:**
```bash
pip install -r requirements.txt
```

**conda** (recommended if you don't already have a working `rasterio`/`geopandas` install —
conda resolves their compiled GDAL/GEOS dependencies more reliably than pip on most systems):
```bash
conda env create -f environment.yml
conda activate moistx-api-examples
```

Both install the same set of packages, including JupyterLab — skip that part of
`requirements.txt`/`environment.yml` if you're already running notebooks through VS Code,
Colab, or another environment that provides its own kernel.

Get an API key from [moistx.com/dashboard](https://moistx.com/dashboard), then create a
`.env` file in the repo root with:

```
MOISTX_API_KEY=your_key
```

(copy `.env.example` as a starting point). Each notebook's setup cell loads it
automatically via `python-dotenv`. `.env` is gitignored, so your key is never committed —
never commit it or paste it directly into a notebook cell.

## Notebooks

### `01_getting_started.ipynb`

The minimal end-to-end example: authenticate, query soil moisture for a single point
(`GET /get/point/soil-moisture`) over a year, load the response into a pandas DataFrame,
and plot a time series coloured by data source (`S2`, `S1_asc`, `S1_desc`, `L89`, or
`harmonized` where available).

### `02_irrigation_detection.ipynb`

A more advanced workflow that detects irrigation and rain events from SSM time series:

1. Auto-generate a ring buffer around a field polygon
2. Download clipped GeoTIFFs for the field and buffer via `POST /get/file/soil-moisture` (cached locally under `cache/`)
3. Cluster buffer pixels into non-irrigated reference vs. potentially-irrigated groups (K-means)
4. Classify field SM rises as **rain** (widespread jump across the buffer) or **irrigation** (localised jump, buffer stable)
5. Summarise detected events with a confidence score

The reusable logic behind this notebook lives in `helpers.py` (detection pipeline) and
`plotting.py` (the chart), keeping the notebook itself to short, readable calls.

### `03_field_soil_moisture_map.ipynb`

Visualizes **per-pixel** (not averaged) surface soil moisture inside a field boundary:

1. Load a field boundary from any OGR-supported vector file (GeoJSON, Shapefile, GeoPackage, ...)
2. Download clipped GeoTIFFs for the field via `POST /get/file/soil-moisture` (cached locally under `cache/`)
3. Extract every pixel's soil moisture value for each acquisition date
4. Pick a date from an interactive dropdown to view that date's pixel-level soil moisture on a map, colored on a fixed scale so colors stay comparable across dates

## `helpers.py`

| Function | Purpose | Used by |
|----------|---------|---------|
| `load_field_geometry` | Load a field boundary from any OGR-supported vector file, reprojected to WGS84 | `03` |
| `compute_ring_buffer` | Build a ring-shaped buffer polygon around a field | `02` |
| `download_sm_files` | Download & locally cache a ZIP of clipped GeoTIFFs from the API | `02`, `03` |
| `load_pixel_timeseries` | Extract per-pixel SM values from a list of GeoTIFFs into a DataFrame | `02`, `03` |
| `compute_pixel_features` | Compute per-pixel temporal features (mean, std, jump count) | `02` |
| `find_reference_pixels` | K-means clustering to identify non-irrigated reference pixels | `02` |
| `compute_jump_fractions` | Per-acquisition fraction of buffer pixels showing a SM jump | `02` |
| `time_centered_smooth` | Rolling mean centered in calendar time (not acquisition count) | `02` |
| `detect_events` | Classify SM rises as rain or irrigation | `02` |
| `summarise_events` | Print a plain-English summary of detected events | `02` |

## `plotting.py`

| Function | Purpose | Used by |
|----------|---------|---------|
| `plot_field_buffer_map` | Basemap showing the field polygon and its buffer ring | `02` |
| `plot_cluster_map` | Scatter map of buffer pixels by reference/irrigated cluster | `02` |
| `plot_sm_events` | Build the field/reference SM chart with rain & irrigation event markers | `02` |
| `plot_pixel_sm_map` | Per-pixel SM map for a single date, continuous color scale | `03` |

## `cache/`

Downloaded GeoTIFFs are cached here, keyed by an MD5 hash of the request polygon and date
range — re-running a notebook with the same parameters reads from disk instead of hitting
the API again. Safe to delete at any time; it will be repopulated on the next run.
