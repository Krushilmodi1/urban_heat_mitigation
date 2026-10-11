"""Central configuration for the Ahmedabad urban heat project.

Everything a user is likely to change lives here. Paths are derived from this
file's location, so the pipeline works from any working directory.

Modules read these values at call time (config.X), so tests and scripts can
override them before running a stage.
"""
import os

# --------------------------------------------------------------------------- paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")               # inputs you supply (never modified)
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")   # everything the pipeline writes
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Inputs ------------------------------------------------------------------------
# Municipal boundary file (GeoJSON, GeoPackage, shapefile, or a .zip containing one).
BOUNDARY_DIR = os.path.join(RAW_DIR, "boundary")
BOUNDARY_FILE = os.path.join(BOUNDARY_DIR, "ahmedabad_municipal_boundary.geojson")
# Your uploaded Open Buildings CSV (or .csv.gz). Put the file here, or point this at it.
OPEN_BUILDINGS_CSV = r"D:\395_buildings.csv"


# Outputs -----------------------------------------------------------------------
STUDY_AREA_DIR = os.path.join(PROCESSED_DIR, "study_area")
BOUNDARY_GEOJSON = os.path.join(STUDY_AREA_DIR, "ahmedabad_city_boundary.geojson")
BUFFER_GEOJSON = os.path.join(STUDY_AREA_DIR, "ahmedabad_buffer_10km.geojson")
BOUNDARY_CONFIG_JSON = os.path.join(STUDY_AREA_DIR, "study_area_config.json")

LST_RASTER_DIR = os.path.join(PROCESSED_DIR, "lst_rasters")
LST_MONTHLY_TIF_DIR = os.path.join(LST_RASTER_DIR, "monthly")
LST_ANNUAL_TIF_DIR = os.path.join(LST_RASTER_DIR, "annual")
GRID_TIF_DIR = os.path.join(PROCESSED_DIR, "grid_rasters")

MONTHLY_CSV = os.path.join(PROCESSED_DIR, "monthly_lst_metrics.csv")
EXTRACTION_MANIFEST = os.path.join(PROCESSED_DIR, "extraction_manifest.json")

OPEN_BUILDINGS_DIR = os.path.join(PROCESSED_DIR, "open_buildings")
BUILDINGS_CLEAN_CSV = os.path.join(OPEN_BUILDINGS_DIR, "buildings_study_area.csv")
BUILDINGS_GPKG = os.path.join(OPEN_BUILDINGS_DIR, "buildings_study_area.gpkg")   # only if geopandas is installed
BUILDINGS_GRID_CSV = os.path.join(OPEN_BUILDINGS_DIR, "building_grid_metrics.csv")
BUILDING_SUMMARY_CSV = os.path.join(OPEN_BUILDINGS_DIR, "building_summary.csv")
BUILDINGS_SCHEMA_REPORT = os.path.join(OPEN_BUILDINGS_DIR, "csv_inspection_report.md")
BUILDINGS_META_JSON = os.path.join(OPEN_BUILDINGS_DIR, "processing_metadata.json")

GRID_CSV = os.path.join(PROCESSED_DIR, "spatial_grid_metrics.csv")
GRID_GEOJSON = os.path.join(PROCESSED_DIR, "spatial_grid_metrics.geojson")
REPORT_MD = os.path.join(PROCESSED_DIR, "summary_report.md")

# Models & Simulation -----------------------------------------------------------
MODELS_DIR = os.path.join(PROCESSED_DIR, "models")
PINN_MODEL_PATH = os.path.join(MODELS_DIR, "pinn_model.keras")
PINN_SCALER_JSON = os.path.join(MODELS_DIR, "pinn_scaler_stats.json")
PINN_METRICS_JSON = os.path.join(MODELS_DIR, "pinn_metrics.json")
SIMULATION_DIR = os.path.join(PROCESSED_DIR, "simulation")
SCENARIO_RESULTS_CSV = os.path.join(SIMULATION_DIR, "cooling_scenarios.csv")
SCENARIO_REPORT_MD = os.path.join(SIMULATION_DIR, "mitigation_recommendations.md")


# ------------------------------------------------------------------- Earth Engine
# The Cloud project is read from the environment so it is not baked into shared
# code:  set EE_PROJECT=your-project-id   (falls back to the project's original id)
EE_PROJECT = os.environ.get("EE_PROJECT", "bah2026-vector-zero")

# ---------------------------------------------------------------- analysis period
# Inclusive dates. The code converts END_DATE to Earth Engine's exclusive end.
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

# --------------------------------------------------------------------- study area
# Where the CITY boundary comes from:
#   "file"         BOUNDARY_FILE above (recommended: Ahmedabad Municipal Corporation limits)
#   "ee_asset"     an Earth Engine table asset you uploaded (BOUNDARY_ASSET)
#   "gaul_district" FAO GAUL 2015 level 2 "Ahmadabad" - this is the DISTRICT (~8,000 km2),
#                  NOT the city. Only for pipeline testing; every output is labelled "district".
BOUNDARY_SOURCE = "file"
BOUNDARY_ASSET = None                  # e.g. "projects/<your-project>/assets/amc_boundary"
BOUNDARY_NAME_FIELD = None             # optional: attribute to filter features in the file/asset
BOUNDARY_NAME_VALUE = None             # e.g. BOUNDARY_NAME_FIELD="name", VALUE="Ahmedabad Municipal Corporation"
GAUL_DATASET = "FAO/GAUL/2015/level2"
GAUL_FILTERS = {"ADM0_NAME": "India", "ADM1_NAME": "Gujarat", "ADM2_NAME": "Ahmadabad"}

BUFFER_M = 10_000                 # outward buffer, metres, computed in PROJECTED_CRS
PROJECTED_CRS = "EPSG:32643"      # WGS 84 / UTM zone 43N (metres) - Ahmedabad lies inside zone 43
EXPECTED_CENTRE = (72.5714, 23.0225)  # lon, lat of central Ahmedabad, used to verify the boundary
# Plausible municipal area range (AMC limits are roughly 460-510 km2). Outside this range
# the run stops unless BOUNDARY_SOURCE is "gaul_district" - it catches district files.
CITY_AREA_KM2_RANGE = (300, 800)
GEOMETRY_SIMPLIFY_M = 20          # vertex tolerance used for the copy sent to Earth Engine

# ------------------------------------------------------------------------ Landsat
LANDSAT_COLLECTIONS = {
    "L8": "LANDSAT/LC08/C02/T1_L2",
    # Landsat 9 (OLI-2/TIRS-2) has the same band names, scale factors and QA layout as
    # Landsat 8, so it is merged without any change to the processing. With both
    # satellites the revisit is 8 days instead of 16, which roughly doubles the number
    # of cloud-free looks per month. Set USE_LANDSAT9 = False for Landsat 8 only.
    "L9": "LANDSAT/LC09/C02/T1_L2",
}
USE_LANDSAT9 = True
SCENE_CLOUD_MAX = 60      # scene-level pre-filter (%); pixel masking does the real work
STATS_SCALE_M = 100       # scale for monthly regional statistics
EXPORT_SCALE_M = 100      # scale of downloaded LST GeoTIFFs (30 m exceeds the direct-download limit)
MIN_VALID_FRACTION = 0.20 # a month needs >= this share of cloud-free pixels to count as "observed"

# --------------------------------------------------------- Open Buildings (CSV)
# Map the project's internal names to the columns in YOUR CSV. Leave a value as None
# to auto-detect from the column names Google publishes (latitude, longitude,
# area_in_meters, confidence, geometry, full_plus_code). Detection only matches exact
# known names - it never guesses. Check data/processed/open_buildings/csv_inspection_report.md.
OPEN_BUILDINGS_COLUMNS = {
    "latitude": None,
    "longitude": None,
    "area_m2": None,
    "confidence": None,
    "geometry_wkt": None,
    "building_id": None,
}
OPEN_BUILDINGS_CRS = "EPSG:4326"  # Google publishes lat/lon and WKT in WGS 84 degrees
# Google: 0.65-0.70 is a reasonable lower bound; 0.75+ for higher precision.
# Only applied when the CSV has a confidence column. None = keep everything.
BUILDING_CONFIDENCE_MIN = 0.65
CSV_CHUNK_ROWS = 250_000          # rows read at a time, so very large CSVs fit in memory
MAX_FOOTPRINTS_TO_DRAW = 400_000  # above this the footprint map draws centroids instead
BUILDINGS_SOURCE_NOTE = ("Google Research Open Buildings (uploaded CSV). Single snapshot - "
                         "see csv_inspection_report.md for the date range if stated in the file.")

# --------------------------------------------------------------------------- grid
GRID_CRS = PROJECTED_CRS  # grid cells are squares in UTM 43N, aligned to multiples of GRID_SCALE_M
GRID_SCALE_M = 1000       # analysis cell size in metres (500 or 1000)
MIN_CELL_VALID_FRACTION = 0.25  # a cell needs >= 25 % clear 30 m pixels for its annual LST to be used
MIN_CELL_STUDY_FRACTION = 0.5   # edge cells less than half inside the study area are flagged

# ----------------------------------------------------------------------- plotting
# Fixed classes so every map is comparable between months and years: 2.5 deg C steps
# from LST_MIN to LST_MAX; values outside use the end colours (colour-bar arrows).
LST_MIN, LST_MAX, LST_STEP = 5.0, 55.0, 2.5
# Colour anchors, blue -> pale yellow -> orange -> red (no green), similar to national LST maps.
LST_COLORS = ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#fff7bc", "#fee391",
              "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#990000", "#67000d"]
MAP_DPI = 200
PLOT_DPI = 300
TREND_MIN_MONTHS = 36     # no long-term trend line is drawn with fewer observed months
TREND_ALPHA = 0.05

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def set_base_dir(base_dir):
    """Re-point every data/plot path at another project root (used by the offline tests)."""
    old = BASE_DIR
    for name, value in list(globals().items()):
        if name.isupper() and isinstance(value, str) and value.startswith(old):
            globals()[name] = os.path.join(base_dir, os.path.relpath(value, old))
    globals()["BASE_DIR"] = base_dir


def ensure_dirs():
    for d in (RAW_DIR, BOUNDARY_DIR, PROCESSED_DIR, PLOTS_DIR, STUDY_AREA_DIR, LST_MONTHLY_TIF_DIR,
              LST_ANNUAL_TIF_DIR, GRID_TIF_DIR, OPEN_BUILDINGS_DIR, MODELS_DIR, SIMULATION_DIR):
        os.makedirs(d, exist_ok=True)

