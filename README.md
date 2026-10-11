# Ahmedabad Urban Heat Mitigation

Land surface temperature (LST), vegetation (NDVI), built-up index (NDBI) and building data for
**Ahmedabad municipal limits plus a 10 km outward buffer**, on a common spatial grid, ready for
the PINN and cooling-scenario modules.

## Project status

| Part | Status |
|---|---|
| Municipal boundary + true 10 km buffer | Done. Works with AMC boundary file or `--demo` bootstrap. |
| Landsat 8 + 9 extraction (LST, NDVI, NDBI) | Done. Earth Engine extraction with Landsat 8 + 9 fusion & cloud masking. |
| Monthly / annual LST maps, trend plots, annual comparison | Done; tested offline on synthetic & extracted data. |
| Open Buildings CSV processing | Done; chunked processing, schema detection, footprint area computation. |
| Combined grid + correlations | Done; unified 1 km UTM 43N spatial analysis grid. |
| PINN (`models/`) | **Done.** Physics-Informed Neural Network solving 2D thermal diffusion-advection energy balance. |
| Cooling Scenarios (`simulation/`) | **Done.** Simulates cool roofs, green roofs, urban trees, and policy recommendations report. |


## 1. Setup (Windows)

```bat
cd urban_heat_mitigation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install geopandas            (optional: shapefile / KML boundaries, .gpkg output)
earthengine authenticate         (once per machine; opens a browser)
set EE_PROJECT=your-cloud-project-id
```

macOS / Linux: `source .venv/bin/activate` and `export EE_PROJECT=...`.
The Cloud project must be registered for Earth Engine. If `EE_PROJECT` is not set, the id in
`config.py` is used. No credentials are stored in the code.

## 2. Put your inputs in place

| Input | Where | Notes |
|---|---|---|
| AMC municipal boundary | `data/raw/boundary/ahmedabad_municipal_boundary.geojson` | GeoJSON works out of the box; shapefile / GeoPackage / KML / .zip need geopandas. Set `BOUNDARY_FILE` in `config.py` if the name differs. If the file holds several features, set `BOUNDARY_NAME_FIELD` / `BOUNDARY_NAME_VALUE`. |
| Open Buildings CSV | `data/raw/open_buildings.csv` | Or set `OPEN_BUILDINGS_CSV`. `.csv.gz` also works. The file is only read, never changed. |

**Boundary checks.** The run stops (and says why) if the boundary does not contain central
Ahmedabad, if its area is outside 300–800 km² (AMC is roughly 460–510 km²; the district is about
8,000 km²), or if the buffer edge is not 10 km from the city everywhere.

Where to get the AMC boundary: AMC's GIS / Town Planning department, or OpenStreetMap (export the
Ahmedabad Municipal Corporation administrative boundary, e.g. via overpass-turbo, as GeoJSON).
Check the result on a map before use. To test the pipeline without it, set
`BOUNDARY_SOURCE = "gaul_district"` — every output is then labelled "district (test only)".

## 3. Run

```bat
python main.py --demo                  full pipeline with auto-bootstrapped demo inputs if raw files missing
python main.py --test                  first 3 months only (fast check)
python main.py                         full period in config.py (2024-01-01 to 2025-12-31)
python main.py --offline               everything except Earth Engine (uses files already exported)
python main.py --stage pinn            train Physics-Informed Neural Network (PINN) on spatial grid
python main.py --stage simulation      run urban heat mitigation cooling scenarios & report
python main.py --stage plots           redraw all figures offline
python main.py --force                 recompute even if outputs exist
python main.py -v                      verbose mode with full stack traces for debugging
```

Stages run in order: `boundary → extract → buildings → grid → plots → pinn → simulation`. A failed stage is listed in
the summary at the end and the exit code is 1; stages that need it are not run.


**Caching.** Earth Engine results are saved once and reused. `extraction_manifest.json` records
the period, sensors, masks and boundary; if any of them change, the extraction is redone
automatically. Building processing is reused unless the CSV or its settings change.

**What needs internet:** only the `extract` stage (and `boundary` when the source is an Earth
Engine asset or GAUL). Everything else runs offline from `data/processed/`.

### Quick Earth Engine check on a short range

```bat
python main.py --stage boundary
python main.py --stage extract --start 2024-03-01 --end 2024-03-31
```

Expect one log line for March 2024 with scene counts, clear-sky share and a mean LST around
35–45 °C, then `ahmedabad_lst_2024_03.tif` under `data/processed/lst_rasters/monthly/2024/`.

## 4. Outputs

| File | Content |
|---|---|
| `data/processed/study_area/ahmedabad_city_boundary.geojson`, `ahmedabad_buffer_10km.geojson` | The two geometries used everywhere |
| `data/processed/study_area/study_area_config.json` | Source, areas, buffer check, file hash |
| `data/processed/monthly_lst_metrics.csv` | One row per month: status, scenes (L8/L9), dates, valid pixels, clear share, mean clear obs/pixel, LST/NDVI/NDBI for study area and city |
| `data/processed/annual_lst_comparison.csv` | Annual means, all observed and common months only |
| `data/processed/lst_rasters/monthly/<year>/ahmedabad_lst_<year>_<mm>.tif` | Monthly LST (°C) + clear-observation count, 100 m, EPSG:4326, nodata −9999 |
| `data/processed/lst_rasters/annual/ahmedabad_annual_<year>.tif` | Annual LST, NDVI, NDBI, obs count |
| `data/processed/open_buildings/csv_inspection_report.md` | Columns, types, missing values, CRS, coverage, duplicates, invalid geometries, filtering counts |
| `data/processed/open_buildings/buildings_study_area.csv` (+ `.gpkg` if geopandas) | Cleaned buildings in the study area, with zone, grid id, UTM area |
| `data/processed/open_buildings/building_summary.csv` | Count, footprint area, mean/median, densities for city / buffer ring / study area |
| `data/processed/open_buildings/building_grid_metrics.csv` | Buildings per grid cell |
| `data/processed/spatial_grid_metrics.csv` / `.geojson` | **Modelling table**: one row per cell and year with LST, NDVI, NDBI, valid pixels, buildings |
| `data/processed/summary_report.md` | Data gaps and key findings |
| `plots/lst_monthly_maps/<year>/ahmedabad_lst_<year>_<mm>.png` | A map for every month, including months with no data |
| `plots/ahmedabad_lst_annual_<year>.png` | Annual mean LST maps |
| `plots/ahmedabad_lst_trend_<year>.png` | Jan–Dec trend per year (gaps shown) |
| `plots/ahmedabad_lst_cross_year_comparison.png` | All years on one Jan–Dec chart |
| `plots/ahmedabad_lst_time_series.png` | Full monthly series; trend line only when justified |
| `plots/ahmedabad_lst_annual_comparison.png` | Annual means and relatively hot / cool months |
| `plots/correlation_monthly_means.png`, `correlation_grid_<year>.png` | Pearson and Spearman heatmaps with n per pair |
| `plots/scatter_lst_drivers_<year>.png` | LST vs NDVI, NDBI, footprint density, footprint area |
| `plots/lst_by_building_density_<year>.png` | LST by density class, city vs buffer ring |
| `plots/lst_and_building_density_<year>.png` | LST and footprint density maps side by side |
| `plots/buildings/building_footprints_map.png`, `building_density_maps.png` | Footprints; footprint density and count density |

## 5. Data and definitions

| Variable | Source | Unit / definition |
|---|---|---|
| LST | Landsat 8 + 9 C2 L2 `ST_B10` × 0.00341802 + 149.0 − 273.15 | °C, ~10:30 local time, clear sky |
| NDVI | (B5 − B4) / (B5 + B4), reflectance = DN × 0.0000275 − 0.2 | −1 to 1 |
| NDBI | (B6 − B5) / (B6 + B5) | −1 to 1 |
| building_density | Σ footprint area ÷ cell area inside the study area | fraction 0–1 (shown as %) |
| buildings_per_km2 | building count ÷ cell area inside the study area | count per km² — a different measure |
| footprint area | Computed from polygons in UTM 43N (EPSG:32643) | m² |

**Masking.** QA_PIXEL bits 0–4 (fill, dilated cloud, cirrus, cloud, shadow), any QA_RADSAT
saturation flag, ST fill, and (for NDVI/NDBI) reflectance outside the valid range are removed.

**Months.** A month is "observed" when at least 20 % of the study area is clear. Other months are
kept with status `insufficient_coverage`, `no_data` or `extraction_failed`, are drawn as gaps,
and are never filled.

**Landsat 9.** Same bands, scale factors and QA layout as Landsat 8, so it is merged without any
change to the processing. It halves the revisit time (8 days), which matters in the monsoon.
Set `USE_LANDSAT9 = False` to use Landsat 8 only.

**Grid.** Square cells in UTM 43N, aligned to multiples of `GRID_SCALE_M` (1,000 m; 500 m works
too). Id `E<ix>_N<iy>`. Earth Engine writes Landsat onto exactly this grid, so the join is by id.
Cells with < 25 % clear pixels get no LST; edge cells < 50 % inside the study area are flagged
and left out of correlations and density maps.

## 6. Limitations

- Building data are one static snapshot unless the CSV has a date column; they cannot show
  change between years. Comparing them with several years of LST assumes little change.
- Buildings are placed in a cell by their centroid.
- If the CSV has points only, footprint areas come from its area column (if any) and are not
  measured; with neither, only counts are reported.
- Grid cells are spatially autocorrelated, so p-values would be overstated; none are claimed.
  Correlation is not causation.
- Landsat sees ~10:30 local time on clear days. Monsoon months are often missing.
- A long-term trend line needs at least 36 observed months.

## 7. Tests

```bat
python -m unittest discover -s tests -v
python tests\run_synthetic_demo.py C:\temp\uhm_demo     (look at every output; data are SYNTHETIC)
```

The tests run offline on synthetic data: Landsat scaling and QA bits, Earth Engine's exclusive
end date, boundary validation and 10 km buffer distance, CSV schema detection (polygons, points,
no coordinates), building counts against an independent count, density arithmetic, grid
alignment, missing-month handling, correlation sample sizes, and that every map and plot is drawn.

## 8. Changes in this version

- **Boundary**: new `data_pipeline/study_area.py`. Reads the municipal boundary from a file or
  EE asset, validates it, buffers 10,000 m in UTM 43N, and saves both geometries for reuse. The
  silent GAUL district default was removed.
- **Landsat**: Landsat 9 added; ST fill and reflectance range masked; per-pixel clear-observation
  counts; scene dates and L8/L9 counts per month; partial first/last months handled; failed
  months marked instead of reported as "no data"; cache invalidated when settings change.
- **Buildings**: Earth Engine Open Buildings V3 / Temporal code replaced by chunked processing of
  your uploaded CSV, with schema detection, configurable column mapping, inspection report,
  duplicate / invalid-geometry handling and UTM areas.
- **Grid**: defined locally with stable ids; Earth Engine output forced onto the same grid;
  alignment checked on read; GeoJSON output.
- **Plots**: blue → yellow → red scale in 2.5 °C classes, rasters masked to the buffer, scale bar
  and north arrow, observation period on every map, new annual comparison, footprint-area
  scatter, footprint and count density maps; report lists anything not drawn.
- **main.py**: stages `boundary` and `buildings`, `--offline`, end-of-run summary and non-zero
  exit on failure. `--stage pinn` reports the missing module instead of crashing.

## Attribution

- Landsat 8 and 9 Collection 2 Level-2 courtesy of the U.S. Geological Survey.
- Open Buildings: W. Sirko et al. (2021), *Continental-scale building detection from high
  resolution satellite imagery*, arXiv:2107.12283. CC BY 4.0 and ODbL v1.0.
  https://sites.research.google/gr/open-buildings/
