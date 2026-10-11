"""Common analysis grid joining Landsat (LST, NDVI, NDBI) with the building CSV.

The grid is defined locally: square cells of GRID_SCALE_M metres in UTM 43N,
aligned to multiples of the cell size, kept where they intersect the study
area. Each cell has a stable id  E<ix>_N<iy>  with ix = floor(x / size).

* Earth Engine aggregates the annual Landsat composites onto exactly this grid
  (fixed crs_transform), and the small rasters are downloaded once.
* The building CSV is aggregated onto the same ids in open_buildings.py.
* build_grid_dataset() joins both on grid_id - offline.

Output: data/processed/spatial_grid_metrics.csv     (one row per cell and year)
        data/processed/spatial_grid_metrics.geojson (cell polygons, WGS 84)
"""
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

log = logging.getLogger("spatial_grid")
LANDSAT_GRID_BANDS = ["LST", "NDVI", "NDBI", "LST_obs_count", "lst_valid_pixels"]


def grid_ids(ix, iy):
    ix, iy = np.asarray(ix).astype(int), np.asarray(iy).astype(int)
    return np.char.add(np.char.add("E", ix.astype(str)), np.char.add("_N", iy.astype(str)))


def landsat_grid_tif(year):
    return os.path.join(config.GRID_TIF_DIR, f"grid_landsat_{year}_{config.GRID_SCALE_M}m.tif")


# ----------------------------------------------------------------- local grid
def make_grid():
    """All cells intersecting the study area, with in-study area and city share."""
    import shapely
    from data_pipeline import study_area
    city, buffered = study_area.load_study_area(config.PROJECTED_CRS)
    s = config.GRID_SCALE_M
    x0, y0, x1, y1 = buffered.bounds
    ixs = np.arange(int(np.floor(x0 / s)), int(np.ceil(x1 / s)))
    iys = np.arange(int(np.floor(y0 / s)), int(np.ceil(y1 / s)))
    IX, IY = [a.ravel() for a in np.meshgrid(ixs, iys)]
    cells = shapely.box(IX * s, IY * s, (IX + 1) * s, (IY + 1) * s)
    shapely.prepare(buffered)
    hit = shapely.intersects(buffered, cells)
    IX, IY, cells = IX[hit], IY[hit], cells[hit]
    in_study = shapely.area(shapely.intersection(cells, buffered))
    in_city = shapely.area(shapely.intersection(cells, city))
    g = pd.DataFrame({"grid_id": grid_ids(IX, IY), "ix": IX, "iy": IY,
                      "x_utm": (IX + 0.5) * s, "y_utm": (IY + 0.5) * s,
                      "grid_size_m": s, "cell_area_in_study_m2": in_study,
                      "study_fraction": in_study / s ** 2, "city_fraction": in_city / s ** 2})
    g["zone"] = np.where(g.city_fraction >= 0.5 * g.study_fraction, "city", "buffer_ring")
    g["edge_cell"] = g.study_fraction < config.MIN_CELL_STUDY_FRACTION
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(config.PROJECTED_CRS, "EPSG:4326", always_xy=True).transform(g.x_utm.values, g.y_utm.values)
    g["lon"], g["lat"] = lon, lat
    return g


def cell_polygons_wgs84(grid):
    import shapely
    from pyproj import Transformer
    s = config.GRID_SCALE_M
    tr = Transformer.from_crs(config.PROJECTED_CRS, "EPSG:4326", always_xy=True)
    boxes = shapely.box(grid.ix * s, grid.iy * s, (grid.ix + 1) * s, (grid.iy + 1) * s)
    return shapely.transform(boxes, lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1])))


# ------------------------------------------------------------------ Earth Engine
def download_grid_rasters(df_monthly, city, buffered, force=False, start_date=None, end_date=None):
    """Per-year annual Landsat means aggregated to the analysis grid (exact alignment)."""
    from data_pipeline import extract_gee_data as gee
    ee = gee.init_ee()
    s = config.GRID_SCALE_M
    grid_proj = ee.Projection(config.GRID_CRS, [s, 0, 0, 0, -s, 0])
    failed = []
    for year, grp in df_monthly[df_monthly.status == "observed"].groupby("Year"):
        months = [int(m) for m in grp.Month]
        annual = (gee.annual_composite(int(year), months, buffered, start_date, end_date)
                  .setDefaultProjection(crs=config.GRID_CRS, scale=30))
        mean = annual.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(grid_proj)
        n_valid = (annual.select("LST").reduceResolution(ee.Reducer.count(), maxPixels=4096)
                   .reproject(grid_proj).rename("lst_valid_pixels"))
        out = mean.select(["LST", "NDVI", "NDBI", "LST_obs_count"]).addBands(n_valid)
        ok = gee.download_geotiff(out, buffered, landsat_grid_tif(int(year)), crs=config.GRID_CRS,
                                  crs_transform=[s, 0, 0, 0, -s, 0], force=force)
        if ok is None:
            failed.append(int(year))
    if failed:
        log.error("Grid raster download failed for year(s) %s - these years are missing from the grid table.", failed)
    return failed


# ------------------------------------------------------------------------- local
def _read(path, names):
    """Read a grid GeoTIFF into a DataFrame keyed by grid_id, checking alignment."""
    import rasterio
    with rasterio.open(path) as src:
        data = src.read().astype(float)
        crs = src.crs.to_string() if src.crs else None
        t = src.transform
        rows, cols = np.indices((src.height, src.width))
        xs, ys = rasterio.transform.xy(t, rows.ravel(), cols.ravel())
    s = config.GRID_SCALE_M
    if crs and crs.upper() != config.GRID_CRS.upper():
        raise ValueError(f"{os.path.basename(path)} is in {crs}, expected {config.GRID_CRS}")
    if abs(abs(t.a) - s) > 1e-6 or abs(abs(t.e) - s) > 1e-6:
        raise ValueError(f"{os.path.basename(path)} has {t.a} m pixels, expected {s} m (GRID_SCALE_M changed?)")
    if abs((t.c / s) - round(t.c / s)) > 1e-6 or abs((t.f / s) - round(t.f / s)) > 1e-6:
        raise ValueError(f"{os.path.basename(path)} is not aligned to the {s} m grid (origin {t.c}, {t.f}).")
    data[data <= -9998] = np.nan
    out = pd.DataFrame({n: band.ravel() for n, band in zip(names, data)})
    xs, ys = np.asarray(xs), np.asarray(ys)
    out["grid_id"] = grid_ids(np.floor(xs / s), np.floor(ys / s))
    return out


def build_grid_dataset():
    """Join the Landsat grids and the building grid into one tidy table (works offline)."""
    grid = make_grid()
    s = config.GRID_SCALE_M
    max_pix = (s / 30.0) ** 2

    buildings = None
    if os.path.exists(config.BUILDINGS_GRID_CSV):
        buildings = pd.read_csv(config.BUILDINGS_GRID_CSV)
        if len(set(buildings.grid_id) & set(grid.grid_id)) == 0:
            raise ValueError("Building grid ids do not match the analysis grid - re-run the buildings stage "
                             "(GRID_SCALE_M or the boundary probably changed).")
        bcols = ["grid_id", "building_count", "footprint_area_m2", "building_density", "buildings_per_km2",
                 "mean_footprint_m2"]
        grid = grid.merge(buildings[bcols], on="grid_id", how="left")
    else:
        log.warning("No building grid (%s) - building columns are empty. Run the buildings stage.",
                    os.path.relpath(config.BUILDINGS_GRID_CSV, config.BASE_DIR))
        for c in ("building_count", "footprint_area_m2", "building_density", "buildings_per_km2", "mean_footprint_m2"):
            grid[c] = np.nan

    frames = []
    suffix = f"_{s}m.tif"
    tifs = sorted(n for n in os.listdir(config.GRID_TIF_DIR)
                  if n.startswith("grid_landsat_") and n.endswith(suffix)) if os.path.isdir(config.GRID_TIF_DIR) else []
    for name in tifs:
        year = int(name[len("grid_landsat_"):-len(suffix)])
        ls = _read(os.path.join(config.GRID_TIF_DIR, name), LANDSAT_GRID_BANDS)
        ls = ls[ls.grid_id.isin(grid.grid_id)]
        # valid share of the 30 m pixels inside the study-area part of the cell
        ls = ls.merge(grid[["grid_id", "study_fraction"]], on="grid_id")
        ls["lst_valid_fraction"] = (ls.lst_valid_pixels.fillna(0) / (max_pix * ls.study_fraction)).clip(upper=1)
        thin = ls.lst_valid_fraction < config.MIN_CELL_VALID_FRACTION
        ls.loc[thin, ["LST", "NDVI", "NDBI"]] = np.nan
        frames.append(grid.merge(ls.drop(columns="study_fraction"), on="grid_id", how="left").assign(year=year))
        log.info("Grid %d: %d cells, %d with valid LST (%d dropped for < %.0f%% clear pixels)", year, len(grid),
                 ls.LST.notna().sum(), int(thin.sum()), 100 * config.MIN_CELL_VALID_FRACTION)
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        log.warning("No Landsat grid rasters found - the grid table holds building columns only.")
        out = grid.assign(year=np.nan, **{c: np.nan for c in LANDSAT_GRID_BANDS + ["lst_valid_fraction"]})

    first = ["grid_id", "year", "zone", "edge_cell", "lon", "lat", "x_utm", "y_utm", "grid_size_m",
             "cell_area_in_study_m2", "study_fraction", "city_fraction",
             "LST", "NDVI", "NDBI", "lst_valid_pixels", "lst_valid_fraction", "LST_obs_count",
             "building_count", "footprint_area_m2", "building_density", "buildings_per_km2", "mean_footprint_m2"]
    out = out[first + [c for c in out.columns if c not in first]]
    out.to_csv(config.GRID_CSV, index=False)
    log.info("Saved %s (%d cells x %d year[s])", config.GRID_CSV, len(grid), max(len(frames), 1))
    _write_grid_geojson(out)
    return out


def _write_grid_geojson(table):
    """Cell polygons with static building columns and LST_<year> / NDVI_<year> / NDBI_<year>."""
    from shapely.geometry import mapping
    static_cols = ["grid_id", "zone", "edge_cell", "cell_area_in_study_m2", "building_count", "footprint_area_m2",
                   "building_density", "buildings_per_km2", "mean_footprint_m2"]
    static = table.drop_duplicates("grid_id")[static_cols + ["x_utm", "y_utm"]].copy()
    wide = static
    if table.year.notna().any():
        piv = table.pivot_table(index="grid_id", columns="year", values=["LST", "NDVI", "NDBI"], dropna=False)
        piv.columns = [f"{v}_{int(y)}" for v, y in piv.columns]
        wide = static.merge(piv.reset_index(), on="grid_id", how="left")
    s = config.GRID_SCALE_M
    wide["ix"] = np.floor(wide.x_utm / s).astype(int)
    wide["iy"] = np.floor(wide.y_utm / s).astype(int)
    polys = cell_polygons_wgs84(wide)
    feats = []
    for rec, poly in zip(wide.drop(columns=["ix", "iy", "x_utm", "y_utm"]).to_dict("records"), polys):
        props = {k: (None if isinstance(v, float) and np.isnan(v) else (v.item() if hasattr(v, "item") else v))
                 for k, v in rec.items()}
        feats.append({"type": "Feature", "properties": props, "geometry": mapping(poly)})
    with open(config.GRID_GEOJSON, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    log.info("Saved %s", config.GRID_GEOJSON)
