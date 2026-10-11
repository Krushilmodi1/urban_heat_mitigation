"""SYNTHETIC test data for the offline smoke test.

Nothing here is real: the "city" polygon is an invented shape of roughly
municipal size centred on Ahmedabad, the buildings are random rectangles, and
the LST rasters are generated numbers. They exist only to exercise the code
path from inputs to outputs without Earth Engine or the real files.
"""
import json
import os

import numpy as np
import pandas as pd

CENTRE = (72.5714, 23.0225)


def synthetic_city_polygon():
    """Irregular ~480 km2 polygon around central Ahmedabad (SYNTHETIC)."""
    from pyproj import Transformer
    from shapely.geometry import Polygon
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    to_ll = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(*CENTRE)
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    r = 12_400 * (1 + 0.12 * np.sin(3 * ang) + 0.05 * np.cos(5 * ang))
    xs, ys = cx + r * np.cos(ang), cy + 0.95 * r * np.sin(ang)
    lon, lat = to_ll.transform(xs, ys)
    return Polygon(zip(lon, lat))


def write_boundary(path, polygon=None):
    from shapely.geometry import mapping
    polygon = polygon or synthetic_city_polygon()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"name": "SYNTHETIC test city"}, "geometry": mapping(polygon)}]}, fh)
    return polygon


def write_buildings_csv(path, n=6000, seed=1):
    """Random footprints in Google's published CSV schema, with known defects added."""
    from pyproj import Transformer
    from shapely.geometry import box
    rng = np.random.default_rng(seed)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    to_ll = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(*CENTRE)
    # denser near the centre, extending beyond the buffer so some are filtered out
    d = np.abs(rng.normal(0, 13_000, n)); a = rng.uniform(0, 2 * np.pi, n)
    x, y = cx + d * np.cos(a), cy + d * np.sin(a)
    w, h = rng.uniform(6, 30, n), rng.uniform(6, 25, n)
    rows = []
    for i in range(n):
        poly_utm = box(x[i] - w[i] / 2, y[i] - h[i] / 2, x[i] + w[i] / 2, y[i] + h[i] / 2)
        lon, lat = to_ll.transform(*np.asarray(poly_utm.exterior.coords).T)
        wkt = "POLYGON((" + ", ".join(f"{a_:.7f} {b_:.7f}" for a_, b_ in zip(lon, lat)) + "))"
        clon, clat = to_ll.transform(x[i], y[i])
        rows.append({"latitude": clat, "longitude": clon, "area_in_meters": round(poly_utm.area, 4),
                     "confidence": round(rng.uniform(0.6, 0.98), 4), "geometry": wkt,
                     "full_plus_code": f"SYN{i:07d}"})
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.iloc[:25]], ignore_index=True)                 # 25 exact duplicates
    df.loc[len(df)] = {**df.iloc[30].to_dict(), "geometry": None, "full_plus_code": "SYNMISSING"}  # missing geometry
    bow = df.iloc[40].to_dict()                                              # self-intersecting "bow-tie"
    lon0, lat0 = bow["longitude"], bow["latitude"]
    e = 0.0001
    bow.update(full_plus_code="SYNBOWTIE", geometry=(f"POLYGON(({lon0-e} {lat0-e}, {lon0+e} {lat0+e}, "
                                                     f"{lon0+e} {lat0-e}, {lon0-e} {lat0+e}, {lon0-e} {lat0-e}))"))
    df.loc[len(df)] = bow
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def write_monthly_csv(path, years=(2024, 2025), missing=((2024, 7), (2025, 8)), thin=((2024, 8),)):
    """Monthly table in the extractor's format, with explicit no-data / thin months."""
    rows = []
    clim = [24, 27, 33, 38, 41, 37, 31, 30, 32, 33, 29, 25]
    for y in years:
        for m in range(1, 13):
            status = "no_data" if (y, m) in missing else ("insufficient_coverage" if (y, m) in thin else "observed")
            lst = np.nan if status == "no_data" else clim[m - 1] + (0.6 if y == 2025 else 0)
            rows.append({"Year": y, "Month": m, "Month_Name": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                                                               "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1],
                         "Date": f"{y}-{m:02d}-01", "window_start": f"{y}-{m:02d}-01",
                         "window_end_exclusive": (pd.Timestamp(y, m, 1) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
                         "status": status, "n_scenes": 0 if status == "no_data" else 4,
                         "n_scenes_L8": 0 if status == "no_data" else 2, "n_scenes_L9": 0 if status == "no_data" else 2,
                         "first_scene": None if status == "no_data" else f"{y}-{m:02d}-03",
                         "last_scene": None if status == "no_data" else f"{y}-{m:02d}-27",
                         "valid_fraction": 0.0 if status == "no_data" else (0.1 if status != "observed" else 0.85),
                         "LST": lst, "NDVI": 0.15 + (0.15 if m in (8, 9) else 0), "NDBI": -0.02,
                         "LST_std": 2.0, "LST_city": lst + 1 if lst == lst else np.nan,
                         "NDVI_city": 0.12, "NDBI_city": 0.01, "valid_fraction_city": 0.85, "error": ""})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def write_lst_tifs(cfg, df, bounds_wgs84):
    """Monthly and annual GeoTIFFs (EPSG:4326, ~100 m) with a hot centre."""
    import rasterio
    from rasterio.transform import from_bounds
    from data_pipeline.extract_gee_data import annual_tif_path, monthly_tif_path
    minx, miny, maxx, maxy = bounds_wgs84
    W, H = 220, 200
    tr = from_bounds(minx, miny, maxx, maxy, W, H)
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot((xx - W / 2) / W, (yy - H / 2) / H)
    rng = np.random.default_rng(0)

    def write(path, arrs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=len(arrs), dtype="float32",
                           crs="EPSG:4326", transform=tr, nodata=-9999) as dst:
            for i, a in enumerate(arrs, 1):
                dst.write(a.astype("float32"), i)

    for r in df[df.status != "no_data"].itertuples():
        base = (r.LST if r.LST == r.LST else 30) + 4 * (0.5 - dist) + rng.normal(0, 0.6, (H, W))
        if r.status != "observed":
            base[rng.random((H, W)) < 0.85] = -9999
        write(monthly_tif_path(r.Year, r.Month), [base, np.full((H, W), 2.0)])
    for y, g in df[df.status == "observed"].groupby("Year"):
        base = g.LST.mean() + 4 * (0.5 - dist)
        write(annual_tif_path(y), [base, np.full((H, W), 0.2), np.full((H, W), 0.0), np.full((H, W), 20.0)])


def write_grid_tifs(cfg, years, bgrid_csv):
    """Aligned UTM grid rasters whose LST rises with synthetic building density."""
    import rasterio
    from rasterio.transform import from_origin
    from data_pipeline.spatial_grid import landsat_grid_tif, make_grid
    s = cfg.GRID_SCALE_M
    g = make_grid()
    b = pd.read_csv(bgrid_csv)[["grid_id", "building_density"]] if os.path.exists(bgrid_csv) else None
    if b is not None:
        g = g.merge(b, on="grid_id", how="left")
    else:
        g["building_density"] = 0.0
    ix0, ix1, iy0, iy1 = g.ix.min(), g.ix.max(), g.iy.min(), g.iy.max()
    W, H = ix1 - ix0 + 1, iy1 - iy0 + 1
    tr = from_origin(ix0 * s, (iy1 + 1) * s, s, s)
    rng = np.random.default_rng(3)
    for y in years:
        bands = np.full((5, H, W), -9999.0)
        for r in g.itertuples():
            row, col = iy1 - r.iy, r.ix - ix0
            dens = 0.0 if np.isnan(r.building_density) else float(r.building_density)
            ndvi = float(np.clip(0.38 - 0.32 * dens + rng.normal(0, 0.02), 0.08, 0.65))
            ndbi = float(np.clip(-0.15 + 0.40 * dens + rng.normal(0, 0.02), -0.25, 0.35))
            lst = float(32.5 + 8.5 * dens - 4.5 * (ndvi - 0.25) + rng.normal(0, 0.35) + (0.4 if y == 2025 else 0))
            bands[0, row, col] = lst
            bands[1, row, col] = ndvi
            bands[2, row, col] = ndbi
            bands[3, row, col] = 18
            bands[4, row, col] = (s / 30) ** 2 * r.study_fraction * 0.9
        with rasterio.open(landsat_grid_tif(y), "w", driver="GTiff", width=W, height=H, count=5,
                           dtype="float32", crs=cfg.GRID_CRS, transform=tr, nodata=-9999) as dst:
            dst.write(bands.astype("float32"))
