"""Open Buildings from YOUR uploaded CSV (no Earth Engine download).

One pass over the CSV, in chunks, does both jobs:

1. Inspection  - columns, data types, row count, missing values, coordinate
                 ranges, geometry type, confidence / area ranges, duplicates,
                 invalid geometries -> csv_inspection_report.md
2. Processing  - confidence filter (only if a confidence column exists),
                 spatial filter to the city + 10 km buffer, footprint areas in
                 UTM 43N, zone (city / buffer ring), grid cell assignment

What can be calculated depends on what the CSV holds:

  polygons (WKT)  footprint map, exact areas, density, counts      -> full analysis
  points only     counts and count density; footprint area only if
                  the CSV has an area column (taken as given, not
                  measured)                                        -> partial
  no coordinates  no spatial filtering, no grid, no maps           -> stops with an explanation

The original CSV is only read, never modified. Outputs go to
data/processed/open_buildings/.

Attribution: W. Sirko et al. (2021), "Continental-scale building detection from
high resolution satellite imagery", arXiv:2107.12283. Open Buildings data are
licensed CC BY 4.0 and ODbL v1.0 - https://sites.research.google/gr/open-buildings/
"""
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

log = logging.getLogger("open_buildings")

# Exact column names published by Google Research Open Buildings, per internal role.
KNOWN_COLUMNS = {
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "area_m2": ["area_in_meters", "area_m2", "area_sq_m"],
    "confidence": ["confidence"],
    "geometry_wkt": ["geometry", "wkt", "geom"],
    "building_id": ["full_plus_code", "building_id", "id"],
}


class BuildingDataError(RuntimeError):
    """The CSV is missing or cannot support the requested analysis."""


# ------------------------------------------------------------------ schema
def resolve_columns(header):
    """Map internal roles to actual CSV columns: config overrides first, then exact known names."""
    lower = {c.lower().strip(): c for c in header}
    mapping, notes = {}, []
    for role, candidates in KNOWN_COLUMNS.items():
        override = config.OPEN_BUILDINGS_COLUMNS.get(role)
        if override:
            if override not in header:
                raise BuildingDataError(f"config.OPEN_BUILDINGS_COLUMNS['{role}'] = {override!r} is not a column "
                                        f"in the CSV. Columns are: {list(header)}")
            mapping[role] = override
            notes.append(f"{role}: '{override}' (set in config)")
            continue
        found = next((lower[c] for c in candidates if c in lower), None)
        mapping[role] = found
        notes.append(f"{role}: '{found}' (auto-detected)" if found else f"{role}: not present")
    return mapping, notes


def representation(mapping):
    if mapping["geometry_wkt"]:
        return "polygons"
    if mapping["latitude"] and mapping["longitude"]:
        return "points"
    return "no_coordinates"


# ------------------------------------------------------------------ geometry
def _to_utm_fn():
    from pyproj import Transformer
    tr = Transformer.from_crs(config.OPEN_BUILDINGS_CRS, config.PROJECTED_CRS, always_xy=True)

    def fn(coords):
        x, y = tr.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])
    return fn, tr


def _parse_wkt(series):
    """WKT -> shapely array; unparsable -> None. Returns (geoms, n_unparsable, n_invalid_repaired)."""
    import shapely
    geoms = shapely.from_wkt(series.fillna("").astype(str).values, on_invalid="ignore")
    empty = series.isna().values | (series.fillna("").astype(str).str.len().values == 0)
    unparsable = int(((geoms == None) | shapely.is_empty(geoms)).sum() - empty.sum())  # noqa: E711
    ok = (geoms != None)  # noqa: E711
    invalid = ok.copy()
    invalid[ok] = ~shapely.is_valid(geoms[ok])
    n_invalid = int(invalid.sum())
    if n_invalid:
        geoms[invalid] = shapely.make_valid(geoms[invalid])
    return geoms, max(unparsable, 0), n_invalid


# -------------------------------------------------------------------- main
def process_buildings(csv_path=None, force=False):
    """Inspect + filter + aggregate the uploaded CSV. Returns the summary DataFrame."""
    import shapely
    from data_pipeline import spatial_grid
    from data_pipeline import study_area

    csv_path = csv_path or config.OPEN_BUILDINGS_CSV
    if not os.path.exists(csv_path):
        raise BuildingDataError(
            f"Open Buildings CSV not found: {csv_path}\n"
            "Copy your uploaded CSV to data/raw/ (or set config.OPEN_BUILDINGS_CSV). "
            "Building metrics, building maps and LST-building correlations cannot run without it.")
    config.ensure_dirs()
    meta_path = config.BUILDINGS_META_JSON
    stamp = {"csv": os.path.abspath(csv_path), "size": os.path.getsize(csv_path),
             "mtime": int(os.path.getmtime(csv_path)), "conf": config.BUILDING_CONFIDENCE_MIN,
             "grid": config.GRID_SCALE_M, "columns": config.OPEN_BUILDINGS_COLUMNS,
             "study_area": study_area.study_area_record().get("study_area_km2")}
    if not force and os.path.exists(meta_path) and os.path.exists(config.BUILDING_SUMMARY_CSV):
        with open(meta_path) as fh:
            if json.load(fh).get("input_stamp") == stamp:
                log.info("Buildings already processed for this CSV and settings - reusing.")
                return pd.read_csv(config.BUILDING_SUMMARY_CSV)

    header = list(pd.read_csv(csv_path, nrows=0).columns)
    mapping, map_notes = resolve_columns(header)
    rep = representation(mapping)
    log.info("CSV columns: %s", header)
    log.info("Column mapping: %s", "; ".join(map_notes))
    log.info("Representation: %s", rep)

    city_utm, buffer_utm = study_area.load_study_area(config.PROJECTED_CRS)
    city_wgs, buffer_wgs = study_area.load_study_area()
    shapely.prepare(buffer_utm); shapely.prepare(city_utm)
    shapely.prepare(buffer_wgs)
    minx, miny, maxx, maxy = buffer_wgs.bounds
    pad = 0.01   # deg, so footprints straddling the edge are not lost before the exact test
    to_utm, tr = _to_utm_fn()
    s = config.GRID_SCALE_M

    # --- running inspection statistics
    insp = {"rows": 0, "missing": defaultdict(int), "dtypes": {}, "lat_range": [np.inf, -np.inf],
            "lon_range": [np.inf, -np.inf], "conf_range": [np.inf, -np.inf], "area_csv_range": [np.inf, -np.inf],
            "geom_types": defaultdict(int), "unparsable_wkt": 0, "invalid_repaired": 0,
            "duplicates": 0, "below_confidence": 0, "outside_study_area": 0, "missing_location": 0,
            "area_check": []}
    seen = set()
    zone_areas = {"city": [], "buffer_ring": []}
    grid_acc = defaultdict(lambda: [0, 0.0])          # grid_id -> [count, footprint m2]
    clean_written = False
    if os.path.exists(config.BUILDINGS_CLEAN_CSV):
        os.remove(config.BUILDINGS_CLEAN_CSV)
    gpkg_frames = []
    want_gpkg = rep == "polygons"

    reader = pd.read_csv(csv_path, chunksize=config.CSV_CHUNK_ROWS, low_memory=False)
    for k, chunk in enumerate(reader, 1):
        n0 = len(chunk)
        insp["rows"] += n0
        if not insp["dtypes"]:
            insp["dtypes"] = {c: str(t) for c, t in chunk.dtypes.items()}
        for c in header:
            insp["missing"][c] += int(chunk[c].isna().sum())

        lat = pd.to_numeric(chunk[mapping["latitude"]], errors="coerce") if mapping["latitude"] else None
        lon = pd.to_numeric(chunk[mapping["longitude"]], errors="coerce") if mapping["longitude"] else None
        if lat is not None and lat.notna().any():
            insp["lat_range"] = [min(insp["lat_range"][0], lat.min()), max(insp["lat_range"][1], lat.max())]
            insp["lon_range"] = [min(insp["lon_range"][0], lon.min()), max(insp["lon_range"][1], lon.max())]
        conf = pd.to_numeric(chunk[mapping["confidence"]], errors="coerce") if mapping["confidence"] else None
        if conf is not None and conf.notna().any():
            insp["conf_range"] = [min(insp["conf_range"][0], conf.min()), max(insp["conf_range"][1], conf.max())]
        area_csv = pd.to_numeric(chunk[mapping["area_m2"]], errors="coerce") if mapping["area_m2"] else None
        if area_csv is not None and area_csv.notna().any():
            insp["area_csv_range"] = [min(insp["area_csv_range"][0], area_csv.min()),
                                      max(insp["area_csv_range"][1], area_csv.max())]

        if rep == "no_coordinates":
            continue

        keep = np.ones(n0, dtype=bool)
        # 1. confidence (only when the column exists)
        if conf is not None and config.BUILDING_CONFIDENCE_MIN is not None:
            low = (conf < config.BUILDING_CONFIDENCE_MIN).fillna(True).values
            insp["below_confidence"] += int((low & keep).sum())
            keep &= ~low

        # Fast bounding-box check if lat and lon columns exist
        cand = keep
        if lat is not None and lon is not None:
            lonv = lon.values.astype(float)
            latv = lat.values.astype(float)
            in_box = (lonv >= minx - pad) & (lonv <= maxx + pad) & (latv >= miny - pad) & (latv <= maxy + pad)
            cand = keep & np.nan_to_num(in_box, nan=0).astype(bool)
            if not cand.any():
                insp["outside_study_area"] += int(keep.sum())
                log.info("Chunk %d: %d rows, 0 in study area (bbox skipped)", k, n0)
                continue

        # 2. location -> geometry / centroid
        geoms = None
        if rep == "polygons":
            cand_idx = np.flatnonzero(cand) if (lat is not None and lon is not None) else np.arange(n0)
            geoms_cand, bad, rep_inv = _parse_wkt(chunk[mapping["geometry_wkt"]].iloc[cand_idx])
            insp["unparsable_wkt"] += bad
            insp["invalid_repaired"] += rep_inv
            has = geoms_cand != None  # noqa: E711
            for t, n in pd.Series(shapely.get_type_id(geoms_cand[has])).value_counts().items():
                insp["geom_types"][int(t)] += int(n)

            if lat is None or lon is None:
                cx = np.full(n0, np.nan); cy = np.full(n0, np.nan)
                cent = shapely.centroid(geoms_cand[has])
                cx[cand_idx[has]], cy[cand_idx[has]] = shapely.get_x(cent), shapely.get_y(cent)
                lon, lat = pd.Series(cx, index=chunk.index), pd.Series(cy, index=chunk.index)
                lonv, latv = lon.values.astype(float), lat.values.astype(float)
                in_box = (lonv >= minx - pad) & (lonv <= maxx + pad) & (latv >= miny - pad) & (latv <= maxy + pad)
                cand = keep & np.nan_to_num(in_box, nan=0).astype(bool)

            geoms = np.empty(n0, dtype=object)
            geoms[cand_idx] = geoms_cand
            nogeom = np.ones(n0, dtype=bool)
            nogeom[cand_idx[has]] = False
        else:
            nogeom = (lat.isna() | lon.isna()).values
        insp["missing_location"] += int((nogeom & keep).sum())
        keep &= ~nogeom

        # 3. exact test on candidate subset
        idx = np.flatnonzero(cand & ~nogeom)
        if rep == "polygons" and len(idx):
            g_utm = shapely.transform(geoms[idx], to_utm)
            inside = shapely.intersects(buffer_utm, g_utm)
        elif len(idx):
            inside = shapely.contains_xy(buffer_wgs, lonv[idx], latv[idx])
        else:
            inside = np.zeros(0, dtype=bool)
        insp["outside_study_area"] += int(keep.sum() - inside.sum())
        idx_in = idx[inside]
        if not len(idx_in):
            log.info("Chunk %d: %d rows, 0 in study area", k, n0)
            continue

        sub = pd.DataFrame(index=chunk.index[idx_in])
        sub["building_id"] = (chunk[mapping["building_id"]].iloc[idx_in].astype(str).values
                              if mapping["building_id"] else None)
        if rep == "polygons":
            gu = g_utm[inside]
            cu = shapely.centroid(gu)
            sub["x_utm"], sub["y_utm"] = shapely.get_x(cu), shapely.get_y(cu)
            sub["area_m2"] = shapely.area(gu)
            sub["area_source"] = "computed_utm43n"
            wkt = chunk[mapping["geometry_wkt"]].iloc[idx_in].values
        else:
            x, y = tr.transform(lonv[idx_in], latv[idx_in])
            sub["x_utm"], sub["y_utm"] = x, y
            sub["area_m2"] = area_csv.iloc[idx_in].values if area_csv is not None else np.nan
            sub["area_source"] = "csv_column" if area_csv is not None else "unavailable"
            wkt = None
        sub["lon"], sub["lat"] = lonv[idx_in], latv[idx_in]
        sub["area_m2_csv"] = area_csv.iloc[idx_in].values if area_csv is not None else np.nan
        sub["confidence"] = conf.iloc[idx_in].values if conf is not None else np.nan

        # 4. duplicates (by id if present, else by rounded location + area)
        keys = (sub.building_id.values if mapping["building_id"]
                else (np.round(sub.x_utm, 1).astype(str) + "_" + np.round(sub.y_utm, 1).astype(str)
                      + "_" + np.round(sub.area_m2.fillna(-1), 1).astype(str)).values)
        dup = np.array([key in seen or seen.add(key) for key in keys], dtype=bool)  # set.add returns None
        insp["duplicates"] += int(dup.sum())
        sub = sub[~dup]
        if wkt is not None:
            wkt = wkt[~dup]

        # 5. zone by centroid; grid cell by centroid
        in_city = shapely.contains_xy(city_utm, sub.x_utm.values, sub.y_utm.values)
        sub["zone"] = np.where(in_city, "city", "buffer_ring")
        ix, iy = np.floor(sub.x_utm / s).astype(int), np.floor(sub.y_utm / s).astype(int)
        sub["grid_id"] = spatial_grid.grid_ids(ix, iy)
        if rep == "polygons" and area_csv is not None:
            both = sub[["area_m2", "area_m2_csv"]].dropna()
            if len(both):
                insp["area_check"].append(both.sample(min(len(both), 2000), random_state=0))
        if wkt is not None:
            sub["geometry_wkt"] = wkt

        for z in ("city", "buffer_ring"):
            zone_areas[z].append(sub.loc[sub.zone == z, "area_m2"].values)
        agg = sub.groupby("grid_id").area_m2.agg(["size", "sum"])
        for gid, row in agg.iterrows():
            grid_acc[gid][0] += int(row["size"]); grid_acc[gid][1] += float(row["sum"])

        sub.to_csv(config.BUILDINGS_CLEAN_CSV, mode="a", header=not clean_written, index=False)
        clean_written = True
        if want_gpkg:
            gpkg_frames.append(sub)
        log.info("Chunk %d: %d rows, %d kept in study area", k, n0, len(sub))

    if rep == "no_coordinates":
        _write_inspection(csv_path, header, mapping, map_notes, rep, insp, None)
        raise BuildingDataError(
            "The CSV has neither a geometry column nor latitude/longitude columns, so buildings cannot be "
            "filtered to Ahmedabad, gridded or mapped. Set config.OPEN_BUILDINGS_COLUMNS if the columns "
            "exist under other names. See csv_inspection_report.md.")
    if insp["rows"] == 0:
        raise BuildingDataError("The CSV has no data rows.")
    _check_coverage(insp, buffer_wgs)

    zone_areas = {z: (np.concatenate(v) if v else np.array([])) for z, v in zone_areas.items()}
    summary = _summary(zone_areas, city_utm, buffer_utm, rep, mapping)
    summary.to_csv(config.BUILDING_SUMMARY_CSV, index=False)
    log.info("Saved %s", config.BUILDING_SUMMARY_CSV)

    grid = spatial_grid.make_grid()
    bg = pd.DataFrame([(g, c, a) for g, (c, a) in grid_acc.items()],
                      columns=["grid_id", "building_count", "footprint_area_m2"])
    bg = grid[["grid_id", "zone", "cell_area_in_study_m2", "study_fraction"]].merge(bg, on="grid_id", how="left")
    bg["building_count"] = bg.building_count.fillna(0).astype(int)
    bg["footprint_area_m2"] = bg.footprint_area_m2.fillna(0.0)
    area_ok = summary.loc[summary.zone == "study_area", "area_source"].iloc[0] != "unavailable"
    if not area_ok:
        bg["footprint_area_m2"] = np.nan
    bg = add_density_columns(bg)
    lost = set(grid_acc) - set(grid.grid_id)
    if lost:
        log.warning("%d building cell id(s) fall outside the analysis grid (edge rounding).", len(lost))
    bg.to_csv(config.BUILDINGS_GRID_CSV, index=False)
    log.info("Saved %s", config.BUILDINGS_GRID_CSV)

    if want_gpkg and gpkg_frames:
        _write_gpkg(pd.concat(gpkg_frames, ignore_index=True))
    _write_inspection(csv_path, header, mapping, map_notes, rep, insp, summary)
    with open(meta_path, "w") as fh:
        json.dump({"input_stamp": stamp, "representation": rep, "column_mapping": mapping,
                   "confidence_threshold_applied": (config.BUILDING_CONFIDENCE_MIN
                                                    if mapping["confidence"] else None),
                   "rows_read": insp["rows"], "rows_kept": int(summary.loc[summary.zone == "study_area",
                                                                           "building_count"].iloc[0]),
                   "grid_size_m": s, "crs_input": config.OPEN_BUILDINGS_CRS, "crs_metrics": config.PROJECTED_CRS,
                   "source_note": config.BUILDINGS_SOURCE_NOTE}, fh, indent=2, default=str)
    return summary


def add_density_columns(bg):
    """Footprint density (fraction of the cell's in-study area) and buildings per km2."""
    area = bg.cell_area_in_study_m2.where(bg.cell_area_in_study_m2 > 0)
    bg["building_density"] = (bg.footprint_area_m2 / area).clip(upper=1.0)   # fraction 0-1
    bg["buildings_per_km2"] = bg.building_count / (area / 1e6)
    bg["mean_footprint_m2"] = np.where(bg.building_count > 0, bg.footprint_area_m2 / bg.building_count.where(bg.building_count > 0), np.nan)
    return bg


def _summary(zone_areas, city_utm, buffer_utm, rep, mapping):
    area_source = ("computed_utm43n" if rep == "polygons" else ("csv_column" if mapping["area_m2"] else "unavailable"))
    zones = {"city": (zone_areas["city"], city_utm.area),
             "buffer_ring": (zone_areas["buffer_ring"], buffer_utm.area - city_utm.area),
             "study_area": (np.concatenate([zone_areas["city"], zone_areas["buffer_ring"]]), buffer_utm.area)}
    rows = []
    for z, (a, zone_m2) in zones.items():
        a_valid = a[~np.isnan(a)] if len(a) else a
        has_area = area_source != "unavailable" and len(a_valid) > 0
        total = float(a_valid.sum()) if has_area else np.nan
        rows.append({
            "zone": z, "zone_area_km2": zone_m2 / 1e6, "building_count": int(len(a)),
            "buildings_per_km2": len(a) / (zone_m2 / 1e6),
            "total_footprint_km2": total / 1e6 if has_area else np.nan,
            "footprint_density_pct": 100 * total / zone_m2 if has_area else np.nan,
            "mean_footprint_m2": float(a_valid.mean()) if has_area else np.nan,
            "median_footprint_m2": float(np.median(a_valid)) if has_area else np.nan,
            "representation": rep, "area_source": area_source,
            "confidence_min": config.BUILDING_CONFIDENCE_MIN if mapping["confidence"] else None,
        })
    return pd.DataFrame(rows)


def _check_coverage(insp, buffer_wgs):
    """Warn if the CSV's extent does not cover the whole study area."""
    if not np.isfinite(insp["lat_range"][0]):
        return
    minx, miny, maxx, maxy = buffer_wgs.bounds
    lo_lon, hi_lon = insp["lon_range"]; lo_lat, hi_lat = insp["lat_range"]
    if not (-180 <= lo_lon <= 180 and -90 <= lo_lat <= 90):
        raise BuildingDataError(f"Coordinates are not in degrees (lon {lo_lon}..{hi_lon}, lat {lo_lat}..{hi_lat}). "
                                "Set config.OPEN_BUILDINGS_CRS to the CSV's projected CRS.")
    covers = lo_lon <= minx and hi_lon >= maxx and lo_lat <= miny and hi_lat >= maxy
    insp["covers_study_area"] = bool(covers)
    if not covers:
        log.warning("The CSV extent (lon %.3f-%.3f, lat %.3f-%.3f) does NOT fully cover the study area "
                    "(lon %.3f-%.3f, lat %.3f-%.3f). Cells outside the CSV extent will show zero buildings, "
                    "which is missing data, not absence of buildings.",
                    lo_lon, hi_lon, lo_lat, hi_lat, minx, maxx, miny, maxy)


def _write_gpkg(df):
    try:
        import geopandas as gpd
        import shapely
    except ImportError:
        log.info("geopandas not installed - footprints saved as WKT in %s (opens in QGIS as a delimited "
                 "text layer, EPSG:4326).", os.path.basename(config.BUILDINGS_CLEAN_CSV))
        return
    gdf = gpd.GeoDataFrame(df.drop(columns=["geometry_wkt"]),
                           geometry=shapely.from_wkt(df.geometry_wkt.values), crs=config.OPEN_BUILDINGS_CRS)
    gdf.to_file(config.BUILDINGS_GPKG, driver="GPKG")
    log.info("Saved %s", config.BUILDINGS_GPKG)


def _fmt_range(r, p=4):
    return "n/a" if not np.isfinite(r[0]) else f"{r[0]:.{p}f} to {r[1]:.{p}f}"


def _write_inspection(path, header, mapping, notes, rep, insp, summary):
    type_names = {0: "Point", 3: "Polygon", 6: "MultiPolygon", 7: "GeometryCollection"}
    lines = ["# Open Buildings CSV - inspection report", "",
             f"- File: `{os.path.basename(path)}` ({os.path.getsize(path) / 1e6:,.1f} MB) - read only, not modified",
             f"- Rows: {insp['rows']:,}",
             f"- Representation: **{rep}**",
             f"- Coordinate reference system assumed: {config.OPEN_BUILDINGS_CRS} "
             "(Google Open Buildings publishes WGS 84 degrees)",
             f"- Latitude range: {_fmt_range(insp['lat_range'])}; longitude range: {_fmt_range(insp['lon_range'])}",
             f"- Covers the whole study area: {insp.get('covers_study_area', 'n/a')}",
             f"- Confidence range: {_fmt_range(insp['conf_range'], 3)}",
             f"- Area column range (m2): {_fmt_range(insp['area_csv_range'], 1)}",
             "", "## Column mapping", ""] + [f"- {n}" for n in notes]
    lines += ["", "## Columns, types and missing values", "", "| Column | Type | Missing |", "|---|---|---|"]
    for c in header:
        lines.append(f"| {c} | {insp['dtypes'].get(c, '?')} | {insp['missing'][c]:,} |")
    dated = [c for c in header if any(k in c.lower() for k in ("date", "year", "time"))]
    lines += ["", "## Time coverage", "",
              f"- Date-like columns: {', '.join(dated) if dated else 'none'}. "
              + ("" if dated else "Treated as a single static snapshot (no construction year).")]
    if rep == "polygons":
        lines += ["", "## Geometry", ""] + [f"- {type_names.get(t, t)}: {n:,}" for t, n in insp["geom_types"].items()]
        lines += [f"- Unparsable WKT: {insp['unparsable_wkt']:,}",
                  f"- Invalid geometries repaired (make_valid): {insp['invalid_repaired']:,}"]
        if insp["area_check"]:
            chk = pd.concat(insp["area_check"])
            rel = (chk.area_m2 - chk.area_m2_csv) / chk.area_m2_csv.where(chk.area_m2_csv > 0)
            lines.append(f"- Computed UTM area vs CSV area column: median difference {rel.median():+.2%}, "
                         f"90 % within {rel.abs().quantile(0.9):.2%} (n = {len(chk):,} sampled)")
    lines += ["", "## Filtering", "",
              f"- Below confidence threshold ({config.BUILDING_CONFIDENCE_MIN}): {insp['below_confidence']:,}"
              if mapping["confidence"] else "- No confidence column - no confidence filter applied",
              f"- Missing location / geometry: {insp['missing_location']:,}",
              f"- Outside city + {config.BUFFER_M // 1000} km buffer: {insp['outside_study_area']:,}",
              f"- Duplicates removed ({'by ' + mapping['building_id'] if mapping['building_id'] else 'by location + area'}): "
              f"{insp['duplicates']:,}"]
    if summary is not None:
        lines += ["", "## Result", "", summary.round(3).to_markdown(index=False) if _has_tabulate()
                  else summary.round(3).to_string(index=False)]
    lines += ["", "## Limitations", "",
              "- Buildings are assigned to a zone and grid cell by their centroid; a footprint crossing a cell "
              "edge is counted wholly in one cell.",
              "- Points-only data give counts, not measured footprint areas.",
              "- A single snapshot cannot show building change between years."]
    with open(config.BUILDINGS_SCHEMA_REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Saved %s", config.BUILDINGS_SCHEMA_REPORT)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
