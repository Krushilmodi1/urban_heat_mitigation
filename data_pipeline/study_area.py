"""Study area: Ahmedabad municipal boundary + a true 10 km outward buffer.

The city boundary is read once (file, Earth Engine asset, or - for testing only -
the GAUL district), validated, buffered by BUFFER_M metres in a projected CRS
(UTM 43N) and saved as two separate GeoJSON files plus a small JSON record.

Every other module (Earth Engine extraction, building filtering, the grid and
all maps) loads these saved files, so the same two geometries are used
everywhere.

Nothing here silently swaps the municipal boundary for the district: if no
municipal boundary is available the step stops with instructions.
"""
import hashlib
import json
import logging
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

log = logging.getLogger("study_area")


class BoundaryError(RuntimeError):
    """The municipal boundary is missing or failed validation."""


# --------------------------------------------------------------------- helpers
def _transformer(src, dst):
    from pyproj import Transformer
    return Transformer.from_crs(src, dst, always_xy=True)


def reproject(geom, src, dst):
    import numpy as np
    import shapely
    tr = _transformer(src, dst)
    return shapely.transform(geom, lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1])))


def _polygonal(geom):
    """Keep only the polygon parts of a geometry, repaired if invalid."""
    from shapely import make_valid
    from shapely.geometry import MultiPolygon, Polygon
    if not geom.is_valid:
        geom = make_valid(geom)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    if not parts:
        raise BoundaryError("The boundary contains no polygon geometry.")
    from shapely.ops import unary_union
    return unary_union(parts)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _geojson_crs(gj):
    """CRS named in an (old-style) GeoJSON 'crs' member; RFC 7946 files are WGS 84."""
    name = (gj.get("crs") or {}).get("properties", {}).get("name", "")
    if not name or "CRS84" in name or name.endswith(":4326") or name.endswith("::4326"):
        return "EPSG:4326"
    if "EPSG" in name:
        return "EPSG:" + name.replace("::", ":").split(":")[-1]
    raise BoundaryError(f"Unrecognised CRS in GeoJSON: {name}")


def _features_from_geojson(path):
    from shapely.geometry import shape
    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)
    crs = _geojson_crs(gj)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else (
        [gj] if gj.get("type") == "Feature" else [{"type": "Feature", "properties": {}, "geometry": gj}])
    out = [(f.get("properties") or {}, shape(f["geometry"])) for f in feats if f.get("geometry")]
    return out, crs


def _features_from_other(path):
    """Shapefile / GeoPackage / KML via geopandas (optional dependency)."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise BoundaryError(
            f"Reading {os.path.basename(path)} needs geopandas (pip install geopandas). "
            "Alternatively save the boundary as GeoJSON.") from exc
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise BoundaryError(f"{path} has no coordinate reference system (.prj missing?).")
    crs = gdf.crs.to_string()
    props = gdf.drop(columns="geometry").to_dict("records")
    return list(zip(props, gdf.geometry)), crs


def _read_boundary_file(path):
    if not os.path.exists(path):
        raise BoundaryError(
            f"Municipal boundary file not found: {path}\n"
            "Provide the Ahmedabad Municipal Corporation (AMC) limits as GeoJSON / shapefile / "
            "GeoPackage and set config.BOUNDARY_FILE to it. Satellite extraction, building "
            "filtering and maps are stopped until then (the district boundary is NOT used instead).")
    if path.lower().endswith(".zip"):
        target = os.path.join(config.STUDY_AREA_DIR, "_unzipped_boundary")
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target)
        cands = [os.path.join(r, f) for r, _, fs in os.walk(target) for f in fs
                 if f.lower().endswith((".geojson", ".json", ".shp", ".gpkg", ".kml"))]
        if len(cands) != 1:
            raise BoundaryError(f"Expected exactly one boundary layer in {path}, found: {cands}")
        path = cands[0]
    if path.lower().endswith((".geojson", ".json")):
        return _features_from_geojson(path)
    return _features_from_other(path)


def _filter_by_name(features):
    field, value = config.BOUNDARY_NAME_FIELD, config.BOUNDARY_NAME_VALUE
    if not field:
        return features
    kept = [(p, g) for p, g in features if str(p.get(field)) == str(value)]
    if not kept:
        values = sorted({str(p.get(field)) for p, _ in features})[:20]
        raise BoundaryError(f"No feature with {field} == {value!r}. Values present: {values}")
    return kept


# --------------------------------------------------------------------- sources
def _city_from_file():
    from shapely.ops import unary_union
    feats, crs = _read_boundary_file(config.BOUNDARY_FILE)
    feats = _filter_by_name(feats)
    geom = unary_union([g for _, g in feats])
    if crs != "EPSG:4326":
        geom = reproject(geom, crs, "EPSG:4326")
    meta = {"source": f"file: {os.path.basename(config.BOUNDARY_FILE)}",
            "source_sha256": _sha256(config.BOUNDARY_FILE), "source_crs": crs,
            "n_features": len(feats), "kind": "municipal",
            "label": "Ahmedabad municipal boundary"}
    return geom, meta


def _city_from_ee(kind):
    from shapely.geometry import shape
    from data_pipeline.extract_gee_data import init_ee
    ee = init_ee()
    if kind == "ee_asset":
        if not config.BOUNDARY_ASSET:
            raise BoundaryError("BOUNDARY_SOURCE is 'ee_asset' but BOUNDARY_ASSET is not set.")
        fc = ee.FeatureCollection(config.BOUNDARY_ASSET)
        if config.BOUNDARY_NAME_FIELD:
            fc = fc.filter(ee.Filter.eq(config.BOUNDARY_NAME_FIELD, config.BOUNDARY_NAME_VALUE))
        meta = {"source": f"Earth Engine asset {config.BOUNDARY_ASSET}", "kind": "municipal",
                "label": "Ahmedabad municipal boundary"}
    else:
        fc = ee.FeatureCollection(config.GAUL_DATASET)
        for key, value in config.GAUL_FILTERS.items():
            fc = fc.filter(ee.Filter.eq(key, value))
        meta = {"source": f"{config.GAUL_DATASET} {config.GAUL_FILTERS}", "kind": "district",
                "label": "Ahmedabad DISTRICT (GAUL) - not the municipal boundary"}
        log.warning("Using the GAUL DISTRICT boundary. This is NOT the Ahmedabad municipal area; "
                    "outputs are labelled 'district'. Use only to test the pipeline.")
    n = fc.size().getInfo()
    if n == 0:
        raise BoundaryError(f"No boundary feature found for {meta['source']}.")
    meta["n_features"] = n
    return shape(fc.geometry().getInfo()), meta


# ------------------------------------------------------------------ main logic
def build_study_area(city_wgs84, meta):
    """Validate the city polygon, buffer it in UTM, return (city, buffer, record)."""
    from shapely.geometry import Point
    city_utm = _polygonal(reproject(_polygonal(city_wgs84), "EPSG:4326", config.PROJECTED_CRS))
    # One canonical, lightly simplified city outline: used for EE, building filtering and maps alike.
    city_utm = _polygonal(city_utm.simplify(config.GEOMETRY_SIMPLIFY_M, preserve_topology=True))
    buffer_utm = _polygonal(city_utm.buffer(config.BUFFER_M, quad_segs=32))

    city_km2 = city_utm.area / 1e6
    buffer_km2 = buffer_utm.area / 1e6
    centre = reproject(Point(config.EXPECTED_CENTRE), "EPSG:4326", config.PROJECTED_CRS)
    contains_centre = city_utm.contains(centre)

    # The buffer edge must be BUFFER_M from the city everywhere (within the arc tolerance).
    ring = buffer_utm.exterior if buffer_utm.geom_type == "Polygon" else max(buffer_utm.geoms, key=lambda g: g.area).exterior
    import numpy as np
    sample = [ring.interpolate(f, normalized=True) for f in np.linspace(0, 1, 400, endpoint=False)]
    dists = np.array([city_utm.distance(p) for p in sample])
    record = dict(meta)
    record.update({
        "projected_crs": config.PROJECTED_CRS, "buffer_m": config.BUFFER_M,
        "simplify_tolerance_m": config.GEOMETRY_SIMPLIFY_M,
        "city_area_km2": round(city_km2, 2), "study_area_km2": round(buffer_km2, 2),
        "buffer_ring_km2": round(buffer_km2 - city_km2, 2),
        "contains_expected_centre": bool(contains_centre),
        "buffer_distance_check_m": {"min": round(float(dists.min()), 1), "max": round(float(dists.max()), 1)},
        "bounds_wgs84": [round(v, 5) for v in reproject(buffer_utm, config.PROJECTED_CRS, "EPSG:4326").bounds],
    })

    log.info("Boundary: %s", record["source"])
    log.info("City %.1f km2, study area (city + %d km) %.1f km2; contains centre: %s; buffer distance %.0f-%.0f m",
             city_km2, config.BUFFER_M // 1000, buffer_km2, contains_centre, dists.min(), dists.max())

    if not contains_centre:
        raise BoundaryError("The boundary does not contain central Ahmedabad - wrong feature or CRS?")
    if abs(dists.min() - config.BUFFER_M) > 25 or abs(dists.max() - config.BUFFER_M) > 25:
        raise BoundaryError(f"Buffer check failed: edge is {dists.min():.0f}-{dists.max():.0f} m from the city.")
    lo, hi = config.CITY_AREA_KM2_RANGE
    if meta["kind"] == "municipal" and not lo <= city_km2 <= hi:
        raise BoundaryError(
            f"City area is {city_km2:.0f} km2, outside the plausible municipal range {lo}-{hi} km2. "
            "This is probably the district or a different unit. Check the file, or adjust "
            "CITY_AREA_KM2_RANGE if you are sure it is correct.")

    city = reproject(city_utm, config.PROJECTED_CRS, "EPSG:4326")
    buffered = reproject(buffer_utm, config.PROJECTED_CRS, "EPSG:4326")
    return city, buffered, record


def _write_geojson(geom, path, props):
    from shapely.geometry import mapping
    from shapely import set_precision
    gj = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": props,
                        "geometry": mapping(set_precision(geom, 1e-6))}]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gj, fh)
    log.info("Saved %s", os.path.relpath(path, config.BASE_DIR))


def prepare_study_area(force=False):
    """Create (or reuse) the saved city and buffer geometries. Returns the record dict."""
    config.ensure_dirs()
    if not force and os.path.exists(config.BOUNDARY_CONFIG_JSON) and os.path.exists(config.BUFFER_GEOJSON):
        with open(config.BOUNDARY_CONFIG_JSON) as fh:
            rec = json.load(fh)
        unchanged = (rec.get("buffer_m") == config.BUFFER_M and rec.get("boundary_source") == config.BOUNDARY_SOURCE
                     and (config.BOUNDARY_SOURCE != "file" or (os.path.exists(config.BOUNDARY_FILE)
                                                               and rec.get("source_sha256") == _sha256(config.BOUNDARY_FILE)))
                     and (config.BOUNDARY_SOURCE != "ee_asset" or str(config.BOUNDARY_ASSET) in rec.get("source", "")))
        if unchanged:
            log.info("Study area already prepared (%s) - reusing.", rec["source"])
            return rec
        log.info("Boundary settings or file changed - rebuilding the study area.")

    src = config.BOUNDARY_SOURCE
    if src == "file":
        city, meta = _city_from_file()
    elif src in ("ee_asset", "gaul_district"):
        city, meta = _city_from_ee(src)
    else:
        raise BoundaryError(f"Unknown BOUNDARY_SOURCE {src!r}")

    city, buffered, rec = build_study_area(city, meta)
    rec["boundary_source"] = src
    _write_geojson(city, config.BOUNDARY_GEOJSON,
                   {"name": rec["label"], "source": rec["source"], "area_km2": rec["city_area_km2"]})
    _write_geojson(buffered, config.BUFFER_GEOJSON,
                   {"name": f"{rec['label']} + {config.BUFFER_M // 1000} km buffer", "source": rec["source"],
                    "area_km2": rec["study_area_km2"], "buffer_m": config.BUFFER_M})
    with open(config.BOUNDARY_CONFIG_JSON, "w") as fh:
        json.dump(rec, fh, indent=2)
    log.info("Saved %s", os.path.relpath(config.BOUNDARY_CONFIG_JSON, config.BASE_DIR))
    return rec


# ---------------------------------------------------------------------- loaders
def _load(path):
    from shapely.geometry import shape
    from shapely.ops import unary_union
    if not os.path.exists(path):
        raise BoundaryError(f"{path} not found - run:  python main.py --stage boundary")
    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)
    return unary_union([shape(f["geometry"]) for f in gj["features"]])


def load_study_area(crs="EPSG:4326"):
    """(city, buffered) shapely geometries from the saved files, in the requested CRS."""
    city, buffered = _load(config.BOUNDARY_GEOJSON), _load(config.BUFFER_GEOJSON)
    if crs != "EPSG:4326":
        city, buffered = reproject(city, "EPSG:4326", crs), reproject(buffered, "EPSG:4326", crs)
    return city, buffered


def study_area_record():
    if not os.path.exists(config.BOUNDARY_CONFIG_JSON):
        return {"label": "Ahmedabad boundary", "kind": "unknown", "source": "unknown"}
    with open(config.BOUNDARY_CONFIG_JSON) as fh:
        return json.load(fh)


def to_ee(geom):
    """shapely (WGS 84) -> ee.Geometry with planar edges (vertices are dense, edges short)."""
    from shapely.geometry import mapping
    from data_pipeline.extract_gee_data import init_ee
    ee = init_ee()
    return ee.Geometry(mapping(geom), None, False)


def load_study_area_ee():
    city, buffered = load_study_area()
    return to_ee(city), to_ee(buffered)
