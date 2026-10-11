"""Google Earth Engine extraction for Ahmedabad (municipal boundary + 10 km buffer).

Landsat 8 (+ Landsat 9) Collection 2 Level-2  ->  LST (deg C), NDVI, NDBI
  * study area = the saved city + buffer geometries from data_pipeline/study_area.py
  * pixel-level fill / cloud / cirrus / shadow / saturation masking (QA_PIXEL, QA_RADSAT),
    ST fill and out-of-range reflectance removed
  * monthly composites, per-pixel clear-observation counts and statistics computed server-side
  * months without usable data are recorded as such, never filled
  * monthly and annual composites downloaded once as GeoTIFF and reused; a manifest
    records the settings so a changed boundary / sensor set triggers re-extraction

Earth Engine is initialised lazily (init_ee), so importing this module never
needs a network connection. All plotting lives in visualization/plots.py and
works offline from the files written here.
"""
import hashlib
import json
import logging
import os
import shutil
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from data_pipeline import landsat as L  # noqa: E402

log = logging.getLogger("extract_gee")
BANDS = ["LST", "NDVI", "NDBI"]
NODATA = -9999
PROCESSING_VERSION = "2026-10-v2"   # bump when masking / compositing logic changes
_ee = None


# ------------------------------------------------------------------ Earth Engine
def init_ee():
    """Initialise Earth Engine once; fall back to interactive authentication."""
    global _ee
    if _ee is not None:
        return _ee
    import ee
    try:
        ee.Initialize(project=config.EE_PROJECT)
    except Exception as exc:  # no stored credentials, or expired
        log.warning("Earth Engine initialisation failed (%s). Starting authentication...", exc)
        ee.Authenticate()
        ee.Initialize(project=config.EE_PROJECT)
    log.info("Earth Engine initialised (project: %s)", config.EE_PROJECT)
    _ee = ee
    return ee


# -------------------------------------------------------------------- study area
def get_study_area(save=True, force=False):
    """Return (city_geometry, buffered_geometry) as ee.Geometry objects.

    Both come from the files written by study_area.prepare_study_area(), so the
    satellite statistics, building filtering and maps share identical geometries.
    """
    from data_pipeline import study_area
    if save:
        study_area.prepare_study_area(force=force)
    return study_area.load_study_area_ee()


# ----------------------------------------------------------------------- Landsat
def mask_and_scale(image):
    """Scaled LST / NDVI / NDBI with invalid pixels masked.

    LST mask:  QA_PIXEL bits 0-4 clear, QA_RADSAT == 0, ST_B10 not fill.
    NDVI/NDBI: additionally Red, NIR and SWIR1 inside the valid SR range.
    """
    ee = init_ee()
    clear = image.select("QA_PIXEL").bitwiseAnd(L.QA_CLEAR_MASK).eq(0)
    unsaturated = image.select("QA_RADSAT").eq(0)
    st_valid = image.select(L.THERMAL).neq(L.ST_FILL_DN)
    lo, hi = L.SR_VALID_DN
    opt = image.select(L.OPTICAL_USED)
    sr_valid = opt.gte(lo).And(opt.lte(hi)).reduce(ee.Reducer.min())   # all three bands valid

    base = clear.And(unsaturated)
    sr = opt.multiply(L.SR_MULT).add(L.SR_ADD)
    ndvi = sr.normalizedDifference([L.NIR, L.RED]).rename("NDVI")      # (NIR-Red)/(NIR+Red)
    ndbi = sr.normalizedDifference([L.SWIR1, L.NIR]).rename("NDBI")    # (SWIR1-NIR)/(SWIR1+NIR)
    lst = (image.select(L.THERMAL).multiply(L.ST_MULT).add(L.ST_ADD)   # Kelvin
           .subtract(L.KELVIN_OFFSET).rename("LST"))                   # -> deg C
    out = ee.Image.cat([lst.updateMask(base.And(st_valid)),
                        ndvi.updateMask(base.And(sr_valid)),
                        ndbi.updateMask(base.And(sr_valid))]).toFloat()
    return ee.Image(out.copyProperties(image, ["system:time_start", "SPACECRAFT_ID", "CLOUD_COVER"]))


def sensors():
    names = ["L8"] + (["L9"] if config.USE_LANDSAT9 else [])
    return {k: config.LANDSAT_COLLECTIONS[k] for k in names}


def _prepare_for(sensor):
    """Per-sensor mapping function (closure, so Earth Engine sees a one-argument function)."""
    ee = init_ee()

    def fn(img):
        return ee.Image(mask_and_scale(img).set("SENSOR", sensor))
    return fn


def landsat_collection(start_date, end_date_exclusive, bounds):
    """Masked Landsat 8 (+9) collection. NOTE: end date is EXCLUSIVE (Earth Engine convention)."""
    ee = init_ee()
    merged = None
    for name, cid in sensors().items():
        c = (ee.ImageCollection(cid)
             .filterBounds(bounds)
             .filterDate(start_date, end_date_exclusive)
             .filter(ee.Filter.lt("CLOUD_COVER", config.SCENE_CLOUD_MAX))
             .map(_prepare_for(name)))
        merged = c if merged is None else merged.merge(c)
    return merged


# Backwards-compatible name used by earlier versions of this project.
l8_data_collection = landsat_collection


def month_list(start_date=None, end_date=None):
    """All (year, month) pairs touched by the inclusive date range."""
    start = pd.Timestamp(start_date or config.START_DATE)
    end = pd.Timestamp(end_date or config.END_DATE)
    if end < start:
        raise ValueError(f"End date {end.date()} is before start date {start.date()}")
    return [(p.year, p.month) for p in pd.period_range(start, end, freq="M")]


def exclusive_end(end_date):
    """Inclusive YYYY-MM-DD -> the exclusive end Earth Engine expects (next day)."""
    return (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def month_window(year, month, start_date=None, end_date=None):
    """[start, end_exclusive) of one calendar month, clipped to the configured period."""
    m0 = pd.Timestamp(year=year, month=month, day=1)
    m1 = m0 + pd.offsets.MonthBegin(1)
    s = max(m0, pd.Timestamp(start_date or config.START_DATE))
    e = min(m1, pd.Timestamp(exclusive_end(end_date or config.END_DATE)))
    return s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")


def monthly_composite(year, month, region, start_date=None, end_date=None):
    """(collection, composite) for one calendar month.

    Composite bands: LST, NDVI, NDBI (per-pixel mean of clear observations) and
    LST_obs_count (number of clear LST observations per pixel). A fully masked
    placeholder keeps the bands present in empty months, so an empty month gives
    an all-masked image (no data) - never a substitute value.
    """
    ee = init_ee()
    s, e = month_window(year, month, start_date, end_date)
    coll = landsat_collection(s, e, region)
    empty = ee.Image.constant([0, 0, 0]).rename(BANDS).toFloat().updateMask(0)
    stack = coll.select(BANDS).merge(ee.ImageCollection([empty]))
    mean = stack.mean().toFloat()
    count = stack.select("LST").count().unmask(0).rename("LST_obs_count").toFloat()
    return coll, mean.addBands(count).clip(region)


def annual_composite(year, months, region, start_date=None, end_date=None):
    """Mean of the monthly composites, so each observed month has equal weight.

    LST_obs_count is the total number of clear observations over those months.
    """
    ee = init_ee()
    imgs = [monthly_composite(year, m, region, start_date, end_date)[1] for m in months]
    coll = ee.ImageCollection(imgs)
    return (coll.select(BANDS).mean()
            .addBands(coll.select("LST_obs_count").sum().rename("LST_obs_count"))
            .toFloat().clip(region))


# ------------------------------------------------------------ monthly statistics
def _region_stats(image, geometry):
    ee = init_ee()
    reducer = (ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
               .combine(ee.Reducer.count(), sharedInputs=True))
    return image.reduceRegion(reducer=reducer, geometry=geometry,
                              scale=config.STATS_SCALE_M, maxPixels=1e10, tileScale=4)


def extract_monthly_average(start_date=None, end_date=None, city=None, buffered=None):
    """Monthly mean LST / NDVI / NDBI for the study area and for the city alone.

    One server round trip per month. Every month in the period gets a row:
      status = observed               enough cloud-free pixels
               insufficient_coverage  scenes exist but < MIN_VALID_FRACTION is clear
               no_data                no scene / no clear pixel (values are NaN)
    """
    ee = init_ee()
    if city is None or buffered is None:
        city, buffered = get_study_area()

    ones = ee.Image.constant(1)
    totals = ee.Dictionary({
        "study": ones.reduceRegion(ee.Reducer.count(), buffered, config.STATS_SCALE_M,
                                   maxPixels=1e10, tileScale=4).get("constant"),
        "city": ones.reduceRegion(ee.Reducer.count(), city, config.STATS_SCALE_M,
                                  maxPixels=1e10, tileScale=4).get("constant"),
    }).getInfo()
    if not totals["study"] or not totals["city"]:
        raise RuntimeError("Study-area pixel count is zero - check the saved boundary geometry.")

    months = month_list(start_date, end_date)
    rows = []
    for i, (year, month) in enumerate(months, 1):
        coll, comp = monthly_composite(year, month, buffered, start_date, end_date)
        s_date, e_date = month_window(year, month, start_date, end_date)
        try:
            res = ee.Dictionary({
                "n_scenes": coll.size(),
                "n_L8": coll.filter(ee.Filter.eq("SENSOR", "L8")).size(),
                "n_L9": coll.filter(ee.Filter.eq("SENSOR", "L9")).size(),
                "first": coll.aggregate_min("system:time_start"),
                "last": coll.aggregate_max("system:time_start"),
                "study": _region_stats(comp.select(BANDS), buffered),
                "city": _region_stats(comp.select(BANDS), city),
                "obs": comp.select("LST_obs_count").reduceRegion(
                    ee.Reducer.mean(), buffered, config.STATS_SCALE_M, maxPixels=1e10, tileScale=4),
            }).getInfo()
            error = ""
        except Exception as exc:
            log.error("[%d/%d] %d-%02d statistics failed: %s", i, len(months), year, month, exc)
            res, error = {"n_scenes": None, "study": {}, "city": {}, "obs": {}}, str(exc)[:200]

        s, c = res["study"] or {}, res["city"] or {}
        valid = (s.get("LST_count") or 0) / totals["study"]
        valid_city = (c.get("LST_count") or 0) / totals["city"]
        if error:
            status = "extraction_failed"
        elif not res["n_scenes"] or s.get("LST_mean") is None:
            status = "no_data"
        elif valid < config.MIN_VALID_FRACTION:
            status = "insufficient_coverage"
        else:
            status = "observed"

        ts = lambda v: pd.to_datetime(v, unit="ms").strftime("%Y-%m-%d") if v else None  # noqa: E731
        rows.append({
            "Year": year, "Month": month, "Month_Name": config.MONTH_NAMES[month - 1],
            "Date": f"{year}-{month:02d}-01", "window_start": s_date, "window_end_exclusive": e_date,
            "status": status, "n_scenes": res.get("n_scenes"), "n_scenes_L8": res.get("n_L8"),
            "n_scenes_L9": res.get("n_L9"), "first_scene": ts(res.get("first")), "last_scene": ts(res.get("last")),
            "valid_pixels": s.get("LST_count"), "total_pixels": totals["study"],
            "valid_fraction": round(valid, 4),
            "mean_clear_obs_per_pixel": (res.get("obs") or {}).get("LST_obs_count"),
            "LST": s.get("LST_mean"), "NDVI": s.get("NDVI_mean"), "NDBI": s.get("NDBI_mean"),
            "LST_std": s.get("LST_stdDev"),
            "LST_city": c.get("LST_mean"), "NDVI_city": c.get("NDVI_mean"),
            "NDBI_city": c.get("NDBI_mean"), "valid_fraction_city": round(valid_city, 4),
            "error": error,
        })
        lst_txt = "n/a" if s.get("LST_mean") is None else f"{s['LST_mean']:.1f} C"
        log.info("[%d/%d] %d-%02d  %-21s scenes=%s  clear=%.0f%%  LST=%s",
                 i, len(months), year, month, status, res.get("n_scenes"), valid * 100, lst_txt)

    df = pd.DataFrame(rows)
    config.ensure_dirs()
    df.to_csv(config.MONTHLY_CSV, index=False)
    log.info("Saved %s", config.MONTHLY_CSV)
    failed = df[df.status == "extraction_failed"]
    if len(failed):
        log.error("%d month(s) failed to extract and are marked 'extraction_failed': %s. "
                  "Re-run to retry them.", len(failed), ", ".join(failed.Date))
    return df


# --------------------------------------------------------------------- downloads
def download_geotiff(image, region, path, scale=None, crs="EPSG:4326", force=False, retries=3,
                     crs_transform=None):
    """Download a (small) image as GeoTIFF. Masked pixels are written as -9999.

    Skips files that already exist, so re-runs make no Earth Engine request.
    If the request is too large the scale is doubled and retried (recorded in a
    .json sidecar). With crs_transform the pixel grid is fixed (no doubling).
    """
    if os.path.exists(path) and not force:
        log.info("Exists, skipping download: %s", os.path.basename(path))
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = image.unmask(NODATA).toFloat().clip(region)
    for attempt in range(1, retries + 1):
        try:
            params = {"region": region, "crs": crs, "format": "GEO_TIFF"}
            if crs_transform:
                params["crs_transform"] = crs_transform
            else:
                params["scale"] = scale
            url = image.getDownloadURL(params)
            with requests.get(url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(path + ".part", "wb") as fh:
                    shutil.copyfileobj(resp.raw, fh)
            os.replace(path + ".part", path)
            with open(os.path.splitext(path)[0] + ".json", "w") as fh:
                json.dump({"scale_m": scale, "crs": crs, "crs_transform": crs_transform,
                           "nodata": NODATA, "processing_version": PROCESSING_VERSION}, fh)
            log.info("Downloaded %s (%s)", os.path.basename(path), f"{scale} m" if scale else "fixed grid")
            return path
        except Exception as exc:
            msg = str(exc)
            log.warning("Download attempt %d failed for %s: %s", attempt, os.path.basename(path), msg[:200])
            too_big = any(k in msg.lower() for k in ("size", "too large", "limit"))
            if too_big and not crs_transform:
                scale *= 2
                log.warning("Request too large - retrying at %d m", scale)
            time.sleep(5 * attempt)
    log.error("Giving up on %s", path)
    return None


def monthly_tif_path(year, month):
    return os.path.join(config.LST_MONTHLY_TIF_DIR, str(year), f"ahmedabad_lst_{year}_{month:02d}.tif")


def annual_tif_path(year):
    return os.path.join(config.LST_ANNUAL_TIF_DIR, f"ahmedabad_annual_{year}.tif")


def download_lst_rasters(df, buffered, force=False, start_date=None, end_date=None):
    """Monthly LST GeoTIFFs (months with data) and annual LST/NDVI/NDBI GeoTIFFs.

    Monthly: band 1 = LST (deg C), band 2 = clear observations per pixel.
    Annual:  LST, NDVI, NDBI, LST_obs_count (observed months only).
    """
    has_data = df[df.status.isin(["observed", "insufficient_coverage"])]
    failed = []
    for i, row in enumerate(has_data.itertuples(), 1):
        log.info("Monthly raster %d/%d", i, len(has_data))
        _, comp = monthly_composite(int(row.Year), int(row.Month), buffered, start_date, end_date)
        ok = download_geotiff(comp.select(["LST", "LST_obs_count"]), buffered,
                              monthly_tif_path(row.Year, row.Month), config.EXPORT_SCALE_M, force=force)
        if ok is None:
            failed.append(f"{row.Year}-{row.Month:02d}")
    for year, grp in has_data.groupby("Year"):
        months = [int(m) for m in grp[grp.status == "observed"].Month]
        if not months:
            log.warning("%d has no observed month - no annual raster.", year)
            continue
        ok = download_geotiff(annual_composite(int(year), months, buffered, start_date, end_date), buffered,
                              annual_tif_path(int(year)), config.EXPORT_SCALE_M, force=force)
        if ok is None:
            failed.append(f"annual {year}")
    if failed:
        log.error("Raster downloads failed for: %s (maps for these will show 'no raster').", ", ".join(failed))
    return failed


# ---------------------------------------------------------------------- caching
def extraction_fingerprint(start_date, end_date):
    from data_pipeline import study_area
    rec = study_area.study_area_record()
    payload = {
        "version": PROCESSING_VERSION, "collections": sensors(), "cloud_max": config.SCENE_CLOUD_MAX,
        "stats_scale": config.STATS_SCALE_M, "export_scale": config.EXPORT_SCALE_M,
        "min_valid": config.MIN_VALID_FRACTION, "start": start_date, "end": end_date,
        "boundary": [rec.get("source"), rec.get("source_sha256"), rec.get("city_area_km2"), rec.get("buffer_m")],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16], payload


def _manifest_matches(fp):
    if not os.path.exists(config.EXTRACTION_MANIFEST):
        return False
    with open(config.EXTRACTION_MANIFEST) as fh:
        return json.load(fh).get("fingerprint") == fp


def run_pipeline(start_date=None, end_date=None, force=False):
    """Boundary -> monthly statistics -> rasters. Returns (DataFrame, city, buffered)."""
    start_date, end_date = start_date or config.START_DATE, end_date or config.END_DATE
    log.info("Extracting %s to %s (inclusive); sensors: %s", start_date, end_date, ", ".join(sensors()))
    city, buffered = get_study_area()
    fp, payload = extraction_fingerprint(start_date, end_date)
    if not force and os.path.exists(config.MONTHLY_CSV) and not _manifest_matches(fp):
        log.warning("Settings, period or boundary changed since the last extraction - re-extracting "
                    "and replacing cached rasters.")
        force = True
    if os.path.exists(config.MONTHLY_CSV) and not force:
        df = pd.read_csv(config.MONTHLY_CSV)
        if (df.status == "extraction_failed").any():
            log.info("Retrying months that failed previously.")
            df = extract_monthly_average(start_date, end_date, city, buffered)
        else:
            log.info("Monthly statistics already extracted - reusing %s", config.MONTHLY_CSV)
    else:
        df = extract_monthly_average(start_date, end_date, city, buffered)
    failed = download_lst_rasters(df, buffered, force=force, start_date=start_date, end_date=end_date)
    with open(config.EXTRACTION_MANIFEST, "w") as fh:
        json.dump({"fingerprint": fp, "settings": payload, "failed_downloads": failed,
                   "written": pd.Timestamp.now().isoformat(timespec="seconds")}, fh, indent=2)
    return df, city, buffered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline()
