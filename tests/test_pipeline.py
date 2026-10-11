"""Offline tests. Run from the project folder:

    python -m unittest discover -s tests -v

No Earth Engine access is needed. The smoke test uses SYNTHETIC data
(tests/synthetic_data.py) in a temporary copy of the folder layout.
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from data_pipeline import landsat as L  # noqa: E402
from data_pipeline import extract_gee_data as gee  # noqa: E402
import synthetic_data as syn  # noqa: E402


class TestLandsatArithmetic(unittest.TestCase):
    def test_surface_temperature(self):
        self.assertAlmostEqual(float(L.st_kelvin(0)), 149.0)
        dn = (300.0 - 149.0) / 0.00341802                    # DN that should give 300 K
        self.assertAlmostEqual(float(L.st_kelvin(dn)), 300.0, places=6)
        self.assertAlmostEqual(float(L.st_celsius(dn)), 26.85, places=6)

    def test_reflectance_and_valid_range(self):
        lo, hi = L.SR_VALID_DN
        self.assertAlmostEqual(float(L.sr_reflectance(lo)), 0.0, places=3)
        self.assertAlmostEqual(float(L.sr_reflectance(hi)), 1.0, places=3)

    def test_indices(self):
        red, nir, swir = 0.10, 0.40, 0.25
        self.assertAlmostEqual(float(L.ndvi(nir, red)), 0.6)
        self.assertAlmostEqual(float(L.ndbi(swir, nir)), (0.25 - 0.40) / 0.65)

    def test_qa_bits(self):
        self.assertTrue(L.qa_clear(21824))          # typical clear-land QA_PIXEL value (bits 0-4 = 0)
        for bit in range(5):
            self.assertFalse(L.qa_clear(21824 | (1 << bit)), f"bit {bit} should mask")
        self.assertTrue(L.qa_clear(21824 | (1 << 5)))   # snow bit is not in the mask


class TestDates(unittest.TestCase):
    def test_exclusive_end(self):
        self.assertEqual(gee.exclusive_end("2025-12-31"), "2026-01-01")
        self.assertEqual(gee.exclusive_end("2024-02-29"), "2024-03-01")

    def test_month_list(self):
        self.assertEqual(len(gee.month_list("2024-01-01", "2025-12-31")), 24)
        self.assertEqual(gee.month_list("2024-01-15", "2024-03-02"), [(2024, 1), (2024, 2), (2024, 3)])
        with self.assertRaises(ValueError):
            gee.month_list("2025-01-01", "2024-01-01")

    def test_month_window_clipped(self):
        self.assertEqual(gee.month_window(2024, 1, "2024-01-15", "2024-03-02"), ("2024-01-15", "2024-02-01"))
        self.assertEqual(gee.month_window(2024, 3, "2024-01-15", "2024-03-02"), ("2024-03-01", "2024-03-03"))
        self.assertEqual(gee.month_window(2024, 12, "2024-01-01", "2024-12-31"), ("2024-12-01", "2025-01-01"))


class TestColumnMapping(unittest.TestCase):
    def test_google_schema(self):
        from data_pipeline.open_buildings import representation, resolve_columns
        m, _ = resolve_columns(["latitude", "longitude", "area_in_meters", "confidence", "geometry", "full_plus_code"])
        self.assertEqual(m["geometry_wkt"], "geometry")
        self.assertEqual(m["area_m2"], "area_in_meters")
        self.assertEqual(representation(m), "polygons")

    def test_points_and_unknown(self):
        from data_pipeline.open_buildings import representation, resolve_columns
        m, _ = resolve_columns(["Latitude", "Longitude"])
        self.assertEqual(representation(m), "points")
        self.assertIsNone(m["confidence"])
        m, _ = resolve_columns(["ward", "building_total"])
        self.assertEqual(representation(m), "no_coordinates")

    def test_override_must_exist(self):
        from data_pipeline.open_buildings import BuildingDataError, resolve_columns
        old = dict(config.OPEN_BUILDINGS_COLUMNS)
        try:
            config.OPEN_BUILDINGS_COLUMNS["latitude"] = "y_coord"
            with self.assertRaises(BuildingDataError):
                resolve_columns(["latitude", "longitude"])
        finally:
            config.OPEN_BUILDINGS_COLUMNS.clear(); config.OPEN_BUILDINGS_COLUMNS.update(old)


class TestCorrelation(unittest.TestCase):
    def test_pairwise_n(self):
        from visualization.plots import corr_table
        df = pd.DataFrame({"a": [1, 2, 3, 4, np.nan], "b": [2, 4, 6, 8, 10], "c": [np.nan, np.nan, 1, 2, 3]})
        r, n = corr_table(df, ["a", "b", "c"])
        self.assertEqual(n.loc["a", "b"], 4)
        self.assertEqual(n.loc["a", "c"], 2)
        self.assertAlmostEqual(r.loc["a", "b"], 1.0)
        self.assertTrue(np.isnan(r.loc["a", "c"]))      # n < 3 -> not reported


class OfflineSmokeTest(unittest.TestCase):
    """Boundary -> buildings -> grid -> plots on SYNTHETIC data, in a temp folder."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="uhm_test_")
        cls.old_base = config.BASE_DIR
        config.set_base_dir(cls.tmp)
        cls.old_chunk = config.CSV_CHUNK_ROWS
        config.CSV_CHUNK_ROWS = 1000                  # force several chunks
        config.ensure_dirs()
        cls.city = syn.write_boundary(config.BOUNDARY_FILE)
        cls.buildings = syn.write_buildings_csv(config.OPEN_BUILDINGS_CSV)
        with open(config.OPEN_BUILDINGS_CSV, "rb") as fh:
            cls.csv_bytes = fh.read()

    @classmethod
    def tearDownClass(cls):
        config.set_base_dir(cls.old_base)
        config.CSV_CHUNK_ROWS = cls.old_chunk
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_1_boundary(self):
        from data_pipeline import study_area
        rec = study_area.prepare_study_area(force=True)
        self.assertTrue(rec["contains_expected_centre"])
        self.assertTrue(300 < rec["city_area_km2"] < 800)
        self.assertAlmostEqual(rec["buffer_distance_check_m"]["min"], config.BUFFER_M, delta=25)
        self.assertAlmostEqual(rec["buffer_distance_check_m"]["max"], config.BUFFER_M, delta=25)
        city, buf = study_area.load_study_area(config.PROJECTED_CRS)
        self.assertTrue(buf.contains(city.buffer(-1)))
        self.assertGreater(buf.area, city.area)

    def test_2_boundary_rejects_district_sized(self):
        from shapely.geometry import Point
        from data_pipeline import study_area
        big = study_area.reproject(study_area.reproject(Point(syn.CENTRE), "EPSG:4326", "EPSG:32643").buffer(50_000),
                                   "EPSG:32643", "EPSG:4326")      # ~7,850 km2
        with self.assertRaises(study_area.BoundaryError):
            study_area.build_study_area(big, {"kind": "municipal", "source": "test", "label": "x"})

    def test_3_buildings(self):
        import shapely
        from data_pipeline import open_buildings, study_area
        summary = open_buildings.process_buildings(force=True)
        # independent count: valid, unique, confident, intersecting the study area
        _, buf = study_area.load_study_area()
        d = self.buildings.drop_duplicates("full_plus_code").dropna(subset=["geometry"])
        d = d[d.confidence >= config.BUILDING_CONFIDENCE_MIN]
        g = shapely.make_valid(shapely.from_wkt(d.geometry.values))
        expected = int(shapely.intersects(buf, g).sum())
        got = int(summary.set_index("zone").loc["study_area", "building_count"])
        self.assertEqual(got, expected)
        s = summary.set_index("zone")
        self.assertEqual(s.loc["city", "building_count"] + s.loc["buffer_ring", "building_count"], got)
        # areas: computed in UTM, close to the CSV's metre areas for these rectangles
        clean = pd.read_csv(config.BUILDINGS_CLEAN_CSV)
        ok = clean[clean.building_id != "SYNBOWTIE"]
        self.assertLess(((ok.area_m2 - ok.area_m2_csv).abs() / ok.area_m2_csv).max(), 0.01)
        self.assertEqual(clean.building_id.duplicated().sum(), 0)
        # grid: totals add up and density is footprint / in-study cell area
        bg = pd.read_csv(config.BUILDINGS_GRID_CSV)
        self.assertEqual(int(bg.building_count.sum()), got)
        self.assertAlmostEqual(bg.footprint_area_m2.sum(), clean.area_m2.sum(), delta=1e-3 * clean.area_m2.sum())
        dens = bg.footprint_area_m2 / bg.cell_area_in_study_m2
        self.assertTrue(np.allclose(bg.building_density, dens.clip(upper=1), equal_nan=True))
        # original CSV untouched
        with open(config.OPEN_BUILDINGS_CSV, "rb") as fh:
            self.assertEqual(fh.read(), self.csv_bytes)
        self.assertTrue(os.path.exists(config.BUILDINGS_SCHEMA_REPORT))

    def test_4_grid_and_plots(self):
        from data_pipeline import spatial_grid
        from visualization import plots
        df = syn.write_monthly_csv(config.MONTHLY_CSV)
        from data_pipeline import study_area
        _, buf = study_area.load_study_area()
        syn.write_lst_tifs(config, df, buf.bounds)
        syn.write_grid_tifs(config, [2024, 2025], config.BUILDINGS_GRID_CSV)
        grid = spatial_grid.build_grid_dataset()
        self.assertEqual(sorted(grid.year.unique()), [2024, 2025])
        self.assertEqual(grid.groupby("year").grid_id.nunique().nunique(), 1)   # same cells every year
        self.assertGreater(grid.LST.notna().mean(), 0.9)
        self.assertTrue(os.path.exists(config.GRID_GEOJSON))
        # missing months stay missing
        obs = plots._observed(pd.read_csv(config.MONTHLY_CSV))
        self.assertTrue(obs.loc[(obs.Year == 2024) & (obs.Month == 8), "LST"].isna().all())   # thin month
        self.assertTrue(obs.loc[(obs.Year == 2024) & (obs.Month == 7), "LST"].isna().all())   # no data
        not_drawn = plots.make_all_plots(pd.read_csv(config.MONTHLY_CSV), grid,
                                         pd.read_csv(config.BUILDING_SUMMARY_CSV))
        self.assertEqual(not_drawn, [])
        maps = [f for _, _, fs in os.walk(os.path.join(config.PLOTS_DIR, "lst_monthly_maps")) for f in fs]
        self.assertEqual(len(maps), 24)
        for name in ("ahmedabad_lst_annual_2024.png", "ahmedabad_lst_trend_2025.png",
                     "ahmedabad_lst_cross_year_comparison.png", "ahmedabad_lst_time_series.png",
                     "ahmedabad_lst_annual_comparison.png", "correlation_monthly_means.png",
                     "correlation_grid_2025.png", "scatter_lst_drivers_2025.png",
                     "lst_and_building_density_2025.png", os.path.join("buildings", "building_density_maps.png"),
                     os.path.join("buildings", "building_footprints_map.png")):
            self.assertTrue(os.path.exists(os.path.join(config.PLOTS_DIR, name)), name)
        tr = plots.long_term_trend(df)
        self.assertFalse(tr["justified"])                      # < 36 observed months -> no trend line

    def test_5_points_only_csv(self):
        """Lat/lon without polygons: counts work, areas come from the CSV column and are labelled so."""
        from data_pipeline import open_buildings
        pts = os.path.join(config.RAW_DIR, "points_only.csv")
        self.buildings.drop(columns=["geometry"]).to_csv(pts, index=False)
        summary = open_buildings.process_buildings(pts, force=True).set_index("zone")
        self.assertEqual(summary.loc["study_area", "representation"], "points")
        self.assertEqual(summary.loc["study_area", "area_source"], "csv_column")
        self.assertGreater(summary.loc["study_area", "building_count"], 0)
        pts2 = os.path.join(config.RAW_DIR, "points_no_area.csv")
        self.buildings[["latitude", "longitude"]].to_csv(pts2, index=False)
        s2 = open_buildings.process_buildings(pts2, force=True).set_index("zone")
        self.assertTrue(np.isnan(s2.loc["study_area", "footprint_density_pct"]))   # no invented areas

    def test_6_no_coordinates_csv(self):
        from data_pipeline import open_buildings
        agg = os.path.join(config.RAW_DIR, "aggregate.csv")
        pd.DataFrame({"ward": ["A", "B"], "building_total": [100, 200]}).to_csv(agg, index=False)
        with self.assertRaises(open_buildings.BuildingDataError):
            open_buildings.process_buildings(agg, force=True)
        self.assertTrue(os.path.exists(config.BUILDINGS_SCHEMA_REPORT))


if __name__ == "__main__":
    unittest.main(verbosity=2)
