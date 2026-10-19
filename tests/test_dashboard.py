"""Unit tests for the interactive dashboard data bundle and HTTP server.
"""
import json
import os
import shutil
import tempfile
import unittest
import urllib.request

import numpy as np
import pandas as pd

import config
from dashboard import data_builder, serve


class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.orig_base = config.BASE_DIR
        config.set_base_dir(self.tmp)
        config.ensure_dirs()

        # Create dummy scenario results CSV and model benchmarks CSV
        n = 20
        scen_rows = []
        for i in range(n):
            scen_rows.append({
                "grid_id": f"G_{i}",
                "zone": "city" if i < 15 else "buffer_ring",
                "lon": 72.5 + i * 0.01,
                "lat": 23.0 + i * 0.01,
                "x_utm": 250000 + i * 500,
                "y_utm": 2540000 + i * 500,
                "LST_baseline": 32.5 + i * 0.1,
                "building_density": 0.15,
                "building_count": 50,
                "NDVI": 0.22,
                "NDBI": 0.05,
                "UHVI": 0.45 + i * 0.01,
                "priority_tier": "High" if i < 5 else "Moderate",
                "policy_recommendation": "Cool Roofs + Trees",
                "delta_LST_cool_roofs": 0.12,
                "delta_LST_green_roofs": 0.08,
                "delta_LST_urban_trees": 0.35,
                "delta_LST_combined": 0.55,
                "delta_T_air_cool_roofs": 0.04,
                "delta_T_air_green_roofs": 0.03,
                "delta_T_air_urban_trees": 0.11,
                "delta_T_air_combined": 0.18,
                "delta_LST_combined_lower95": 0.0,
                "delta_LST_combined_upper95": 1.33,
            })
        df_scen = pd.DataFrame(scen_rows)
        df_scen.to_csv(config.SCENARIO_RESULTS_CSV, index=False)

        # Benchmarks CSV
        benchmarks = [
            {"Model": "Baseline 0: Mean Predictor", "Test RMSE (C)": 0.4074, "Test MAE (C)": 0.3271, "Test R2": 0.0000},
            {"Model": "Baseline 1: Ridge Regression", "Test RMSE (C)": 0.3959, "Test MAE (C)": 0.3160, "Test R2": 0.0555},
            {"Model": "PINN (Physical Constraint lambda=0.1)", "Test RMSE (C)": 0.4045, "Test MAE (C)": 0.3234, "Test R2": 0.0144},
        ]
        pd.DataFrame(benchmarks).to_csv(os.path.join(config.MODELS_DIR, "model_benchmarks.csv"), index=False)

        # Building Summary CSV
        bld_summary = [
            {"zone": "study_area", "building_count": 4777, "zone_area_km2": 1570.08, "footprint_density_pct": 0.08},
            {"zone": "city", "building_count": 3380, "zone_area_km2": 462.01, "footprint_density_pct": 0.15},
            {"zone": "buffer_ring", "building_count": 1397, "zone_area_km2": 1108.07, "footprint_density_pct": 0.05},
        ]
        pd.DataFrame(bld_summary).to_csv(config.BUILDING_SUMMARY_CSV, index=False)

        # Spatial Grid Multi-Year Metrics CSV
        grid_rows = []
        for yr in [2024, 2025]:
            for i in range(n):
                grid_rows.append({
                    "grid_id": f"G_{i}",
                    "year": yr,
                    "LST": 31.0 + i * 0.1 if yr == 2024 else 32.5 + i * 0.1,
                    "NDVI": 0.20 if yr == 2024 else 0.22,
                    "NDBI": 0.06 if yr == 2024 else 0.05,
                })
        pd.DataFrame(grid_rows).to_csv(os.path.join(config.PROCESSED_DIR, "spatial_grid_metrics.csv"), index=False)

        # Annual LST Comparison CSV
        ann_rows = [
            {"Year": 2024, "observed_months": 10, "mean_LST_all_observed": 31.42, "mean_LST_common_months": 31.42, "hottest_month": "May", "hottest_LST": 40.1, "coolest_month": "January", "coolest_LST": 22.3},
            {"Year": 2025, "observed_months": 11, "mean_LST_all_observed": 32.18, "mean_LST_common_months": 32.18, "hottest_month": "May", "hottest_LST": 41.2, "coolest_month": "January", "coolest_LST": 23.1},
        ]
        pd.DataFrame(ann_rows).to_csv(os.path.join(config.PROCESSED_DIR, "annual_lst_comparison.csv"), index=False)

        # Copy real index.html into test tempdir so server can serve it
        orig_html = os.path.join(self.orig_base, "dashboard", "index.html")
        if os.path.exists(orig_html):
            target_dash = os.path.join(self.tmp, "dashboard")
            os.makedirs(target_dash, exist_ok=True)
            shutil.copy2(orig_html, os.path.join(target_dash, "index.html"))

    def tearDown(self):
        config.set_base_dir(self.orig_base)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_dashboard_data(self):
        bundle = data_builder.build_dashboard_data()
        self.assertIn("meta", bundle)
        self.assertIn("benchmarks", bundle)
        self.assertIn("scenarios_summary", bundle)
        self.assertIn("grid_cells", bundle)
        self.assertIn("top_hotspots", bundle)

        self.assertEqual(len(bundle["grid_cells"]), 20)
        self.assertLessEqual(len(bundle["top_hotspots"]), 50)

        # Check export files exist
        dashboard_dir = os.path.join(config.BASE_DIR, "dashboard")
        json_path = os.path.join(dashboard_dir, "data.json")
        js_path = os.path.join(dashboard_dir, "data.js")

        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(js_path))

        with open(json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
            self.assertEqual(loaded_json["meta"]["total_cells"], 20)
            self.assertEqual(loaded_json["meta"]["available_years"], [2024, 2025])
            self.assertIn("2024", loaded_json["meta"]["yearly_metrics"])
            self.assertIn("2025", loaded_json["meta"]["yearly_metrics"])

        with open(js_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertTrue(content.startswith("window.DASHBOARD_DATA = {"))
            self.assertTrue(content.strip().endswith("};"))

    def test_multi_year_cell_integrity(self):
        bundle = data_builder.build_dashboard_data()
        first_cell = bundle["grid_cells"][0]

        # Verify years dictionary is present
        self.assertIn("years", first_cell)
        self.assertIn("2024", first_cell["years"])
        self.assertIn("2025", first_cell["years"])

        # 2024 has observed satellite metrics but no modeled scenarios (scientific integrity)
        y2024 = first_cell["years"]["2024"]
        self.assertEqual(y2024["year"], 2024)
        self.assertEqual(y2024["period"], "day")
        self.assertEqual(y2024["lst"], 31.0)
        self.assertEqual(y2024["ndvi"], 0.20)
        self.assertEqual(y2024["ndbi"], 0.06)
        self.assertIsNone(y2024["uhvi"])
        self.assertIsNone(y2024["dLST_comb"])

        # 2025 has both observed satellite metrics and modeled mitigation scenarios
        y2025 = first_cell["years"]["2025"]
        self.assertEqual(y2025["year"], 2025)
        self.assertEqual(y2025["period"], "day")
        self.assertEqual(y2025["lst"], 32.5)
        self.assertEqual(y2025["ndvi"], 0.22)
        self.assertEqual(y2025["ndbi"], 0.05)
        self.assertIsNotNone(y2025["uhvi"])
        self.assertIsNotNone(y2025["dLST_comb"])

    def test_day_night_metadata_and_provenance(self):
        bundle = data_builder.build_dashboard_data()
        day_night = bundle["meta"]["day_night"]

        self.assertFalse(day_night["has_separate_night_data"])
        self.assertEqual(day_night["available_periods"], ["day"])
        self.assertNotIn("night", day_night["available_periods"])
        self.assertIn("Landsat", day_night["sensor"])
        self.assertIn("10:30 AM", day_night["orbit_overpass"])
        self.assertIn("MODIS", day_night["explanation"])

    def test_presentation_deck_completely_removed(self):
        html_path = os.path.join(self.orig_base, "dashboard", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Check navigation tab button is removed
        self.assertNotIn("tab-presentation", html)
        self.assertNotIn("switchTab('presentation')", html)
        self.assertNotIn("Presentation Deck", html)

        # Check presentation section is removed
        self.assertNotIn('<section id="tab-presentation"', html)

        # Check slide navigation functions are removed
        self.assertNotIn("changeSlide(", html)
        self.assertNotIn("currentSlide", html)
        self.assertNotIn(".presentation-deck", html)

        # Check remaining functional tabs are preserved
        self.assertIn('id="tab-overview"', html)
        self.assertIn('id="tab-explorer"', html)
        self.assertIn('id="tab-simulator"', html)
        self.assertIn('id="tab-benchmarks"', html)
        self.assertIn('id="tab-research"', html)
        self.assertIn('id="tab-downloads"', html)
        self.assertIn("switchTab('overview')", html)
        self.assertIn("switchTab('explorer')", html)
        self.assertIn("switchTab('simulator')", html)
        self.assertIn("switchTab('benchmarks')", html)
        self.assertIn("switchTab('research')", html)
        self.assertIn("switchTab('downloads')", html)

    def test_ui_controls_and_inspector_elements(self):
        html_path = os.path.join(self.orig_base, "dashboard", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Year filter selector and options
        self.assertIn('id="sel-year-filter"', html)
        self.assertIn('value="all"', html)
        self.assertIn('value="2025"', html)
        self.assertIn('value="2024"', html)

        # Day/Night selector and modal
        self.assertIn('id="btn-period-day"', html)
        self.assertIn('id="btn-period-night"', html)
        self.assertIn('id="modal-night-info"', html)
        self.assertIn('showNightUnavailableModal()', html)

        # Cell Inspector fields
        self.assertIn('id="insp-year-period"', html)
        self.assertIn('id="insp-coords"', html)
        self.assertIn('id="insp-ndbi"', html)
        self.assertIn('id="insp-uhvi"', html)
        self.assertIn('id="insp-scenarios-section"', html)

    def test_fallback_when_multiyear_grid_csv_missing(self):
        # Create a separate temporary folder without spatial_grid_metrics.csv
        fresh_tmp = tempfile.mkdtemp()
        orig = config.BASE_DIR
        try:
            # Source paths from self.tmp
            src_scen = os.path.join(self.tmp, "data", "processed", "simulation", "cooling_scenarios.csv")
            src_bld = os.path.join(self.tmp, "data", "processed", "open_buildings", "building_summary.csv")

            config.set_base_dir(fresh_tmp)
            config.ensure_dirs()

            shutil.copy2(src_scen, config.SCENARIO_RESULTS_CSV)
            if os.path.exists(src_bld):
                shutil.copy2(src_bld, config.BUILDING_SUMMARY_CSV)

            # Should not raise FileNotFoundError, should fallback gracefully
            bundle = data_builder.build_dashboard_data()
            self.assertIn("grid_cells", bundle)
            self.assertGreater(len(bundle["grid_cells"]), 0)
            self.assertIn("2025", bundle["grid_cells"][0]["years"])
        finally:
            config.set_base_dir(orig)
            shutil.rmtree(fresh_tmp, ignore_errors=True)

    def test_serve_dashboard_http(self):
        # Build real bundle first
        data_builder.build_dashboard_data()

        # Start non-blocking server on high port
        server, port = serve.run_dashboard(port=8912, open_browser=False, blocking=False)
        try:
            url = f"http://127.0.0.1:{port}/data.json"
            with urllib.request.urlopen(url, timeout=5) as response:
                self.assertEqual(response.status, 200)
                # Verify anti-caching headers
                self.assertIn("no-cache", response.headers.get("Cache-Control", ""))
                self.assertEqual(response.headers.get("Pragma"), "no-cache")
                data = json.loads(response.read().decode("utf-8"))
                self.assertEqual(data["meta"]["total_cells"], 20)

            # Check index.html serving with anti-cache headers
            root_url = f"http://127.0.0.1:{port}/index.html"
            with urllib.request.urlopen(root_url, timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertIn("no-cache", response.headers.get("Cache-Control", ""))
                html_body = response.read().decode("utf-8")
                self.assertIn("Ahmedabad Urban Heat Mitigation", html_body)
                self.assertNotIn("Presentation Deck", html_body)
                self.assertIn("sel-year-filter", html_body)
        finally:
            try:
                server.shutdown()
            except Exception:
                pass
            server.server_close()

    def test_csv_exports_and_year_filter_in_html(self):
        html_path = os.path.join(self.orig_base, "dashboard", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Check year-aware export handler exists
        self.assertIn("downloadHotspotsCSV()", html)
        self.assertIn("downloadAllScenariosCSV()", html)
        self.assertIn("downloadBenchmarksCSV()", html)
        self.assertIn("ahmedabad_priority_hotspots_top50_", html)

        # Check map search centering
        self.assertIn("searchGridCell", html)
        self.assertIn("resetMapExtent", html)

        # Check scientific refinement
        self.assertNotIn("overfits local coordinates", html)
        self.assertIn("exhibits large spatial holdout generalization gap", html)
        self.assertIn("Thermal Advection-Diffusion PDE Regularizer", html)


if __name__ == "__main__":
    unittest.main()

