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

        with open(js_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertTrue(content.startswith("window.DASHBOARD_DATA = {"))
            self.assertTrue(content.strip().endswith("};"))

    def test_serve_dashboard_http(self):
        # Build real bundle first
        data_builder.build_dashboard_data()

        # Start non-blocking server on high port
        server, port = serve.run_dashboard(port=8912, open_browser=False, blocking=False)
        try:
            url = f"http://127.0.0.1:{port}/data.json"
            with urllib.request.urlopen(url, timeout=5) as response:
                self.assertEqual(response.status, 200)
                data = json.loads(response.read().decode("utf-8"))
                self.assertEqual(data["meta"]["total_cells"], 20)

            # Check index.html serving
            root_url = f"http://127.0.0.1:{port}/index.html"
            with urllib.request.urlopen(root_url, timeout=5) as response:
                self.assertEqual(response.status, 200)
                html_body = response.read().decode("utf-8")
                self.assertIn("Ahmedabad Urban Heat Mitigation", html_body)
        finally:
            try:
                server.shutdown()
            except Exception:
                pass
            server.server_close()


if __name__ == "__main__":
    unittest.main()
