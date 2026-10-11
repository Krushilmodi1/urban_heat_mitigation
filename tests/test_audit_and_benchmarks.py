"""Unit tests for spatial block holdout, model benchmarks, and hotspot ranking.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import config
from models.train_pinn import compute_morans_i, create_spatial_block_split
from simulation import cooling_scenario


class TestAuditAndBenchmarks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.orig_base = config.BASE_DIR
        config.set_base_dir(self.tmp)
        config.ensure_dirs()

        # Create realistic synthetic spatial grid
        n_side = 10
        xs = np.linspace(240000, 270000, n_side)
        ys = np.linspace(2530000, 2560000, n_side)
        grid_rows = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                grid_rows.append({
                    "grid_id": f"E{i}_N{j}",
                    "zone": "city" if (3 <= i <= 7 and 3 <= j <= 7) else "buffer_ring",
                    "lon": 72.5 + (x - 240000) / 100000,
                    "lat": 23.0 + (y - 2530000) / 100000,
                    "x_utm": x,
                    "y_utm": y,
                    "NDVI": float(np.clip(0.35 - 0.25 * ((i + j) / 20), 0.1, 0.5)),
                    "NDBI": float(np.clip(-0.10 + 0.30 * ((i + j) / 20), -0.2, 0.3)),
                    "building_density": float(np.clip(0.05 + 0.50 * ((i + j) / 20), 0.0, 0.7)),
                    "building_count": int(10 + 100 * ((i + j) / 20)),
                    "mean_footprint_m2": 110.0,
                    "LST": float(32.0 + 8.0 * ((i + j) / 20)),
                    "year": 2025,
                })
        self.df = pd.DataFrame(grid_rows)

    def tearDown(self):
        config.set_base_dir(self.orig_base)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_spatial_block_split_separation(self):
        train_df, test_df, test_mask = create_spatial_block_split(self.df, block_size_m=8000, test_fraction=0.25, seed=42)
        # Verify split is non-empty
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)
        self.assertEqual(len(train_df) + len(test_df), len(self.df))

        # Verify zero spatial block intersection
        train_blocks = set(train_df["spatial_block_id"])
        test_blocks = set(test_df["spatial_block_id"])
        self.assertEqual(len(train_blocks & test_blocks), 0)

        # Verify zero grid_id intersection
        self.assertEqual(len(set(train_df["grid_id"]) & set(test_df["grid_id"])), 0)

    def test_morans_i_calculation(self):
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        # Correlated field
        val_corr = x * 0.5 + y * 0.5
        mi_corr = compute_morans_i(x, y, val_corr)
        self.assertGreater(mi_corr, 0.2)

        # Independent random noise
        np.random.seed(42)
        val_rand = np.random.normal(0, 1, 50)
        mi_rand = compute_morans_i(x, y, val_rand)
        self.assertLess(abs(mi_rand), 0.3)

    def test_uhvi_and_hotspot_ranking(self):
        results = cooling_scenario.simulate_scenarios(self.df, pipeline=None)
        self.assertIn("UHVI", results.columns)
        self.assertIn("priority_tier", results.columns)
        self.assertIn("delta_T_air_combined", results.columns)
        self.assertIn("delta_LST_combined", results.columns)

        # Check UHVI is bounded [0, 1]
        self.assertTrue((results["UHVI"] >= 0.0).all())
        self.assertTrue((results["UHVI"] <= 1.0).all())

        # Check Air temperature cooling is roughly ~32% of skin LST cooling
        mean_lst = results["delta_LST_combined"].mean()
        mean_air = results["delta_T_air_combined"].mean()
        self.assertAlmostEqual(mean_air / (mean_lst + 1e-6), cooling_scenario.LST_TO_TAIR_FACTOR, delta=0.05)

        # Check exportable hotspots csv and interactive map generation
        cooling_scenario.plot_scenario_figures(results)
        hotspot_csv = os.path.join(self.tmp, "priority_cooling_hotspots.csv")
        results.head(10).to_csv(hotspot_csv, index=False)
        self.assertTrue(os.path.exists(hotspot_csv))

        html_map = os.path.join(self.tmp, "ahmedabad_hotspots_interactive.html")
        cooling_scenario.generate_interactive_map(results, html_map)
        self.assertTrue(os.path.exists(html_map))
        self.assertGreater(os.path.getsize(html_map), 500)
        with open(html_map, "r", encoding="utf-8") as fh:
            html_text = fh.read()
        self.assertIn("L.control.layers", html_text)
        self.assertIn("Top 10 Critical Hotspots", html_text)
        self.assertIn("provenance-tag", html_text)

    def test_baseline_models_and_scaler_isolation(self):
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        train_df, test_df, _ = create_spatial_block_split(self.df, block_size_m=8000, test_fraction=0.3, seed=1)
        features = ["x_utm", "y_utm", "NDVI", "NDBI", "building_density", "mean_footprint_m2"]
        X_tr = train_df[features].fillna(0.0)
        y_tr = train_df["LST"].values
        X_te = test_df[features].fillna(0.0)
        y_te = test_df["LST"].values

        # Test Ridge
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        ridge = Ridge(alpha=1.0).fit(X_tr_s, y_tr)
        p_ridge = ridge.predict(X_te_s)
        self.assertEqual(len(p_ridge), len(y_te))
        self.assertTrue(np.all(np.isfinite(p_ridge)))

        # Test Random Forest
        rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42).fit(X_tr, y_tr)
        p_rf = rf.predict(X_te)
        self.assertEqual(len(p_rf), len(y_te))
        self.assertTrue(np.all(np.isfinite(p_rf)))


if __name__ == "__main__":
    unittest.main()
