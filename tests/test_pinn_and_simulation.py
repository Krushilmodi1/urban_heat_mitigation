"""Unit tests for the Physics-Informed Neural Network (PINN) and Cooling Simulation.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

import config
from models.pinn import PINNPipeline, UrbanHeatPINN
from simulation import cooling_scenario


class TestPINN(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.orig_base = config.BASE_DIR
        config.set_base_dir(self.tmp)
        config.ensure_dirs()

        # Create dummy grid dataframe
        n = 50
        self.df = pd.DataFrame({
            "grid_id": [f"E{i}_N{i}" for i in range(n)],
            "zone": ["city"] * (n // 2) + ["buffer_ring"] * (n - n // 2),
            "x_utm": np.linspace(250000, 270000, n),
            "y_utm": np.linspace(2540000, 2560000, n),
            "NDVI": np.random.uniform(0.05, 0.45, n),
            "NDBI": np.random.uniform(-0.1, 0.3, n),
            "building_density": np.random.uniform(0.0, 0.6, n),
            "mean_footprint_m2": np.random.uniform(20.0, 300.0, n),
            "LST": np.random.uniform(32.0, 46.0, n),
            "year": [2025] * n,
        })

    def tearDown(self):
        config.set_base_dir(self.orig_base)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pinn_model_forward_and_residual(self):
        model = UrbanHeatPINN(hidden_units=(16, 16))
        x_c = tf.constant([[0.5]], dtype=tf.float32)
        y_c = tf.constant([[0.5]], dtype=tf.float32)
        ndvi = tf.constant([[0.2]], dtype=tf.float32)
        ndbi = tf.constant([[0.1]], dtype=tf.float32)
        bld = tf.constant([[0.3]], dtype=tf.float32)
        fp = tf.constant([[0.4]], dtype=tf.float32)

        res, t_pred = model.compute_pde_residual(x_c, y_c, ndvi, ndbi, bld, fp)
        self.assertEqual(t_pred.shape, (1, 1))
        self.assertEqual(res.shape, (1, 1))
        self.assertFalse(np.isnan(res.numpy()[0, 0]))

    def test_pinn_pipeline_train_and_predict(self):
        pipeline = PINNPipeline(lambda_pde=0.05, lr=0.01)
        history = pipeline.train(self.df, epochs=5, batch_size=25)
        self.assertEqual(len(history["loss_total"]), 5)
        preds = pipeline.predict(self.df)
        self.assertEqual(len(preds), len(self.df))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_cooling_simulation_scenarios(self):
        results = cooling_scenario.simulate_scenarios(self.df, pipeline=None)
        self.assertIn("delta_T_cool_roofs", results.columns)
        self.assertIn("delta_T_green_roofs", results.columns)
        self.assertIn("delta_T_urban_trees", results.columns)
        self.assertIn("delta_T_combined", results.columns)

        # Cooling should be non-negative
        self.assertTrue((results["delta_T_cool_roofs"] >= 0).all())
        self.assertTrue((results["delta_T_combined"] >= 0).all())

        # Test report and plot generation
        cooling_scenario.plot_scenario_figures(results)
        cooling_scenario.write_policy_report(results)
        self.assertTrue(os.path.exists(config.SCENARIO_REPORT_MD))


if __name__ == "__main__":
    unittest.main()
