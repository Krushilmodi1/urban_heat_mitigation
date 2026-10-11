"""Physics-Informed Neural Network (PINN) for Urban Surface Heat Dynamics.

Governing Physics:
  2D Thermal energy balance and heat diffusion-advection across urban cells:
    alpha * (d2T/dx2 + d2T/dy2) + S(NDVI, NDBI, building_density) - beta * (T - T_ambient) = 0

The PINN loss balances empirical observation fidelity with physical consistency:
  Loss = MSE(T_pred, T_obs) + lambda_physics * MSE(PDE_residual)
"""
import json
import logging
import os

import numpy as np
import tensorflow as tf

log = logging.getLogger("models.pinn")

FEATURE_NAMES = ["x_norm", "y_norm", "NDVI", "NDBI", "building_density", "mean_footprint_norm"]


class UrbanHeatPINN(tf.keras.Model):
    def __init__(self, hidden_units=(64, 64, 32), activation="tanh",
                 alpha=0.05, beta=0.1, lambda_pde=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.Variable(alpha, trainable=True, dtype=tf.float32, name="thermal_diffusivity")
        self.beta = tf.Variable(beta, trainable=True, dtype=tf.float32, name="cooling_rate")
        self.gamma_bld = tf.Variable(1.2, trainable=True, dtype=tf.float32, name="gamma_bld")
        self.gamma_ndbi = tf.Variable(1.5, trainable=True, dtype=tf.float32, name="gamma_ndbi")
        self.gamma_ndvi = tf.Variable(-2.0, trainable=True, dtype=tf.float32, name="gamma_ndvi")
        self.lambda_pde = lambda_pde

        self.hidden_layers = [
            tf.keras.layers.Dense(units, activation=activation, kernel_initializer="glorot_normal")
            for units in hidden_units
        ]
        self.out_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out_layer(x)

    def compute_pde_residual(self, x_coord, y_coord, ndvi, ndbi, bld_dens, mean_footprint):
        """Compute the PDE residual using automatic differentiation."""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x_coord, y_coord])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x_coord, y_coord])
                features = tf.concat([x_coord, y_coord, ndvi, ndbi, bld_dens, mean_footprint], axis=1)
                t_pred = self(features, training=True)

            t_x = tape1.gradient(t_pred, x_coord)
            t_y = tape1.gradient(t_pred, y_coord)
            del tape1

        t_xx = tape2.gradient(t_x, x_coord)
        t_yy = tape2.gradient(t_y, y_coord)
        del tape2

        # Safe fallback if second derivatives are zero or None
        t_xx = tf.zeros_like(x_coord) if t_xx is None else t_xx
        t_yy = tf.zeros_like(y_coord) if t_yy is None else t_yy
        laplacian = t_xx + t_yy

        # Source / sink term from land surface characteristics
        source = (self.gamma_bld * bld_dens +
                  self.gamma_ndbi * ndbi +
                  self.gamma_ndvi * ndvi)

        # Quasi-steady thermal energy balance: alpha * laplacian + source - beta * T = 0
        residual = self.alpha * laplacian + source - self.beta * t_pred
        return residual, t_pred

    def train_step_pinn(self, x_data, y_target, optimizer):
        x_c = tf.expand_dims(x_data[:, 0], 1)
        y_c = tf.expand_dims(x_data[:, 1], 1)
        ndvi = tf.expand_dims(x_data[:, 2], 1)
        ndbi = tf.expand_dims(x_data[:, 3], 1)
        bld_dens = tf.expand_dims(x_data[:, 4], 1)
        mean_fp = tf.expand_dims(x_data[:, 5], 1)

        with tf.GradientTape() as tape:
            res, t_pred = self.compute_pde_residual(x_c, y_c, ndvi, ndbi, bld_dens, mean_fp)
            loss_data = tf.reduce_mean(tf.square(t_pred - y_target))
            loss_pde = tf.reduce_mean(tf.square(res))
            loss_total = loss_data + self.lambda_pde * loss_pde

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss_total, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))

        return {
            "loss_total": float(loss_total.numpy()),
            "loss_data": float(loss_data.numpy()),
            "loss_pde": float(loss_pde.numpy()),
        }


class PINNPipeline:
    """Wrapper handling data preprocessing, training, evaluation and persistence."""

    def __init__(self, lambda_pde=0.1, lr=1e-3):
        self.lambda_pde = lambda_pde
        self.lr = lr
        self.model = UrbanHeatPINN(lambda_pde=lambda_pde)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.scaler_stats = {}

    def fit_scaler(self, df):
        """Compute normalization parameters from training dataframe."""
        stats = {}
        for col in ["x_utm", "y_utm", "NDVI", "NDBI", "building_density", "mean_footprint_m2", "LST"]:
            valid = df[col].dropna()
            stats[col] = {
                "mean": float(valid.mean()) if len(valid) else 0.0,
                "std": float(valid.std()) if len(valid) and valid.std() > 1e-6 else 1.0,
                "min": float(valid.min()) if len(valid) else 0.0,
                "max": float(valid.max()) if len(valid) else 1.0,
            }
        self.scaler_stats = stats
        return stats

    def transform(self, df):
        """Prepare normalized feature tensor and target tensor."""
        x_norm = ((df["x_utm"] - self.scaler_stats["x_utm"]["min"]) /
                  (self.scaler_stats["x_utm"]["max"] - self.scaler_stats["x_utm"]["min"] + 1e-6)).values
        y_norm = ((df["y_utm"] - self.scaler_stats["y_utm"]["min"]) /
                  (self.scaler_stats["y_utm"]["max"] - self.scaler_stats["y_utm"]["min"] + 1e-6)).values

        ndvi = df["NDVI"].fillna(self.scaler_stats["NDVI"]["mean"]).values
        ndbi = df["NDBI"].fillna(self.scaler_stats["NDBI"]["mean"]).values
        bld_dens = df["building_density"].fillna(0.0).values
        fp_norm = (df["mean_footprint_m2"].fillna(0.0) /
                   (self.scaler_stats["mean_footprint_m2"]["max"] + 1e-6)).values

        X = np.stack([x_norm, y_norm, ndvi, ndbi, bld_dens, fp_norm], axis=1).astype(np.float32)

        if "LST" in df.columns:
            y = ((df["LST"] - self.scaler_stats["LST"]["mean"]) /
                 self.scaler_stats["LST"]["std"]).values.astype(np.float32)
            y = np.expand_dims(y, 1)
            return X, y
        return X, None

    def inverse_transform_target(self, y_pred_norm):
        """Convert normalized temperature back to degrees Celsius."""
        mean = self.scaler_stats["LST"]["mean"]
        std = self.scaler_stats["LST"]["std"]
        return y_pred_norm * std + mean

    def train(self, df_train, epochs=100, batch_size=128):
        self.fit_scaler(df_train)
        valid_mask = df_train["LST"].notna() & df_train["NDVI"].notna()
        df_valid = df_train[valid_mask].copy()

        if len(df_valid) == 0:
            raise ValueError("No valid rows with both LST and NDVI found for training.")

        X, y = self.transform(df_valid)
        n_samples = len(X)
        history = {"loss_total": [], "loss_data": [], "loss_pde": []}

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]

            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                xb = tf.convert_to_tensor(X_shuffled[i:i + batch_size], dtype=tf.float32)
                yb = tf.convert_to_tensor(y_shuffled[i:i + batch_size], dtype=tf.float32)
                losses = self.model.train_step_pinn(xb, yb, self.optimizer)
                epoch_losses.append(losses)

            mean_total = np.mean([l["loss_total"] for l in epoch_losses])
            mean_data = np.mean([l["loss_data"] for l in epoch_losses])
            mean_pde = np.mean([l["loss_pde"] for l in epoch_losses])

            history["loss_total"].append(float(mean_total))
            history["loss_data"].append(float(mean_data))
            history["loss_pde"].append(float(mean_pde))

            if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
                log.info("Epoch %3d/%3d | Total: %.4f | Data MSE: %.4f | PDE Residual: %.4f",
                         epoch, epochs, mean_total, mean_data, mean_pde)

        return history

    def predict(self, df):
        X, _ = self.transform(df)
        pred_norm = self.model(tf.convert_to_tensor(X, dtype=tf.float32), training=False).numpy()
        return self.inverse_transform_target(pred_norm).ravel()

    def save(self, model_path, stats_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_weights(model_path + ".weights.h5")
        with open(stats_path, "w") as fh:
            json.dump({"scaler_stats": self.scaler_stats, "lambda_pde": self.lambda_pde}, fh, indent=2)
        log.info("Saved PINN model weights and scalers to %s", model_path)

    def load(self, model_path, stats_path):
        with open(stats_path) as fh:
            meta = json.load(fh)
        self.scaler_stats = meta["scaler_stats"]
        self.lambda_pde = meta.get("lambda_pde", 0.1)
        dummy_x = tf.zeros((1, 6), dtype=tf.float32)
        _ = self.model(dummy_x)
        self.model.load_weights(model_path + ".weights.h5")
        log.info("Loaded PINN model from %s", model_path)
