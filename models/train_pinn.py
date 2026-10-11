"""Train and Evaluate Physics-Informed Neural Network (PINN) and ML Baselines.

Implements rigorous spatial methodology:
  1. Spatially Contiguous Block Holdout Split (prevents spatial autocorrelation leakage).
  2. Multi-Model Benchmark Suite:
     - Ridge Regression (linear spatial baseline)
     - Random Forest Regressor (non-linear ensemble baseline)
     - Standard Deep MLP (neural ablation without PDE constraint, lambda_pde = 0.0)
     - Physics-Informed Neural Network (PINN, lambda_pde = 0.1)
  3. Comprehensive Diagnostic Reporting:
     - Out-of-sample R², RMSE, MAE
     - Residual diagnostics (heteroscedasticity, error distribution, Moran's I spatial autocorrelation)
     - 95% empirical prediction uncertainty intervals
  4. Publication-grade diagnostic and comparison plots.
"""
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.pinn import PINNPipeline

log = logging.getLogger("models.train_pinn")


def ensure_training_data():
    """Ensure spatial_grid_metrics.csv exists, bootstrapping synthetic data if needed for demo."""
    if os.path.exists(config.GRID_CSV):
        df = pd.read_csv(config.GRID_CSV)
        if df["LST"].notna().sum() > 20:
            return df

    log.warning("Valid LST grid data not found at %s. Bootstrapping synthetic dataset for training demo.",
                config.GRID_CSV)
    from tests import synthetic_data as syn
    from data_pipeline import open_buildings, spatial_grid, study_area
    config.ensure_dirs()
    if not os.path.exists(config.BOUNDARY_FILE):
        syn.write_boundary(config.BOUNDARY_FILE)
    study_area.prepare_study_area(force=False)
    if not os.path.exists(config.OPEN_BUILDINGS_CSV):
        syn.write_buildings_csv(config.OPEN_BUILDINGS_CSV, n=5000)
    open_buildings.process_buildings(force=False)
    mdf = syn.write_monthly_csv(config.MONTHLY_CSV)
    _, buf = study_area.load_study_area()
    syn.write_lst_tifs(config, mdf, buf.bounds)
    syn.write_grid_tifs(config, [2024, 2025], config.BUILDINGS_GRID_CSV)
    return spatial_grid.build_grid_dataset()


def create_spatial_block_split(df, block_size_m=8000, test_fraction=0.25, seed=42):
    """Partition cells into contiguous geographic blocks to prevent spatial autocorrelation leakage.
    
    Both years (2024, 2025) for any grid cell are strictly assigned to the same split.
    """
    df_copy = df.copy()
    bx = np.floor(df_copy["x_utm"] / block_size_m).astype(int)
    by = np.floor(df_copy["y_utm"] / block_size_m).astype(int)
    df_copy["spatial_block_id"] = bx.astype(str) + "_" + by.astype(str)

    unique_blocks = sorted(df_copy["spatial_block_id"].unique())
    rng = np.random.default_rng(seed)
    n_test_blocks = max(1, int(round(len(unique_blocks) * test_fraction)))
    test_blocks = set(rng.choice(unique_blocks, size=n_test_blocks, replace=False))

    test_mask = df_copy["spatial_block_id"].isin(test_blocks)
    train_mask = ~test_mask

    train_df = df_copy[train_mask].copy()
    test_df = df_copy[test_mask].copy()

    log.info("Spatial Block Split (Block size: %.1f km): %d total blocks -> %d train blocks (%d rows), %d test blocks (%d rows)",
             block_size_m / 1000, len(unique_blocks), len(unique_blocks) - n_test_blocks, len(train_df),
             n_test_blocks, len(test_df))
    return train_df, test_df, test_mask


def compute_morans_i(x_coords, y_coords, residuals, k=8):
    """Compute Moran's I spatial autocorrelation coefficient for model residuals."""
    n = len(residuals)
    if n < k + 1:
        return 0.0
    z = residuals - np.mean(residuals)
    s0 = np.sum(z ** 2)
    if s0 == 0:
        return 0.0

    coords = np.column_stack([x_coords, y_coords])
    tree = KDTree(coords)
    dists, idxs = tree.query(coords, k=k + 1)

    num = 0.0
    w_sum = 0.0
    for i in range(n):
        for j_idx, d in zip(idxs[i][1:], dists[i][1:]):
            if d > 0:
                w = 1.0 / d
                num += w * z[i] * z[j_idx]
                w_sum += w
    return float((n / (w_sum + 1e-12)) * (num / s0))


def plot_spatial_split_map(df, test_mask):
    """Plot the spatial distribution of the train and held-out test blocks across Ahmedabad."""
    fig, ax = plt.subplots(figsize=(8, 7))
    train_df = df[~test_mask]
    test_df = df[test_mask]

    ax.scatter(train_df["x_utm"] / 1000, train_df["y_utm"] / 1000,
               c="#1b6ca8", s=18, alpha=0.7, label=f"Train Blocks (n={len(train_df)})")
    ax.scatter(test_df["x_utm"] / 1000, test_df["y_utm"] / 1000,
               c="#d95f02", s=22, alpha=0.85, edgecolors="black", lw=0.5,
               label=f"Spatial Test Holdout (n={len(test_df)})")

    ax.set_xlabel("UTM 43N Easting (km)", fontsize=11)
    ax.set_ylabel("UTM 43N Northing (km)", fontsize=11)
    ax.set_title("Ahmedabad Spatial Holdout Split (Contiguous Block Validation)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "pinn_spatial_holdout_map.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_baseline_comparison(results_df):
    """Generate bar chart comparing all models on the held-out spatial test set."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    models = results_df["Model"].tolist()
    r2_scores = results_df["Test_R2"].tolist()
    rmse_scores = results_df["Test_RMSE"].tolist()
    palette = ["#7f7f7f", "#7b4ea3", "#2a9d5c", "#b8860b", "#1b6ca8"]
    colors = palette[:len(models)]

    # 1. R² Bar
    bars1 = ax1.bar(models, r2_scores, color=colors, width=0.55)
    ax1.set_ylabel("Out-of-Sample Test R²", fontsize=11)
    ax1.set_title("Test Generalization Accuracy (R²)", fontsize=12, fontweight="bold")
    ax1.set_ylim([0, max(0.10, max(r2_scores) * 1.35)])
    ax1.grid(True, axis="y", ls="--", alpha=0.5)
    for b in bars1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, max(0.002, yval + 0.003), f"{yval:.3f}", ha="center", fontsize=9, fontweight="bold")

    # 2. RMSE Bar
    bars2 = ax2.bar(models, rmse_scores, color=colors, width=0.55)
    ax2.set_ylabel("Out-of-Sample Test RMSE (°C)", fontsize=11)
    ax2.set_title("Test Prediction Error (RMSE in °C)", fontsize=12, fontweight="bold")
    ax2.set_ylim([0, max(0.50, max(rmse_scores) * 1.15)])
    ax2.grid(True, axis="y", ls="--", alpha=0.5)
    for b in bars2:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, yval + 0.015, f"{yval:.3f}°C", ha="center", fontsize=8.5, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "pinn_baseline_comparison.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_residual_diagnostics(y_test, y_pred, x_utm_test, y_utm_test):
    """Generate 4-panel residual diagnostics plot."""
    residuals = y_test - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # 1. Histogram / Normal Distribution
    ax = axes[0, 0]
    ax.hist(residuals, bins=25, density=True, color="#1b6ca8", alpha=0.7, edgecolor="black")
    mu, sigma = float(np.mean(residuals)), float(np.std(residuals))
    x_grid = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 100)
    ax.plot(x_grid, sp_stats.norm.pdf(x_grid, mu, sigma), "r--", lw=2,
            label=f"Normal Fit (μ={mu:.2f}, σ={sigma:.2f})")
    ax.set_xlabel("Residual: Observed - Predicted (°C)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Test Residual Error Distribution", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)

    # 2. Residuals vs Predicted (Heteroscedasticity check)
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.5, color="#2a9d5c", edgecolors="none", s=25)
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Predicted LST (°C)", fontsize=10)
    ax.set_ylabel("Residual (°C)", fontsize=10)
    ax.set_title("Residuals vs. Predicted LST", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)

    # 3. Observed vs Predicted with 95% Confidence Band
    ax = axes[1, 0]
    ax.scatter(y_test, y_pred, alpha=0.4, color="#1b6ca8", edgecolors="none", s=25)
    lims = [min(min(y_test), min(y_pred)) - 1, max(max(y_test), max(y_pred)) + 1]
    ax.plot(lims, lims, "r--", lw=1.5, label="1:1 Perfect Fit")
    # Empirical 95% error interval (+/- 1.96 std)
    ax.fill_between(lims, np.array(lims) - 1.96 * sigma, np.array(lims) + 1.96 * sigma,
                    color="#1b6ca8", alpha=0.15, label="±1.96σ (95% Error Interval)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed LST (°C)", fontsize=10)
    ax.set_ylabel("Predicted LST (°C)", fontsize=10)
    ax.set_title("Test Scatter with 95% Prediction Bounds", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)

    # 4. Spatial Map of Residuals
    ax = axes[1, 1]
    sc = ax.scatter(x_utm_test / 1000, y_utm_test / 1000, c=residuals,
                    cmap="coolwarm", vmin=-2.0, vmax=2.0, s=28, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Residual (°C)")
    ax.set_xlabel("UTM 43N Easting (km)", fontsize=10)
    ax.set_ylabel("UTM 43N Northing (km)", fontsize=10)
    moran_i = compute_morans_i(x_utm_test, y_utm_test, residuals)
    ax.set_title(f"Spatial Error Map (Moran's I = {moran_i:.3f})", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "pinn_residuals_diagnostic.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_pinn_loss_and_scatter(history, y_true_test, y_pred_test, r2_test, rmse_test):
    """Plot PINN training loss curves and independent test scatter."""
    # 1. Loss Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["loss_total"]) + 1)
    ax.plot(epochs, history["loss_total"], label="Total Loss", color="#1b6ca8", lw=2)
    ax.plot(epochs, history["loss_data"], label="Data Loss (MSE)", color="#2a9d5c", lw=1.5, ls="--")
    ax.plot(epochs, history["loss_pde"], label="Physics Residual (PDE)", color="#d95f02", lw=1.5, ls=":")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Urban Heat PINN Loss Convergence", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    loss_path = os.path.join(config.PLOTS_DIR, "pinn_loss_curve.png")
    fig.tight_layout()
    fig.savefig(loss_path, dpi=200)
    plt.close(fig)
    log.info("Saved %s", loss_path)

    # 2. Test Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_test, y_pred_test, alpha=0.4, color="#1b6ca8", edgecolors="none", s=25)
    lims = [min(min(y_true_test), min(y_pred_test)) - 1, max(max(y_true_test), max(y_pred_test)) + 1]
    ax.plot(lims, lims, color="#d95f02", lw=1.5, ls="--", label="1:1 Perfect Fit")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed LST (°C)", fontsize=11)
    ax.set_ylabel("PINN Predicted LST (°C)", fontsize=11)
    ax.set_title(f"PINN Out-of-Sample Test (R² = {r2_test:.3f}, RMSE = {rmse_test:.2f} °C)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, ls="--", alpha=0.5)
    scatter_path = os.path.join(config.PLOTS_DIR, "pinn_prediction_scatter.png")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)
    log.info("Saved %s", scatter_path)


def train_pinn(epochs=60, batch_size=128):
    """Train and evaluate the Physics-Informed Neural Network alongside ML baselines."""
    config.ensure_dirs()
    df = ensure_training_data()

    # Filter to valid grid cells
    valid = df[df["LST"].notna() & df["NDVI"].notna()].copy()
    log.info("Total available grid samples: %d", len(valid))

    # 1. Create Spatially Contiguous Block Holdout Split
    train_df, test_df, test_mask = create_spatial_block_split(valid, block_size_m=8000, test_fraction=0.25, seed=42)
    plot_spatial_split_map(valid, test_mask)

    # Feature definitions
    feature_cols = ["x_utm", "y_utm", "NDVI", "NDBI", "building_density", "mean_footprint_m2"]
    X_train_raw = train_df[feature_cols].fillna(0.0)
    y_train = train_df["LST"].values
    X_test_raw = test_df[feature_cols].fillna(0.0)
    y_test = test_df["LST"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    benchmark_records = []

    # ------------------------------------------------------------- Baseline 0: Mean Baseline
    log.info("Evaluating Baseline 0: Mean Predictor Baseline (predicting training mean)...")
    p_mean_tr = np.full_like(y_train, float(np.mean(y_train)))
    p_mean_te = np.full_like(y_test, float(np.mean(y_train)))
    benchmark_records.append({
        "Model": "Mean Baseline",
        "Train_R2": round(float(r2_score(y_train, p_mean_tr)), 4),
        "Train_RMSE": round(float(np.sqrt(mean_squared_error(y_train, p_mean_tr))), 4),
        "Test_R2": round(float(r2_score(y_test, p_mean_te)), 4),
        "Test_RMSE": round(float(np.sqrt(mean_squared_error(y_test, p_mean_te))), 4),
        "Test_MAE": round(float(mean_absolute_error(y_test, p_mean_te)), 4),
    })

    # ------------------------------------------------------------- Baseline 1: Ridge
    log.info("Training Baseline 1: Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    p_ridge_tr = ridge.predict(X_train_scaled)
    p_ridge_te = ridge.predict(X_test_scaled)
    benchmark_records.append({
        "Model": "Ridge Regression",
        "Train_R2": round(r2_score(y_train, p_ridge_tr), 4),
        "Train_RMSE": round(np.sqrt(mean_squared_error(y_train, p_ridge_tr)), 4),
        "Test_R2": round(r2_score(y_test, p_ridge_te), 4),
        "Test_RMSE": round(np.sqrt(mean_squared_error(y_test, p_ridge_te)), 4),
        "Test_MAE": round(mean_absolute_error(y_test, p_ridge_te), 4),
    })

    # ------------------------------------------------------------- Baseline 2: Random Forest
    log.info("Training Baseline 2: Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
    rf.fit(X_train_raw, y_train)
    p_rf_tr = rf.predict(X_train_raw)
    p_rf_te = rf.predict(X_test_raw)
    benchmark_records.append({
        "Model": "Random Forest",
        "Train_R2": round(r2_score(y_train, p_rf_tr), 4),
        "Train_RMSE": round(np.sqrt(mean_squared_error(y_train, p_rf_tr)), 4),
        "Test_R2": round(r2_score(y_test, p_rf_te), 4),
        "Test_RMSE": round(np.sqrt(mean_squared_error(y_test, p_rf_te)), 4),
        "Test_MAE": round(mean_absolute_error(y_test, p_rf_te), 4),
    })

    # ------------------------------------------------------------- Baseline 3: MLP (No Physics)
    log.info("Training Baseline 3: Deep MLP (Neural Ablation, lambda_pde = 0.0)...")
    mlp_pipeline = PINNPipeline(lambda_pde=0.0, lr=0.002)
    mlp_pipeline.train(train_df, epochs=epochs, batch_size=batch_size)
    p_mlp_tr = mlp_pipeline.predict(train_df)
    p_mlp_te = mlp_pipeline.predict(test_df)
    benchmark_records.append({
        "Model": "Deep MLP (No PDE)",
        "Train_R2": round(r2_score(y_train, p_mlp_tr), 4),
        "Train_RMSE": round(np.sqrt(mean_squared_error(y_train, p_mlp_tr)), 4),
        "Test_R2": round(r2_score(y_test, p_mlp_te), 4),
        "Test_RMSE": round(np.sqrt(mean_squared_error(y_test, p_mlp_te)), 4),
        "Test_MAE": round(mean_absolute_error(y_test, p_mlp_te), 4),
    })

    # ------------------------------------------------------------- Proposed: PINN
    log.info("Training Proposed: Physics-Informed Neural Network (PINN, lambda_pde = 0.1)...")
    pinn_pipeline = PINNPipeline(lambda_pde=0.1, lr=0.002)
    history = pinn_pipeline.train(train_df, epochs=epochs, batch_size=batch_size)
    p_pinn_tr = pinn_pipeline.predict(train_df)
    p_pinn_te = pinn_pipeline.predict(test_df)
    benchmark_records.append({
        "Model": "Urban Heat PINN",
        "Train_R2": round(r2_score(y_train, p_pinn_tr), 4),
        "Train_RMSE": round(np.sqrt(mean_squared_error(y_train, p_pinn_tr)), 4),
        "Test_R2": round(r2_score(y_test, p_pinn_te), 4),
        "Test_RMSE": round(np.sqrt(mean_squared_error(y_test, p_pinn_te)), 4),
        "Test_MAE": round(mean_absolute_error(y_test, p_pinn_te), 4),
    })

    # ------------------------------------------------------------- Benchmark Results Table
    benchmarks_df = pd.DataFrame(benchmark_records)
    bench_csv = os.path.join(config.MODELS_DIR, "model_benchmarks.csv")
    benchmarks_df.to_csv(bench_csv, index=False)
    log.info("Saved benchmark results to %s\n%s", bench_csv, benchmarks_df.to_string(index=False))

    plot_baseline_comparison(benchmarks_df)

    # Save primary PINN model
    pinn_pipeline.save(config.PINN_MODEL_PATH, config.PINN_SCALER_JSON)

    # Residual analysis for PINN
    residuals_pinn = y_test - p_pinn_te
    moran_i = compute_morans_i(test_df["x_utm"].values, test_df["y_utm"].values, residuals_pinn)
    res_std = float(np.std(residuals_pinn))

    # Save metrics JSON
    metrics = {
        "spatial_validation_method": "Contiguous 8km Spatial Block Holdout",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_fraction": round(len(test_df) / len(valid), 4),
        "test_r2_score": float(benchmarks_df.loc[benchmarks_df["Model"] == "Urban Heat PINN", "Test_R2"].values[0]),
        "test_rmse_celsius": float(benchmarks_df.loc[benchmarks_df["Model"] == "Urban Heat PINN", "Test_RMSE"].values[0]),
        "test_mae_celsius": float(benchmarks_df.loc[benchmarks_df["Model"] == "Urban Heat PINN", "Test_MAE"].values[0]),
        "train_r2_score": float(benchmarks_df.loc[benchmarks_df["Model"] == "Urban Heat PINN", "Train_R2"].values[0]),
        "train_rmse_celsius": float(benchmarks_df.loc[benchmarks_df["Model"] == "Urban Heat PINN", "Train_RMSE"].values[0]),
        "residual_mean_celsius": float(np.mean(residuals_pinn)),
        "residual_std_celsius": res_std,
        "residual_morans_i": moran_i,
        "prediction_interval_95pct_celsius": 1.96 * res_std,
        "epochs": epochs,
        "lambda_pde": pinn_pipeline.lambda_pde,
        "learned_diffusivity_alpha": float(pinn_pipeline.model.alpha.numpy()),
        "learned_cooling_rate_beta": float(pinn_pipeline.model.beta.numpy()),
        "benchmarks": benchmark_records,
    }
    with open(config.PINN_METRICS_JSON, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Visualizations
    plot_pinn_loss_and_scatter(history, y_test, p_pinn_te, metrics["test_r2_score"], metrics["test_rmse_celsius"])
    plot_residual_diagnostics(y_test, p_pinn_te, test_df["x_utm"].values, test_df["y_utm"].values)

    # Write evaluation report markdown
    report_md = os.path.join(config.MODELS_DIR, "model_evaluation_report.md")
    with open(report_md, "w", encoding="utf-8") as fh:
        fh.write("# Model Evaluation & Benchmark Report\n\n")
        fh.write("## Spatial Block Holdout Validation\n")
        fh.write(f"- **Methodology**: 8 km contiguous spatial block holdout ({metrics['train_samples']} train samples, {metrics['test_samples']} test samples, {metrics['test_fraction']*100:.1f}% spatial holdout).\n")
        fh.write("- **Spatial Autocorrelation**: Tested with Moran's I on out-of-sample residuals.\n\n")
        fh.write("## Benchmark Comparison\n\n")
        fh.write("| Model | Train R² | Train RMSE (°C) | Test R² | Test RMSE (°C) | Test MAE (°C) |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for r in benchmark_records:
            fh.write(f"| **{r['Model']}** | {r['Train_R2']:.3f} | {r['Train_RMSE']:.2f} | **{r['Test_R2']:.3f}** | **{r['Test_RMSE']:.2f}** | {r['Test_MAE']:.2f} |\n")
        fh.write(f"\n## Residual Diagnostics (Urban Heat PINN)\n")
        fh.write(f"- **Residual Mean**: {metrics['residual_mean_celsius']:.3f} °C\n")
        fh.write(f"- **Residual Std (σ)**: {metrics['residual_std_celsius']:.3f} °C\n")
        fh.write(f"- **95% Error Margin (±1.96σ)**: ±{metrics['prediction_interval_95pct_celsius']:.2f} °C\n")
        fh.write(f"- **Residual Spatial Moran's I**: {metrics['residual_morans_i']:.3f}\n")
    log.info("Saved %s", report_md)

    return metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    return train_pinn()


if __name__ == "__main__":
    main()
