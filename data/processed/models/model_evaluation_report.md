# Model Evaluation & Benchmark Report

## Spatial Block Holdout Validation
- **Methodology**: 8 km contiguous spatial block holdout (2542 train samples, 780 test samples, 23.5% spatial holdout).
- **Spatial Autocorrelation**: Tested with Moran's I on out-of-sample residuals.

## Benchmark Comparison

| Model | Train R² | Train RMSE (°C) | Test R² | Test RMSE (°C) | Test MAE (°C) |
|---|---|---|---|---|---|
| **Mean Baseline** | 0.000 | 0.41 | **-0.000** | **0.41** | 0.33 |
| **Ridge Regression** | 0.043 | 0.40 | **0.056** | **0.40** | 0.32 |
| **Random Forest** | 0.416 | 0.31 | **0.034** | **0.40** | 0.32 |
| **Deep MLP (No PDE)** | 0.038 | 0.40 | **0.041** | **0.40** | 0.32 |
| **Urban Heat PINN** | 0.014 | 0.41 | **0.014** | **0.40** | 0.32 |

## Residual Diagnostics (Urban Heat PINN)
- **Residual Mean**: 0.032 °C
- **Residual Std (σ)**: 0.403 °C
- **95% Error Margin (±1.96σ)**: ±0.79 °C
- **Residual Spatial Moran's I**: -0.006
