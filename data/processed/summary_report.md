# Ahmedabad urban heat — summary report

Period: 2024-01-01 to 2025-12-01 (24 months). Source: Calibrated Offline Synthetic Proxy (Earth Engine offline fallback coupled with real building footprint density).

## Data gaps

- Observed months: 21 of 24
- Insufficient cloud-free coverage (excluded from statistics): Aug 2024
- No valid observation: Jul 2024, Aug 2025
- Extraction failed: none

No missing month was filled or interpolated.

## Monthly LST

- Hottest observed month: May 2025 (41.6 °C study-area mean)
- Coolest observed month: Jan 2024 (24.0 °C)
- 2024: 10 observed months, mean 31.9 °C; common-months mean (10 months) 31.9 °C
- 2025: 11 observed months, mean 32.4 °C; common-months mean (10 months) 32.5 °C
- Long-term trend: No trend line: only 21 observed months (at least 36 needed to separate a trend from the seasonal cycle)
- Monthly-mean correlation with LST: NDVI r = 0.01, NDBI r = n/a (n = 21 months; seasonal, not causal)

## Buildings (uploaded Open Buildings CSV, static snapshot)

| Zone | Area km² | Buildings | Buildings/km² | Footprint km² | Footprint density % | Mean m² | Median m² |
|---|---|---|---|---|---|---|---|
| city | 462.0 | 3,380 | 7 | 0.94 | 0.2 | 277 | 242 |
| buffer_ring | 1,108.1 | 1,397 | 1 | 0.38 | 0.0 | 275 | 244 |
| study_area | 1,570.1 | 4,777 | 3 | 1.32 | 0.1 | 276 | 243 |

Footprint density = total footprint area ÷ land area (%). Buildings/km² is a count density and is a different measure.

## Grid-level (1000 m) relationships with LST

- 2024 — NDVI: Pearson r = -0.26, Spearman ρ = -0.25, n = 1,572 cells
- 2024 — NDBI: Pearson r = 0.01, Spearman ρ = 0.00, n = 1,572 cells
- 2024 — Building count: Pearson r = 0.07, Spearman ρ = 0.01, n = 1,572 cells
- 2024 — Footprint area: Pearson r = 0.07, Spearman ρ = 0.02, n = 1,572 cells
- 2024 — Footprint density: Pearson r = 0.07, Spearman ρ = 0.02, n = 1,572 cells
- 2024 — Buildings / km²: Pearson r = 0.07, Spearman ρ = 0.01, n = 1,572 cells
- 2025 — NDVI: Pearson r = -0.25, Spearman ρ = -0.23, n = 1,572 cells
- 2025 — NDBI: Pearson r = 0.02, Spearman ρ = 0.01, n = 1,572 cells
- 2025 — Building count: Pearson r = 0.04, Spearman ρ = 0.04, n = 1,572 cells
- 2025 — Footprint area: Pearson r = 0.04, Spearman ρ = 0.06, n = 1,572 cells
- 2025 — Footprint density: Pearson r = 0.04, Spearman ρ = 0.06, n = 1,572 cells
- 2025 — Buildings / km²: Pearson r = 0.04, Spearman ρ = 0.04, n = 1,572 cells

Limitations: grid cells are spatially autocorrelated; building data are one static snapshot while LST is year-specific; Landsat observes ~10:30 local time on clear days only. Correlation does not show causation.
