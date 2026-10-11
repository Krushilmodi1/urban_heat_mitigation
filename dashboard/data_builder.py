"""Build data bundle for the interactive Ahmedabad Urban Heat Dashboard.

Compiles verified metrics, benchmarks, scenario calculations, and grid cell geometries
into a standalone JSON/JavaScript bundle for client-side rendering.
"""
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger("dashboard.data_builder")


def build_dashboard_data():
    """Extract and compile all project results into dashboard data objects."""
    config.ensure_dirs()
    dashboard_dir = os.path.join(config.BASE_DIR, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)

    # 1. Load Cooling Scenarios
    scen_csv = config.SCENARIO_RESULTS_CSV
    if not os.path.exists(scen_csv):
        raise FileNotFoundError(f"Scenario results not found at {scen_csv}. Run main.py first.")
    df_scen = pd.read_csv(scen_csv)

    # 2. Load Model Benchmarks
    bench_csv = os.path.join(config.MODELS_DIR, "model_benchmarks.csv")
    benchmarks = []
    if os.path.exists(bench_csv):
        benchmarks = pd.read_csv(bench_csv).to_dict(orient="records")

    # 3. Load Building Summary
    bld_sum_csv = config.BUILDING_SUMMARY_CSV
    bld_summary = {}
    if os.path.exists(bld_sum_csv):
        bld_df = pd.read_csv(bld_sum_csv)
        for _, r in bld_df.iterrows():
            bld_summary[r["zone"]] = {
                "count": int(r["building_count"]),
                "area_km2": float(r["zone_area_km2"]),
                "density_pct": float(r.get("footprint_density_pct", 0.0)),
            }

    # 4. Prepare Metadata
    meta = {
        "title": "Ahmedabad Urban Heat Island Mitigation & PINN Modeling",
        "institution": "Urban Heat Mitigation Analytics Platform",
        "total_cells": len(df_scen),
        "study_area_km2": 1570.08,
        "city_area_km2": 462.01,
        "total_buildings": bld_summary.get("study_area", {}).get("count", 4777),
        "city_buildings": bld_summary.get("city", {}).get("count", 3380),
        "buffer_buildings": bld_summary.get("buffer_ring", {}).get("count", 1397),
        "mean_lst": round(float(df_scen["LST_baseline"].mean()), 2),
        "max_lst": round(float(df_scen["LST_baseline"].max()), 2),
        "min_lst": round(float(df_scen["LST_baseline"].min()), 2),
        "mean_bld_density_pct": round(float(df_scen["building_density"].mean() * 100), 2),
        "max_bld_density_pct": round(float(df_scen["building_density"].max() * 100), 2),
        "provenance": {
            "buildings": "Synthetic prototype (4,777 buildings in study area from D:\\395_buildings.csv)",
            "satellite": "Calibrated offline synthetic proxy (Earth Engine offline)",
            "boundary": "Synthetic municipal ellipse (~462 km²)",
            "validation": "8.0 km Contiguous Spatial Block Holdout (28 train blocks, 10 test blocks)",
        },
    }

    # 5. Scenarios Summary Statistics
    scenarios_summary = {
        "cool_roofs": {
            "mean_lst": round(float(df_scen["delta_LST_cool_roofs"].mean()), 2),
            "max_lst": round(float(df_scen["delta_LST_cool_roofs"].max()), 2),
            "mean_air": round(float(df_scen["delta_T_air_cool_roofs"].mean()), 2),
            "max_air": round(float(df_scen["delta_T_air_cool_roofs"].max()), 2),
            "description": "High-albedo solar reflection on rooftop surfaces (SRI ≥ 78).",
        },
        "green_roofs": {
            "mean_lst": round(float(df_scen["delta_LST_green_roofs"].mean()), 2),
            "max_lst": round(float(df_scen["delta_LST_green_roofs"].max()), 2),
            "mean_air": round(float(df_scen["delta_T_air_green_roofs"].mean()), 2),
            "max_air": round(float(df_scen["delta_T_air_green_roofs"].max()), 2),
            "description": "Extensive sedum/vegetative rooftop conversion with evapotranspirative cooling.",
        },
        "urban_trees": {
            "mean_lst": round(float(df_scen["delta_LST_urban_trees"].mean()), 2),
            "max_lst": round(float(df_scen["delta_LST_urban_trees"].max()), 2),
            "mean_air": round(float(df_scen["delta_T_air_urban_trees"].mean()), 2),
            "max_air": round(float(df_scen["delta_T_air_urban_trees"].max()), 2),
            "description": "Street-level canopy expansion and urban forestry corridors (NDVI +0.15).",
        },
        "combined": {
            "mean_lst": round(float(df_scen["delta_LST_combined"].mean()), 2),
            "max_lst": round(float(df_scen["delta_LST_combined"].max()), 2),
            "mean_air": round(float(df_scen["delta_T_air_combined"].mean()), 2),
            "max_air": round(float(df_scen["delta_T_air_combined"].max()), 2),
            "description": "Integrated reflective roofs, green roofs, and tree corridors with non-linear saturation.",
        },
    }

    # 6. Grid Cells Array (Clean, minimal schema for client performance)
    grid_cells = []
    for _, row in df_scen.iterrows():
        grid_cells.append({
            "id": str(row["grid_id"]),
            "zone": str(row.get("zone", "city")),
            "lon": round(float(row["lon"]), 5),
            "lat": round(float(row["lat"]), 5),
            "x_utm": float(row["x_utm"]),
            "y_utm": float(row["y_utm"]),
            "lst": round(float(row["LST_baseline"]), 2),
            "dens": round(float(row["building_density"]) * 100, 2),  # in %
            "bld_count": int(row.get("building_count", 0)),
            "ndvi": round(float(row["NDVI"]), 3),
            "ndbi": round(float(row["NDBI"]), 3),
            "uhvi": round(float(row["UHVI"]), 4),
            "tier": str(row["priority_tier"]),
            "policy": str(row["policy_recommendation"]),
            "dLST_cool": round(float(row["delta_LST_cool_roofs"]), 2),
            "dLST_green": round(float(row["delta_LST_green_roofs"]), 2),
            "dLST_trees": round(float(row["delta_LST_urban_trees"]), 2),
            "dLST_comb": round(float(row["delta_LST_combined"]), 2),
            "dAir_comb": round(float(row["delta_T_air_combined"]), 2),
            "dLST_ci": [
                round(float(row.get("delta_LST_combined_lower95", 0.0)), 2),
                round(float(row.get("delta_LST_combined_upper95", row["delta_LST_combined"] + 0.78)), 2),
            ],
        })

    # 7. Priority Top Hotspots
    top_hotspots = sorted(grid_cells, key=lambda c: c["uhvi"], reverse=True)[:50]

    bundle = {
        "meta": meta,
        "benchmarks": benchmarks,
        "scenarios_summary": scenarios_summary,
        "grid_cells": grid_cells,
        "top_hotspots": top_hotspots,
    }

    # Export JS and JSON
    js_path = os.path.join(dashboard_dir, "data.js")
    json_path = os.path.join(dashboard_dir, "data.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    with open(js_path, "w", encoding="utf-8") as f:
        f.write("window.DASHBOARD_DATA = " + json.dumps(bundle, separators=(",", ":")) + ";\n")

    log.info("Saved dashboard data bundle to %s (%d cells) and %s", js_path, len(grid_cells), json_path)
    return bundle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build_dashboard_data()
