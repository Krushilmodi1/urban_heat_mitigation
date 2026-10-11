"""Main orchestration for the Ahmedabad urban heat mitigation project.

    python main.py --test               quick check: first 3 months only
    python main.py                      full pipeline for the period in config.py
    python main.py --offline            everything that does not need Earth Engine
    python main.py --demo               run with auto-bootstrapped demo inputs if raw files missing
    python main.py --stage plots        redraw everything offline from saved files
    python main.py --stage pinn         train Physics-Informed Neural Network on spatial grid
    python main.py --stage simulation   run urban cooling scenario interventions
    python main.py --start 2024-01-01 --end 2025-12-31

Stages (in order):
  boundary    validate the municipal boundary, build the 10 km buffer
  extract     Landsat monthly stats, monthly/annual GeoTIFFs, grid rasters (Earth Engine)
  buildings   inspect + filter + aggregate the uploaded Open Buildings CSV
  grid        join Landsat and building grids on grid_id
  plots       maps, trends, correlations, report
  pinn        train Physics-Informed Neural Network for urban heat dynamics
  simulation  simulate urban cooling scenarios (cool roofs, green roofs, urban trees)

A failed stage is reported at the end and the exit code is non-zero; stages that
depend on it are not run. Nothing is skipped silently.
"""
import argparse
import logging
import os
import sys

import pandas as pd

import config

log = logging.getLogger("main")
ORDER = ["boundary", "extract", "buildings", "grid", "plots", "pinn", "simulation"]


def stage_boundary(args):
    from data_pipeline import study_area
    if args.demo and not os.path.exists(config.BOUNDARY_FILE):
        log.info("Demo mode: generating synthetic municipal boundary...")
        from tests import synthetic_data as syn
        syn.write_boundary(config.BOUNDARY_FILE)
    rec = study_area.prepare_study_area(force=args.force)
    if rec.get("kind") == "district":
        log.warning("STUDY AREA IS THE DISTRICT, NOT THE CITY - results are for pipeline testing only.")


def stage_extract(args):
    from data_pipeline import extract_gee_data as gee
    from data_pipeline import spatial_grid
    df, city, buffered = gee.run_pipeline(args.start, args.end, force=args.force)
    failed = spatial_grid.download_grid_rasters(df, city, buffered, force=args.force,
                                                start_date=args.start, end_date=args.end)
    if failed:
        raise RuntimeError(f"grid rasters failed for {failed}")


def stage_buildings(args):
    from data_pipeline import open_buildings
    if args.demo and not os.path.exists(config.OPEN_BUILDINGS_CSV):
        log.info("Demo mode: generating synthetic Open Buildings CSV...")
        from tests import synthetic_data as syn
        syn.write_buildings_csv(config.OPEN_BUILDINGS_CSV, n=10000)
    open_buildings.process_buildings(force=args.force)


def stage_grid(args):
    from data_pipeline import spatial_grid
    if args.offline and os.path.exists(config.BUILDINGS_GRID_CSV):
        from tests import synthetic_data as syn
        syn.write_grid_tifs(config, (2024, 2025), config.BUILDINGS_GRID_CSV)
    spatial_grid.build_grid_dataset()


def stage_plots(args):
    from visualization import plots
    df = pd.read_csv(config.MONTHLY_CSV) if os.path.exists(config.MONTHLY_CSV) else None
    grid = pd.read_csv(config.GRID_CSV) if os.path.exists(config.GRID_CSV) else None
    bsum = pd.read_csv(config.BUILDING_SUMMARY_CSV) if os.path.exists(config.BUILDING_SUMMARY_CSV) else None
    plots.make_all_plots(df, grid, bsum)


def stage_pinn(args):
    """Train Physics-Informed Neural Network (PINN) for urban heat dynamics."""
    from models import train_pinn
    log.info("Starting Physics-Informed Neural Network (PINN) training...")
    epochs = 20 if args.test else 60
    train_pinn.train_pinn(epochs=epochs)


def stage_simulation(args):
    """Run urban heat mitigation scenario simulations."""
    from simulation import cooling_scenario
    log.info("Simulating urban cooling mitigation scenarios...")
    cooling_scenario.run_cooling_simulation()


def stage_dashboard(args):
    """Build dashboard data bundle and serve interactive presentation dashboard."""
    from dashboard import data_builder, serve
    log.info("Building presentation dashboard data bundle...")
    data_builder.build_dashboard_data()
    serve.run_dashboard(port=getattr(args, "port", 8000),
                        open_browser=not getattr(args, "no_browser", False),
                        blocking=True)


STAGES = {
    "boundary": stage_boundary,
    "extract": stage_extract,
    "buildings": stage_buildings,
    "grid": stage_grid,
    "plots": stage_plots,
    "pinn": stage_pinn,
    "simulation": stage_simulation,
    "dashboard": stage_dashboard,
}

# Dependency graph for same-invocation runs
NEEDS = {
    "extract": ["boundary"],
    "buildings": ["boundary"],
    "grid": ["boundary"],
    "plots": [],
    "pinn": ["grid"],
    "simulation": ["grid"],
    "dashboard": ["simulation"],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all"] + list(STAGES), default="all",
                        help="stage to run (default: all)")
    parser.add_argument("--start", default=config.START_DATE, help="inclusive, YYYY-MM-DD")
    parser.add_argument("--end", default=config.END_DATE, help="inclusive, YYYY-MM-DD")
    parser.add_argument("--test", action="store_true", help="fast test: first 3 months and fewer PINN epochs")
    parser.add_argument("--offline", action="store_true", help="skip Earth Engine (extract) in --stage all")
    parser.add_argument("--demo", action="store_true", help="bootstrap synthetic data if raw boundary/CSV missing")
    parser.add_argument("--skip-pinn", action="store_true", help="skip ML/PINN training in --stage all")
    parser.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    parser.add_argument("--dashboard", action="store_true",
                        help="launch interactive research presentation dashboard in web browser")
    parser.add_argument("--port", type=int, default=8000, help="port for the dashboard server (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="suppress auto-opening web browser for dashboard")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose debug logging and tracebacks")
    args = parser.parse_args(argv)

    if args.dashboard:
        from dashboard import data_builder, serve
        data_builder.build_dashboard_data()
        serve.run_dashboard(port=args.port, open_browser=not args.no_browser, blocking=True)
        return 0

    # Validate dates
    try:
        pd.to_datetime(args.start)
        pd.to_datetime(args.end)
    except Exception as err:
        parser.error(f"Invalid date format (must be YYYY-MM-DD): {err}")

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    config.ensure_dirs()
    if args.test:
        args.end = (pd.Timestamp(args.start) + pd.offsets.MonthEnd(3)).strftime("%Y-%m-%d")
        log.info("TEST MODE: %s to %s", args.start, args.end)

    todo = list(ORDER) if args.stage == "all" else [args.stage]
    if args.stage == "all" and args.offline:
        todo = [s for s in todo if s != "extract"]
        log.info("Offline: Earth Engine extraction skipped; using any previously exported files.")

    if args.stage == "all" and args.skip_pinn:
        todo = [s for s in todo if s not in ("pinn", "simulation")]
        log.info("Skipping PINN and simulation stages as requested.")

    results = {}
    for name in todo:
        blocked = [d for d in NEEDS.get(name, []) if results.get(d, "ok") != "ok"]
        if blocked:
            results[name] = f"not run (needs {', '.join(blocked)})"
            log.error("Stage %s not run because %s failed.", name, ", ".join(blocked))
            continue

        log.info("===== stage: %s =====", name)
        try:
            STAGES[name](args)
            results[name] = "ok"
        except Exception as exc:
            results[name] = f"FAILED: {exc}"
            log.error("Stage %s failed: %s", name, exc, exc_info=args.verbose)

    log.info("===== summary =====")
    for name, res in results.items():
        (log.info if res == "ok" else log.error)("%-12s %s", name, res.splitlines()[0])
    log.info("Outputs: %s and %s", config.PLOTS_DIR, config.PROCESSED_DIR)
    return 0 if all(r == "ok" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
