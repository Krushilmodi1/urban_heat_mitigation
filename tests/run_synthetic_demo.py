"""Run the whole offline chain on SYNTHETIC data in a separate folder, so you
can look at every map and table without Earth Engine or the real inputs.

    python tests/run_synthetic_demo.py  C:\\temp\\uhm_demo

Nothing is written inside the project folder. All values are invented.
"""
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402

import config  # noqa: E402
import synthetic_data as syn  # noqa: E402


def main(out_dir):
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    config.set_base_dir(os.path.abspath(out_dir))
    config.ensure_dirs()
    from data_pipeline import open_buildings, spatial_grid, study_area
    from visualization import plots
    syn.write_boundary(config.BOUNDARY_FILE)
    study_area.prepare_study_area(force=True)
    syn.write_buildings_csv(config.OPEN_BUILDINGS_CSV, n=20000)
    open_buildings.process_buildings(force=True)
    df = syn.write_monthly_csv(config.MONTHLY_CSV)
    _, buf = study_area.load_study_area()
    syn.write_lst_tifs(config, df, buf.bounds)
    syn.write_grid_tifs(config, [2024, 2025], config.BUILDINGS_GRID_CSV)
    grid = spatial_grid.build_grid_dataset()
    plots.make_all_plots(df, grid, pd.read_csv(config.BUILDING_SUMMARY_CSV))
    print(f"\nSYNTHETIC demo outputs written to {config.BASE_DIR}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "uhm_synthetic_demo"))
