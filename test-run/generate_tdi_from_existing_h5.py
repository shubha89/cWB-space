"""Generate pyTDI XYZ/AET CSVs from existing test-run HDF5 products.

This is a post-processing step only. It reuses the existing orbit, GW, and
lisainstrument HDF5 files and writes:

- outputs/tdi_michelson_XYZ_AET.csv
- outputs/tdi_gw_only_XYZ_AET.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cluster.config import load_pipeline_config
from scripts.six_link_generation import LisaLinkSimulationConfig, LisaLinkSimulationPaths
from scripts.tdi_computation import compute_tdi_from_gws, compute_tdi_from_instrument


def build_lisa_config(config, row: pd.Series) -> LisaLinkSimulationConfig:
    sim = config.simulation
    return LisaLinkSimulationConfig(
        dt=sim.dt_s,
        science_duration=sim.science_duration_s,
        burn_in_seconds=sim.burn_in_seconds,
        center_waveform_peak=True,
        ra=float(row["ra"]),
        dec=float(row["dec"]),
        central_freq_hz=float(sim.central_freq_hz),
        orbit_dt=sim.orbit_dt_s,
        orbit_duration_factor=sim.orbit_duration_factor,
        seed=sim.seed,
        frequency_low_hz=sim.frequency_low_hz,
        frequency_high_hz=sim.frequency_high_hz,
        tdi_generation=sim.tdi_generation,
        interpolation_order=sim.interpolation_order,
        tdi_unit="frequency",
    )


def run_source(row: pd.Series, *, config, runs_root: Path, make_plots: bool, force: bool) -> dict:
    work_dir = runs_root / str(row["source_id"])
    paths = LisaLinkSimulationPaths.from_work_dir(work_dir)
    paths.make_dirs()

    required = [paths.instrument_h5, paths.gws_h5, paths.orbits_h5, paths.gw_links_csv]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{row['source_id']} is missing required files: {missing}")

    diagnostics_path = paths.output_dir / "tdi_from_existing_h5_diagnostics.json"
    if (
        paths.noisy_tdi_csv.exists()
        and paths.gw_tdi_csv.exists()
        and diagnostics_path.exists()
        and not force
    ):
        return json.loads(diagnostics_path.read_text(encoding="utf-8"))

    lisa_config = build_lisa_config(config, row)
    noisy_diag = compute_tdi_from_instrument(lisa_config, paths, make_plots=make_plots)
    gw_diag = compute_tdi_from_gws(lisa_config, paths, make_plots=make_plots)
    diagnostics = {
        "source_id": str(row["source_id"]),
        "work_dir": str(work_dir),
        "noisy_tdi_csv": str(paths.noisy_tdi_csv),
        "gw_tdi_csv": str(paths.gw_tdi_csv),
        **noisy_diag,
        **gw_diag,
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pyTDI AET CSVs from existing test-run HDF5 files.")
    parser.add_argument("--config", default="test-run/config.yaml")
    parser.add_argument("--catalog", default="test-run/catalog.csv")
    parser.add_argument("--runs-root", default="test-run/simulations")
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    catalog = pd.read_csv(args.catalog)
    if args.source_id:
        catalog = catalog[catalog["source_id"].isin(args.source_id)].reset_index(drop=True)
    if catalog.empty:
        raise ValueError("No catalog rows selected.")

    for _, row in catalog.iterrows():
        diagnostics = run_source(
            row,
            config=config,
            runs_root=Path(args.runs_root),
            make_plots=args.make_plots,
            force=args.force,
        )
        print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
