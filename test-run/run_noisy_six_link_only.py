"""Generate noisy six-link products for the fixed test-run catalog.

This intentionally stops after ``six_links_noisy.csv`` and
``six_links_gw_only.csv`` because the coronagraphic WDM+kappa analysis consumes
six one-way links directly and does not need TDI CSV/HDF products.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cluster.config import load_pipeline_config
from cluster.waveforms import generate_waveform_for_source
from scripts.lisa_link_simulation import load_npz_waveform
from scripts.six_link_generation import (
    LisaLinkSimulationConfig,
    LisaLinkSimulationPaths,
    compute_gw_links,
    create_orbit_file,
    prepare_science_waveform,
    run_lisa_instrument,
)


def build_lisa_config(config) -> LisaLinkSimulationConfig:
    sim = config.simulation
    return LisaLinkSimulationConfig(
        dt=sim.dt_s,
        science_duration=sim.science_duration_s,
        burn_in_seconds=sim.burn_in_seconds,
        center_waveform_peak=True,
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


def run_source(row: pd.Series, *, config, runs_root: Path, force: bool) -> Path:
    work_dir = runs_root / str(row["source_id"])
    paths = LisaLinkSimulationPaths.from_work_dir(work_dir)
    paths.make_dirs()

    if paths.noisy_links_csv.exists() and paths.gw_links_csv.exists() and not force:
        return work_dir

    waveform_path, waveform_metadata_path = generate_waveform_for_source(
        row,
        paths.input_dir,
        config=config.waveform,
        force=force,
    )
    t, hplus, hcross = load_npz_waveform(waveform_path)
    lisa_config = build_lisa_config(config)
    lisa_config = LisaLinkSimulationConfig(
        **{
            **lisa_config.__dict__,
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
        }
    )

    prepared = prepare_science_waveform(
        t,
        hplus,
        hcross,
        lisa_config,
        output_file=paths.aligned_waveform_npz,
        plot_file=None,
    )
    orbit = create_orbit_file(prepared, lisa_config, paths.orbits_h5)
    gw_diag = compute_gw_links(prepared, lisa_config, paths, make_plots=False)
    noisy_diag = run_lisa_instrument(lisa_config, paths, make_plots=False)

    diagnostics = {
        "source_id": str(row["source_id"]),
        "waveform_path": str(waveform_path),
        "waveform_metadata_path": str(waveform_metadata_path),
        "orbit_t0": orbit.t0,
        "orbit_t1": orbit.t1,
        "orbit_span": orbit.span,
        "orbit_size": orbit.size,
        **gw_diag,
        **noisy_diag,
    }
    (paths.output_dir / "six_link_only_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return work_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run noisy six-link-only test-run simulations.")
    parser.add_argument("--config", default="test-run/config.yaml")
    parser.add_argument("--catalog", default="test-run/catalog.csv")
    parser.add_argument("--runs-root", default="test-run/simulations")
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    catalog = pd.read_csv(args.catalog)
    if args.source_id:
        catalog = catalog[catalog["source_id"].isin(args.source_id)].reset_index(drop=True)
    if catalog.empty:
        raise ValueError("No catalog rows selected.")

    for _, row in catalog.iterrows():
        print(run_source(row, config=config, runs_root=Path(args.runs_root), force=args.force))


if __name__ == "__main__":
    main()
