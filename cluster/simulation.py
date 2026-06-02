"""Wrappers around the existing LISA link and TDI simulation pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import LisaLinkSimulationConfig, load_npz_waveform, run_lisa_link_simulation

from .config import SimulationConfig, WaveformConfig
from .waveforms import generate_waveform_for_source


def source_work_dir(runs_root: str | Path, source_id: str) -> Path:
    return Path(runs_root) / str(source_id)


def run_source_simulation(
    source: pd.Series,
    *,
    runs_root: str | Path,
    waveform_config: WaveformConfig,
    simulation_config: SimulationConfig,
    force_waveform: bool = False,
    force_simulation: bool = False,
    make_plots: bool = False,
) -> Path:
    """Generate one waveform and run the existing link/TDI pipeline."""

    work_dir = source_work_dir(runs_root, str(source["source_id"]))
    input_dir = work_dir / "inputs"
    output_dir = work_dir / "outputs"
    diagnostics_path = output_dir / "cluster_simulation_diagnostics.json"
    if diagnostics_path.exists() and not force_simulation:
        return work_dir

    waveform_path, waveform_metadata_path = generate_waveform_for_source(
        source,
        input_dir,
        config=waveform_config,
        force=force_waveform,
    )
    t, hplus, hcross = load_npz_waveform(waveform_path)
    lisa_config = LisaLinkSimulationConfig(
        dt=simulation_config.dt_s,
        science_duration=simulation_config.science_duration_s,
        burn_in_seconds=simulation_config.burn_in_seconds,
        center_waveform_peak=True,
        ra=float(source["ra"]),
        dec=float(source["dec"]),
        central_freq_hz=simulation_config.central_freq_hz,
        orbit_dt=simulation_config.orbit_dt_s,
        orbit_duration_factor=simulation_config.orbit_duration_factor,
        seed=simulation_config.seed,
        frequency_low_hz=simulation_config.frequency_low_hz,
        frequency_high_hz=simulation_config.frequency_high_hz,
        tdi_generation=simulation_config.tdi_generation,
        interpolation_order=simulation_config.interpolation_order,
        tdi_unit="frequency",
    )
    products = run_lisa_link_simulation(
        t,
        hplus,
        hcross,
        work_dir=work_dir,
        config=lisa_config,
        make_plots=make_plots,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "source_id": str(source["source_id"]),
        "waveform_path": str(waveform_path),
        "waveform_metadata_path": str(waveform_metadata_path),
        **dict(products.diagnostics),
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return work_dir


def run_catalog_simulations(
    catalog: pd.DataFrame,
    *,
    runs_root: str | Path,
    waveform_config: WaveformConfig,
    simulation_config: SimulationConfig,
    start_index: int = 0,
    limit: int | None = None,
    force_waveform: bool = False,
    force_simulation: bool = False,
    make_plots: bool = False,
) -> list[Path]:
    """Run simulations for each row in a catalog."""

    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative when provided.")

    rows = catalog.iloc[start_index:]
    rows = rows.head(limit) if limit is not None else rows
    work_dirs: list[Path] = []
    for _, source in rows.iterrows():
        work_dirs.append(
            run_source_simulation(
                source,
                runs_root=runs_root,
                waveform_config=waveform_config,
                simulation_config=simulation_config,
                force_waveform=force_waveform,
                force_simulation=force_simulation,
                make_plots=make_plots,
            )
        )
    return work_dirs
