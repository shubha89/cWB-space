"""Reusable LISA link simulation pipeline.

Compatibility façade for the modular implementation:
- ``six_link_generation`` prepares waveforms, creates orbits, and generates links.
- ``tdi_computation`` computes noisy and GW-only XYZ/AET TDI.
- ``plotting_utils`` contains plotting helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .six_link_generation import (
    LINKS,
    LINK_LABELS,
    LisaLinkSimulationConfig,
    LisaLinkSimulationPaths,
    OrbitInfo,
    PreparedWaveform,
    compute_gw_links,
    create_orbit_file,
    load_npz_waveform,
    prepare_science_waveform,
    read_orbit_time_range,
    run_lisa_instrument,
)
from .tdi_computation import (
    apply_frequency_cut,
    compute_tdi_from_gws,
    compute_tdi_from_instrument,
)
from .plotting_utils import plot_noisy_with_gw_signal


@dataclass(frozen=True)
class LisaLinkSimulationProducts:
    """Returned paths and scalar diagnostics from a completed run."""

    paths: LisaLinkSimulationPaths
    diagnostics: Mapping[str, float | int]


def run_lisa_link_simulation(
    t: Sequence[float] | np.ndarray,
    hplus: Sequence[float] | np.ndarray,
    hcross: Sequence[float] | np.ndarray,
    *,
    work_dir: str | Path,
    config: LisaLinkSimulationConfig | None = None,
    make_plots: bool = True,
) -> LisaLinkSimulationProducts:
    """Run the full link simulation for a user-provided waveform.

    Args:
        t, hplus, hcross:
            User waveform arrays. ``t`` must be strictly increasing.
        work_dir:
            Directory where ``inputs/`` and ``outputs/`` will be written.
        config:
            Simulation settings. Use ``LisaLinkSimulationConfig()`` for notebook-equivalent
            defaults.
        make_plots:
            Whether to write PNG diagnostic plots.
    """

    if config is None:
        config = LisaLinkSimulationConfig()
    if not isinstance(config, LisaLinkSimulationConfig):
        raise TypeError("config must be a LisaLinkSimulationConfig instance.")

    paths = LisaLinkSimulationPaths.from_work_dir(work_dir)
    paths.make_dirs()

    prepared = prepare_science_waveform(
        t,
        hplus,
        hcross,
        config,
        output_file=paths.aligned_waveform_npz,
        plot_file=paths.waveform_plot if make_plots else None,
    )
    orbit = create_orbit_file(prepared, config, paths.orbits_h5)
    gw_diag = compute_gw_links(prepared, config, paths, make_plots=make_plots)
    instrument_diag = run_lisa_instrument(config, paths, make_plots=make_plots)
    noisy_tdi_diag = compute_tdi_from_instrument(config, paths, make_plots=make_plots)
    gw_tdi_diag = compute_tdi_from_gws(config, paths, make_plots=make_plots)
    overlay_diag = plot_noisy_with_gw_signal(paths, suffix=str(config.tdi_generation)) if make_plots else {}

    diagnostics: dict[str, float | int] = {
        "science_samples": len(prepared.t),
        "science_dt": prepared.dt,
        "science_fft_df_hz": 1.0 / (len(prepared.t) * prepared.dt),
        "waveform_peak_time": prepared.peak_time,
        "waveform_max_strain": float(np.max(np.sqrt(prepared.hplus**2 + prepared.hcross**2))),
        "orbit_t0": orbit.t0,
        "orbit_t1": orbit.t1,
        "orbit_span": orbit.span,
        "orbit_size": orbit.size,
        **gw_diag,
        **instrument_diag,
        **noisy_tdi_diag,
        **gw_tdi_diag,
        **overlay_diag,
    }

    return LisaLinkSimulationProducts(paths=paths, diagnostics=diagnostics)


__all__ = [
    "LINKS",
    "LINK_LABELS",
    "LisaLinkSimulationConfig",
    "LisaLinkSimulationPaths",
    "LisaLinkSimulationProducts",
    "OrbitInfo",
    "PreparedWaveform",
    "apply_frequency_cut",
    "compute_gw_links",
    "compute_tdi_from_gws",
    "compute_tdi_from_instrument",
    "create_orbit_file",
    "load_npz_waveform",
    "prepare_science_waveform",
    "read_orbit_time_range",
    "run_lisa_instrument",
    "run_lisa_link_simulation",
]
