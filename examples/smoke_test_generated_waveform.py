"""Small smoke test for the reusable pipeline scripts.

This uses the generated waveform in ``lisa_link_sim/inputs/my_waveform.npz`` and
runs only the deterministic lightweight stages:

- waveform loading and alignment
- short orbit file generation
- six GW-only links
- GW-only XYZ/AET TDI

It intentionally skips the full lisainstrument noise stage so the check remains
fast and does not regenerate the main project outputs.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.six_link_generation import (
    LisaLinkSimulationConfig,
    LisaLinkSimulationPaths,
    compute_gw_links,
    create_orbit_file,
    load_npz_waveform,
    prepare_science_waveform,
)
from scripts.tdi_computation import compute_tdi_from_gws


def project_root() -> Path:
    return PROJECT_ROOT


def run_smoke_test(work_dir: Path) -> dict[str, float | int | str]:
    root = project_root()
    waveform_file = root / "lisa_link_sim" / "inputs" / "my_waveform.npz"
    t, hplus, hcross = load_npz_waveform(waveform_file)

    paths = LisaLinkSimulationPaths.from_work_dir(work_dir)
    paths.make_dirs()

    config = LisaLinkSimulationConfig(
        dt=0.1,
        science_duration=512.0,
        burn_in_seconds=128.0,
        orbit_dt=60.0,
        orbit_duration_factor=20.0,
        frequency_low_hz=1e-4,
        frequency_high_hz=5e-2,
        interpolation_order=3,
        seed=12345,
    )

    prepared = prepare_science_waveform(
        t,
        hplus,
        hcross,
        config,
        output_file=paths.aligned_waveform_npz,
        plot_file=None,
    )
    orbit = create_orbit_file(prepared, config, paths.orbits_h5)
    gw_diag = compute_gw_links(prepared, config, paths, make_plots=False)
    tdi_diag = compute_tdi_from_gws(config, paths, make_plots=False)

    if len(prepared.t) != 5120:
        raise AssertionError(f"expected 5120 prepared samples, got {len(prepared.t)}")
    if gw_diag["n_total"] != 6400:
        raise AssertionError(f"expected 6400 total GW samples, got {gw_diag['n_total']}")
    if tdi_diag["gw_tdi_samples"] != 5120:
        raise AssertionError(f"expected 5120 GW TDI samples, got {tdi_diag['gw_tdi_samples']}")
    if tdi_diag["gw_a_max_hz"] <= 0.0:
        raise AssertionError("GW-only A-channel TDI max must be positive")

    return {
        "work_dir": str(work_dir),
        "waveform_file": str(waveform_file),
        "prepared_samples": len(prepared.t),
        "peak_time_s": prepared.peak_time,
        "orbit_span_s": orbit.span,
        "gw_total_samples": int(gw_diag["n_total"]),
        "gw_tdi_samples": int(tdi_diag["gw_tdi_samples"]),
        "gw_a_max_hz": float(tdi_diag["gw_a_max_hz"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small generated-waveform pipeline smoke test.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to a temporary directory.",
    )
    args = parser.parse_args()

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="lisa_tf_smoke_") as tmp:
            summary = run_smoke_test(Path(tmp))
            for key, value in summary.items():
                print(f"{key}: {value}")
    else:
        summary = run_smoke_test(args.work_dir)
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
