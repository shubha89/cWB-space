from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import pandas as pd
from lisainstrument import Instrument

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import LisaLinkSimulationConfig, LisaLinkSimulationPaths, compute_tdi_from_instrument


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
OUTPUT_DIR = WORK_DIR / "outputs"
DEFAULT_NOISE_ONLY_INSTRUMENT = OUTPUT_DIR / "lisa_instrument_noise_only.h5"
DEFAULT_NOISE_ONLY_TDI = OUTPUT_DIR / "tdi_noise_only_XYZ_AET.csv"
DEFAULT_REPORT = OUTPUT_DIR / "noise_only_tdi_report.json"


def output_paths(seed: int) -> tuple[Path, Path, Path]:
    if seed == 54321:
        return DEFAULT_NOISE_ONLY_INSTRUMENT, DEFAULT_NOISE_ONLY_TDI, DEFAULT_REPORT
    return (
        OUTPUT_DIR / f"lisa_instrument_noise_only_seed{seed}.h5",
        OUTPUT_DIR / f"tdi_noise_only_seed{seed}_XYZ_AET.csv",
        OUTPUT_DIR / f"noise_only_tdi_seed{seed}_report.json",
    )


def main() -> None:
    parser = ArgumentParser(description="Generate a realistic noise-only MBHB diagnostic TDI file.")
    parser.add_argument("--seed", type=int, default=54321)
    args = parser.parse_args()

    noise_only_instrument, noise_only_tdi, report = output_paths(args.seed)
    base_paths = LisaLinkSimulationPaths.from_work_dir(WORK_DIR)
    timing = pd.read_csv(base_paths.gw_links_csv, usecols=["t_waveform", "t_lisa", "is_science"])
    dt = float(timing["t_waveform"].diff().median())
    n_total = len(timing)
    t0_lisa = float(timing["t_lisa"].iloc[0])

    config = LisaLinkSimulationConfig(
        dt=0.1,
        science_duration=150_000.0,
        burn_in_seconds=600.0,
        center_waveform_peak=True,
        edge_taper_seconds=10_000.0,
        central_freq_hz=2.816e14,
        orbit_name="esa-trailing",
        orbit_dt=600.0,
        orbit_duration_factor=5.0,
        seed=args.seed,
        frequency_low_hz=1e-5,
        frequency_high_hz=1e-1,
        tdi_generation=2,
        interpolation_order=31,
        tdi_unit="frequency",
        gw_dataset="tcb",
        orbit_dataset="tcb/ltt",
    )

    if noise_only_instrument.exists():
        noise_only_instrument.unlink()

    instrument = Instrument(
        size=n_total,
        dt=dt,
        t0=t0_lisa,
        orbits=str(base_paths.orbits_h5),
        gws=None,
        orbit_dataset=config.orbit_dataset,
        central_freq=config.central_freq_hz,
        seed=config.seed,
        noises_f_min_hz=config.frequency_low_hz,
        clock_f_min_hz=config.frequency_low_hz,
        chunk_gw_files=True,
    )
    if hasattr(instrument, "export_hdf5"):
        instrument.export_hdf5(path=str(noise_only_instrument), overwrite=True)
    elif hasattr(instrument, "write"):
        instrument.write(str(noise_only_instrument), overwrite=True)
    else:
        instrument.simulate()
        raise RuntimeError("Could not find Instrument.export_hdf5() or Instrument.write().")

    paths = replace(
        base_paths,
        instrument_h5=noise_only_instrument,
        noisy_tdi_csv=noise_only_tdi,
    )
    diag = compute_tdi_from_instrument(config, paths, make_plots=False)
    report.write_text(
        json.dumps(
            {
                "instrument_h5": str(noise_only_instrument),
                "tdi_csv": str(noise_only_tdi),
                "seed": config.seed,
                "orbit_dataset": config.orbit_dataset,
                "diagnostics": diag,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(diag, indent=2))
    print(f"instrument = {noise_only_instrument}")
    print(f"tdi        = {noise_only_tdi}")
    print(f"report     = {report}")


if __name__ == "__main__":
    main()
