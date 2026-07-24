from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import LisaLinkSimulationConfig, LisaLinkSimulationPaths
from scripts.tdi_computation import (
    compute_sagnac_tdi_from_gws,
    compute_sagnac_tdi_from_instrument,
    compute_tdi_from_gws,
    compute_tdi_from_instrument,
)


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
REPORT = WORK_DIR / "outputs" / "pytdi_tdi_products_report.json"


def mbhb_config() -> LisaLinkSimulationConfig:
    return LisaLinkSimulationConfig(
        dt=0.1,
        science_duration=150_000.0,
        burn_in_seconds=600.0,
        center_waveform_peak=True,
        edge_taper_seconds=10_000.0,
        ra=3.141592653589793 / 4.0,
        dec=0.3,
        central_freq_hz=2.816e14,
        orbit_name="esa-trailing",
        orbit_dt=600.0,
        orbit_duration_factor=5.0,
        seed=12345,
        frequency_low_hz=1e-5,
        frequency_high_hz=1e-1,
        tdi_generation=2,
        interpolation_order=31,
        tdi_unit="frequency",
        gw_dataset="tcb",
        orbit_dataset="tcb/ltt",
    )


def main() -> None:
    parser = ArgumentParser(description="Regenerate PyTDI Michelson and Sagnac products for the MBHB diagnostic.")
    parser.add_argument("--skip-michelson", action="store_true", help="Only compute Sagnac products.")
    parser.add_argument("--skip-sagnac", action="store_true", help="Only compute Michelson products.")
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic plots.")
    args = parser.parse_args()

    if args.skip_michelson and args.skip_sagnac:
        raise ValueError("At least one of Michelson or Sagnac products must be enabled.")

    config = mbhb_config()
    paths = LisaLinkSimulationPaths.from_work_dir(WORK_DIR)
    make_plots = not args.no_plots
    report: dict[str, object] = {
        "config": {
            "tdi_generation": config.tdi_generation,
            "interpolation_order": config.interpolation_order,
            "tdi_unit": config.tdi_unit,
            "frequency_low_hz": config.frequency_low_hz,
            "frequency_high_hz": config.frequency_high_hz,
            "gw_dataset": config.gw_dataset,
            "orbit_dataset": config.orbit_dataset,
            "central_freq_hz": config.central_freq_hz,
        },
        "outputs": {},
    }

    if not args.skip_michelson:
        report["michelson_noisy"] = compute_tdi_from_instrument(config, paths, make_plots=make_plots)
        report["michelson_gw_only"] = compute_tdi_from_gws(config, paths, make_plots=make_plots)
        report["outputs"].update(
            {
                "noisy_michelson_tdi_csv": str(paths.noisy_tdi_csv),
                "gw_michelson_tdi_csv": str(paths.gw_tdi_csv),
                "noisy_xyz_plot": str(paths.noisy_tdi_xyz_plot),
                "noisy_aet_plot": str(paths.noisy_tdi_aet_plot),
                "gw_xyz_plot": str(paths.gw_tdi_xyz_plot),
                "gw_aet_plot": str(paths.gw_tdi_aet_plot),
            }
        )

    if not args.skip_sagnac:
        report["sagnac_noisy"] = compute_sagnac_tdi_from_instrument(config, paths, make_plots=make_plots)
        report["sagnac_gw_only"] = compute_sagnac_tdi_from_gws(config, paths, make_plots=make_plots)
        report["outputs"].update(
            {
                "noisy_sagnac_tdi_csv": str(paths.noisy_sagnac_tdi_csv),
                "gw_sagnac_tdi_csv": str(paths.gw_sagnac_tdi_csv),
                "noisy_sagnac_plot": str(paths.noisy_tdi_sagnac_plot),
                "gw_sagnac_plot": str(paths.gw_tdi_sagnac_plot),
            }
        )

    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report = {REPORT}")


if __name__ == "__main__":
    main()
