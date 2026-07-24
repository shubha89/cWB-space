from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import LisaLinkSimulationConfig, LisaLinkSimulationPaths
from scripts.pytdi_coronagraphic_tdi import (
    compute_kappa_from_sagnac_time,
    geometry_from_orbit_file,
    kappa_summary,
    load_pytdi_sagnac_csv,
    write_kappa_time_csv,
)
from scripts.tdi_computation import compute_sagnac_tdi_from_gws, compute_sagnac_tdi_from_instrument


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
LOW_HZ = 1e-5
HIGH_HZ = 1e-1
SIGNAL_BETA = 0.3
SIGNAL_LAMBDA = np.pi / 4.0
FAR_BETA = -0.6
FAR_LAMBDA = np.mod(SIGNAL_LAMBDA + np.pi, 2.0 * np.pi)


def mbhb_config(generation: int) -> LisaLinkSimulationConfig:
    if generation not in {1, 2}:
        raise ValueError("generation must be 1 or 2.")
    return LisaLinkSimulationConfig(
        dt=0.1,
        science_duration=150_000.0,
        burn_in_seconds=600.0,
        center_waveform_peak=True,
        edge_taper_seconds=10_000.0,
        ra=SIGNAL_LAMBDA,
        dec=SIGNAL_BETA,
        central_freq_hz=2.816e14,
        orbit_name="esa-trailing",
        orbit_dt=600.0,
        orbit_duration_factor=5.0,
        seed=12345,
        frequency_low_hz=LOW_HZ,
        frequency_high_hz=HIGH_HZ,
        tdi_generation=generation,
        interpolation_order=31,
        tdi_unit="frequency",
        gw_dataset="tcb",
        orbit_dataset="tcb/ltt",
    )


def sagnac_paths_for_generation(base: LisaLinkSimulationPaths, generation: int) -> LisaLinkSimulationPaths:
    if generation == 2:
        return base
    if generation != 1:
        raise ValueError("generation must be 1 or 2.")
    return replace(
        base,
        noisy_sagnac_tdi_csv=base.output_dir / "tdi_sagnac_g1_noisy.csv",
        gw_sagnac_tdi_csv=base.output_dir / "tdi_sagnac_g1_gw_only.csv",
        noisy_tdi_sagnac_plot=base.plot_dir / "04_tdi_sagnac_g1_orbit_time.png",
        gw_tdi_sagnac_plot=base.plot_dir / "05_tdi_gw_only_sagnac_g1.png",
    )


def signal_reference_time(paths: LisaLinkSimulationPaths, fallback: np.ndarray) -> float:
    report_path = paths.output_dir / "mbhb_sixlink_diagnostic_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        value = report.get("aet_signal_center_lisa")
        if value is not None:
            return float(value)
    return float(np.median(fallback))


def compute_for_sky_positions(
    t_waveform: np.ndarray,
    t_lisa: np.ndarray,
    channels: dict[str, np.ndarray],
    geometry,
) -> dict[str, dict[str, np.ndarray]]:
    dt = float(np.median(np.diff(t_waveform)))
    positions = {
        "signal_direction": (SIGNAL_BETA, SIGNAL_LAMBDA),
        "far_direction": (FAR_BETA, FAR_LAMBDA),
    }
    return {
        name: compute_kappa_from_sagnac_time(
            channels["alpha"],
            channels["beta"],
            channels["gamma"],
            dt=dt,
            beta=beta,
            lamb=lamb,
            geometry=geometry,
        )
        for name, (beta, lamb) in positions.items()
    }


def plot_kappa_time(
    output: Path,
    t_waveform: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, result in results.items():
        ax.plot(t_waveform / 86400.0, np.abs(result["kappa"]), lw=0.8, label=name.replace("_", " "))
    ax.set_title(title)
    ax.set_xlabel("Waveform time [days]")
    ax.set_ylabel(r"$|\kappa(t)|$ [Hz]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_kappa_frequency(
    output: Path,
    results: dict[str, dict[str, np.ndarray]],
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, result in results.items():
        freqs = result["freqs_hz"]
        mask = (freqs > 0.0) & (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
        ax.loglog(freqs[mask], np.abs(result["kappa_tilde"][mask]), lw=0.9, label=name.replace("_", " "))
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$|\tilde{\kappa}(f)|$ [Hz samples]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def report_for_results(results: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    return {
        name: kappa_summary(result["kappa"], result["freqs_hz"], (LOW_HZ, HIGH_HZ))
        for name, result in results.items()
    }


def main() -> None:
    parser = ArgumentParser(description="Run fresh coronagraphic TDI from PyTDI Sagnac channels.")
    parser.add_argument(
        "--generation",
        type=int,
        choices=(1, 2),
        default=1,
        help="PyTDI Sagnac generation to use. G1 is the paper-faithful default; G2 is experimental.",
    )
    parser.add_argument("--reuse-sagnac", action="store_true", help="Reuse existing Sagnac CSVs if present.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG diagnostic plots.")
    args = parser.parse_args()

    generation = int(args.generation)
    config = mbhb_config(generation)
    paths = sagnac_paths_for_generation(LisaLinkSimulationPaths.from_work_dir(WORK_DIR), generation)
    paths.make_dirs()
    make_plots = not args.no_plots

    if not args.reuse_sagnac or not paths.noisy_sagnac_tdi_csv.exists():
        noisy_report = compute_sagnac_tdi_from_instrument(config, paths, make_plots=make_plots)
    else:
        noisy_report = {"reused": str(paths.noisy_sagnac_tdi_csv)}

    if not args.reuse_sagnac or not paths.gw_sagnac_tdi_csv.exists():
        gw_report = compute_sagnac_tdi_from_gws(config, paths, make_plots=make_plots)
    else:
        gw_report = {"reused": str(paths.gw_sagnac_tdi_csv)}

    t_noisy, lisa_noisy, noisy_channels = load_pytdi_sagnac_csv(paths.noisy_sagnac_tdi_csv, generation=generation)
    t_gw, lisa_gw, gw_channels = load_pytdi_sagnac_csv(paths.gw_sagnac_tdi_csv, generation=generation, suffix="_gw_hz")
    if not (np.array_equal(t_noisy, t_gw) and np.array_equal(lisa_noisy, lisa_gw)):
        raise ValueError("Noisy and GW-only Sagnac products do not share the same time grid.")

    t_ref = signal_reference_time(paths, lisa_gw)
    geometry = geometry_from_orbit_file(paths.orbits_h5, t_lisa_reference=t_ref)

    noisy_results = compute_for_sky_positions(t_noisy, lisa_noisy, noisy_channels, geometry)
    gw_results = compute_for_sky_positions(t_gw, lisa_gw, gw_channels, geometry)

    tag = f"g{generation}"
    noisy_csv = paths.output_dir / f"coronagraphic_pytdi_{tag}_noisy.csv"
    gw_csv = paths.output_dir / f"coronagraphic_pytdi_{tag}_gw_only.csv"
    write_kappa_time_csv(noisy_csv, t_waveform=t_noisy, t_lisa=lisa_noisy, results=noisy_results)
    write_kappa_time_csv(gw_csv, t_waveform=t_gw, t_lisa=lisa_gw, results=gw_results)

    plots = {}
    if make_plots:
        noisy_time_plot = paths.plot_dir / f"07_coronagraphic_pytdi_{tag}_time_noisy.png"
        gw_time_plot = paths.plot_dir / f"07_coronagraphic_pytdi_{tag}_time_gw_only.png"
        gw_freq_plot = paths.plot_dir / f"07_coronagraphic_pytdi_{tag}_frequency_gw_only.png"
        plot_kappa_time(
            noisy_time_plot,
            t_noisy,
            noisy_results,
            f"Coronagraphic TDI from noisy PyTDI Sagnac G{generation}",
        )
        plot_kappa_time(
            gw_time_plot,
            t_gw,
            gw_results,
            f"Coronagraphic TDI from GW-only PyTDI Sagnac G{generation}",
        )
        plot_kappa_frequency(
            gw_freq_plot,
            gw_results,
            f"GW-only coronagraphic TDI from PyTDI Sagnac G{generation}",
        )
        plots = {
            "noisy_time": str(noisy_time_plot),
            "gw_time": str(gw_time_plot),
            "gw_frequency": str(gw_freq_plot),
        }

    report = {
        "implementation": f"fresh PyTDI Sagnac G{generation} coronagraphic TDI",
        "paper_convention": "frequency-domain first-generation Sagnac alpha, beta, gamma",
        "sagnac_generation": generation,
        "status": "paper-faithful" if generation == 1 else "experimental optional test case",
        "generation_note": (
            "G1 uses the Sagnac basis used in the coronagraphic TDI paper. "
            "G2 applies the same static projection to PyTDI alpha2/beta2/gamma2 and is for development diagnostics."
        ),
        "tdi_unit": config.tdi_unit,
        "frequency_band_hz": [LOW_HZ, HIGH_HZ],
        "sky_positions": {
            "signal_direction": {"beta": SIGNAL_BETA, "lambda": SIGNAL_LAMBDA},
            "far_direction": {"beta": FAR_BETA, "lambda": FAR_LAMBDA},
        },
        "geometry": {
            "source": geometry.source,
            "t_lisa_reference": geometry.t_lisa_reference,
            "arm_lengths_light_seconds": {key: float(value) for key, value in geometry.L.items()},
        },
        "time_grid": {
            "samples": int(len(t_gw)),
            "dt_seconds": float(np.median(np.diff(t_gw))),
            "t_waveform_start": float(t_gw[0]),
            "t_waveform_stop": float(t_gw[-1]),
            "t_lisa_start": float(lisa_gw[0]),
            "t_lisa_stop": float(lisa_gw[-1]),
        },
        "sagnac_noisy": noisy_report,
        "sagnac_gw_only": gw_report,
        "kappa_noisy": report_for_results(noisy_results),
        "kappa_gw_only": report_for_results(gw_results),
        "outputs": {
            "noisy_csv": str(noisy_csv),
            "gw_only_csv": str(gw_csv),
            "plots": plots,
        },
    }
    report_path = paths.output_dir / f"coronagraphic_pytdi_{tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report = {report_path}")


if __name__ == "__main__":
    main()
