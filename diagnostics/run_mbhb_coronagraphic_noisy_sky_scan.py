from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import LisaLinkSimulationConfig, LisaLinkSimulationPaths
from scripts.pytdi_coronagraphic_tdi import (
    compute_kappa_frequency,
    geometry_from_orbit_file,
    kappa_summary,
    load_pytdi_sagnac_csv,
    sky_to_unit_source_vector,
)


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
LOW_HZ = 1e-5
HIGH_HZ = 1e-1
SIGNAL_BETA = 0.3
SIGNAL_LAMBDA = np.pi / 4.0


def mbhb_config(generation: int) -> LisaLinkSimulationConfig:
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


def ten_sky_positions() -> list[dict[str, float | str | bool]]:
    positions: list[dict[str, float | str | bool]] = [
        {
            "label": "signal",
            "beta": SIGNAL_BETA,
            "lambda": SIGNAL_LAMBDA,
            "is_signal_direction": True,
        }
    ]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for idx in range(9):
        z = 1.0 - 2.0 * (idx + 0.5) / 9.0
        beta = float(np.arcsin(z))
        lamb = float(np.mod((idx + 0.5) * golden_angle, 2.0 * np.pi))
        positions.append(
            {
                "label": f"sky_{idx + 1:02d}",
                "beta": beta,
                "lambda": lamb,
                "is_signal_direction": False,
            }
        )
    source = sky_to_unit_source_vector(SIGNAL_BETA, SIGNAL_LAMBDA)
    for item in positions:
        vec = sky_to_unit_source_vector(float(item["beta"]), float(item["lambda"]))
        sep = np.arccos(np.clip(float(np.dot(source, vec)), -1.0, 1.0))
        item["separation_from_signal_deg"] = float(np.rad2deg(sep))
    return positions


def time_to_frequency(x: np.ndarray) -> np.ndarray:
    return np.fft.ifft(x) * x.size


def frequency_to_time(x: np.ndarray) -> np.ndarray:
    return np.fft.fft(x) / x.size


def scan_generation(generation: int, positions: list[dict[str, float | str | bool]]) -> pd.DataFrame:
    config = mbhb_config(generation)
    paths = sagnac_paths_for_generation(LisaLinkSimulationPaths.from_work_dir(WORK_DIR), generation)
    if not paths.noisy_sagnac_tdi_csv.exists():
        raise FileNotFoundError(f"Missing noisy Sagnac CSV for G{generation}: {paths.noisy_sagnac_tdi_csv}")

    t_waveform, t_lisa, channels = load_pytdi_sagnac_csv(paths.noisy_sagnac_tdi_csv, generation=generation)
    dt = float(np.median(np.diff(t_waveform)))
    freqs = np.fft.fftfreq(len(t_waveform), d=dt)
    alpha_f = time_to_frequency(channels["alpha"])
    beta_f = time_to_frequency(channels["beta"])
    gamma_f = time_to_frequency(channels["gamma"])
    geometry = geometry_from_orbit_file(paths.orbits_h5, t_lisa_reference=signal_reference_time(paths, t_lisa))

    rows: list[dict[str, float | str | int | bool]] = []
    for item in positions:
        for normalize_vector in (False, True):
            kappa_f, avec = compute_kappa_frequency(
                freqs,
                alpha_f,
                beta_f,
                gamma_f,
                beta=float(item["beta"]),
                lamb=float(item["lambda"]),
                geometry=geometry,
                normalize_vector=normalize_vector,
            )
            kappa_t = frequency_to_time(kappa_f)
            summary = kappa_summary(kappa_t, freqs, (LOW_HZ, HIGH_HZ))
            avec_norm = np.sqrt(np.sum(np.abs(avec) ** 2, axis=0))
            rows.append(
                {
                    "generation": generation,
                    "label": str(item["label"]),
                    "is_signal_direction": bool(item["is_signal_direction"]),
                    "beta": float(item["beta"]),
                    "lambda": float(item["lambda"]),
                    "separation_from_signal_deg": float(item["separation_from_signal_deg"]),
                    "normalize_vector": normalize_vector,
                    "projection_norm_median": float(np.median(avec_norm)),
                    "projection_norm_max": float(np.max(avec_norm)),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def plot_scan(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, normalize_vector in zip(axes, (False, True)):
        sub = df[df["normalize_vector"] == normalize_vector]
        for generation, marker in [(1, "o"), (2, "s")]:
            g = sub[sub["generation"] == generation].sort_values("separation_from_signal_deg")
            ax.scatter(
                g["separation_from_signal_deg"],
                g["time_abs_rms"],
                marker=marker,
                s=42,
                label=f"G{generation}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Separation from signal direction [deg]")
        ax.set_ylabel(r"Noisy $|\kappa(t)|$ RMS [Hz]")
        ax.set_title("Vector-normalized" if normalize_vector else "Raw")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Noisy coronagraphic TDI sky scan")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_summary(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = (
        df[(df["generation"] == 1) & (df["normalize_vector"])]
        .sort_values("separation_from_signal_deg")["label"]
        .tolist()
    )
    x = np.arange(len(ordered))
    width = 0.36
    colors = {1: "#2563eb", 2: "#dc2626"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    metrics = [("time_abs_rms", r"Noisy $|\kappa(t)|$ RMS [Hz]"), ("time_abs_max", r"Noisy $|\kappa(t)|$ max [Hz]")]
    for row, (metric, ylabel) in enumerate(metrics):
        for col, normalize_vector in enumerate((False, True)):
            ax = axes[row, col]
            for generation, offset in [(1, -width / 2.0), (2, width / 2.0)]:
                sub = (
                    df[(df["generation"] == generation) & (df["normalize_vector"] == normalize_vector)]
                    .set_index("label")
                    .loc[ordered]
                    .reset_index()
                )
                ax.bar(
                    x + offset,
                    sub[metric],
                    width=width,
                    color=colors[generation],
                    alpha=0.82,
                    label=f"G{generation}",
                )
            ax.scatter([0], [df[(df["label"] == "signal") & (df["generation"] == 1) & (df["normalize_vector"] == normalize_vector)][metric].iloc[0]],
                       marker="*", s=170, color="black", zorder=5)
            ax.scatter([0], [df[(df["label"] == "signal") & (df["generation"] == 2) & (df["normalize_vector"] == normalize_vector)][metric].iloc[0]],
                       marker="*", s=170, color="black", zorder=5)
            ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.set_title("Vector-normalized" if normalize_vector else "Raw")
            ax.grid(True, axis="y", which="both", alpha=0.25)
            ax.legend(loc="best")

    labels = [
        f"{label}\n{df[df['label'] == label]['separation_from_signal_deg'].iloc[0]:.0f} deg"
        for label in ordered
    ]
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Sky point, ordered by separation from signal")
    fig.suptitle("Noisy coronagraphic TDI 10-sky scan")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    positions = ten_sky_positions()
    scans = [scan_generation(generation, positions) for generation in (1, 2)]
    df = pd.concat(scans, ignore_index=True)

    output_dir = WORK_DIR / "outputs"
    plot_dir = output_dir / "plots"
    stats_csv = output_dir / "coronagraphic_noisy_10sky_g1_g2_stats.csv"
    report_json = output_dir / "coronagraphic_noisy_10sky_g1_g2_report.json"
    plot_png = plot_dir / "08_coronagraphic_noisy_10sky_g1_g2_rms.png"
    summary_plot_png = plot_dir / "08_coronagraphic_noisy_10sky_g1_g2_summary.png"

    df.to_csv(stats_csv, index=False)
    plot_scan(df, plot_png)
    plot_summary(df, summary_plot_png)

    summary = {}
    for normalize_vector in (False, True):
        key = "normalized" if normalize_vector else "raw"
        summary[key] = {}
        for generation in (1, 2):
            sub = df[(df["generation"] == generation) & (df["normalize_vector"] == normalize_vector)]
            signal = sub[sub["is_signal_direction"]].iloc[0]
            others = sub[~sub["is_signal_direction"]]
            summary[key][f"G{generation}"] = {
                "signal_time_abs_rms": float(signal["time_abs_rms"]),
                "signal_time_abs_max": float(signal["time_abs_max"]),
                "median_other_time_abs_rms": float(others["time_abs_rms"].median()),
                "max_other_time_abs_rms": float(others["time_abs_rms"].max()),
                "median_other_over_signal_rms": float(others["time_abs_rms"].median() / signal["time_abs_rms"]),
                "max_other_over_signal_rms": float(others["time_abs_rms"].max() / signal["time_abs_rms"]),
            }

    report = {
        "input": "noisy PyTDI Sagnac channels containing GW signal plus realistic LISA noise",
        "sky_points": positions,
        "frequency_band_hz": [LOW_HZ, HIGH_HZ],
        "summary": summary,
        "outputs": {
            "stats_csv": str(stats_csv),
            "rms_plot": str(plot_png),
            "summary_plot": str(summary_plot_png),
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"stats = {stats_csv}")
    print(f"rms_plot = {plot_png}")
    print(f"summary_plot = {summary_plot_png}")


if __name__ == "__main__":
    main()
