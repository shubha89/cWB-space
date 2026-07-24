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
    coronagraphic_vector,
    geometry_from_orbit_file,
    load_pytdi_sagnac_csv,
    sky_to_unit_source_vector,
)
from scripts.tdi_computation import compute_sagnac_tdi_from_instrument


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
NOISE_ONLY_INSTRUMENT = OUTPUT_DIR / "lisa_instrument_noise_only_seed12345.h5"
LOW_HZ = 1e-5
HIGH_HZ = 1e-1
SIGNAL_BETA = 0.3
SIGNAL_LAMBDA = np.pi / 4.0
SMOOTH_BINS = 401
EPSILON = 1e-30


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


def noise_only_sagnac_path(generation: int) -> Path:
    return OUTPUT_DIR / f"tdi_sagnac_g{generation}_noise_only.csv"


def ensure_noise_only_sagnac(generation: int) -> dict[str, object]:
    if not NOISE_ONLY_INSTRUMENT.exists():
        raise FileNotFoundError(f"Missing noise-only instrument HDF5: {NOISE_ONLY_INSTRUMENT}")

    out = noise_only_sagnac_path(generation)
    if out.exists():
        return {"reused": str(out)}

    config = mbhb_config(generation)
    base = sagnac_paths_for_generation(LisaLinkSimulationPaths.from_work_dir(WORK_DIR), generation)
    paths = replace(
        base,
        instrument_h5=NOISE_ONLY_INSTRUMENT,
        noisy_sagnac_tdi_csv=out,
        noisy_tdi_sagnac_plot=PLOT_DIR / f"04_tdi_sagnac_g{generation}_noise_only.png",
    )
    diag = compute_sagnac_tdi_from_instrument(config, paths, make_plots=False)
    return {"generated": str(out), "diagnostics": diag}


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


def smooth_complex(values: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 1:
        return values
    if bins % 2 == 0:
        bins += 1
    kernel = np.ones(bins, dtype=float) / float(bins)
    real = np.convolve(np.real(values), kernel, mode="same")
    imag = np.convolve(np.imag(values), kernel, mode="same")
    return real + 1j * imag


def estimate_noise_covariance(noise_f: np.ndarray, smooth_bins: int) -> np.ndarray:
    n_channels, n_freqs = noise_f.shape
    cov = np.empty((n_channels, n_channels, n_freqs), dtype=np.complex128)
    for i in range(n_channels):
        for j in range(n_channels):
            cross = noise_f[i] * np.conjugate(noise_f[j])
            cov[i, j] = smooth_complex(cross, smooth_bins)
    for idx in range(n_freqs):
        mat = 0.5 * (cov[:, :, idx] + cov[:, :, idx].conjugate().T)
        diag_floor = EPSILON * max(1.0, float(np.max(np.real(np.diag(mat)))))
        mat = mat + diag_floor * np.eye(n_channels)
        cov[:, :, idx] = mat
    return cov


def scan_generation(
    generation: int,
    positions: list[dict[str, float | str | bool]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    noise_sagnac_report = ensure_noise_only_sagnac(generation)
    base_paths = sagnac_paths_for_generation(LisaLinkSimulationPaths.from_work_dir(WORK_DIR), generation)
    if not base_paths.noisy_sagnac_tdi_csv.exists():
        raise FileNotFoundError(f"Missing noisy Sagnac CSV for G{generation}: {base_paths.noisy_sagnac_tdi_csv}")

    t_noisy, lisa_noisy, noisy = load_pytdi_sagnac_csv(base_paths.noisy_sagnac_tdi_csv, generation=generation)
    t_noise, lisa_noise, noise = load_pytdi_sagnac_csv(noise_only_sagnac_path(generation), generation=generation)
    if not (np.array_equal(t_noisy, t_noise) and np.array_equal(lisa_noisy, lisa_noise)):
        raise ValueError(f"Noisy and noise-only G{generation} Sagnac products do not share the same time grid.")

    dt = float(np.median(np.diff(t_noisy)))
    freqs = np.fft.fftfreq(len(t_noisy), d=dt)
    band = (freqs > 0.0) & (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    freqs_band = freqs[band]

    noisy_f = np.vstack([time_to_frequency(noisy[name])[band] for name in ("alpha", "beta", "gamma")])
    noise_f = np.vstack([time_to_frequency(noise[name])[band] for name in ("alpha", "beta", "gamma")])
    cov = estimate_noise_covariance(noise_f, SMOOTH_BINS)

    geometry = geometry_from_orbit_file(base_paths.orbits_h5, t_lisa_reference=signal_reference_time(base_paths, lisa_noisy))
    rows: list[dict[str, float | str | int | bool]] = []
    for item in positions:
        avec = coronagraphic_vector(
            freqs_band,
            float(item["beta"]),
            float(item["lambda"]),
            geometry,
            normalize=False,
        )
        kappa = np.sum(avec * noisy_f, axis=0)
        sigma2 = np.einsum("if,ijf,jf->f", avec, cov, np.conjugate(avec))
        sigma = np.sqrt(np.maximum(np.real(sigma2), EPSILON))
        z = kappa / sigma
        rows.append(
            {
                "generation": generation,
                "label": str(item["label"]),
                "is_signal_direction": bool(item["is_signal_direction"]),
                "beta": float(item["beta"]),
                "lambda": float(item["lambda"]),
                "separation_from_signal_deg": float(item["separation_from_signal_deg"]),
                "raw_freq_band_abs_rms": float(np.sqrt(np.mean(np.abs(kappa) ** 2))),
                "raw_freq_band_abs_max": float(np.max(np.abs(kappa))),
                "noise_sigma_median": float(np.median(sigma)),
                "noise_sigma_min": float(np.min(sigma)),
                "noise_sigma_max": float(np.max(sigma)),
                "z_abs_rms": float(np.sqrt(np.mean(np.abs(z) ** 2))),
                "z_abs_median": float(np.median(np.abs(z))),
                "z_abs_max": float(np.max(np.abs(z))),
            }
        )
    meta = {
        "generation": generation,
        "noise_sagnac": noise_sagnac_report,
        "noisy_sagnac": str(base_paths.noisy_sagnac_tdi_csv),
        "noise_only_sagnac": str(noise_only_sagnac_path(generation)),
        "frequency_bins": int(len(freqs_band)),
        "frequency_start_hz": float(freqs_band[0]),
        "frequency_stop_hz": float(freqs_band[-1]),
        "smooth_bins": SMOOTH_BINS,
    }
    return pd.DataFrame(rows), meta


def plot_normalized_scan(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = (
        df[df["generation"] == 1]
        .sort_values("separation_from_signal_deg")["label"]
        .tolist()
    )
    labels = [
        f"{label}\n{df[df['label'] == label]['separation_from_signal_deg'].iloc[0]:.0f} deg"
        for label in ordered
    ]
    x = np.arange(len(ordered))
    width = 0.36
    colors = {1: "#2563eb", 2: "#dc2626"}

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "z_abs_rms", r"Noise-normalized $|\kappa|$ RMS"),
        (axes[1], "z_abs_max", r"Noise-normalized max $|\kappa|$"),
    ]:
        for generation, offset in [(1, -width / 2.0), (2, width / 2.0)]:
            sub = df[df["generation"] == generation].set_index("label").loc[ordered].reset_index()
            ax.bar(
                x + offset,
                sub[metric],
                width=width,
                color=colors[generation],
                alpha=0.84,
                label=f"G{generation}",
            )
        for generation in (1, 2):
            signal = df[(df["generation"] == generation) & (df["label"] == "signal")].iloc[0]
            ax.scatter([0], [signal[metric]], marker="*", s=180, color="black", zorder=5)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Sky point, ordered by separation from signal")
    fig.suptitle("Noise-normalized coronagraphic TDI sky scan")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    positions = ten_sky_positions()
    generation_results = [scan_generation(generation, positions) for generation in (1, 2)]
    df = pd.concat([item[0] for item in generation_results], ignore_index=True)
    metadata = {f"G{item[1]['generation']}": item[1] for item in generation_results}

    stats_csv = OUTPUT_DIR / "coronagraphic_noise_normalized_10sky_g1_g2_stats.csv"
    report_json = OUTPUT_DIR / "coronagraphic_noise_normalized_10sky_g1_g2_report.json"
    plot_png = PLOT_DIR / "09_coronagraphic_noise_normalized_10sky_g1_g2.png"
    df.to_csv(stats_csv, index=False)
    plot_normalized_scan(df, plot_png)

    summary = {}
    for generation in (1, 2):
        sub = df[df["generation"] == generation]
        signal = sub[sub["is_signal_direction"]].iloc[0]
        others = sub[~sub["is_signal_direction"]]
        summary[f"G{generation}"] = {
            "signal_z_abs_rms": float(signal["z_abs_rms"]),
            "signal_z_abs_max": float(signal["z_abs_max"]),
            "median_other_z_abs_rms": float(others["z_abs_rms"].median()),
            "max_other_z_abs_rms": float(others["z_abs_rms"].max()),
            "median_other_over_signal_z_rms": float(others["z_abs_rms"].median() / signal["z_abs_rms"]),
            "max_other_over_signal_z_rms": float(others["z_abs_rms"].max() / signal["z_abs_rms"]),
        }

    report = {
        "method": "kappa(f, sky) divided by sqrt(A(f, sky) C_noise(f) A(f, sky)^dagger)",
        "input": "noisy PyTDI Sagnac channels containing GW signal plus realistic LISA noise",
        "noise_covariance_source": str(NOISE_ONLY_INSTRUMENT),
        "frequency_band_hz": [LOW_HZ, HIGH_HZ],
        "smooth_bins": SMOOTH_BINS,
        "sky_points": positions,
        "metadata": metadata,
        "summary": summary,
        "outputs": {
            "stats_csv": str(stats_csv),
            "plot": str(plot_png),
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"stats = {stats_csv}")
    print(f"plot = {plot_png}")


if __name__ == "__main__":
    main()
