from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
LOW_HZ = 1e-5
HIGH_HZ = 1e-1


def smooth_real(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return uniform_filter(np.asarray(values, dtype=float), size=size, mode="nearest")


def smooth_complex(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return smooth_real(np.real(values), size) + 1j * smooth_real(np.imag(values), size)


def local_ae_coherence(
    a: np.ndarray,
    e: np.ndarray,
    *,
    smoothing: tuple[int, int],
    epsilon: float = 1e-300,
) -> dict[str, np.ndarray]:
    """Estimate pixel-wise local A/E coherence from complex WDM coefficients.

    The local smoothing is needed because a single complex pixel has trivial
    magnitude coherence when normalized by its own amplitudes.
    """

    a_power = np.abs(a) ** 2
    e_power = np.abs(e) ** 2
    cross = a * np.conjugate(e)
    saa = smooth_real(a_power, smoothing)
    see = smooth_real(e_power, smoothing)
    sae = smooth_complex(cross, smoothing)
    denom = np.sqrt(np.maximum(saa * see, epsilon))
    coherence_abs = np.clip(np.abs(sae) / denom, 0.0, 1.0)
    phase_correlation = np.clip(np.real(sae) / denom, -1.0, 1.0)
    return {
        "a_power": a_power,
        "e_power": e_power,
        "energy": a_power + e_power,
        "cross_real": np.real(cross),
        "cross_imag": np.imag(cross),
        "coherence_abs": coherence_abs,
        "phase_correlation": phase_correlation,
        "coherent_excess_power": (a_power + e_power) * coherence_abs,
        "positive_phase_excess_power": (a_power + e_power) * np.maximum(phase_correlation, 0.0),
        "signed_phase_excess_power": (a_power + e_power) * phase_correlation,
    }


def load_complex_pair(maps: np.lib.npyio.NpzFile, *, source: str, resolution: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix_a = f"{source}_A_{resolution}"
    prefix_e = f"{source}_E_{resolution}"
    a = maps[f"{prefix_a}_twhitened_real"].astype(float) + 1j * maps[f"{prefix_a}_twhitened_imag"].astype(float)
    e = maps[f"{prefix_e}_twhitened_real"].astype(float) + 1j * maps[f"{prefix_e}_twhitened_imag"].astype(float)
    times = maps[f"{prefix_a}_times_rel"].astype(float)
    freqs = maps[f"{prefix_a}_freqs_hz"].astype(float)
    if a.shape != e.shape:
        raise ValueError("A and E T-whitened WDM maps have different shapes.")
    if not np.array_equal(times, maps[f"{prefix_e}_times_rel"]) or not np.array_equal(freqs, maps[f"{prefix_e}_freqs_hz"]):
        raise ValueError("A and E T-whitened WDM maps are not aligned.")
    return a, e, times, freqs


def time_bin_overlap_mask(times_rel: np.ndarray, start_rel: float, stop_rel: float) -> np.ndarray:
    if len(times_rel) == 0:
        return np.zeros(0, dtype=bool)
    if len(times_rel) == 1:
        dt = max(stop_rel - start_rel, 0.0)
    else:
        dt = float(np.median(np.diff(times_rel)))
    half_dt = 0.5 * dt
    return (times_rel + half_dt >= start_rel) & (times_rel - half_dt <= stop_rel)


def summarize_map(
    *,
    source: str,
    resolution: str,
    products: dict[str, np.ndarray],
    times_rel: np.ndarray,
    freqs_hz: np.ndarray,
    signal: np.ndarray,
) -> dict[str, float | int | str]:
    signal_start_rel = float(signal[4])
    signal_stop_rel = float(signal[5])
    signal_t = time_bin_overlap_mask(times_rel, signal_start_rel, signal_stop_rel)
    offsource_t = ~time_bin_overlap_mask(times_rel, signal_start_rel - 500.0, signal_stop_rel + 500.0)
    band_f = (freqs_hz >= LOW_HZ) & (freqs_hz <= HIGH_HZ)
    rows: dict[str, float | int | str] = {
        "source": source,
        "resolution": resolution,
        "n_band_freq_bins": int(np.sum(band_f)),
        "n_signal_time_bins": int(np.sum(signal_t)),
        "n_offsource_time_bins": int(np.sum(offsource_t)),
    }
    for name in ("energy", "coherence_abs", "phase_correlation", "coherent_excess_power", "positive_phase_excess_power"):
        arr = products[name]
        signal_band = arr[np.ix_(band_f, signal_t)]
        offsource_band = arr[np.ix_(band_f, offsource_t)]
        full_band = arr[band_f, :]
        max_idx = np.unravel_index(int(np.nanargmax(full_band)), full_band.shape) if full_band.size else (0, 0)
        band_freqs = freqs_hz[band_f]
        rows[f"{name}_signal_mean"] = float(np.nanmean(signal_band)) if signal_band.size else np.nan
        rows[f"{name}_signal_median"] = float(np.nanmedian(signal_band)) if signal_band.size else np.nan
        rows[f"{name}_signal_max"] = float(np.nanmax(signal_band)) if signal_band.size else np.nan
        rows[f"{name}_offsource_mean"] = float(np.nanmean(offsource_band)) if offsource_band.size else np.nan
        rows[f"{name}_offsource_median"] = float(np.nanmedian(offsource_band)) if offsource_band.size else np.nan
        rows[f"{name}_signal_over_offsource_mean"] = (
            float(np.nanmean(signal_band) / np.nanmean(offsource_band))
            if signal_band.size and offsource_band.size and np.nanmean(offsource_band) != 0.0
            else np.nan
        )
        rows[f"{name}_peak_time_rel"] = float(times_rel[max_idx[1]]) if full_band.size else np.nan
        rows[f"{name}_peak_freq_hz"] = float(band_freqs[max_idx[0]]) if full_band.size else np.nan
    return rows


def plot_products(
    output: Path,
    *,
    source: str,
    resolution: str,
    products: dict[str, np.ndarray],
    times_rel: np.ndarray,
    freqs_hz: np.ndarray,
    signal: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    band_f = (freqs_hz >= LOW_HZ) & (freqs_hz <= HIGH_HZ)
    times_days = times_rel / 86400.0
    signal_start_days = float(signal[4]) / 86400.0
    signal_stop_days = float(signal[5]) / 86400.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        ("energy", r"$\log_{10}(|A_T|^2 + |E_T|^2)$"),
        ("coherence_abs", r"$|C_{AE}|$"),
        ("coherent_excess_power", r"$\log_{10}(E_{AE}|C_{AE}|)$"),
    ]
    images = []
    for ax, (name, label) in zip(axes, panels):
        values = products[name][band_f, :]
        if name == "coherence_abs":
            shown = values
            vmin, vmax = 0.0, 1.0
        else:
            shown = np.log10(np.maximum(values, 1e-300))
            vmin, vmax = None, None
        image = ax.pcolormesh(times_days, freqs_hz[band_f], shown, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        images.append((image, label))
        ax.set_yscale("log")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Time [days]")
        ax.axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.16, lw=0)
        ax.grid(True, which="both", alpha=0.15)
    axes[0].set_ylabel("Frequency [Hz]")
    for image, label in images:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.015)
        cbar.set_label(label)
    fig.suptitle(f"A/E T-whitened coherent excess power: {source}, {resolution}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")
    flat = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(flat)
    if not np.any(finite):
        return np.zeros_like(values, dtype=bool)
    threshold = np.quantile(flat[finite], 1.0 - fraction)
    return np.asarray(values, dtype=float) >= threshold


def plot_top_pixels_comparison(
    output: Path,
    *,
    source: str,
    resolution: str,
    products: dict[str, np.ndarray],
    times_rel: np.ndarray,
    freqs_hz: np.ndarray,
    signal: np.ndarray,
    top_fraction: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    band_f = (freqs_hz >= LOW_HZ) & (freqs_hz <= HIGH_HZ)
    times_days = times_rel / 86400.0
    signal_start_days = float(signal[4]) / 86400.0
    signal_stop_days = float(signal[5]) / 86400.0

    before = products["energy"][band_f, :]
    after = products["coherent_excess_power"][band_f, :]
    before_masked = np.where(top_fraction_mask(before, top_fraction), before, np.nan)
    after_masked = np.where(top_fraction_mask(after, top_fraction), after, np.nan)
    top_label = f"top {100.0 * top_fraction:.0f}%"

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        (before_masked, rf"{top_label} square-sum $|A_T|^2+|E_T|^2$"),
        (after_masked, rf"{top_label} coherence-weighted excess"),
    ]
    for ax, (values, title) in zip(axes, panels):
        image = ax.pcolormesh(
            times_days,
            freqs_hz[band_f],
            np.log10(np.maximum(values, 1e-300)),
            shading="auto",
            cmap="magma",
        )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Time [days]")
        ax.axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.16, lw=0)
        ax.grid(True, which="both", alpha=0.15)
    axes[0].set_ylabel("Frequency [Hz]")
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.84, pad=0.02)
    cbar.set_label(r"$\log_{10}$ retained pixel value")
    fig.suptitle(f"Brightest {100.0 * top_fraction:.0f}% WDM pixels before/after coherence reweighting: {source}, {resolution}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Compute A/E T-whitened coherent excess-power WDM maps.")
    parser.add_argument("--input-prefix", default="twhitened_wdm_smoke_aet")
    parser.add_argument("--resolution", default="dt1_M512")
    parser.add_argument("--smooth-freq", type=int, default=3)
    parser.add_argument("--smooth-time", type=int, default=3)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    input_npz = OUTPUT_DIR / f"{args.input_prefix}_maps.npz"
    if not input_npz.exists():
        raise FileNotFoundError(f"Missing T-whitened WDM map file: {input_npz}")

    smoothing = (int(args.smooth_freq), int(args.smooth_time))
    maps = np.load(input_npz)
    signal = np.asarray(maps["signal"], dtype=float)
    payload: dict[str, np.ndarray] = {
        "signal": signal,
        "smoothing": np.asarray(smoothing, dtype=int),
    }
    rows = []
    plots = {}
    for source in ("noisy", "gw_only"):
        a, e, times_rel, freqs_hz = load_complex_pair(maps, source=source, resolution=args.resolution)
        products = local_ae_coherence(a, e, smoothing=smoothing)
        rows.append(
            summarize_map(
                source=source,
                resolution=args.resolution,
                products=products,
                times_rel=times_rel,
                freqs_hz=freqs_hz,
                signal=signal,
            )
        )
        prefix = f"{source}_{args.resolution}"
        payload[f"{prefix}_times_rel"] = times_rel.astype(np.float64)
        payload[f"{prefix}_freqs_hz"] = freqs_hz.astype(np.float64)
        for name, arr in products.items():
            payload[f"{prefix}_{name}"] = arr.astype(np.float32)
        if not args.no_plot:
            plot_path = PLOT_DIR / f"10_wdm_coherent_excess_{source}_{args.resolution}.png"
            top_plot_path = PLOT_DIR / f"11_wdm_top{int(100 * args.top_fraction):02d}_before_after_excess_{source}_{args.resolution}.png"
            plot_products(
                plot_path,
                source=source,
                resolution=args.resolution,
                products=products,
                times_rel=times_rel,
                freqs_hz=freqs_hz,
                signal=signal,
            )
            plot_top_pixels_comparison(
                top_plot_path,
                source=source,
                resolution=args.resolution,
                products=products,
                times_rel=times_rel,
                freqs_hz=freqs_hz,
                signal=signal,
                top_fraction=float(args.top_fraction),
            )
            plots[source] = str(plot_path)
            plots[f"{source}_top_pixels"] = str(top_plot_path)

    tag = f"{args.input_prefix}_{args.resolution}_smooth{args.smooth_freq}x{args.smooth_time}"
    stats = pd.DataFrame(rows)
    stats_csv = OUTPUT_DIR / f"wdm_coherent_excess_{tag}_stats.csv"
    maps_npz = OUTPUT_DIR / f"wdm_coherent_excess_{tag}_maps.npz"
    report_json = OUTPUT_DIR / f"wdm_coherent_excess_{tag}_report.json"
    stats.to_csv(stats_csv, index=False)
    np.savez_compressed(maps_npz, **payload)

    report = {
        "method": (
            "T-whitened A/E pixel energy reweighted by local A/E coherence. "
            "Local coherence is smoothed A E* divided by smoothed auto powers."
        ),
        "input_npz": str(input_npz),
        "resolution": args.resolution,
        "smoothing": {"frequency_bins": smoothing[0], "time_bins": smoothing[1]},
        "top_fraction": float(args.top_fraction),
        "frequency_band_hz": [LOW_HZ, HIGH_HZ],
        "outputs": {
            "stats_csv": str(stats_csv),
            "maps_npz": str(maps_npz),
            "plots": plots,
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(stats.to_string(index=False))
    print(f"report = {report_json}")


if __name__ == "__main__":
    main()
