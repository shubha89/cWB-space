from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clean_wdm import (
    CleanWDMConfig,
    CleanWDMPaths,
    WDMResolution,
    compute_wdm_transform,
    downsample_to_target_dt,
    load_simulation_channels,
    locate_signal_window,
    require_wdm_dependencies,
)


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
NOISE_ONLY_TDI_CSV = OUTPUT_DIR / "tdi_noise_only_seed12345_XYZ_AET.csv"


RESOLUTION_SETS = {
    "smoke": (
        WDMResolution("dt1_M512", 1.0, 512),
        WDMResolution("dt2_M2048", 2.0, 2048),
    ),
    "dt2_M4096": (
        WDMResolution("dt2_M4096", 2.0, 4096),
    ),
    "pycwb_mra_smoke": (
        WDMResolution("dt2_M512", 2.0, 512),
        WDMResolution("dt2_M1024", 2.0, 1024),
    ),
    "pycwb_mra_full": (
        WDMResolution("dt2_M512", 2.0, 512),
        WDMResolution("dt2_M1024", 2.0, 1024),
        WDMResolution("dt2_M2048", 2.0, 2048),
        WDMResolution("dt2_M4096", 2.0, 4096),
    ),
    "default": (
        WDMResolution("dt1_M512", 1.0, 512),
        WDMResolution("dt1_M1024", 1.0, 1024),
        WDMResolution("dt2_M2048", 2.0, 2048),
        WDMResolution("dt2_M4096", 2.0, 4096),
    ),
}


def load_noise_only_t_channel(paths: CleanWDMPaths, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    if not NOISE_ONLY_TDI_CSV.exists():
        raise FileNotFoundError(f"Missing noise-only TDI CSV: {NOISE_ONLY_TDI_CSV}")
    frame = pd.read_csv(NOISE_ONLY_TDI_CSV, usecols=["t_lisa", f"T{suffix}"])
    t_noise = frame["t_lisa"].to_numpy(dtype=float)
    if not np.allclose(pd.read_csv(paths.noisy_tdi_csv, usecols=["t_lisa"])["t_lisa"].to_numpy(dtype=float), t_noise, rtol=0.0, atol=1e-6):
        raise ValueError("Noise-only TDI is not aligned to the noisy TDI time grid.")
    return t_noise, frame[f"T{suffix}"].to_numpy(dtype=float)


def signal_masks(times_lisa: np.ndarray, signal: dict[str, float], guard_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    signal_t = (
        (times_lisa >= signal["signal_window_start_lisa"])
        & (times_lisa <= signal["signal_window_stop_lisa"])
    )
    offsource_t = (
        (times_lisa < signal["signal_window_start_lisa"] - guard_seconds)
        | (times_lisa > signal["signal_window_stop_lisa"] + guard_seconds)
    )
    return signal_t, offsource_t


def summarize_whitened(
    *,
    source: str,
    channel: str,
    resolution: WDMResolution,
    whitened: np.ndarray,
    t_floor: np.ndarray,
    times_lisa: np.ndarray,
    freqs_hz: np.ndarray,
    signal: dict[str, float],
    config: CleanWDMConfig,
) -> dict[str, float | int | str]:
    power = np.abs(whitened) ** 2
    signal_t, offsource_t = signal_masks(times_lisa, signal, config.offsource_guard_seconds)
    band_f = (freqs_hz >= config.low_freq_hz) & (freqs_hz <= config.high_freq_hz)
    signal_band = power[np.ix_(band_f, signal_t)]
    offsource_band = power[np.ix_(band_f, offsource_t)]
    band_power = power[band_f, :]
    band_freqs = freqs_hz[band_f]
    peak_idx = np.unravel_index(int(np.nanargmax(band_power)), band_power.shape) if band_power.size else (0, 0)
    return {
        "source": source,
        "channel": channel,
        "resolution": resolution.label,
        "target_dt_s": float(resolution.target_dt),
        "wdm_M": int(resolution.M),
        "n_time_bins": int(len(times_lisa)),
        "n_freq_bins": int(len(freqs_hz)),
        "n_band_freq_bins": int(np.sum(band_f)),
        "n_signal_time_bins": int(np.sum(signal_t)),
        "t_floor_median": float(np.nanmedian(t_floor[band_f])) if np.any(band_f) else np.nan,
        "t_floor_min": float(np.nanmin(t_floor[band_f])) if np.any(band_f) else np.nan,
        "t_floor_max": float(np.nanmax(t_floor[band_f])) if np.any(band_f) else np.nan,
        "signal_twhitened_power_mean": float(np.nanmean(signal_band)) if signal_band.size else np.nan,
        "signal_twhitened_power_median": float(np.nanmedian(signal_band)) if signal_band.size else np.nan,
        "signal_twhitened_power_max": float(np.nanmax(signal_band)) if signal_band.size else np.nan,
        "offsource_twhitened_power_mean": float(np.nanmean(offsource_band)) if offsource_band.size else np.nan,
        "offsource_twhitened_power_median": float(np.nanmedian(offsource_band)) if offsource_band.size else np.nan,
        "signal_over_offsource_mean": (
            float(np.nanmean(signal_band) / np.nanmean(offsource_band))
            if signal_band.size and offsource_band.size
            else np.nan
        ),
        "band_peak_time_rel": float(times_lisa[peak_idx[1]] - times_lisa[0]) if band_power.size else np.nan,
        "band_peak_freq_hz": float(band_freqs[peak_idx[0]]) if band_power.size else np.nan,
        "band_peak_twhitened_power": float(np.nanmax(band_power)) if band_power.size else np.nan,
    }


def plot_twhitened(npz_path: Path, stats: pd.DataFrame, output: Path, *, source: str, resolution: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    maps = np.load(npz_path)
    signal = np.asarray(maps["signal"], dtype=float)
    signal_start_days = signal[4] / 86400.0
    signal_stop_days = signal[5] / 86400.0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ax, channel in zip(axes, ("A", "E")):
        prefix = f"{source}_{channel}_{resolution}"
        power = np.asarray(maps[f"{prefix}_twhitened_power"], dtype=float)
        times = np.asarray(maps[f"{prefix}_times_rel"], dtype=float) / 86400.0
        freqs = np.asarray(maps[f"{prefix}_freqs_hz"], dtype=float)
        band = (freqs >= 1e-5) & (freqs <= 1e-1)
        values = np.log10(np.maximum(power[band, :], 1e-300))
        image = ax.pcolormesh(times, freqs[band], values, shading="auto", cmap="magma")
        ax.set_yscale("log")
        ax.set_title(f"T-whitened {source} {channel}")
        ax.set_xlabel("Time [days]")
        ax.axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.16, lw=0)
        row = stats[(stats["source"] == source) & (stats["channel"] == channel) & (stats["resolution"] == resolution)]
        if len(row):
            ax.scatter(
                float(row["band_peak_time_rel"].iloc[0]) / 86400.0,
                float(row["band_peak_freq_hz"].iloc[0]),
                marker="*",
                s=110,
                color="cyan",
            )
        ax.grid(True, which="both", alpha=0.15)
    axes[0].set_ylabel("Frequency [Hz]")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.84, pad=0.02)
        cbar.set_label(r"$\log_{10}(|W|^2 / P_T(f))$")
    fig.suptitle(f"T-whitened WDM A/E: {source}, {resolution}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Compute T-whitened WDM A/E coefficients for MBHB diagnostic data.")
    parser.add_argument("--resolution-set", choices=tuple(RESOLUTION_SETS), default="smoke")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    prefix = args.prefix or f"twhitened_wdm_{args.resolution_set}_aet"
    config = CleanWDMConfig(resolutions=RESOLUTION_SETS[args.resolution_set], groups=("AET",), sources=("noisy", "gw_only"))
    resample_poly, WDM = require_wdm_dependencies()
    paths = CleanWDMPaths.from_work_dir(WORK_DIR, prefix=prefix)
    paths.make_dirs()

    t_lisa, data = load_simulation_channels(paths, tdi_suffix=config.tdi_suffix)
    _, t_noise = load_noise_only_t_channel(paths, config.tdi_suffix)
    native_dt = float(np.median(np.diff(t_lisa)))
    signal = locate_signal_window(t_lisa, data["gw_only"]["AET"], threshold_fraction=config.signal_threshold_fraction)

    rows: list[dict[str, float | int | str]] = []
    payload: dict[str, np.ndarray] = {
        "signal": np.asarray(
            [
                signal["signal_peak_lisa"],
                signal["signal_peak_rel"],
                signal["signal_window_start_lisa"],
                signal["signal_window_stop_lisa"],
                signal["signal_window_start_rel"],
                signal["signal_window_stop_rel"],
            ],
            dtype=float,
        ),
        "native_t_lisa_start": np.asarray(float(t_lisa[0])),
        "native_dt_s": np.asarray(native_dt),
    }

    for resolution in config.resolutions:
        wdm = WDM(
            M=resolution.M,
            K=config.wdm_k,
            beta_order=config.beta_order,
            precision=config.precision,
            backend=config.backend,
        )
        t_noise_ds, dt_ds, _ = downsample_to_target_dt(t_noise, native_dt, resolution.target_dt, resample_poly)
        t_coeff, times_wdm, freqs_wdm, _, _ = compute_wdm_transform(
            wdm,
            t_noise_ds,
            sample_rate=1.0 / dt_ds,
            t0=float(t_lisa[0]),
            original_t1=float(t_lisa[-1]),
        )
        t_floor = np.nanmedian(np.abs(t_coeff) ** 2, axis=1)
        t_floor = np.where(np.isfinite(t_floor) & (t_floor > 0.0), t_floor, config.min_noise_floor)
        payload[f"T_noise_{resolution.label}_floor"] = t_floor.astype(np.float32)
        payload[f"T_noise_{resolution.label}_times_rel"] = (times_wdm - t_lisa[0]).astype(np.float64)
        payload[f"T_noise_{resolution.label}_freqs_hz"] = freqs_wdm.astype(np.float64)

        for source in ("noisy", "gw_only"):
            for channel in ("A", "E"):
                x_ds, x_dt_ds, _ = downsample_to_target_dt(
                    data[source]["AET"][channel],
                    native_dt,
                    resolution.target_dt,
                    resample_poly,
                )
                if not np.isclose(x_dt_ds, dt_ds, rtol=0.0, atol=1e-12):
                    raise RuntimeError("Internal downsampled dt mismatch.")
                coeff, x_times, x_freqs, _, _ = compute_wdm_transform(
                    wdm,
                    x_ds,
                    sample_rate=1.0 / x_dt_ds,
                    t0=float(t_lisa[0]),
                    original_t1=float(t_lisa[-1]),
                )
                if coeff.shape != t_coeff.shape or not np.allclose(x_times, times_wdm) or not np.allclose(x_freqs, freqs_wdm):
                    raise ValueError("T and A/E WDM maps are not aligned.")
                whitened = coeff / np.sqrt(t_floor[:, None])
                power = np.abs(whitened) ** 2
                rows.append(
                    summarize_whitened(
                        source=source,
                        channel=channel,
                        resolution=resolution,
                        whitened=whitened,
                        t_floor=t_floor,
                        times_lisa=times_wdm,
                        freqs_hz=freqs_wdm,
                        signal=signal,
                        config=config,
                    )
                )
                key = f"{source}_{channel}_{resolution.label}"
                payload[f"{key}_twhitened_real"] = np.real(whitened).astype(np.float32)
                payload[f"{key}_twhitened_imag"] = np.imag(whitened).astype(np.float32)
                payload[f"{key}_twhitened_power"] = power.astype(np.float32)
                payload[f"{key}_times_rel"] = (times_wdm - t_lisa[0]).astype(np.float64)
                payload[f"{key}_freqs_hz"] = freqs_wdm.astype(np.float64)

    stats = pd.DataFrame(rows)
    stats.to_csv(paths.stats_csv, index=False)
    np.savez_compressed(paths.maps_npz, **payload)

    plots = {}
    if not args.no_plot:
        resolution = config.resolutions[0].label
        for source in ("noisy", "gw_only"):
            plot_path = paths.plot_dir / f"{prefix}_{source}_{resolution}.png"
            plot_twhitened(paths.maps_npz, stats, plot_path, source=source, resolution=resolution)
            plots[source] = str(plot_path)

    report = {
        "method": "complex WDM A/E coefficients divided by sqrt(median_t |WDM(T_noise_only)|^2) per frequency",
        "noise_reference": str(NOISE_ONLY_TDI_CSV),
        "config": {
            "resolution_set": args.resolution_set,
            "resolutions": [
                {"label": item.label, "target_dt": item.target_dt, "M": item.M}
                for item in config.resolutions
            ],
            "low_freq_hz": config.low_freq_hz,
            "high_freq_hz": config.high_freq_hz,
        },
        "diagnostics": {
            "native_samples": int(len(t_lisa)),
            "native_dt_s": native_dt,
            **signal,
            "rows": int(len(stats)),
        },
        "outputs": {
            "stats_csv": str(paths.stats_csv),
            "maps_npz": str(paths.maps_npz),
            "plots": plots,
        },
    }
    report_path = paths.output_dir / f"{prefix}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report = {report_path}")


if __name__ == "__main__":
    main()
