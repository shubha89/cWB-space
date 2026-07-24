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

from diagnostics.run_mbhb_t_whitened_wdm import load_noise_only_t_channel
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
LOW_HZ = 1e-5
HIGH_HZ = 1e-1


def plot_source_tf_maps(
    output: Path,
    *,
    maps: dict[str, np.ndarray],
    source: str,
    resolution: str,
    signal: np.ndarray,
    show_signal: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        ("A", "raw_power", r"$\log_{10}|W_A|^2$"),
        ("A", "twhitened_power", r"$\log_{10}(|W_A|^2/P_T)$"),
        ("E", "raw_power", r"$\log_{10}|W_E|^2$"),
        ("E", "twhitened_power", r"$\log_{10}(|W_E|^2/P_T)$"),
    ]
    panel_values = {}
    for channel, kind, _ in panels:
        prefix = f"{source}_{channel}_{resolution}"
        power = np.asarray(maps[f"{prefix}_{kind}"], dtype=float)
        times = np.asarray(maps[f"{prefix}_times_rel"], dtype=float) / 86400.0
        freqs = np.asarray(maps[f"{prefix}_freqs_hz"], dtype=float)
        band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
        panel_values[channel, kind] = (
            times,
            freqs[band],
            np.log10(np.maximum(power[band, :], 1e-300)),
        )
    raw_values = np.concatenate([panel_values[channel, "raw_power"][2].ravel() for channel in ("A", "E")])
    white_values = np.concatenate([panel_values[channel, "twhitened_power"][2].ravel() for channel in ("A", "E")])
    raw_vmin, raw_vmax = np.nanpercentile(raw_values[np.isfinite(raw_values)], [1.0, 99.7])
    white_vmin, white_vmax = np.nanpercentile(white_values[np.isfinite(white_values)], [1.0, 99.7])

    signal_start_days = float(signal[4]) / 86400.0
    signal_stop_days = float(signal[5]) / 86400.0
    images = {}
    for ax, (channel, kind, label) in zip(axes.ravel(), panels):
        times, freqs, values = panel_values[channel, kind]
        vmin, vmax = (raw_vmin, raw_vmax) if kind == "raw_power" else (white_vmin, white_vmax)
        image = ax.pcolormesh(times, freqs, values, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        images[kind] = (image, label)
        ax.set_yscale("log")
        ax.set_title(f"{channel} {'T-whitened' if kind == 'twhitened_power' else 'raw'}")
        ax.grid(True, which="both", alpha=0.14)
        if show_signal:
            ax.axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.13, lw=0)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time [days]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency [Hz]")
    raw_image, raw_label = images["raw_power"]
    raw_cbar = fig.colorbar(raw_image, ax=axes[:, 0].tolist(), shrink=0.82, pad=0.012)
    raw_cbar.set_label(raw_label.replace("_A", "").replace("_E", ""))
    white_image, white_label = images["twhitened_power"]
    white_cbar = fig.colorbar(white_image, ax=axes[:, 1].tolist(), shrink=0.82, pad=0.012)
    white_cbar.set_label(white_label.replace("_A", "").replace("_E", ""))
    fig.suptitle(f"A/E WDM time-frequency maps: {source}, {resolution}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize_map(
    *,
    source: str,
    channel: str,
    resolution: WDMResolution,
    raw_power: np.ndarray,
    twhitened_power: np.ndarray,
    times_rel: np.ndarray,
    freqs_hz: np.ndarray,
    signal: dict[str, float],
) -> dict[str, float | int | str]:
    band = (freqs_hz >= LOW_HZ) & (freqs_hz <= HIGH_HZ)
    signal_t = (times_rel >= signal["signal_window_start_rel"]) & (times_rel <= signal["signal_window_stop_rel"])
    row: dict[str, float | int | str] = {
        "source": source,
        "channel": channel,
        "resolution": resolution.label,
        "target_dt_s": float(resolution.target_dt),
        "wdm_M": int(resolution.M),
        "n_time_bins": int(len(times_rel)),
        "n_freq_bins": int(len(freqs_hz)),
        "n_band_freq_bins": int(np.sum(band)),
        "n_signal_time_bins": int(np.sum(signal_t)),
    }
    for name, values in (("raw_power", raw_power), ("twhitened_power", twhitened_power)):
        band_values = values[band, :]
        signal_values = values[np.ix_(band, signal_t)]
        peak_idx = np.unravel_index(int(np.nanargmax(band_values)), band_values.shape)
        band_freqs = freqs_hz[band]
        row[f"{name}_band_median"] = float(np.nanmedian(band_values))
        row[f"{name}_band_max"] = float(np.nanmax(band_values))
        row[f"{name}_signal_median"] = float(np.nanmedian(signal_values)) if signal_values.size else np.nan
        row[f"{name}_signal_max"] = float(np.nanmax(signal_values)) if signal_values.size else np.nan
        row[f"{name}_peak_time_rel"] = float(times_rel[peak_idx[1]])
        row[f"{name}_peak_freq_hz"] = float(band_freqs[peak_idx[0]])
    return row


def main() -> None:
    parser = ArgumentParser(description="Produce raw and T-whitened M512 A/E WDM time-frequency maps.")
    parser.add_argument("--source", nargs="+", choices=("noisy", "gw_only"), default=["noisy", "gw_only"])
    parser.add_argument("--target-dt", type=float, default=2.0)
    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--prefix", default="aet_tf_maps_dt2_M512")
    parser.add_argument("--show-signal", action="store_true")
    args = parser.parse_args()

    resolution = WDMResolution(f"dt{args.target_dt:g}_M{args.M}".replace(".", "p"), float(args.target_dt), int(args.M))
    config = CleanWDMConfig(resolutions=(resolution,), groups=("AET",), sources=tuple(args.source))
    resample_poly, WDM = require_wdm_dependencies()
    paths = CleanWDMPaths.from_work_dir(WORK_DIR, prefix=args.prefix)
    paths.make_dirs()

    t_lisa, data = load_simulation_channels(paths, tdi_suffix=config.tdi_suffix)
    _, t_noise = load_noise_only_t_channel(paths, config.tdi_suffix)
    native_dt = float(np.median(np.diff(t_lisa)))
    signal = locate_signal_window(t_lisa, data["gw_only"]["AET"], threshold_fraction=config.signal_threshold_fraction)

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

    signal_payload = np.asarray(
        [
            signal["signal_peak_lisa"],
            signal["signal_peak_rel"],
            signal["signal_window_start_lisa"],
            signal["signal_window_stop_lisa"],
            signal["signal_window_start_rel"],
            signal["signal_window_stop_rel"],
        ],
        dtype=float,
    )
    payload: dict[str, np.ndarray] = {
        "signal": signal_payload,
        "native_t_lisa_start": np.asarray(float(t_lisa[0])),
        "native_dt_s": np.asarray(native_dt),
        f"T_noise_{resolution.label}_floor": t_floor.astype(np.float32),
        f"T_noise_{resolution.label}_times_rel": (times_wdm - t_lisa[0]).astype(np.float64),
        f"T_noise_{resolution.label}_freqs_hz": freqs_wdm.astype(np.float64),
    }
    rows = []
    for source in args.source:
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
                raise ValueError("A/E WDM maps are not aligned with the T-noise floor map.")
            raw_power = np.abs(coeff) ** 2
            twhitened_power = raw_power / t_floor[:, None]
            key = f"{source}_{channel}_{resolution.label}"
            payload[f"{key}_raw_real"] = np.real(coeff).astype(np.float32)
            payload[f"{key}_raw_imag"] = np.imag(coeff).astype(np.float32)
            payload[f"{key}_raw_power"] = raw_power.astype(np.float32)
            payload[f"{key}_twhitened_power"] = twhitened_power.astype(np.float32)
            payload[f"{key}_times_rel"] = (x_times - t_lisa[0]).astype(np.float64)
            payload[f"{key}_freqs_hz"] = x_freqs.astype(np.float64)
            rows.append(
                summarize_map(
                    source=source,
                    channel=channel,
                    resolution=resolution,
                    raw_power=raw_power,
                    twhitened_power=twhitened_power,
                    times_rel=x_times - t_lisa[0],
                    freqs_hz=x_freqs,
                    signal=signal,
                )
            )

    stats = pd.DataFrame(rows)
    stats_csv = paths.output_dir / f"{args.prefix}_stats.csv"
    maps_npz = paths.output_dir / f"{args.prefix}_maps.npz"
    stats.to_csv(stats_csv, index=False)
    np.savez_compressed(maps_npz, **payload)

    plots = {}
    for source in args.source:
        plot_path = paths.plot_dir / f"{args.prefix}_{source}_{resolution.label}_raw_vs_twhitened_AE.png"
        plot_source_tf_maps(
            plot_path,
            maps=payload,
            source=source,
            resolution=resolution.label,
            signal=signal_payload,
            show_signal=bool(args.show_signal),
        )
        plots[source] = str(plot_path)

    report = {
        "method": "raw and T-channel-whitened WDM TF maps for A/E channels; no A/E cross-correlation weighting",
        "config": {
            "resolution": {"label": resolution.label, "target_dt_s": resolution.target_dt, "M": resolution.M},
            "sources": list(args.source),
            "low_freq_hz": LOW_HZ,
            "high_freq_hz": HIGH_HZ,
            "show_signal": bool(args.show_signal),
        },
        "diagnostics": {
            "native_samples": int(len(t_lisa)),
            "native_dt_s": native_dt,
            **signal,
        },
        "outputs": {
            "stats_csv": str(stats_csv),
            "maps_npz": str(maps_npz),
            "plots": plots,
        },
    }
    report_json = paths.output_dir / f"{args.prefix}_report.json"
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report = {report_json}")


if __name__ == "__main__":
    main()
