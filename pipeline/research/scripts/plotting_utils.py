"""Plotting and visualization helpers for the reusable LISA pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

LINK_LABELS: tuple[str, ...] = ("12", "23", "31", "13", "32", "21")
XYZ_LABELS: tuple[str, ...] = ("X", "Y", "Z")
AET_LABELS: tuple[str, ...] = ("A", "E", "T")
GROUP_CHANNELS: Mapping[str, tuple[str, ...]] = {
    "links": LINK_LABELS,
    "XYZ": XYZ_LABELS,
    "AET": AET_LABELS,
}


def _get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        raise ModuleNotFoundError(
            "Plotting pipeline products requires matplotlib. "
            "Install the project environment from env.yml, or disable plotting."
        ) from exc
    return plt


def plot_waveform(prepared, outfile: str | Path) -> None:
    plt = _get_pyplot()
    plt.figure(figsize=(12, 4))
    plt.plot(prepared.tau, prepared.hplus, label="hplus")
    plt.plot(prepared.tau, prepared.hcross, label="hcross")
    plt.xlabel("time relative to waveform peak [s]")
    plt.ylabel("strain")
    plt.title("Aligned waveform: peak amplitude centered")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_link_columns(
    x: np.ndarray,
    channels: Mapping[str, np.ndarray],
    title: str,
    ylabel: str,
    outfile: str | Path,
) -> None:
    plt = _get_pyplot()
    plt.figure(figsize=(13, 6))
    for name, y in channels.items():
        plt.plot(x, y, lw=0.8, label=name)
    plt.xlabel("LISA orbit time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(ncol=3, loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def _robust_absmax(arr: np.ndarray) -> float:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0.0
    return float(np.nanmax(np.abs(arr[finite])))


def _signal_window(signal_arrays: Sequence[np.ndarray], t: np.ndarray, threshold_fraction: float) -> tuple[float, float] | None:
    power = np.zeros(len(t), dtype=float)
    for arr in signal_arrays:
        scale = _robust_absmax(arr)
        if scale > 0.0:
            power += (np.asarray(arr, dtype=float) / scale) ** 2
    if not np.any(power > 0.0):
        return None

    threshold = threshold_fraction * np.nanmax(power)
    idx = np.flatnonzero(power >= threshold)
    if len(idx) == 0:
        return None
    return float(t[idx[0]]), float(t[idx[-1]])


def plot_noisy_with_gw_signal(
    paths: LisaLinkSimulationPaths,
    *,
    suffix: str = "2",
    threshold_fraction: float = 1e-3,
) -> dict[str, float]:
    """Overlay Hz-consistent GW-only TDI on noisy TDI plots."""

    plt = _get_pyplot()
    noisy = pd.read_csv(paths.noisy_tdi_csv)
    signal = pd.read_csv(paths.gw_tdi_csv)

    diagnostics: dict[str, float] = {}
    channel_sets = [
        (
            [f"X{suffix}", f"Y{suffix}", f"Z{suffix}"],
            [f"X{suffix}_gw_hz", f"Y{suffix}_gw_hz", f"Z{suffix}_gw_hz"],
            "Michelson TDI with lisainstrument noise and Hz-converted GW-only signal",
            paths.overlay_xyz_plot,
            "xyz",
        ),
        (
            [f"A{suffix}", f"E{suffix}", f"T{suffix}"],
            [f"A{suffix}_gw_hz", f"E{suffix}_gw_hz", f"T{suffix}_gw_hz"],
            "Orthogonal TDI with lisainstrument noise and Hz-converted GW-only signal",
            paths.overlay_aet_plot,
            "aet",
        ),
    ]

    for noisy_cols, signal_cols, title, outfile, tag in channel_sets:
        t = noisy["t_lisa"].to_numpy()
        signal_t = signal["t_lisa"].to_numpy()
        if len(signal_t) != len(t) or not np.allclose(signal_t, t, rtol=0.0, atol=1e-6):
            signal = signal.set_index("t_lisa").reindex(t, method="nearest", tolerance=1e-6).reset_index()

        noisy_channels = {name: noisy[name].to_numpy() for name in noisy_cols}
        signal_channels = {name: signal[src].to_numpy() for name, src in zip(noisy_cols, signal_cols)}
        noisy_max = max(_robust_absmax(arr) for arr in noisy_channels.values())
        signal_max = max(_robust_absmax(arr) for arr in signal_channels.values())
        visual_scale = 1.0 if noisy_max == 0.0 or signal_max == 0.0 else 0.35 * noisy_max / signal_max
        window = _signal_window(list(signal_channels.values()), t, threshold_fraction)

        plt.figure(figsize=(13, 6))
        for name, arr in noisy_channels.items():
            plt.plot(t, arr, lw=0.7, alpha=0.55, label=f"{name} noise+GW")
        if window is not None:
            plt.axvspan(window[0], window[1], color="#fff176", alpha=0.22, label="GW signal interval")
            plt.axvline(0.5 * (window[0] + window[1]), color="#ff1744", lw=1.5, ls="--", alpha=0.9, label="signal center")
            diagnostics[f"{tag}_signal_start_lisa"] = window[0]
            diagnostics[f"{tag}_signal_stop_lisa"] = window[1]
            diagnostics[f"{tag}_signal_center_lisa"] = 0.5 * (window[0] + window[1])
        for name, arr in signal_channels.items():
            plt.plot(t, visual_scale * arr, lw=1.2, alpha=0.95, label=f"{name} GW-only [Hz] x {visual_scale:.2e}")
        plt.xlabel("LISA orbit time [s]")
        plt.ylabel("TDI frequency combination [Hz]")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend(ncol=2, loc="upper right")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()
        diagnostics[f"{tag}_visual_signal_scale"] = float(visual_scale)

    return diagnostics

def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        width = 1.0
        return np.array([values[0] - 0.5 * width, values[0] + 0.5 * width])
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def _tf_plot_edges(times_rel_wdm: np.ndarray, freqs: np.ndarray, tf_dt: float) -> tuple[np.ndarray, np.ndarray]:
    time_edges = np.concatenate(
        [
            [times_rel_wdm[0] - 0.5 * tf_dt],
            0.5 * (times_rel_wdm[:-1] + times_rel_wdm[1:]),
            [times_rel_wdm[-1] + 0.5 * tf_dt],
        ]
    )
    freq_edges = _centers_to_edges(freqs)
    freq_edges = np.maximum(freq_edges, np.nextafter(0.0, 1.0))
    return time_edges, freq_edges


def _robust_limits(arrays: Sequence[np.ndarray], low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    finite = []
    for arr in arrays:
        vals = np.asarray(arr)[np.isfinite(arr)]
        if vals.size:
            finite.append(vals)
    if not finite:
        return -1.0, 1.0
    vals = np.concatenate(finite)
    vmin, vmax = np.nanpercentile(vals, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    return float(vmin), float(vmax)


def _log_power(power: np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(power, dtype=float) + 1e-30)


def _to_float32_npz(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    limit = np.finfo(np.float32).max
    return np.clip(arr, -limit, limit).astype(np.float32)


def _plot_multires_grid(
    source_name: str,
    group: str,
    wdm_products: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    paths: TDIWDMPipelinePaths,
    config: TDIWDMPipelineConfig,
) -> None:
    plt = _get_pyplot()
    channels = GROUP_CHANNELS[group]
    nrows = len(config.resolutions)
    ncols = len(channels)
    prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    images_for_limits = []

    for r, resolution in enumerate(config.resolutions):
        for c, ch in enumerate(channels):
            prod = wdm_products[(source_name, group, ch, resolution.label)]
            freqs = np.asarray(prod["freqs_hz"])
            fmask = (freqs > 0.0) & (freqs <= config.high_freq_cut)
            image = _log_power(np.asarray(prod["normalized_power"])[fmask])
            time_edges, freq_edges = _tf_plot_edges(np.asarray(prod["times_rel"]), freqs[fmask], float(prod["tf"].dt))
            prepared[(r, c)] = (time_edges, freq_edges, image)
            images_for_limits.append(image)

    vmin, vmax = _robust_limits(images_for_limits)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(3.2 * ncols, 9), max(1.35 * nrows, 12)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)
    last_im = None

    for r, resolution in enumerate(config.resolutions):
        for c, ch in enumerate(channels):
            ax = axes[r, c]
            time_edges, freq_edges, image = prepared[(r, c)]
            last_im = ax.pcolormesh(
                time_edges,
                freq_edges,
                image,
                shading="auto",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_yscale("log")
            ax.set_ylim(config.low_freq_cut, config.high_freq_cut)
            if r == 0:
                ax.set_title(ch, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{resolution.label}\ndt={resolution.target_dt:g}s, M={resolution.M}", fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            if r < nrows - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time [s]")

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, extend="both")
    cbar.set_label(r"$\log_{10}$ off-source-normalized WDM power")
    fig.suptitle(
        f"Multi-resolution WDM grid: {source_name} {group}\n"
        f"frequency range {config.low_freq_cut:.0e}-{config.high_freq_cut:.0e} Hz",
        fontsize=12,
    )
    outfile = paths.plot_dir / f"15_multires_grid_{source_name}_{group}.png"
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


def _plot_t_weighted_grid(
    wdm_products: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    t_weighted_payload: Mapping[str, np.ndarray],
    paths: TDIWDMPipelinePaths,
    config: TDIWDMPipelineConfig,
) -> None:
    plt = _get_pyplot()
    channels = ("A", "E")
    nrows = len(config.resolutions)
    prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    images_for_limits = []

    for r, resolution in enumerate(config.resolutions):
        for c, ch in enumerate(channels):
            prod = wdm_products[("noisy", "AET", ch, resolution.label)]
            key = f"noisy_AET_{ch}_Twhiten_{resolution.label}_power"
            freqs = np.asarray(prod["freqs_hz"])
            fmask = (freqs > 0.0) & (freqs <= config.high_freq_cut)
            image = _log_power(t_weighted_payload[key][fmask])
            time_edges, freq_edges = _tf_plot_edges(np.asarray(prod["times_rel"]), freqs[fmask], float(prod["tf"].dt))
            prepared[(r, c)] = (time_edges, freq_edges, image)
            images_for_limits.append(image)

    vmin, vmax = _robust_limits(images_for_limits)
    fig, axes = plt.subplots(
        nrows,
        len(channels),
        figsize=(8, max(1.35 * nrows, 12)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(nrows, len(channels))
    last_im = None

    for r, resolution in enumerate(config.resolutions):
        for c, ch in enumerate(channels):
            ax = axes[r, c]
            time_edges, freq_edges, image = prepared[(r, c)]
            last_im = ax.pcolormesh(
                time_edges,
                freq_edges,
                image,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_yscale("log")
            ax.set_ylim(config.low_freq_cut, config.high_freq_cut)
            if r == 0:
                ax.set_title(f"{ch} / T-noise proxy", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{resolution.label}\ndt={resolution.target_dt:g}s, M={resolution.M}", fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            if r < nrows - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time [s]")

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, extend="both")
    cbar.set_label(r"$\log_{10}$ T-weighted WDM power")
    fig.suptitle(
        f"Multi-resolution WDM grid: noisy A/E weighted by T\n"
        f"frequency range {config.low_freq_cut:.0e}-{config.high_freq_cut:.0e} Hz",
        fontsize=12,
    )
    outfile = paths.plot_dir / "15_multires_grid_noisy_AE_Twhiten.png"
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


def _combined_best_pixel_map(
    source_name: str,
    group: str,
    channel: str,
    wdm_products: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    t_weighted_payload: Mapping[str, np.ndarray],
    t_rel: np.ndarray,
    config: TDIWDMPipelineConfig,
    *,
    weighted_by_t: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_edges = np.linspace(t_rel[0], t_rel[-1], config.best_pixel_time_bins + 1)
    freq_edges = np.geomspace(config.low_freq_cut, config.high_freq_cut, config.best_pixel_freq_bins + 1)
    best = np.full((config.best_pixel_freq_bins, config.best_pixel_time_bins), np.nan, dtype=float)
    best_resolution = np.full((config.best_pixel_freq_bins, config.best_pixel_time_bins), "", dtype=object)

    for resolution in config.resolutions:
        if weighted_by_t:
            prod = wdm_products[("noisy", "AET", channel, resolution.label)]
            power = t_weighted_payload[f"noisy_AET_{channel}_Twhiten_{resolution.label}_power"]
        else:
            prod = wdm_products[(source_name, group, channel, resolution.label)]
            power = np.asarray(prod["normalized_power"])
        freqs = np.asarray(prod["freqs_hz"])
        times = np.asarray(prod["times_rel"])
        fmask = (freqs >= config.low_freq_cut) & (freqs <= config.high_freq_cut)
        if not np.any(fmask) or len(times) == 0:
            continue

        values = power[fmask]
        freq_centers = freqs[fmask]
        tt = np.tile(times, len(freq_centers))
        ff = np.repeat(freq_centers, len(times))
        vv = values.reshape(-1)
        ti = np.digitize(tt, time_edges) - 1
        fi = np.digitize(ff, freq_edges) - 1
        valid = np.isfinite(vv) & (ti >= 0) & (ti < config.best_pixel_time_bins) & (fi >= 0) & (
            fi < config.best_pixel_freq_bins
        )
        for f_idx, t_idx, val in zip(fi[valid], ti[valid], vv[valid]):
            if np.isnan(best[f_idx, t_idx]) or val > best[f_idx, t_idx]:
                best[f_idx, t_idx] = val
                best_resolution[f_idx, t_idx] = resolution.label

    return time_edges, freq_edges, best, best_resolution


def _summarize_best_pixel(source_name: str, group: str, channel: str, best: np.ndarray, best_resolution: np.ndarray):
    if not np.any(np.isfinite(best)):
        return {
            "source": source_name,
            "group": group,
            "channel": channel,
            "best_pixel_power": np.nan,
            "best_pixel_log10_power": np.nan,
            "best_pixel_resolution": "",
            "finite_bins": 0,
        }
    idx = np.nanargmax(best)
    f_idx, t_idx = np.unravel_index(idx, best.shape)
    return {
        "source": source_name,
        "group": group,
        "channel": channel,
        "best_pixel_power": float(best[f_idx, t_idx]),
        "best_pixel_log10_power": float(np.log10(best[f_idx, t_idx] + 1e-30)),
        "best_pixel_resolution": best_resolution[f_idx, t_idx],
        "finite_bins": int(np.sum(np.isfinite(best))),
    }


def _save_and_plot_best_pixels(
    wdm_products: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    t_weighted_payload: Mapping[str, np.ndarray],
    t_rel: np.ndarray,
    paths: TDIWDMPipelinePaths,
    config: TDIWDMPipelineConfig,
) -> pd.DataFrame:
    plt = _get_pyplot() if config.make_best_pixel_plots else None
    rows = []
    payload: dict[str, np.ndarray] = {
        "time_edges_rel": np.linspace(t_rel[0], t_rel[-1], config.best_pixel_time_bins + 1),
        "freq_edges_hz": np.geomspace(config.low_freq_cut, config.high_freq_cut, config.best_pixel_freq_bins + 1),
    }

    def plot_group(source_name: str, group: str) -> None:
        channels = GROUP_CHANNELS[group]
        fig, axes = plt.subplots(1, len(channels), figsize=(4.2 * len(channels), 3.8), sharey=True)
        axes = np.asarray(axes).reshape(-1)
        maps = {}
        images = []
        for ch in channels:
            time_edges, freq_edges, best, best_resolution = _combined_best_pixel_map(
                source_name, group, ch, wdm_products, t_weighted_payload, t_rel, config
            )
            maps[ch] = (time_edges, freq_edges, best, best_resolution)
            images.append(_log_power(best))
        vmin, vmax = _robust_limits(images)
        last_im = None
        for ax, ch in zip(axes, channels):
            time_edges, freq_edges, best, best_resolution = maps[ch]
            rows.append(_summarize_best_pixel(source_name, group, ch, best, best_resolution))
            prefix = f"{source_name}_{group}_{ch}_bestpixel"
            payload[f"{prefix}_power"] = _to_float32_npz(best)
            payload[f"{prefix}_resolution"] = best_resolution.astype(str)
            last_im = ax.pcolormesh(
                time_edges,
                freq_edges,
                _log_power(best),
                shading="auto",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_yscale("log")
            ax.set_ylim(config.low_freq_cut, config.high_freq_cut)
            ax.set_title(ch)
            ax.set_xlabel("time [s]")
        axes[0].set_ylabel("frequency [Hz]")
        fig.colorbar(last_im, ax=axes, label=r"$\log_{10}$ best normalized WDM power", extend="both")
        fig.suptitle(f"Combined best-pixel WDM: {source_name} {group}")
        fig.tight_layout()
        fig.savefig(paths.plot_dir / f"14_bestpixel_wdm_{source_name}_{group}.png", dpi=150)
        plt.close(fig)

    if config.make_best_pixel_plots:
        for source_name in ("noisy", "gw_only"):
            for group in ("links", "XYZ", "AET"):
                plot_group(source_name, group)

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
        maps = {}
        images = []
        for ch in ("A", "E"):
            time_edges, freq_edges, best, best_resolution = _combined_best_pixel_map(
                "noisy", "AET", ch, wdm_products, t_weighted_payload, t_rel, config, weighted_by_t=True
            )
            maps[ch] = (time_edges, freq_edges, best, best_resolution)
            images.append(_log_power(best))
        vmin, vmax = _robust_limits(images)
        last_im = None
        for ax, ch in zip(axes, ("A", "E")):
            time_edges, freq_edges, best, best_resolution = maps[ch]
            rows.append(_summarize_best_pixel("noisy", "AET_Tweighted", ch, best, best_resolution))
            prefix = f"noisy_AET_{ch}_Twhiten_bestpixel"
            payload[f"{prefix}_power"] = _to_float32_npz(best)
            payload[f"{prefix}_resolution"] = best_resolution.astype(str)
            last_im = ax.pcolormesh(
                time_edges,
                freq_edges,
                _log_power(best),
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_yscale("log")
            ax.set_ylim(config.low_freq_cut, config.high_freq_cut)
            ax.set_title(f"{ch} / T-noise proxy")
            ax.set_xlabel("time [s]")
        axes[0].set_ylabel("frequency [Hz]")
        fig.colorbar(last_im, ax=axes, label=r"$\log_{10}$ best T-weighted WDM power", extend="both")
        fig.suptitle("Combined best-pixel WDM: noisy A/E weighted by T")
        fig.tight_layout()
        fig.savefig(paths.plot_dir / "14_bestpixel_wdm_noisy_AE_Twhiten.png", dpi=150)
        plt.close(fig)
    else:
        for source_name in ("noisy", "gw_only"):
            for group in ("links", "XYZ", "AET"):
                for ch in GROUP_CHANNELS[group]:
                    time_edges, freq_edges, best, best_resolution = _combined_best_pixel_map(
                        source_name, group, ch, wdm_products, t_weighted_payload, t_rel, config
                    )
                    rows.append(_summarize_best_pixel(source_name, group, ch, best, best_resolution))
                    prefix = f"{source_name}_{group}_{ch}_bestpixel"
                    payload[f"{prefix}_power"] = _to_float32_npz(best)
                    payload[f"{prefix}_resolution"] = best_resolution.astype(str)
        for ch in ("A", "E"):
            time_edges, freq_edges, best, best_resolution = _combined_best_pixel_map(
                "noisy", "AET", ch, wdm_products, t_weighted_payload, t_rel, config, weighted_by_t=True
            )
            rows.append(_summarize_best_pixel("noisy", "AET_Tweighted", ch, best, best_resolution))
            prefix = f"noisy_AET_{ch}_Twhiten_bestpixel"
            payload[f"{prefix}_power"] = _to_float32_npz(best)
            payload[f"{prefix}_resolution"] = best_resolution.astype(str)

    bestpixel_df = pd.DataFrame(rows)
    bestpixel_df.to_csv(paths.bestpixel_stats_csv, index=False)
    np.savez_compressed(paths.bestpixel_maps_npz, **payload)
    return bestpixel_df
