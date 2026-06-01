from __future__ import annotations

import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MAPS_FILE = OUTPUT_DIR / "1e5_wdm_multires_maps.npz"
METADATA_FILE = OUTPUT_DIR / "1e5_plot_and_wdm_metadata.json"

AET_CHANNELS = ["A", "E", "T"]
ZOOM_TIME_HALF_BINS = 10
ZOOM_FREQ_HALF_BINS = 6
ZOOM_REL_LOG_FLOOR = -4.0


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        width = 1.0
        return np.array([values[0] - 0.5 * width, values[0] + 0.5 * width])
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.maximum(edges, np.nextafter(0.0, 1.0))


def robust_limits(images: list[np.ndarray]) -> tuple[float, float]:
    finite = [im[np.isfinite(im)] for im in images if np.any(np.isfinite(im))]
    if not finite:
        return -1.0, 1.0
    vals = np.concatenate(finite)
    lo, hi = np.nanpercentile(vals, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    return float(lo), float(hi)


def centered_slice(size: int, center: int, half_width: int) -> slice:
    start = max(0, center - half_width)
    stop = min(size, center + half_width + 1)
    return slice(start, stop)


def peak_zoom(
    power: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    peak_time: float,
    peak_freq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(peak_time) or not np.isfinite(peak_freq):
        return np.array([]), np.array([]), np.empty((0, 0))

    time_idx = int(np.nanargmin(np.abs(times - peak_time)))
    freq_idx = int(np.nanargmin(np.abs(freqs - peak_freq)))
    ts = centered_slice(times.size, time_idx, ZOOM_TIME_HALF_BINS)
    fs = centered_slice(freqs.size, freq_idx, ZOOM_FREQ_HALF_BINS)

    crop_freqs = freqs[fs]
    crop = power[fs, ts]
    positive = crop_freqs > 0.0
    crop_freqs = crop_freqs[positive]
    crop = crop[positive]
    finite = np.isfinite(crop) & (crop > 0)
    if not np.any(finite):
        return times[ts], crop_freqs, np.full(crop.shape, np.nan)

    local_max = float(np.nanmax(crop[finite]))
    image = np.log10(np.maximum(crop / local_max, 1e-30))
    return times[ts], crop_freqs, np.maximum(image, ZOOM_REL_LOG_FLOOR)


def peak_pixel(
    power: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    low_hz: float,
    high_hz: float,
    edge_guard_s: float,
    time_window: tuple[float, float] | None = None,
) -> dict[str, float | int]:
    fmask = (freqs >= low_hz) & (freqs <= high_hz)
    tmask = (times >= edge_guard_s) & (times <= times[-1] - edge_guard_s)
    if time_window is not None:
        tmask &= (times >= time_window[0]) & (times <= time_window[1])

    sub = power[np.ix_(fmask, tmask)]
    if sub.size == 0 or not np.any(np.isfinite(sub)):
        return {
            "time_s": np.nan,
            "frequency_hz": np.nan,
            "power": np.nan,
            "freq_index": -1,
            "time_index": -1,
        }

    local_flat = int(np.nanargmax(sub))
    local_f, local_t = np.unravel_index(local_flat, sub.shape)
    freq_indices = np.flatnonzero(fmask)
    time_indices = np.flatnonzero(tmask)
    f_idx = int(freq_indices[local_f])
    t_idx = int(time_indices[local_t])
    return {
        "time_s": float(times[t_idx]),
        "frequency_hz": float(freqs[f_idx]),
        "power": float(power[f_idx, t_idx]),
        "freq_index": f_idx,
        "time_index": t_idx,
    }


def main() -> None:
    maps = np.load(MAPS_FILE)
    metadata = json.loads(METADATA_FILE.read_text())
    configs = metadata["wdm_configs"]
    low_hz = float(metadata["low_freq_cut_hz"])
    high_hz = float(metadata["high_freq_cut_hz"])
    signal_window = tuple(float(x) for x in maps["signal_window_rel"])
    signal_peak_rel = float(maps["signal_peak_rel"])
    edge_guard_s = float(metadata.get("analysis_edge_guard_seconds", 10_000.0))

    prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    zoom_prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    images: list[np.ndarray] = []
    rows: list[dict[str, float | int | str]] = []

    for r, config in enumerate(configs):
        label = config["label"]
        for c, channel in enumerate(AET_CHANNELS):
            noisy_prefix = f"noisy_AET_{channel}_{label}"
            gw_prefix = f"gw_only_AET_{channel}_{label}"

            noisy_power = maps[f"{noisy_prefix}_normalized_power"]
            gw_power = maps[f"{gw_prefix}_normalized_power"]
            times = maps[f"{noisy_prefix}_times_rel"]
            freqs = maps[f"{noisy_prefix}_freqs_hz"]

            fmask = (freqs > 0.0) & (freqs <= high_hz)
            image = np.log10(noisy_power[fmask] + 1e-30)
            prepared[(r, c)] = (times, freqs[fmask], image)

            noisy_peak = peak_pixel(noisy_power, times, freqs, low_hz=low_hz, high_hz=high_hz, edge_guard_s=edge_guard_s)
            gw_peak = peak_pixel(
                gw_power,
                times,
                freqs,
                low_hz=low_hz,
                high_hz=high_hz,
                edge_guard_s=edge_guard_s,
                time_window=signal_window,
            )
            zoom_prepared[(r, c)] = peak_zoom(
                noisy_power,
                times,
                freqs,
                float(noisy_peak["time_s"]),
                float(noisy_peak["frequency_hz"]),
            )
            images.append(image)

            rows.append(
                {
                    "channel": channel,
                    "resolution": label,
                    "target_dt_s": float(config["target_dt"]),
                    "wdm_M": int(config["M"]),
                    "noisy_peak_time_s": noisy_peak["time_s"],
                    "noisy_peak_frequency_hz": noisy_peak["frequency_hz"],
                    "noisy_peak_power": noisy_peak["power"],
                    "signal_peak_time_s": gw_peak["time_s"],
                    "signal_peak_frequency_hz": gw_peak["frequency_hz"],
                    "signal_peak_power": gw_peak["power"],
                    "time_offset_noisy_minus_signal_s": float(noisy_peak["time_s"] - gw_peak["time_s"]),
                    "frequency_ratio_noisy_over_signal": float(noisy_peak["frequency_hz"] / gw_peak["frequency_hz"])
                    if np.isfinite(noisy_peak["frequency_hz"]) and np.isfinite(gw_peak["frequency_hz"]) and gw_peak["frequency_hz"] > 0
                    else np.nan,
                    "reference_aet_peak_time_s": signal_peak_rel,
                    "reference_window_start_s": signal_window[0],
                    "reference_window_stop_s": signal_window[1],
                }
            )

    vmin, vmax = robust_limits(images)
    fig, axes = plt.subplots(
        len(configs),
        len(AET_CHANNELS),
        figsize=(10.5, max(1.35 * len(configs), 8.5)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(configs), len(AET_CHANNELS))
    last_im = None

    peak_df = pd.DataFrame(rows)
    for r, config in enumerate(configs):
        label = config["label"]
        for c, channel in enumerate(AET_CHANNELS):
            ax = axes[r, c]
            times, freqs_plot, image = prepared[(r, c)]
            time_edges = centers_to_edges(times)
            freq_edges = centers_to_edges(freqs_plot)
            last_im = ax.pcolormesh(
                time_edges,
                freq_edges,
                image,
                shading="auto",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            row = peak_df[(peak_df["resolution"] == label) & (peak_df["channel"] == channel)].iloc[0]
            ax.scatter(
                row["noisy_peak_time_s"],
                row["noisy_peak_frequency_hz"],
                marker="o",
                s=34,
                facecolor="#00e5ff",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )
            ax.scatter(
                row["signal_peak_time_s"],
                row["signal_peak_frequency_hz"],
                marker="*",
                s=95,
                facecolor="#ffea00",
                edgecolor="black",
                linewidth=0.6,
                zorder=6,
            )
            ax.set_yscale("log")
            ax.set_ylim(low_hz, high_hz)
            if r == 0:
                ax.set_title(channel)
            if c == 0:
                ax.set_ylabel(f"{label}\ndt={config['target_dt']:g}s, M={config['M']}", fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            if r == len(configs) - 1:
                ax.set_xlabel("time in science segment [s]")
            else:
                ax.tick_params(labelbottom=False)

    dot = mlines.Line2D(
        [],
        [],
        color="#00e5ff",
        marker="o",
        linestyle="None",
        markeredgecolor="black",
        markersize=7,
        label="noisy AET WDM peak",
    )
    star = mlines.Line2D(
        [],
        [],
        color="#ffea00",
        marker="*",
        linestyle="None",
        markeredgecolor="black",
        markersize=11,
        label="GW-only signal WDM peak",
    )
    fig.legend(handles=[dot, star], loc="upper right", bbox_to_anchor=(0.985, 0.99))
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.88, extend="both")
    cbar.set_label(r"$\log_{10}$ noisy off-source-normalized WDM power")
    fig.suptitle(
        "1e5 AET multi-resolution WDM peak check\n"
        "dot: peak in noisy map; star: peak in GW-only signal map inside signal window",
        fontsize=12,
    )

    plot_file = PLOT_DIR / "09_1e5_multires_wdm_AET_peak_markers.png"
    stats_file = OUTPUT_DIR / "1e5_wdm_AET_peak_marker_stats.csv"
    fig.savefig(plot_file, dpi=170)
    peak_df.to_csv(stats_file, index=False)
    plt.close(fig)

    zoom_fig, zoom_axes = plt.subplots(
        len(configs),
        len(AET_CHANNELS),
        figsize=(10.5, max(1.35 * len(configs), 8.5)),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    zoom_axes = np.asarray(zoom_axes).reshape(len(configs), len(AET_CHANNELS))
    zoom_im = None
    for r, config in enumerate(configs):
        label = config["label"]
        for c, channel in enumerate(AET_CHANNELS):
            ax = zoom_axes[r, c]
            times, freqs_plot, image = zoom_prepared[(r, c)]
            row = peak_df[(peak_df["resolution"] == label) & (peak_df["channel"] == channel)].iloc[0]
            if image.size:
                zoom_im = ax.pcolormesh(
                    centers_to_edges(times),
                    centers_to_edges(freqs_plot),
                    image,
                    shading="auto",
                    cmap="magma",
                    vmin=ZOOM_REL_LOG_FLOOR,
                    vmax=0.0,
                )
                ax.scatter(
                    row["noisy_peak_time_s"],
                    row["noisy_peak_frequency_hz"],
                    marker="o",
                    s=34,
                    facecolor="#00e5ff",
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=5,
                )
                if np.isfinite(row["signal_peak_time_s"]) and np.isfinite(row["signal_peak_frequency_hz"]):
                    ax.scatter(
                        row["signal_peak_time_s"],
                        row["signal_peak_frequency_hz"],
                        marker="*",
                        s=95,
                        facecolor="#ffea00",
                        edgecolor="black",
                        linewidth=0.6,
                        zorder=6,
                    )
                ax.set_yscale("log")
                ax.set_xlim(float(times[0]), float(times[-1]))
                ax.set_ylim(float(freqs_plot[0]), float(freqs_plot[-1]))
                ax.text(
                    0.02,
                    0.95,
                    f"log10 Pmax={np.log10(row['noisy_peak_power']):.2f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.5},
                )
            else:
                ax.text(0.5, 0.5, "no finite peak", transform=ax.transAxes, ha="center", va="center")
            if r == 0:
                ax.set_title(channel)
            if c == 0:
                ax.set_ylabel(f"{label}\ndt={config['target_dt']:g}s, M={config['M']}", fontsize=8)
            if r == len(configs) - 1:
                ax.set_xlabel("time [s]")

    zoom_fig.legend(handles=[dot, star], loc="upper right", bbox_to_anchor=(0.985, 0.99))
    if zoom_im is not None:
        zoom_cbar = zoom_fig.colorbar(zoom_im, ax=zoom_axes, shrink=0.88, extend="min")
        zoom_cbar.set_label(r"$\log_{10}(P/P_{\rm local\ max})$")
    zoom_fig.suptitle(
        "1e5 AET WDM local zoom around noisy peak\n"
        "each panel is centered on the dot and uses a relative local z scale",
        fontsize=12,
    )
    zoom_plot_file = PLOT_DIR / "11_1e5_multires_wdm_AET_peak_zoom.png"
    zoom_fig.savefig(zoom_plot_file, dpi=190)
    plt.close(zoom_fig)

    print(f"saved {plot_file}")
    print(f"saved {zoom_plot_file}")
    print(f"saved {stats_file}")
    print(
        peak_df[
            [
                "channel",
                "resolution",
                "noisy_peak_time_s",
                "noisy_peak_frequency_hz",
                "signal_peak_time_s",
                "signal_peak_frequency_hz",
                "time_offset_noisy_minus_signal_s",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
