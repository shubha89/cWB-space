from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_1e5_plots_and_wdm import (
    AET_LABELS,
    HIGH_FREQ_CUT,
    LOW_FREQ_CUT,
    MIN_NOISE_FLOOR,
    OFFSOURCE_GUARD_SECONDS,
    ANALYSIS_EDGE_GUARD_SECONDS,
    OUTPUT_DIR,
    PLOT_DIR,
    WDM_CONFIGS,
    centers_to_edges,
    robust_limits,
    signal_window_from_gw_aet,
    wdm_transform_channel,
)


ZOOM_TIME_HALF_BINS = 10
ZOOM_FREQ_HALF_BINS = 6
ZOOM_REL_LOG_FLOOR = -4.0


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
    edge_guard_s: float,
    time_window: tuple[float, float] | None = None,
) -> dict[str, float | int]:
    fmask = (freqs >= LOW_FREQ_CUT) & (freqs <= HIGH_FREQ_CUT)
    tmask = (times >= edge_guard_s) & (times <= times[-1] - edge_guard_s)
    if time_window is not None:
        tmask &= (times >= time_window[0]) & (times <= time_window[1])

    sub = power[np.ix_(fmask, tmask)]
    if sub.size == 0 or not np.any(np.isfinite(sub)):
        return {"time_s": np.nan, "frequency_hz": np.nan, "power": np.nan}

    local_f, local_t = np.unravel_index(int(np.nanargmax(sub)), sub.shape)
    freq_indices = np.flatnonzero(fmask)
    time_indices = np.flatnonzero(tmask)
    f_idx = int(freq_indices[local_f])
    t_idx = int(time_indices[local_t])
    return {
        "time_s": float(times[t_idx]),
        "frequency_hz": float(freqs[f_idx]),
        "power": float(power[f_idx, t_idx]),
    }


def main() -> None:
    noisy_tdi = pd.read_csv(OUTPUT_DIR / "tdi_michelson_XYZ_AET.csv")
    gw_tdi = pd.read_csv(OUTPUT_DIR / "tdi_gw_only_XYZ_AET.csv")

    t_lisa = noisy_tdi["t_lisa"].to_numpy(dtype=float)
    native_dt = float(np.median(np.diff(t_lisa)))
    t0 = float(t_lisa[0])
    t1 = float(t_lisa[-1])

    signal_start, signal_stop, signal_peak = signal_window_from_gw_aet(gw_tdi)
    signal_window_rel = (signal_start - t0, signal_stop - t0)

    rows: list[dict[str, float | int | str]] = []
    prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    zoom_prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    images: list[np.ndarray] = []
    payload: dict[str, np.ndarray] = {
        "signal_window_rel": np.asarray(signal_window_rel, dtype=float),
        "signal_peak_rel": np.asarray(signal_peak - t0, dtype=float),
    }

    for r, config in enumerate(WDM_CONFIGS):
        label = config["label"]
        print(f"running {label}")

        noisy_t = wdm_transform_channel(
            noisy_tdi["T2"].to_numpy(dtype=float),
            t0,
            t1,
            native_dt,
            config,
        )
        off_mask = (
            (noisy_t["times"] < signal_start - OFFSOURCE_GUARD_SECONDS)
            | (noisy_t["times"] > signal_stop + OFFSOURCE_GUARD_SECONDS)
        )
        t_floor = np.nanmedian(noisy_t["power"][:, off_mask], axis=1)
        t_floor = np.where(np.isfinite(t_floor) & (t_floor > 0), t_floor, MIN_NOISE_FLOOR)

        for c, channel in enumerate(["A", "E"]):
            noisy = wdm_transform_channel(
                noisy_tdi[f"{channel}2"].to_numpy(dtype=float),
                t0,
                t1,
                native_dt,
                config,
            )
            signal = wdm_transform_channel(
                gw_tdi[f"{channel}2_gw_hz"].to_numpy(dtype=float),
                t0,
                t1,
                native_dt,
                config,
            )

            noisy_twhiten = noisy["power"] / np.maximum(t_floor[:, None], MIN_NOISE_FLOOR)
            signal_twhiten = signal["power"] / np.maximum(t_floor[:, None], MIN_NOISE_FLOOR)
            times_rel = noisy["times"] - t0

            noisy_peak = peak_pixel(noisy_twhiten, times_rel, noisy["freqs"], edge_guard_s=ANALYSIS_EDGE_GUARD_SECONDS)
            signal_peak_pixel = peak_pixel(
                signal_twhiten,
                times_rel,
                signal["freqs"],
                edge_guard_s=ANALYSIS_EDGE_GUARD_SECONDS,
                time_window=signal_window_rel,
            )

            fmask = (noisy["freqs"] > 0.0) & (noisy["freqs"] <= HIGH_FREQ_CUT)
            image = np.log10(noisy_twhiten[fmask] + 1e-30)
            prepared[(r, c)] = (times_rel, noisy["freqs"][fmask], image)
            zoom_prepared[(r, c)] = peak_zoom(
                noisy_twhiten,
                times_rel,
                noisy["freqs"],
                float(noisy_peak["time_s"]),
                float(noisy_peak["frequency_hz"]),
            )
            images.append(image)

            prefix = f"noisy_AET_{channel}_Twhiten_{label}"
            payload[f"{prefix}_power"] = noisy_twhiten.astype(np.float32)
            payload[f"{prefix}_times_rel"] = times_rel.astype(np.float64)
            payload[f"{prefix}_freqs_hz"] = noisy["freqs"].astype(np.float64)

            rows.append(
                {
                    "channel": channel,
                    "resolution": label,
                    "target_dt_s": float(config["target_dt"]),
                    "wdm_M": int(config["M"]),
                    "noisy_twhiten_peak_time_s": noisy_peak["time_s"],
                    "noisy_twhiten_peak_frequency_hz": noisy_peak["frequency_hz"],
                    "noisy_twhiten_peak_power": noisy_peak["power"],
                    "signal_twhiten_peak_time_s": signal_peak_pixel["time_s"],
                    "signal_twhiten_peak_frequency_hz": signal_peak_pixel["frequency_hz"],
                    "signal_twhiten_peak_power": signal_peak_pixel["power"],
                    "time_offset_noisy_minus_signal_s": float(noisy_peak["time_s"] - signal_peak_pixel["time_s"]),
                    "frequency_ratio_noisy_over_signal": float(
                        noisy_peak["frequency_hz"] / signal_peak_pixel["frequency_hz"]
                    )
                    if np.isfinite(noisy_peak["frequency_hz"])
                    and np.isfinite(signal_peak_pixel["frequency_hz"])
                    and signal_peak_pixel["frequency_hz"] > 0
                    else np.nan,
                    "reference_aet_peak_time_s": float(signal_peak - t0),
                    "reference_window_start_s": signal_window_rel[0],
                    "reference_window_stop_s": signal_window_rel[1],
                }
            )

    vmin, vmax = robust_limits(images)
    fig, axes = plt.subplots(
        len(WDM_CONFIGS),
        2,
        figsize=(8.5, max(1.35 * len(WDM_CONFIGS), 8.5)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(WDM_CONFIGS), 2)
    peak_df = pd.DataFrame(rows)
    last_im = None

    for r, config in enumerate(WDM_CONFIGS):
        label = config["label"]
        for c, channel in enumerate(["A", "E"]):
            ax = axes[r, c]
            times, freqs, image = prepared[(r, c)]
            last_im = ax.pcolormesh(
                centers_to_edges(times),
                centers_to_edges(freqs),
                image,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            row = peak_df[(peak_df["resolution"] == label) & (peak_df["channel"] == channel)].iloc[0]
            ax.scatter(
                row["noisy_twhiten_peak_time_s"],
                row["noisy_twhiten_peak_frequency_hz"],
                marker="o",
                s=34,
                facecolor="#00e5ff",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )
            ax.scatter(
                row["signal_twhiten_peak_time_s"],
                row["signal_twhiten_peak_frequency_hz"],
                marker="*",
                s=95,
                facecolor="#ffea00",
                edgecolor="black",
                linewidth=0.6,
                zorder=6,
            )
            ax.set_yscale("log")
            ax.set_ylim(LOW_FREQ_CUT, HIGH_FREQ_CUT)
            if r == 0:
                ax.set_title(f"{channel} / T")
            if c == 0:
                ax.set_ylabel(f"{label}\ndt={config['target_dt']:g}s, M={config['M']}", fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            if r == len(WDM_CONFIGS) - 1:
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
        label="noisy T-whitened peak",
    )
    star = mlines.Line2D(
        [],
        [],
        color="#ffea00",
        marker="*",
        linestyle="None",
        markeredgecolor="black",
        markersize=11,
        label="GW-only T-whitened signal peak",
    )
    fig.legend(handles=[dot, star], loc="upper right", bbox_to_anchor=(0.985, 0.99))
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.88, extend="both")
    cbar.set_label(r"$\log_{10}$ T-whitened WDM power")
    fig.suptitle(
        "1e5 A/E multi-resolution WDM peak check with T whitening\n"
        "dot: noisy A/E divided by off-source T floor; star: GW-only A/E divided by same T floor",
        fontsize=12,
    )

    plot_file = PLOT_DIR / "10_1e5_multires_wdm_AE_Twhiten_peak_markers.png"
    stats_file = OUTPUT_DIR / "1e5_wdm_AE_Twhiten_peak_marker_stats.csv"
    maps_file = OUTPUT_DIR / "1e5_wdm_AE_Twhiten_peak_marker_maps.npz"

    fig.savefig(plot_file, dpi=170)
    peak_df.to_csv(stats_file, index=False)
    np.savez_compressed(maps_file, **payload)
    plt.close(fig)

    zoom_fig, zoom_axes = plt.subplots(
        len(WDM_CONFIGS),
        2,
        figsize=(8.5, max(1.35 * len(WDM_CONFIGS), 8.5)),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    zoom_axes = np.asarray(zoom_axes).reshape(len(WDM_CONFIGS), 2)
    zoom_im = None
    for r, config in enumerate(WDM_CONFIGS):
        label = config["label"]
        for c, channel in enumerate(["A", "E"]):
            ax = zoom_axes[r, c]
            times, freqs, image = zoom_prepared[(r, c)]
            row = peak_df[(peak_df["resolution"] == label) & (peak_df["channel"] == channel)].iloc[0]
            if image.size:
                zoom_im = ax.pcolormesh(
                    centers_to_edges(times),
                    centers_to_edges(freqs),
                    image,
                    shading="auto",
                    cmap="viridis",
                    vmin=ZOOM_REL_LOG_FLOOR,
                    vmax=0.0,
                )
                ax.scatter(
                    row["noisy_twhiten_peak_time_s"],
                    row["noisy_twhiten_peak_frequency_hz"],
                    marker="o",
                    s=34,
                    facecolor="#00e5ff",
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=5,
                )
                if np.isfinite(row["signal_twhiten_peak_time_s"]) and np.isfinite(row["signal_twhiten_peak_frequency_hz"]):
                    ax.scatter(
                        row["signal_twhiten_peak_time_s"],
                        row["signal_twhiten_peak_frequency_hz"],
                        marker="*",
                        s=95,
                        facecolor="#ffea00",
                        edgecolor="black",
                        linewidth=0.6,
                        zorder=6,
                    )
                ax.set_yscale("log")
                ax.set_xlim(float(times[0]), float(times[-1]))
                ax.set_ylim(float(freqs[0]), float(freqs[-1]))
                ax.text(
                    0.02,
                    0.95,
                    f"log10 Pmax={np.log10(row['noisy_twhiten_peak_power']):.2f}",
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
                ax.set_title(f"{channel} / T")
            if c == 0:
                ax.set_ylabel(f"{label}\ndt={config['target_dt']:g}s, M={config['M']}", fontsize=8)
            if r == len(WDM_CONFIGS) - 1:
                ax.set_xlabel("time [s]")

    zoom_fig.legend(handles=[dot, star], loc="upper right", bbox_to_anchor=(0.985, 0.99))
    if zoom_im is not None:
        zoom_cbar = zoom_fig.colorbar(zoom_im, ax=zoom_axes, shrink=0.88, extend="min")
        zoom_cbar.set_label(r"$\log_{10}(P/P_{\rm local\ max})$")
    zoom_fig.suptitle(
        "1e5 A/E T-whitened WDM local zoom around noisy peak\n"
        "each panel is centered on the dot and uses a relative local z scale",
        fontsize=12,
    )
    zoom_plot_file = PLOT_DIR / "12_1e5_multires_wdm_AE_Twhiten_peak_zoom.png"
    zoom_fig.savefig(zoom_plot_file, dpi=190)
    plt.close(zoom_fig)

    print(f"saved {plot_file}")
    print(f"saved {zoom_plot_file}")
    print(f"saved {stats_file}")
    print(f"saved {maps_file}")
    print(
        peak_df[
            [
                "channel",
                "resolution",
                "noisy_twhiten_peak_time_s",
                "noisy_twhiten_peak_frequency_hz",
                "signal_twhiten_peak_time_s",
                "signal_twhiten_peak_frequency_hz",
                "time_offset_noisy_minus_signal_s",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
