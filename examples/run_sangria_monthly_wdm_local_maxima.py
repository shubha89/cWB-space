from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label, maximum_filter
from scipy.signal import resample_poly
from wdm_wavelet.wdm import WDM


SECONDS_PER_DAY = 86_400.0
MIN_NOISE_FLOOR = 1e-300


@dataclass(frozen=True)
class WDMResolution:
    label: str
    target_dt: float
    M: int


SMBHB_LIKE_RESOLUTIONS = (
    WDMResolution("dt5_M512", 5.0, 512),
    WDMResolution("dt5_M1024", 5.0, 1024),
    WDMResolution("dt5_M2048", 5.0, 2048),
    WDMResolution("dt5_M4096", 5.0, 4096),
    WDMResolution("dt10_M2048", 10.0, 2048),
    WDMResolution("dt10_M4096", 10.0, 4096),
    WDMResolution("dt10_M8192", 10.0, 8192),
)

# The SMBHB notebooks used a ladder like dt1_M128, dt1_M256, ..., dt10_M8192.
# Sangria is sampled at 5 s, so these are the native-cadence-compatible analogs.
NATIVE_5S_SMBHB_RESOLUTIONS = (
    WDMResolution("dt5_M128", 5.0, 128),
    WDMResolution("dt5_M256", 5.0, 256),
    WDMResolution("dt5_M512", 5.0, 512),
    WDMResolution("dt5_M1024", 5.0, 1024),
    WDMResolution("dt10_M2048", 10.0, 2048),
    WDMResolution("dt25_M4096", 25.0, 4096),
    WDMResolution("dt50_M8192", 50.0, 8192),
)

DEFAULT_RESOLUTIONS = NATIVE_5S_SMBHB_RESOLUTIONS


def xyz_to_aet(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = (z - x) / np.sqrt(2.0)
    e = (x - 2.0 * y + z) / np.sqrt(6.0)
    t = (x + y + z) / np.sqrt(3.0)
    return a, e, t


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def robust_limits(images: list[np.ndarray], low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    finite = [image[np.isfinite(image)] for image in images if np.any(np.isfinite(image))]
    if not finite:
        return -1.0, 1.0
    vals = np.concatenate(finite)
    vmin, vmax = np.nanpercentile(vals, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    return float(vmin), float(vmax)


def edge_taper(x: np.ndarray, dt: float, taper_seconds: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n_taper = int(round(taper_seconds / dt))
    if n_taper <= 1 or 2 * n_taper >= len(x):
        return x.copy()
    window = np.ones(len(x), dtype=float)
    phase = np.arange(n_taper, dtype=float) / n_taper
    ramp = 0.5 * (1.0 - np.cos(np.pi * phase))
    window[:n_taper] = ramp
    window[-n_taper:] = ramp[::-1]
    return x * window


def downsample_for_wdm(x: np.ndarray, native_dt: float, target_dt: float) -> tuple[np.ndarray, float]:
    factor = int(round(target_dt / native_dt))
    if factor < 1 or not np.isclose(factor * native_dt, target_dt):
        raise ValueError(f"target_dt={target_dt} must be an integer multiple of native dt={native_dt}.")
    if factor == 1:
        return np.asarray(x, dtype=float), native_dt
    return resample_poly(np.asarray(x, dtype=float), up=1, down=factor), factor * native_dt


def pad_for_wdm(x: np.ndarray, wdm: WDM) -> tuple[np.ndarray, int]:
    min_len = max(len(x), 2 * int(wdm.m_H) + 1)
    if len(x) >= min_len:
        return x, 0
    pad_right = min_len - len(x)
    return np.pad(x, (0, pad_right), mode="constant"), pad_right


def wdm_normalized_power(
    x: np.ndarray,
    *,
    native_dt: float,
    t0: float,
    t1: float,
    resolution: WDMResolution,
    taper_seconds: float,
    wdm_k: int,
    wdm_beta_order: int,
    wdm_precision: int,
    wdm_backend: str,
) -> dict[str, np.ndarray | float | int]:
    x_tapered = edge_taper(x, native_dt, taper_seconds)
    x_ds, dt_wdm = downsample_for_wdm(x_tapered, native_dt, resolution.target_dt)
    wdm = WDM(
        M=resolution.M,
        K=wdm_k,
        beta_order=wdm_beta_order,
        precision=wdm_precision,
        backend=wdm_backend,
    )
    x_pad, pad_right = pad_for_wdm(x_ds, wdm)
    tf = wdm.t2w(x_pad, sample_rate=1.0 / dt_wdm, t0=t0)
    power = np.abs(tf.data) ** 2
    freqs = np.arange(tf.n_freq, dtype=float) * tf.df
    times = tf.t0 + (np.arange(tf.n_time, dtype=float) + 0.5) * tf.dt
    valid = times <= t1
    power = power[:, valid]
    times = times[valid]
    floor = np.nanmedian(power, axis=1)
    floor = np.where(np.isfinite(floor) & (floor > 0.0), floor, MIN_NOISE_FLOOR)
    return {
        "power": power,
        "normalized_power": power / floor[:, None],
        "freqs_hz": freqs,
        "times_s": times,
        "tf_dt_s": float(tf.dt),
        "tf_df_hz": float(tf.df),
        "pad_right": int(pad_right),
    }


def local_maxima(
    normalized_power: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    low_freq: float,
    high_freq: float,
    percentile: float,
    max_peaks: int,
    min_time_separation_bins: int,
    min_freq_separation_bins: int,
    edge_guard_seconds: float,
) -> list[dict[str, float | int]]:
    fmask = (freqs >= low_freq) & (freqs <= high_freq)
    tmask = (times >= times[0] + edge_guard_seconds) & (times <= times[-1] - edge_guard_seconds)
    band = np.asarray(normalized_power[np.ix_(fmask, tmask)], dtype=float)
    if band.size == 0:
        return []

    finite = np.isfinite(band) & (band > 0.0)
    if not np.any(finite):
        return []

    threshold = float(np.nanpercentile(band[finite], percentile))
    filled = np.where(finite, band, -np.inf)
    footprint = (max(1, 2 * min_freq_separation_bins + 1), max(1, 2 * min_time_separation_bins + 1))
    local = filled == maximum_filter(filled, size=footprint, mode="nearest")
    candidates = np.argwhere(local & finite & (band >= threshold))
    if candidates.size == 0:
        return []

    values = band[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(values)[::-1]
    candidates = candidates[order]
    values = values[order]

    freq_indices = np.flatnonzero(fmask)
    time_indices = np.flatnonzero(tmask)
    peaks = []
    for (local_f, local_t), value in zip(candidates, values):
        f_idx = int(freq_indices[int(local_f)])
        t_idx = int(time_indices[int(local_t)])
        peaks.append(
            {
                "time_s": float(times[t_idx]),
                "frequency_hz": float(freqs[f_idx]),
                "power": float(value),
                "log10_power": float(np.log10(value)),
                "freq_index": f_idx,
                "time_index": t_idx,
                "threshold": threshold,
            }
        )
        if max_peaks > 0 and len(peaks) >= max_peaks:
            break
    return peaks


def high_energy_clusters(
    normalized_power: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    low_freq: float,
    high_freq: float,
    percentile: float,
    max_clusters: int,
    min_cluster_pixels: int,
    edge_guard_seconds: float,
) -> list[dict[str, float | int]]:
    fmask = (freqs >= low_freq) & (freqs <= high_freq)
    tmask = (times >= times[0] + edge_guard_seconds) & (times <= times[-1] - edge_guard_seconds)
    band = np.asarray(normalized_power[np.ix_(fmask, tmask)], dtype=float)
    if band.size == 0:
        return []

    finite = np.isfinite(band) & (band > 0.0)
    if not np.any(finite):
        return []

    threshold = float(np.nanpercentile(band[finite], percentile))
    hot = finite & (band >= threshold)
    if not np.any(hot):
        return []

    labels, n_labels = label(hot, structure=np.ones((3, 3), dtype=int))
    freq_indices = np.flatnonzero(fmask)
    time_indices = np.flatnonzero(tmask)
    clusters = []

    for cluster_id in range(1, n_labels + 1):
        coords = np.argwhere(labels == cluster_id)
        if coords.shape[0] < min_cluster_pixels:
            continue

        values = band[coords[:, 0], coords[:, 1]]
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue

        local_peak = int(np.nanargmax(values))
        local_f = int(coords[local_peak, 0])
        local_t = int(coords[local_peak, 1])
        f_idx = int(freq_indices[local_f])
        t_idx = int(time_indices[local_t])

        local_freqs = freqs[freq_indices[coords[:, 0]]]
        local_times = times[time_indices[coords[:, 1]]]
        excess = np.maximum(finite_values - threshold, 0.0)
        total_power = float(np.nansum(finite_values))
        total_excess_power = float(np.nansum(excess))
        max_power = float(values[local_peak])

        clusters.append(
            {
                "time_s": float(times[t_idx]),
                "frequency_hz": float(freqs[f_idx]),
                "power": max_power,
                "log10_power": float(np.log10(max_power)),
                "freq_index": f_idx,
                "time_index": t_idx,
                "threshold": threshold,
                "cluster_id": int(cluster_id),
                "cluster_pixels": int(coords.shape[0]),
                "cluster_total_power": total_power,
                "cluster_total_excess_power": total_excess_power,
                "cluster_mean_power": float(np.nanmean(finite_values)),
                "cluster_time_start_s": float(np.nanmin(local_times)),
                "cluster_time_stop_s": float(np.nanmax(local_times)),
                "cluster_frequency_low_hz": float(np.nanmin(local_freqs)),
                "cluster_frequency_high_hz": float(np.nanmax(local_freqs)),
            }
        )

    clusters.sort(
        key=lambda row: (
            float(row["cluster_total_excess_power"]),
            float(row["power"]),
            int(row["cluster_pixels"]),
        ),
        reverse=True,
    )
    if max_clusters > 0:
        clusters = clusters[:max_clusters]
    return clusters


def zoom_slice(size: int, center: int, half_width: int) -> slice:
    return slice(max(0, center - half_width), min(size, center + half_width + 1))


def top_energy_log_image(power: np.ndarray, top_fraction: float) -> tuple[np.ndarray, float]:
    finite = np.isfinite(power) & (power > 0.0)
    image = np.full(power.shape, np.nan, dtype=float)
    if not np.any(finite):
        return image, np.nan

    percentile = 100.0 * (1.0 - top_fraction)
    threshold = float(np.nanpercentile(power[finite], percentile))
    top_mask = finite & (power >= threshold)
    image[top_mask] = np.log10(power[top_mask])
    return image, threshold


def crop_around_time_frequency(
    normalized_power: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    time_s: float,
    frequency_hz: float,
    *,
    time_half_bins: int,
    freq_half_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    t_idx = int(np.nanargmin(np.abs(times - time_s)))
    f_idx = int(np.nanargmin(np.abs(freqs - frequency_hz)))
    ts = zoom_slice(len(times), t_idx, time_half_bins)
    fs = zoom_slice(len(freqs), f_idx, freq_half_bins)
    crop_times = times[ts]
    crop_freqs = freqs[fs]
    crop = normalized_power[fs, ts]
    positive = crop_freqs > 0.0
    crop_freqs = crop_freqs[positive]
    crop = crop[positive]
    return crop_times, crop_freqs, crop, t_idx, f_idx


def plot_peak_zoom(
    normalized_power: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    peak: dict[str, float | int],
    *,
    chunk_label: str,
    channel: str,
    resolution: WDMResolution,
    outfile: Path,
    time_half_bins: int,
    freq_half_bins: int,
    top_energy_fraction: float,
) -> None:
    crop_times, crop_freqs, crop, _, _ = crop_around_time_frequency(
        normalized_power,
        times,
        freqs,
        float(peak["time_s"]),
        float(peak["frequency_hz"]),
        time_half_bins=time_half_bins,
        freq_half_bins=freq_half_bins,
    )
    image, top_threshold = top_energy_log_image(crop, top_energy_fraction)
    vmin, vmax = robust_limits([image], low=1.0, high=99.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    im = ax.pcolormesh(
        centers_to_edges(crop_times),
        centers_to_edges(crop_freqs),
        image,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.scatter(
        float(peak["time_s"]),
        float(peak["frequency_hz"]),
        s=42,
        marker="o",
        facecolor="#00e5ff",
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(
        f"{chunk_label} {channel} {resolution.label} local maximum\n"
        f"t={float(peak['time_s']):.1f}s, f={float(peak['frequency_hz']):.4g}Hz, "
        f"log10 P={float(peak['log10_power']):.2f}, top {100 * top_energy_fraction:.0f}% pixels"
    )
    ax.text(
        0.02,
        0.95,
        f"crop threshold={top_threshold:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none", "pad": 2.0},
    )
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ chunk-median-normalized WDM power")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=170)
    plt.close(fig)


def plot_multires_peak_zoom(
    products: dict[str, dict[str, np.ndarray | float | int]],
    resolutions: tuple[WDMResolution, ...],
    peak: dict[str, float | int],
    *,
    chunk_label: str,
    channel: str,
    peak_resolution: WDMResolution,
    outfile: Path,
    time_half_bins: int,
    freq_half_bins: int,
    top_energy_fraction: float,
) -> None:
    prepared = []
    images = []
    peak_time = float(peak["time_s"])
    peak_freq = float(peak["frequency_hz"])

    for resolution in resolutions:
        product = products[resolution.label]
        normalized_power = np.asarray(product["normalized_power"])
        times = np.asarray(product["times_s"])
        freqs = np.asarray(product["freqs_hz"])
        crop_times, crop_freqs, crop, nearest_t_idx, nearest_f_idx = crop_around_time_frequency(
            normalized_power,
            times,
            freqs,
            peak_time,
            peak_freq,
            time_half_bins=time_half_bins,
            freq_half_bins=freq_half_bins,
        )
        image, threshold = top_energy_log_image(crop, top_energy_fraction)
        prepared.append(
            {
                "resolution": resolution,
                "times": crop_times,
                "freqs": crop_freqs,
                "image": image,
                "threshold": threshold,
                "nearest_time_s": float(times[nearest_t_idx]),
                "nearest_frequency_hz": float(freqs[nearest_f_idx]),
                "nearest_power": float(normalized_power[nearest_f_idx, nearest_t_idx]),
                "wdm_dt_s": float(product["tf_dt_s"]),
                "wdm_df_hz": float(product["tf_df_hz"]),
            }
        )
        images.append(image)

    vmin, vmax = robust_limits(images, low=1.0, high=99.0)
    fig, axes = plt.subplots(
        len(prepared),
        1,
        figsize=(7.4, max(1.45 * len(prepared), 6.5)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    last_im = None

    for ax, panel in zip(axes, prepared):
        resolution = panel["resolution"]
        last_im = ax.pcolormesh(
            centers_to_edges(panel["times"]),
            centers_to_edges(panel["freqs"]),
            panel["image"],
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.scatter(
            panel["nearest_time_s"],
            panel["nearest_frequency_hz"],
            s=28,
            marker="o",
            facecolor="#00e5ff",
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )
        ax.axvline(peak_time, color="white", lw=0.7, ls="--", alpha=0.75)
        ax.set_yscale("log")
        ax.set_ylabel(f"{resolution.label}\ndt={resolution.target_dt:g}s\nM={resolution.M}", fontsize=8)
        ax.text(
            0.02,
            0.93,
            f"thr={panel['threshold']:.3g}, nearest log10P={np.log10(max(panel['nearest_power'], MIN_NOISE_FLOOR)):.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none", "pad": 1.6},
        )

    axes[-1].set_xlabel("time [s]")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.9, extend="both")
        cbar.set_label(r"$\log_{10}$ normalized WDM power, local top pixels only")
    fig.suptitle(
        f"{chunk_label} {channel} multi-resolution zoom around "
        f"{peak_resolution.label} peak\n"
        f"t={peak_time:.1f}s, f={peak_freq:.4g}Hz; retaining top {100 * top_energy_fraction:.0f}% energy pixels per panel",
        fontsize=11,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=190)
    plt.close(fig)


def parse_resolutions(text: str) -> tuple[WDMResolution, ...]:
    if text.strip().lower() in {"default", "smbhb", "native5s-smbhb", ""}:
        return DEFAULT_RESOLUTIONS
    if text.strip().lower() in {"dense", "old-default"}:
        return SMBHB_LIKE_RESOLUTIONS
    parsed = []
    by_label = {resolution.label: resolution for resolution in (*DEFAULT_RESOLUTIONS, *SMBHB_LIKE_RESOLUTIONS)}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item in by_label:
            parsed.append(by_label[item])
            continue
        try:
            dt_part, m_part = item.split("_")
            target_dt = float(dt_part.removeprefix("dt").replace("p", "."))
            m = int(m_part.removeprefix("M"))
        except Exception as exc:
            raise ValueError(
                f"Could not parse resolution {item!r}. Use labels like dt5_M1024,dt10_M4096."
            ) from exc
        parsed.append(WDMResolution(item, target_dt, m))
    return tuple(parsed)


def chunk_ranges(
    n_samples: int,
    dt: float,
    chunk_days: float,
    max_chunks: int | None,
    start_chunk: int,
) -> list[tuple[int, int, int]]:
    chunk_samples = int(round(chunk_days * SECONDS_PER_DAY / dt))
    if chunk_samples <= 0:
        raise ValueError("chunk_days is too small for the data cadence.")
    if start_chunk < 1:
        raise ValueError("start_chunk is 1-based and must be >= 1.")
    ranges = []
    for chunk_number, start in enumerate(range(0, n_samples, chunk_samples), start=1):
        if chunk_number < start_chunk:
            continue
        stop = min(n_samples, start + chunk_samples)
        ranges.append((chunk_number, start, stop))
        if max_chunks is not None and len(ranges) >= max_chunks:
            break
    return ranges


def run(args: argparse.Namespace) -> None:
    input_hdf = Path(args.input_hdf)
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots" / "local_max_zoom"
    multires_plot_dir = output_dir / "plots" / "local_max_multires_zoom"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    multires_plot_dir.mkdir(parents=True, exist_ok=True)

    resolutions = parse_resolutions(args.resolutions)
    channels = tuple(ch.strip().upper() for ch in args.channels.split(",") if ch.strip())
    unsupported = set(channels) - {"A", "E", "T", "X", "Y", "Z"}
    if unsupported:
        raise ValueError(f"Unsupported channels: {sorted(unsupported)}")

    rows: list[dict[str, float | int | str]] = []
    with h5py.File(input_hdf, "r") as h5:
        dataset = h5["obs/tdi"]
        n_samples = int(dataset.shape[0])
        dt = float(dataset.attrs["dt"])
        ranges = chunk_ranges(n_samples, dt, args.chunk_days, args.max_chunks, args.start_chunk)

        for chunk_number, start, stop in ranges:
            arr = dataset[start:stop, 0]
            t = arr["t"].astype(float)
            if len(t) < 2:
                continue
            chunk_label = f"chunk{chunk_number:02d}_{t[0] / SECONDS_PER_DAY:.1f}d_{t[-1] / SECONDS_PER_DAY:.1f}d"
            print(f"{chunk_label}: samples={len(t)}, t={t[0]:.1f}-{t[-1]:.1f}s")

            x = arr["X"].astype(float)
            y = arr["Y"].astype(float)
            z = arr["Z"].astype(float)
            a, e, tt = xyz_to_aet(x, y, z)
            series = {"X": x, "Y": y, "Z": z, "A": a, "E": e, "T": tt}

            for channel in channels:
                channel_products: dict[str, dict[str, np.ndarray | float | int]] = {}
                for resolution in resolutions:
                    print(f"  {channel} {resolution.label}")
                    product = wdm_normalized_power(
                        series[channel],
                        native_dt=dt,
                        t0=float(t[0]),
                        t1=float(t[-1]),
                        resolution=resolution,
                        taper_seconds=args.taper_seconds,
                        wdm_k=args.wdm_k,
                        wdm_beta_order=args.wdm_beta_order,
                        wdm_precision=args.wdm_precision,
                        wdm_backend=args.wdm_backend,
                    )
                    channel_products[resolution.label] = product

                for resolution in resolutions:
                    product = channel_products[resolution.label]
                    normalized_power = np.asarray(product["normalized_power"])
                    times = np.asarray(product["times_s"])
                    freqs = np.asarray(product["freqs_hz"])
                    if args.detection_mode == "clusters":
                        detections = high_energy_clusters(
                            normalized_power,
                            times,
                            freqs,
                            low_freq=args.low_freq,
                            high_freq=args.high_freq,
                            percentile=args.peak_percentile,
                            max_clusters=args.top_peaks_per_map,
                            min_cluster_pixels=args.min_cluster_pixels,
                            edge_guard_seconds=args.edge_guard_seconds,
                        )
                    else:
                        detections = local_maxima(
                            normalized_power,
                            times,
                            freqs,
                            low_freq=args.low_freq,
                            high_freq=args.high_freq,
                            percentile=args.peak_percentile,
                            max_peaks=args.top_peaks_per_map,
                            min_time_separation_bins=args.min_time_separation_bins,
                            min_freq_separation_bins=args.min_freq_separation_bins,
                            edge_guard_seconds=args.edge_guard_seconds,
                        )
                    print(f"    {channel} {resolution.label}: {len(detections)} {args.detection_mode}")
                    for peak_rank, peak in enumerate(detections, start=1):
                        zoom_name = (
                            f"{chunk_label}_{channel}_{resolution.label}_"
                            f"peak{peak_rank:02d}_t{float(peak['time_s']):.0f}_"
                            f"f{float(peak['frequency_hz']):.6g}.png"
                        )
                        zoom_file = plot_dir / zoom_name
                        plot_peak_zoom(
                            normalized_power,
                            times,
                            freqs,
                            peak,
                            chunk_label=chunk_label,
                            channel=channel,
                            resolution=resolution,
                            outfile=zoom_file,
                            time_half_bins=args.zoom_time_half_bins,
                            freq_half_bins=args.zoom_freq_half_bins,
                            top_energy_fraction=args.zoom_top_energy_fraction,
                        )
                        multires_zoom_name = (
                            f"{chunk_label}_{channel}_{resolution.label}_"
                            f"peak{peak_rank:02d}_t{float(peak['time_s']):.0f}_"
                            f"f{float(peak['frequency_hz']):.6g}_multires_top10.png"
                        )
                        multires_zoom_file = multires_plot_dir / multires_zoom_name
                        plot_multires_peak_zoom(
                            channel_products,
                            resolutions,
                            peak,
                            chunk_label=chunk_label,
                            channel=channel,
                            peak_resolution=resolution,
                            outfile=multires_zoom_file,
                            time_half_bins=args.zoom_time_half_bins,
                            freq_half_bins=args.zoom_freq_half_bins,
                            top_energy_fraction=args.zoom_top_energy_fraction,
                        )
                        rows.append(
                            {
                                "chunk": chunk_number,
                                "chunk_start_index": start,
                                "chunk_stop_index": stop,
                                "chunk_start_s": float(t[0]),
                                "chunk_stop_s": float(t[-1]),
                                "channel": channel,
                                "resolution": resolution.label,
                                "target_dt_s": resolution.target_dt,
                                "wdm_M": resolution.M,
                                "wdm_dt_s": float(product["tf_dt_s"]),
                                "wdm_df_hz": float(product["tf_df_hz"]),
                                "rank": peak_rank,
                                "detection_mode": args.detection_mode,
                                "time_s": float(peak["time_s"]),
                                "frequency_hz": float(peak["frequency_hz"]),
                                "power": float(peak["power"]),
                                "log10_power": float(peak["log10_power"]),
                                "threshold": float(peak["threshold"]),
                                "cluster_id": int(peak.get("cluster_id", -1)),
                                "cluster_pixels": int(peak.get("cluster_pixels", 1)),
                                "cluster_total_power": float(peak.get("cluster_total_power", peak["power"])),
                                "cluster_total_excess_power": float(
                                    peak.get("cluster_total_excess_power", max(0.0, peak["power"] - peak["threshold"]))
                                ),
                                "cluster_mean_power": float(peak.get("cluster_mean_power", peak["power"])),
                                "cluster_time_start_s": float(peak.get("cluster_time_start_s", peak["time_s"])),
                                "cluster_time_stop_s": float(peak.get("cluster_time_stop_s", peak["time_s"])),
                                "cluster_frequency_low_hz": float(
                                    peak.get("cluster_frequency_low_hz", peak["frequency_hz"])
                                ),
                                "cluster_frequency_high_hz": float(
                                    peak.get("cluster_frequency_high_hz", peak["frequency_hz"])
                                ),
                                "zoom_plot": str(zoom_file),
                                "multires_zoom_plot": str(multires_zoom_file),
                            }
                        )

    peaks_df = pd.DataFrame(rows)
    peaks_csv = output_dir / "sangria_monthly_wdm_local_maxima.csv"
    peaks_df.to_csv(peaks_csv, index=False)
    config_json = output_dir / "sangria_monthly_wdm_local_maxima_config.json"
    config_json.write_text(
        json.dumps(
            {
                "input_hdf": str(input_hdf),
                "output_dir": str(output_dir),
                "channels": channels,
                "resolutions": [resolution.__dict__ for resolution in resolutions],
                "chunk_days": args.chunk_days,
                "low_freq": args.low_freq,
                "high_freq": args.high_freq,
                "peak_percentile": args.peak_percentile,
                "top_peaks_per_map": args.top_peaks_per_map,
                "detection_mode": args.detection_mode,
                "min_cluster_pixels": args.min_cluster_pixels,
                "zoom_top_energy_fraction": args.zoom_top_energy_fraction,
                "edge_guard_seconds": args.edge_guard_seconds,
                "taper_seconds": args.taper_seconds,
            },
            indent=2,
        )
    )
    print(f"saved {peaks_csv}")
    print(f"saved {config_json}")
    print(f"saved zoom plots under {plot_dir}")
    print(f"saved multi-resolution zoom plots under {multires_plot_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-hdf", default="data/LDC2_sangria_blind_v2.h5")
    parser.add_argument("--output-dir", default="data/sangria_wdm_monthly")
    parser.add_argument("--chunk-days", type=float, default=30.0)
    parser.add_argument("--start-chunk", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--channels", default="X,Y,Z")
    parser.add_argument("--resolutions", default="native5s-smbhb")
    parser.add_argument("--low-freq", type=float, default=1e-5)
    parser.add_argument("--high-freq", type=float, default=2e-2)
    parser.add_argument("--peak-percentile", type=float, default=99.9)
    parser.add_argument("--top-peaks-per-map", type=int, default=5)
    parser.add_argument("--detection-mode", choices=("clusters", "peaks"), default="clusters")
    parser.add_argument("--min-cluster-pixels", type=int, default=3)
    parser.add_argument("--min-time-separation-bins", type=int, default=3)
    parser.add_argument("--min-freq-separation-bins", type=int, default=3)
    parser.add_argument("--edge-guard-seconds", type=float, default=43_200.0)
    parser.add_argument("--taper-seconds", type=float, default=43_200.0)
    parser.add_argument("--zoom-time-half-bins", type=int, default=10)
    parser.add_argument("--zoom-freq-half-bins", type=int, default=8)
    parser.add_argument("--zoom-top-energy-fraction", type=float, default=0.10)
    parser.add_argument("--wdm-k", type=int, default=32)
    parser.add_argument("--wdm-beta-order", type=int, default=6)
    parser.add_argument("--wdm-precision", type=int, default=6)
    parser.add_argument("--wdm-backend", default="numba")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
