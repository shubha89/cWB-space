from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.run_mbhb_wdm_reconstruction import (
    DAY,
    AE_CHANNELS,
    reconstruction_metrics,
    scalar_projection_scale,
    wdm_inverse_quadrature_mean,
)
from scripts.blind_event_clustering import SeededChirpEventConfig, find_seeded_chirp_events
from scripts.clean_wdm import require_wdm_dependencies


RUN_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_observed_cluster_wdm_reconstruction"
OUTPUT_DIR = RUN_DIR / "outputs"
TWO_WEEK_OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "simulated_2week_mbhb_then_glitch_highrate" / "outputs"
PRODUCTS_NPZ = TWO_WEEK_OUTPUT_DIR / "two_week_mbhb_then_glitch_tdi.npz"
MBHB_CATALOG_CSV = TWO_WEEK_OUTPUT_DIR / "mbhb_catalog.csv"

LOW_HZ = 1.0e-5
HIGH_HZ = 1.0e-1
PEAK_ZOOM_HALF_WINDOW_DAYS = 0.008


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Build seeded noisy-observed A/E WDM cluster masks around the MBHB injections, "
            "reconstruct masked observed A/E time series, and match them to source-only MBHBs."
        )
    )
    parser.add_argument("--products", type=Path, default=PRODUCTS_NPZ)
    parser.add_argument("--catalog", type=Path, default=MBHB_CATALOG_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--beta-order", type=int, default=6)
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--backend", default="numba", choices=("numba", "jax"))
    parser.add_argument("--low-hz", type=float, default=LOW_HZ)
    parser.add_argument("--high-hz", type=float, default=HIGH_HZ)
    parser.add_argument("--seed-quantile", type=float, default=0.99)
    parser.add_argument("--grow-quantile", type=float, default=0.95)
    parser.add_argument("--max-freq-jump-bins", type=int, default=4)
    parser.add_argument("--same-time-connect-bins", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=8)
    parser.add_argument("--min-time-bins", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--search-half-window-days", type=float, default=0.75)
    parser.add_argument("--local-recovery-half-window-days", type=float, default=0.35)
    parser.add_argument(
        "--peak-zoom-half-window-days",
        type=float,
        default=PEAK_ZOOM_HALF_WINDOW_DAYS,
        help="Half-width of the high-amplitude time-domain zoom panels.",
    )
    parser.add_argument(
        "--dilate-freq-bins",
        type=int,
        default=1,
        help="Frequency-bin half-width for the additional dilated reconstruction mask.",
    )
    parser.add_argument(
        "--dilate-time-bins",
        type=int,
        default=1,
        help="Time-bin half-width for the additional dilated reconstruction mask.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def ensure_inputs(args: Namespace) -> None:
    missing = [path for path in (args.products, args.catalog) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(str(path) for path in missing))
    if not (0.0 <= args.low_hz < args.high_hz):
        raise ValueError("Require 0 <= --low-hz < --high-hz.")
    if not (0.0 < args.seed_quantile < 1.0) or not (0.0 < args.grow_quantile < 1.0):
        raise ValueError("Seed and grow quantiles must be in (0, 1).")
    if args.grow_quantile > args.seed_quantile:
        raise ValueError("--grow-quantile must be <= --seed-quantile.")
    if args.peak_zoom_half_window_days <= 0.0:
        raise ValueError("--peak-zoom-half-window-days must be positive.")
    if args.dilate_freq_bins < 0 or args.dilate_time_bins < 0:
        raise ValueError("Mask dilation bin counts must be non-negative.")


def clean_floor(floor: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(floor) & (floor > 0.0), floor, 1.0e-300)


def compute_raw_tf_maps(wdm, t_lisa: np.ndarray, channels: dict[str, np.ndarray]) -> dict[str, object]:
    native_dt = float(np.median(np.diff(t_lisa)))
    sample_rate = 1.0 / native_dt
    tf_maps = {}
    for channel, values in channels.items():
        tf_maps[channel] = wdm.t2w(np.asarray(values, dtype=float), sample_rate=sample_rate, t0=float(t_lisa[0]))
    return tf_maps


def build_cluster_maps(t_lisa: np.ndarray, raw_tf_maps: dict[str, object]) -> dict[str, np.ndarray]:
    a_coeff_raw = np.asarray(raw_tf_maps["A2"].data, dtype=np.complex128)
    e_coeff_raw = np.asarray(raw_tf_maps["E2"].data, dtype=np.complex128)
    if a_coeff_raw.shape != e_coeff_raw.shape:
        raise ValueError("Observed A/E WDM maps are not shape-aligned.")
    a_floor = clean_floor(np.nanmedian(np.abs(a_coeff_raw) ** 2, axis=1))
    e_floor = clean_floor(np.nanmedian(np.abs(e_coeff_raw) ** 2, axis=1))
    a_coeff = a_coeff_raw / np.sqrt(a_floor[:, None])
    e_coeff = e_coeff_raw / np.sqrt(e_floor[:, None])
    tf_ref = raw_tf_maps["A2"]
    times_rel_s = (np.arange(tf_ref.n_time, dtype=float) + 0.5) * float(tf_ref.dt)
    freqs_hz = np.arange(tf_ref.n_freq, dtype=float) * float(tf_ref.df)
    return {
        "times_rel_s": times_rel_s,
        "times_day": times_rel_s / DAY,
        "freqs_hz": freqs_hz,
        "A_coeff": a_coeff,
        "E_coeff": e_coeff,
        "A_power": np.abs(a_coeff) ** 2,
        "E_power": np.abs(e_coeff) ** 2,
        "AE_power": np.abs(a_coeff) ** 2 + np.abs(e_coeff) ** 2,
        "A_floor": a_floor,
        "E_floor": e_floor,
        "wdm_df_hz": np.asarray(float(tf_ref.df)),
        "wdm_dt_s": np.asarray(float(tf_ref.dt)),
        "native_dt_s": np.asarray(float(np.median(np.diff(t_lisa)))),
    }


def source_label_from_catalog_index(index: int) -> str:
    return f"mbhb{index + 1}"


def case_windows(case_row: pd.Series, args: Namespace) -> dict[str, float]:
    target_day = float(case_row["response_peak_day"])
    return {
        "target_day": target_day,
        "search_start_day": max(0.0, target_day - float(args.search_half_window_days)),
        "search_stop_day": min(7.0, target_day + float(args.search_half_window_days)),
        "catalog_start_day": float(case_row["injection_start_rel_s"]) / DAY,
        "catalog_stop_day": float(case_row["injection_stop_rel_s"]) / DAY,
        "local_start_day": max(0.0, target_day - float(args.local_recovery_half_window_days)),
        "local_stop_day": min(7.0, target_day + float(args.local_recovery_half_window_days)),
    }


def annotate_events(events: pd.DataFrame, *, source_label: str, case_row: pd.Series, windows: dict[str, float]) -> pd.DataFrame:
    out = events.copy()
    out["source_label"] = source_label
    out["case_id"] = str(case_row["case_id"])
    out["target_day"] = windows["target_day"]
    out["target_time_error_days"] = out["peak_time_day"] - windows["target_day"]
    out["catalog_start_day"] = windows["catalog_start_day"]
    out["catalog_stop_day"] = windows["catalog_stop_day"]
    out["local_recovery_start_day"] = windows["local_start_day"]
    out["local_recovery_stop_day"] = windows["local_stop_day"]
    out["overlaps_catalog_time"] = (out["stop_day"] >= windows["catalog_start_day"]) & (
        out["start_day"] <= windows["catalog_stop_day"]
    )
    out["peak_in_local_recovery_window"] = (out["peak_time_day"] >= windows["local_start_day"]) & (
        out["peak_time_day"] <= windows["local_stop_day"]
    )
    return out


def select_event(events: pd.DataFrame) -> pd.Series:
    for column in ("peak_in_local_recovery_window", "overlaps_catalog_time"):
        eligible = events[events[column].astype(bool)]
        if not eligible.empty:
            return eligible.sort_values("event_statistic", ascending=False).iloc[0]
    return events.sort_values("event_statistic", ascending=False).iloc[0]


def dilate_mask(mask: np.ndarray, *, freq_bins: int, time_bins: int) -> np.ndarray:
    if freq_bins == 0 and time_bins == 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    n_freq, n_time = mask.shape
    for df in range(-freq_bins, freq_bins + 1):
        src_f0 = max(0, -df)
        src_f1 = min(n_freq, n_freq - df)
        dst_f0 = max(0, df)
        dst_f1 = min(n_freq, n_freq + df)
        for dt in range(-time_bins, time_bins + 1):
            src_t0 = max(0, -dt)
            src_t1 = min(n_time, n_time - dt)
            dst_t0 = max(0, dt)
            dst_t1 = min(n_time, n_time + dt)
            out[dst_f0:dst_f1, dst_t0:dst_t1] |= mask[src_f0:src_f1, src_t0:src_t1]
    return out


def inverse_masked_tf(wdm, tf_map, mask: np.ndarray, n_samples: int) -> np.ndarray:
    masked_tf = replace(tf_map, data=np.where(mask, tf_map.data, 0.0 + 0.0j))
    return wdm_inverse_quadrature_mean(wdm, masked_tf, n_samples)


def add_metrics_with_prefix(row: dict[str, object], prefix: str, reference: np.ndarray, reconstruction: np.ndarray) -> None:
    metrics = reconstruction_metrics(reference, reconstruction)
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value
    scale = scalar_projection_scale(reference, reconstruction)
    row[f"{prefix}_fit_scale"] = scale
    fit_metrics = reconstruction_metrics(reference, scale * reconstruction)
    for key, value in fit_metrics.items():
        row[f"{prefix}_fit_{key}"] = value


def analyze_mask_variant(
    *,
    wdm,
    source_label: str,
    case_id: str,
    channel: str,
    mask_label: str,
    mask: np.ndarray,
    source_series: np.ndarray,
    observed_tf_map,
    source_tf_map,
    noise_tf_map,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    n_samples = len(source_series)
    observed_reconstruction = inverse_masked_tf(wdm, observed_tf_map, mask, n_samples)
    source_mask_reconstruction = inverse_masked_tf(wdm, source_tf_map, mask, n_samples)
    noise_mask_reconstruction = inverse_masked_tf(wdm, noise_tf_map, mask, n_samples)

    source_tf_power = np.abs(np.asarray(source_tf_map.data, dtype=np.complex128)) ** 2
    observed_tf_power = np.abs(np.asarray(observed_tf_map.data, dtype=np.complex128)) ** 2
    noise_tf_power = np.abs(np.asarray(noise_tf_map.data, dtype=np.complex128)) ** 2
    source_norm = float(np.linalg.norm(source_series))
    observed_norm = float(np.linalg.norm(observed_reconstruction))
    source_mask_norm = float(np.linalg.norm(source_mask_reconstruction))
    noise_mask_norm = float(np.linalg.norm(noise_mask_reconstruction))

    row: dict[str, object] = {
        "source_label": source_label,
        "case_id": case_id,
        "channel": channel,
        "mask_label": mask_label,
        "n_mask_pixels": int(np.sum(mask)),
        "mask_time_bins": int(np.sum(np.any(mask, axis=0))),
        "mask_freq_bins": int(np.sum(np.any(mask, axis=1))),
        "selected_source_tf_power_fraction": float(np.sum(source_tf_power[mask]) / np.sum(source_tf_power)),
        "selected_observed_tf_power_fraction": float(np.sum(observed_tf_power[mask]) / np.sum(observed_tf_power)),
        "selected_noise_tf_power_fraction": float(np.sum(noise_tf_power[mask]) / np.sum(noise_tf_power)),
        "source_norm": source_norm,
        "observed_cluster_norm": observed_norm,
        "source_mask_norm": source_mask_norm,
        "noise_mask_norm": noise_mask_norm,
        "observed_cluster_norm_over_source": observed_norm / source_norm if source_norm > 0.0 else np.nan,
        "source_mask_norm_over_source": source_mask_norm / source_norm if source_norm > 0.0 else np.nan,
        "noise_mask_norm_over_source": noise_mask_norm / source_norm if source_norm > 0.0 else np.nan,
        "source_mask_noise_norm_ratio": noise_mask_norm / source_mask_norm if source_mask_norm > 0.0 else np.nan,
    }
    add_metrics_with_prefix(row, "observed_cluster", source_series, observed_reconstruction)
    add_metrics_with_prefix(row, "source_mask", source_series, source_mask_reconstruction)
    add_metrics_with_prefix(row, "noise_mask", source_series, noise_mask_reconstruction)

    observed_fit_scale = float(row["observed_cluster_fit_scale"])
    series_payload = {
        f"{source_label}_{channel}_{mask_label}_observed_cluster": observed_reconstruction.astype(np.float64),
        f"{source_label}_{channel}_{mask_label}_observed_cluster_fit": (
            observed_fit_scale * observed_reconstruction
        ).astype(np.float64),
        f"{source_label}_{channel}_{mask_label}_source_mask": source_mask_reconstruction.astype(np.float64),
        f"{source_label}_{channel}_{mask_label}_noise_mask": noise_mask_reconstruction.astype(np.float64),
    }
    return row, series_payload


def compute_ae_combined_metrics(channel_metrics: pd.DataFrame, payload: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (source_label, mask_label), group in channel_metrics.groupby(["source_label", "mask_label"], sort=False):
        case_id = str(group.iloc[0]["case_id"])
        source = np.concatenate([payload[f"{source_label}_{channel}_source"] for channel in AE_CHANNELS])
        reconstructions = {
            "observed_cluster": np.concatenate(
                [payload[f"{source_label}_{channel}_{mask_label}_observed_cluster"] for channel in AE_CHANNELS]
            ),
            "observed_cluster_fit": np.concatenate(
                [payload[f"{source_label}_{channel}_{mask_label}_observed_cluster_fit"] for channel in AE_CHANNELS]
            ),
            "source_mask": np.concatenate(
                [payload[f"{source_label}_{channel}_{mask_label}_source_mask"] for channel in AE_CHANNELS]
            ),
            "noise_mask": np.concatenate(
                [payload[f"{source_label}_{channel}_{mask_label}_noise_mask"] for channel in AE_CHANNELS]
            ),
        }
        for reconstruction_label, reconstruction in reconstructions.items():
            metrics = reconstruction_metrics(source, reconstruction)
            scale = scalar_projection_scale(source, reconstruction)
            fit_metrics = reconstruction_metrics(source, scale * reconstruction)
            row = {
                "source_label": str(source_label),
                "case_id": case_id,
                "channel_combination": "AE",
                "mask_label": str(mask_label),
                "reconstruction": reconstruction_label,
                "fit_scale": scale,
                "norm_over_source": float(np.linalg.norm(reconstruction) / np.linalg.norm(source)),
                **metrics,
            }
            for key, value in fit_metrics.items():
                row[f"fit_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def plot_cluster_masks(
    output: Path,
    *,
    maps: dict[str, np.ndarray],
    selected_events: pd.DataFrame,
    masks: dict[str, np.ndarray],
    args: Namespace,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(maps["times_day"], dtype=float)
    freqs = np.asarray(maps["freqs_hz"], dtype=float)
    score = np.asarray(maps["AE_power"], dtype=float)
    band = (freqs >= float(args.low_hz)) & (freqs <= float(args.high_hz))

    fig, axes = plt.subplots(len(selected_events), 1, figsize=(12.0, 3.4 * len(selected_events)), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, selected_events.itertuples(index=False)):
        time = (times >= row.search_start_day) & (times <= row.search_stop_day)
        values = np.log10(np.maximum(score[np.ix_(band, time)], 1.0e-300))
        finite = values[np.isfinite(values)]
        vmin, vmax = np.nanpercentile(finite, [2.0, 99.7])
        image = ax.pcolormesh(times[time], freqs[band], values, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        mask = masks[f"{row.source_label}_cluster"]
        local_mask = mask[np.ix_(band, time)]
        fi, ti = np.nonzero(local_mask)
        ax.scatter(times[time][ti], freqs[band][fi], s=6, marker="s", color="#00ffff", alpha=0.82, linewidths=0)
        ax.axvline(row.target_day, color="#00ffff", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylim(float(args.low_hz), float(args.high_hz))
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(f"{row.source_label}: selected noisy-observed cluster, match {row.ae_observed_cluster_match:.6f}")
        ax.grid(True, which="both", alpha=0.14)
        cbar = fig.colorbar(image, ax=ax, pad=0.01)
        cbar.set_label(r"$\log_{10}$ self-AE median power")
    axes[-1].set_xlabel("Time since product start [days]")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def peak_center_day(
    *,
    t_rel_s: np.ndarray,
    selected,
    source_label: str,
    payload: dict[str, np.ndarray],
) -> float:
    target_s = float(selected.target_day) * DAY
    search_half_width_s = max(0.12 * DAY, 6.0 * float(selected.wdm_dt_s))
    search_window = (t_rel_s >= target_s - search_half_width_s) & (t_rel_s <= target_s + search_half_width_s)
    indices = np.flatnonzero(search_window)
    if len(indices) == 0:
        return float(selected.target_day)

    ae_amplitude2 = np.zeros(len(indices), dtype=float)
    for channel in AE_CHANNELS:
        source = np.asarray(payload[f"{source_label}_{channel}_source"], dtype=float)
        ae_amplitude2 += source[indices] ** 2
    peak_index = int(indices[int(np.argmax(ae_amplitude2))])
    return float(t_rel_s[peak_index] / DAY)


def time_window(t_rel_s: np.ndarray, *, center_day: float, half_width_s: float):
    window = (t_rel_s >= center_day * DAY - half_width_s) & (t_rel_s <= center_day * DAY + half_width_s)
    if int(np.sum(window)) < 8:
        return slice(None)
    return window


def draw_time_domain_overlay(
    ax,
    *,
    days: np.ndarray,
    source: np.ndarray,
    observed_fit: np.ndarray,
    source_mask: np.ndarray,
) -> None:
    ax.plot(
        days,
        source_mask,
        color="#2563eb",
        lw=0.9,
        ls=(0, (1.0, 2.0)),
        alpha=0.32,
        zorder=1,
        label="source through same mask",
    )
    ax.plot(
        days,
        source,
        color="#111827",
        lw=2.2,
        alpha=0.72,
        zorder=2,
        label="source-only MBHB",
    )
    ax.plot(
        days,
        observed_fit,
        color="#dc2626",
        lw=1.15,
        ls=(0, (4.0, 2.0)),
        alpha=0.95,
        zorder=3,
        label="observed cluster, fit scale",
    )


def plot_time_domain_overlays(
    output_dir: Path,
    *,
    t_rel_s: np.ndarray,
    selected_events: pd.DataFrame,
    channel_metrics: pd.DataFrame,
    payload: dict[str, np.ndarray],
    peak_zoom_half_window_days: float,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}
    for selected in selected_events.itertuples(index=False):
        source_label = str(selected.source_label)
        output = output_dir / f"{source_label}_observed_cluster_reconstruction_overlay.png"
        fig, axes = plt.subplots(2, 2, figsize=(15.0, 7.4), sharex="col", constrained_layout=True)
        full_half_width_s = max(0.35 * DAY, 8.0 * float(selected.wdm_dt_s))
        full_window = time_window(t_rel_s, center_day=float(selected.target_day), half_width_s=full_half_width_s)
        peak_day = peak_center_day(
            t_rel_s=t_rel_s,
            selected=selected,
            source_label=source_label,
            payload=payload,
        )
        peak_half_width_s = float(peak_zoom_half_window_days) * DAY
        peak_window = time_window(t_rel_s, center_day=peak_day, half_width_s=peak_half_width_s)
        rows = channel_metrics[
            (channel_metrics["source_label"] == source_label) & (channel_metrics["mask_label"] == "cluster")
        ].set_index("channel")
        for row_index, channel in enumerate(AE_CHANNELS):
            source = payload[f"{source_label}_{channel}_source"]
            observed_fit = payload[f"{source_label}_{channel}_cluster_observed_cluster_fit"]
            source_mask = payload[f"{source_label}_{channel}_cluster_source_mask"]
            full_days = t_rel_s[full_window] / DAY
            peak_days = t_rel_s[peak_window] / DAY
            full_ax = axes[row_index, 0]
            peak_ax = axes[row_index, 1]
            draw_time_domain_overlay(
                full_ax,
                days=full_days,
                source=source[full_window],
                observed_fit=observed_fit[full_window],
                source_mask=source_mask[full_window],
            )
            draw_time_domain_overlay(
                peak_ax,
                days=peak_days,
                source=source[peak_window],
                observed_fit=observed_fit[peak_window],
                source_mask=source_mask[peak_window],
            )
            full_ax.axvline(float(selected.target_day), color="#0f766e", lw=0.9, alpha=0.75)
            peak_ax.axvline(peak_day, color="#0f766e", lw=0.9, alpha=0.75)
            row = rows.loc[channel]
            text = (
                f"obs match={row['observed_cluster_match']:.6f}, "
                f"source-mask match={row['source_mask_match']:.6f}"
            )
            full_ax.text(0.015, 0.78, text, transform=full_ax.transAxes, ha="left", va="top", fontsize=9)
            full_ax.set_title(f"{channel} full event window")
            peak_ax.set_title(f"{channel} high-amplitude zoom, +/- {peak_zoom_half_window_days * 24.0 * 60.0:.1f} min")
            for ax in (full_ax, peak_ax):
                ax.set_ylabel(f"{channel} [Hz]")
                ax.grid(True, alpha=0.22)
        axes[-1, 0].set_xlabel("Time since product start [days]")
        axes[-1, 1].set_xlabel("Time since product start [days]")
        axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
        fig.suptitle(f"Noisy observed cluster reconstruction: {selected.case_id}")
        fig.savefig(output, dpi=180)
        plt.close(fig)
        plots[source_label] = str(output)
    return plots


def plot_combined_time_domain_overlaps(
    output: Path,
    *,
    t_rel_s: np.ndarray,
    selected_events: pd.DataFrame,
    channel_metrics: pd.DataFrame,
    payload: dict[str, np.ndarray],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(selected_events)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14.5, 3.25 * n_rows), squeeze=False)

    for row_index, selected in enumerate(selected_events.itertuples(index=False)):
        source_label = str(selected.source_label)
        half_width_s = max(0.35 * DAY, 8.0 * float(selected.wdm_dt_s))
        window = time_window(t_rel_s, center_day=float(selected.target_day), half_width_s=half_width_s)
        days = t_rel_s[window] / DAY
        rows = channel_metrics[
            (channel_metrics["source_label"] == source_label) & (channel_metrics["mask_label"] == "cluster")
        ].set_index("channel")

        for column_index, channel in enumerate(AE_CHANNELS):
            ax = axes[row_index, column_index]
            source = payload[f"{source_label}_{channel}_source"]
            observed_fit = payload[f"{source_label}_{channel}_cluster_observed_cluster_fit"]
            source_mask = payload[f"{source_label}_{channel}_cluster_source_mask"]
            draw_time_domain_overlay(
                ax,
                days=days,
                source=source[window],
                observed_fit=observed_fit[window],
                source_mask=source_mask[window],
            )
            ax.axvline(float(selected.target_day), color="#0f766e", lw=0.9, alpha=0.75)
            metric_row = rows.loc[channel]
            ax.set_title(
                f"{source_label} {channel}: obs match={metric_row['observed_cluster_match']:.6f}, "
                f"source-mask={metric_row['source_mask_match']:.6f}",
                fontsize=10,
            )
            ax.set_ylabel(f"{channel} [Hz]")
            ax.grid(True, alpha=0.22)
            if row_index == 0 and column_index == 0:
                ax.legend(loc="upper left", frameon=False, fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time since product start [days]")
    fig.suptitle("Noisy observed WDM cluster time-domain overlaps", fontsize=15)
    fig.subplots_adjust(top=0.92, hspace=0.43, wspace=0.24)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_peak_zoom_time_domain_overlaps(
    output: Path,
    *,
    t_rel_s: np.ndarray,
    selected_events: pd.DataFrame,
    channel_metrics: pd.DataFrame,
    payload: dict[str, np.ndarray],
    peak_zoom_half_window_days: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(selected_events)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14.5, 3.25 * n_rows), squeeze=False)
    peak_half_width_s = float(peak_zoom_half_window_days) * DAY

    for row_index, selected in enumerate(selected_events.itertuples(index=False)):
        source_label = str(selected.source_label)
        peak_day = peak_center_day(
            t_rel_s=t_rel_s,
            selected=selected,
            source_label=source_label,
            payload=payload,
        )
        window = time_window(t_rel_s, center_day=peak_day, half_width_s=peak_half_width_s)
        days = t_rel_s[window] / DAY
        rows = channel_metrics[
            (channel_metrics["source_label"] == source_label) & (channel_metrics["mask_label"] == "cluster")
        ].set_index("channel")

        for column_index, channel in enumerate(AE_CHANNELS):
            ax = axes[row_index, column_index]
            source = payload[f"{source_label}_{channel}_source"]
            observed_fit = payload[f"{source_label}_{channel}_cluster_observed_cluster_fit"]
            source_mask = payload[f"{source_label}_{channel}_cluster_source_mask"]
            draw_time_domain_overlay(
                ax,
                days=days,
                source=source[window],
                observed_fit=observed_fit[window],
                source_mask=source_mask[window],
            )
            ax.axvline(peak_day, color="#0f766e", lw=0.9, alpha=0.75)
            metric_row = rows.loc[channel]
            ax.set_title(
                f"{source_label} {channel}: peak zoom, obs match={metric_row['observed_cluster_match']:.6f}",
                fontsize=10,
            )
            ax.set_ylabel(f"{channel} [Hz]")
            ax.grid(True, alpha=0.22)
            if row_index == 0 and column_index == 0:
                ax.legend(loc="upper left", frameon=False, fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time since product start [days]")
    fig.suptitle(
        f"Noisy observed WDM cluster high-amplitude zoom (+/- {peak_zoom_half_window_days * 24.0 * 60.0:.1f} min)",
        fontsize=15,
    )
    fig.subplots_adjust(top=0.92, hspace=0.43, wspace=0.24)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    args: Namespace,
    selected_events: pd.DataFrame,
    ae_metrics: pd.DataFrame,
    plots: dict[str, str],
) -> None:
    observed_cluster = ae_metrics[
        (ae_metrics["mask_label"] == "cluster") & (ae_metrics["reconstruction"] == "observed_cluster")
    ]
    observed_dilated = ae_metrics[
        (ae_metrics["mask_label"] == "cluster_dilated") & (ae_metrics["reconstruction"] == "observed_cluster")
    ]
    lines = [
        "# Noisy Observed Cluster WDM Reconstruction",
        "",
        "## Method",
        "",
        "- Build raw observed A/E WDM maps from `observed_A2` and `observed_E2`.",
        "- Normalize each channel by its per-frequency median observed WDM power for clustering only.",
        "- Find seeded connected chirp-like clusters in known MBHB search windows.",
        "- Apply the selected noisy-observed cluster mask to raw observed A/E coefficients and invert with `0.5*(w2t+w2tQ)`.",
        "- Compare the reconstructed A/E time series with the source-only MBHB injection using a zero-lag normalized time-domain match.",
        "",
        "## Configuration",
        "",
        f"- WDM: `M={args.M}`, `K={args.K}`, `beta_order={args.beta_order}`, `precision={args.precision}`, `backend={args.backend}`",
        f"- Cluster band: `{args.low_hz:g}` to `{args.high_hz:g}` Hz",
        f"- Seed/grow quantiles: `{args.seed_quantile:g}` / `{args.grow_quantile:g}`",
        f"- Search half-window: `{args.search_half_window_days:g}` days",
        f"- Peak zoom half-window: `{args.peak_zoom_half_window_days:g}` days",
        f"- Additional dilation variant: `freq={args.dilate_freq_bins}` bins, `time={args.dilate_time_bins}` bins",
        "",
        "## Combined AE Matches",
        "",
        "Strict selected cluster:",
        "```text",
        observed_cluster[
            ["source_label", "match", "mismatch", "fit_rel_l2", "norm_over_source"]
        ].to_string(index=False),
        "```",
        "",
        "Dilated cluster sensitivity check:",
        "```text",
        observed_dilated[
            ["source_label", "match", "mismatch", "fit_rel_l2", "norm_over_source"]
        ].to_string(index=False),
        "```",
        "",
        "Selected events:",
        "```text",
        selected_events[
            [
                "source_label",
                "peak_time_day",
                "peak_freq_hz",
                "n_pixels",
                "n_time_bins",
                "freq_min_hz",
                "freq_max_hz",
                "ae_balance",
                "monotonic_fraction",
                "event_statistic",
                "selection_reason",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Outputs",
        "",
        f"- Candidate clusters: `{args.output_dir / 'observed_cluster_candidates.csv'}`",
        f"- Selected clusters: `{args.output_dir / 'selected_observed_cluster_events.csv'}`",
        f"- Per-channel reconstruction metrics: `{args.output_dir / 'observed_cluster_reconstruction_metrics.csv'}`",
        f"- Combined AE matches: `{args.output_dir / 'observed_cluster_ae_combined_matches.csv'}`",
        f"- Masks/maps: `{args.output_dir / 'observed_cluster_masks_and_maps.npz'}`",
        f"- Reconstructed series: `{args.output_dir / 'observed_cluster_reconstructed_ae_timeseries.npz'}`",
    ]
    for label, plot in plots.items():
        lines.append(f"- Plot `{label}`: `{plot}`")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- These matches are not PSD-weighted and are not maximized over time or phase.",
            "- The cluster search uses known MBHB time windows for this injection-recovery diagnostic; it is not a fully blind all-data search.",
            "- The reconstructed observed cluster contains selected noise as well as signal. The `source_mask` and `noise_mask` columns quantify the same mask applied to source-only and noise-only WDM maps.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ensure_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _, WDM = require_wdm_dependencies()
    wdm = WDM(M=args.M, K=args.K, beta_order=args.beta_order, precision=args.precision, backend=args.backend)
    cluster_config = SeededChirpEventConfig(
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        seed_quantile=args.seed_quantile,
        grow_quantile=args.grow_quantile,
        max_freq_jump_bins=args.max_freq_jump_bins,
        same_time_connect_bins=args.same_time_connect_bins,
        min_pixels=args.min_pixels,
        min_time_bins=args.min_time_bins,
        max_events=args.max_events,
    )

    catalog = pd.read_csv(args.catalog).reset_index(drop=True)
    if args.max_cases is not None:
        catalog = catalog.head(args.max_cases)

    candidate_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    mask_payload: dict[str, np.ndarray] = {}
    series_payload: dict[str, np.ndarray] = {}

    with np.load(args.products) as products:
        t_lisa = np.asarray(products["t_lisa"], dtype=float)
        t_rel_s = np.asarray(products["time_rel_s"], dtype=float)
        series_payload["t_lisa"] = t_lisa.astype(np.float64)
        series_payload["time_rel_s"] = t_rel_s.astype(np.float64)
        series_payload["time_day"] = np.asarray(products["time_day"], dtype=float).astype(np.float64)

        observed_tf_maps = compute_raw_tf_maps(
            wdm,
            t_lisa,
            {"A2": np.asarray(products["observed_A2"], dtype=float), "E2": np.asarray(products["observed_E2"], dtype=float)},
        )
        noise_tf_maps = compute_raw_tf_maps(
            wdm,
            t_lisa,
            {"A2": np.asarray(products["noise_A2"], dtype=float), "E2": np.asarray(products["noise_E2"], dtype=float)},
        )
        cluster_maps = build_cluster_maps(t_lisa, observed_tf_maps)
        mask_payload.update(
            {
                "times_day": np.asarray(cluster_maps["times_day"], dtype=float),
                "times_rel_s": np.asarray(cluster_maps["times_rel_s"], dtype=float),
                "freqs_hz": np.asarray(cluster_maps["freqs_hz"], dtype=float),
                "observed_A_power_normalized": np.asarray(cluster_maps["A_power"], dtype=np.float32),
                "observed_E_power_normalized": np.asarray(cluster_maps["E_power"], dtype=np.float32),
                "observed_AE_power_normalized": np.asarray(cluster_maps["AE_power"], dtype=np.float32),
                "observed_A_floor": np.asarray(cluster_maps["A_floor"], dtype=np.float64),
                "observed_E_floor": np.asarray(cluster_maps["E_floor"], dtype=np.float64),
            }
        )

        for index, case_row in catalog.iterrows():
            source_label = source_label_from_catalog_index(index)
            windows = case_windows(case_row, args)
            events, masks = find_seeded_chirp_events(
                cluster_maps,
                config=cluster_config,
                start_day=windows["search_start_day"],
                stop_day=windows["search_stop_day"],
                label=str(case_row["case_id"]),
            )
            if events.empty:
                selected_rows.append(
                    {
                        "source_label": source_label,
                        "case_id": str(case_row["case_id"]),
                        "selection_reason": "no_cluster_found",
                    }
                )
                continue
            events = annotate_events(events, source_label=source_label, case_row=case_row, windows=windows)
            candidate_frames.append(events)
            selected = select_event(events)
            if bool(selected["peak_in_local_recovery_window"]):
                selection_reason = "max_event_statistic_with_peak_in_local_recovery_window"
            elif bool(selected["overlaps_catalog_time"]):
                selection_reason = "max_event_statistic_overlapping_catalog_window"
            else:
                selection_reason = "max_event_statistic_in_search_window"
            cluster_mask = masks[str(selected["event_id"])]
            dilated_mask = dilate_mask(cluster_mask, freq_bins=args.dilate_freq_bins, time_bins=args.dilate_time_bins)
            mask_payload[f"{source_label}_cluster"] = cluster_mask.astype(bool)
            mask_payload[f"{source_label}_cluster_dilated"] = dilated_mask.astype(bool)

            source_tf_maps = compute_raw_tf_maps(
                wdm,
                t_lisa,
                {
                    "A2": np.asarray(products[f"{source_label}_A2"], dtype=float),
                    "E2": np.asarray(products[f"{source_label}_E2"], dtype=float),
                },
            )
            for channel in AE_CHANNELS:
                source_series = np.asarray(products[f"{source_label}_{channel}"], dtype=float)
                series_payload[f"{source_label}_{channel}_source"] = source_series.astype(np.float64)
                for mask_label, mask in (("cluster", cluster_mask), ("cluster_dilated", dilated_mask)):
                    row, series = analyze_mask_variant(
                        wdm=wdm,
                        source_label=source_label,
                        case_id=str(case_row["case_id"]),
                        channel=channel,
                        mask_label=mask_label,
                        mask=mask,
                        source_series=source_series,
                        observed_tf_map=observed_tf_maps[channel],
                        source_tf_map=source_tf_maps[channel],
                        noise_tf_map=noise_tf_maps[channel],
                    )
                    channel_rows.append(
                        {
                            **row,
                            "selected_event_id": str(selected["event_id"]),
                            "selected_event_statistic": float(selected["event_statistic"]),
                            "selected_event_peak_time_day": float(selected["peak_time_day"]),
                            "selected_event_peak_freq_hz": float(selected["peak_freq_hz"]),
                        }
                    )
                    series_payload.update(series)
            selected_row = selected.to_dict()
            selected_row["selection_reason"] = selection_reason
            selected_row["strict_mask_pixels"] = int(np.sum(cluster_mask))
            selected_row["dilated_mask_pixels"] = int(np.sum(dilated_mask))
            selected_row["wdm_dt_s"] = float(cluster_maps["wdm_dt_s"])
            selected_row["wdm_df_hz"] = float(cluster_maps["wdm_df_hz"])
            selected_rows.append(selected_row)

    candidate_table = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    selected_table = pd.DataFrame(selected_rows)
    channel_metrics = pd.DataFrame(channel_rows)
    ae_metrics = compute_ae_combined_metrics(channel_metrics, series_payload) if not channel_metrics.empty else pd.DataFrame()

    if not selected_table.empty and not ae_metrics.empty:
        strict_ae = ae_metrics[(ae_metrics["mask_label"] == "cluster") & (ae_metrics["reconstruction"] == "observed_cluster")]
        strict_ae = strict_ae.set_index("source_label")
        selected_table["ae_observed_cluster_match"] = selected_table["source_label"].map(strict_ae["match"])
        selected_table["ae_observed_cluster_mismatch"] = selected_table["source_label"].map(strict_ae["mismatch"])

    candidate_csv = args.output_dir / "observed_cluster_candidates.csv"
    selected_csv = args.output_dir / "selected_observed_cluster_events.csv"
    channel_csv = args.output_dir / "observed_cluster_reconstruction_metrics.csv"
    ae_csv = args.output_dir / "observed_cluster_ae_combined_matches.csv"
    candidate_table.to_csv(candidate_csv, index=False)
    selected_table.to_csv(selected_csv, index=False)
    channel_metrics.to_csv(channel_csv, index=False)
    ae_metrics.to_csv(ae_csv, index=False)
    (args.output_dir / "observed_cluster_candidates.json").write_text(
        json.dumps(candidate_table.to_dict(orient="records"), indent=2) + "\n"
    )
    (args.output_dir / "selected_observed_cluster_events.json").write_text(
        json.dumps(selected_table.to_dict(orient="records"), indent=2) + "\n"
    )
    (args.output_dir / "observed_cluster_reconstruction_metrics.json").write_text(
        json.dumps(channel_metrics.to_dict(orient="records"), indent=2) + "\n"
    )
    (args.output_dir / "observed_cluster_ae_combined_matches.json").write_text(
        json.dumps(ae_metrics.to_dict(orient="records"), indent=2) + "\n"
    )
    np.savez_compressed(args.output_dir / "observed_cluster_masks_and_maps.npz", **mask_payload)
    np.savez_compressed(args.output_dir / "observed_cluster_reconstructed_ae_timeseries.npz", **series_payload)

    plots: dict[str, str] = {}
    if not selected_table.empty:
        mask_plot = plot_dir / "observed_cluster_masks.png"
        plot_cluster_masks(mask_plot, maps=cluster_maps, selected_events=selected_table, masks=mask_payload, args=args)
        plots["masks"] = str(mask_plot)
        plots.update(
            plot_time_domain_overlays(
                plot_dir,
                t_rel_s=series_payload["time_rel_s"],
                selected_events=selected_table,
                channel_metrics=channel_metrics,
                payload=series_payload,
                peak_zoom_half_window_days=float(args.peak_zoom_half_window_days),
            )
        )
        combined_time_plot = plot_dir / "observed_cluster_time_domain_overlaps.png"
        plot_combined_time_domain_overlaps(
            combined_time_plot,
            t_rel_s=series_payload["time_rel_s"],
            selected_events=selected_table,
            channel_metrics=channel_metrics,
            payload=series_payload,
        )
        plots["time_domain_overlaps"] = str(combined_time_plot)
        peak_zoom_time_plot = plot_dir / "observed_cluster_time_domain_peak_zoom_overlaps.png"
        plot_peak_zoom_time_domain_overlaps(
            peak_zoom_time_plot,
            t_rel_s=series_payload["time_rel_s"],
            selected_events=selected_table,
            channel_metrics=channel_metrics,
            payload=series_payload,
            peak_zoom_half_window_days=float(args.peak_zoom_half_window_days),
        )
        plots["time_domain_peak_zoom_overlaps"] = str(peak_zoom_time_plot)

    write_summary(
        args.output_dir / "run_summary.md",
        args=args,
        selected_events=selected_table,
        ae_metrics=ae_metrics,
        plots=plots,
    )

    print(
        json.dumps(
            {
                "candidate_csv": str(candidate_csv),
                "selected_csv": str(selected_csv),
                "channel_metrics_csv": str(channel_csv),
                "ae_metrics_csv": str(ae_csv),
                "summary": str(args.output_dir / "run_summary.md"),
                "plots": plots,
                "strict_ae_match_min": float(
                    ae_metrics[
                        (ae_metrics["mask_label"] == "cluster")
                        & (ae_metrics["reconstruction"] == "observed_cluster")
                    ]["match"].min()
                )
                if not ae_metrics.empty
                else np.nan,
                "dilated_ae_match_min": float(
                    ae_metrics[
                        (ae_metrics["mask_label"] == "cluster_dilated")
                        & (ae_metrics["reconstruction"] == "observed_cluster")
                    ]["match"].min()
                )
                if not ae_metrics.empty
                else np.nan,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
