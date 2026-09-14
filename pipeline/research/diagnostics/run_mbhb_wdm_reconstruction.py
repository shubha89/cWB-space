from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clean_wdm import require_wdm_dependencies


RUN_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_wdm_reconstruction"
OUTPUT_DIR = RUN_DIR / "outputs"

TWO_WEEK_OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "simulated_2week_mbhb_then_glitch_highrate" / "outputs"
PRODUCTS_NPZ = TWO_WEEK_OUTPUT_DIR / "two_week_mbhb_then_glitch_tdi.npz"
MBHB_CATALOG_CSV = TWO_WEEK_OUTPUT_DIR / "mbhb_catalog.csv"

AE_CHANNELS = ("A2", "E2")
DAY = 86_400.0
LOW_HZ = 1.0e-5
HIGH_HZ = 1.0e-1


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Reconstruct source-only MBHB A/E time series from WDM maps and quantify "
            "full-map and loud-pixel inverse errors."
        )
    )
    parser.add_argument("--products", type=Path, default=PRODUCTS_NPZ)
    parser.add_argument("--catalog", type=Path, default=MBHB_CATALOG_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--M",
        type=int,
        default=512,
        help="WDM frequency parameter. The default keeps the M512 grid used by the TF diagnostics.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=256,
        help="WDM filter parameter. The default was chosen from an inverse round-trip sanity scan.",
    )
    parser.add_argument("--beta-order", type=int, default=6)
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--backend", default="numba", choices=("numba", "jax"))
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=5.0e-3,
        help="Fraction of selectable WDM pixels retained in the masked reconstruction.",
    )
    parser.add_argument(
        "--mask-frequency-scope",
        choices=("all", "analysis-band"),
        default="all",
        help="Select loud pixels over all WDM bins or only the standard analysis band.",
    )
    parser.add_argument("--low-hz", type=float, default=LOW_HZ)
    parser.add_argument("--high-hz", type=float, default=HIGH_HZ)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def ensure_inputs(args: Namespace) -> None:
    missing = [path for path in (args.products, args.catalog) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(str(path) for path in missing))
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top-fraction must be in (0, 1].")
    if args.low_hz < 0.0 or args.high_hz <= args.low_hz:
        raise ValueError("Require 0 <= --low-hz < --high-hz.")


def wdm_inverse_quadrature_mean(wdm, tf_map, n_samples: int) -> np.ndarray:
    """Combine the 00 and 90 degree inverse WDM quadratures for a real strain series."""

    w2t = np.asarray(wdm.w2t(tf_map).value, dtype=float)[:n_samples]
    w2tq = np.asarray(wdm.w2tQ(tf_map).value, dtype=float)[:n_samples]
    return 0.5 * (w2t + w2tq)


def scalar_projection_scale(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    denom = float(np.dot(reconstruction, reconstruction))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(reference, reconstruction) / denom)


def safe_correlation(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    if float(np.std(reference)) <= 0.0 or float(np.std(reconstruction)) <= 0.0:
        return np.nan
    return float(np.corrcoef(reference, reconstruction)[0, 1])


def normalized_time_domain_match(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    reference_norm = float(np.linalg.norm(reference))
    reconstruction_norm = float(np.linalg.norm(reconstruction))
    denom = reference_norm * reconstruction_norm
    if denom <= 0.0:
        return np.nan
    return float(np.dot(reference, reconstruction) / denom)


def reconstruction_metrics(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    reference_norm = float(np.linalg.norm(reference))
    residual = reference - reconstruction
    reconstruction_norm = float(np.linalg.norm(reconstruction))
    if reference_norm <= 0.0:
        rel_l2 = np.nan
        energy_ratio = np.nan
    else:
        rel_l2 = float(np.linalg.norm(residual) / reference_norm)
        energy_ratio = float(reconstruction_norm**2 / reference_norm**2)
    peak_ref = int(np.argmax(np.abs(reference))) if reference.size else -1
    peak_rec = int(np.argmax(np.abs(reconstruction))) if reconstruction.size else -1
    return {
        "rel_l2": rel_l2,
        "match": normalized_time_domain_match(reference, reconstruction),
        "mismatch": 1.0 - normalized_time_domain_match(reference, reconstruction),
        "corr": safe_correlation(reference, reconstruction),
        "max_abs_residual": float(np.max(np.abs(residual))) if residual.size else np.nan,
        "energy_ratio": energy_ratio,
        "peak_index_error_samples": int(peak_rec - peak_ref),
    }


def selectable_pixel_mask(tf_map, *, mask_frequency_scope: str, low_hz: float, high_hz: float) -> np.ndarray:
    data = np.asarray(tf_map.data)
    if mask_frequency_scope == "all":
        return np.ones(data.shape, dtype=bool)
    freqs_hz = np.arange(tf_map.n_freq, dtype=float) * float(tf_map.df)
    freq_mask = (freqs_hz >= low_hz) & (freqs_hz <= high_hz)
    return np.broadcast_to(freq_mask[:, None], data.shape).copy()


def top_power_mask(power: np.ndarray, selectable: np.ndarray, top_fraction: float) -> np.ndarray:
    selectable_values = np.asarray(power[selectable], dtype=float)
    if selectable_values.size == 0:
        raise ValueError("No selectable WDM pixels for the requested frequency scope.")
    n_keep = max(1, int(round(top_fraction * selectable_values.size)))
    cutoff = float(np.partition(selectable_values, selectable_values.size - n_keep)[selectable_values.size - n_keep])
    return selectable & (power >= cutoff)


def source_label_from_catalog_index(index: int) -> str:
    return f"mbhb{index + 1}"


def analyze_channel(
    *,
    wdm,
    t_lisa: np.ndarray,
    source_label: str,
    case_row: pd.Series,
    channel: str,
    source_series: np.ndarray,
    top_fraction: float,
    mask_frequency_scope: str,
    low_hz: float,
    high_hz: float,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    n_samples = int(len(source_series))
    native_dt = float(np.median(np.diff(t_lisa)))
    sample_rate = 1.0 / native_dt
    tf_map = wdm.t2w(source_series, sample_rate=sample_rate, t0=float(t_lisa[0]))

    full_raw = wdm_inverse_quadrature_mean(wdm, tf_map, n_samples)
    full_scale = scalar_projection_scale(source_series, full_raw)
    full_reconstruction = full_scale * full_raw
    full = reconstruction_metrics(source_series, full_reconstruction)

    power = np.abs(np.asarray(tf_map.data, dtype=np.complex128)) ** 2
    selectable = selectable_pixel_mask(tf_map, mask_frequency_scope=mask_frequency_scope, low_hz=low_hz, high_hz=high_hz)
    mask = top_power_mask(power, selectable, top_fraction)
    masked_tf_map = replace(tf_map, data=np.where(mask, tf_map.data, 0.0 + 0.0j))
    masked_raw = wdm_inverse_quadrature_mean(wdm, masked_tf_map, n_samples)
    masked_same_scale = full_scale * masked_raw
    masked_same = reconstruction_metrics(source_series, masked_same_scale)
    masked_fit_scale = scalar_projection_scale(source_series, masked_raw)
    masked_fit = reconstruction_metrics(source_series, masked_fit_scale * masked_raw)

    total_tf_power = float(np.sum(power))
    selected_tf_power = float(np.sum(power[mask]))
    source_abs = np.abs(source_series)
    response_peak_idx = int(np.argmax(source_abs))
    response_peak_rel_s = float(t_lisa[response_peak_idx] - t_lisa[0])

    row: dict[str, float | int | str] = {
        "source_label": source_label,
        "case_id": str(case_row["case_id"]),
        "channel": channel,
        "wdm_M": int(wdm.M),
        "wdm_K": int(wdm.K),
        "wdm_beta_order": int(wdm.beta_order),
        "wdm_precision": int(wdm.precision),
        "inverse_method": "0.5*(w2t+w2tQ)",
        "native_dt_s": native_dt,
        "sample_rate_hz": sample_rate,
        "tf_dt_s": float(tf_map.dt),
        "tf_df_hz": float(tf_map.df),
        "n_time_samples": n_samples,
        "n_tf_time_bins": int(tf_map.n_time),
        "n_tf_freq_bins": int(tf_map.n_freq),
        "n_tf_pixels": int(power.size),
        "top_fraction": float(top_fraction),
        "mask_frequency_scope": mask_frequency_scope,
        "mask_low_hz": float(low_hz),
        "mask_high_hz": float(high_hz),
        "n_selectable_pixels": int(np.sum(selectable)),
        "n_selected_pixels": int(np.sum(mask)),
        "selected_tf_power_fraction": selected_tf_power / total_tf_power if total_tf_power > 0.0 else np.nan,
        "original_l2_norm": float(np.linalg.norm(source_series)),
        "original_absmax": float(np.max(source_abs)) if source_abs.size else np.nan,
        "response_peak_rel_s_measured": response_peak_rel_s,
        "response_peak_day_measured": response_peak_rel_s / DAY,
        "response_peak_rel_s_catalog": float(case_row["response_peak_rel_s"]),
        "response_peak_day_catalog": float(case_row["response_peak_day"]),
        "full_amplitude_scale": full_scale,
        "masked_fit_amplitude_scale": masked_fit_scale,
    }
    for prefix, values in (("full", full), ("masked_same_scale", masked_same), ("masked_fit", masked_fit)):
        for key, value in values.items():
            row[f"{prefix}_{key}"] = value
    row["full_peak_time_error_s"] = row["full_peak_index_error_samples"] * native_dt
    row["masked_same_scale_peak_time_error_s"] = row["masked_same_scale_peak_index_error_samples"] * native_dt
    row["masked_fit_peak_time_error_s"] = row["masked_fit_peak_index_error_samples"] * native_dt

    series_payload = {
        f"{source_label}_{channel}_original": source_series.astype(np.float64),
        f"{source_label}_{channel}_wdm_full": full_reconstruction.astype(np.float64),
        f"{source_label}_{channel}_wdm_top": masked_same_scale.astype(np.float64),
        f"{source_label}_{channel}_wdm_top_fit_scale": (masked_fit_scale * masked_raw).astype(np.float64),
    }
    return row, series_payload


def plot_case_overlay(
    output: Path,
    *,
    t_rel_s: np.ndarray,
    rows: pd.DataFrame,
    payload: dict[str, np.ndarray],
    source_label: str,
    case_id: str,
    top_fraction: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True, constrained_layout=True)
    colors = {
        "original": "#111827",
        "wdm_full": "#2563eb",
        "wdm_top": "#dc2626",
    }
    case_rows = rows[rows["source_label"] == source_label].set_index("channel")
    response_peak = float(case_rows.iloc[0]["response_peak_rel_s_catalog"])
    half_width = max(0.35 * DAY, 8.0 * float(case_rows.iloc[0]["tf_dt_s"]))
    window = (t_rel_s >= response_peak - half_width) & (t_rel_s <= response_peak + half_width)
    if int(np.sum(window)) < 8:
        window = slice(None)

    for ax, channel in zip(axes, AE_CHANNELS):
        original = payload[f"{source_label}_{channel}_original"]
        full = payload[f"{source_label}_{channel}_wdm_full"]
        masked = payload[f"{source_label}_{channel}_wdm_top"]
        days = t_rel_s[window] / DAY
        ax.plot(days, original[window], color=colors["original"], lw=1.2, label="source-only")
        ax.plot(days, full[window], color=colors["wdm_full"], lw=1.0, alpha=0.9, label="full WDM inverse")
        ax.plot(
            days,
            masked[window],
            color=colors["wdm_top"],
            lw=1.0,
            alpha=0.9,
            label=f"top {100.0 * top_fraction:.3g}% pixels",
        )
        row = case_rows.loc[channel]
        text = (
            f"full rel L2={row['full_rel_l2']:.3g}, "
            f"top rel L2={row['masked_same_scale_rel_l2']:.3g}"
        )
        ax.text(0.015, 0.92, text, transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.set_ylabel(f"{channel} [Hz]")
        ax.grid(True, alpha=0.22)
    axes[-1].set_xlabel("Time since product start [days]")
    axes[0].legend(loc="upper right", frameon=False, ncol=3)
    fig.suptitle(f"WDM time-series reconstruction: {case_id}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_summary(output: Path, rows: pd.DataFrame) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    labels = [f"{row.source_label} {row.channel}" for row in rows.itertuples()]
    x = np.arange(len(labels))

    axes[0].bar(x - 0.18, rows["full_rel_l2"], width=0.36, color="#2563eb", label="full inverse")
    axes[0].bar(x + 0.18, rows["masked_same_scale_rel_l2"], width=0.36, color="#dc2626", label="top-pixel inverse")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Relative L2 error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].grid(True, which="both", axis="y", alpha=0.22)
    axes[0].legend(frameon=False)

    axes[1].bar(x, rows["selected_tf_power_fraction"], color="#059669")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("Selected WDM power fraction")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].grid(True, axis="y", alpha=0.22)
    fig.suptitle("Source-only MBHB A/E WDM reconstruction diagnostics")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def compute_ae_combined_matches(metrics: pd.DataFrame, payload: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    reconstruction_keys = {
        "full": "wdm_full",
        "top_same_scale": "wdm_top",
        "top_fit_scale": "wdm_top_fit_scale",
    }
    for source_label in metrics["source_label"].drop_duplicates():
        case_rows = metrics[metrics["source_label"] == source_label]
        case_id = str(case_rows.iloc[0]["case_id"])
        reference = np.concatenate([payload[f"{source_label}_{channel}_original"] for channel in AE_CHANNELS])
        for reconstruction_label, reconstruction_key in reconstruction_keys.items():
            reconstruction = np.concatenate(
                [payload[f"{source_label}_{channel}_{reconstruction_key}"] for channel in AE_CHANNELS]
            )
            values = reconstruction_metrics(reference, reconstruction)
            rows.append(
                {
                    "source_label": str(source_label),
                    "case_id": case_id,
                    "channel_combination": "AE",
                    "reconstruction": reconstruction_label,
                    **values,
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    output: Path,
    *,
    args: Namespace,
    metrics: pd.DataFrame,
    ae_combined_matches: pd.DataFrame,
    plots: dict[str, str],
) -> None:
    full_rel = metrics["full_rel_l2"]
    top_rel = metrics["masked_same_scale_rel_l2"]
    full_match = metrics["full_match"]
    top_match = metrics["masked_same_scale_match"]
    ae_full_match = ae_combined_matches[ae_combined_matches["reconstruction"] == "full"]["match"]
    ae_top_match = ae_combined_matches[ae_combined_matches["reconstruction"] == "top_same_scale"]["match"]
    selected = metrics["selected_tf_power_fraction"]
    lines = [
        "# MBHB WDM Reconstruction Summary",
        "",
        "## Inputs",
        "",
        f"- Products: `{args.products}`",
        f"- Catalog: `{args.catalog}`",
        f"- WDM inverse: `0.5*(w2t+w2tQ)`",
        f"- WDM parameters: `M={args.M}`, `K={args.K}`, `beta_order={args.beta_order}`, `precision={args.precision}`, `backend={args.backend}`",
        f"- Top-pixel selection: `{args.top_fraction:g}` of `{args.mask_frequency_scope}` WDM pixels",
        "",
        "## Quantitative Checks",
        "",
        f"- Full-map inverse relative L2 range: `{full_rel.min():.6g}` to `{full_rel.max():.6g}`",
        f"- Full-map inverse median relative L2: `{full_rel.median():.6g}`",
        f"- Full-map inverse match range: `{full_match.min():.12g}` to `{full_match.max():.12g}`",
        f"- Combined-AE full inverse match range: `{ae_full_match.min():.12g}` to `{ae_full_match.max():.12g}`",
        f"- Top-pixel inverse relative L2 range: `{top_rel.min():.6g}` to `{top_rel.max():.6g}`",
        f"- Top-pixel inverse median relative L2: `{top_rel.median():.6g}`",
        f"- Top-pixel inverse match range: `{top_match.min():.12g}` to `{top_match.max():.12g}`",
        f"- Combined-AE top-pixel inverse match range: `{ae_top_match.min():.12g}` to `{ae_top_match.max():.12g}`",
        f"- Selected WDM power fraction range: `{selected.min():.6g}` to `{selected.max():.6g}`",
        "",
        "## Interpretation",
        "",
        "- The reconstruction is performed on source-only MBHB A/E streams, so the metrics test the WDM inverse and sparse signal support without instrument-noise contamination.",
        "- The reported match is a zero-lag normalized time-domain overlap, `<h,h_rec>/sqrt(<h,h><h_rec,h_rec>)`; it is not PSD-weighted and is not maximized over time or phase.",
        "- The scalar amplitude calibration is fitted only from the full-map inverse and then reused for the top-pixel reconstruction.",
        "- For this `wdm_wavelet` build, inverse accuracy depends on the WDM filter support; the default `M=512, K=256` was chosen from a round-trip sanity scan for these products.",
        "",
        "## Outputs",
        "",
        f"- Metrics CSV: `{args.output_dir / 'mbhb_wdm_reconstruction_metrics.csv'}`",
        f"- Metrics JSON: `{args.output_dir / 'mbhb_wdm_reconstruction_metrics.json'}`",
        f"- Combined AE match CSV: `{args.output_dir / 'mbhb_wdm_reconstruction_ae_combined_matches.csv'}`",
        f"- Combined AE match JSON: `{args.output_dir / 'mbhb_wdm_reconstruction_ae_combined_matches.json'}`",
        f"- Reconstructed series NPZ: `{args.output_dir / 'mbhb_wdm_reconstructed_ae_timeseries.npz'}`",
    ]
    for label, path in plots.items():
        lines.append(f"- Plot `{label}`: `{path}`")
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ensure_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _, WDM = require_wdm_dependencies()
    wdm = WDM(M=args.M, K=args.K, beta_order=args.beta_order, precision=args.precision, backend=args.backend)

    catalog = pd.read_csv(args.catalog)
    if args.max_cases is not None:
        catalog = catalog.head(args.max_cases)

    rows: list[dict[str, float | int | str]] = []
    payload: dict[str, np.ndarray] = {}
    with np.load(args.products) as products:
        t_lisa = np.asarray(products["t_lisa"], dtype=float)
        t_rel_s = np.asarray(products["time_rel_s"], dtype=float)
        payload["t_lisa"] = t_lisa.astype(np.float64)
        payload["time_rel_s"] = t_rel_s.astype(np.float64)
        payload["time_day"] = np.asarray(products["time_day"], dtype=float).astype(np.float64)
        for index, case_row in catalog.reset_index(drop=True).iterrows():
            source_label = source_label_from_catalog_index(index)
            for channel in AE_CHANNELS:
                key = f"{source_label}_{channel}"
                if key not in products:
                    raise KeyError(f"Missing source-only channel {key!r} in {args.products}.")
                row, series = analyze_channel(
                    wdm=wdm,
                    t_lisa=t_lisa,
                    source_label=source_label,
                    case_row=case_row,
                    channel=channel,
                    source_series=np.asarray(products[key], dtype=float),
                    top_fraction=args.top_fraction,
                    mask_frequency_scope=args.mask_frequency_scope,
                    low_hz=args.low_hz,
                    high_hz=args.high_hz,
                )
                rows.append(row)
                payload.update(series)

    metrics = pd.DataFrame(rows)
    ae_combined_matches = compute_ae_combined_matches(metrics, payload)
    metrics_csv = args.output_dir / "mbhb_wdm_reconstruction_metrics.csv"
    metrics_json = args.output_dir / "mbhb_wdm_reconstruction_metrics.json"
    ae_match_csv = args.output_dir / "mbhb_wdm_reconstruction_ae_combined_matches.csv"
    ae_match_json = args.output_dir / "mbhb_wdm_reconstruction_ae_combined_matches.json"
    series_npz = args.output_dir / "mbhb_wdm_reconstructed_ae_timeseries.npz"
    metrics.to_csv(metrics_csv, index=False)
    metrics_json.write_text(json.dumps(metrics.to_dict(orient="records"), indent=2) + "\n")
    ae_combined_matches.to_csv(ae_match_csv, index=False)
    ae_match_json.write_text(json.dumps(ae_combined_matches.to_dict(orient="records"), indent=2) + "\n")
    np.savez_compressed(series_npz, **payload)

    plots: dict[str, str] = {}
    summary_plot = plot_dir / "mbhb_wdm_reconstruction_summary.png"
    plot_summary(summary_plot, metrics)
    plots["summary"] = str(summary_plot)
    for source_label, group in metrics.groupby("source_label", sort=False):
        case_id = str(group.iloc[0]["case_id"])
        plot_path = plot_dir / f"{source_label}_wdm_reconstruction_overlay.png"
        plot_case_overlay(
            plot_path,
            t_rel_s=payload["time_rel_s"],
            rows=metrics,
            payload=payload,
            source_label=str(source_label),
            case_id=case_id,
            top_fraction=args.top_fraction,
        )
        plots[str(source_label)] = str(plot_path)

    write_summary(
        args.output_dir / "run_summary.md",
        args=args,
        metrics=metrics,
        ae_combined_matches=ae_combined_matches,
        plots=plots,
    )
    print(json.dumps({
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "ae_combined_match_csv": str(ae_match_csv),
        "ae_combined_match_json": str(ae_match_json),
        "series_npz": str(series_npz),
        "summary": str(args.output_dir / "run_summary.md"),
        "plots": plots,
        "full_rel_l2_min": float(metrics["full_rel_l2"].min()),
        "full_rel_l2_max": float(metrics["full_rel_l2"].max()),
        "top_rel_l2_min": float(metrics["masked_same_scale_rel_l2"].min()),
        "top_rel_l2_max": float(metrics["masked_same_scale_rel_l2"].max()),
        "ae_full_match_min": float(
            ae_combined_matches[ae_combined_matches["reconstruction"] == "full"]["match"].min()
        ),
        "ae_top_match_min": float(
            ae_combined_matches[ae_combined_matches["reconstruction"] == "top_same_scale"]["match"].min()
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
