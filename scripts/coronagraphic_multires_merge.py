"""Merge coronagraphic WDM clusters across WDM resolutions.

This module operates after the per-resolution coronagraphic scan:

1. Read ``pixel_clusters_all_resolutions.csv`` from a multi-resolution run.
2. Summarize every coherent per-resolution HDBSCAN cluster.
3. Run a second HDBSCAN pass on those cluster summaries.
4. Attach the resulting cross-resolution event labels back to every pixel.

The merge is deliberately performed on physical coordinates and kappa-null
features, not on WDM bin indices, because each WDM resolution has a different
time-frequency grid.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoronagraphicMergeConfig:
    """Settings for cross-resolution coronagraphic cluster merging."""

    run_dir: str
    output_dir: str | None = None
    input_csv: str = "pixel_clusters_all_resolutions.csv"
    min_cluster_size: int = 2
    min_samples: int | None = None
    min_resolutions: int = 2
    make_plots: bool = True


def _require_dependencies():
    try:
        import hdbscan
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        raise ModuleNotFoundError(
            "Cross-resolution merge requires hdbscan and scikit-learn. "
            "Use the lisa-tf conda environment from env.yml."
        ) from exc
    return hdbscan, StandardScaler


def _safe_log10(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.log10(np.clip(arr, 1.0e-300, 1.0e300))


def _normalize_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
    vec = np.asarray([x, y, z], dtype=float)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm == 0.0:
        return np.nan, np.nan, np.nan
    vec = vec / norm
    return float(vec[0]), float(vec[1]), float(vec[2])


def _as_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="raise").astype(int)


def load_multires_pixels(run_dir: Path, input_csv: str) -> pd.DataFrame:
    path = run_dir / input_csv
    if not path.exists():
        raise FileNotFoundError(f"Missing multi-resolution pixel CSV: {path}")
    pixels = pd.read_csv(path)
    required = {
        "wdm_m",
        "cluster",
        "time_s",
        "freq_hz",
        "best_x",
        "best_y",
        "best_z",
        "kappa_min_power",
        "sky_contrast",
        "sky_sharpness",
    }
    missing = sorted(required - set(pixels.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    pixels = pixels.copy()
    pixels["wdm_m"] = _as_int_series(pixels["wdm_m"])
    pixels["cluster"] = _as_int_series(pixels["cluster"])
    return pixels


def build_layer_cluster_representatives(pixels: pd.DataFrame) -> pd.DataFrame:
    coherent = pixels[pixels["cluster"] >= 0].copy()
    if coherent.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (wdm_m, cluster_id), group in coherent.groupby(["wdm_m", "cluster"], sort=True):
        median_x, median_y, median_z = _normalize_xyz(
            float(np.nanmedian(group["best_x"])),
            float(np.nanmedian(group["best_y"])),
            float(np.nanmedian(group["best_z"])),
        )
        row: dict[str, object] = {
            "layer_cluster_id": f"M{int(wdm_m)}_c{int(cluster_id)}",
            "wdm_m": int(wdm_m),
            "cluster": int(cluster_id),
            "n_pixels": int(len(group)),
            "median_time_s": float(np.nanmedian(group["time_s"])),
            "time_start_s": float(np.nanmin(group["time_s"])),
            "time_stop_s": float(np.nanmax(group["time_s"])),
            "median_freq_hz": float(np.nanmedian(group["freq_hz"])),
            "freq_min_hz": float(np.nanmin(group["freq_hz"])),
            "freq_max_hz": float(np.nanmax(group["freq_hz"])),
            "median_best_x": median_x,
            "median_best_y": median_y,
            "median_best_z": median_z,
            "median_kappa_min_power": float(np.nanmedian(group["kappa_min_power"])),
            "median_sky_contrast": float(np.nanmedian(group["sky_contrast"])),
            "median_sky_sharpness": float(np.nanmedian(group["sky_sharpness"])),
            "median_best_ra": float(np.nanmedian(group["best_ra"])) if "best_ra" in group else np.nan,
            "median_best_dec": float(np.nanmedian(group["best_dec"])) if "best_dec" in group else np.nan,
        }
        if "true_sky_sep_deg" in group:
            row["median_true_sky_sep_deg"] = float(np.nanmedian(group["true_sky_sep_deg"]))
        rows.append(row)

    reps = pd.DataFrame(rows)
    reps["log10_median_freq_hz"] = _safe_log10(reps["median_freq_hz"])
    reps["log10_median_kappa_min_power"] = _safe_log10(reps["median_kappa_min_power"])
    reps["log10_median_sky_contrast"] = _safe_log10(reps["median_sky_contrast"])
    reps["log10_median_sky_sharpness"] = _safe_log10(reps["median_sky_sharpness"])
    return reps


def merge_layer_clusters(
    representatives: pd.DataFrame,
    *,
    min_cluster_size: int,
    min_samples: int | None,
) -> pd.DataFrame:
    out = representatives.copy()
    out["event_cluster"] = -1
    out["event_probability"] = 0.0
    out["event_cluster_method"] = "hdbscan_layer_cluster_merge"
    if out.empty:
        return out

    feature_cols = [
        "median_time_s",
        "log10_median_freq_hz",
        "median_best_x",
        "median_best_y",
        "median_best_z",
        "log10_median_kappa_min_power",
        "log10_median_sky_contrast",
        "log10_median_sky_sharpness",
    ]
    features = out[feature_cols].to_numpy(dtype=float)
    valid = np.all(np.isfinite(features), axis=1)
    if int(np.sum(valid)) < max(2, int(min_cluster_size)):
        return out

    hdbscan, StandardScaler = _require_dependencies()
    scaled = StandardScaler().fit_transform(features[valid])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(scaled)
    probabilities = getattr(clusterer, "probabilities_", np.zeros(len(labels), dtype=float))

    valid_indices = np.flatnonzero(valid)
    out.loc[valid_indices, "event_cluster"] = labels.astype(int)
    out.loc[valid_indices, "event_probability"] = probabilities.astype(float)
    return out


def attach_event_labels_to_pixels(pixels: pd.DataFrame, representatives: pd.DataFrame) -> pd.DataFrame:
    labels = representatives[
        ["wdm_m", "cluster", "layer_cluster_id", "event_cluster", "event_probability", "event_cluster_method"]
    ].copy()
    merged = pixels.merge(labels, on=["wdm_m", "cluster"], how="left")
    merged["event_cluster"] = merged["event_cluster"].fillna(-1).astype(int)
    merged["event_probability"] = merged["event_probability"].fillna(0.0).astype(float)
    merged["layer_cluster_id"] = merged["layer_cluster_id"].fillna("")
    merged["event_cluster_method"] = merged["event_cluster_method"].fillna("unclustered_pixel")
    return merged


def summarize_events(merged_pixels: pd.DataFrame, representatives: pd.DataFrame, min_resolutions: int) -> pd.DataFrame:
    coherent = merged_pixels[merged_pixels["event_cluster"] >= 0].copy()
    if coherent.empty:
        return pd.DataFrame(
            columns=[
                "event_cluster",
                "n_pixels",
                "n_layer_clusters",
                "n_resolutions",
                "wdm_ms",
                "passes_min_resolutions",
            ]
        )

    rep_lookup = representatives[representatives["event_cluster"] >= 0].copy()
    rep_grouped = rep_lookup.groupby("event_cluster", sort=True)

    rows: list[dict[str, object]] = []
    for event_cluster, group in coherent.groupby("event_cluster", sort=True):
        reps = rep_grouped.get_group(event_cluster) if event_cluster in rep_grouped.groups else pd.DataFrame()
        wdm_ms = sorted(int(value) for value in group["wdm_m"].unique())
        layer_ids = sorted(str(value) for value in reps["layer_cluster_id"].unique()) if not reps.empty else []
        row: dict[str, object] = {
            "event_cluster": int(event_cluster),
            "n_pixels": int(len(group)),
            "n_layer_clusters": int(len(layer_ids)),
            "n_resolutions": int(len(wdm_ms)),
            "wdm_ms": ",".join(str(value) for value in wdm_ms),
            "layer_cluster_ids": ",".join(layer_ids),
            "passes_min_resolutions": int(len(wdm_ms)) >= int(min_resolutions),
            "median_time_s": float(np.nanmedian(group["time_s"])),
            "time_start_s": float(np.nanmin(group["time_s"])),
            "time_stop_s": float(np.nanmax(group["time_s"])),
            "median_freq_hz": float(np.nanmedian(group["freq_hz"])),
            "freq_min_hz": float(np.nanmin(group["freq_hz"])),
            "freq_max_hz": float(np.nanmax(group["freq_hz"])),
            "median_best_x": float(np.nanmedian(group["best_x"])),
            "median_best_y": float(np.nanmedian(group["best_y"])),
            "median_best_z": float(np.nanmedian(group["best_z"])),
            "median_kappa_min_power": float(np.nanmedian(group["kappa_min_power"])),
            "median_sky_contrast": float(np.nanmedian(group["sky_contrast"])),
            "median_sky_sharpness": float(np.nanmedian(group["sky_sharpness"])),
            "median_event_probability": float(np.nanmedian(group["event_probability"])),
        }
        if "true_sky_sep_deg" in group:
            row["median_true_sky_sep_deg"] = float(np.nanmedian(group["true_sky_sep_deg"]))
        rows.append(row)

    return pd.DataFrame(rows)


def make_merge_plots(merged_pixels: pd.DataFrame, event_summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    coherent = merged_pixels[merged_pixels["event_cluster"] >= 0].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    noise = merged_pixels[merged_pixels["event_cluster"] < 0]
    if len(noise):
        ax.scatter(noise["time_s"], noise["freq_hz"] * 1.0e3, s=10, c="#bdbdbd", alpha=0.35, label="unmerged/noise")
    if len(coherent):
        scatter = ax.scatter(
            coherent["time_s"],
            coherent["freq_hz"] * 1.0e3,
            c=coherent["event_cluster"],
            s=22,
            cmap="tab20",
            alpha=0.85,
        )
        fig.colorbar(scatter, ax=ax, label="cross-resolution event cluster")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [mHz]")
    ax.set_title("Cross-resolution coronagraphic event clusters")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "multires_event_clusters_time_frequency.png", dpi=170)
    plt.close(fig)

    if len(coherent) and {"best_ra", "best_dec"}.issubset(coherent.columns):
        fig, ax = plt.subplots(figsize=(9, 5))
        scatter = ax.scatter(
            np.degrees(coherent["best_ra"]),
            np.degrees(coherent["best_dec"]),
            c=coherent["event_cluster"],
            s=24,
            cmap="tab20",
            alpha=0.85,
        )
        fig.colorbar(scatter, ax=ax, label="cross-resolution event cluster")
        ax.set_xlabel("best-null RA [deg]")
        ax.set_ylabel("best-null Dec [deg]")
        ax.set_title("Cross-resolution event clusters in kappa-minimum sky space")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "multires_event_clusters_sky.png", dpi=170)
        plt.close(fig)

    if len(event_summary):
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#2ca02c" if value else "#d62728" for value in event_summary["passes_min_resolutions"]]
        ax.bar(event_summary["event_cluster"].astype(str), event_summary["n_resolutions"], color=colors)
        ax.set_xlabel("event cluster")
        ax.set_ylabel("number of WDM resolutions")
        ax.set_title("Cross-resolution support per event cluster")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "multires_event_resolution_support.png", dpi=170)
        plt.close(fig)


def run_merge(config: CoronagraphicMergeConfig) -> dict[str, object]:
    run_dir = Path(config.run_dir)
    output_dir = Path(config.output_dir) if config.output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pixels = load_multires_pixels(run_dir, config.input_csv)
    representatives = build_layer_cluster_representatives(pixels)
    representatives = merge_layer_clusters(
        representatives,
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
    )
    merged_pixels = attach_event_labels_to_pixels(pixels, representatives)
    event_summary = summarize_events(merged_pixels, representatives, config.min_resolutions)

    representatives.to_csv(output_dir / "layer_cluster_representatives.csv", index=False)
    merged_pixels.to_csv(output_dir / "merged_pixel_clusters.csv", index=False)
    event_summary.to_csv(output_dir / "multires_event_summary.csv", index=False)
    (output_dir / "multires_merge_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    if config.make_plots:
        make_merge_plots(merged_pixels, event_summary, output_dir)

    hdbscan_event_labels = sorted(
        int(value)
        for value in merged_pixels["event_cluster"].unique()
        if int(value) >= 0
    )
    n_multires_supported = int(np.sum(event_summary["passes_min_resolutions"].to_numpy(dtype=bool))) if len(event_summary) else 0
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "n_pixels": int(len(pixels)),
        "n_coherent_pixels": int(np.sum(pixels["cluster"].to_numpy(dtype=int) >= 0)),
        "n_layer_clusters": int(len(representatives)),
        "n_hdbscan_event_clusters": int(len(hdbscan_event_labels)),
        "n_multires_event_candidates": n_multires_supported,
        "event_cluster_counts": (
            merged_pixels["event_cluster"].value_counts().sort_index().astype(int).to_dict()
            if "event_cluster" in merged_pixels
            else {}
        ),
        "event_summary_csv": str(output_dir / "multires_event_summary.csv"),
        "merged_pixels_csv": str(output_dir / "merged_pixel_clusters.csv"),
        "representatives_csv": str(output_dir / "layer_cluster_representatives.csv"),
    }
    (output_dir / "multires_merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> CoronagraphicMergeConfig:
    parser = argparse.ArgumentParser(description="Merge coronagraphic WDM clusters across WDM resolutions.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-csv", default="pixel_clusters_all_resolutions.csv")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--min-resolutions", type=int, default=2)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    return CoronagraphicMergeConfig(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        input_csv=args.input_csv,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        min_resolutions=args.min_resolutions,
        make_plots=not args.no_plots,
    )


def main() -> None:
    summary = run_merge(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
