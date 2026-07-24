"""Make representative slide plots for the coronagraphic clustering stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHORO_DIR = PROJECT_ROOT / "choronagraphic-cluster"
if str(CHORO_DIR) not in sys.path:
    sys.path.insert(0, str(CHORO_DIR))

import plot_multires_wdm_overlay as overlay  # noqa: E402


def load_run(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return summary, float(summary["chunk_start_s"])


def plot_selected_pixels(run_dir: Path, out_dir: Path, min_freq_mhz: float, max_freq_mhz: float) -> Path:
    _, chunk_start = load_run(run_dir)
    candidates = pd.read_csv(run_dir / "candidate_pixels_all_resolutions.csv")
    candidates["time_offset_s"] = candidates["time_s"].astype(float) - chunk_start
    candidates["freq_mhz"] = candidates["freq_hz"].astype(float) * 1.0e3

    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    markers = {128: "o", 256: "s", 512: "^"}
    colors = {128: "#00a6fb", 256: "#f9c74f", 512: "#f94144"}
    for m in sorted(candidates["wdm_m"].astype(int).unique()):
        rows = candidates[candidates["wdm_m"].astype(int) == m]
        ax.scatter(
            rows["time_offset_s"],
            rows["freq_mhz"],
            s=30,
            marker=markers.get(int(m), "o"),
            facecolor=colors.get(int(m), "#cccccc"),
            edgecolor="black",
            linewidth=0.25,
            alpha=0.75,
            label=f"M={m}",
        )
    ax.set_xlabel("time since chunk start [s]")
    ax.set_ylabel("frequency [mHz]")
    ax.set_ylim(min_freq_mhz, max_freq_mhz)
    ax.grid(alpha=0.25)
    ax.legend(title="WDM resolution", loc="upper right")
    ax.set_title("1. Top WDM-noise-weighted pixels before clustering")
    out = out_dir / "06_selected_pixels_before_clustering.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_selected_pixels_with_kappa_labels(
    run_dir: Path,
    out_dir: Path,
    min_freq_mhz: float,
    max_freq_mhz: float,
    label_top_n: int,
) -> Path:
    _, chunk_start = load_run(run_dir)
    pixels = pd.read_csv(run_dir / "pixel_clusters_all_resolutions.csv")
    pixels["time_offset_s"] = pixels["time_s"].astype(float) - chunk_start
    pixels["freq_mhz"] = pixels["freq_hz"].astype(float) * 1.0e3
    pixels = pixels.sort_values("selection_power", ascending=False).head(label_top_n).copy()

    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    sc = ax.scatter(
        pixels["time_offset_s"],
        pixels["freq_mhz"],
        c=np.log10(np.maximum(pixels["kappa_min_power"].astype(float), 1.0e-300)),
        s=60,
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
    )
    fig.colorbar(sc, ax=ax, label="log10 minimum kappa/null power")

    for _, row in pixels.iterrows():
        kappa = float(row["kappa_min_power"])
        label = f"{kappa:.1e}" if kappa > 0.0 else "0"
        ax.annotate(
            label,
            (float(row["time_offset_s"]), float(row["freq_mhz"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="black",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.65},
        )

    ax.set_xlabel("time since chunk start [s]")
    ax.set_ylabel("frequency [mHz]")
    ax.set_ylim(min_freq_mhz, max_freq_mhz)
    ax.grid(alpha=0.25)
    ax.set_title(f"1b. Selected pixels tagged with min kappa/null value (top {label_top_n})")
    out = out_dir / f"06b_selected_pixels_min_kappa_labels_top{label_top_n}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def representative_pixel(run_dir: Path, wdm_m: int) -> tuple[pd.Series, int]:
    clusters = pd.read_csv(run_dir / f"wdm_m_{wdm_m}" / "pixel_clusters.csv")
    clustered = clusters[clusters["cluster"].astype(int) >= 0].copy()
    if clustered.empty:
        row = clusters.sort_values("selection_power", ascending=False).iloc[0]
    else:
        row = clustered.sort_values(["cluster_probability", "selection_power"], ascending=False).iloc[0]
    return row, int(row.name)


def plot_kappa_sky_map(run_dir: Path, out_dir: Path, wdm_m: int) -> Path:
    row, row_idx = representative_pixel(run_dir, wdm_m)
    sky = pd.read_csv(run_dir / "healpix_sky_grid.csv")
    responses = np.load(run_dir / f"wdm_m_{wdm_m}" / "kappa_sky_responses.npz")["responses"]
    vals = responses[row_idx]
    positive = vals[np.isfinite(vals) & (vals > 0.0)]
    floor = float(np.nanmin(positive)) if positive.size else 1.0e-300
    image = np.log10(np.maximum(vals, floor))

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    sc = ax.scatter(
        np.degrees(sky["ra"]),
        np.degrees(sky["dec"]),
        c=image,
        s=70,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.25,
    )
    fig.colorbar(sc, ax=ax, label="log10 kappa/null WDM power")
    ax.scatter(
        [np.degrees(float(row["best_ra"]))],
        [np.degrees(float(row["best_dec"]))],
        marker="*",
        s=220,
        color="#ff006e",
        edgecolor="white",
        linewidth=0.8,
        label="minimum null direction",
    )
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(
        f"2. Coronagraphic/null TDI sky scan for one WDM pixel\n"
        f"M={wdm_m}, t={float(row['time_s']):.0f}s, f={1e3 * float(row['freq_hz']):.2f} mHz"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    out = out_dir / "07_kappa_null_sky_scan_one_pixel.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_kappa_minimum_curve(run_dir: Path, out_dir: Path, wdm_m: int) -> Path:
    row, row_idx = representative_pixel(run_dir, wdm_m)
    responses = np.load(run_dir / f"wdm_m_{wdm_m}" / "kappa_sky_responses.npz")["responses"]
    vals = responses[row_idx]
    finite = vals[np.isfinite(vals)]
    positive = finite[finite > 0.0]
    floor = float(np.nanmin(positive)) if positive.size else 1.0e-300
    sorted_vals = np.sort(np.maximum(finite, floor))

    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    ax.plot(np.arange(len(sorted_vals)), sorted_vals, color="#005f73", lw=2)
    ax.scatter([0], [sorted_vals[0]], s=90, color="#ff006e", zorder=5, label="minimum")
    ax.set_yscale("log")
    ax.set_xlabel("sky directions sorted by null power")
    ax.set_ylabel("kappa/null WDM power")
    ax.set_title("3. Null variable minimizes at the best-matching sky direction")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    out = out_dir / "08_kappa_minimum_sorted_sky_directions.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_healpix_tagging(run_dir: Path, out_dir: Path) -> Path:
    pixels = pd.read_csv(run_dir / "pixel_clusters_all_resolutions.csv")
    pixels["best_ra_deg"] = np.degrees(pixels["best_ra"].astype(float))
    pixels["best_dec_deg"] = np.degrees(pixels["best_dec"].astype(float))
    pixels["freq_mhz"] = pixels["freq_hz"].astype(float) * 1.0e3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    sc = axes[0].scatter(
        pixels["best_ra_deg"],
        pixels["best_dec_deg"],
        c=pixels["best_healpix"].astype(int),
        s=22,
        cmap="tab20",
        alpha=0.8,
    )
    fig.colorbar(sc, ax=axes[0], label="best HEALPix pixel")
    axes[0].set_xlabel("best-null RA [deg]")
    axes[0].set_ylabel("best-null Dec [deg]")
    axes[0].set_title("4a. Pixels tagged by best null HEALPix direction (nside=4)")
    axes[0].grid(alpha=0.25)

    counts = pixels["best_healpix"].astype(int).value_counts().head(12).sort_values()
    axes[1].barh([str(i) for i in counts.index], counts.values, color="#2a9d8f")
    axes[1].set_xlabel("number of selected WDM pixels")
    axes[1].set_ylabel("HEALPix index")
    axes[1].set_title("4b. Dominant sky patches among selected pixels")
    axes[1].grid(axis="x", alpha=0.25)

    out = out_dir / "09_healpix_best_sky_tagging.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_hdbscan_events(run_dir: Path, out_dir: Path, min_freq_mhz: float, max_freq_mhz: float) -> Path:
    _, chunk_start = load_run(run_dir)
    pixels = pd.read_csv(run_dir / "merged_pixel_clusters.csv")
    pixels["time_offset_s"] = pixels["time_s"].astype(float) - chunk_start
    pixels["freq_mhz"] = pixels["freq_hz"].astype(float) * 1.0e3
    pixels["event_cluster_int"] = pixels["event_cluster"].astype(int)

    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    noise = pixels[pixels["event_cluster_int"] < 0]
    if len(noise):
        ax.scatter(
            noise["time_offset_s"],
            noise["freq_mhz"],
            s=18,
            marker="x",
            color="#999999",
            alpha=0.45,
            label="HDBSCAN noise/unmerged",
        )
    clustered = pixels[pixels["event_cluster_int"] >= 0]
    sc = ax.scatter(
        clustered["time_offset_s"],
        clustered["freq_mhz"],
        c=clustered["event_cluster_int"],
        s=38,
        cmap="tab10",
        edgecolor="black",
        linewidth=0.25,
        alpha=0.85,
        label="event clusters",
    )
    fig.colorbar(sc, ax=ax, label="HDBSCAN event cluster")
    ax.set_xlabel("time since chunk start [s]")
    ax.set_ylabel("frequency [mHz]")
    ax.set_ylim(min_freq_mhz, max_freq_mhz)
    ax.grid(alpha=0.25)
    ax.set_title("5. HDBSCAN groups tagged pixels into density-based event candidates")
    ax.legend(loc="upper right")
    out = out_dir / "10_hdbscan_event_clusters_time_frequency.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_event_summary(run_dir: Path, out_dir: Path) -> Path:
    summary = pd.read_csv(run_dir / "multires_event_summary.csv")
    summary = summary[summary["passes_min_resolutions"].astype(bool)].copy()
    summary = summary.sort_values("n_pixels", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    labels = [f"C{int(c)}" for c in summary["event_cluster"]]
    axes[0].barh(labels, summary["n_pixels"], color="#264653")
    axes[0].set_xlabel("pixels in merged event")
    axes[0].set_title("6a. Event size")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(labels, summary["n_resolutions"], color="#e76f51")
    axes[1].set_xlabel("supporting WDM resolutions")
    axes[1].set_xlim(0, 3.2)
    axes[1].set_title("6b. Multi-resolution support")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("6. HDBSCAN event candidates after cross-resolution merge")
    out = out_dir / "11_hdbscan_event_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create representative plots for coronagraphic clustering slides.")
    parser.add_argument("--source-id", default="smbhb_m1e6")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-dir", default="test-run/slide_plots")
    parser.add_argument("--wdm-m", type=int, default=512)
    parser.add_argument("--min-freq-mhz", type=float, default=0.01)
    parser.add_argument("--max-freq-mhz", type=float, default=100.0)
    parser.add_argument("--label-top-n", type=int, default=40)
    args = parser.parse_args()

    source_id = args.source_id
    run_dir = PROJECT_ROOT / (
        args.run_dir or f"test-run/coronagraphic/{source_id}_noisy_pytdi_aet_offsource_wdmnorm"
    )
    out_dir = PROJECT_ROOT / args.out_dir / source_id / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_selected_pixels(run_dir, out_dir, args.min_freq_mhz, args.max_freq_mhz),
        plot_selected_pixels_with_kappa_labels(
            run_dir,
            out_dir,
            args.min_freq_mhz,
            args.max_freq_mhz,
            args.label_top_n,
        ),
        plot_kappa_sky_map(run_dir, out_dir, args.wdm_m),
        plot_kappa_minimum_curve(run_dir, out_dir, args.wdm_m),
        plot_healpix_tagging(run_dir, out_dir),
        plot_hdbscan_events(run_dir, out_dir, args.min_freq_mhz, args.max_freq_mhz),
        plot_event_summary(run_dir, out_dir),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
