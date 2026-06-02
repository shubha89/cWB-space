"""Plot selected coronagraphic pixels on the native WDM bin grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_one_smbhb as proto  # noqa: E402


def reconstruct_ae_wdm(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = proto.RunConfig(**summary["config"])
    _, _, resample_poly, _, WDM = proto.require_dependencies()

    work_dir = PROJECT_ROOT / config.work_dir
    t_raw, link_raw = proto.load_links(work_dir, config.source)
    center_s = float(t_raw[0]) + 0.5 * (float(t_raw[-1]) - float(t_raw[0]))
    t, links, dt = proto.crop_and_downsample_links(
        t_raw,
        link_raw,
        center_s=center_s,
        duration_s=config.chunk_duration_s,
        target_dt_s=config.target_dt_s,
        resample_poly=resample_poly,
    )
    geometry = proto.LISAGeometry.equilateral(proto.LISA_ARM_M / proto.C_M_PER_S)
    michelson = proto.compute_michelson_response_from_links(
        links,
        dt=dt,
        geometry=geometry,
        generation="second",
        channels="aet",
        output_domain="time",
    )
    a_t = np.asarray(michelson["time"]["A"])
    e_t = np.asarray(michelson["time"]["E"])
    power_a, freqs, times = proto.wdm_power(a_t, dt=dt, t0=float(t[0]), config=config, WDM=WDM)
    power_e, _, _ = proto.wdm_power(e_t, dt=dt, t0=float(t[0]), config=config, WDM=WDM)
    return power_a + power_e, freqs, times


def plot_pixels(run_dir: Path) -> tuple[Path, Path]:
    clusters = pd.read_csv(run_dir / "pixel_clusters.csv")
    ae_power, _, _ = reconstruct_ae_wdm(run_dir)
    background = np.log10(np.maximum(ae_power, 1.0e-300))

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids = sorted(int(value) for value in clusters["cluster"].unique() if int(value) >= 0)
    palette = plt.get_cmap("tab10")
    cluster_colors = {cluster_id: palette(i % palette.N) for i, cluster_id in enumerate(cluster_ids)}
    colors = {-1: "#f0f0f0", **cluster_colors}
    markers = {-1: "x", 0: "o", 1: "s", 2: "^", 3: "D", 4: "P", 5: "v", 6: "X", 7: "*"}
    background_cmap = "magma"

    all_path = plot_dir / "selected_pixels_wdm_resolution.png"
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(background, origin="lower", aspect="auto", cmap=background_cmap)
    fig.colorbar(im, ax=ax, label="log10(A+E WDM power)")
    for cluster_id in sorted(clusters["cluster"].unique()):
        cluster_id = int(cluster_id)
        rows = clusters[clusters["cluster"] == cluster_id]
        label = "noise / unclustered" if cluster_id == -1 else f"coherent cluster {cluster_id}"
        if cluster_id == -1:
            ax.scatter(
                rows["time_idx"],
                rows["freq_idx"],
                s=60,
                marker=markers[-1],
                color=colors[-1],
                linewidths=1.2,
                label=f"{label} (n={len(rows)})",
            )
        else:
            ax.scatter(
                rows["time_idx"],
                rows["freq_idx"],
                s=78,
                marker=markers.get(cluster_id, "o"),
                edgecolors="white",
                linewidths=1.1,
                facecolors=colors.get(cluster_id, "#9467bd"),
                label=f"{label} (n={len(rows)})",
            )
    ax.set_xlabel("WDM time bin index")
    ax.set_ylabel("WDM frequency bin index")
    ax.set_title("Selected coronagraphic pixels at native WDM resolution")
    ax.set_xlim(-1, ae_power.shape[1])
    ax.set_ylim(-1, ae_power.shape[0])
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(all_path, dpi=180)
    plt.close(fig)

    coherent_path = plot_dir / "coherent_pixels_wdm_resolution.png"
    coherent = clusters[clusters["cluster"] >= 0]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(background, origin="lower", aspect="auto", cmap=background_cmap)
    fig.colorbar(im, ax=ax, label="log10(A+E WDM power)")
    for cluster_id in sorted(coherent["cluster"].unique()):
        cluster_id = int(cluster_id)
        rows = coherent[coherent["cluster"] == cluster_id]
        ax.scatter(
            rows["time_idx"],
            rows["freq_idx"],
            s=82,
            marker=markers.get(cluster_id, "o"),
            edgecolors="white",
            linewidths=1.1,
            facecolors=colors.get(cluster_id, "#9467bd"),
            label=f"coherent cluster {cluster_id} (n={len(rows)})",
        )
    ax.set_xlabel("WDM time bin index")
    ax.set_ylabel("WDM frequency bin index")
    ax.set_title("Coherent coronagraphic pixel clusters at native WDM resolution")
    ax.set_xlim(-1, ae_power.shape[1])
    ax.set_ylim(-1, ae_power.shape[0])
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(coherent_path, dpi=180)
    plt.close(fig)
    return all_path, coherent_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot coronagraphic selected pixels on the WDM grid.")
    parser.add_argument("--run-dir", default="choronagraphic-cluster/runs/smbhb_00000")
    args = parser.parse_args()
    all_path, coherent_path = plot_pixels(PROJECT_ROOT / args.run_dir)
    print(all_path)
    print(coherent_path)


if __name__ == "__main__":
    main()
