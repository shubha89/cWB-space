from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.plot_mbhb_mass_sweep_tube_clustering import CASES, collect_products, positive_score, score_norm


OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"


def plot_summary(products: dict[str, dict[str, object]]) -> Path:
    output = PLOT_DIR / "twhitened_wdm_mbhb_mass_sweep_unclustered_vs_tube_summary.png"
    norm = score_norm(products)
    fig, axes = plt.subplots(2, len(CASES), figsize=(21.5, 6.9), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for col, (tag, title) in enumerate(CASES):
        data = products[tag]["data"]
        signal = np.asarray(data["signal"], dtype=float)
        for row, mode in enumerate(("unclustered", "tube-selected")):
            ax = axes[row, col]
            for part in ("low", "high"):
                times = np.asarray(data[f"gw_only_{part}_times_rel"], dtype=float) / 86400.0
                freqs = np.asarray(data[f"gw_only_{part}_freqs_hz"], dtype=float)
                score = positive_score(data, part)
                if mode == "tube-selected":
                    score = np.where(np.asarray(products[tag][part]["tube_mask"], dtype=bool), score, np.nan)
                image = ax.pcolormesh(times, freqs, score, shading="auto", cmap="viridis", norm=norm)
                if mode == "tube-selected":
                    path = np.asarray(products[tag][part]["path"], dtype=int)
                    if len(path):
                        ax.plot(times[path[:, 1]], freqs[path[:, 0]], color="white", lw=0.75, alpha=0.9)
            ax.axvspan(signal[4] / 86400.0, signal[5] / 86400.0, color="cyan", alpha=0.12, lw=0)
            ax.axhline(float(data["split_hz"]), color="white", lw=0.6, alpha=0.7)
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1e-1)
            ax.set_xlim(0.0, 150000.0 / 86400.0)
            ax.grid(True, which="both", alpha=0.11)
            if row == 0:
                ax.set_title(rf"$M={title}\ M_\odot$", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{mode}\nFrequency [Hz]")
            if row == 1:
                ax.set_xlabel("Time [days]")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.01)
        cbar.set_label("positive frequency-normalized score")
    fig.suptitle("GW-only events: unclustered score maps versus signal-anchored tube selection", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def write_metric_summary(products: dict[str, dict[str, object]]) -> Path:
    rows = []
    for tag, _ in CASES:
        unclustered_path = OUTPUT_DIR / f"twhitened_wdm_mbhb_{tag}_combined_M4096low_M512high_score_energy.csv"
        metrics = pd.read_csv(unclustered_path)
        unclustered = metrics[
            (metrics["source"] == "gw_only")
            & (metrics["component"] == "combined_frequency_split")
            & (metrics["region"] == "signal_window")
        ].iloc[0]
        tube = {
            "n_tube_pixels": 0,
            "sum_score": 0.0,
            "sum_ae_power": 0.0,
            "path_score": 0.0,
            "max_score": 0.0,
            "max_ae_power": 0.0,
        }
        for part in ("low", "high"):
            product = products[tag][part]
            tube["n_tube_pixels"] += int(product.get("n_tube_pixels", 0))
            tube["sum_score"] += float(product.get("sum_score", 0.0))
            tube["sum_ae_power"] += float(product.get("sum_ae_power", 0.0))
            path_score = float(product.get("path_score", 0.0))
            tube["path_score"] += path_score if np.isfinite(path_score) else 0.0
            max_score = float(product.get("max_score", 0.0))
            max_power = float(product.get("max_ae_power", 0.0))
            tube["max_score"] = max(tube["max_score"], max_score if np.isfinite(max_score) else 0.0)
            tube["max_ae_power"] = max(tube["max_ae_power"], max_power if np.isfinite(max_power) else 0.0)
        rows.append(
            {
                "tag": tag,
                "unclustered_signal_pixels": int(unclustered["n_pixels"]),
                "unclustered_sum_score": float(unclustered["sum_score"]),
                "unclustered_mean_score": float(unclustered["mean_score"]),
                "unclustered_max_score": float(unclustered["max_score"]),
                "unclustered_ae_power": float(unclustered["sum_ae_twhitened_power"]),
                "tube_pixels": int(tube["n_tube_pixels"]),
                "tube_sum_score": float(tube["sum_score"]),
                "tube_path_score": float(tube["path_score"]),
                "tube_max_score": float(tube["max_score"]),
                "tube_ae_power": float(tube["sum_ae_power"]),
                "tube_score_fraction_of_unclustered_signal": (
                    float(tube["sum_score"]) / float(unclustered["sum_score"])
                    if float(unclustered["sum_score"]) > 0.0
                    else np.nan
                ),
                "tube_power_fraction_of_unclustered_signal": (
                    float(tube["sum_ae_power"]) / float(unclustered["sum_ae_twhitened_power"])
                    if float(unclustered["sum_ae_twhitened_power"]) > 0.0
                    else np.nan
                ),
            }
        )
    output = OUTPUT_DIR / "twhitened_wdm_mbhb_mass_sweep_unclustered_vs_tube_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    products = collect_products()
    plot_path = plot_summary(products)
    csv_path = write_metric_summary(products)
    print(plot_path)
    print(csv_path)


if __name__ == "__main__":
    main()
