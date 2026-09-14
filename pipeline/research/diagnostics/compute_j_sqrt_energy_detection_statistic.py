from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
NOISE_CSV = OUTPUT_DIR / "noise_only_ae_cluster_background_summary.csv"
SIGNAL_CSV = OUTPUT_DIR / "twhitened_wdm_mbhb_mass_sweep_jc_channel_energy_summary.csv"
MASS_ORDER = ["1e4", "1e4p5", "1e5", "1e6", "1e7", "1e7p5", "1e8", "1e8p5"]
MASS_LABELS = {
    "1e4": r"$10^4$",
    "1e4p5": r"$10^{4.5}$",
    "1e5": r"$10^5$",
    "1e6": r"$10^6$",
    "1e7": r"$10^7$",
    "1e7p5": r"$10^{7.5}$",
    "1e8": r"$10^8$",
    "1e8p5": r"$10^{8.5}$",
}
DISTANCE_LABELS = {
    "baseline": "6230 Mpc",
    "double_distance": "12460 Mpc",
}
DISTANCE_COLORS = {
    "baseline": "#dc2626",
    "double_distance": "#2563eb",
}


def statistic(j: pd.Series | np.ndarray, energy: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(j, dtype=float) * np.sqrt(np.maximum(np.asarray(energy, dtype=float), 0.0))


def tail_fraction(value: float, background: np.ndarray) -> float:
    return float(np.mean(background >= value))


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    noise = pd.read_csv(NOISE_CSV)
    noise = noise[noise["part"] == "combined_low_high"].copy()
    noise["detection_statistic"] = statistic(noise["jaccard"], noise["ae_total_power"])
    noise["sample_type"] = "noise_only"

    signals = pd.read_csv(SIGNAL_CSV)
    signals = signals[
        (signals["distance_case"].isin(DISTANCE_LABELS))
        & (signals["source"] == "noisy")
    ].copy()
    signals = signals[signals["base_tag"].isin(MASS_ORDER)].copy()
    signals["detection_statistic"] = statistic(signals["jaccard"], signals["ae_total_power"])
    signals["noise_tail_fraction"] = [
        tail_fraction(value, noise["detection_statistic"].to_numpy(dtype=float))
        for value in signals["detection_statistic"]
    ]
    signals["passes_noise_max"] = signals["detection_statistic"] > float(noise["detection_statistic"].max())
    signals["sample_type"] = "gw_plus_noise"
    signals["mass_order"] = signals["base_tag"].map({tag: i for i, tag in enumerate(MASS_ORDER)})
    signals = signals.sort_values(["distance_case", "mass_order"])

    noise_out = noise[
        [
            "sample_type",
            "seed",
            "jaccard",
            "ae_total_power",
            "detection_statistic",
            "a_power_fraction",
            "union_coherence",
        ]
    ].copy()
    noise_out["mass"] = ""
    noise_out["noise_tail_fraction"] = np.nan
    noise_out["passes_noise_max"] = False
    signal_out = signals[
        [
            "sample_type",
            "base_tag",
            "distance_case",
            "jaccard",
            "ae_total_power",
            "detection_statistic",
            "a_power_fraction",
            "union_coherence",
            "noise_tail_fraction",
            "passes_noise_max",
        ]
    ].rename(columns={"base_tag": "mass"})
    signal_out["seed"] = np.nan
    combined = pd.concat([noise_out, signal_out], ignore_index=True, sort=False)
    out_csv = OUTPUT_DIR / "j_sqrt_energy_detection_statistic.csv"
    combined.to_csv(out_csv, index=False)

    x = np.arange(len(MASS_ORDER), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), constrained_layout=True)
    noise_stat = noise["detection_statistic"].to_numpy(dtype=float)
    for ax in axes:
        ax.axhspan(float(noise_stat.min()), float(noise_stat.max()), color="0.7", alpha=0.25, label="noise-only min-max")
        ax.axhline(float(np.median(noise_stat)), color="0.35", ls="--", lw=1.0, label="noise-only median")
        for distance_case, label in DISTANCE_LABELS.items():
            case = signals[signals["distance_case"] == distance_case].sort_values("mass_order")
            ax.plot(
                x,
                case["detection_statistic"],
                marker="o",
                lw=1.5,
                color=DISTANCE_COLORS[distance_case],
                label=f"GW + noise, {label}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MASS_LABELS[tag] for tag in MASS_ORDER], rotation=35, ha="right")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$D = J \sqrt{E_A + E_E}$")
    axes[0].set_title("linear scale")
    axes[1].set_yscale("log")
    axes[1].set_title("log scale")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(r"Detection statistic from A/E tube overlap and energy: $D = J\sqrt{E}$", fontsize=13)
    out_plot = PLOT_DIR / "j_sqrt_energy_detection_statistic.png"
    fig.savefig(out_plot, dpi=190)
    plt.close(fig)

    print(out_csv)
    print(out_plot)
    print("noise-only D min/median/max =", float(noise_stat.min()), float(np.median(noise_stat)), float(noise_stat.max()))
    print(
        signals[
            [
                "distance_case",
                "base_tag",
                "jaccard",
                "ae_total_power",
                "detection_statistic",
                "noise_tail_fraction",
                "passes_noise_max",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
