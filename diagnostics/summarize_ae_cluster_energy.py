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
INPUTS = {
    "baseline": OUTPUT_DIR / "twhitened_wdm_mbhb_mass_sweep_ae_separate_cluster_correlation.csv",
    "double_distance": OUTPUT_DIR / "twhitened_wdm_mbhb_mass_sweep_ae_separate_cluster_correlation_double_distance.csv",
}
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


def base_tag(tag: str) -> str:
    return tag.removesuffix("_d2")


def load_summary() -> pd.DataFrame:
    frames = []
    for distance_case, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df = df[df["part"] == "combined_low_high"].copy()
        df["distance_case"] = distance_case
        df["base_tag"] = df["tag"].map(base_tag)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["ae_total_power"] = out["a_sum_power"] + out["e_sum_power"]
    out["a_power_fraction"] = out["a_sum_power"] / out["ae_total_power"].replace(0.0, np.nan)
    out["e_over_a_power"] = out["e_sum_power"] / out["a_sum_power"].replace(0.0, np.nan)
    out["log10_a_power"] = np.log10(np.maximum(out["a_sum_power"], 1e-300))
    out["log10_e_power"] = np.log10(np.maximum(out["e_sum_power"], 1e-300))
    out["log10_ae_total_power"] = np.log10(np.maximum(out["ae_total_power"], 1e-300))
    out["channel_power_asymmetry"] = (out["a_sum_power"] - out["e_sum_power"]) / out["ae_total_power"].replace(0.0, np.nan)
    out["mass_order"] = out["base_tag"].map({tag: i for i, tag in enumerate(MASS_ORDER)})
    return out.sort_values(["source", "distance_case", "mass_order"]).reset_index(drop=True)


def plot_energy(summary: pd.DataFrame) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output = PLOT_DIR / "twhitened_wdm_mbhb_mass_sweep_jc_channel_energy_summary.png"
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), sharex=True, constrained_layout=True)
    x = np.arange(len(MASS_ORDER), dtype=float)
    xticklabels = [MASS_LABELS[tag] for tag in MASS_ORDER]
    colors = {"baseline": "#2563eb", "double_distance": "#dc2626"}
    linestyles = {"gw_only": "-", "noisy": "--"}

    for source in ("gw_only", "noisy"):
        for distance_case in ("baseline", "double_distance"):
            df = summary[(summary["source"] == source) & (summary["distance_case"] == distance_case)].set_index("base_tag").loc[MASS_ORDER]
            label = f"{source}, {distance_case}"
            style = {"color": colors[distance_case], "ls": linestyles[source], "marker": "o", "ms": 4, "label": label}
            axes[0, 0].plot(x, df["jaccard"], **style)
            axes[0, 1].plot(x, df["union_coherence"], **style)
            axes[0, 2].plot(x, df["a_power_fraction"], **style)
            axes[1, 0].plot(x, df["log10_a_power"], **style)
            axes[1, 1].plot(x, df["log10_e_power"], **style)
            axes[1, 2].plot(x, df["log10_ae_total_power"], **style)

    titles = [
        "J: A/E tube overlap",
        "C: complex coherence",
        "A energy fraction",
        r"$\log_{10} E_A$",
        r"$\log_{10} E_E$",
        r"$\log_{10}(E_A+E_E)$",
    ]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=35, ha="right")
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 1].set_ylim(-0.03, 1.03)
    axes[0, 2].set_ylim(-0.03, 1.03)
    axes[0, 1].legend(fontsize=8, loc="lower left")
    fig.suptitle("A/E cluster consistency and per-channel T-whitened energy estimates", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def main() -> None:
    summary = load_summary()
    out_csv = OUTPUT_DIR / "twhitened_wdm_mbhb_mass_sweep_jc_channel_energy_summary.csv"
    cols = [
        "distance_case",
        "source",
        "base_tag",
        "jaccard",
        "union_coherence",
        "n_a_tube",
        "n_e_tube",
        "n_intersection",
        "n_union",
        "a_sum_power",
        "e_sum_power",
        "ae_total_power",
        "a_power_fraction",
        "e_over_a_power",
        "channel_power_asymmetry",
        "log10_a_power",
        "log10_e_power",
        "log10_ae_total_power",
        "a_sum_score",
        "e_sum_score",
    ]
    summary[cols].to_csv(out_csv, index=False)
    plot_path = plot_energy(summary)
    print(out_csv)
    print(plot_path)


if __name__ == "__main__":
    main()
