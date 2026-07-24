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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    noise = pd.read_csv(NOISE_CSV)
    noise = noise[noise["part"] == "combined_low_high"].copy()
    signals = pd.read_csv(SIGNAL_CSV)
    signals = signals[signals["distance_case"] == "baseline"].copy()
    signals = signals[signals["base_tag"].isin(MASS_ORDER)].copy()
    signals["mass_order"] = signals["base_tag"].map({tag: i for i, tag in enumerate(MASS_ORDER)})
    signals = signals.sort_values(["source", "mass_order"])
    return noise, signals


def empirical_tail(value: float, background: np.ndarray) -> float:
    return float(np.mean(background >= value))


def write_comparison(noise: pd.DataFrame, signals: pd.DataFrame) -> Path:
    noise_j = noise["jaccard"].to_numpy(dtype=float)
    noise_c = noise["union_coherence"].to_numpy(dtype=float)
    noise_e = noise["ae_total_power"].to_numpy(dtype=float)
    noise_loge = np.log10(np.maximum(noise_e, 1e-300))

    rows = []
    for _, row in signals.iterrows():
        loge = float(np.log10(max(row["ae_total_power"], 1e-300)))
        rows.append(
            {
                "source": row["source"],
                "mass": row["base_tag"],
                "jaccard": float(row["jaccard"]),
                "coherence": float(row["union_coherence"]),
                "ae_total_power": float(row["ae_total_power"]),
                "log10_ae_total_power": loge,
                "a_power_fraction": float(row["a_power_fraction"]),
                "j_noise_tail_fraction": empirical_tail(float(row["jaccard"]), noise_j),
                "c_noise_tail_fraction": empirical_tail(float(row["union_coherence"]), noise_c),
                "e_noise_tail_fraction": empirical_tail(float(row["ae_total_power"]), noise_e),
                "loge_noise_tail_fraction": empirical_tail(loge, noise_loge),
                "passes_noise_max_j": bool(float(row["jaccard"]) > np.max(noise_j)),
                "passes_noise_max_c": bool(float(row["union_coherence"]) > np.max(noise_c)),
                "passes_noise_max_e": bool(float(row["ae_total_power"]) > np.max(noise_e)),
            }
        )
    output = OUTPUT_DIR / "signal_vs_noise_jce_comparison.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def plot_comparison(noise: pd.DataFrame, signals: pd.DataFrame) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output = PLOT_DIR / "signal_vs_noise_jce_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    x = np.arange(len(MASS_ORDER), dtype=float)
    labels = [MASS_LABELS[tag] for tag in MASS_ORDER]
    source_styles = {
        "gw_only": {"label": "GW only", "color": "#2563eb", "marker": "o"},
        "noisy": {"label": "GW + noise", "color": "#dc2626", "marker": "s"},
    }

    noise_stats = {
        "jaccard": noise["jaccard"].to_numpy(dtype=float),
        "union_coherence": noise["union_coherence"].to_numpy(dtype=float),
        "log10_ae_total_power": np.log10(np.maximum(noise["ae_total_power"].to_numpy(dtype=float), 1e-300)),
    }
    fields = [
        ("jaccard", "J: A/E tube overlap", None),
        ("union_coherence", "C: complex coherence", None),
        ("log10_ae_total_power", r"$\log_{10}(E_A+E_E)$", None),
    ]
    for ax, (field, title, _) in zip(axes, fields):
        bg = noise_stats[field]
        ax.axhspan(np.min(bg), np.max(bg), color="0.7", alpha=0.22, label="noise-only min-max")
        ax.axhline(np.median(bg), color="0.35", ls="--", lw=1.0, label="noise-only median")
        for source, style in source_styles.items():
            sub = signals[signals["source"] == source].set_index("base_tag").loc[MASS_ORDER]
            y = np.log10(np.maximum(sub["ae_total_power"], 1e-300)) if field == "log10_ae_total_power" else sub[field]
            ax.plot(x, y, color=style["color"], marker=style["marker"], lw=1.4, ms=4, label=style["label"])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Signals versus blind noise-only background: J, C, and tube energy", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def main() -> None:
    noise, signals = load_inputs()
    csv_path = write_comparison(noise, signals)
    plot_path = plot_comparison(noise, signals)
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
