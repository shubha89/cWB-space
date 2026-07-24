from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
CASES = [
    ("1e4", r"$10^4$"),
    ("1e4p5", r"$10^{4.5}$"),
    ("1e5", r"$10^5$"),
    ("1e6", r"$10^6$"),
    ("1e7", r"$10^7$"),
    ("1e7p5", r"$10^{7.5}$"),
    ("1e8", r"$10^8$"),
    ("1e8p5", r"$10^{8.5}$"),
]
SOURCES = [("noisy", "noisy"), ("gw_only", "GW only")]


def load_products(tag: str) -> np.lib.npyio.NpzFile:
    path = OUTPUT_DIR / f"twhitened_wdm_mbhb_{tag}_combined_M4096low_M512high_score_maps.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing combined score map: {path}")
    return np.load(path)


def positive_score(data: np.lib.npyio.NpzFile, source: str, part: str) -> np.ndarray:
    score = np.asarray(data[f"{source}_{part}_score"], dtype=float)
    return np.where(np.isfinite(score), np.maximum(score, 0.0), np.nan)


def score_limits(products: dict[str, np.lib.npyio.NpzFile]) -> tuple[float, float]:
    scores = []
    for data in products.values():
        for source, _ in SOURCES:
            for part in ("low", "high"):
                score = positive_score(data, source, part)
                scores.append(score[np.isfinite(score) & (score > 0.0)])
    all_scores = np.concatenate(scores)
    return 0.0, float(np.nanpercentile(all_scores, 99.5))


def plot_panel(ax: plt.Axes, data: np.lib.npyio.NpzFile, source: str, norm: PowerNorm) -> None:
    image = None
    for part in ("low", "high"):
        times = np.asarray(data[f"{source}_{part}_times_rel"], dtype=float) / 86400.0
        freqs = np.asarray(data[f"{source}_{part}_freqs_hz"], dtype=float)
        score = positive_score(data, source, part)
        image = ax.pcolormesh(times, freqs, score, shading="auto", cmap="viridis", norm=norm)
    ax.axhline(float(data["split_hz"]), color="white", lw=0.65, alpha=0.7)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e-1)
    ax.set_xlim(0.0, 150000.0 / 86400.0)
    ax.grid(True, which="both", alpha=0.12)
    return image


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    products = {tag: load_products(tag) for tag, _ in CASES}
    vmin, vmax = score_limits(products)
    norm = PowerNorm(gamma=0.55, vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        len(SOURCES),
        len(CASES),
        figsize=(18.0, 6.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for col, (tag, title) in enumerate(CASES):
        for row, (source, source_label) in enumerate(SOURCES):
            ax = axes[row, col]
            image = plot_panel(ax, products[tag], source, norm)
            if row == 0:
                ax.set_title(rf"$M={title}\ M_\odot$", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{source_label}\nFrequency [Hz]")
            if row == len(SOURCES) - 1:
                ax.set_xlabel("Time [days]")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.01)
        cbar.set_label("positive frequency-normalized score")

    output = PLOT_DIR / "twhitened_wdm_mbhb_mass_sweep_combined_frequency_scores.png"
    fig.suptitle("MBHB mass sweep, combined WDM frequency-normalized scores", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
