from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.run_mbhb_aet_chirp_track_morphology import monotonic_chirp_path, tube_from_path


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
SOURCE = "noisy"
PARTS = {
    "low": {"max_jump_bins": 2, "tube_width_bins": 0, "step_penalty": 0.03, "jump_penalty": 0.03},
    "high": {"max_jump_bins": 10, "tube_width_bins": 2, "step_penalty": 0.03, "jump_penalty": 0.004},
}


def load_products(tag: str) -> np.lib.npyio.NpzFile:
    path = OUTPUT_DIR / f"twhitened_wdm_mbhb_{tag}_combined_M4096low_M512high_score_maps.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def positive_score(data: np.lib.npyio.NpzFile, part: str) -> np.ndarray:
    score = np.asarray(data[f"{SOURCE}_{part}_score"], dtype=float)
    return np.where(np.isfinite(score), np.maximum(score, 0.0), 0.0)


def time_overlap_mask(times: np.ndarray, signal: np.ndarray) -> np.ndarray:
    half_width = 0.5 * float(np.nanmedian(np.diff(times))) if len(times) > 1 else 0.0
    starts = times - half_width
    stops = times + half_width
    return (stops >= signal[4]) & (starts <= signal[5])


def anchored_tube(data: np.lib.npyio.NpzFile, part: str) -> dict[str, object]:
    score = positive_score(data, part)
    power = np.asarray(data[f"{SOURCE}_{part}_ae_power"], dtype=float)
    times = np.asarray(data[f"{SOURCE}_{part}_times_rel"], dtype=float)
    freqs = np.asarray(data[f"{SOURCE}_{part}_freqs_hz"], dtype=float)
    signal = np.asarray(data["signal"], dtype=float)

    time_mask = time_overlap_mask(times, signal)
    search_power = np.where(time_mask[None, :], power, -np.inf)
    anchored_to_signal = bool(np.any(np.isfinite(search_power)))
    if not anchored_to_signal:
        search_power = power
    end_cell = tuple(int(v) for v in np.unravel_index(int(np.nanargmax(search_power)), search_power.shape))

    settings = PARTS[part]
    path, path_score, _ = monotonic_chirp_path(
        score,
        max_jump_bins=int(settings["max_jump_bins"]),
        step_penalty=float(settings["step_penalty"]),
        jump_penalty=float(settings["jump_penalty"]),
        end_cell=end_cell,
    )
    tube_mask = tube_from_path(path, score.shape, width_bins=int(settings["tube_width_bins"]))
    product: dict[str, object] = {
        "score": score,
        "power": power,
        "tube_mask": tube_mask,
        "path": path,
        "times": times,
        "freqs": freqs,
        "path_score": float(path_score),
        "anchored_to_signal": anchored_to_signal,
    }
    if np.any(tube_mask):
        fi, ti = np.nonzero(tube_mask)
        product.update(
            {
                "n_tube_pixels": int(np.sum(tube_mask)),
                "n_path_pixels": int(len(path)),
                "sum_score": float(np.sum(score[tube_mask])),
                "sum_ae_power": float(np.sum(power[tube_mask])),
                "max_score": float(np.max(score[tube_mask])),
                "max_ae_power": float(np.max(power[tube_mask])),
                "time_start_rel": float(times[np.min(ti)]),
                "time_stop_rel": float(times[np.max(ti)]),
                "freq_min_hz": float(freqs[np.min(fi)]),
                "freq_max_hz": float(freqs[np.max(fi)]),
            }
        )
    return product


def collect_products() -> dict[str, dict[str, object]]:
    products = {}
    for tag, _ in CASES:
        data = load_products(tag)
        products[tag] = {
            "data": data,
            "low": anchored_tube(data, "low"),
            "high": anchored_tube(data, "high"),
        }
    return products


def score_norm(products: dict[str, dict[str, object]]) -> PowerNorm:
    values = []
    for product in products.values():
        for part in ("low", "high"):
            score = np.asarray(product[part]["score"], dtype=float)
            values.append(score[np.isfinite(score) & (score > 0.0)])
    return PowerNorm(gamma=0.55, vmin=0.0, vmax=float(np.nanpercentile(np.concatenate(values), 99.5)))


def plot_summary(products: dict[str, dict[str, object]]) -> Path:
    output = PLOT_DIR / "twhitened_wdm_mbhb_mass_sweep_noisy_unclustered_vs_tube_summary.png"
    norm = score_norm(products)
    fig, axes = plt.subplots(2, len(CASES), figsize=(21.5, 6.9), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for col, (tag, title) in enumerate(CASES):
        data = products[tag]["data"]
        signal = np.asarray(data["signal"], dtype=float)
        for row, mode in enumerate(("unclustered", "tube-selected")):
            ax = axes[row, col]
            for part in ("low", "high"):
                times = np.asarray(data[f"{SOURCE}_{part}_times_rel"], dtype=float) / 86400.0
                freqs = np.asarray(data[f"{SOURCE}_{part}_freqs_hz"], dtype=float)
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
    fig.suptitle("Noisy injected events: unclustered score maps versus signal-anchored tube selection", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def write_metric_summary(products: dict[str, dict[str, object]]) -> Path:
    rows = []
    for tag, _ in CASES:
        metrics = pd.read_csv(OUTPUT_DIR / f"twhitened_wdm_mbhb_{tag}_combined_M4096low_M512high_score_energy.csv")
        unclustered = metrics[
            (metrics["source"] == SOURCE)
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
            "anchored_to_signal": True,
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
            tube["anchored_to_signal"] = tube["anchored_to_signal"] and bool(product.get("anchored_to_signal", False))
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
                "anchored_to_signal": bool(tube["anchored_to_signal"]),
            }
        )
    output = OUTPUT_DIR / "twhitened_wdm_mbhb_mass_sweep_noisy_unclustered_vs_tube_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    products = collect_products()
    print(plot_summary(products))
    print(write_metric_summary(products))


if __name__ == "__main__":
    main()
