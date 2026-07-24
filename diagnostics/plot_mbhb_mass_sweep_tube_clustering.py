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
    score = np.asarray(data[f"gw_only_{part}_score"], dtype=float)
    return np.where(np.isfinite(score), np.maximum(score, 0.0), 0.0)


def time_overlap_mask(times: np.ndarray, signal: np.ndarray) -> np.ndarray:
    if len(times) > 1:
        half_width = 0.5 * float(np.nanmedian(np.diff(times)))
    else:
        half_width = 0.0
    starts = times - half_width
    stops = times + half_width
    return (stops >= signal[4]) & (starts <= signal[5])


def anchored_tube(data: np.lib.npyio.NpzFile, part: str) -> dict[str, np.ndarray | float | int | bool]:
    score = positive_score(data, part)
    power = np.asarray(data[f"gw_only_{part}_ae_power"], dtype=float)
    times = np.asarray(data[f"gw_only_{part}_times_rel"], dtype=float)
    freqs = np.asarray(data[f"gw_only_{part}_freqs_hz"], dtype=float)
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
    if not np.any(tube_mask):
        return {
            "score": score,
            "power": power,
            "tube_mask": tube_mask,
            "path": path,
            "times": times,
            "freqs": freqs,
            "path_score": float(path_score),
            "anchored_to_signal": anchored_to_signal,
        }

    fi, ti = np.nonzero(tube_mask)
    return {
        "score": score,
        "power": power,
        "tube_mask": tube_mask,
        "path": path,
        "times": times,
        "freqs": freqs,
        "path_score": float(path_score),
        "anchored_to_signal": anchored_to_signal,
        "time_start_rel": float(times[np.min(ti)]),
        "time_stop_rel": float(times[np.max(ti)]),
        "freq_min_hz": float(freqs[np.min(fi)]),
        "freq_max_hz": float(freqs[np.max(fi)]),
        "sum_score": float(np.sum(score[tube_mask])),
        "sum_ae_power": float(np.sum(power[tube_mask])),
        "n_tube_pixels": int(np.sum(tube_mask)),
        "n_path_pixels": int(len(path)),
        "max_score": float(np.max(score[tube_mask])),
        "max_ae_power": float(np.max(power[tube_mask])),
    }


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
    all_values = np.concatenate(values)
    return PowerNorm(gamma=0.55, vmin=0.0, vmax=float(np.nanpercentile(all_values, 99.5)))


def plot_unclustered(products: dict[str, dict[str, object]], norm: PowerNorm) -> Path:
    output = PLOT_DIR / "twhitened_wdm_mbhb_mass_sweep_gw_only_unclustered_scores.png"
    fig, axes = plt.subplots(1, len(CASES), figsize=(21.0, 3.8), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ax, (tag, title) in zip(axes, CASES):
        data = products[tag]["data"]
        for part in ("low", "high"):
            times = np.asarray(data[f"gw_only_{part}_times_rel"], dtype=float) / 86400.0
            freqs = np.asarray(data[f"gw_only_{part}_freqs_hz"], dtype=float)
            image = ax.pcolormesh(times, freqs, positive_score(data, part), shading="auto", cmap="viridis", norm=norm)
        signal = np.asarray(data["signal"], dtype=float)
        ax.axvspan(signal[4] / 86400.0, signal[5] / 86400.0, color="cyan", alpha=0.12, lw=0)
        ax.axhline(float(data["split_hz"]), color="white", lw=0.65, alpha=0.7)
        ax.set_title(rf"$M={title}\ M_\odot$", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1e-1)
        ax.set_xlim(0.0, 150000.0 / 86400.0)
        ax.grid(True, which="both", alpha=0.12)
        ax.set_xlabel("Time [days]")
    axes[0].set_ylabel("Frequency [Hz]")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.86, pad=0.01)
        cbar.set_label("positive frequency-normalized score")
    fig.suptitle("GW-only mass sweep, unclustered frequency-normalized scores", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def plot_tube_clustered(products: dict[str, dict[str, object]], norm: PowerNorm) -> Path:
    output = PLOT_DIR / "twhitened_wdm_mbhb_mass_sweep_gw_only_tube_clustered_scores.png"
    fig, axes = plt.subplots(1, len(CASES), figsize=(21.0, 3.8), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ax, (tag, title) in zip(axes, CASES):
        data = products[tag]["data"]
        for part in ("low", "high"):
            times = np.asarray(data[f"gw_only_{part}_times_rel"], dtype=float) / 86400.0
            freqs = np.asarray(data[f"gw_only_{part}_freqs_hz"], dtype=float)
            score = np.where(np.asarray(products[tag][part]["tube_mask"], dtype=bool), positive_score(data, part), np.nan)
            image = ax.pcolormesh(times, freqs, score, shading="auto", cmap="viridis", norm=norm)
            path = np.asarray(products[tag][part]["path"], dtype=int)
            if len(path):
                ax.plot(times[path[:, 1]], freqs[path[:, 0]], color="white", lw=0.8, alpha=0.9)
        signal = np.asarray(data["signal"], dtype=float)
        ax.axvspan(signal[4] / 86400.0, signal[5] / 86400.0, color="cyan", alpha=0.12, lw=0)
        ax.axhline(float(data["split_hz"]), color="white", lw=0.65, alpha=0.7)
        ax.set_title(rf"$M={title}\ M_\odot$", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1e-1)
        ax.set_xlim(0.0, 150000.0 / 86400.0)
        ax.grid(True, which="both", alpha=0.12)
        ax.set_xlabel("Time [days]")
    axes[0].set_ylabel("Frequency [Hz]")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.86, pad=0.01)
        cbar.set_label("tube-selected positive score")
    fig.suptitle("GW-only mass sweep, signal-anchored monotonic tube selection", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def write_summary(products: dict[str, dict[str, object]]) -> Path:
    rows = []
    for tag, _ in CASES:
        data = products[tag]["data"]
        signal = np.asarray(data["signal"], dtype=float)
        total = {
            "tag": tag,
            "component": "combined_low_high",
            "signal_start_rel_s": float(signal[4]),
            "signal_stop_rel_s": float(signal[5]),
            "n_tube_pixels": 0,
            "n_path_pixels": 0,
            "sum_score": 0.0,
            "sum_ae_power": 0.0,
            "path_score": 0.0,
            "max_score": 0.0,
            "max_ae_power": 0.0,
            "anchored_to_signal": True,
        }
        for part in ("low", "high"):
            product = products[tag][part]
            row = {
                "tag": tag,
                "component": part,
                "signal_start_rel_s": float(signal[4]),
                "signal_stop_rel_s": float(signal[5]),
                "n_tube_pixels": int(product.get("n_tube_pixels", 0)),
                "n_path_pixels": int(product.get("n_path_pixels", 0)),
                "sum_score": float(product.get("sum_score", 0.0)),
                "sum_ae_power": float(product.get("sum_ae_power", 0.0)),
                "path_score": float(product.get("path_score", np.nan)),
                "max_score": float(product.get("max_score", np.nan)),
                "max_ae_power": float(product.get("max_ae_power", np.nan)),
                "time_start_rel_s": float(product.get("time_start_rel", np.nan)),
                "time_stop_rel_s": float(product.get("time_stop_rel", np.nan)),
                "freq_min_hz": float(product.get("freq_min_hz", np.nan)),
                "freq_max_hz": float(product.get("freq_max_hz", np.nan)),
                "anchored_to_signal": bool(product.get("anchored_to_signal", False)),
            }
            rows.append(row)
            total["n_tube_pixels"] += row["n_tube_pixels"]
            total["n_path_pixels"] += row["n_path_pixels"]
            total["sum_score"] += row["sum_score"]
            total["sum_ae_power"] += row["sum_ae_power"]
            total["path_score"] += row["path_score"] if np.isfinite(row["path_score"]) else 0.0
            total["max_score"] = max(total["max_score"], row["max_score"] if np.isfinite(row["max_score"]) else 0.0)
            total["max_ae_power"] = max(total["max_ae_power"], row["max_ae_power"] if np.isfinite(row["max_ae_power"]) else 0.0)
            total["anchored_to_signal"] = total["anchored_to_signal"] and row["anchored_to_signal"]
        rows.append(total)
    output = OUTPUT_DIR / "twhitened_wdm_mbhb_mass_sweep_gw_only_tube_cluster_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    products = collect_products()
    norm = score_norm(products)
    unclustered = plot_unclustered(products, norm)
    tube_clustered = plot_tube_clustered(products, norm)
    summary = write_summary(products)
    print(unclustered)
    print(tube_clustered)
    print(summary)


if __name__ == "__main__":
    main()
