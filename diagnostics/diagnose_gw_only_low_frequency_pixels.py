from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import numpy as np
import pandas as pd

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


def load_map(tag: str) -> np.lib.npyio.NpzFile:
    path = OUTPUT_DIR / f"twhitened_wdm_mbhb_{tag}_combined_M4096low_M512high_score_maps.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def overlap_mask(times: np.ndarray, signal: np.ndarray) -> np.ndarray:
    dt = float(np.nanmedian(np.diff(times))) if len(times) > 1 else 0.0
    starts = times - 0.5 * dt
    stops = times + 0.5 * dt
    return (stops >= signal[4]) & (starts <= signal[5])


def top_pixel_rows(tag: str, data: np.lib.npyio.NpzFile, *, n: int = 16) -> list[dict[str, float | int | str | bool]]:
    times = np.asarray(data["gw_only_low_times_rel"], dtype=float)
    freqs = np.asarray(data["gw_only_low_freqs_hz"], dtype=float)
    score = np.asarray(data["gw_only_low_score"], dtype=float)
    power = np.asarray(data["gw_only_low_ae_power"], dtype=float)
    signal = np.asarray(data["signal"], dtype=float)
    overlaps = overlap_mask(times, signal)
    order = np.argsort(score.ravel())[::-1]
    rows = []
    for rank, flat_idx in enumerate(order[:n], start=1):
        fi, ti = np.unravel_index(int(flat_idx), score.shape)
        rows.append(
            {
                "tag": tag,
                "rank": rank,
                "freq_hz": float(freqs[fi]),
                "time_rel_s": float(times[ti]),
                "time_days": float(times[ti] / 86400.0),
                "overlaps_1e-3_peak_signal_window": bool(overlaps[ti]),
                "score": float(score[fi, ti]),
                "ae_twhitened_power": float(power[fi, ti]),
                "signal_start_rel_s": float(signal[4]),
                "signal_stop_rel_s": float(signal[5]),
            }
        )
    return rows


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    products = {tag: load_map(tag) for tag, _ in CASES}

    scores = []
    powers = []
    for data in products.values():
        score = np.maximum(np.asarray(data["gw_only_low_score"], dtype=float), 0.0)
        power = np.asarray(data["gw_only_low_ae_power"], dtype=float)
        scores.append(score[np.isfinite(score) & (score > 0.0)])
        powers.append(power[np.isfinite(power) & (power > 0.0)])
    score_norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=float(np.nanpercentile(np.concatenate(scores), 99.0)))
    power_norm = LogNorm(
        vmin=float(np.nanpercentile(np.concatenate(powers), 2.0)),
        vmax=float(np.nanpercentile(np.concatenate(powers), 99.5)),
    )

    fig, axes = plt.subplots(2, len(CASES), figsize=(21.0, 6.5), sharex=True, sharey=True, constrained_layout=True)
    score_image = None
    power_image = None
    rows: list[dict[str, float | int | str | bool]] = []
    for col, (tag, title) in enumerate(CASES):
        data = products[tag]
        times = np.asarray(data["gw_only_low_times_rel"], dtype=float) / 86400.0
        freqs = np.asarray(data["gw_only_low_freqs_hz"], dtype=float)
        signal = np.asarray(data["signal"], dtype=float)
        score = np.maximum(np.asarray(data["gw_only_low_score"], dtype=float), 0.0)
        power = np.asarray(data["gw_only_low_ae_power"], dtype=float)

        score_image = axes[0, col].pcolormesh(times, freqs, score, shading="auto", cmap="viridis", norm=score_norm)
        power_image = axes[1, col].pcolormesh(times, freqs, power, shading="auto", cmap="magma", norm=power_norm)
        for ax in axes[:, col]:
            ax.axvspan(signal[4] / 86400.0, signal[5] / 86400.0, color="cyan", alpha=0.13, lw=0)
            ax.set_yscale("log")
            ax.set_ylim(5.5e-5, 4.6e-4)
            ax.grid(True, which="both", alpha=0.12)
        axes[0, col].set_title(rf"$M={title}\ M_\odot$", fontsize=10)
        axes[1, col].set_xlabel("Time [days]")
        rows.extend(top_pixel_rows(tag, data))

    axes[0, 0].set_ylabel("score\nFrequency [Hz]")
    axes[1, 0].set_ylabel("AE power\nFrequency [Hz]")
    if score_image is not None:
        cbar = fig.colorbar(score_image, ax=axes[0, :].tolist(), shrink=0.88, pad=0.01)
        cbar.set_label("positive frequency-normalized score")
    if power_image is not None:
        cbar = fig.colorbar(power_image, ax=axes[1, :].tolist(), shrink=0.88, pad=0.01)
        cbar.set_label("GW-only AE T-whitened power")

    plot_path = PLOT_DIR / "gw_only_low_frequency_score_vs_power_diagnostic.png"
    table_path = OUTPUT_DIR / "gw_only_low_frequency_top_score_pixels.csv"
    fig.suptitle("GW-only low-frequency WDM pixels: normalized score versus absolute power", fontsize=13)
    fig.savefig(plot_path, dpi=190)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(table_path, index=False)
    print(plot_path)
    print(table_path)


if __name__ == "__main__":
    main()
