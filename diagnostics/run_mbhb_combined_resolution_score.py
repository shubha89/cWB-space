from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.run_mbhb_aet_chirp_track_morphology import robust_frequency_score


OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
LOW_HZ = 1e-5
HIGH_HZ = 1e-1
SPLIT_HZ = 4.882812427240425e-4


def load_ae_power(maps: np.lib.npyio.NpzFile, *, source: str, resolution: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(maps[f"{source}_A_{resolution}_twhitened_power"], dtype=float)
    e = np.asarray(maps[f"{source}_E_{resolution}_twhitened_power"], dtype=float)
    times = np.asarray(maps[f"{source}_A_{resolution}_times_rel"], dtype=float)
    freqs = np.asarray(maps[f"{source}_A_{resolution}_freqs_hz"], dtype=float)
    return a + e, times, freqs


def split_score_products(
    maps: np.lib.npyio.NpzFile,
    *,
    source: str,
    resolution: str,
    low_hz: float,
    high_hz: float,
    smooth_time_bins: int,
) -> dict[str, np.ndarray]:
    ae_power, times, freqs = load_ae_power(maps, source=source, resolution=resolution)
    band = (freqs >= low_hz) & (freqs < high_hz)
    band_power = ae_power[band, :]
    score = robust_frequency_score(
        band_power,
        z_floor=0.35,
        smooth_size=(3, int(smooth_time_bins)),
    )
    return {
        "ae_power": band_power,
        "score": score,
        "times_rel": times,
        "freqs_hz": freqs[band],
    }


def signal_time_overlap_mask(times: np.ndarray, signal: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        return np.zeros(0, dtype=bool)
    if len(times) > 1:
        half_width = 0.5 * float(np.nanmedian(np.diff(times)))
    else:
        half_width = 0.0
    starts = times - half_width
    stops = times + half_width
    return (stops >= signal[4]) & (starts <= signal[5])


def finite_sum(values: float | int | np.floating) -> float:
    value = float(values)
    return value if np.isfinite(value) else 0.0


def metric_rows(source: str, component: str, products: dict[str, np.ndarray], signal: np.ndarray) -> list[dict[str, float | int | str]]:
    score = np.asarray(products["score"], dtype=float)
    power = np.asarray(products["ae_power"], dtype=float)
    times = np.asarray(products["times_rel"], dtype=float)
    finite = np.isfinite(score)
    signal_t = signal_time_overlap_mask(times, signal)
    masks = {
        "full": finite,
        "positive": finite & (score > 0),
        "signal_window": finite & signal_t[None, :],
        "signal_window_positive": finite & signal_t[None, :] & (score > 0),
        "off_signal_window": finite & ~signal_t[None, :],
        "off_signal_window_positive": finite & ~signal_t[None, :] & (score > 0),
    }
    rows = []
    for region, mask in masks.items():
        s = score[mask]
        p = power[mask]
        rows.append(
            {
                "source": source,
                "component": component,
                "region": region,
                "n_pixels": int(s.size),
                "sum_score": float(np.nansum(s)) if s.size else np.nan,
                "sum_score_squared": float(np.nansum(s**2)) if s.size else np.nan,
                "mean_score": float(np.nanmean(s)) if s.size else np.nan,
                "max_score": float(np.nanmax(s)) if s.size else np.nan,
                "sum_ae_twhitened_power": float(np.nansum(p)) if p.size else np.nan,
            }
        )
    return rows


def combined_metric_rows(source: str, low: dict[str, np.ndarray], high: dict[str, np.ndarray], signal: np.ndarray) -> list[dict[str, float | int | str]]:
    rows = []
    for low_row, high_row in zip(metric_rows(source, "low_M4096", low, signal), metric_rows(source, "high_M512", high, signal)):
        n_pixels = int(low_row["n_pixels"]) + int(high_row["n_pixels"])
        max_values = [float(low_row["max_score"]), float(high_row["max_score"])]
        max_values = [value for value in max_values if np.isfinite(value)]
        sum_score = finite_sum(low_row["sum_score"]) + finite_sum(high_row["sum_score"])
        sum_score_squared = finite_sum(low_row["sum_score_squared"]) + finite_sum(high_row["sum_score_squared"])
        sum_power = finite_sum(low_row["sum_ae_twhitened_power"]) + finite_sum(high_row["sum_ae_twhitened_power"])
        rows.append(
            {
                "source": source,
                "component": "combined_frequency_split",
                "region": low_row["region"],
                "n_pixels": n_pixels,
                "sum_score": sum_score,
                "sum_score_squared": sum_score_squared,
                "mean_score": sum_score / n_pixels if n_pixels > 0 else np.nan,
                "max_score": max(max_values) if max_values else np.nan,
                "sum_ae_twhitened_power": sum_power,
            }
        )
    return rows


def plot_combined(
    output: Path,
    products: dict[str, dict[str, dict[str, np.ndarray]]],
    signal: np.ndarray,
    *,
    title: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = ["noisy", "gw_only"]
    all_scores = []
    for source in sources:
        for part in ("low", "high"):
            score = products[source][part]["score"]
            all_scores.append(score[np.isfinite(score)])
    finite_scores = np.concatenate(all_scores)
    vmin, vmax = np.nanpercentile(finite_scores, [1.0, 99.7])

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.2), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ax, source in zip(axes, sources):
        for part in ("low", "high"):
            part_products = products[source][part]
            times = part_products["times_rel"] / 86400.0
            freqs = part_products["freqs_hz"]
            score = part_products["score"]
            image = ax.pcolormesh(times, freqs, score, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.axhline(SPLIT_HZ, color="white", lw=0.8, alpha=0.75)
        ax.axvspan(signal[4] / 86400.0, signal[5] / 86400.0, color="cyan", alpha=0.12, lw=0)
        ax.set_yscale("log")
        ax.set_ylabel(f"{source}\nFrequency [Hz]")
        ax.grid(True, which="both", alpha=0.14)
    axes[-1].set_xlabel("Time [days]")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.86, pad=0.02)
        cbar.set_label("frequency-normalized robust chirp score")
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine low-M4096 and high-M512 WDM scores.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--m512-maps", type=Path, required=True)
    parser.add_argument("--m4096-maps", type=Path, required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.output_prefix or f"twhitened_wdm_mbhb_{args.tag}_combined_M4096low_M512high"
    title = args.title or rf"{args.tag} $M_\odot$ combined WDM score: M4096 low-f, M512 high-f"
    if not args.m512_maps.exists():
        raise FileNotFoundError(f"Missing M512 map file: {args.m512_maps}")
    if not args.m4096_maps.exists():
        raise FileNotFoundError(f"Missing M4096 map file: {args.m4096_maps}")
    m512 = np.load(args.m512_maps)
    m4096 = np.load(args.m4096_maps)
    signal = np.asarray(m512["signal"], dtype=float)

    products: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    rows = []
    for source in ("noisy", "gw_only"):
        low = split_score_products(
            m4096,
            source=source,
            resolution="dt2_M4096",
            low_hz=LOW_HZ,
            high_hz=SPLIT_HZ,
            smooth_time_bins=1,
        )
        high = split_score_products(
            m512,
            source=source,
            resolution="dt2_M512",
            low_hz=SPLIT_HZ,
            high_hz=HIGH_HZ,
            smooth_time_bins=3,
        )
        products[source] = {"low": low, "high": high}
        rows.extend(metric_rows(source, "low_M4096", low, signal))
        rows.extend(metric_rows(source, "high_M512", high, signal))
        rows.extend(combined_metric_rows(source, low, high, signal))

    metrics = pd.DataFrame(rows)
    metrics_csv = OUTPUT_DIR / f"{prefix}_score_energy.csv"
    plot_png = PLOT_DIR / f"{prefix}_score.png"
    products_npz = OUTPUT_DIR / f"{prefix}_score_maps.npz"
    report_json = OUTPUT_DIR / f"{prefix}_report.json"
    metrics.to_csv(metrics_csv, index=False)
    payload: dict[str, np.ndarray] = {"signal": signal, "split_hz": np.asarray(SPLIT_HZ)}
    for source, source_products in products.items():
        for part, part_products in source_products.items():
            for name, value in part_products.items():
                payload[f"{source}_{part}_{name}"] = value
    np.savez_compressed(products_npz, **payload)
    plot_combined(plot_png, products, signal, title=title)

    report = {
        "method": "frequency-split combined diagnostic using separate orthogonal WDM bases",
        "tag": args.tag,
        "split_hz": SPLIT_HZ,
        "low_component": {"resolution": "dt2_M4096", "frequency_range_hz": [LOW_HZ, SPLIT_HZ]},
        "high_component": {"resolution": "dt2_M512", "frequency_range_hz": [SPLIT_HZ, HIGH_HZ]},
        "note": "The components are plotted together but kept on their native WDM time grids.",
        "inputs": {"m512_maps": str(args.m512_maps), "m4096_maps": str(args.m4096_maps)},
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "products_npz": str(products_npz),
            "plot": str(plot_png),
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(metrics[metrics["component"] == "combined_frequency_split"].to_string(index=False))


if __name__ == "__main__":
    main()
