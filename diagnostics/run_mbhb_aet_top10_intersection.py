from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"


def select_intersection(
    masks: np.lib.npyio.NpzFile,
    *,
    source: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    a_prefix = f"{source}_A"
    e_prefix = f"{source}_E"
    times = np.asarray(masks[f"{a_prefix}_times_rel"], dtype=float)
    freqs = np.asarray(masks[f"{a_prefix}_freqs_hz"], dtype=float)
    band = np.asarray(masks[f"{a_prefix}_band"], dtype=bool)
    band_freq_indices = np.flatnonzero(band)

    e_times = np.asarray(masks[f"{e_prefix}_times_rel"], dtype=float)
    e_freqs = np.asarray(masks[f"{e_prefix}_freqs_hz"], dtype=float)
    e_band = np.asarray(masks[f"{e_prefix}_band"], dtype=bool)
    if not np.array_equal(times, e_times) or not np.array_equal(freqs, e_freqs) or not np.array_equal(band, e_band):
        raise ValueError(f"A and E grids are not aligned for {source}.")

    a_power_full = np.asarray(masks[f"{a_prefix}_twhitened_power"], dtype=float)
    e_power_full = np.asarray(masks[f"{e_prefix}_twhitened_power"], dtype=float)
    a_top = np.asarray(masks[f"{a_prefix}_top_mask_band"], dtype=bool)
    e_top = np.asarray(masks[f"{e_prefix}_top_mask_band"], dtype=bool)
    common = a_top & e_top
    fi_local, ti = np.nonzero(common)
    fi = band_freq_indices[fi_local]

    a_power = a_power_full[fi, ti]
    e_power = e_power_full[fi, ti]
    joint_min = np.minimum(a_power, e_power)
    joint_geom = np.sqrt(np.maximum(a_power, 0.0) * np.maximum(e_power, 0.0))
    order = np.argsort(joint_min)[::-1]
    fi = fi[order]
    ti = ti[order]
    fi_local = fi_local[order]
    a_power = a_power[order]
    e_power = e_power[order]
    joint_min = joint_min[order]
    joint_geom = joint_geom[order]

    dt = float(np.median(np.diff(times))) if len(times) > 1 else np.nan
    df = float(np.median(np.diff(freqs))) if len(freqs) > 1 else np.nan
    frame = pd.DataFrame(
        {
            "source": source,
            "time_index": ti.astype(int),
            "freq_index": fi.astype(int),
            "band_freq_index": fi_local.astype(int),
            "time_rel": times[ti],
            "freq_hz": freqs[fi],
            "dt_s": dt,
            "df_hz": df,
            "a_twhitened_power": a_power,
            "e_twhitened_power": e_power,
            "joint_min_twhitened_power": joint_min,
            "joint_geometric_mean_twhitened_power": joint_geom,
        }
    )
    products = {
        "times_rel": times,
        "freqs_hz": freqs,
        "band": band,
        "a_twhitened_power": a_power_full,
        "e_twhitened_power": e_power_full,
        "a_top_mask_band": a_top,
        "e_top_mask_band": e_top,
        "common_top_mask_band": common,
    }
    return frame, products


def plot_intersection(
    output: Path,
    *,
    products_by_source: dict[str, dict[str, np.ndarray]],
    signal: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = list(products_by_source)
    fig, axes = plt.subplots(
        len(sources),
        3,
        figsize=(15.0, max(4.0, 3.5 * len(sources))),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(sources) == 1:
        axes = np.asarray([axes])

    signal_start_days = float(signal[4]) / 86400.0
    signal_stop_days = float(signal[5]) / 86400.0
    last_images = []
    for row_axes, source in zip(axes, sources):
        products = products_by_source[source]
        times_days = products["times_rel"] / 86400.0
        freqs = products["freqs_hz"]
        band = products["band"]
        panels = [
            ("A top 10%", np.where(products["a_top_mask_band"], products["a_twhitened_power"][band, :], np.nan)),
            ("E top 10%", np.where(products["e_top_mask_band"], products["e_twhitened_power"][band, :], np.nan)),
            ("A and E common", np.where(products["common_top_mask_band"], np.minimum(products["a_twhitened_power"][band, :], products["e_twhitened_power"][band, :]), np.nan)),
        ]
        for ax, (title, values) in zip(row_axes, panels):
            shown = np.log10(np.maximum(values, 1e-300))
            image = ax.pcolormesh(times_days, freqs[band], shown, shading="auto", cmap="magma")
            last_images.append(image)
            ax.set_title(title)
            ax.set_yscale("log")
            ax.axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.13, lw=0)
            ax.grid(True, which="both", alpha=0.14)
            ax.set_xlabel("Time [days]")
        row_axes[0].set_ylabel(f"{source}\nFrequency [Hz]")

    if last_images:
        cbar = fig.colorbar(last_images[-1], ax=axes.ravel().tolist(), shrink=0.78, pad=0.012)
        cbar.set_label(r"$\log_{10}$ selected T-whitened power")
    fig.suptitle("Pixels above the top-10% threshold in both A and E")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Select pixels present in both A and E top-threshold T-whitened maps.")
    parser.add_argument("--input-prefix", default="aet_tf_maps_dt2_M512_dt2_M512_top10_twhitened_A_E_separate")
    parser.add_argument("--sources", nargs="+", choices=("noisy", "gw_only"), default=["noisy", "gw_only"])
    args = parser.parse_args()

    input_npz = OUTPUT_DIR / f"{args.input_prefix}_masks.npz"
    if not input_npz.exists():
        raise FileNotFoundError(f"Missing per-channel top-pixel mask file: {input_npz}")

    masks = np.load(input_npz)
    signal = np.asarray(masks["signal"], dtype=float)
    frames = []
    products_by_source = {}
    for source in args.sources:
        frame, products = select_intersection(masks, source=source)
        frames.append(frame)
        products_by_source[source] = products

    selected = pd.concat(frames, ignore_index=True)
    tag = f"{args.input_prefix}_A_E_intersection"
    selected_csv = OUTPUT_DIR / f"{tag}_pixels.csv"
    intersection_npz = OUTPUT_DIR / f"{tag}_masks.npz"
    plot_png = PLOT_DIR / f"{tag}.png"
    report_json = OUTPUT_DIR / f"{tag}_report.json"

    selected.to_csv(selected_csv, index=False)
    payload: dict[str, np.ndarray] = {"signal": signal}
    for source, products in products_by_source.items():
        for name, value in products.items():
            payload[f"{source}_{name}"] = value
    np.savez_compressed(intersection_npz, **payload)
    plot_intersection(plot_png, products_by_source=products_by_source, signal=signal)

    report = {
        "method": "intersection of separately selected A and E top-10% T-whitened pixels",
        "input_npz": str(input_npz),
        "sources": list(args.sources),
        "selection": {
            source: {
                "n_common_pixels": int(np.sum(products_by_source[source]["common_top_mask_band"])),
                "n_A_top_pixels": int(np.sum(products_by_source[source]["a_top_mask_band"])),
                "n_E_top_pixels": int(np.sum(products_by_source[source]["e_top_mask_band"])),
            }
            for source in args.sources
        },
        "outputs": {
            "selected_csv": str(selected_csv),
            "intersection_npz": str(intersection_npz),
            "plot": str(plot_png),
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report = {report_json}")


if __name__ == "__main__":
    main()
