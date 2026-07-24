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
LOW_HZ = 1e-5
HIGH_HZ = 1e-1


def top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    if not (0.0 < fraction <= 1.0):
        raise ValueError("top fraction must be in (0, 1].")
    finite = np.isfinite(values)
    threshold = np.quantile(values[finite], 1.0 - fraction)
    return finite & (values >= threshold)


def select_top_pixels(
    maps: np.lib.npyio.NpzFile,
    *,
    source: str,
    channel: str,
    resolution: str,
    fraction: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    prefix = f"{source}_{channel}_{resolution}"
    power = np.asarray(maps[f"{prefix}_twhitened_power"], dtype=float)
    times = np.asarray(maps[f"{prefix}_times_rel"], dtype=float)
    freqs = np.asarray(maps[f"{prefix}_freqs_hz"], dtype=float)
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    band_freq_indices = np.flatnonzero(band)
    band_power = power[band, :]
    mask_band = top_fraction_mask(band_power, fraction)
    fi_local, ti = np.nonzero(mask_band)
    fi = band_freq_indices[fi_local]
    order = np.argsort(band_power[fi_local, ti])[::-1]
    fi = fi[order]
    ti = ti[order]

    dt = float(np.median(np.diff(times))) if len(times) > 1 else np.nan
    df = float(np.median(np.diff(freqs))) if len(freqs) > 1 else np.nan
    frame = pd.DataFrame(
        {
            "source": source,
            "channel": channel,
            "resolution": resolution,
            "time_index": ti.astype(int),
            "freq_index": fi.astype(int),
            "time_rel": times[ti],
            "freq_hz": freqs[fi],
            "dt_s": dt,
            "df_hz": df,
            "twhitened_power": power[fi, ti],
        }
    )
    products = {
        "times_rel": times,
        "freqs_hz": freqs,
        "twhitened_power": power,
        "top_mask_band": mask_band,
        "band": band,
    }
    return frame, products


def plot_top_pixels(
    output: Path,
    *,
    products_by_key: dict[str, dict[str, np.ndarray]],
    selected_by_key: dict[str, pd.DataFrame],
    signal: np.ndarray,
    fraction: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(products_by_key)
    fig, axes = plt.subplots(
        len(keys),
        2,
        figsize=(12.8, max(4.4, 3.0 * len(keys))),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(keys) == 1:
        axes = np.asarray([axes])

    signal_start_days = float(signal[4]) / 86400.0
    signal_stop_days = float(signal[5]) / 86400.0
    for row_axes, key in zip(axes, keys):
        products = products_by_key[key]
        times_days = products["times_rel"] / 86400.0
        freqs = products["freqs_hz"]
        band = products["band"]
        full_values = np.log10(np.maximum(products["twhitened_power"][band, :], 1e-300))
        masked_power = np.where(products["top_mask_band"], products["twhitened_power"][band, :], np.nan)
        masked_values = np.log10(np.maximum(masked_power, 1e-300))

        image_full = row_axes[0].pcolormesh(times_days, freqs[band], full_values, shading="auto", cmap="magma")
        image_top = row_axes[1].pcolormesh(times_days, freqs[band], masked_values, shading="auto", cmap="magma")
        for ax in row_axes:
            ax.set_yscale("log")
            ax.axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.13, lw=0)
            ax.grid(True, which="both", alpha=0.14)
            ax.set_ylabel(f"{key}\nFrequency [Hz]")
        row_axes[0].set_title(r"all channel power")
        row_axes[1].set_title(f"top {100.0 * fraction:.0f}% pixels")
        selected = selected_by_key[key]
        row_axes[1].scatter(
            selected["time_rel"] / 86400.0,
            selected["freq_hz"],
            s=4,
            facecolors="none",
            edgecolors="white",
            linewidths=0.25,
            alpha=0.35,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [days]")
    cbar_full = fig.colorbar(image_full, ax=axes[:, 0].tolist(), shrink=0.78, pad=0.014)
    cbar_full.set_label(r"$\log_{10}|W_{A/E,T}|^2$")
    cbar_top = fig.colorbar(image_top, ax=axes[:, 1].tolist(), shrink=0.78, pad=0.014)
    cbar_top.set_label(r"$\log_{10}$ top-pixel power")
    fig.suptitle("Top T-whitened A and E pixels selected separately")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Select top T-whitened A/E WDM pixels.")
    parser.add_argument("--input-prefix", default="aet_tf_maps_dt2_M512")
    parser.add_argument("--resolution", default="dt2_M512")
    parser.add_argument("--sources", nargs="+", choices=("noisy", "gw_only"), default=["noisy", "gw_only"])
    parser.add_argument("--channels", nargs="+", choices=("A", "E"), default=["A", "E"])
    parser.add_argument("--top-fraction", type=float, default=0.10)
    args = parser.parse_args()

    input_npz = OUTPUT_DIR / f"{args.input_prefix}_maps.npz"
    if not input_npz.exists():
        raise FileNotFoundError(f"Missing A/E TF map file: {input_npz}")
    maps = np.load(input_npz)
    signal = np.asarray(maps["signal"], dtype=float)

    selected_frames = []
    products_by_key = {}
    selected_by_key = {}
    for source in args.sources:
        for channel in args.channels:
            frame, products = select_top_pixels(
                maps,
                source=source,
                channel=channel,
                resolution=args.resolution,
                fraction=float(args.top_fraction),
            )
            key = f"{source}_{channel}"
            selected_frames.append(frame)
            selected_by_key[key] = frame
            products_by_key[key] = products

    selected = pd.concat(selected_frames, ignore_index=True)
    tag = f"{args.input_prefix}_{args.resolution}_top{int(round(100 * args.top_fraction)):02d}_twhitened_A_E_separate"
    selected_csv = OUTPUT_DIR / f"{tag}_pixels.csv"
    masks_npz = OUTPUT_DIR / f"{tag}_masks.npz"
    plot_png = PLOT_DIR / f"{tag}.png"
    report_json = OUTPUT_DIR / f"{tag}_report.json"

    selected.to_csv(selected_csv, index=False)
    payload: dict[str, np.ndarray] = {
        "signal": signal,
    }
    for source_channel, products in products_by_key.items():
        for product_name, value in products.items():
            payload[f"{source_channel}_{product_name}"] = value
    np.savez_compressed(masks_npz, **payload)
    plot_top_pixels(
        plot_png,
        products_by_key=products_by_key,
        selected_by_key=selected_by_key,
        signal=signal,
        fraction=float(args.top_fraction),
    )

    report = {
        "method": "top pixels selected separately in A and E using per-channel T-whitened power; no A/E quadrature sum or cross-correlation",
        "input_npz": str(input_npz),
        "resolution": args.resolution,
        "top_fraction": float(args.top_fraction),
        "sources": list(args.sources),
        "channels": list(args.channels),
        "selection": {
            source_channel: {
                "n_selected": int(len(frame)),
                "max_twhitened_power": float(frame["twhitened_power"].max()),
                "min_selected_twhitened_power": float(frame["twhitened_power"].min()),
            }
            for source_channel, frame in selected_by_key.items()
        },
        "outputs": {
            "selected_csv": str(selected_csv),
            "masks_npz": str(masks_npz),
            "plot": str(plot_png),
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report = {report_json}")


if __name__ == "__main__":
    main()
