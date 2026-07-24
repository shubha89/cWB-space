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

from diagnostics.run_mbhb_wdm_coherent_excess_power import local_ae_coherence
from scripts.pycwb_xtalk import PycWBXTalkCatalog


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
DEFAULT_CATALOG = OUTPUT_DIR / "pycwb_mra_xtalk_512_4096_scale_extended.npz"
LOW_HZ = 1e-5
HIGH_HZ = 1e-1


def load_complex_pair(maps: np.lib.npyio.NpzFile, *, source: str, resolution: str):
    prefix_a = f"{source}_A_{resolution}"
    prefix_e = f"{source}_E_{resolution}"
    a = maps[f"{prefix_a}_twhitened_real"].astype(float) + 1j * maps[f"{prefix_a}_twhitened_imag"].astype(float)
    e = maps[f"{prefix_e}_twhitened_real"].astype(float) + 1j * maps[f"{prefix_e}_twhitened_imag"].astype(float)
    times = maps[f"{prefix_a}_times_rel"].astype(float)
    freqs = maps[f"{prefix_a}_freqs_hz"].astype(float)
    return a, e, times, freqs


def parse_layer(resolution: str) -> int:
    for part in resolution.split("_"):
        if part.startswith("M"):
            return int(part[1:])
    raise ValueError(f"Could not parse WDM layer from resolution label {resolution!r}.")


def candidate_pixels(
    maps: np.lib.npyio.NpzFile,
    *,
    source: str,
    resolution: str,
    smoothing: tuple[int, int],
    top_fraction: float,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    a, e, times, freqs = load_complex_pair(maps, source=source, resolution=resolution)
    products = local_ae_coherence(a, e, smoothing=smoothing)
    score = products["coherent_excess_power"]
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    band_freq_indices = np.flatnonzero(band)
    band_score = score[band, :]
    threshold = np.quantile(band_score[np.isfinite(band_score)], 1.0 - top_fraction)
    fi_local, ti = np.nonzero(band_score >= threshold)
    values = band_score[fi_local, ti]
    order = np.argsort(values)[::-1]
    if top_n > 0:
        order = order[:top_n]
    fi = band_freq_indices[fi_local[order]]
    ti = ti[order]
    values = values[order]

    layer = parse_layer(resolution)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else np.nan
    df = float(np.median(np.diff(freqs))) if len(freqs) > 1 else np.nan
    frame = pd.DataFrame(
        {
            "candidate_id": np.arange(len(order), dtype=int),
            "source": source,
            "resolution": resolution,
            "layer": layer,
            "time_index": ti.astype(int),
            "freq_index": fi.astype(int),
            "pixel_index": (ti.astype(int) * (layer + 1) + fi.astype(int)).astype(int),
            "time_rel": times[ti],
            "freq_hz": freqs[fi],
            "dt_s": dt,
            "df_hz": df,
            "score": values,
            "energy": products["energy"][fi, ti],
            "coherence_abs": products["coherence_abs"][fi, ti],
        }
    )
    return frame, {"times": times, "freqs": freqs, **products}


def possibly_near(row: pd.Series, selected: pd.Series, *, time_factor: float, freq_factor: float) -> bool:
    time_scale = time_factor * max(float(row["dt_s"]), float(selected["dt_s"]))
    freq_scale = freq_factor * max(float(row["df_hz"]), float(selected["df_hz"]))
    return (
        abs(float(row["time_rel"]) - float(selected["time_rel"])) <= time_scale
        and abs(float(row["freq_hz"]) - float(selected["freq_hz"])) <= freq_scale
    )


def xtalk_prune(
    candidates: pd.DataFrame,
    catalog: PycWBXTalkCatalog,
    *,
    overlap_threshold: float,
    time_factor: float,
    freq_factor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = candidates.sort_values("score", ascending=False).reset_index(drop=True)
    selected_rows = []
    rejected_rows = []
    for _, row in ordered.iterrows():
        rejected = False
        reject_info = {}
        for selected in selected_rows:
            if not possibly_near(row, selected, time_factor=time_factor, freq_factor=freq_factor):
                continue
            overlap = catalog.overlap_power(
                int(row["layer"]),
                int(row["pixel_index"]),
                int(selected["layer"]),
                int(selected["pixel_index"]),
            )
            if overlap >= overlap_threshold:
                rejected = True
                reject_info = {
                    "selected_blocker_resolution": selected["resolution"],
                    "selected_blocker_time_rel": selected["time_rel"],
                    "selected_blocker_freq_hz": selected["freq_hz"],
                    "selected_blocker_score": selected["score"],
                    "xtalk_overlap_power": overlap,
                }
                break
        record = row.to_dict()
        if rejected:
            record.update({"selected": False, **reject_info})
            rejected_rows.append(record)
        else:
            record.update({"selected": True, "xtalk_overlap_power": 0.0})
            selected_rows.append(pd.Series(record))

    return pd.DataFrame(selected_rows), pd.DataFrame(rejected_rows)


def plot_selected(
    output: Path,
    *,
    products_by_resolution: dict[str, dict[str, np.ndarray]],
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    signal: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    resolutions = list(products_by_resolution)
    fig, axes = plt.subplots(
        1,
        len(resolutions),
        figsize=(6.0 * len(resolutions), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(resolutions) == 1:
        axes = [axes]
    signal_start_days = float(signal[4]) / 86400.0
    signal_stop_days = float(signal[5]) / 86400.0
    for ax, resolution in zip(axes, resolutions):
        prod = products_by_resolution[resolution]
        times_days = prod["times"] / 86400.0
        freqs = prod["freqs"]
        band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
        image = ax.pcolormesh(
            times_days,
            freqs[band],
            np.log10(np.maximum(prod["coherent_excess_power"][band, :], 1e-300)),
            shading="auto",
            cmap="magma",
        )
        raw = candidates[candidates["resolution"] == resolution]
        keep = selected[selected["resolution"] == resolution]
        ax.scatter(raw["time_rel"] / 86400.0, raw["freq_hz"], s=8, facecolors="none", edgecolors="white", alpha=0.35, linewidths=0.4)
        ax.scatter(keep["time_rel"] / 86400.0, keep["freq_hz"], s=18, color="cyan", marker="x", linewidths=0.9)
        ax.axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.13, lw=0)
        ax.set_yscale("log")
        ax.set_title(resolution)
        ax.set_xlabel("Time [days]")
        ax.grid(True, which="both", alpha=0.15)
    axes[0].set_ylabel("Frequency [Hz]")
    cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label(r"$\log_{10}$ coherent excess power")
    fig.suptitle("Multiresolution WDM candidates after pycWB cross-talk pruning")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _nearest_common_axes(products_by_resolution: dict[str, dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    finest_time = min(products_by_resolution, key=lambda key: np.median(np.diff(products_by_resolution[key]["times"])))
    finest_freq = min(products_by_resolution, key=lambda key: np.median(np.diff(products_by_resolution[key]["freqs"])))
    times = products_by_resolution[finest_time]["times"].astype(float)
    freqs = products_by_resolution[finest_freq]["freqs"].astype(float)
    freqs = freqs[(freqs >= LOW_HZ) & (freqs <= HIGH_HZ)]
    return times, freqs


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        step = 1.0
        return np.asarray([values[0] - 0.5 * step, values[0] + 0.5 * step])
    midpoints = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def combine_best_pixels(
    selected: pd.DataFrame,
    *,
    products_by_resolution: dict[str, dict[str, np.ndarray]],
    metric: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Project selected MRA pixels onto one common grid and keep the largest metric."""

    if metric not in {"energy", "score"}:
        raise ValueError("combine metric must be either 'energy' or 'score'.")
    common_times, common_freqs = _nearest_common_axes(products_by_resolution)
    if len(common_times) == 0 or len(common_freqs) == 0:
        raise ValueError("Cannot combine pixels onto an empty common grid.")

    resolution_names = list(products_by_resolution)
    resolution_to_id = {name: idx + 1 for idx, name in enumerate(resolution_names)}
    id_to_resolution = np.asarray(["none", *resolution_names])

    best_metric = np.full((len(common_freqs), len(common_times)), -np.inf, dtype=float)
    best_energy = np.full_like(best_metric, np.nan)
    best_score = np.full_like(best_metric, np.nan)
    best_coherence = np.full_like(best_metric, np.nan)
    best_resolution_id = np.zeros(best_metric.shape, dtype=np.int16)
    best_source_row = np.full(best_metric.shape, -1, dtype=np.int32)

    records = []
    ordered = selected.sort_values(metric, ascending=False).reset_index(drop=True)
    for source_row, row in ordered.iterrows():
        time_min = float(row["time_rel"]) - 0.5 * float(row["dt_s"])
        time_max = float(row["time_rel"]) + 0.5 * float(row["dt_s"])
        freq_min = max(LOW_HZ, float(row["freq_hz"]) - 0.5 * float(row["df_hz"]))
        freq_max = min(HIGH_HZ, float(row["freq_hz"]) + 0.5 * float(row["df_hz"]))
        t0 = int(np.searchsorted(common_times, time_min, side="left"))
        t1 = int(np.searchsorted(common_times, time_max, side="right"))
        f0 = int(np.searchsorted(common_freqs, freq_min, side="left"))
        f1 = int(np.searchsorted(common_freqs, freq_max, side="right"))
        if t0 >= t1 or f0 >= f1:
            continue

        value = float(row[metric])
        window = best_metric[f0:f1, t0:t1]
        update = value > window
        if not np.any(update):
            continue
        window[update] = value
        best_energy[f0:f1, t0:t1][update] = float(row["energy"])
        best_score[f0:f1, t0:t1][update] = float(row["score"])
        best_coherence[f0:f1, t0:t1][update] = float(row["coherence_abs"])
        best_resolution_id[f0:f1, t0:t1][update] = resolution_to_id[str(row["resolution"])]
        best_source_row[f0:f1, t0:t1][update] = source_row
        record = row.to_dict()
        record.update(
            {
                "combine_metric": metric,
                "common_time_start": common_times[t0],
                "common_time_stop": common_times[t1 - 1],
                "common_freq_start_hz": common_freqs[f0],
                "common_freq_stop_hz": common_freqs[f1 - 1],
                "common_cells_filled": int(np.sum(update)),
            }
        )
        records.append(record)

    best_metric[~np.isfinite(best_metric)] = np.nan
    combined_pixels = pd.DataFrame(records)
    combined_maps = {
        "times_rel": common_times,
        "freqs_hz": common_freqs,
        "best_metric": best_metric,
        "best_energy": best_energy,
        "best_score": best_score,
        "best_coherence_abs": best_coherence,
        "best_resolution_id": best_resolution_id,
        "best_source_row": best_source_row,
        "resolution_labels": id_to_resolution,
    }
    return combined_pixels, combined_maps


def combine_all_pixels(
    *,
    products_by_resolution: dict[str, dict[str, np.ndarray]],
    metric: str,
) -> dict[str, np.ndarray]:
    """Project every WDM pixel in-band to one common grid and keep the largest metric."""

    if metric not in {"energy", "score"}:
        raise ValueError("combine metric must be either 'energy' or 'score'.")
    common_times, common_freqs = _nearest_common_axes(products_by_resolution)
    resolution_names = list(products_by_resolution)
    resolution_to_id = {name: idx + 1 for idx, name in enumerate(resolution_names)}
    id_to_resolution = np.asarray(["none", *resolution_names])

    best_metric = np.full((len(common_freqs), len(common_times)), -np.inf, dtype=float)
    best_energy = np.full_like(best_metric, np.nan)
    best_score = np.full_like(best_metric, np.nan)
    best_coherence = np.full_like(best_metric, np.nan)
    best_resolution_id = np.zeros(best_metric.shape, dtype=np.int16)

    for resolution, prod in products_by_resolution.items():
        times = prod["times"].astype(float)
        freqs = prod["freqs"].astype(float)
        native_dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        native_df = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 1.0
        metric_map = prod["energy"] if metric == "energy" else prod["coherent_excess_power"]
        energy_map = prod["energy"]
        score_map = prod["coherent_excess_power"]
        coherence_map = prod["coherence_abs"]

        time_ranges = [
            (
                int(np.searchsorted(common_times, time - 0.5 * native_dt, side="left")),
                int(np.searchsorted(common_times, time + 0.5 * native_dt, side="right")),
            )
            for time in times
        ]
        band_freq_indices = np.flatnonzero((freqs >= LOW_HZ) & (freqs <= HIGH_HZ))
        freq_ranges = {
            int(fi): (
                int(np.searchsorted(common_freqs, max(LOW_HZ, freqs[fi] - 0.5 * native_df), side="left")),
                int(np.searchsorted(common_freqs, min(HIGH_HZ, freqs[fi] + 0.5 * native_df), side="right")),
            )
            for fi in band_freq_indices
        }

        resolution_id = resolution_to_id[resolution]
        for fi in band_freq_indices:
            f0, f1 = freq_ranges[int(fi)]
            if f0 >= f1:
                continue
            for ti, (t0, t1) in enumerate(time_ranges):
                if t0 >= t1:
                    continue
                value = float(metric_map[fi, ti])
                if not np.isfinite(value):
                    continue
                window = best_metric[f0:f1, t0:t1]
                update = value > window
                if not np.any(update):
                    continue
                window[update] = value
                best_energy[f0:f1, t0:t1][update] = float(energy_map[fi, ti])
                best_score[f0:f1, t0:t1][update] = float(score_map[fi, ti])
                best_coherence[f0:f1, t0:t1][update] = float(coherence_map[fi, ti])
                best_resolution_id[f0:f1, t0:t1][update] = resolution_id

    best_metric[~np.isfinite(best_metric)] = np.nan
    return {
        "times_rel": common_times,
        "freqs_hz": common_freqs,
        "best_metric": best_metric,
        "best_energy": best_energy,
        "best_score": best_score,
        "best_coherence_abs": best_coherence,
        "best_resolution_id": best_resolution_id,
        "resolution_labels": id_to_resolution,
    }


def combined_grid_frame(combined_maps: dict[str, np.ndarray]) -> pd.DataFrame:
    filled_f, filled_t = np.nonzero(combined_maps["best_resolution_id"] > 0)
    labels = np.asarray(combined_maps["resolution_labels"])
    resolution_id = combined_maps["best_resolution_id"][filled_f, filled_t].astype(int)
    return pd.DataFrame(
        {
            "time_index": filled_t.astype(int),
            "freq_index": filled_f.astype(int),
            "time_rel": combined_maps["times_rel"][filled_t],
            "freq_hz": combined_maps["freqs_hz"][filled_f],
            "winning_resolution": labels[resolution_id],
            "best_metric": combined_maps["best_metric"][filled_f, filled_t],
            "best_energy": combined_maps["best_energy"][filled_f, filled_t],
            "best_score": combined_maps["best_score"][filled_f, filled_t],
            "best_coherence_abs": combined_maps["best_coherence_abs"][filled_f, filled_t],
        }
    )


def plot_combined_best(
    output: Path,
    *,
    combined_maps: dict[str, np.ndarray],
    combined_pixels: pd.DataFrame,
    signal: np.ndarray,
    metric: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    times = combined_maps["times_rel"].astype(float)
    freqs = combined_maps["freqs_hz"].astype(float)
    values = combined_maps["best_metric"].astype(float)
    resolution_id = combined_maps["best_resolution_id"].astype(int)
    labels = [str(x) for x in combined_maps["resolution_labels"].tolist()]
    time_edges = _centers_to_edges(times) / 86400.0
    freq_edges = _centers_to_edges(freqs)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True, sharey=True, constrained_layout=True)
    shown = np.log10(np.maximum(values, 1e-300))
    shown[~np.isfinite(values)] = np.nan
    image = axes[0].pcolormesh(time_edges, freq_edges, shown, shading="auto", cmap="magma")
    axes[0].set_title(f"best {metric}")
    cbar = fig.colorbar(image, ax=axes[0], pad=0.015)
    cbar.set_label(rf"$\log_{{10}}$ best {metric}")

    cmap = plt.get_cmap("tab10", max(1, len(labels)))
    res_image = axes[1].pcolormesh(
        time_edges,
        freq_edges,
        np.where(resolution_id > 0, resolution_id, np.nan),
        shading="auto",
        cmap=cmap,
        vmin=0.5,
        vmax=max(1.5, len(labels) - 0.5),
    )
    axes[1].set_title("winning resolution")
    res_cbar = fig.colorbar(res_image, ax=axes[1], ticks=np.arange(1, len(labels)), pad=0.015)
    res_cbar.ax.set_yticklabels(labels[1:])

    signal_start_days = float(signal[4]) / 86400.0
    signal_stop_days = float(signal[5]) / 86400.0
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Time [days]")
        ax.axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.13, lw=0)
        ax.grid(True, which="both", alpha=0.15)
    axes[0].set_ylabel("Frequency [Hz]")

    if not combined_pixels.empty:
        axes[0].scatter(
            combined_pixels["time_rel"] / 86400.0,
            combined_pixels["freq_hz"],
            s=10,
            facecolors="none",
            edgecolors="white",
            linewidths=0.45,
            alpha=0.45,
        )
    fig.suptitle("Combined multiresolution WDM best-pixel map")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Multiresolution WDM candidate selection with pycWB cross-talk coefficients.")
    parser.add_argument("--input-prefix", default="twhitened_wdm_pycwb_mra_full_aet")
    parser.add_argument("--source", default="noisy", choices=("noisy", "gw_only"))
    parser.add_argument("--resolutions", nargs="+", default=["dt2_M512", "dt2_M1024", "dt2_M2048", "dt2_M4096"])
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--smooth-freq", type=int, default=3)
    parser.add_argument("--smooth-time", type=int, default=3)
    parser.add_argument("--top-fraction", type=float, default=0.03)
    parser.add_argument("--top-n-per-resolution", type=int, default=500)
    parser.add_argument("--overlap-threshold", type=float, default=0.2)
    parser.add_argument("--time-factor", type=float, default=2.5)
    parser.add_argument("--freq-factor", type=float, default=2.5)
    parser.add_argument("--combine-metric", choices=("energy", "score"), default="energy")
    parser.add_argument("--combine-all-pixels", action="store_true")
    args = parser.parse_args()

    input_npz = OUTPUT_DIR / f"{args.input_prefix}_maps.npz"
    if not input_npz.exists():
        raise FileNotFoundError(f"Missing T-whitened WDM map file: {input_npz}")
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing pycWB cross-talk catalog: {catalog_path}")

    catalog = PycWBXTalkCatalog.load(catalog_path)
    maps = np.load(input_npz)
    signal = np.asarray(maps["signal"], dtype=float)
    smoothing = (int(args.smooth_freq), int(args.smooth_time))

    frames = []
    products_by_resolution = {}
    for resolution in args.resolutions:
        layer = parse_layer(resolution)
        if layer not in set(catalog.layers.tolist()):
            raise ValueError(f"Resolution {resolution} has layer {layer}, not present in catalog {catalog.layers.tolist()}.")
        frame, products = candidate_pixels(
            maps,
            source=args.source,
            resolution=resolution,
            smoothing=smoothing,
            top_fraction=float(args.top_fraction),
            top_n=int(args.top_n_per_resolution),
        )
        frames.append(frame)
        products_by_resolution[resolution] = products

    candidates = pd.concat(frames, ignore_index=True)
    selected, rejected = xtalk_prune(
        candidates,
        catalog,
        overlap_threshold=float(args.overlap_threshold),
        time_factor=float(args.time_factor),
        freq_factor=float(args.freq_factor),
    )

    tag = f"{args.input_prefix}_{args.source}_{'-'.join(args.resolutions)}_xtalk"
    candidates_csv = OUTPUT_DIR / f"multires_wdm_{tag}_candidates.csv"
    selected_csv = OUTPUT_DIR / f"multires_wdm_{tag}_selected.csv"
    rejected_csv = OUTPUT_DIR / f"multires_wdm_{tag}_rejected.csv"
    combined_csv = OUTPUT_DIR / f"multires_wdm_{tag}_combined_best_{args.combine_metric}.csv"
    combined_npz = OUTPUT_DIR / f"multires_wdm_{tag}_combined_best_{args.combine_metric}.npz"
    all_pixels_csv = OUTPUT_DIR / f"multires_wdm_{tag}_combined_all_pixels_best_{args.combine_metric}.csv"
    all_pixels_npz = OUTPUT_DIR / f"multires_wdm_{tag}_combined_all_pixels_best_{args.combine_metric}.npz"
    report_json = OUTPUT_DIR / f"multires_wdm_{tag}_report.json"
    plot_png = PLOT_DIR / f"12_multires_wdm_{tag}.png"
    combined_plot_png = PLOT_DIR / f"13_multires_wdm_{tag}_combined_best_{args.combine_metric}.png"
    all_pixels_plot_png = PLOT_DIR / f"14_multires_wdm_{tag}_combined_all_pixels_best_{args.combine_metric}.png"
    candidates.to_csv(candidates_csv, index=False)
    selected.to_csv(selected_csv, index=False)
    rejected.to_csv(rejected_csv, index=False)
    plot_selected(plot_png, products_by_resolution=products_by_resolution, candidates=candidates, selected=selected, signal=signal)
    combined_pixels, combined_maps = combine_best_pixels(
        selected,
        products_by_resolution=products_by_resolution,
        metric=str(args.combine_metric),
    )
    combined_pixels.to_csv(combined_csv, index=False)
    np.savez_compressed(combined_npz, **combined_maps)
    plot_combined_best(
        combined_plot_png,
        combined_maps=combined_maps,
        combined_pixels=combined_pixels,
        signal=signal,
        metric=str(args.combine_metric),
    )
    all_pixels_maps = None
    all_pixels_frame = None
    if args.combine_all_pixels:
        all_pixels_maps = combine_all_pixels(
            products_by_resolution=products_by_resolution,
            metric=str(args.combine_metric),
        )
        all_pixels_frame = combined_grid_frame(all_pixels_maps)
        all_pixels_frame.to_csv(all_pixels_csv, index=False)
        np.savez_compressed(all_pixels_npz, **all_pixels_maps)
        plot_combined_best(
            all_pixels_plot_png,
            combined_maps=all_pixels_maps,
            combined_pixels=pd.DataFrame(),
            signal=signal,
            metric=str(args.combine_metric),
        )

    report = {
        "method": "multiresolution WDM candidate selection with pycWB catalog cross-talk overlap pruning",
        "input_npz": str(input_npz),
        "source": args.source,
        "resolutions": args.resolutions,
        "catalog": {
            "path": str(catalog_path),
            "layers": catalog.layers.astype(int).tolist(),
            "n_res": catalog.n_res,
        },
        "selection": {
            "top_fraction": float(args.top_fraction),
            "top_n_per_resolution": int(args.top_n_per_resolution),
            "overlap_threshold": float(args.overlap_threshold),
            "time_factor": float(args.time_factor),
            "freq_factor": float(args.freq_factor),
            "n_candidates": int(len(candidates)),
            "n_selected": int(len(selected)),
            "n_rejected": int(len(rejected)),
            "combine_metric": str(args.combine_metric),
            "n_combined_source_pixels": int(len(combined_pixels)),
            "n_combined_filled_cells": int(np.sum(combined_maps["best_resolution_id"] > 0)),
            "combine_all_pixels": bool(args.combine_all_pixels),
            "n_all_pixel_combined_filled_cells": (
                int(np.sum(all_pixels_maps["best_resolution_id"] > 0)) if all_pixels_maps is not None else 0
            ),
        },
        "outputs": {
            "candidates_csv": str(candidates_csv),
            "selected_csv": str(selected_csv),
            "rejected_csv": str(rejected_csv),
            "combined_csv": str(combined_csv),
            "combined_npz": str(combined_npz),
            "plot": str(plot_png),
            "combined_plot": str(combined_plot_png),
            "all_pixels_combined_csv": str(all_pixels_csv) if args.combine_all_pixels else None,
            "all_pixels_combined_npz": str(all_pixels_npz) if args.combine_all_pixels else None,
            "all_pixels_combined_plot": str(all_pixels_plot_png) if args.combine_all_pixels else None,
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report = {report_json}")


if __name__ == "__main__":
    main()
