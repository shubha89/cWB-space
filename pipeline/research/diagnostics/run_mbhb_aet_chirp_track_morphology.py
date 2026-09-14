from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
LOW_HZ = 1e-5
HIGH_HZ = 1e-1


def load_complex_channel(
    maps: np.lib.npyio.NpzFile,
    *,
    source: str,
    channel: str,
    resolution: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"{source}_{channel}_{resolution}"
    real = np.asarray(maps[f"{prefix}_twhitened_real"], dtype=float)
    imag = np.asarray(maps[f"{prefix}_twhitened_imag"], dtype=float)
    times = np.asarray(maps[f"{prefix}_times_rel"], dtype=float)
    freqs = np.asarray(maps[f"{prefix}_freqs_hz"], dtype=float)
    return real + 1j * imag, times, freqs


def robust_frequency_score(power: np.ndarray, *, z_floor: float, smooth_size: tuple[int, int]) -> np.ndarray:
    log_power = np.log10(np.maximum(power, 1e-300))
    median = np.nanmedian(log_power, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(log_power - median), axis=1, keepdims=True)
    sigma = np.maximum(1.4826 * mad, 1e-6)
    z_score = (log_power - median) / sigma
    score = np.maximum(z_score - float(z_floor), 0.0)
    if smooth_size[0] > 1 or smooth_size[1] > 1:
        score = uniform_filter(score, size=smooth_size, mode="nearest")
    return score


def monotonic_chirp_path(
    score: np.ndarray,
    *,
    max_jump_bins: int,
    step_penalty: float,
    jump_penalty: float,
    end_cell: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    n_freq, n_time = score.shape
    if n_freq == 0 or n_time == 0:
        return np.empty((0, 2), dtype=int), np.nan, np.empty_like(score)

    dp = np.full((n_freq, n_time), -np.inf, dtype=float)
    back = np.full((n_freq, n_time), -1, dtype=np.int32)
    effective = np.asarray(score, dtype=float) - float(step_penalty)
    dp[:, 0] = effective[:, 0]

    jumps = int(max_jump_bins)
    for ti in range(1, n_time):
        previous = dp[:, ti - 1]
        for fi in range(n_freq):
            lo = max(0, fi - jumps)
            prev_indices = np.arange(lo, fi + 1)
            transition = previous[lo : fi + 1] - float(jump_penalty) * (fi - prev_indices) ** 2
            best_offset = int(np.nanargmax(transition))
            best_value = float(transition[best_offset])
            if best_value > 0.0:
                dp[fi, ti] = effective[fi, ti] + best_value
                back[fi, ti] = int(prev_indices[best_offset])
            else:
                dp[fi, ti] = effective[fi, ti]

    if end_cell is None:
        end_fi, end_ti = np.unravel_index(int(np.nanargmax(dp)), dp.shape)
    else:
        end_fi, end_ti = int(end_cell[0]), int(end_cell[1])
    best_score = float(dp[end_fi, end_ti])
    if not np.isfinite(best_score) or best_score <= 0.0:
        return np.empty((0, 2), dtype=int), best_score, dp

    path: list[tuple[int, int]] = []
    fi = int(end_fi)
    ti = int(end_ti)
    while ti >= 0 and fi >= 0 and np.isfinite(dp[fi, ti]) and dp[fi, ti] > 0.0:
        path.append((fi, ti))
        prev_fi = int(back[fi, ti])
        if prev_fi < 0:
            break
        fi = prev_fi
        ti -= 1
    path.reverse()
    return np.asarray(path, dtype=int), best_score, dp


def tube_from_path(path: np.ndarray, shape: tuple[int, int], *, width_bins: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(path) == 0:
        return mask
    n_freq, _ = shape
    width = int(width_bins)
    for fi, ti in path:
        lo = max(0, int(fi) - width)
        hi = min(n_freq, int(fi) + width + 1)
        mask[lo:hi, int(ti)] = True
    return mask


def tube_statistics(
    *,
    source: str,
    resolution: str,
    a_band: np.ndarray,
    e_band: np.ndarray,
    ae_power: np.ndarray,
    score: np.ndarray,
    tube_mask: np.ndarray,
    path: np.ndarray,
    path_score: float,
    times: np.ndarray,
    freqs_band: np.ndarray,
    signal: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_rows = []
    for order, (fi, ti) in enumerate(path):
        path_rows.append(
            {
                "source": source,
                "resolution": resolution,
                "path_order": int(order),
                "time_index": int(ti),
                "freq_index_band": int(fi),
                "time_rel": float(times[ti]),
                "time_days": float(times[ti] / 86400.0),
                "freq_hz": float(freqs_band[fi]),
                "ae_twhitened_power": float(ae_power[fi, ti]),
                "chirp_score": float(score[fi, ti]),
            }
        )

    if not np.any(tube_mask):
        summary = {
            "source": source,
            "resolution": resolution,
            "path_score": float(path_score),
            "n_path_pixels": int(len(path)),
            "n_tube_pixels": 0,
            "cluster_energy": np.nan,
            "cluster_ae_coherence": np.nan,
            "coherent_cluster_energy": np.nan,
            "overlaps_signal_window": False,
        }
        return pd.DataFrame([summary]), pd.DataFrame(path_rows)

    fi, ti = np.nonzero(tube_mask)
    aw = a_band[tube_mask]
    ew = e_band[tube_mask]
    a_sum = float(np.sum(np.abs(aw) ** 2))
    e_sum = float(np.sum(np.abs(ew) ** 2))
    cross = np.sum(np.conjugate(aw) * ew)
    denom = np.sqrt(max(a_sum * e_sum, 1e-300))
    coherence = float(np.clip(np.abs(cross) / denom, 0.0, 1.0))
    cluster_energy = float(a_sum + e_sum)
    coherent_energy = float(cluster_energy * coherence)
    peak_local = int(np.argmax(ae_power[tube_mask]))
    peak_fi = int(fi[peak_local])
    peak_ti = int(ti[peak_local])
    time_start = float(times[np.min(ti)])
    time_stop = float(times[np.max(ti)])
    signal_start_rel = float(signal[4])
    signal_stop_rel = float(signal[5])

    summary = {
        "source": source,
        "resolution": resolution,
        "path_score": float(path_score),
        "n_path_pixels": int(len(path)),
        "n_tube_pixels": int(np.sum(tube_mask)),
        "time_start_rel": time_start,
        "time_stop_rel": time_stop,
        "freq_min_hz": float(freqs_band[np.min(fi)]),
        "freq_max_hz": float(freqs_band[np.max(fi)]),
        "peak_time_rel": float(times[peak_ti]),
        "peak_freq_hz": float(freqs_band[peak_fi]),
        "peak_ae_twhitened_power": float(ae_power[peak_fi, peak_ti]),
        "a_energy": a_sum,
        "e_energy": e_sum,
        "cluster_energy": cluster_energy,
        "cross_energy_real": float(np.real(cross)),
        "cross_energy_imag": float(np.imag(cross)),
        "cross_energy_abs": float(np.abs(cross)),
        "cross_energy_phase_rad": float(np.angle(cross)),
        "cluster_ae_coherence": coherence,
        "coherent_cluster_energy": coherent_energy,
        "overlaps_signal_window": bool(time_stop >= signal_start_rel and time_start <= signal_stop_rel),
    }
    return pd.DataFrame([summary]), pd.DataFrame(path_rows)


def run_source(
    maps: np.lib.npyio.NpzFile,
    *,
    source: str,
    resolution: str,
    search_low_hz: float,
    search_high_hz: float,
    z_floor: float,
    smooth_freq_bins: int,
    smooth_time_bins: int,
    max_jump_bins: int,
    step_penalty: float,
    jump_penalty: float,
    tube_width_bins: int,
    anchor_end_global_peak: bool,
    end_at_path_peak: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    a, times, freqs = load_complex_channel(maps, source=source, channel="A", resolution=resolution)
    e, e_times, e_freqs = load_complex_channel(maps, source=source, channel="E", resolution=resolution)
    if a.shape != e.shape or not np.array_equal(times, e_times) or not np.array_equal(freqs, e_freqs):
        raise ValueError(f"A and E maps are not aligned for {source} {resolution}.")

    display_band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    a_display = a[display_band, :]
    e_display = e[display_band, :]
    freqs_display = freqs[display_band]
    search_band = (freqs_display >= float(search_low_hz)) & (freqs_display <= float(search_high_hz))
    if not np.any(search_band):
        raise ValueError("The requested chirp search band contains no WDM frequency bins.")

    ae_power = np.abs(a_display) ** 2 + np.abs(e_display) ** 2
    ae_search = ae_power[search_band, :]
    score_search = robust_frequency_score(
        ae_search,
        z_floor=z_floor,
        smooth_size=(int(smooth_freq_bins), int(smooth_time_bins)),
    )
    end_cell = None
    if anchor_end_global_peak:
        peak_cell = np.unravel_index(int(np.nanargmax(ae_search)), ae_search.shape)
        end_cell = (int(peak_cell[0]), int(peak_cell[1]))
    path_search, path_score, dp_search = monotonic_chirp_path(
        score_search,
        max_jump_bins=int(max_jump_bins),
        step_penalty=float(step_penalty),
        jump_penalty=float(jump_penalty),
        end_cell=end_cell,
    )
    display_search_indices = np.flatnonzero(search_band)
    if len(path_search):
        path = np.column_stack([display_search_indices[path_search[:, 0]], path_search[:, 1]]).astype(int)
    else:
        path = np.empty((0, 2), dtype=int)
    if end_at_path_peak and not anchor_end_global_peak and len(path):
        path_power = ae_power[path[:, 0], path[:, 1]]
        peak_order = int(np.nanargmax(path_power))
        path = path[: peak_order + 1]
    score = np.full(ae_power.shape, np.nan, dtype=float)
    dp = np.full(ae_power.shape, np.nan, dtype=float)
    score[search_band, :] = score_search
    dp[search_band, :] = dp_search
    tube_mask = tube_from_path(path, score.shape, width_bins=int(tube_width_bins))
    summary, path_table = tube_statistics(
        source=source,
        resolution=resolution,
        a_band=a_display,
        e_band=e_display,
        ae_power=ae_power,
        score=score,
        tube_mask=tube_mask,
        path=path,
        path_score=path_score,
        times=times,
        freqs_band=freqs_display,
        signal=np.asarray(maps["signal"], dtype=float),
    )
    products = {
        "times_rel": times,
        "freqs_hz": freqs_display,
        "band": np.ones_like(freqs_display, dtype=bool),
        "search_band": search_band,
        "ae_power_band": ae_power,
        "chirp_score_band": score,
        "dp_score_band": dp,
        "tube_mask_band": tube_mask,
        "path_freq_indices_band": path[:, 0] if len(path) else np.array([], dtype=int),
        "path_time_indices": path[:, 1] if len(path) else np.array([], dtype=int),
    }
    return summary, path_table, products


def plot_products(
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
        figsize=(15.8, max(4.2, 3.8 * len(sources))),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(sources) == 1:
        axes = np.asarray([axes])

    signal_start_days = float(signal[4]) / 86400.0
    signal_stop_days = float(signal[5]) / 86400.0
    energy_values = []
    score_values = []
    tube_values = []
    for products in products_by_source.values():
        energy_values.append(np.log10(np.maximum(products["ae_power_band"], 1e-300)))
        score_values.append(products["chirp_score_band"])
        tube_values.append(np.log10(np.maximum(np.where(products["tube_mask_band"], products["ae_power_band"], np.nan), 1e-300)))
    energy_vmin, energy_vmax = np.nanpercentile(np.concatenate([v.ravel() for v in energy_values]), [1.0, 99.7])
    score_vmin, score_vmax = np.nanpercentile(np.concatenate([v.ravel() for v in score_values]), [1.0, 99.7])
    tube_finite = np.concatenate([v[np.isfinite(v)] for v in tube_values if np.any(np.isfinite(v))])
    tube_vmin, tube_vmax = np.nanpercentile(tube_finite, [1.0, 99.7]) if tube_finite.size else (None, None)

    energy_image = None
    score_image = None
    tube_image = None
    for row_axes, source in zip(axes, sources):
        products = products_by_source[source]
        times_days = products["times_rel"] / 86400.0
        freqs_band = products["freqs_hz"][products["band"]]
        path_ti = products["path_time_indices"]
        path_fi = products["path_freq_indices_band"]
        path_times = times_days[path_ti] if len(path_ti) else np.array([])
        path_freqs = freqs_band[path_fi] if len(path_fi) else np.array([])

        energy = np.log10(np.maximum(products["ae_power_band"], 1e-300))
        score = products["chirp_score_band"]
        tube = np.log10(np.maximum(np.where(products["tube_mask_band"], products["ae_power_band"], np.nan), 1e-300))
        panels = [
            ("A+E T-whitened energy", energy, "magma", energy_vmin, energy_vmax),
            ("frequency-normalized chirp score", score, "viridis", score_vmin, score_vmax),
            ("morphology tube", tube, "magma", tube_vmin, tube_vmax),
        ]
        for ax, (title, values, cmap, vmin, vmax) in zip(row_axes, panels):
            image = ax.pcolormesh(times_days, freqs_band, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            if title.startswith("A+E"):
                energy_image = image
            elif title.startswith("frequency"):
                score_image = image
            else:
                tube_image = image
            if len(path_times):
                ax.plot(path_times, path_freqs, color="cyan", lw=1.4)
                ax.scatter(path_times, path_freqs, s=5, color="white", alpha=0.7, linewidths=0)
            ax.axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.12, lw=0)
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Time [days]")
            ax.grid(True, which="both", alpha=0.14)
        row_axes[0].set_ylabel(f"{source}\nFrequency [Hz]")

    if energy_image is not None:
        cbar = fig.colorbar(energy_image, ax=axes[:, 0].tolist(), shrink=0.78, pad=0.012)
        cbar.set_label(r"$\log_{10}(|A_T|^2 + |E_T|^2)$")
    if score_image is not None:
        cbar = fig.colorbar(score_image, ax=axes[:, 1].tolist(), shrink=0.78, pad=0.012)
        cbar.set_label("robust chirp score")
    if tube_image is not None:
        cbar = fig.colorbar(tube_image, ax=axes[:, 2].tolist(), shrink=0.78, pad=0.012)
        cbar.set_label(r"$\log_{10}$ tube energy")
    fig.suptitle("Monotonic chirp morphology track on T-whitened A/E maps")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Find a monotonic chirp-like morphology track in T-whitened A/E WDM maps.")
    parser.add_argument("--input-prefix", default="twhitened_wdm_pycwb_mra_full_aet")
    parser.add_argument("--resolution", default="dt2_M512")
    parser.add_argument("--sources", nargs="+", choices=("noisy", "gw_only"), default=["noisy", "gw_only"])
    parser.add_argument("--search-low-hz", type=float, default=5e-4)
    parser.add_argument("--search-high-hz", type=float, default=3e-2)
    parser.add_argument("--z-floor", type=float, default=0.25)
    parser.add_argument("--smooth-freq-bins", type=int, default=3)
    parser.add_argument("--smooth-time-bins", type=int, default=3)
    parser.add_argument("--max-jump-bins", type=int, default=10)
    parser.add_argument("--step-penalty", type=float, default=0.08)
    parser.add_argument("--jump-penalty", type=float, default=0.02)
    parser.add_argument("--tube-width-bins", type=int, default=2)
    parser.add_argument("--no-anchor-end-global-peak", action="store_true")
    parser.add_argument("--no-end-at-path-peak", action="store_true")
    args = parser.parse_args()

    input_npz = OUTPUT_DIR / f"{args.input_prefix}_maps.npz"
    if not input_npz.exists():
        raise FileNotFoundError(f"Missing complex T-whitened map file: {input_npz}")

    maps = np.load(input_npz)
    signal = np.asarray(maps["signal"], dtype=float)
    summaries = []
    path_tables = []
    products_by_source = {}
    for source in args.sources:
        summary, path_table, products = run_source(
            maps,
            source=source,
            resolution=args.resolution,
            search_low_hz=float(args.search_low_hz),
            search_high_hz=float(args.search_high_hz),
            z_floor=float(args.z_floor),
            smooth_freq_bins=int(args.smooth_freq_bins),
            smooth_time_bins=int(args.smooth_time_bins),
            max_jump_bins=int(args.max_jump_bins),
            step_penalty=float(args.step_penalty),
            jump_penalty=float(args.jump_penalty),
            tube_width_bins=int(args.tube_width_bins),
            anchor_end_global_peak=not bool(args.no_anchor_end_global_peak),
            end_at_path_peak=not bool(args.no_end_at_path_peak),
        )
        summaries.append(summary)
        path_tables.append(path_table)
        products_by_source[source] = products

    summary_table = pd.concat(summaries, ignore_index=True)
    path_table = pd.concat(path_tables, ignore_index=True)
    tag = (
        f"{args.input_prefix}_{args.resolution}_chirp_track"
        f"_s{args.search_low_hz:g}-{args.search_high_hz:g}"
        f"_zf{args.z_floor:g}_mj{args.max_jump_bins}_sp{args.step_penalty:g}_jp{args.jump_penalty:g}_tw{args.tube_width_bins}"
        f"_{'anchorpeak' if not args.no_anchor_end_global_peak else ('endpeak' if not args.no_end_at_path_peak else 'fullpath')}"
    )
    summary_csv = OUTPUT_DIR / f"{tag}_summary.csv"
    path_csv = OUTPUT_DIR / f"{tag}_path.csv"
    products_npz = OUTPUT_DIR / f"{tag}_maps.npz"
    plot_png = PLOT_DIR / f"{tag}.png"
    report_json = OUTPUT_DIR / f"{tag}_report.json"

    summary_table.to_csv(summary_csv, index=False)
    path_table.to_csv(path_csv, index=False)
    payload: dict[str, np.ndarray] = {"signal": signal}
    for source, products in products_by_source.items():
        for name, value in products.items():
            payload[f"{source}_{name}"] = value
    np.savez_compressed(products_npz, **payload)
    plot_products(plot_png, products_by_source=products_by_source, signal=signal)

    report = {
        "method": "monotonic dynamic-programming chirp track on frequency-normalized T-whitened A/E energy; A/E complex cross-energy is computed inside a tube around the path",
        "input_npz": str(input_npz),
        "resolution": args.resolution,
        "config": {
            "low_hz": LOW_HZ,
            "high_hz": HIGH_HZ,
            "search_low_hz": float(args.search_low_hz),
            "search_high_hz": float(args.search_high_hz),
            "z_floor": float(args.z_floor),
            "smooth_freq_bins": int(args.smooth_freq_bins),
            "smooth_time_bins": int(args.smooth_time_bins),
            "max_jump_bins": int(args.max_jump_bins),
            "step_penalty": float(args.step_penalty),
            "jump_penalty": float(args.jump_penalty),
            "tube_width_bins": int(args.tube_width_bins),
            "anchor_end_global_peak": not bool(args.no_anchor_end_global_peak),
            "end_at_path_peak": not bool(args.no_end_at_path_peak),
        },
        "summary": json.loads(summary_table.to_json(orient="records")),
        "outputs": {
            "summary_csv": str(summary_csv),
            "path_csv": str(path_csv),
            "products_npz": str(products_npz),
            "plot": str(plot_png),
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report = {report_json}")


if __name__ == "__main__":
    main()
