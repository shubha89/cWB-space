from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lisainstrument import Instrument
from pytdi import Data
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import diagnostics.generate_2week_mbhb_then_glitch_highrate_tdi as week2
import diagnostics.run_broad_injection_recovery_3mo as base
from scripts.clean_wdm import WDMResolution, compute_wdm_transform, require_wdm_dependencies
from scripts.six_link_generation import LisaLinkSimulationConfig
from scripts.tdi_computation import _evaluate_sagnac_tdi, apply_frequency_cut


RUN_DIR = PROJECT_ROOT / "diagnostics" / "simulated_2week_mbhb_then_glitch_highrate"
OUTPUT_DIR = RUN_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
TMP_DIR = RUN_DIR / "tmp_zeta_wdm"
PRODUCTS_NPZ = OUTPUT_DIR / "two_week_mbhb_then_glitch_tdi.npz"
MBHB_CATALOG = OUTPUT_DIR / "mbhb_catalog.csv"
ORBIT_H5 = RUN_DIR / "inputs" / "orbits_2week.h5"

DAY = 86_400.0
WEEK_DURATION = 7.0 * DAY
CHUNK_DURATION = DAY
OVERLAP_SECONDS = 3_000.0
NOISE_BURN_IN_SECONDS = 600.0
HIGH_DT = 0.1
TARGET_DT = 4.0
ORBIT_DT = 600.0
CENTRAL_FREQ_HZ = 2.816e14
LOW_HZ = 1.0e-5
HIGH_HZ = 1.0e-1
NOISE_SEED = 752001
MAX_WORKERS = int(os.environ.get("LISA_TF_MAX_WORKERS", "2"))

RESOLUTION = WDMResolution("dt4_M512", TARGET_DT, 512)
ABC_CHANNELS = ("alpha2", "beta2", "gamma2")
ZETA_CHANNELS = ("zeta21", "zeta22", "zeta23")
SAGNAC_CHANNELS = (*ABC_CHANNELS, *ZETA_CHANNELS)
LOCAL_HALF_WINDOW_DAYS = 0.35


def main() -> None:
    started = time.time()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    products = np.load(PRODUCTS_NPZ)
    catalog = pd.read_csv(MBHB_CATALOG)
    week_samples = int(round(WEEK_DURATION / TARGET_DT))
    t_lisa = np.asarray(products["t_lisa"][:week_samples], dtype=float)

    configure_source_generation()
    grid = make_week1_grid(t_lisa)

    print("generate source-only MBHB Sagnac products", flush=True)
    mbhb_sources, source_windows = generate_mbhb_sagnac_sources(grid)
    source_sum = sum_products(mbhb_sources)

    print(f"generate week-1 high-rate Sagnac noise chunks with max_workers={MAX_WORKERS}", flush=True)
    noise, chunk_rows = generate_week1_noise_sagnac(t_lisa)
    observed = {channel: noise[channel] + source_sum[channel] for channel in SAGNAC_CHANNELS}

    sagnac_npz = OUTPUT_DIR / "mbhb_week1_sagnac_zeta_products.npz"
    save_sagnac_products(sagnac_npz, t_lisa, noise, source_sum, observed, mbhb_sources)

    print("compute zeta WDM maps and summaries", flush=True)
    analysis = compute_zeta_wdm(t_lisa, noise, observed)
    summary = summarize_mbhbs(analysis, catalog)
    summary_csv = OUTPUT_DIR / "mbhb_week1_zeta_wdm_summary.csv"
    summary.to_csv(summary_csv, index=False)
    maps_npz = OUTPUT_DIR / "mbhb_week1_zeta_wdm_maps.npz"
    save_zeta_maps(maps_npz, analysis)
    plot_path = plot_zeta_wdm(analysis, catalog)
    qc_csv = OUTPUT_DIR / "mbhb_week1_zeta_noise_chunk_qc.csv"
    pd.DataFrame(chunk_rows).to_csv(qc_csv, index=False)
    report = {
        "elapsed_seconds": time.time() - started,
        "input_tdi_products": str(PRODUCTS_NPZ),
        "sagnac_products": str(sagnac_npz),
        "summary_csv": str(summary_csv),
        "maps_npz": str(maps_npz),
        "plot": str(plot_path),
        "chunk_qc_csv": str(qc_csv),
        "wdm_resolution": RESOLUTION.label,
        "zeta_statistic": "G2 zeta-WDM uses noise-median normalized Sagnac WDM powers; positive excess is max(observed/noise_floor - noise/noise_floor, 0).",
    }
    report_path = OUTPUT_DIR / "mbhb_week1_zeta_wdm_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(report_path)
    print(plot_path)


def configure_source_generation() -> None:
    week2.configure_base()
    base.SCIENCE_DURATION = WEEK_DURATION
    base.DT = TARGET_DT
    base.BURN_IN_SECONDS = week2.SOURCE_BURN_IN_SECONDS
    base.ORBIT_DT = ORBIT_DT
    base.LOW_HZ = LOW_HZ
    base.HIGH_HZ = HIGH_HZ
    base.CENTRAL_FREQ_HZ = CENTRAL_FREQ_HZ
    base.TMP_DIR = TMP_DIR
    base.TMP_DIR.mkdir(parents=True, exist_ok=True)


def make_week1_grid(t_lisa_science: np.ndarray) -> base.RunGrid:
    burn_in_size = int(round(week2.SOURCE_BURN_IN_SECONDS / TARGET_DT))
    n_science = len(t_lisa_science)
    n_total = n_science + burn_in_size
    t0_lisa = float(t_lisa_science[0] - burn_in_size * TARGET_DT)
    t_lisa_full = t0_lisa + np.arange(n_total, dtype=float) * TARGET_DT
    t_waveform_full = (np.arange(n_total, dtype=float) - burn_in_size) * TARGET_DT
    is_science = np.arange(n_total) >= burn_in_size
    return base.RunGrid(
        n_science=n_science,
        burn_in_size=burn_in_size,
        n_total=n_total,
        t_lisa_full=t_lisa_full,
        t_lisa_science=t_lisa_full[is_science],
        t_waveform_full=t_waveform_full,
        is_science=is_science,
        t0_lisa=t0_lisa,
        orbits_h5=ORBIT_H5,
    )


def generate_mbhb_sagnac_sources(grid: base.RunGrid) -> tuple[list[dict[str, np.ndarray]], list[dict[str, float]]]:
    products = []
    windows = []
    for case in week2.MBHB_CASES:
        product, window = base.generate_mbhb_gw_product(case, grid)
        products.append({channel: np.asarray(product.sagnac[channel], dtype=float) for channel in SAGNAC_CHANNELS})
        windows.append(window)
    return products, windows


def sum_products(products: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {channel: sum(product[channel] for product in products) for channel in SAGNAC_CHANNELS}


def generate_week1_noise_sagnac(t_lisa: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    n_target = len(t_lisa)
    noise = {channel: np.zeros(n_target, dtype=float) for channel in SAGNAC_CHANNELS}
    tasks = []
    for chunk_id, start_s in enumerate(np.arange(0.0, WEEK_DURATION, CHUNK_DURATION), start=1):
        tasks.append(
            {
                "chunk_id": chunk_id,
                "start_s": float(start_s),
                "stop_s": float(min(WEEK_DURATION, start_s + CHUNK_DURATION)),
                "t0_lisa": float(t_lisa[0]),
                "orbits_h5": str(ORBIT_H5),
                "tmp_dir": str(TMP_DIR),
            }
        )
    if MAX_WORKERS <= 1:
        results = [compute_sagnac_noise_chunk(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(compute_sagnac_noise_chunk, task): task["chunk_id"] for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
                print(f"finished Sagnac noise chunk {futures[future]:02d}/{len(tasks):02d}", flush=True)

    rows = []
    for chunk, row in sorted(results, key=lambda item: int(item[1]["chunk_id"])):
        out_start = int(round(float(row["science_start_s"]) / TARGET_DT))
        out_stop = int(round(float(row["science_stop_s"]) / TARGET_DT))
        expected = out_stop - out_start
        for channel in SAGNAC_CHANNELS:
            values = np.asarray(chunk[channel], dtype=float)[:expected]
            if len(values) != expected:
                raise ValueError(f"Chunk {row['chunk_id']} {channel} length {len(values)} does not match {expected}.")
            noise[channel][out_start:out_stop] = values
        rows.append(row)
    return noise, sorted(rows, key=lambda row: int(row["chunk_id"]))


def compute_sagnac_noise_chunk(task: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    chunk_id = int(task["chunk_id"])
    start_s = float(task["start_s"])
    stop_s = float(task["stop_s"])
    padded_start_s = max(0.0, start_s - OVERLAP_SECONDS)
    padded_stop_s = min(WEEK_DURATION, stop_s + OVERLAP_SECONDS)
    instrument_start_s = padded_start_s - NOISE_BURN_IN_SECONDS
    chunk_duration = padded_stop_s - instrument_start_s
    n_high = int(round(chunk_duration / HIGH_DT))
    crop_start = int(round((start_s - instrument_start_s) / HIGH_DT))
    crop_stop = crop_start + int(round((stop_s - start_s) / HIGH_DT))
    h5_path = Path(str(task["tmp_dir"])) / f"sagnac_noise_chunk_{chunk_id:02d}.h5"
    h5_path.unlink(missing_ok=True)

    instrument = Instrument(
        size=n_high,
        dt=HIGH_DT,
        t0=float(task["t0_lisa"]) + instrument_start_s,
        orbits=str(task["orbits_h5"]),
        gws=None,
        orbit_dataset="tcb/ltt",
        central_freq=CENTRAL_FREQ_HZ,
        seed=NOISE_SEED + chunk_id - 1,
        noises_f_min_hz=LOW_HZ,
        clock_f_min_hz=LOW_HZ,
        chunk_gw_files=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        instrument.export_hdf5(path=str(h5_path), overwrite=True)

    config = LisaLinkSimulationConfig(
        dt=HIGH_DT,
        science_duration=chunk_duration,
        burn_in_seconds=0.0,
        central_freq_hz=CENTRAL_FREQ_HZ,
        orbit_dt=ORBIT_DT,
        frequency_low_hz=LOW_HZ,
        frequency_high_hz=HIGH_HZ,
        tdi_generation=2,
        interpolation_order=31,
        tdi_unit="frequency",
        orbit_dataset="tcb/ltt",
        gw_dataset="tcb",
    )
    data = Data.from_instrument(str(h5_path), signals="fluctuations", skipped=0)
    channels, _ = _evaluate_sagnac_tdi(data, config)
    factor = int(round(TARGET_DT / HIGH_DT))
    out = {}
    for channel in SAGNAC_CHANNELS:
        science_high = np.asarray(channels[channel][crop_start:crop_stop], dtype=float)
        filtered = apply_frequency_cut(science_high, HIGH_DT, LOW_HZ, HIGH_HZ)
        out[channel] = resample_poly(filtered, up=1, down=factor)
    h5_path.unlink(missing_ok=True)
    row = {
        "chunk_id": chunk_id,
        "science_start_s": start_s,
        "science_stop_s": stop_s,
        **{f"{channel}_std_hz": float(np.std(out[channel])) for channel in SAGNAC_CHANNELS},
        **{f"{channel}_absmax_hz": float(np.max(np.abs(out[channel]))) for channel in SAGNAC_CHANNELS},
    }
    return out, row


def save_sagnac_products(
    path: Path,
    t_lisa: np.ndarray,
    noise: dict[str, np.ndarray],
    source_sum: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    mbhb_sources: list[dict[str, np.ndarray]],
) -> None:
    arrays: dict[str, np.ndarray] = {"t_lisa": t_lisa, "time_day": (t_lisa - t_lisa[0]) / DAY}
    for channel in SAGNAC_CHANNELS:
        arrays[f"noise_{channel}"] = noise[channel]
        arrays[f"source_sum_{channel}"] = source_sum[channel]
        arrays[f"observed_{channel}"] = observed[channel]
    for index, product in enumerate(mbhb_sources, start=1):
        for channel in SAGNAC_CHANNELS:
            arrays[f"mbhb{index}_{channel}"] = product[channel]
    np.savez_compressed(path, **arrays)


def compute_zeta_wdm(
    t_lisa: np.ndarray,
    noise: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    _, WDM = require_wdm_dependencies()
    native_dt = float(np.median(np.diff(t_lisa)))
    wdm = WDM(M=RESOLUTION.M, K=32, beta_order=6, precision=6, backend="numba")
    noise_power = {}
    observed_power = {}
    normalized = {}
    excess = {}
    floors = {}
    times_ref = None
    freqs_ref = None
    for channel in SAGNAC_CHANNELS:
        noise_coeff, times, freqs, _, _ = compute_wdm_transform(
            wdm,
            np.asarray(noise[channel], dtype=float),
            sample_rate=1.0 / native_dt,
            t0=float(t_lisa[0]),
            original_t1=float(t_lisa[-1]),
        )
        obs_coeff, obs_times, obs_freqs, _, _ = compute_wdm_transform(
            wdm,
            np.asarray(observed[channel], dtype=float),
            sample_rate=1.0 / native_dt,
            t0=float(t_lisa[0]),
            original_t1=float(t_lisa[-1]),
        )
        if times_ref is None:
            times_ref = times
            freqs_ref = freqs
        elif not np.array_equal(times_ref, times) or not np.array_equal(freqs_ref, freqs):
            raise ValueError(f"WDM grid mismatch for {channel}.")
        if not np.array_equal(times, obs_times) or not np.array_equal(freqs, obs_freqs):
            raise ValueError(f"Noise/observed WDM grid mismatch for {channel}.")
        noise_power[channel] = np.abs(noise_coeff) ** 2
        observed_power[channel] = np.abs(obs_coeff) ** 2
        floor = np.nanmedian(noise_power[channel], axis=1)
        floors[channel] = np.where(np.isfinite(floor) & (floor > 0.0), floor, 1.0e-300)
        normalized[channel] = observed_power[channel] / floors[channel][:, None]
        noise_norm = noise_power[channel] / floors[channel][:, None]
        excess[channel] = np.maximum(normalized[channel] - noise_norm, 0.0)
    abc_excess = sum(excess[channel] for channel in ABC_CHANNELS)
    zeta_excess = sum(excess[channel] for channel in ZETA_CHANNELS)
    ratio = np.divide(
        zeta_excess,
        abc_excess,
        out=np.full_like(zeta_excess, np.nan, dtype=float),
        where=abc_excess > 0.0,
    )
    fraction = zeta_excess / np.maximum(abc_excess + zeta_excess, 1.0e-300)
    return {
        "times_day": (np.asarray(times_ref, dtype=float) - float(t_lisa[0])) / DAY,
        "freqs_hz": np.asarray(freqs_ref, dtype=float),
        "floors": floors,
        "normalized": normalized,
        "excess": excess,
        "abc_excess": abc_excess,
        "zeta_excess": zeta_excess,
        "zeta_over_abc_pixel": ratio,
        "zeta_fraction_pixel": fraction,
    }


def summarize_mbhbs(analysis: dict[str, object], catalog: pd.DataFrame) -> pd.DataFrame:
    times = np.asarray(analysis["times_day"], dtype=float)
    freqs = np.asarray(analysis["freqs_hz"], dtype=float)
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    abc = np.asarray(analysis["abc_excess"], dtype=float)
    zeta = np.asarray(analysis["zeta_excess"], dtype=float)
    rows = []
    for row in catalog.itertuples(index=False):
        peak = float(row.response_peak_day)
        local = (times >= peak - LOCAL_HALF_WINDOW_DAYS) & (times <= peak + LOCAL_HALF_WINDOW_DAYS)
        mask = np.ix_(band, local)
        abc_sum = float(np.nansum(abc[mask]))
        zeta_sum = float(np.nansum(zeta[mask]))
        n_pixels = int(np.sum(band) * np.sum(local))
        rows.append(
            {
                "case_id": row.case_id,
                "response_peak_day": peak,
                "local_half_window_days": LOCAL_HALF_WINDOW_DAYS,
                "n_pixels": n_pixels,
                "abc_positive_excess_sum": abc_sum,
                "zeta_positive_excess_sum": zeta_sum,
                "zeta_wdm_ratio": zeta_sum / abc_sum if abc_sum > 0.0 else np.nan,
                "abc_positive_excess_per_pixel_per_channel": abc_sum / (n_pixels * len(ABC_CHANNELS)),
                "zeta_positive_excess_per_pixel_per_channel": zeta_sum / (n_pixels * len(ZETA_CHANNELS)),
            }
        )
    return pd.DataFrame(rows)


def save_zeta_maps(path: Path, analysis: dict[str, object]) -> None:
    arrays = {
        "times_day": np.asarray(analysis["times_day"], dtype=float),
        "freqs_hz": np.asarray(analysis["freqs_hz"], dtype=float),
        "abc_positive_excess": np.asarray(analysis["abc_excess"], dtype=float),
        "zeta_positive_excess": np.asarray(analysis["zeta_excess"], dtype=float),
        "zeta_over_abc_pixel": np.asarray(analysis["zeta_over_abc_pixel"], dtype=float),
        "zeta_fraction_pixel": np.asarray(analysis["zeta_fraction_pixel"], dtype=float),
    }
    for channel in SAGNAC_CHANNELS:
        arrays[f"{channel}_noise_floor"] = np.asarray(analysis["floors"][channel], dtype=float)
        arrays[f"{channel}_normalized_power"] = np.asarray(analysis["normalized"][channel], dtype=float)
        arrays[f"{channel}_positive_excess"] = np.asarray(analysis["excess"][channel], dtype=float)
    np.savez_compressed(path, **arrays)


def plot_zeta_wdm(analysis: dict[str, object], catalog: pd.DataFrame) -> Path:
    output = PLOT_DIR / "mbhb_week1_zeta_wdm_positive_excess.png"
    times = np.asarray(analysis["times_day"], dtype=float)
    freqs = np.asarray(analysis["freqs_hz"], dtype=float)
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    maps = [
        ("ABC positive excess", np.asarray(analysis["abc_excess"], dtype=float)),
        ("zeta positive excess", np.asarray(analysis["zeta_excess"], dtype=float)),
        ("zeta fraction", np.asarray(analysis["zeta_fraction_pixel"], dtype=float)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.4), sharex=True, sharey=True, constrained_layout=True)
    for ax, (title, values) in zip(axes, maps):
        if title == "zeta fraction":
            data = values[band, :]
            finite = data[np.isfinite(data)]
            vmin, vmax = 0.0, min(1.0, float(np.nanpercentile(finite, 99.5))) if finite.size else 1.0
            image = ax.pcolormesh(times, freqs[band], data, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        else:
            data = np.log10(np.maximum(values[band, :], 1.0e-300))
            finite = data[np.isfinite(data) & (data > -250)]
            vmin, vmax = np.nanpercentile(finite, [2.0, 99.5]) if finite.size else (-1.0, 1.0)
            image = ax.pcolormesh(times, freqs[band], data, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        for source in catalog.itertuples(index=False):
            ax.axvline(float(source.response_peak_day), color="#00ffff", lw=0.9, alpha=0.78)
        ax.set_yscale("log")
        ax.set_ylim(LOW_HZ, HIGH_HZ)
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.14)
        cbar = fig.colorbar(image, ax=ax, pad=0.01)
        cbar.set_label(r"$\zeta/(ABC+\zeta)$" if title == "zeta fraction" else r"$\log_{10}$ value")
    axes[-1].set_xlabel("time from dataset start [days]")
    fig.suptitle("Week-1 MBHB G2 zeta-WDM analysis")
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


if __name__ == "__main__":
    main()
