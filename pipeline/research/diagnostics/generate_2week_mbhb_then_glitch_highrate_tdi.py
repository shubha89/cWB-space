from __future__ import annotations

import json
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from lisainstrument import Instrument
from lisaorbits import OEMOrbits
from pytdi import Data
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import diagnostics.run_broad_injection_recovery_3mo as base
from scripts.six_link_generation import LisaLinkSimulationConfig, read_orbit_time_range
from scripts.tdi_computation import _evaluate_michelson_tdi, apply_frequency_cut


RUN_DIR = PROJECT_ROOT / "diagnostics" / "simulated_2week_mbhb_then_glitch_highrate"
INPUT_DIR = RUN_DIR / "inputs"
OUTPUT_DIR = RUN_DIR / "outputs"
TMP_DIR = RUN_DIR / "tmp"

DAY = 86_400.0
SCIENCE_DURATION = 14.0 * DAY
WEEK_DURATION = 7.0 * DAY
CHUNK_DURATION = DAY
OVERLAP_SECONDS = 3_000.0
NOISE_BURN_IN_SECONDS = 600.0
SOURCE_BURN_IN_SECONDS = 600.0
HIGH_DT = 0.1
TARGET_DT = 4.0
ORBIT_DT = 600.0
ORBIT_MARGIN_SECONDS = 2.0 * DAY
CENTRAL_FREQ_HZ = 2.816e14
LOW_HZ = 1.0e-5
HIGH_HZ = 1.0e-1
NOISE_SEED = 752001
MAX_WORKERS = int(os.environ.get("LISA_TF_MAX_WORKERS", "2"))

MICHELSON_CHANNELS = ("X2", "Y2", "Z2")
AET_CHANNELS = ("A2", "E2", "T2")
TDI_CHANNELS = (*MICHELSON_CHANNELS, *AET_CHANNELS)

MBHB_CASES = (
    base.MBHBCase("mbhb_week1_m1e5_q1_day2p5", 1.0e5, 1.0, 0.45, 0.35, 2.5, 0.35, 0.2, 0.1, -0.1, NOISE_SEED),
    base.MBHBCase("mbhb_week1_m3e5_q06_day4p0", 3.0e5, 0.6, 1.75, -0.45, 4.0, 0.75, 1.1, -0.2, 0.2, NOISE_SEED),
    base.MBHBCase("mbhb_week1_m1e6_q03_day5p5", 1.0e6, 0.3, 3.10, 0.10, 5.5, 1.05, 2.4, 0.3, -0.3, NOISE_SEED),
)

GLITCH_CASES = (
    base.GlitchCase("glitch_week2_12_day9p0", "readout_isi_carrier_12", 9.0, 1400.0, 1.8e-4, NOISE_SEED),
    base.GlitchCase("glitch_week2_23_day10p7", "readout_isi_carrier_23", 10.7, 2600.0, 1.2e-4, NOISE_SEED),
    base.GlitchCase("glitch_week2_31_day12p3", "readout_isi_carrier_31", 12.3, 1000.0, 3.0e-4, NOISE_SEED),
)


def main() -> None:
    started = time.time()
    reset_dirs()
    configure_base()
    grid = make_grid()
    write_config()

    log("generate source-only MBHB TDI for week 1")
    mbhb_products, mbhb_catalog = generate_sources(MBHB_CASES, grid, generator=base.generate_mbhb_gw_product)
    log("generate source-only glitch TDI for week 2")
    glitch_products, glitch_catalog = generate_sources(GLITCH_CASES, grid, generator=base.generate_glitch_product)

    mbhb_sum = sum_source_products(mbhb_products)
    glitch_sum = sum_source_products(glitch_products)
    source_sum = add_channel_products(mbhb_sum, glitch_sum)

    log(f"generate 14 one-day high-rate noise chunks with max_workers={MAX_WORKERS}")
    noise, chunk_rows = generate_noise_product(grid)
    observed = add_channel_products(noise, source_sum)

    noise_qc = noise_quality_rows(noise, chunk_rows)
    boundary_rows = boundary_diagnostics(noise)
    source_rows = source_summary_rows(mbhb_products, glitch_products, mbhb_sum, glitch_sum, source_sum)

    save_dataset(grid, noise, source_sum, mbhb_sum, glitch_sum, observed, mbhb_products, glitch_products)
    pd.DataFrame(mbhb_catalog).to_csv(OUTPUT_DIR / "mbhb_catalog.csv", index=False)
    pd.DataFrame(glitch_catalog).to_csv(OUTPUT_DIR / "glitch_catalog.csv", index=False)
    pd.DataFrame(chunk_rows).to_csv(OUTPUT_DIR / "noise_chunk_qc.csv", index=False)
    pd.DataFrame(noise_qc).to_csv(OUTPUT_DIR / "noise_level_qc.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(OUTPUT_DIR / "noise_boundary_qc.csv", index=False)
    pd.DataFrame(source_rows).to_csv(OUTPUT_DIR / "source_level_summary.csv", index=False)

    report = {
        "elapsed_seconds": time.time() - started,
        "run_dir": str(RUN_DIR),
        "products_npz": str(OUTPUT_DIR / "two_week_mbhb_then_glitch_tdi.npz"),
        "noise_level_qc": str(OUTPUT_DIR / "noise_level_qc.csv"),
        "noise_chunk_qc": str(OUTPUT_DIR / "noise_chunk_qc.csv"),
        "noise_boundary_qc": str(OUTPUT_DIR / "noise_boundary_qc.csv"),
        "mbhb_catalog": str(OUTPUT_DIR / "mbhb_catalog.csv"),
        "glitch_catalog": str(OUTPUT_DIR / "glitch_catalog.csv"),
        "noise_ok": noise_ok(noise_qc),
        "noise_std_hz": {
            channel: float(np.std(noise[channel]))
            for channel in AET_CHANNELS
        },
        "max_abs_boundary_mean_jump_hz": max_boundary_jump(boundary_rows),
    }
    (OUTPUT_DIR / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_summary(report, noise_qc, chunk_rows, source_rows)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(pd.DataFrame(noise_qc).to_string(index=False))
    print(OUTPUT_DIR / "two_week_mbhb_then_glitch_tdi.npz")


def reset_dirs() -> None:
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    for path in (INPUT_DIR, OUTPUT_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_base() -> None:
    base.RUN_DIR = RUN_DIR
    base.INPUT_DIR = INPUT_DIR
    base.OUTPUT_DIR = OUTPUT_DIR
    base.PLOT_DIR = OUTPUT_DIR / "plots"
    base.TMP_DIR = TMP_DIR
    base.SCIENCE_DURATION = SCIENCE_DURATION
    base.DT = TARGET_DT
    base.BURN_IN_SECONDS = SOURCE_BURN_IN_SECONDS
    base.ORBIT_DT = ORBIT_DT
    base.ORBIT_MARGIN_SECONDS = ORBIT_MARGIN_SECONDS
    base.LOW_HZ = LOW_HZ
    base.HIGH_HZ = HIGH_HZ
    base.CENTRAL_FREQ_HZ = CENTRAL_FREQ_HZ
    base.MBHB_CASES = MBHB_CASES
    base.GLITCH_CASES = GLITCH_CASES


def write_config() -> None:
    config = {
        "science_duration_days": SCIENCE_DURATION / DAY,
        "week1": "MBHB injections",
        "week2": "glitch injections",
        "target_dt_s": TARGET_DT,
        "noise_highrate_dt_s": HIGH_DT,
        "noise_generation": "lisainstrument default anti-aliasing at dt=0.1 s; PyTDI high-rate; crop science; frequency cut; downsample to 4 s",
        "frequency_low_hz": LOW_HZ,
        "frequency_high_hz": HIGH_HZ,
        "chunk_duration_days": CHUNK_DURATION / DAY,
        "overlap_seconds": OVERLAP_SECONDS,
        "noise_burn_in_seconds": NOISE_BURN_IN_SECONDS,
        "source_burn_in_seconds": SOURCE_BURN_IN_SECONDS,
        "central_freq_hz": CENTRAL_FREQ_HZ,
        "noise_seed": NOISE_SEED,
        "max_workers": MAX_WORKERS,
        "tdi_channels": list(TDI_CHANNELS),
        "mbhb_cases": [base.asdict(case) | {"eta": case.eta, "distance_mpc": base.DISTANCE_MPC} for case in MBHB_CASES],
        "glitch_cases": [base.asdict(case) for case in GLITCH_CASES],
        "checkpoint": str(PROJECT_ROOT / "docs" / "checkpoints" / "highrate_antialias_sampling_checkpoint_2026-07-27.md"),
        "sagnac_zeta": "not included in this product; this run saves Michelson XYZ and orthogonal AET only",
    }
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")


def make_grid() -> base.RunGrid:
    n_science = int(round(SCIENCE_DURATION / TARGET_DT))
    burn_in_size = int(round(SOURCE_BURN_IN_SECONDS / TARGET_DT))
    n_total = n_science + burn_in_size
    orbit_size = int(np.ceil((n_total * TARGET_DT + 2.0 * ORBIT_MARGIN_SECONDS) / ORBIT_DT)) + 1
    orbits_h5 = INPUT_DIR / "orbits_2week.h5"
    orbits = OEMOrbits.from_included("esa-trailing", version="latest")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orbits.write(str(orbits_h5), t0="init", dt=ORBIT_DT, size=orbit_size, mode="x")
    orbit = read_orbit_time_range(orbits_h5)
    t0_lisa = orbit.t0 + ORBIT_MARGIN_SECONDS
    t_lisa_full = t0_lisa + np.arange(n_total, dtype=float) * TARGET_DT
    if t_lisa_full[-1] > orbit.t1:
        raise ValueError("The 2-week time grid does not fit inside the generated orbit file.")
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
        orbits_h5=orbits_h5,
    )


def generate_sources(cases, grid: base.RunGrid, *, generator) -> tuple[list[dict[str, np.ndarray]], list[dict[str, object]]]:
    products = []
    catalog_rows = []
    for case in cases:
        log(f"source {case.case_id}")
        product, window = generator(case, grid)
        products.append(product_channels(product))
        catalog_rows.append({**base.case_metadata(case), **window})
    return products, catalog_rows


def product_channels(product: base.TDIProduct) -> dict[str, np.ndarray]:
    out = {channel: np.asarray(product.aet[channel], dtype=float) for channel in AET_CHANNELS}
    out.update(aet_to_xyz(out))
    return {channel: out[channel] for channel in TDI_CHANNELS}


def aet_to_xyz(channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    a = np.asarray(channels["A2"], dtype=float)
    e = np.asarray(channels["E2"], dtype=float)
    t = np.asarray(channels["T2"], dtype=float)
    s1 = np.sqrt(2.0) * a
    s2 = np.sqrt(6.0) * e
    s3 = np.sqrt(3.0) * t
    x = (s3 - 1.5 * s1 + 0.5 * s2) / 3.0
    y = x + 0.5 * s1 - 0.5 * s2
    z = x + s1
    return {"X2": x, "Y2": y, "Z2": z}


def sum_source_products(products: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not products:
        raise ValueError("No source products to sum.")
    return {channel: sum(product[channel] for product in products) for channel in TDI_CHANNELS}


def add_channel_products(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {channel: np.asarray(a[channel], dtype=float) + np.asarray(b[channel], dtype=float) for channel in TDI_CHANNELS}


def generate_noise_product(grid: base.RunGrid) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    n_target = grid.n_science
    noise = {channel: np.zeros(n_target, dtype=float) for channel in TDI_CHANNELS}
    rows: list[dict[str, object]] = []
    tasks = []
    for chunk_id, start_s in enumerate(np.arange(0.0, SCIENCE_DURATION, CHUNK_DURATION), start=1):
        stop_s = min(SCIENCE_DURATION, start_s + CHUNK_DURATION)
        tasks.append(
            {
                "chunk_id": chunk_id,
                "start_s": float(start_s),
                "stop_s": float(stop_s),
                "t0_lisa": float(grid.t0_lisa),
                "orbits_h5": str(grid.orbits_h5),
                "tmp_dir": str(TMP_DIR),
            }
        )

    if MAX_WORKERS <= 1:
        results = [compute_noise_chunk(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(compute_noise_chunk, task): task["chunk_id"] for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
                log(f"finished high-rate noise chunk {futures[future]:02d}/{len(tasks):02d}")

    for chunk, row in sorted(results, key=lambda item: int(item[1]["chunk_id"])):
        out_start = int(round(float(row["science_start_s"]) / TARGET_DT))
        out_stop = int(round(float(row["science_stop_s"]) / TARGET_DT))
        expected = out_stop - out_start
        for channel in TDI_CHANNELS:
            values = np.asarray(chunk[channel], dtype=float)[:expected]
            if len(values) != expected:
                raise ValueError(f"Chunk {row['chunk_id']} {channel} length {len(values)} does not match expected {expected}.")
            noise[channel][out_start:out_stop] = values
        rows.append(row)
    return noise, sorted(rows, key=lambda row: int(row["chunk_id"]))


def compute_noise_chunk(task: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    chunk_id = int(task["chunk_id"])
    start_s = float(task["start_s"])
    stop_s = float(task["stop_s"])
    padded_start_s = max(0.0, start_s - OVERLAP_SECONDS)
    padded_stop_s = min(SCIENCE_DURATION, stop_s + OVERLAP_SECONDS)
    instrument_start_s = padded_start_s - NOISE_BURN_IN_SECONDS
    chunk_duration = padded_stop_s - instrument_start_s
    n_high = int(round(chunk_duration / HIGH_DT))
    crop_start = int(round((start_s - instrument_start_s) / HIGH_DT))
    crop_stop = crop_start + int(round((stop_s - start_s) / HIGH_DT))
    t0_lisa = float(task["t0_lisa"]) + instrument_start_s
    tmp_dir = Path(str(task["tmp_dir"]))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    h5_path = tmp_dir / f"noise_chunk_{chunk_id:02d}.h5"
    h5_path.unlink(missing_ok=True)

    instrument = Instrument(
        size=n_high,
        dt=HIGH_DT,
        t0=t0_lisa,
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
    x, y, z, a, e, t, _ = _evaluate_michelson_tdi(data, config)
    raw = {"X2": x, "Y2": y, "Z2": z, "A2": a, "E2": e, "T2": t}
    factor = int(round(TARGET_DT / HIGH_DT))
    out = {}
    for channel, values in raw.items():
        science_high = np.asarray(values[crop_start:crop_stop], dtype=float)
        if len(science_high) != int(round((stop_s - start_s) / HIGH_DT)):
            raise ValueError(f"Chunk {chunk_id} {channel} high-rate crop has wrong length.")
        filtered = apply_frequency_cut(science_high, HIGH_DT, LOW_HZ, HIGH_HZ)
        out[channel] = resample_poly(filtered, up=1, down=factor)
    h5_path.unlink(missing_ok=True)

    row = {
        "chunk_id": chunk_id,
        "science_start_s": start_s,
        "science_stop_s": stop_s,
        "science_start_day": start_s / DAY,
        "science_stop_day": stop_s / DAY,
        "padded_start_day": padded_start_s / DAY,
        "padded_stop_day": padded_stop_s / DAY,
        "instrument_start_day": instrument_start_s / DAY,
        "burn_in_seconds": NOISE_BURN_IN_SECONDS,
        "overlap_seconds": OVERLAP_SECONDS,
        "highrate_samples": n_high,
        "crop_start_highrate": crop_start,
        "crop_stop_highrate": crop_stop,
        **{f"{channel}_std_hz": float(np.std(out[channel])) for channel in TDI_CHANNELS},
        **{f"{channel}_absmax_hz": float(np.max(np.abs(out[channel]))) for channel in TDI_CHANNELS},
    }
    return out, row


def noise_quality_rows(noise: dict[str, np.ndarray], chunk_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    segments = {
        "full_14d": slice(None),
        "week1_mbhb": slice(0, int(round(WEEK_DURATION / TARGET_DT))),
        "week2_glitch": slice(int(round(WEEK_DURATION / TARGET_DT)), None),
    }
    for segment, slc in segments.items():
        for channel in TDI_CHANNELS:
            values = np.asarray(noise[channel][slc], dtype=float)
            rows.append(
                {
                    "segment": segment,
                    "channel": channel,
                    "std_hz": float(np.std(values)),
                    "mean_hz": float(np.mean(values)),
                    "p99_abs_hz": float(np.percentile(np.abs(values), 99.0)),
                    "absmax_hz": float(np.max(np.abs(values))),
                    "passes_aet_noise_scale_gate": bool(passes_noise_gate(channel, float(np.std(values)))),
                }
            )
    for channel in AET_CHANNELS:
        stds = np.asarray([float(row[f"{channel}_std_hz"]) for row in chunk_rows], dtype=float)
        rows.append(
            {
                "segment": "chunks",
                "channel": channel,
                "std_hz": float(np.median(stds)),
                "mean_hz": np.nan,
                "p99_abs_hz": float(np.percentile(stds, 99.0)),
                "absmax_hz": float(np.max(stds)),
                "passes_aet_noise_scale_gate": bool(np.all([passes_noise_gate(channel, value) for value in stds])),
            }
        )
    return rows


def passes_noise_gate(channel: str, std_hz: float) -> bool:
    if channel not in AET_CHANNELS:
        return True
    return 3.0e-6 <= std_hz <= 3.0e-5


def boundary_diagnostics(noise: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    half_width = 50
    n = len(noise["A2"])
    for boundary_s in np.arange(CHUNK_DURATION, SCIENCE_DURATION, CHUNK_DURATION):
        idx = int(round(boundary_s / TARGET_DT))
        left = slice(max(0, idx - half_width), idx)
        right = slice(idx, min(n, idx + half_width))
        for channel in TDI_CHANNELS:
            values = noise[channel]
            rows.append(
                {
                    "boundary_day": boundary_s / DAY,
                    "channel": channel,
                    "left_mean_hz": float(np.mean(values[left])),
                    "right_mean_hz": float(np.mean(values[right])),
                    "mean_jump_hz": float(np.mean(values[right]) - np.mean(values[left])),
                    "left_std_hz": float(np.std(values[left])),
                    "right_std_hz": float(np.std(values[right])),
                }
            )
    return rows


def source_summary_rows(
    mbhb_products: list[dict[str, np.ndarray]],
    glitch_products: list[dict[str, np.ndarray]],
    mbhb_sum: dict[str, np.ndarray],
    glitch_sum: dict[str, np.ndarray],
    source_sum: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    grouped = {
        **{f"mbhb{i}": product for i, product in enumerate(mbhb_products, start=1)},
        **{f"glitch{i}": product for i, product in enumerate(glitch_products, start=1)},
        "mbhb_sum": mbhb_sum,
        "glitch_sum": glitch_sum,
        "source_sum": source_sum,
    }
    for label, product in grouped.items():
        for channel in TDI_CHANNELS:
            values = np.asarray(product[channel], dtype=float)
            rows.append(
                {
                    "source": label,
                    "channel": channel,
                    "std_hz": float(np.std(values)),
                    "p99_abs_hz": float(np.percentile(np.abs(values), 99.0)),
                    "absmax_hz": float(np.max(np.abs(values))),
                    "peak_day": float(np.argmax(np.abs(values)) * TARGET_DT / DAY),
                }
            )
    return rows


def save_dataset(
    grid: base.RunGrid,
    noise: dict[str, np.ndarray],
    source_sum: dict[str, np.ndarray],
    mbhb_sum: dict[str, np.ndarray],
    glitch_sum: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    mbhb_products: list[dict[str, np.ndarray]],
    glitch_products: list[dict[str, np.ndarray]],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "t_lisa": grid.t_lisa_science,
        "time_rel_s": np.arange(grid.n_science, dtype=float) * TARGET_DT,
        "time_day": np.arange(grid.n_science, dtype=float) * TARGET_DT / DAY,
        "week1_mbhb_mask": np.arange(grid.n_science) < int(round(WEEK_DURATION / TARGET_DT)),
        "week2_glitch_mask": np.arange(grid.n_science) >= int(round(WEEK_DURATION / TARGET_DT)),
    }
    for channel in TDI_CHANNELS:
        arrays[f"noise_{channel}"] = noise[channel]
        arrays[f"source_sum_{channel}"] = source_sum[channel]
        arrays[f"mbhb_sum_{channel}"] = mbhb_sum[channel]
        arrays[f"glitch_sum_{channel}"] = glitch_sum[channel]
        arrays[f"observed_{channel}"] = observed[channel]
    for index, product in enumerate(mbhb_products, start=1):
        for channel in TDI_CHANNELS:
            arrays[f"mbhb{index}_{channel}"] = product[channel]
    for index, product in enumerate(glitch_products, start=1):
        for channel in TDI_CHANNELS:
            arrays[f"glitch{index}_{channel}"] = product[channel]
    np.savez_compressed(OUTPUT_DIR / "two_week_mbhb_then_glitch_tdi.npz", **arrays)


def noise_ok(rows: list[dict[str, object]]) -> bool:
    return all(bool(row["passes_aet_noise_scale_gate"]) for row in rows if row["channel"] in AET_CHANNELS)


def max_boundary_jump(rows: list[dict[str, object]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    return frame.groupby("channel")["mean_jump_hz"].apply(lambda x: float(np.max(np.abs(x)))).to_dict()


def write_summary(
    report: dict[str, object],
    noise_qc: list[dict[str, object]],
    chunk_rows: list[dict[str, object]],
    source_rows: list[dict[str, object]],
) -> None:
    noise = pd.DataFrame(noise_qc)
    chunks = pd.DataFrame(chunk_rows)
    source = pd.DataFrame(source_rows)
    full_aet = noise[(noise["segment"] == "full_14d") & (noise["channel"].isin(AET_CHANNELS))]
    week_aet = noise[(noise["segment"].isin(["week1_mbhb", "week2_glitch"])) & (noise["channel"].isin(AET_CHANNELS))]
    source_peaks = source[(source["source"].isin(["mbhb_sum", "glitch_sum", "source_sum"])) & (source["channel"].isin(AET_CHANNELS))]
    lines = [
        "# 2-Week MBHB-Then-Glitch High-Rate TDI Dataset",
        "",
        "Generated simulation products only; no blind WDM/clustering analysis was run.",
        "",
        "Injection layout:",
        "- Week 1: three MBHB injections.",
        "- Week 2: three link glitches.",
        "",
        "Noise convention:",
        "- `lisainstrument` at `dt=0.1 s` with its default anti-aliasing filter.",
        "- PyTDI evaluated at high rate.",
        "- Central science samples cropped before frequency cuts.",
        "- Frequency cut: `1e-5 <= f <= 1e-1 Hz`.",
        "- Downsampled final TDI to `dt=4 s`.",
        "",
        f"Noise level gate passed: `{report['noise_ok']}`",
        "",
        "Full 14-day A/E/T noise std:",
        "```text",
        full_aet[["channel", "std_hz", "p99_abs_hz", "absmax_hz"]].to_string(index=False),
        "```",
        "",
        "Week-level A/E/T noise std:",
        "```text",
        week_aet[["segment", "channel", "std_hz"]].to_string(index=False),
        "```",
        "",
        "Chunk A/E/T std range:",
        "```text",
        chunks[
            [
                "chunk_id",
                "science_start_day",
                "science_stop_day",
                "A2_std_hz",
                "E2_std_hz",
                "T2_std_hz",
            ]
        ].to_string(index=False),
        "```",
        "",
        "Source peak levels:",
        "```text",
        source_peaks[["source", "channel", "std_hz", "absmax_hz", "peak_day"]].to_string(index=False),
        "```",
        "",
        "Outputs:",
        f"- `{report['products_npz']}`",
        f"- `{report['noise_level_qc']}`",
        f"- `{report['noise_chunk_qc']}`",
        f"- `{report['noise_boundary_qc']}`",
        f"- `{report['mbhb_catalog']}`",
        f"- `{report['glitch_catalog']}`",
        "",
        f"Elapsed seconds: `{report['elapsed_seconds']:.1f}`",
    ]
    (OUTPUT_DIR / "run_summary.md").write_text("\n".join(lines) + "\n")


def log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
