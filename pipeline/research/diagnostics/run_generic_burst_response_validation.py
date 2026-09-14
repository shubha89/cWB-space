from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import shutil
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lisagwresponse import ReadStrain
from pytdi import Data
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import diagnostics.run_broad_injection_recovery_3mo as base
from scripts.blind_event_clustering import (
    SeededChirpEventConfig,
    compute_self_ae_median_wdm,
    find_seeded_chirp_events,
)
from scripts.clean_wdm import WDMResolution, compute_wdm_transform, require_wdm_dependencies


DAY = 86_400.0
SCIENCE_DURATION = 14.0 * DAY
DT = 4.0
BURN_IN_SECONDS = 600.0
ORBIT_DT = 600.0
CENTRAL_FREQ_HZ = 2.816e14
LOW_HZ = 1.0e-5
HIGH_HZ = 1.0e-1
SEARCH_HIGH_HZ = 2.0e-2
INTERPOLATION_ORDER = 31
RESOLUTION = WDMResolution("dt4_M512", DT, 512)

RUN_DIR = PROJECT_ROOT / "diagnostics" / "generic_burst_response_validation"
INPUT_DIR = RUN_DIR / "inputs"
OUTPUT_DIR = RUN_DIR / "outputs"
TMP_DIR = RUN_DIR / "tmp"
PLOT_DIR = OUTPUT_DIR / "plots"

CHECKPOINT_DIR = PROJECT_ROOT / "diagnostics" / "simulated_2week_mbhb_then_glitch_highrate"
CHECKPOINT_OUTPUTS = CHECKPOINT_DIR / "outputs"
CHECKPOINT_ORBITS = CHECKPOINT_DIR / "inputs" / "orbits_2week.h5"
CHECKPOINT_TDI = CHECKPOINT_OUTPUTS / "two_week_mbhb_then_glitch_tdi.npz"
CHECKPOINT_WEEK1_SAGNAC = CHECKPOINT_OUTPUTS / "mbhb_week1_sagnac_zeta_products.npz"
CHECKPOINT_WEEK2_SAGNAC = CHECKPOINT_OUTPUTS / "glitch_week2_sagnac_zeta_products.npz"

ABC_CHANNELS = ("alpha2", "beta2", "gamma2")
ZETA_CHANNELS = ("zeta21", "zeta22", "zeta23")
SAGNAC_CHANNELS = (*ABC_CHANNELS, *ZETA_CHANNELS)
AET_CHANNELS = ("A2", "E2", "T2")


@dataclass(frozen=True)
class ResponseBurstCase:
    case_id: str
    case_type: str
    morphology: str
    center_day: float
    duration_s: float
    ra: float
    dec: float
    strain_amp: float
    frequency_hz: float | None = None
    band_low_hz: float | None = None
    band_high_hz: float | None = None
    phase: float = 0.0
    seed: int = 0


SIGNAL_CASES = (
    ResponseBurstCase("response_sg_01", "signal_like", "sine_gaussian", 2.15, 9_000.0, 0.45, 0.35, 8.0e-18, frequency_hz=0.0022, phase=0.3),
    ResponseBurstCase("response_wnb_01", "signal_like", "white_noise_burst", 4.45, 11_000.0, 1.55, -0.40, 7.0e-18, band_low_hz=0.0012, band_high_hz=0.0060, seed=1101),
    ResponseBurstCase("response_sg_02", "signal_like", "sine_gaussian", 9.25, 6_500.0, 3.10, 0.12, 8.0e-18, frequency_hz=0.0075, phase=1.4),
    ResponseBurstCase("response_wnb_02", "signal_like", "white_noise_burst", 10.95, 8_000.0, 4.60, -0.62, 7.5e-18, band_low_hz=0.0030, band_high_hz=0.0120, seed=1102),
)

GLITCH_CASES = (
    base.GlitchCase("response_glitch_12_day6p85", "readout_isi_carrier_12", 6.85, 5_500.0, 2.0e-4, 2301),
    base.GlitchCase("response_glitch_31_day12p35", "readout_isi_carrier_31", 12.35, 2_500.0, 3.0e-4, 2302),
)


def main() -> None:
    reset_dirs()
    configure_base()
    grid = reconstruct_checkpoint_grid()
    noise = load_checkpoint_noise(grid)

    source_products = []
    catalog_rows = []
    for case in SIGNAL_CASES:
        print(f"response source {case.case_id}", flush=True)
        product, metadata = generate_response_burst_product(case, grid)
        source_products.append(product)
        catalog_rows.append(metadata)

    glitch_products = []
    for case in GLITCH_CASES:
        print(f"glitch source {case.case_id}", flush=True)
        product, window = base.generate_glitch_product(case, grid)
        glitch_products.append(product)
        catalog_rows.append(
            {
                "case_id": case.case_id,
                "case_type": "glitch_like",
                "morphology": "lisaglitch_rectangle",
                "center_day": case.inj_day,
                "duration_s": case.width_s,
                "ra": np.nan,
                "dec": np.nan,
                "strain_amp": np.nan,
                "frequency_hz": np.nan,
                "band_low_hz": np.nan,
                "band_high_hz": np.nan,
                "injection_start_day": window["injection_start_rel_s"] / DAY,
                "injection_stop_day": window["injection_stop_rel_s"] / DAY,
                **window,
            }
        )

    source_sum = sum_products([*source_products, *glitch_products], grid)
    observed = base.add_products(noise, source_sum)
    catalog = pd.DataFrame(catalog_rows)
    save_dataset(grid, noise, source_sum, observed, catalog)

    maps = compute_self_ae_median_wdm(
        observed.t_lisa,
        observed.aet["A2"],
        observed.aet["E2"],
        resolution=RESOLUTION,
    )
    t_map = compute_single_channel_normalized_wdm(observed.t_lisa, observed.aet["T2"])
    zeta_maps = compute_zeta_wdm(noise, observed)

    config = SeededChirpEventConfig(
        low_hz=LOW_HZ,
        high_hz=SEARCH_HIGH_HZ,
        seed_quantile=0.99,
        grow_quantile=0.95,
        max_freq_jump_bins=4,
        same_time_connect_bins=1,
        min_pixels=8,
        min_time_bins=2,
        freq_tolerance_bins=1.0,
        max_events=30,
    )
    events, masks = find_seeded_chirp_events(
        maps,
        config=config,
        start_day=0.5,
        stop_day=13.5,
        label="generic_burst_response_realistic_noise",
    )
    events = annotate_events(events, masks, maps, t_map, zeta_maps, catalog)
    recovery = summarize_recovery(events, catalog)
    save_analysis_outputs(maps, t_map, zeta_maps, masks, catalog, events, recovery)

    aet_plot = plot_aet_maps(maps, t_map, events, catalog)
    zeta_plot = plot_zeta_maps(zeta_maps, events, catalog)
    scatter_plot = plot_event_summary(events)
    report = {
        "method": "SG/WNB signal-like transients are injected as h_plus/h_cross, passed through lisagwresponse.ReadStrain, converted to six links, evaluated by PyTDI, then added to realistic checkpoint noise. Glitches use lisaglitch and the same PyTDI route.",
        "response_path": "hplus/hcross -> ReadStrain.write -> Data.from_gws -> PyTDI Michelson/Sagnac -> AET/zeta WDM",
        "noise_path": "realistic two-week high-rate anti-aliased checkpoint noise reused for AET and Sagnac channels",
        "resolution": asdict(RESOLUTION),
        "event_config": config.__dict__,
        "catalog_csv": str(OUTPUT_DIR / "response_burst_injection_catalog.csv"),
        "events_csv": str(OUTPUT_DIR / "response_burst_candidate_events.csv"),
        "recovery_csv": str(OUTPUT_DIR / "response_burst_recovery_summary.csv"),
        "plots": [str(aet_plot), str(zeta_plot), str(scatter_plot)],
        "zeta_definition": "Event zeta-WDM ratio = sum positive matched-noise-normalized zeta excess / sum positive matched-noise-normalized alpha,beta,gamma excess over the AE event mask.",
    }
    (OUTPUT_DIR / "response_burst_validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_summary(OUTPUT_DIR / "response_burst_validation_summary.md", catalog, recovery, events, report)
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    cols = [
        "case_id",
        "case_type",
        "morphology",
        "center_day",
        "recovered",
        "best_event_rank",
        "best_peak_day",
        "best_peak_freq_hz",
        "best_event_statistic",
        "best_zeta_wdm_ratio",
        "best_t_over_ae",
    ]
    print(recovery[cols].to_string(index=False))
    print(f"saved outputs under {OUTPUT_DIR}")


def reset_dirs() -> None:
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    for path in (INPUT_DIR, OUTPUT_DIR, PLOT_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_base() -> None:
    base.RUN_DIR = RUN_DIR
    base.INPUT_DIR = INPUT_DIR
    base.OUTPUT_DIR = OUTPUT_DIR
    base.PLOT_DIR = PLOT_DIR
    base.TMP_DIR = TMP_DIR
    base.SCIENCE_DURATION = SCIENCE_DURATION
    base.DT = DT
    base.BURN_IN_SECONDS = BURN_IN_SECONDS
    base.ORBIT_DT = ORBIT_DT
    base.CENTRAL_FREQ_HZ = CENTRAL_FREQ_HZ
    base.LOW_HZ = LOW_HZ
    base.HIGH_HZ = HIGH_HZ
    base.INTERPOLATION_ORDER = INTERPOLATION_ORDER
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def reconstruct_checkpoint_grid() -> base.RunGrid:
    data = np.load(CHECKPOINT_TDI)
    t_science = np.asarray(data["t_lisa"], dtype=float)
    n_science = len(t_science)
    burn_in_size = int(round(BURN_IN_SECONDS / DT))
    n_total = n_science + burn_in_size
    t0_lisa = float(t_science[0] - burn_in_size * DT)
    t_lisa_full = t0_lisa + np.arange(n_total, dtype=float) * DT
    t_waveform_full = (np.arange(n_total, dtype=float) - burn_in_size) * DT
    is_science = np.arange(n_total) >= burn_in_size
    if not CHECKPOINT_ORBITS.exists():
        raise FileNotFoundError(CHECKPOINT_ORBITS)
    return base.RunGrid(
        n_science=n_science,
        burn_in_size=burn_in_size,
        n_total=n_total,
        t_lisa_full=t_lisa_full,
        t_lisa_science=t_lisa_full[is_science],
        t_waveform_full=t_waveform_full,
        is_science=is_science,
        t0_lisa=t0_lisa,
        orbits_h5=CHECKPOINT_ORBITS,
    )


def load_checkpoint_noise(grid: base.RunGrid) -> base.TDIProduct:
    tdi = np.load(CHECKPOINT_TDI)
    if len(tdi["t_lisa"]) != grid.n_science:
        raise ValueError("Checkpoint AET noise length does not match reconstructed grid.")
    week1 = np.load(CHECKPOINT_WEEK1_SAGNAC)
    week2 = np.load(CHECKPOINT_WEEK2_SAGNAC)
    sagnac = {}
    for channel in SAGNAC_CHANNELS:
        sagnac[channel] = np.concatenate(
            [np.asarray(week1[f"noise_{channel}"], dtype=float), np.asarray(week2[f"noise_{channel}"], dtype=float)]
        )
        if len(sagnac[channel]) != grid.n_science:
            raise ValueError(f"Checkpoint Sagnac noise length mismatch for {channel}.")
    return base.TDIProduct(
        t_lisa=grid.t_lisa_science.copy(),
        aet={
            "A2": np.asarray(tdi["noise_A2"], dtype=float),
            "E2": np.asarray(tdi["noise_E2"], dtype=float),
            "T2": np.asarray(tdi["noise_T2"], dtype=float),
        },
        sagnac=sagnac,
    )


def generate_response_burst_product(case: ResponseBurstCase, grid: base.RunGrid) -> tuple[base.TDIProduct, dict[str, object]]:
    hplus, hcross = make_science_waveform(case, grid)
    hplus_full = np.zeros(grid.n_total, dtype=float)
    hcross_full = np.zeros(grid.n_total, dtype=float)
    hplus_full[grid.burn_in_size :] = hplus
    hcross_full[grid.burn_in_size :] = hcross
    gws_h5 = TMP_DIR / f"{case.case_id}_gws.h5"
    gws_h5.unlink(missing_ok=True)
    source = ReadStrain(
        grid.t_lisa_full,
        hplus_full,
        hcross_full,
        orbits=str(grid.orbits_h5),
        ra=case.ra,
        dec=case.dec,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        source.write(str(gws_h5), t0=float(grid.t_lisa_full[0]), dt=DT, size=grid.n_total)
    data = Data.from_gws(
        str(gws_h5),
        str(grid.orbits_h5),
        skipped=0,
        gw_dataset="tcb",
        orbit_dataset="tcb/ltt",
    )
    product = base.product_from_data(data, grid, multiply_by_central_frequency=True)
    gws_h5.unlink(missing_ok=True)
    response_window = base.signal_window_from_product(product, threshold_fraction=1.0e-3)
    half = 0.5 * case.duration_s
    metadata = {
        **asdict(case),
        "injection_start_rel_s": case.center_day * DAY - half,
        "injection_stop_rel_s": case.center_day * DAY + half,
        "injection_peak_rel_s": case.center_day * DAY,
        "injection_start_day": case.center_day - half / DAY,
        "injection_stop_day": case.center_day + half / DAY,
        **response_window,
        "response_A2_absmax_hz": float(np.max(np.abs(product.aet["A2"]))),
        "response_E2_absmax_hz": float(np.max(np.abs(product.aet["E2"]))),
        "response_T2_absmax_hz": float(np.max(np.abs(product.aet["T2"]))),
    }
    return product, metadata


def make_science_waveform(case: ResponseBurstCase, grid: base.RunGrid) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(grid.n_science, dtype=float) * DT
    center = case.center_day * DAY
    sigma = case.duration_s / 6.0
    envelope = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    if case.morphology == "sine_gaussian":
        if case.frequency_hz is None:
            raise ValueError(f"{case.case_id} missing frequency.")
        phase = 2.0 * np.pi * case.frequency_hz * (t - center) + case.phase
        hp = envelope * np.sin(phase)
        hc = 0.8 * envelope * np.cos(phase + 0.4)
    elif case.morphology == "white_noise_burst":
        if case.band_low_hz is None or case.band_high_hz is None:
            raise ValueError(f"{case.case_id} missing WNB band.")
        rng = np.random.default_rng(case.seed)
        sos = butter(5, [case.band_low_hz, case.band_high_hz], btype="bandpass", fs=1.0 / DT, output="sos")
        hp = envelope * sosfiltfilt(sos, rng.normal(size=len(t)))
        hc = 0.8 * envelope * sosfiltfilt(sos, rng.normal(size=len(t)))
    else:
        raise ValueError(f"Unsupported morphology {case.morphology!r}.")
    amp = np.sqrt(hp**2 + hc**2)
    active = envelope > np.exp(-0.5 * 3.0**2)
    scale = float(np.max(amp[active])) if np.any(active) else float(np.max(amp))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Could not normalize waveform for {case.case_id}.")
    return case.strain_amp * hp / scale, case.strain_amp * hc / scale


def sum_products(products: list[base.TDIProduct], grid: base.RunGrid) -> base.TDIProduct:
    aet = {channel: np.zeros(grid.n_science, dtype=float) for channel in AET_CHANNELS}
    sagnac = {channel: np.zeros(grid.n_science, dtype=float) for channel in SAGNAC_CHANNELS}
    for product in products:
        for channel in AET_CHANNELS:
            aet[channel] += product.aet[channel]
        for channel in SAGNAC_CHANNELS:
            sagnac[channel] += product.sagnac[channel]
    return base.TDIProduct(t_lisa=grid.t_lisa_science.copy(), aet=aet, sagnac=sagnac)


def save_dataset(
    grid: base.RunGrid,
    noise: base.TDIProduct,
    source_sum: base.TDIProduct,
    observed: base.TDIProduct,
    catalog: pd.DataFrame,
) -> None:
    arrays = {
        "t_lisa": grid.t_lisa_science,
        "time_day": (grid.t_lisa_science - grid.t_lisa_science[0]) / DAY,
    }
    for channel in AET_CHANNELS:
        short = channel[0]
        arrays[f"noise_{channel}"] = noise.aet[channel]
        arrays[f"source_sum_{channel}"] = source_sum.aet[channel]
        arrays[f"observed_{channel}"] = observed.aet[channel]
        arrays[f"noise_{short}"] = noise.aet[channel]
        arrays[f"source_sum_{short}"] = source_sum.aet[channel]
        arrays[f"observed_{short}"] = observed.aet[channel]
    for channel in SAGNAC_CHANNELS:
        arrays[f"noise_{channel}"] = noise.sagnac[channel]
        arrays[f"source_sum_{channel}"] = source_sum.sagnac[channel]
        arrays[f"observed_{channel}"] = observed.sagnac[channel]
    np.savez_compressed(OUTPUT_DIR / "response_burst_dataset.npz", **arrays)
    catalog.to_csv(OUTPUT_DIR / "response_burst_injection_catalog.csv", index=False)


def compute_single_channel_normalized_wdm(t_lisa: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    _, WDM = require_wdm_dependencies()
    wdm = WDM(M=RESOLUTION.M, K=32, beta_order=6, precision=6, backend="numba")
    dt = float(np.median(np.diff(t_lisa)))
    coeff, times, freqs, tf, _ = compute_wdm_transform(
        wdm,
        np.asarray(x, dtype=float),
        sample_rate=1.0 / dt,
        t0=float(t_lisa[0]),
        original_t1=float(t_lisa[-1]),
    )
    power = np.abs(coeff) ** 2
    floor = clean_floor(np.nanmedian(power, axis=1))
    return {
        "times_day": (times - float(t_lisa[0])) / DAY,
        "freqs_hz": freqs,
        "power": power / floor[:, None],
        "wdm_df_hz": np.asarray(float(tf.df)),
        "wdm_dt_s": np.asarray(float(tf.dt)),
    }


def compute_zeta_wdm(noise: base.TDIProduct, observed: base.TDIProduct) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    _, WDM = require_wdm_dependencies()
    t_lisa = observed.t_lisa
    dt = float(np.median(np.diff(t_lisa)))
    wdm = WDM(M=RESOLUTION.M, K=32, beta_order=6, precision=6, backend="numba")
    excess = {}
    times_ref = None
    freqs_ref = None
    for channel in SAGNAC_CHANNELS:
        noise_coeff, times, freqs, _, _ = compute_wdm_transform(
            wdm,
            noise.sagnac[channel],
            sample_rate=1.0 / dt,
            t0=float(t_lisa[0]),
            original_t1=float(t_lisa[-1]),
        )
        obs_coeff, obs_times, obs_freqs, _, _ = compute_wdm_transform(
            wdm,
            observed.sagnac[channel],
            sample_rate=1.0 / dt,
            t0=float(t_lisa[0]),
            original_t1=float(t_lisa[-1]),
        )
        if times_ref is None:
            times_ref = times
            freqs_ref = freqs
        elif not np.array_equal(times_ref, times) or not np.array_equal(freqs_ref, freqs):
            raise ValueError(f"WDM grid mismatch for {channel}.")
        if not np.array_equal(obs_times, times) or not np.array_equal(obs_freqs, freqs):
            raise ValueError(f"Noise/observed WDM grid mismatch for {channel}.")
        noise_power = np.abs(noise_coeff) ** 2
        obs_power = np.abs(obs_coeff) ** 2
        floor = clean_floor(np.nanmedian(noise_power, axis=1))
        excess[channel] = np.maximum((obs_power - noise_power) / floor[:, None], 0.0)
    abc_excess = sum(np.asarray(excess[channel], dtype=float) for channel in ABC_CHANNELS)
    zeta_excess = sum(np.asarray(excess[channel], dtype=float) for channel in ZETA_CHANNELS)
    return {
        "times_day": (np.asarray(times_ref, dtype=float) - float(t_lisa[0])) / DAY,
        "freqs_hz": np.asarray(freqs_ref, dtype=float),
        "excess": excess,
        "abc_positive_excess": abc_excess,
        "zeta_positive_excess": zeta_excess,
        "zeta_fraction": zeta_excess / np.maximum(abc_excess + zeta_excess, 1.0e-300),
    }


def clean_floor(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values) & (values > 0.0), values, 1.0e-300)


def annotate_events(
    events: pd.DataFrame,
    masks: dict[str, np.ndarray],
    maps: dict[str, np.ndarray],
    t_map: dict[str, np.ndarray],
    zeta_maps: dict[str, np.ndarray | dict[str, np.ndarray]],
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return events
    out = events.copy()
    out["matched_case_id"] = ""
    out["matched_case_type"] = ""
    out["matched_morphology"] = ""
    out["matched_center_day"] = np.nan
    out["matched_time_error_days"] = np.nan
    out["t_norm_power_sum"] = np.nan
    out["t_over_ae"] = np.nan
    out["zeta_wdm_abc_excess_sum"] = np.nan
    out["zeta_wdm_zeta_excess_sum"] = np.nan
    out["zeta_wdm_ratio"] = np.nan
    out["zeta_wdm_fraction"] = np.nan

    t_power = np.asarray(t_map["power"], dtype=float)
    abc = np.asarray(zeta_maps["abc_positive_excess"], dtype=float)
    zeta = np.asarray(zeta_maps["zeta_positive_excess"], dtype=float)
    for index, row in out.iterrows():
        mask = masks[str(row["event_id"])]
        fi, ti = np.nonzero(mask)
        t_sum = float(np.nansum(t_power[fi, ti]))
        abc_sum = float(np.nansum(abc[fi, ti]))
        zeta_sum = float(np.nansum(zeta[fi, ti]))
        out.loc[index, "t_norm_power_sum"] = t_sum
        out.loc[index, "t_over_ae"] = t_sum / float(row["ae_total_power"]) if float(row["ae_total_power"]) > 0.0 else np.nan
        out.loc[index, "zeta_wdm_abc_excess_sum"] = abc_sum
        out.loc[index, "zeta_wdm_zeta_excess_sum"] = zeta_sum
        out.loc[index, "zeta_wdm_ratio"] = zeta_sum / abc_sum if abc_sum > 0.0 else np.nan
        out.loc[index, "zeta_wdm_fraction"] = zeta_sum / (abc_sum + zeta_sum) if abc_sum + zeta_sum > 0.0 else np.nan
        match = match_case(float(row["start_day"]), float(row["stop_day"]), float(row["peak_time_day"]), catalog)
        if match is not None:
            out.loc[index, "matched_case_id"] = str(match.case_id)
            out.loc[index, "matched_case_type"] = str(match.case_type)
            out.loc[index, "matched_morphology"] = str(match.morphology)
            out.loc[index, "matched_center_day"] = float(match.center_day)
            out.loc[index, "matched_time_error_days"] = float(row["peak_time_day"]) - float(match.center_day)
    return out.sort_values("event_statistic", ascending=False).reset_index(drop=True)


def match_case(start_day: float, stop_day: float, peak_day: float, catalog: pd.DataFrame) -> pd.Series | None:
    expanded = catalog.copy()
    expanded["match_start_day"] = expanded["injection_start_day"] - 0.20
    expanded["match_stop_day"] = expanded["injection_stop_day"] + 0.20
    overlapping = expanded[(expanded["match_start_day"] <= stop_day) & (expanded["match_stop_day"] >= start_day)]
    if overlapping.empty:
        nearby = expanded[np.abs(expanded["center_day"] - peak_day) <= 0.30]
        if nearby.empty:
            return None
        overlapping = nearby
    return overlapping.loc[(overlapping["center_day"] - peak_day).abs().idxmin()]


def summarize_recovery(events: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, case in catalog.iterrows():
        matched = events[events["matched_case_id"] == case["case_id"]].copy()
        base_row = case.to_dict()
        if matched.empty:
            rows.append(
                {
                    **base_row,
                    "recovered": False,
                    "best_event_rank": np.nan,
                    "best_peak_day": np.nan,
                    "best_peak_freq_hz": np.nan,
                    "best_event_statistic": np.nan,
                    "best_zeta_wdm_ratio": np.nan,
                    "best_t_over_ae": np.nan,
                }
            )
            continue
        best = matched.sort_values("event_statistic", ascending=False).iloc[0]
        rows.append(
            {
                **base_row,
                "recovered": True,
                "best_event_rank": int(best["rank"]),
                "best_peak_day": float(best["peak_time_day"]),
                "best_peak_freq_hz": float(best["peak_freq_hz"]),
                "best_event_statistic": float(best["event_statistic"]),
                "best_zeta_wdm_ratio": float(best["zeta_wdm_ratio"]),
                "best_zeta_wdm_fraction": float(best["zeta_wdm_fraction"]),
                "best_t_over_ae": float(best["t_over_ae"]),
                "best_ae_complex_coherence": float(best["ae_complex_coherence"]),
                "best_monotonic_fraction": float(best["monotonic_fraction"]),
            }
        )
    return pd.DataFrame(rows)


def save_analysis_outputs(
    maps: dict[str, np.ndarray],
    t_map: dict[str, np.ndarray],
    zeta_maps: dict[str, np.ndarray | dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
    catalog: pd.DataFrame,
    events: pd.DataFrame,
    recovery: pd.DataFrame,
) -> None:
    events.to_csv(OUTPUT_DIR / "response_burst_candidate_events.csv", index=False)
    recovery.to_csv(OUTPUT_DIR / "response_burst_recovery_summary.csv", index=False)
    catalog.to_csv(OUTPUT_DIR / "response_burst_injection_catalog.csv", index=False)
    np.savez_compressed(
        OUTPUT_DIR / "response_burst_wdm_maps.npz",
        times_day=maps["times_day"],
        freqs_hz=maps["freqs_hz"],
        A_power=maps["A_power"],
        E_power=maps["E_power"],
        AE_power=maps["AE_power"],
        T_power=t_map["power"],
        abc_positive_excess=zeta_maps["abc_positive_excess"],
        zeta_positive_excess=zeta_maps["zeta_positive_excess"],
        zeta_fraction=zeta_maps["zeta_fraction"],
    )
    np.savez_compressed(OUTPUT_DIR / "response_burst_event_masks.npz", **masks)


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def plot_aet_maps(
    maps: dict[str, np.ndarray],
    t_map: dict[str, np.ndarray],
    events: pd.DataFrame,
    catalog: pd.DataFrame,
) -> Path:
    path = PLOT_DIR / "response_burst_aet_wdm_summary.png"
    times = np.asarray(maps["times_day"], dtype=float)
    freqs = np.asarray(maps["freqs_hz"], dtype=float)
    band = (freqs >= LOW_HZ) & (freqs <= SEARCH_HIGH_HZ)
    images = [
        np.log10(np.maximum(np.asarray(maps["A_power"])[band], 1.0e-300)),
        np.log10(np.maximum(np.asarray(maps["E_power"])[band], 1.0e-300)),
        np.log10(np.maximum(np.asarray(t_map["power"])[band], 1.0e-300)),
        np.log10(np.maximum(np.asarray(maps["AE_power"])[band], 1.0e-300)),
    ]
    finite = np.concatenate([image[np.isfinite(image)] for image in images])
    vmin, vmax = np.nanpercentile(finite, [2.0, 99.6])
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 10.0), sharex=True, sharey=True, constrained_layout=True)
    for ax, image, label in zip(axes, images, ("A", "E", "T", "A+E")):
        im = ax.pcolormesh(
            centers_to_edges(times),
            centers_to_edges(freqs[band]),
            image,
            shading="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        draw_case_markers(ax, catalog)
        for _, event in events.head(12).iterrows():
            ax.scatter(float(event["peak_time_day"]), float(event["peak_freq_hz"]), s=32, facecolor="white", edgecolor="black", linewidth=0.5, zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(LOW_HZ, SEARCH_HIGH_HZ)
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(f"{label} self-normalized WDM")
        ax.grid(True, which="both", alpha=0.12)
    axes[-1].set_xlabel("time [days]")
    fig.colorbar(im, ax=axes, label=r"$\log_{10}$ normalized WDM power")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_zeta_maps(
    zeta_maps: dict[str, np.ndarray | dict[str, np.ndarray]],
    events: pd.DataFrame,
    catalog: pd.DataFrame,
) -> Path:
    path = PLOT_DIR / "response_burst_zeta_wdm_summary.png"
    times = np.asarray(zeta_maps["times_day"], dtype=float)
    freqs = np.asarray(zeta_maps["freqs_hz"], dtype=float)
    band = (freqs >= LOW_HZ) & (freqs <= SEARCH_HIGH_HZ)
    abc = np.asarray(zeta_maps["abc_positive_excess"], dtype=float)
    zeta = np.asarray(zeta_maps["zeta_positive_excess"], dtype=float)
    fraction = np.asarray(zeta_maps["zeta_fraction"], dtype=float)
    panels = [
        ("ABC positive excess", np.log10(1.0 + np.maximum(abc[band], 0.0)), "magma", None),
        ("zeta positive excess", np.log10(1.0 + np.maximum(zeta[band], 0.0)), "magma", None),
        ("zeta fraction", fraction[band], "viridis", (0.0, 1.0)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.4), sharex=True, sharey=True, constrained_layout=True)
    for ax, (title, image, cmap, fixed) in zip(axes, panels):
        if fixed is None:
            finite = image[np.isfinite(image)]
            vmin, vmax = 0.0, float(np.nanpercentile(finite, 99.6))
        else:
            vmin, vmax = fixed
        im = ax.pcolormesh(
            centers_to_edges(times),
            centers_to_edges(freqs[band]),
            image,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        draw_case_markers(ax, catalog)
        for _, event in events.head(12).iterrows():
            ax.scatter(float(event["peak_time_day"]), float(event["peak_freq_hz"]), s=28, facecolor="white", edgecolor="black", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_ylim(LOW_HZ, SEARCH_HIGH_HZ)
        ax.set_ylabel("frequency [Hz]")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.12)
        fig.colorbar(im, ax=ax, pad=0.01, label="fraction" if title == "zeta fraction" else r"$\log_{10}(1 + \mathrm{excess})$")
    axes[-1].set_xlabel("time [days]")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_case_markers(ax, catalog: pd.DataFrame) -> None:
    for _, case in catalog.iterrows():
        color = "#00e5ff" if case["case_type"] == "signal_like" else "#ffea00"
        ax.axvspan(float(case["injection_start_day"]), float(case["injection_stop_day"]), color=color, alpha=0.12, lw=0)
        ax.axvline(float(case["center_day"]), color=color, lw=0.8, alpha=0.9)


def plot_event_summary(events: pd.DataFrame) -> Path:
    path = PLOT_DIR / "response_burst_detection_vs_zeta.png"
    if events.empty:
        return path
    top = events[events["matched_case_id"].astype(str) != ""].copy()
    if top.empty:
        top = events.head(20).copy()
    colors = top["matched_case_type"].map({"signal_like": "#0072B2", "glitch_like": "#D55E00", "": "#666666"}).fillna("#666666")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    axes[0].scatter(top["event_statistic"], top["zeta_wdm_ratio"], c=colors, s=54, edgecolor="black", linewidth=0.5)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("AE event statistic")
    axes[0].set_ylabel("zeta-WDM ratio")
    axes[0].grid(True, which="both", alpha=0.2)
    axes[1].scatter(top["peak_time_day"], top["zeta_wdm_ratio"], c=colors, s=54, edgecolor="black", linewidth=0.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time [days]")
    axes[1].set_ylabel("zeta-WDM ratio")
    axes[1].grid(True, which="both", alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_summary(
    path: Path,
    catalog: pd.DataFrame,
    recovery: pd.DataFrame,
    events: pd.DataFrame,
    report: dict[str, object],
) -> None:
    lines = [
        "# Generic Burst Response Validation",
        "",
        "Signal-like SG/WNB injections are generated as sky-dependent `h_plus,h_cross` strains and passed through the same LISA response and PyTDI path used for MBHB injections.",
        "",
        "Injected cases:",
        "```text",
        catalog[["case_id", "case_type", "morphology", "center_day", "ra", "dec", "strain_amp", "frequency_hz", "band_low_hz", "band_high_hz"]].to_string(index=False),
        "```",
        "",
        "Recovery summary:",
        "```text",
        recovery[
            [
                "case_id",
                "case_type",
                "morphology",
                "center_day",
                "recovered",
                "best_event_rank",
                "best_peak_day",
                "best_peak_freq_hz",
                "best_event_statistic",
                "best_zeta_wdm_ratio",
                "best_t_over_ae",
            ]
        ].to_string(index=False),
        "```",
        "",
        "Top events:",
        "```text",
        events[events["matched_case_id"].astype(str) != ""].head(12)[
            [
                "rank",
                "matched_case_id",
                "matched_case_type",
                "peak_time_day",
                "peak_freq_hz",
                "event_statistic",
                "ae_total_power",
                "ae_complex_coherence",
                "t_over_ae",
                "zeta_wdm_ratio",
            ]
        ].to_string(index=False),
        "```",
        "",
        "Outputs:",
        f"- `{report['catalog_csv']}`",
        f"- `{report['events_csv']}`",
        f"- `{report['recovery_csv']}`",
    ]
    for plot in report["plots"]:
        lines.append(f"- `{plot}`")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
