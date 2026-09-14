from __future__ import annotations

import gc
import json
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lisaglitch import RectangleGlitch
from lisagwresponse import ReadStrain
from lisainstrument import Instrument
from lisaorbits import OEMOrbits
from pytdi import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cluster.parameters import component_masses
from cluster.waveforms import _construct_phenom_model, _unpack_polarizations
from diagnostics.analyze_noise_only_ae_cluster_background import append_combined_rows, coherence
from diagnostics.compute_j_sqrt_energy_detection_statistic import statistic, tail_fraction
from diagnostics.run_mbhb_aet_chirp_track_morphology import (
    monotonic_chirp_path,
    robust_frequency_score,
    tube_from_path,
)
from scripts.clean_wdm import WDMResolution, compute_wdm_transform, downsample_to_target_dt, require_wdm_dependencies
from scripts.six_link_generation import LINKS, LisaLinkSimulationConfig, read_orbit_time_range
from scripts.tdi_computation import (
    _evaluate_michelson_tdi,
    _evaluate_sagnac_tdi,
    apply_frequency_cut,
)


RUN_DIR = PROJECT_ROOT / "diagnostics" / "broad_injection_recovery_3mo"
INPUT_DIR = RUN_DIR / "inputs"
OUTPUT_DIR = RUN_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
TMP_DIR = RUN_DIR / "tmp"

DAY = 86400.0
SCIENCE_DURATION = 90.0 * DAY
DT = 4.0
BURN_IN_SECONDS = 600.0
ORBIT_DT = 600.0
ORBIT_MARGIN_SECONDS = 2.0 * DAY
CENTRAL_FREQ_HZ = 2.816e14
LOW_HZ = 1.0e-5
HIGH_HZ = 1.0e-1
SPLIT_HZ = 2.44140625e-4
INTERPOLATION_ORDER = 7
RAW_WAVEFORM_HALF_DURATION = 3.0e6
RAW_WAVEFORM_DT = 4.0
DISTANCE_MPC = 6230.0

NOISE_SEEDS = (730001, 730002, 730003, 730004, 730005, 730006, 730007, 730008, 730009, 730010)
NOISE_SHIFTS_DAYS = (0.0, 8.5, 17.0, 25.5, 34.0, 42.5, 51.0, 59.5, 68.0, 76.5)
RESOLUTIONS = {
    "low": WDMResolution("dt4_M32768", 4.0, 32768),
    "high": WDMResolution("dt4_M512", 4.0, 512),
}
PART_SETTINGS = {
    "low": {
        "low_hz": LOW_HZ,
        "high_hz": SPLIT_HZ,
        "smooth_time_bins": 1,
        "max_jump_bins": 2,
        "tube_width_bins": 0,
    },
    "high": {
        "low_hz": SPLIT_HZ,
        "high_hz": HIGH_HZ,
        "smooth_time_bins": 3,
        "max_jump_bins": 10,
        "tube_width_bins": 2,
    },
}
ABC_CHANNELS = ("alpha2", "beta2", "gamma2")
ZETA_CHANNELS = ("zeta21", "zeta22", "zeta23")
SAGNAC_CHANNELS = (*ABC_CHANNELS, *ZETA_CHANNELS)


@dataclass(frozen=True)
class MBHBCase:
    case_id: str
    total_mass: float
    q: float
    ra: float
    dec: float
    peak_day: float
    iota: float
    phi_ref: float
    s1: float
    s2: float
    noise_seed: int

    @property
    def eta(self) -> float:
        return float(self.q / (1.0 + self.q) ** 2)


@dataclass(frozen=True)
class GlitchCase:
    case_id: str
    injection_point: str
    inj_day: float
    width_s: float
    level_hz: float
    noise_seed: int


@dataclass
class TDIProduct:
    t_lisa: np.ndarray
    aet: dict[str, np.ndarray]
    sagnac: dict[str, np.ndarray]


@dataclass(frozen=True)
class RunGrid:
    n_science: int
    burn_in_size: int
    n_total: int
    t_lisa_full: np.ndarray
    t_lisa_science: np.ndarray
    t_waveform_full: np.ndarray
    is_science: np.ndarray
    t0_lisa: float
    orbits_h5: Path


MBHB_CASES = (
    MBHBCase("mbhb_m1e5_q1_sky0", 1.0e5, 1.0, 0.45, 0.35, 40.0, 0.35, 0.2, 0.1, -0.1, NOISE_SEEDS[0]),
    MBHBCase("mbhb_m3e5_q06_sky1", 3.0e5, 0.6, 1.75, -0.45, 48.0, 0.75, 1.1, -0.2, 0.2, NOISE_SEEDS[1]),
    MBHBCase("mbhb_m1e6_q03_sky2", 1.0e6, 0.3, 3.10, 0.10, 56.0, 1.05, 2.4, 0.3, -0.3, NOISE_SEEDS[2]),
    MBHBCase("mbhb_m3e6_q02_sky3", 3.0e6, 0.2, 4.45, -0.75, 64.0, 1.25, 3.5, -0.1, 0.4, NOISE_SEEDS[3]),
    MBHBCase("mbhb_m1e7_q01_sky4", 1.0e7, 0.1, 5.70, 0.70, 72.0, 0.95, 4.7, 0.0, 0.0, NOISE_SEEDS[4]),
)

GLITCH_CASES = (
    GlitchCase("glitch_12_day16", "readout_isi_carrier_12", 16.0, 1400.0, 1.8e-4, NOISE_SEEDS[5]),
    GlitchCase("glitch_23_day29", "readout_isi_carrier_23", 29.0, 2600.0, 1.2e-4, NOISE_SEEDS[6]),
    GlitchCase("glitch_31_day43", "readout_isi_carrier_31", 43.0, 1000.0, 3.0e-4, NOISE_SEEDS[7]),
    GlitchCase("glitch_13_day67", "readout_isi_carrier_13", 67.0, 3200.0, 1.0e-4, NOISE_SEEDS[8]),
    GlitchCase("glitch_32_day82", "readout_isi_carrier_32", 82.0, 1800.0, 2.2e-4, NOISE_SEEDS[9]),
)


def main() -> None:
    started = time.time()
    make_dirs()
    config = write_run_config()
    resample_poly, WDM = require_wdm_dependencies()
    grid = prepare_grid()

    noise_detection_rows: list[dict[str, object]] = []
    noise_tubes: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    floor_samples: dict[str, dict[str, list[np.ndarray]]] = {
        part: {channel: [] for channel in SAGNAC_CHANNELS} for part in RESOLUTIONS
    }
    grid_refs: dict[str, dict[str, np.ndarray]] = {}

    log(f"base noise seed {NOISE_SEEDS[0]}: generate 3-month realistic LISA noise")
    base_noise = generate_noise_product(NOISE_SEEDS[0], grid)
    collect_sagnac_floor_samples(base_noise, floor_samples, grid_refs, resample_poly=resample_poly, WDM=WDM)

    for seed in NOISE_SEEDS:
        log(f"noise seed {seed}: shifted noise-only analysis")
        product = shifted_noise_product(base_noise, grid, seed)
        products, rows = analyze_aet_product(product, seed=seed, resample_poly=resample_poly, WDM=WDM)
        noise_tubes[seed] = tubes_from_products(products)
        for row in rows:
            row["case_id"] = f"noise_seed{seed}"
            row["case_type"] = "noise_only"
        noise_detection_rows.extend(rows)
        del product, products
        gc.collect()

    sagnac_floors = aggregate_floor_samples(floor_samples)
    floor_npz = OUTPUT_DIR / "sagnac_noise_floor_vectors.npz"
    save_floor_npz(floor_npz, sagnac_floors, grid_refs)

    noise_detection = append_case_combined_rows(noise_detection_rows)
    noise_detection["detection_statistic"] = statistic(noise_detection["jaccard"], noise_detection["ae_total_power"])

    zeta_rows = []
    for seed in NOISE_SEEDS:
        log(f"noise seed {seed}: zeta-WDM")
        product = shifted_noise_product(base_noise, grid, seed)
        row = summarize_zeta_product(
            f"noise_seed{seed}",
            "noise_only",
            product,
            noise_tubes[seed],
            sagnac_floors,
            resample_poly=resample_poly,
            WDM=WDM,
            matched_noise=None,
        )
        zeta_rows.append(row)
        del product
        gc.collect()

    noise_zeta = pd.DataFrame(zeta_rows)
    case_rows = []
    injection_catalog_rows = []

    for case in MBHB_CASES:
        log(f"{case.case_id}: generate MBHB and recover")
        noise = shifted_noise_product(base_noise, grid, case.noise_seed)
        gw, signal_window = generate_mbhb_gw_product(case, grid)
        observed = add_products(noise, gw)
        products, rows = analyze_aet_product(observed, seed=case.noise_seed, resample_poly=resample_poly, WDM=WDM)
        tubes = tubes_from_products(products)
        det = case_detection_row(rows, case.case_id, "mbhb_plus_noise", signal_window, products)
        zeta = summarize_zeta_product(
            case.case_id,
            "mbhb_plus_noise",
            observed,
            tubes,
            sagnac_floors,
            resample_poly=resample_poly,
            WDM=WDM,
            matched_noise=noise,
        )
        row = {**case_metadata(case), **det, **zeta}
        case_rows.append(row)
        injection_catalog_rows.append({**case_metadata(case), **signal_window})
        del noise, gw, observed, products
        gc.collect()

    for case in GLITCH_CASES:
        log(f"{case.case_id}: generate glitch and recover")
        noise = shifted_noise_product(base_noise, grid, case.noise_seed)
        glitch, signal_window = generate_glitch_product(case, grid)
        observed = add_products(noise, glitch)
        products, rows = analyze_aet_product(observed, seed=case.noise_seed, resample_poly=resample_poly, WDM=WDM)
        tubes = tubes_from_products(products)
        det = case_detection_row(rows, case.case_id, "glitch_plus_noise", signal_window, products)
        zeta = summarize_zeta_product(
            case.case_id,
            "glitch_plus_noise",
            observed,
            tubes,
            sagnac_floors,
            resample_poly=resample_poly,
            WDM=WDM,
            matched_noise=noise,
        )
        row = {**case_metadata(case), **det, **zeta}
        case_rows.append(row)
        injection_catalog_rows.append({**case_metadata(case), **signal_window})
        del noise, glitch, observed, products
        gc.collect()

    cases = pd.DataFrame(case_rows)
    background = build_background_summary(noise_detection, noise_zeta)
    add_empirical_tail_columns(cases, noise_detection, noise_zeta)

    noise_detection.to_csv(OUTPUT_DIR / "noise_only_detection_statistics.csv", index=False)
    noise_zeta.to_csv(OUTPUT_DIR / "noise_only_zeta_wdm_statistics.csv", index=False)
    cases.to_csv(OUTPUT_DIR / "injection_recovery_summary.csv", index=False)
    pd.DataFrame(injection_catalog_rows).to_csv(OUTPUT_DIR / "injection_catalog.csv", index=False)
    pd.DataFrame(background).to_csv(OUTPUT_DIR / "empirical_noise_floor_summary.csv", index=False)
    plot_summary(noise_detection, noise_zeta, cases)

    del base_noise
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    report = {
        "run_dir": str(RUN_DIR),
        "elapsed_seconds": time.time() - started,
        "config": config,
        "outputs": {
            "noise_detection": str(OUTPUT_DIR / "noise_only_detection_statistics.csv"),
            "noise_zeta_wdm": str(OUTPUT_DIR / "noise_only_zeta_wdm_statistics.csv"),
            "injection_recovery_summary": str(OUTPUT_DIR / "injection_recovery_summary.csv"),
            "empirical_noise_floor_summary": str(OUTPUT_DIR / "empirical_noise_floor_summary.csv"),
            "plot": str(PLOT_DIR / "broad_injection_recovery_statistics.png"),
        },
    }
    (OUTPUT_DIR / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def make_dirs() -> None:
    for path in (INPUT_DIR, OUTPUT_DIR, PLOT_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_run_config() -> dict[str, object]:
    config = {
        "science_duration_s": SCIENCE_DURATION,
        "science_duration_days": SCIENCE_DURATION / DAY,
        "dt_s": DT,
        "burn_in_seconds": BURN_IN_SECONDS,
        "orbit_dt_s": ORBIT_DT,
        "orbit_margin_seconds": ORBIT_MARGIN_SECONDS,
        "central_freq_hz": CENTRAL_FREQ_HZ,
        "frequency_low_hz": LOW_HZ,
        "frequency_high_hz": HIGH_HZ,
        "tdi_generation": 2,
        "interpolation_order": INTERPOLATION_ORDER,
        "pytdi_zeta_channels": list(ZETA_CHANNELS),
        "wdm_resolutions": {part: asdict(resolution) for part, resolution in RESOLUTIONS.items()},
        "split_hz": SPLIT_HZ,
        "noise_seeds": list(NOISE_SEEDS),
        "noise_shift_days": list(NOISE_SHIFTS_DAYS),
        "noise_background_method": "one realistic 3-month lisainstrument/PyTDI noise realization, analyzed at ten circular time shifts",
        "distance_mpc": DISTANCE_MPC,
        "raw_waveform_half_duration_s": RAW_WAVEFORM_HALF_DURATION,
        "raw_waveform_dt_s": RAW_WAVEFORM_DT,
        "mbhb_cases": [asdict(case) | {"eta": case.eta} for case in MBHB_CASES],
        "glitch_cases": [asdict(case) for case in GLITCH_CASES],
        "storage_policy": "raw HDF5/time-series caches are temporary; final outputs are summaries, config, floors, and plots.",
    }
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    pd.DataFrame(config["mbhb_cases"]).to_csv(INPUT_DIR / "mbhb_catalog.csv", index=False)
    pd.DataFrame(config["glitch_cases"]).to_csv(INPUT_DIR / "glitch_catalog.csv", index=False)
    return config


def prepare_grid() -> RunGrid:
    n_science = int(round(SCIENCE_DURATION / DT))
    burn_in_size = int(round(BURN_IN_SECONDS / DT))
    n_total = n_science + burn_in_size
    orbit_size = int(np.ceil((n_total * DT + 2.0 * ORBIT_MARGIN_SECONDS) / ORBIT_DT)) + 1
    orbits_h5 = INPUT_DIR / "orbits_3mo.h5"
    if orbits_h5.exists():
        orbits_h5.unlink()
    orbits = OEMOrbits.from_included("esa-trailing", version="latest")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orbits.write(str(orbits_h5), t0="init", dt=ORBIT_DT, size=orbit_size, mode="x")
    orbit = read_orbit_time_range(orbits_h5)
    t0_lisa = orbit.t0 + ORBIT_MARGIN_SECONDS
    t_lisa_full = t0_lisa + np.arange(n_total, dtype=float) * DT
    if t_lisa_full[-1] > orbit.t1:
        raise ValueError("The 3-month time grid does not fit inside the generated orbit file.")
    t_waveform_full = (np.arange(n_total, dtype=float) - burn_in_size) * DT
    is_science = np.arange(n_total) >= burn_in_size
    return RunGrid(
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


def tdi_config() -> LisaLinkSimulationConfig:
    return LisaLinkSimulationConfig(
        dt=DT,
        science_duration=SCIENCE_DURATION,
        burn_in_seconds=BURN_IN_SECONDS,
        central_freq_hz=CENTRAL_FREQ_HZ,
        orbit_dt=ORBIT_DT,
        frequency_low_hz=LOW_HZ,
        frequency_high_hz=HIGH_HZ,
        tdi_generation=2,
        interpolation_order=INTERPOLATION_ORDER,
        tdi_unit="frequency",
        orbit_dataset="tcb/ltt",
        gw_dataset="tcb",
    )


def crop_and_filter(
    arrays: list[np.ndarray],
    grid: RunGrid,
    *,
    multiply_by_central_frequency: bool,
) -> list[np.ndarray]:
    start = grid.burn_in_size
    stop = start + grid.n_science
    out = []
    for arr in arrays:
        values = np.asarray(arr[: grid.n_total], dtype=float)[start:stop]
        values = apply_frequency_cut(values, DT, LOW_HZ, HIGH_HZ)
        if multiply_by_central_frequency:
            values = CENTRAL_FREQ_HZ * values
        out.append(values)
    return out


def generate_noise_product(seed: int, grid: RunGrid) -> TDIProduct:
    h5_path = TMP_DIR / f"noise_seed{seed}.h5"
    h5_path.unlink(missing_ok=True)
    instrument = Instrument(
        size=grid.n_total,
        dt=DT,
        t0=grid.t0_lisa,
        orbits=str(grid.orbits_h5),
        gws=None,
        orbit_dataset="tcb/ltt",
        central_freq=CENTRAL_FREQ_HZ,
        seed=seed,
        noises_f_min_hz=LOW_HZ,
        clock_f_min_hz=LOW_HZ,
        chunk_gw_files=True,
        aafilter=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        instrument.export_hdf5(path=str(h5_path), overwrite=True)
    data = Data.from_instrument(str(h5_path), signals="fluctuations", skipped=0)
    product = product_from_data(data, grid, multiply_by_central_frequency=False)
    h5_path.unlink(missing_ok=True)
    return product


def generate_glitch_product(case: GlitchCase, grid: RunGrid) -> tuple[TDIProduct, dict[str, float]]:
    glitch_file = TMP_DIR / f"{case.case_id}.h5"
    instrument_h5 = TMP_DIR / f"{case.case_id}_glitch_only_instrument.h5"
    glitch_file.unlink(missing_ok=True)
    instrument_h5.unlink(missing_ok=True)
    margin_s = 200.0
    t_inj_lisa = grid.t_lisa_science[0] + case.inj_day * DAY
    glitch = RectangleGlitch(
        inj_point=case.injection_point,
        t_inj=t_inj_lisa,
        width=case.width_s,
        level=case.level_hz,
    )
    glitch.write(
        glitch_file,
        dt=DT,
        size=int(np.ceil((grid.n_total * DT + 2.0 * margin_s) / DT)) + 1,
        t0=grid.t0_lisa - margin_s,
        mode="w",
    )
    instrument = Instrument(
        size=grid.n_total,
        dt=DT,
        t0=grid.t0_lisa,
        orbits=str(grid.orbits_h5),
        gws=None,
        glitches=str(glitch_file),
        orbit_dataset="tcb/ltt",
        central_freq=CENTRAL_FREQ_HZ,
        seed=case.noise_seed,
        laser_asds=0.0,
        modulation_asds=None,
        clock_asds=0.0,
        backlink_asds=0.0,
        testmass_asds=0.0,
        oms_asds=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        moc_time_correlation_asds=0.0,
        mosa_longitudinal_jitter_asds=0.0,
        sc_angular_jitter_asds=(0.0, 0.0, 0.0),
        mosa_angular_jitter_asds=(0.0, 0.0),
        dws_asds=0.0,
        ranging_asds=0.0,
        noises_f_min_hz=None,
        clock_f_min_hz=LOW_HZ,
        chunk_gw_files=True,
        aafilter=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        instrument.export_hdf5(path=str(instrument_h5), overwrite=True)
    data = Data.from_instrument(str(instrument_h5), signals="fluctuations", skipped=0)
    product = product_from_data(data, grid, multiply_by_central_frequency=False)
    glitch_file.unlink(missing_ok=True)
    instrument_h5.unlink(missing_ok=True)
    window = {
        "injection_start_rel_s": case.inj_day * DAY - 0.5 * case.width_s,
        "injection_stop_rel_s": case.inj_day * DAY + 0.5 * case.width_s,
        "injection_peak_rel_s": case.inj_day * DAY,
        "injection_peak_day": case.inj_day,
    }
    return product, window


def generate_mbhb_gw_product(case: MBHBCase, grid: RunGrid) -> tuple[TDIProduct, dict[str, float]]:
    gws_h5 = TMP_DIR / f"{case.case_id}_gws.h5"
    gws_h5.unlink(missing_ok=True)
    times = np.arange(-RAW_WAVEFORM_HALF_DURATION, RAW_WAVEFORM_HALF_DURATION + RAW_WAVEFORM_DT, RAW_WAVEFORM_DT)
    params = {
        "source_id": case.case_id,
        "eta": case.eta,
        "total_mass": case.total_mass,
        "distance_mpc": DISTANCE_MPC,
        "s1": case.s1,
        "s2": case.s2,
        "f_min_hz": 1.0e-4,
        "iota": case.iota,
        "phi_ref": case.phi_ref,
    }
    model, used_kwargs = _construct_phenom_model(params)
    t_raw, hplus_raw, hcross_raw = _unpack_polarizations(model.compute_polarizations(times), times)
    finite = np.isfinite(t_raw) & np.isfinite(hplus_raw) & np.isfinite(hcross_raw)
    t_raw = t_raw[finite]
    hplus_raw = hplus_raw[finite]
    hcross_raw = hcross_raw[finite]
    amp_raw = np.sqrt(hplus_raw**2 + hcross_raw**2)
    peak_raw = float(t_raw[int(np.argmax(amp_raw))])
    source_t = t_raw - peak_raw
    science_t = np.arange(grid.n_science, dtype=float) * DT
    peak_rel = case.peak_day * DAY
    hp_science = np.interp(science_t - peak_rel, source_t, hplus_raw, left=0.0, right=0.0)
    hc_science = np.interp(science_t - peak_rel, source_t, hcross_raw, left=0.0, right=0.0)
    hplus_full = np.zeros(grid.n_total, dtype=float)
    hcross_full = np.zeros(grid.n_total, dtype=float)
    hplus_full[grid.burn_in_size :] = hp_science
    hcross_full[grid.burn_in_size :] = hc_science
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
    product = product_from_data(data, grid, multiply_by_central_frequency=True)
    gws_h5.unlink(missing_ok=True)
    window = signal_window_from_product(product, threshold_fraction=1.0e-3)
    window.update(
        {
            "injection_peak_rel_s": peak_rel,
            "injection_peak_day": case.peak_day,
            "waveform_constructor_kwargs": json.dumps(used_kwargs),
        }
    )
    return product, window


def product_from_data(data: Data, grid: RunGrid, *, multiply_by_central_frequency: bool) -> TDIProduct:
    config = tdi_config()
    x, y, z, a, e, t, _ = _evaluate_michelson_tdi(data, config)
    alpha_beta_gamma_zeta, _ = _evaluate_sagnac_tdi(data, config)
    aet_arrays = crop_and_filter([a, e, t], grid, multiply_by_central_frequency=multiply_by_central_frequency)
    sagnac_arrays = crop_and_filter(
        [alpha_beta_gamma_zeta[name] for name in SAGNAC_CHANNELS],
        grid,
        multiply_by_central_frequency=multiply_by_central_frequency,
    )
    return TDIProduct(
        t_lisa=grid.t_lisa_science.copy(),
        aet={"A2": aet_arrays[0], "E2": aet_arrays[1], "T2": aet_arrays[2]},
        sagnac={name: arr for name, arr in zip(SAGNAC_CHANNELS, sagnac_arrays)},
    )


def cache_product(product: TDIProduct, label: str) -> Path:
    path = TMP_DIR / f"{label}.npz"
    path.unlink(missing_ok=True)
    np.savez(
        path,
        t_lisa=product.t_lisa,
        A2=product.aet["A2"],
        E2=product.aet["E2"],
        T2=product.aet["T2"],
        **{name: product.sagnac[name] for name in SAGNAC_CHANNELS},
    )
    return path


def load_cached_product(path: Path) -> TDIProduct:
    data = np.load(path)
    return TDIProduct(
        t_lisa=np.asarray(data["t_lisa"], dtype=float),
        aet={name: np.asarray(data[name], dtype=float) for name in ("A2", "E2", "T2")},
        sagnac={name: np.asarray(data[name], dtype=float) for name in SAGNAC_CHANNELS},
    )


def shifted_noise_product(product: TDIProduct, grid: RunGrid, seed: int) -> TDIProduct:
    index = NOISE_SEEDS.index(seed)
    shift_samples = int(round(NOISE_SHIFTS_DAYS[index] * DAY / DT)) % grid.n_science
    if shift_samples == 0:
        return TDIProduct(
            t_lisa=product.t_lisa.copy(),
            aet={name: value.copy() for name, value in product.aet.items()},
            sagnac={name: value.copy() for name, value in product.sagnac.items()},
        )
    return TDIProduct(
        t_lisa=product.t_lisa.copy(),
        aet={name: np.roll(value, shift_samples) for name, value in product.aet.items()},
        sagnac={name: np.roll(value, shift_samples) for name, value in product.sagnac.items()},
    )


def add_products(a: TDIProduct, b: TDIProduct) -> TDIProduct:
    if len(a.t_lisa) != len(b.t_lisa) or not np.allclose(a.t_lisa, b.t_lisa, rtol=0.0, atol=1.0e-6):
        raise ValueError("Cannot add TDI products with different time grids.")
    return TDIProduct(
        t_lisa=a.t_lisa.copy(),
        aet={name: a.aet[name] + b.aet[name] for name in ("A2", "E2", "T2")},
        sagnac={name: a.sagnac[name] + b.sagnac[name] for name in SAGNAC_CHANNELS},
    )


def signal_window_from_product(product: TDIProduct, *, threshold_fraction: float) -> dict[str, float]:
    power = product.aet["A2"] ** 2 + product.aet["E2"] ** 2
    peak_idx = int(np.nanargmax(power))
    peak = float(power[peak_idx])
    if peak <= 0.0:
        return {
            "injection_start_rel_s": np.nan,
            "injection_stop_rel_s": np.nan,
            "response_peak_rel_s": np.nan,
            "response_peak_day": np.nan,
            "response_peak_power": 0.0,
        }
    keep = np.flatnonzero(power >= threshold_fraction * peak)
    rel = product.t_lisa - product.t_lisa[0]
    return {
        "injection_start_rel_s": float(rel[keep[0]]),
        "injection_stop_rel_s": float(rel[keep[-1]]),
        "response_peak_rel_s": float(rel[peak_idx]),
        "response_peak_day": float(rel[peak_idx] / DAY),
        "response_peak_power": peak,
    }


def analyze_aet_product(
    product: TDIProduct,
    *,
    seed: int,
    resample_poly,
    WDM,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    maps = compute_aet_maps(product, resample_poly=resample_poly, WDM=WDM)
    products: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for part in ("low", "high"):
        a = channel_cluster(maps[part], channel="A", part=part)
        e = channel_cluster(maps[part], channel="E", part=part)
        a_tube = np.asarray(a["tube"], dtype=bool)
        e_tube = np.asarray(e["tube"], dtype=bool)
        intersection = a_tube & e_tube
        union = a_tube | e_tube
        c_inter, phase_inter, inter_a_power, inter_e_power = coherence(
            np.asarray(a["coeff"], dtype=np.complex128),
            np.asarray(e["coeff"], dtype=np.complex128),
            intersection,
        )
        c_union, phase_union, union_a_power, union_e_power = coherence(
            np.asarray(a["coeff"], dtype=np.complex128),
            np.asarray(e["coeff"], dtype=np.complex128),
            union,
        )
        products[part] = {"A": a, "E": e, "intersection": intersection, "union": union}
        rows.append(
            {
                "seed": seed,
                "part": part,
                "n_a_tube": int(np.sum(a_tube)),
                "n_e_tube": int(np.sum(e_tube)),
                "n_intersection": int(np.sum(intersection)),
                "n_union": int(np.sum(union)),
                "jaccard": float(np.sum(intersection) / np.sum(union)) if np.any(union) else np.nan,
                "a_sum_score": float(np.sum(np.asarray(a["score"])[a_tube])) if np.any(a_tube) else 0.0,
                "e_sum_score": float(np.sum(np.asarray(e["score"])[e_tube])) if np.any(e_tube) else 0.0,
                "a_sum_power": float(np.sum(np.asarray(a["power"])[a_tube])) if np.any(a_tube) else 0.0,
                "e_sum_power": float(np.sum(np.asarray(e["power"])[e_tube])) if np.any(e_tube) else 0.0,
                "intersection_coherence": c_inter,
                "intersection_phase_rad": phase_inter,
                "intersection_a_power": inter_a_power,
                "intersection_e_power": inter_e_power,
                "union_coherence": c_union,
                "union_phase_rad": phase_union,
                "union_a_power": union_a_power,
                "union_e_power": union_e_power,
            }
        )
    return products, rows


def compute_aet_maps(product: TDIProduct, *, resample_poly, WDM) -> dict[str, dict[str, np.ndarray]]:
    t_lisa = product.t_lisa
    native_dt = float(np.median(np.diff(t_lisa)))
    maps: dict[str, dict[str, np.ndarray]] = {}
    for part, resolution in RESOLUTIONS.items():
        wdm = WDM(M=resolution.M, K=32, beta_order=6, precision=6, backend="numba")
        t_ds, dt_ds, _ = downsample_to_target_dt(product.aet["T2"], native_dt, resolution.target_dt, resample_poly)
        t_coeff, times, freqs, _, _ = compute_wdm_transform(
            wdm,
            t_ds,
            sample_rate=1.0 / dt_ds,
            t0=float(t_lisa[0]),
            original_t1=float(t_lisa[-1]),
        )
        floor = np.nanmedian(np.abs(t_coeff) ** 2, axis=1)
        floor = np.where(np.isfinite(floor) & (floor > 0.0), floor, 1.0e-300)
        maps[part] = {"times_rel": times - t_lisa[0], "freqs_hz": freqs, "floor": floor}
        for channel, key in (("A", "A2"), ("E", "E2")):
            x_ds, x_dt, _ = downsample_to_target_dt(product.aet[key], native_dt, resolution.target_dt, resample_poly)
            coeff, x_times, x_freqs, _, _ = compute_wdm_transform(
                wdm,
                x_ds,
                sample_rate=1.0 / x_dt,
                t0=float(t_lisa[0]),
                original_t1=float(t_lisa[-1]),
            )
            if coeff.shape != t_coeff.shape or not np.array_equal(x_times, times) or not np.array_equal(x_freqs, freqs):
                raise ValueError(f"WDM shape/time/frequency mismatch for {channel} {resolution.label}.")
            whitened = coeff / np.sqrt(floor[:, None])
            maps[part][f"{channel}_coeff"] = whitened
            maps[part][f"{channel}_power"] = np.abs(whitened) ** 2
    return maps


def channel_cluster(products: dict[str, np.ndarray], *, channel: str, part: str) -> dict[str, object]:
    settings = PART_SETTINGS[part]
    freqs = np.asarray(products["freqs_hz"], dtype=float)
    times = np.asarray(products["times_rel"], dtype=float)
    coeff_full = np.asarray(products[f"{channel}_coeff"], dtype=np.complex128)
    power_full = np.asarray(products[f"{channel}_power"], dtype=float)
    band = (freqs >= settings["low_hz"]) & (freqs < settings["high_hz"])
    coeff = coeff_full[band, :]
    power = power_full[band, :]
    freqs_band = freqs[band]
    score = robust_frequency_score(power, z_floor=0.35, smooth_size=(3, int(settings["smooth_time_bins"])))
    score = np.maximum(score, 0.0)
    end_cell = tuple(int(v) for v in np.unravel_index(int(np.nanargmax(power)), power.shape))
    path, path_score, _ = monotonic_chirp_path(
        score,
        max_jump_bins=int(settings["max_jump_bins"]),
        step_penalty=0.03,
        jump_penalty=0.03 if part == "low" else 0.004,
        end_cell=end_cell,
    )
    tube = tube_from_path(path, score.shape, width_bins=int(settings["tube_width_bins"]))
    return {
        "times": times,
        "freqs": freqs_band,
        "coeff": coeff,
        "power": power,
        "score": score,
        "path": path,
        "tube": tube,
        "path_score": float(path_score),
    }


def tubes_from_products(products: dict[str, dict[str, object]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        part: {
            "a_tube": np.asarray(products[part]["A"]["tube"], dtype=bool),
            "e_tube": np.asarray(products[part]["E"]["tube"], dtype=bool),
            "intersection": np.asarray(products[part]["intersection"], dtype=bool),
            "union": np.asarray(products[part]["union"], dtype=bool),
            "times_rel_s": np.asarray(products[part]["A"]["times"], dtype=float),
            "freqs_hz": np.asarray(products[part]["A"]["freqs"], dtype=float),
        }
        for part in ("low", "high")
    }


def append_case_combined_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = append_combined_rows(rows)
    case_meta = pd.DataFrame(rows)[["seed", "case_id", "case_type"]].drop_duplicates("seed")
    df = df.drop(columns=[column for column in ("case_id", "case_type") if column in df.columns])
    df = df.merge(case_meta, on="seed", how="left")
    return df


def case_detection_row(
    rows: list[dict[str, object]],
    case_id: str,
    case_type: str,
    injection_window: dict[str, float],
    products: dict[str, dict[str, object]],
) -> dict[str, object]:
    for row in rows:
        row["case_id"] = case_id
        row["case_type"] = case_type
    summary = append_case_combined_rows(rows)
    combined = summary[summary["part"] == "combined_low_high"].iloc[0]
    d_value = float(statistic(np.asarray([combined["jaccard"]]), np.asarray([combined["ae_total_power"]]))[0])
    extents = tube_extents(products)
    return {
        "case_id": case_id,
        "case_type": case_type,
        "jaccard": float(combined["jaccard"]),
        "union_coherence": float(combined["union_coherence"]),
        "ae_total_power": float(combined["ae_total_power"]),
        "log10_ae_total_power": float(combined["log10_ae_total_power"]),
        "a_power_fraction": float(combined["a_power_fraction"]),
        "detection_statistic": d_value,
        "tube_overlaps_injection": bool(tube_overlaps(extents, injection_window)),
        **extents,
    }


def tube_extents(products: dict[str, dict[str, object]]) -> dict[str, float]:
    times = []
    freqs = []
    for part in ("low", "high"):
        union = np.asarray(products[part]["union"], dtype=bool)
        if not np.any(union):
            continue
        fi, ti = np.nonzero(union)
        part_times = np.asarray(products[part]["A"]["times"], dtype=float)
        part_freqs = np.asarray(products[part]["A"]["freqs"], dtype=float)
        times.extend(part_times[ti].tolist())
        freqs.extend(part_freqs[fi].tolist())
    if not times:
        return {
            "tube_start_rel_s": np.nan,
            "tube_stop_rel_s": np.nan,
            "tube_start_day": np.nan,
            "tube_stop_day": np.nan,
            "tube_freq_min_hz": np.nan,
            "tube_freq_max_hz": np.nan,
        }
    return {
        "tube_start_rel_s": float(np.min(times)),
        "tube_stop_rel_s": float(np.max(times)),
        "tube_start_day": float(np.min(times) / DAY),
        "tube_stop_day": float(np.max(times) / DAY),
        "tube_freq_min_hz": float(np.min(freqs)),
        "tube_freq_max_hz": float(np.max(freqs)),
    }


def tube_overlaps(extents: dict[str, float], injection_window: dict[str, float]) -> bool:
    start = float(injection_window.get("injection_start_rel_s", np.nan))
    stop = float(injection_window.get("injection_stop_rel_s", np.nan))
    tube_start = float(extents.get("tube_start_rel_s", np.nan))
    tube_stop = float(extents.get("tube_stop_rel_s", np.nan))
    return bool(np.isfinite(start) and np.isfinite(stop) and tube_stop >= start and tube_start <= stop)


def collect_sagnac_floor_samples(
    product: TDIProduct,
    floor_samples: dict[str, dict[str, list[np.ndarray]]],
    grid_refs: dict[str, dict[str, np.ndarray]],
    *,
    resample_poly,
    WDM,
) -> None:
    for part, resolution in RESOLUTIONS.items():
        powers, times, freqs = sagnac_wdm_powers(product, resolution, resample_poly=resample_poly, WDM=WDM)
        if part not in grid_refs:
            grid_refs[part] = {"times_rel_s": times - product.t_lisa[0], "freqs_hz": freqs}
        else:
            if not np.array_equal(grid_refs[part]["freqs_hz"], freqs):
                raise ValueError(f"Sagnac floor frequency-grid mismatch for {part}.")
        for channel in SAGNAC_CHANNELS:
            floor = np.nanmedian(powers[channel], axis=1)
            floor_samples[part][channel].append(np.where(np.isfinite(floor) & (floor > 0.0), floor, 1.0e-300))


def aggregate_floor_samples(floor_samples: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    floors: dict[str, dict[str, np.ndarray]] = {}
    for part, channel_samples in floor_samples.items():
        floors[part] = {}
        for channel, samples in channel_samples.items():
            stack = np.vstack(samples)
            floor = np.nanmedian(stack, axis=0)
            floors[part][channel] = np.where(np.isfinite(floor) & (floor > 0.0), floor, 1.0e-300)
    return floors


def save_floor_npz(
    path: Path,
    floors: dict[str, dict[str, np.ndarray]],
    grid_refs: dict[str, dict[str, np.ndarray]],
) -> None:
    arrays = {}
    for part in RESOLUTIONS:
        arrays[f"{part}_freqs_hz"] = grid_refs[part]["freqs_hz"]
        arrays[f"{part}_times_rel_s"] = grid_refs[part]["times_rel_s"]
        for channel in SAGNAC_CHANNELS:
            arrays[f"{part}_{channel}_floor"] = floors[part][channel]
    np.savez(path, **arrays)


def sagnac_wdm_powers(
    product: TDIProduct,
    resolution: WDMResolution,
    *,
    resample_poly,
    WDM,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    t_lisa = product.t_lisa
    native_dt = float(np.median(np.diff(t_lisa)))
    wdm = WDM(M=resolution.M, K=32, beta_order=6, precision=6, backend="numba")
    powers: dict[str, np.ndarray] = {}
    times_ref = None
    freqs_ref = None
    for channel in SAGNAC_CHANNELS:
        x_ds, dt_ds, _ = downsample_to_target_dt(product.sagnac[channel], native_dt, resolution.target_dt, resample_poly)
        coeff, times, freqs, _, _ = compute_wdm_transform(
            wdm,
            x_ds,
            sample_rate=1.0 / dt_ds,
            t0=float(t_lisa[0]),
            original_t1=float(t_lisa[-1]),
        )
        if times_ref is None:
            times_ref = times
            freqs_ref = freqs
        elif not np.array_equal(times_ref, times) or not np.array_equal(freqs_ref, freqs):
            raise ValueError(f"Sagnac WDM-grid mismatch for {channel} {resolution.label}.")
        powers[channel] = np.abs(coeff) ** 2
    return powers, np.asarray(times_ref, dtype=float), np.asarray(freqs_ref, dtype=float)


def normalized_sagnac_powers(
    product: TDIProduct,
    part: str,
    floors: dict[str, dict[str, np.ndarray]],
    *,
    resample_poly,
    WDM,
) -> dict[str, np.ndarray]:
    resolution = RESOLUTIONS[part]
    low = PART_SETTINGS[part]["low_hz"]
    high = PART_SETTINGS[part]["high_hz"]
    raw_powers, times, freqs = sagnac_wdm_powers(product, resolution, resample_poly=resample_poly, WDM=WDM)
    band = (freqs >= low) & (freqs < high)
    out: dict[str, np.ndarray] = {
        "_times_rel_s": times - product.t_lisa[0],
        "_freqs_hz": freqs[band],
    }
    for channel in SAGNAC_CHANNELS:
        out[channel] = raw_powers[channel][band, :] / floors[part][channel][band, None]
    return out


def summarize_zeta_product(
    case_id: str,
    case_type: str,
    product: TDIProduct,
    tubes: dict[str, dict[str, np.ndarray]],
    floors: dict[str, dict[str, np.ndarray]],
    *,
    resample_poly,
    WDM,
    matched_noise: TDIProduct | None,
) -> dict[str, object]:
    raw_rows = []
    excess_rows = []
    for part in ("low", "high"):
        powers = normalized_sagnac_powers(product, part, floors, resample_poly=resample_poly, WDM=WDM)
        raw_rows.append(summarize_zeta_part(powers, tubes[part]))
        if matched_noise is not None:
            noise_powers = normalized_sagnac_powers(matched_noise, part, floors, resample_poly=resample_poly, WDM=WDM)
            excess_rows.append(summarize_zeta_part(positive_excess(powers, noise_powers), tubes[part]))
    raw = combine_zeta_parts(raw_rows)
    row = {
        "case_id": case_id,
        "case_type": case_type,
        "zeta_raw_abc_power": raw["abc_power"],
        "zeta_raw_zeta_power": raw["zeta_power"],
        "zeta_raw_ratio": raw["ratio"],
        "zeta_raw_n_pixels": raw["n_pixels"],
    }
    if excess_rows:
        excess = combine_zeta_parts(excess_rows)
        row.update(
            {
                "zeta_positive_excess_abc_power": excess["abc_power"],
                "zeta_positive_excess_zeta_power": excess["zeta_power"],
                "zeta_positive_excess_ratio": excess["ratio"],
                "zeta_positive_excess_n_pixels": excess["n_pixels"],
            }
        )
    else:
        row.update(
            {
                "zeta_positive_excess_abc_power": np.nan,
                "zeta_positive_excess_zeta_power": np.nan,
                "zeta_positive_excess_ratio": np.nan,
                "zeta_positive_excess_n_pixels": np.nan,
            }
        )
    return row


def summarize_zeta_part(powers: dict[str, np.ndarray], tube: dict[str, np.ndarray]) -> dict[str, float]:
    mask = np.asarray(tube["union"], dtype=bool)
    if not np.any(mask):
        return {"abc_power": 0.0, "zeta_power": 0.0, "n_pixels": 0.0}
    abc_power = float(sum(np.nansum(np.asarray(powers[channel], dtype=float)[mask]) for channel in ABC_CHANNELS))
    zeta_power = float(sum(np.nansum(np.asarray(powers[channel], dtype=float)[mask]) for channel in ZETA_CHANNELS))
    return {"abc_power": abc_power, "zeta_power": zeta_power, "n_pixels": float(np.sum(mask))}


def combine_zeta_parts(rows: list[dict[str, float]]) -> dict[str, float]:
    abc_power = float(sum(row["abc_power"] for row in rows))
    zeta_power = float(sum(row["zeta_power"] for row in rows))
    n_pixels = int(sum(row["n_pixels"] for row in rows))
    if abc_power > 0.0:
        ratio = (zeta_power / len(ZETA_CHANNELS)) / (abc_power / len(ABC_CHANNELS))
    elif zeta_power > 0.0:
        ratio = np.inf
    else:
        ratio = np.nan
    return {"abc_power": abc_power, "zeta_power": zeta_power, "n_pixels": n_pixels, "ratio": float(ratio)}


def positive_excess(powers: dict[str, np.ndarray], noise_powers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {"_times_rel_s": powers["_times_rel_s"], "_freqs_hz": powers["_freqs_hz"]}
    for channel in SAGNAC_CHANNELS:
        out[channel] = np.maximum(
            np.asarray(powers[channel], dtype=float) - np.asarray(noise_powers[channel], dtype=float),
            0.0,
        )
    return out


def case_metadata(case: MBHBCase | GlitchCase) -> dict[str, object]:
    if isinstance(case, MBHBCase):
        m1, m2, q = component_masses(case.total_mass, case.eta)
        return {
            "case_id": case.case_id,
            "case_type": "mbhb_plus_noise",
            "noise_seed": case.noise_seed,
            "total_mass": case.total_mass,
            "m1": m1,
            "m2": m2,
            "q": q,
            "eta": case.eta,
            "distance_mpc": DISTANCE_MPC,
            "ra": case.ra,
            "dec": case.dec,
            "injection_peak_day": case.peak_day,
            "injection_point": "",
            "glitch_width_s": np.nan,
            "glitch_level_hz": np.nan,
        }
    return {
        "case_id": case.case_id,
        "case_type": "glitch_plus_noise",
        "noise_seed": case.noise_seed,
        "total_mass": np.nan,
        "m1": np.nan,
        "m2": np.nan,
        "q": np.nan,
        "eta": np.nan,
        "distance_mpc": np.nan,
        "ra": np.nan,
        "dec": np.nan,
        "injection_peak_day": case.inj_day,
        "injection_point": case.injection_point,
        "glitch_width_s": case.width_s,
        "glitch_level_hz": case.level_hz,
    }


def build_background_summary(noise_detection: pd.DataFrame, noise_zeta: pd.DataFrame) -> list[dict[str, object]]:
    det = noise_detection[noise_detection["part"] == "combined_low_high"]["detection_statistic"].to_numpy(dtype=float)
    zeta = noise_zeta["zeta_raw_ratio"].to_numpy(dtype=float)
    rows = []
    for name, values in (("detection_statistic", det), ("zeta_raw_ratio", zeta)):
        rows.append(
            {
                "statistic": name,
                "n_noise": int(len(values)),
                "min": float(np.nanmin(values)),
                "median": float(np.nanmedian(values)),
                "max": float(np.nanmax(values)),
                "p90": float(np.nanpercentile(values, 90.0)),
                "p95": float(np.nanpercentile(values, 95.0)),
                "p99": float(np.nanpercentile(values, 99.0)),
            }
        )
    return rows


def add_empirical_tail_columns(
    cases: pd.DataFrame,
    noise_detection: pd.DataFrame,
    noise_zeta: pd.DataFrame,
) -> None:
    noise_d = noise_detection[noise_detection["part"] == "combined_low_high"]["detection_statistic"].to_numpy(dtype=float)
    noise_r = noise_zeta["zeta_raw_ratio"].to_numpy(dtype=float)
    cases["detection_noise_tail_fraction"] = [tail_fraction(v, noise_d) for v in cases["detection_statistic"]]
    cases["detection_over_noise_median"] = cases["detection_statistic"] / float(np.nanmedian(noise_d))
    cases["detection_above_noise_max"] = cases["detection_statistic"] > float(np.nanmax(noise_d))
    cases["zeta_raw_high_tail_fraction"] = [tail_fraction(v, noise_r) for v in cases["zeta_raw_ratio"]]
    cases["zeta_raw_over_noise_median"] = cases["zeta_raw_ratio"] / float(np.nanmedian(noise_r))
    cases["zeta_raw_above_noise_max"] = cases["zeta_raw_ratio"] > float(np.nanmax(noise_r))


def plot_summary(noise_detection: pd.DataFrame, noise_zeta: pd.DataFrame, cases: pd.DataFrame) -> Path:
    output = PLOT_DIR / "broad_injection_recovery_statistics.png"
    noise_d = noise_detection[noise_detection["part"] == "combined_low_high"]["detection_statistic"].to_numpy(dtype=float)
    noise_r = noise_zeta["zeta_raw_ratio"].to_numpy(dtype=float)
    x = np.arange(len(cases))
    colors = cases["case_type"].map({"mbhb_plus_noise": "#2563eb", "glitch_plus_noise": "#111827"}).to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.axhspan(float(np.nanmin(noise_d)), float(np.nanmax(noise_d)), color="0.75", alpha=0.35, label="10 noise-only min-max")
    ax.axhline(float(np.nanmedian(noise_d)), color="0.35", ls="--", lw=1.0, label="noise median")
    ax.scatter(x, cases["detection_statistic"], c=colors, s=54)
    ax.set_yscale("log")
    ax.set_ylabel(r"$D = J\sqrt{E_A+E_E}$")
    ax.set_xticks(x)
    ax.set_xticklabels(cases["case_id"], rotation=45, ha="right", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("checkpoint detection statistic")

    ax = axes[1]
    ax.axhspan(float(np.nanmin(noise_r)), float(np.nanmax(noise_r)), color="0.75", alpha=0.35, label="10 noise-only min-max")
    ax.axhline(float(np.nanmedian(noise_r)), color="0.35", ls="--", lw=1.0, label="noise median")
    ax.scatter(x, cases["zeta_raw_ratio"], c=colors, s=54)
    ax.set_ylabel(r"$R_{\mathrm{zeta-WDM}}$")
    ax.set_xticks(x)
    ax.set_xticklabels(cases["case_id"], rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_title("raw zeta-WDM ratio")

    ax = axes[2]
    ax.scatter(cases["detection_statistic"], cases["zeta_raw_ratio"], c=colors, s=60)
    ax.axvspan(float(np.nanmin(noise_d)), float(np.nanmax(noise_d)), color="0.75", alpha=0.20)
    ax.axhspan(float(np.nanmin(noise_r)), float(np.nanmax(noise_r)), color="0.75", alpha=0.20)
    ax.set_xscale("log")
    ax.set_xlabel(r"$D = J\sqrt{E_A+E_E}$")
    ax.set_ylabel(r"$R_{\mathrm{zeta-WDM}}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("detection vs glitch veto")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563eb", label="MBHB + noise", markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#111827", label="glitch + noise", markersize=7),
        plt.Line2D([0], [0], color="0.55", lw=6, alpha=0.35, label="noise-only range"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("3-month broad injection-recovery checkpoint study", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


if __name__ == "__main__":
    main()
