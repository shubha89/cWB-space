from __future__ import annotations

import json
import shutil
import sys
import time
from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import diagnostics.run_broad_injection_recovery_3mo as base
from diagnostics.compute_j_sqrt_energy_detection_statistic import statistic, tail_fraction
from scripts.clean_wdm import require_wdm_dependencies


RUN_DIR = PROJECT_ROOT / "diagnostics" / "multilink_glitch_detection_rzeta_study"
INPUT_DIR = RUN_DIR / "inputs"
OUTPUT_DIR = RUN_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
TMP_DIR = RUN_DIR / "tmp"

TWO_WEEK_RUN_DIR = PROJECT_ROOT / "diagnostics" / "simulated_2week_mbhb_then_glitch_highrate"
TWO_WEEK_OUTPUT_DIR = TWO_WEEK_RUN_DIR / "outputs"
PRODUCTS_NPZ = TWO_WEEK_OUTPUT_DIR / "two_week_mbhb_then_glitch_tdi.npz"
MBHB_SAGNAC_NPZ = TWO_WEEK_OUTPUT_DIR / "mbhb_week1_sagnac_zeta_products.npz"
GLITCH_SAGNAC_NPZ = TWO_WEEK_OUTPUT_DIR / "glitch_week2_sagnac_zeta_products.npz"
MBHB_CATALOG_CSV = TWO_WEEK_OUTPUT_DIR / "mbhb_catalog.csv"
ORBIT_H5 = TWO_WEEK_RUN_DIR / "inputs" / "orbits_2week.h5"

LOCKED_BROAD_BENCHMARK_CSV = PROJECT_ROOT / "diagnostics" / "broad_injection_recovery_3mo" / "outputs" / "injection_recovery_summary.csv"
LOCKED_RZETA_METRICS_JSON = (
    PROJECT_ROOT / "diagnostics" / "lisaglitch_noisy_kappa" / "outputs" / "zeta_wdm_broad_g2_glitch_vs_5mbhb_metrics.json"
)

DAY = 86_400.0
WEEK_DURATION = 7.0 * DAY
WEEK2_START_DAY = 7.0
TARGET_DT = 4.0
SOURCE_BURN_IN_SECONDS = 600.0
ORBIT_DT = 600.0
CENTRAL_FREQ_HZ = 2.816e14
LOW_HZ = 1.0e-5
HIGH_HZ = 1.0e-1
NOISE_SEED = 752001

AET_CHANNELS = ("A2", "E2", "T2")
ABC_CHANNELS = base.ABC_CHANNELS
ZETA_CHANNELS = base.ZETA_CHANNELS
SAGNAC_CHANNELS = base.SAGNAC_CHANNELS


@dataclass(frozen=True)
class GlitchComponent:
    injection_point: str
    local_peak_day: float
    width_s: float
    level_hz: float

    @property
    def full_peak_day(self) -> float:
        return WEEK2_START_DAY + self.local_peak_day


@dataclass(frozen=True)
class GlitchEventCase:
    case_id: str
    components: tuple[GlitchComponent, ...]
    noise_seed: int = NOISE_SEED

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def glitch_family(self) -> str:
        return "single_link" if self.component_count == 1 else "multi_link"


GLITCH_CASES = (
    GlitchEventCase(
        "single_12_narrow_high_day8p2",
        (GlitchComponent("readout_isi_carrier_12", 1.2, 600.0, 3.5e-4),),
    ),
    GlitchEventCase(
        "single_23_wide_low_day9p1",
        (GlitchComponent("readout_isi_carrier_23", 2.1, 4_000.0, 8.0e-5),),
    ),
    GlitchEventCase(
        "single_31_mid_day10p3",
        (GlitchComponent("readout_isi_carrier_31", 3.3, 1_400.0, 1.8e-4),),
    ),
    GlitchEventCase(
        "single_13_high_day11p2",
        (GlitchComponent("readout_isi_carrier_13", 4.2, 1_000.0, 4.0e-4),),
    ),
    GlitchEventCase(
        "single_21_low_day12p4",
        (GlitchComponent("readout_isi_carrier_21", 5.4, 3_000.0, 1.0e-4),),
    ),
    GlitchEventCase(
        "single_32_mid_day13p0",
        (GlitchComponent("readout_isi_carrier_32", 6.0, 1_800.0, 2.2e-4),),
    ),
    GlitchEventCase(
        "pair_reciprocal_12_21_day8p7",
        (
            GlitchComponent("readout_isi_carrier_12", 1.7, 1_500.0, 1.5e-4),
            GlitchComponent("readout_isi_carrier_21", 1.7, 1_500.0, 1.5e-4),
        ),
    ),
    GlitchEventCase(
        "pair_adjacent_12_23_stagger_day9p8",
        (
            GlitchComponent("readout_isi_carrier_12", 2.8, 1_400.0, 1.8e-4),
            GlitchComponent("readout_isi_carrier_23", 2.8 + 180.0 / DAY, 1_400.0, 1.4e-4),
        ),
    ),
    GlitchEventCase(
        "pair_cross_13_31_stagger_day10p6",
        (
            GlitchComponent("readout_isi_carrier_13", 3.6 - 240.0 / DAY, 900.0, 2.5e-4),
            GlitchComponent("readout_isi_carrier_31", 3.6 + 240.0 / DAY, 900.0, 2.5e-4),
        ),
    ),
    GlitchEventCase(
        "triplet_clockwise_12_23_31_day11p4",
        (
            GlitchComponent("readout_isi_carrier_12", 4.4, 1_600.0, 1.4e-4),
            GlitchComponent("readout_isi_carrier_23", 4.4, 1_600.0, 1.4e-4),
            GlitchComponent("readout_isi_carrier_31", 4.4, 1_600.0, 1.4e-4),
        ),
    ),
    GlitchEventCase(
        "triplet_counter_13_32_21_day12p1",
        (
            GlitchComponent("readout_isi_carrier_13", 5.1, 2_000.0, 1.2e-4),
            GlitchComponent("readout_isi_carrier_32", 5.1, 2_000.0, 1.2e-4),
            GlitchComponent("readout_isi_carrier_21", 5.1, 2_000.0, 1.2e-4),
        ),
    ),
    GlitchEventCase(
        "sixlink_common_day13p3",
        tuple(
            GlitchComponent(f"readout_isi_carrier_{link}", 6.3, 1_800.0, 7.0e-5)
            for link in ("12", "13", "21", "23", "31", "32")
        ),
    ),
)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Compare A/E detection statistic and G2 R_zeta-WDM for a diverse single- "
            "and multi-link lisaglitch ensemble against the existing MBHB benchmark."
        )
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Run only the first N glitch cases.")
    parser.add_argument("--case-id", action="append", default=None, help="Run only matching glitch case IDs; can be repeated.")
    parser.add_argument("--reuse-source-cache", action="store_true", help="Reuse cached source-only glitch products when available.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write config without running WDM/TDI analysis.")
    parser.add_argument("--force", action="store_true", help="Remove previous outputs for this diagnostic before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.force:
        shutil.rmtree(RUN_DIR, ignore_errors=True)
    make_dirs()
    configure_base()
    cases = selected_cases(args)
    validate_inputs()
    write_config(cases)
    if args.dry_run:
        print(OUTPUT_DIR / "run_config.json")
        return

    resample_poly, WDM = require_wdm_dependencies()
    products = np.load(PRODUCTS_NPZ)
    mbhb_sagnac = np.load(MBHB_SAGNAC_NPZ)
    glitch_sagnac = np.load(GLITCH_SAGNAC_NPZ)

    week1_noise = load_week_product(products, mbhb_sagnac, week_index=0, prefix="noise")
    week2_noise = load_week_product(products, glitch_sagnac, week_index=1, prefix="noise")
    grid = make_week2_grid(week2_noise.t_lisa)
    sagnac_floors = build_sagnac_floors((week1_noise, week2_noise), resample_poly=resample_poly, WDM=WDM)
    matched_noise_powers = {
        "week1": precompute_noise_sagnac_powers(week1_noise, sagnac_floors, resample_poly=resample_poly, WDM=WDM),
        "week2": precompute_noise_sagnac_powers(week2_noise, sagnac_floors, resample_poly=resample_poly, WDM=WDM),
    }

    noise_rows = analyze_noise_background((("week1_noise", week1_noise), ("week2_noise", week2_noise)), sagnac_floors, resample_poly, WDM)
    benchmark_rows = analyze_mbhb_benchmark(
        products,
        mbhb_sagnac,
        week1_noise,
        matched_noise_powers["week1"],
        sagnac_floors,
        resample_poly,
        WDM,
    )
    glitch_rows = analyze_glitch_cases(
        cases,
        week2_noise,
        grid,
        matched_noise_powers["week2"],
        sagnac_floors,
        resample_poly,
        WDM,
        reuse_source_cache=args.reuse_source_cache,
    )

    summary = pd.DataFrame([*noise_rows, *benchmark_rows, *glitch_rows])
    add_comparison_columns(summary)
    summary_csv = OUTPUT_DIR / "multilink_glitch_detection_rzeta_summary.csv"
    summary.to_csv(summary_csv, index=False)

    metrics = build_metrics(summary, started)
    metrics_json = OUTPUT_DIR / "multilink_glitch_detection_rzeta_metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    plot_path = plot_summary(summary, metrics)
    summary_md = write_summary(metrics, summary_csv, metrics_json, plot_path)
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    print(summary_csv)
    print(metrics_json)
    print(plot_path)
    print(summary_md)
    print(
        summary[summary["case_type"].isin(["mbhb_plus_noise", "glitch_plus_noise"])][
            [
                "case_id",
                "case_type",
                "glitch_family",
                "component_count",
                "detection_statistic",
                "zeta_positive_excess_ratio",
                "rzeta_above_internal_mbhb_max",
                "rzeta_above_locked_broad_simple_threshold",
            ]
        ].to_string(index=False)
    )


def make_dirs() -> None:
    for path in (INPUT_DIR, OUTPUT_DIR, PLOT_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_base() -> None:
    base.RUN_DIR = RUN_DIR
    base.INPUT_DIR = INPUT_DIR
    base.OUTPUT_DIR = OUTPUT_DIR
    base.PLOT_DIR = PLOT_DIR
    base.TMP_DIR = TMP_DIR
    base.SCIENCE_DURATION = WEEK_DURATION
    base.DT = TARGET_DT
    base.BURN_IN_SECONDS = SOURCE_BURN_IN_SECONDS
    base.ORBIT_DT = ORBIT_DT
    base.LOW_HZ = LOW_HZ
    base.HIGH_HZ = HIGH_HZ
    base.CENTRAL_FREQ_HZ = CENTRAL_FREQ_HZ


def selected_cases(args: Namespace) -> tuple[GlitchEventCase, ...]:
    cases = GLITCH_CASES
    if args.case_id:
        wanted = set(args.case_id)
        cases = tuple(case for case in cases if case.case_id in wanted)
        missing = sorted(wanted.difference(case.case_id for case in cases))
        if missing:
            raise ValueError(f"Unknown glitch case IDs: {missing}")
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError("No glitch cases selected.")
    return cases


def validate_inputs() -> None:
    required = (PRODUCTS_NPZ, MBHB_SAGNAC_NPZ, GLITCH_SAGNAC_NPZ, MBHB_CATALOG_CSV, ORBIT_H5)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(path) for path in missing))


def write_config(cases: tuple[GlitchEventCase, ...]) -> None:
    config = {
        "science_duration_days": WEEK_DURATION / DAY,
        "week1_role": "existing MBHB benchmark segment",
        "week2_role": "existing noise segment plus newly generated source-only glitches",
        "target_dt_s": TARGET_DT,
        "frequency_low_hz": LOW_HZ,
        "frequency_high_hz": HIGH_HZ,
        "tdi_generation": 2,
        "central_freq_hz": CENTRAL_FREQ_HZ,
        "source_burn_in_seconds": SOURCE_BURN_IN_SECONDS,
        "inputs": {
            "two_week_aet_products": str(PRODUCTS_NPZ),
            "mbhb_week1_sagnac_products": str(MBHB_SAGNAC_NPZ),
            "glitch_week2_sagnac_noise_products": str(GLITCH_SAGNAC_NPZ),
            "mbhb_catalog": str(MBHB_CATALOG_CSV),
            "orbits_h5": str(ORBIT_H5),
        },
        "detection_statistic": "D = J * sqrt(E_A + E_E), using the current A/E WDM tube code.",
        "rzeta_statistic": (
            "G2 R_zeta-WDM = per-channel positive excess in zeta21/zeta22/zeta23 divided by "
            "per-channel positive excess in alpha2/beta2/gamma2 inside the A/E union tube."
        ),
        "linearity_assumption": (
            "Multi-link source-only glitch products are built as sums of independent single-link lisaglitch "
            "RectangleGlitch TDI responses. This uses linearity of the link-to-TDI response for additive readout glitches."
        ),
        "glitch_cases": [
            {
                "case_id": case.case_id,
                "component_count": case.component_count,
                "glitch_family": case.glitch_family,
                "components": [asdict(component) | {"full_peak_day": component.full_peak_day} for component in case.components],
            }
            for case in cases
        ],
    }
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")


def load_week_product(
    products: np.lib.npyio.NpzFile,
    sagnac: np.lib.npyio.NpzFile,
    *,
    week_index: int,
    prefix: str,
) -> base.TDIProduct:
    n_week = int(round(WEEK_DURATION / TARGET_DT))
    start = week_index * n_week
    stop = start + n_week
    t_lisa = np.asarray(products["t_lisa"][start:stop], dtype=float)
    if len(t_lisa) != n_week:
        raise ValueError(f"Week {week_index + 1} A/E segment has {len(t_lisa)} samples, expected {n_week}.")
    if len(sagnac["t_lisa"]) != n_week:
        raise ValueError(f"Week {week_index + 1} Sagnac segment has {len(sagnac['t_lisa'])} samples, expected {n_week}.")
    if not np.allclose(t_lisa, np.asarray(sagnac["t_lisa"], dtype=float), rtol=0.0, atol=1.0e-6):
        raise ValueError(f"A/E and Sagnac week {week_index + 1} grids do not match.")
    return base.TDIProduct(
        t_lisa=t_lisa,
        aet={channel: np.asarray(products[f"{prefix}_{channel}"][start:stop], dtype=float) for channel in AET_CHANNELS},
        sagnac={channel: np.asarray(sagnac[f"{prefix}_{channel}"], dtype=float) for channel in SAGNAC_CHANNELS},
    )


def load_week_source(
    products: np.lib.npyio.NpzFile,
    sagnac: np.lib.npyio.NpzFile,
    *,
    week_index: int,
    prefix: str,
) -> base.TDIProduct:
    n_week = int(round(WEEK_DURATION / TARGET_DT))
    start = week_index * n_week
    stop = start + n_week
    t_lisa = np.asarray(products["t_lisa"][start:stop], dtype=float)
    return base.TDIProduct(
        t_lisa=t_lisa,
        aet={channel: np.asarray(products[f"{prefix}_{channel}"][start:stop], dtype=float) for channel in AET_CHANNELS},
        sagnac={channel: np.asarray(sagnac[f"{prefix}_{channel}"], dtype=float) for channel in SAGNAC_CHANNELS},
    )


def make_week2_grid(t_lisa_science: np.ndarray) -> base.RunGrid:
    burn_in_size = int(round(SOURCE_BURN_IN_SECONDS / TARGET_DT))
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


def build_sagnac_floors(
    noise_products: tuple[base.TDIProduct, ...],
    *,
    resample_poly,
    WDM,
) -> dict[str, dict[str, np.ndarray]]:
    floor_samples = {part: {channel: [] for channel in SAGNAC_CHANNELS} for part in base.RESOLUTIONS}
    grid_refs: dict[str, dict[str, np.ndarray]] = {}
    for product in noise_products:
        base.collect_sagnac_floor_samples(product, floor_samples, grid_refs, resample_poly=resample_poly, WDM=WDM)
    floors = base.aggregate_floor_samples(floor_samples)
    save_floor_npz(OUTPUT_DIR / "local_sagnac_noise_floor_vectors.npz", floors, grid_refs)
    return floors


def precompute_noise_sagnac_powers(
    noise: base.TDIProduct,
    sagnac_floors: dict[str, dict[str, np.ndarray]],
    *,
    resample_poly,
    WDM,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        part: base.normalized_sagnac_powers(noise, part, sagnac_floors, resample_poly=resample_poly, WDM=WDM)
        for part in ("low", "high")
    }


def save_floor_npz(
    path: Path,
    floors: dict[str, dict[str, np.ndarray]],
    grid_refs: dict[str, dict[str, np.ndarray]],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for part in base.RESOLUTIONS:
        arrays[f"{part}_freqs_hz"] = np.asarray(grid_refs[part]["freqs_hz"], dtype=float)
        arrays[f"{part}_times_rel_s"] = np.asarray(grid_refs[part]["times_rel_s"], dtype=float)
        for channel in SAGNAC_CHANNELS:
            arrays[f"{part}_{channel}_floor"] = np.asarray(floors[part][channel], dtype=float)
    np.savez_compressed(path, **arrays)


def analyze_noise_background(
    noise_products: tuple[tuple[str, base.TDIProduct], ...],
    sagnac_floors: dict[str, dict[str, np.ndarray]],
    resample_poly,
    WDM,
) -> list[dict[str, object]]:
    rows = []
    for label, product in noise_products:
        products, part_rows = base.analyze_aet_product(product, seed=NOISE_SEED, resample_poly=resample_poly, WDM=WDM)
        detection = base.case_detection_row(
            part_rows,
            label,
            "noise_only",
            {"injection_start_rel_s": np.nan, "injection_stop_rel_s": np.nan},
            products,
        )
        zeta = base.summarize_zeta_product(
            label,
            "noise_only",
            product,
            base.tubes_from_products(products),
            sagnac_floors,
            resample_poly=resample_poly,
            WDM=WDM,
            matched_noise=None,
        )
        rows.append(with_common_metadata({**detection, **zeta}, glitch_family="noise_only", component_count=0))
    return rows


def analyze_mbhb_benchmark(
    products: np.lib.npyio.NpzFile,
    sagnac: np.lib.npyio.NpzFile,
    noise: base.TDIProduct,
    matched_noise_powers: dict[str, dict[str, np.ndarray]],
    sagnac_floors: dict[str, dict[str, np.ndarray]],
    resample_poly,
    WDM,
) -> list[dict[str, object]]:
    catalog = pd.read_csv(MBHB_CATALOG_CSV)
    rows = []
    for index, mbhb in enumerate(catalog.itertuples(index=False), start=1):
        source = load_week_source(products, sagnac, week_index=0, prefix=f"mbhb{index}")
        observed = base.add_products(noise, source)
        products_by_part, part_rows = base.analyze_aet_product(observed, seed=NOISE_SEED, resample_poly=resample_poly, WDM=WDM)
        window = {
            "injection_start_rel_s": float(mbhb.injection_start_rel_s),
            "injection_stop_rel_s": float(mbhb.injection_stop_rel_s),
            "injection_peak_rel_s": float(mbhb.injection_peak_rel_s),
            "injection_peak_day": float(mbhb.injection_peak_day),
        }
        detection = base.case_detection_row(part_rows, mbhb.case_id, "mbhb_plus_noise", window, products_by_part)
        zeta = summarize_zeta_product_cached_noise(
            mbhb.case_id,
            "mbhb_plus_noise",
            observed,
            base.tubes_from_products(products_by_part),
            sagnac_floors,
            matched_noise_powers,
            resample_poly=resample_poly,
            WDM=WDM,
        )
        metadata = {
            "total_mass": float(mbhb.total_mass),
            "q": float(mbhb.q),
            "eta": float(mbhb.eta),
            "distance_mpc": float(mbhb.distance_mpc),
            "ra": float(mbhb.ra),
            "dec": float(mbhb.dec),
            "injection_peak_day": float(mbhb.injection_peak_day),
            "full_injection_peak_day": float(mbhb.injection_peak_day),
        }
        rows.append(with_common_metadata({**metadata, **detection, **zeta}, glitch_family="mbhb_benchmark", component_count=0))
    return rows


def analyze_glitch_cases(
    cases: tuple[GlitchEventCase, ...],
    noise: base.TDIProduct,
    grid: base.RunGrid,
    matched_noise_powers: dict[str, dict[str, np.ndarray]],
    sagnac_floors: dict[str, dict[str, np.ndarray]],
    resample_poly,
    WDM,
    *,
    reuse_source_cache: bool,
) -> list[dict[str, object]]:
    rows = []
    catalog_rows = []
    for case in cases:
        log(f"{case.case_id}: generate source-only multi-link glitch and analyze")
        source, window = generate_multilink_glitch_product(case, grid, reuse_source_cache=reuse_source_cache)
        observed = base.add_products(noise, source)
        products_by_part, part_rows = base.analyze_aet_product(observed, seed=case.noise_seed, resample_poly=resample_poly, WDM=WDM)
        detection = base.case_detection_row(part_rows, case.case_id, "glitch_plus_noise", window, products_by_part)
        zeta = summarize_zeta_product_cached_noise(
            case.case_id,
            "glitch_plus_noise",
            observed,
            base.tubes_from_products(products_by_part),
            sagnac_floors,
            matched_noise_powers,
            resample_poly=resample_poly,
            WDM=WDM,
        )
        metadata = glitch_metadata(case, window)
        catalog_rows.append(metadata)
        rows.append(with_common_metadata({**metadata, **detection, **zeta}, glitch_family=case.glitch_family, component_count=case.component_count))
    pd.DataFrame(catalog_rows).to_csv(OUTPUT_DIR / "multilink_glitch_catalog.csv", index=False)
    return rows


def summarize_zeta_product_cached_noise(
    case_id: str,
    case_type: str,
    product: base.TDIProduct,
    tubes: dict[str, dict[str, np.ndarray]],
    sagnac_floors: dict[str, dict[str, np.ndarray]],
    matched_noise_powers: dict[str, dict[str, np.ndarray]],
    *,
    resample_poly,
    WDM,
) -> dict[str, object]:
    raw_rows = []
    excess_rows = []
    for part in ("low", "high"):
        powers = base.normalized_sagnac_powers(product, part, sagnac_floors, resample_poly=resample_poly, WDM=WDM)
        raw_rows.append(base.summarize_zeta_part(powers, tubes[part]))
        excess_rows.append(base.summarize_zeta_part(base.positive_excess(powers, matched_noise_powers[part]), tubes[part]))
    raw = base.combine_zeta_parts(raw_rows)
    excess = base.combine_zeta_parts(excess_rows)
    return {
        "case_id": case_id,
        "case_type": case_type,
        "zeta_raw_abc_power": raw["abc_power"],
        "zeta_raw_zeta_power": raw["zeta_power"],
        "zeta_raw_ratio": raw["ratio"],
        "zeta_raw_n_pixels": raw["n_pixels"],
        "zeta_positive_excess_abc_power": excess["abc_power"],
        "zeta_positive_excess_zeta_power": excess["zeta_power"],
        "zeta_positive_excess_ratio": excess["ratio"],
        "zeta_positive_excess_n_pixels": excess["n_pixels"],
    }


def generate_multilink_glitch_product(
    case: GlitchEventCase,
    grid: base.RunGrid,
    *,
    reuse_source_cache: bool,
) -> tuple[base.TDIProduct, dict[str, float]]:
    cache = OUTPUT_DIR / "source_cache" / f"{case.case_id}_source_tdi.npz"
    if reuse_source_cache and cache.exists():
        return load_cached_product(cache), combined_window(case)

    products = []
    for index, component in enumerate(case.components, start=1):
        component_case = base.GlitchCase(
            f"{case.case_id}_component{index}",
            component.injection_point,
            component.local_peak_day,
            component.width_s,
            component.level_hz,
            case.noise_seed,
        )
        product, _ = base.generate_glitch_product(component_case, grid)
        products.append(product)
    source = sum_tdi_products(products)
    cache_product(cache, source)
    return source, combined_window(case)


def combined_window(case: GlitchEventCase) -> dict[str, float]:
    starts = [component.local_peak_day * DAY - 0.5 * component.width_s for component in case.components]
    stops = [component.local_peak_day * DAY + 0.5 * component.width_s for component in case.components]
    peak_days = [component.local_peak_day for component in case.components]
    return {
        "injection_start_rel_s": float(min(starts)),
        "injection_stop_rel_s": float(max(stops)),
        "injection_peak_rel_s": float(np.median(peak_days) * DAY),
        "injection_peak_day": float(np.median(peak_days)),
        "full_injection_peak_day": float(WEEK2_START_DAY + np.median(peak_days)),
    }


def sum_tdi_products(products: list[base.TDIProduct]) -> base.TDIProduct:
    if not products:
        raise ValueError("No TDI products to sum.")
    total = products[0]
    for product in products[1:]:
        total = base.add_products(total, product)
    return total


def cache_product(path: Path, product: base.TDIProduct) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t_lisa=product.t_lisa,
        **{channel: product.aet[channel] for channel in AET_CHANNELS},
        **{channel: product.sagnac[channel] for channel in SAGNAC_CHANNELS},
    )


def load_cached_product(path: Path) -> base.TDIProduct:
    data = np.load(path)
    return base.TDIProduct(
        t_lisa=np.asarray(data["t_lisa"], dtype=float),
        aet={channel: np.asarray(data[channel], dtype=float) for channel in AET_CHANNELS},
        sagnac={channel: np.asarray(data[channel], dtype=float) for channel in SAGNAC_CHANNELS},
    )


def glitch_metadata(case: GlitchEventCase, window: dict[str, float]) -> dict[str, object]:
    component_points = [component.injection_point for component in case.components]
    return {
        "case_id": case.case_id,
        "case_type": "glitch_plus_noise",
        "noise_seed": case.noise_seed,
        "glitch_family": case.glitch_family,
        "component_count": case.component_count,
        "injection_points": ";".join(component_points),
        "component_local_peak_days": json.dumps([component.local_peak_day for component in case.components]),
        "component_full_peak_days": json.dumps([component.full_peak_day for component in case.components]),
        "component_widths_s": json.dumps([component.width_s for component in case.components]),
        "component_levels_hz": json.dumps([component.level_hz for component in case.components]),
        "min_width_s": float(min(component.width_s for component in case.components)),
        "max_width_s": float(max(component.width_s for component in case.components)),
        "max_abs_level_hz": float(max(abs(component.level_hz) for component in case.components)),
        **window,
    }


def with_common_metadata(
    row: dict[str, object],
    *,
    glitch_family: str,
    component_count: int,
) -> dict[str, object]:
    out = {
        "case_id": row.get("case_id", ""),
        "case_type": row.get("case_type", ""),
        "noise_seed": row.get("noise_seed", NOISE_SEED),
        "glitch_family": glitch_family,
        "component_count": component_count,
        "injection_points": row.get("injection_points", ""),
        "component_local_peak_days": row.get("component_local_peak_days", ""),
        "component_full_peak_days": row.get("component_full_peak_days", ""),
        "component_widths_s": row.get("component_widths_s", ""),
        "component_levels_hz": row.get("component_levels_hz", ""),
        "min_width_s": row.get("min_width_s", np.nan),
        "max_width_s": row.get("max_width_s", np.nan),
        "max_abs_level_hz": row.get("max_abs_level_hz", np.nan),
        "total_mass": row.get("total_mass", np.nan),
        "q": row.get("q", np.nan),
        "eta": row.get("eta", np.nan),
        "distance_mpc": row.get("distance_mpc", np.nan),
        "ra": row.get("ra", np.nan),
        "dec": row.get("dec", np.nan),
        "injection_peak_day": row.get("injection_peak_day", np.nan),
        "full_injection_peak_day": row.get("full_injection_peak_day", row.get("injection_peak_day", np.nan)),
        "injection_start_rel_s": row.get("injection_start_rel_s", np.nan),
        "injection_stop_rel_s": row.get("injection_stop_rel_s", np.nan),
    }
    out.update(row)
    return out


def add_comparison_columns(summary: pd.DataFrame) -> None:
    noise = summary[summary["case_type"] == "noise_only"]
    mbhb = summary[summary["case_type"] == "mbhb_plus_noise"]
    glitch = summary[summary["case_type"] == "glitch_plus_noise"]
    if mbhb.empty:
        raise ValueError("Cannot add comparison columns without MBHB benchmark rows.")

    mbhb_d = finite_values(mbhb["detection_statistic"])
    mbhb_r = finite_values(mbhb["zeta_positive_excess_ratio"])
    noise_d = finite_values(noise["detection_statistic"])
    noise_r = finite_values(noise["zeta_raw_ratio"])

    summary["detection_over_internal_mbhb_median"] = summary["detection_statistic"] / np.nanmedian(mbhb_d)
    summary["detection_over_internal_mbhb_max"] = summary["detection_statistic"] / np.nanmax(mbhb_d)
    summary["rzeta_over_internal_mbhb_median"] = summary["zeta_positive_excess_ratio"] / np.nanmedian(mbhb_r)
    summary["rzeta_over_internal_mbhb_max"] = summary["zeta_positive_excess_ratio"] / np.nanmax(mbhb_r)
    summary["rzeta_above_internal_mbhb_max"] = summary["zeta_positive_excess_ratio"] > np.nanmax(mbhb_r)

    if len(noise_d):
        summary["detection_noise_tail_fraction_local"] = [tail_fraction(value, noise_d) for value in summary["detection_statistic"]]
        summary["detection_above_local_noise_max"] = summary["detection_statistic"] > np.nanmax(noise_d)
    else:
        summary["detection_noise_tail_fraction_local"] = np.nan
        summary["detection_above_local_noise_max"] = False
    if len(noise_r):
        summary["rzeta_noise_tail_fraction_local"] = [tail_fraction(value, noise_r) for value in summary["zeta_raw_ratio"]]
        summary["rzeta_raw_above_local_noise_max"] = summary["zeta_raw_ratio"] > np.nanmax(noise_r)
    else:
        summary["rzeta_noise_tail_fraction_local"] = np.nan
        summary["rzeta_raw_above_local_noise_max"] = False

    locked_metrics = load_locked_rzeta_metrics()
    if locked_metrics:
        locked_simple = float(locked_metrics.get("simple_threshold_midpoint", np.nan))
        locked_mbhb_max = float(locked_metrics.get("mbhb_max", np.nan))
        summary["rzeta_above_locked_broad_simple_threshold"] = summary["zeta_positive_excess_ratio"] > locked_simple
        summary["rzeta_above_locked_broad_mbhb_max"] = summary["zeta_positive_excess_ratio"] > locked_mbhb_max
    else:
        summary["rzeta_above_locked_broad_simple_threshold"] = False
        summary["rzeta_above_locked_broad_mbhb_max"] = False

    if not glitch.empty:
        summary["is_glitch"] = summary["case_type"] == "glitch_plus_noise"


def finite_values(values: pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def load_locked_rzeta_metrics() -> dict[str, object]:
    if not LOCKED_RZETA_METRICS_JSON.exists():
        return {}
    return json.loads(LOCKED_RZETA_METRICS_JSON.read_text())


def build_metrics(summary: pd.DataFrame, started: float) -> dict[str, object]:
    noise = summary[summary["case_type"] == "noise_only"]
    mbhb = summary[summary["case_type"] == "mbhb_plus_noise"]
    glitch = summary[summary["case_type"] == "glitch_plus_noise"]
    single = glitch[glitch["glitch_family"] == "single_link"]
    multi = glitch[glitch["glitch_family"] == "multi_link"]
    mbhb_d = finite_values(mbhb["detection_statistic"])
    mbhb_r = finite_values(mbhb["zeta_positive_excess_ratio"])
    glitch_d = finite_values(glitch["detection_statistic"])
    glitch_r = finite_values(glitch["zeta_positive_excess_ratio"])
    locked_metrics = load_locked_rzeta_metrics()
    locked_simple = float(locked_metrics.get("simple_threshold_midpoint", np.nan)) if locked_metrics else np.nan
    locked_mbhb_max = float(locked_metrics.get("mbhb_max", np.nan)) if locked_metrics else np.nan

    metrics: dict[str, object] = {
        "elapsed_seconds": time.time() - started,
        "n_noise": int(len(noise)),
        "n_mbhb_benchmark": int(len(mbhb)),
        "n_glitch": int(len(glitch)),
        "n_single_link_glitch": int(len(single)),
        "n_multi_link_glitch": int(len(multi)),
        "detection_statistic": {
            "definition": "D = J * sqrt(E_A + E_E)",
            "noise_range": range_stats(noise["detection_statistic"]),
            "mbhb_range": range_stats(mbhb["detection_statistic"]),
            "glitch_range": range_stats(glitch["detection_statistic"]),
            "single_link_glitch_range": range_stats(single["detection_statistic"]),
            "multi_link_glitch_range": range_stats(multi["detection_statistic"]),
            "fraction_glitches_above_mbhb_max": fraction_above(glitch_d, np.nanmax(mbhb_d)),
        },
        "rzeta_positive_excess_ratio": {
            "definition": "per-channel positive zeta excess / per-channel positive alpha-beta-gamma excess inside A/E union tubes",
            "mbhb_range": range_stats(mbhb["zeta_positive_excess_ratio"]),
            "glitch_range": range_stats(glitch["zeta_positive_excess_ratio"]),
            "single_link_glitch_range": range_stats(single["zeta_positive_excess_ratio"]),
            "multi_link_glitch_range": range_stats(multi["zeta_positive_excess_ratio"]),
            "internal_threshold_mbhb_max": float(np.nanmax(mbhb_r)),
            "internal_threshold_mbhb_p95": float(np.nanpercentile(mbhb_r, 95.0)),
            "fraction_glitches_above_internal_mbhb_max": fraction_above(glitch_r, np.nanmax(mbhb_r)),
            "fraction_single_link_above_internal_mbhb_max": fraction_above(finite_values(single["zeta_positive_excess_ratio"]), np.nanmax(mbhb_r)),
            "fraction_multi_link_above_internal_mbhb_max": fraction_above(finite_values(multi["zeta_positive_excess_ratio"]), np.nanmax(mbhb_r)),
            "locked_broad_simple_threshold": locked_simple,
            "locked_broad_mbhb_max": locked_mbhb_max,
            "fraction_glitches_above_locked_broad_simple_threshold": fraction_above(glitch_r, locked_simple),
            "fraction_glitches_above_locked_broad_mbhb_max": fraction_above(glitch_r, locked_mbhb_max),
        },
        "locked_broad_rzeta_metrics_file": str(LOCKED_RZETA_METRICS_JSON) if LOCKED_RZETA_METRICS_JSON.exists() else "",
        "locked_broad_benchmark_csv": str(LOCKED_BROAD_BENCHMARK_CSV) if LOCKED_BROAD_BENCHMARK_CSV.exists() else "",
    }
    return metrics


def range_stats(values: pd.Series) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    finite = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "n_inf": 0, "min": np.nan, "median": np.nan, "max": np.nan, "p95": np.nan}
    return {
        "n": int(len(array)),
        "n_inf": int(np.sum(np.isinf(array))),
        "min": float(np.nanmin(array)),
        "median": float(np.nanmedian(finite)) if len(finite) else np.inf,
        "max": float(np.nanmax(array)),
        "p95": float(np.nanpercentile(finite, 95.0)) if len(finite) else np.inf,
    }


def fraction_above(values: np.ndarray, threshold: float) -> float:
    valid = values[~np.isnan(values)]
    if not len(valid) or not np.isfinite(threshold):
        return np.nan
    return float(np.mean(valid > threshold))


def plot_summary(summary: pd.DataFrame, metrics: dict[str, object]) -> Path:
    output = PLOT_DIR / "multilink_glitch_detection_rzeta_summary.png"
    plot_rows = summary[summary["case_type"].isin(["mbhb_plus_noise", "glitch_plus_noise"])].copy()
    colors = {
        "mbhb_benchmark": "#2563eb",
        "single_link": "#7c2d12",
        "multi_link": "#111827",
    }
    plot_rows["color"] = plot_rows["glitch_family"].map(colors).fillna("#6b7280")
    x = np.arange(len(plot_rows))
    rzeta_metrics = metrics["rzeta_positive_excess_ratio"]
    internal_threshold = float(rzeta_metrics["internal_threshold_mbhb_max"])
    locked_threshold = float(rzeta_metrics["locked_broad_simple_threshold"])

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)
    ax = axes[0]
    for family, label in (
        ("mbhb_benchmark", "MBHB benchmark"),
        ("single_link", "single-link glitch"),
        ("multi_link", "multi-link glitch"),
    ):
        sub = plot_rows[plot_rows["glitch_family"] == family]
        ax.scatter(
            sub["detection_statistic"],
            sub["zeta_positive_excess_ratio"],
            s=42 + 10 * sub["component_count"].to_numpy(dtype=float),
            color=colors[family],
            label=label,
            alpha=0.88,
        )
    ax.axhline(internal_threshold, color="#2563eb", ls="--", lw=1.0, label="local MBHB max Rzeta")
    if np.isfinite(locked_threshold):
        ax.axhline(locked_threshold, color="#111827", ls=":", lw=1.2, label="locked broad midpoint")
    ax.set_xscale("log")
    ax.set_xlabel(r"$D = J\sqrt{E_A+E_E}$")
    ax.set_ylabel(r"G2 $R_{\zeta\mathrm{-WDM}}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("detection statistic vs zeta ratio")

    ax = axes[1]
    ax.bar(x, plot_rows["zeta_positive_excess_ratio"], color=plot_rows["color"])
    ax.axhline(internal_threshold, color="#2563eb", ls="--", lw=1.0)
    if np.isfinite(locked_threshold):
        ax.axhline(locked_threshold, color="#111827", ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_rows["case_id"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"G2 $R_{\zeta\mathrm{-WDM}}$")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("case-by-case Rzeta")
    fig.suptitle("Multi-link lisaglitch impact on A/E detection statistic and G2 Rzeta")
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def write_summary(metrics: dict[str, object], summary_csv: Path, metrics_json: Path, plot_path: Path) -> Path:
    output = OUTPUT_DIR / "run_summary.md"
    d_metrics = metrics["detection_statistic"]
    r_metrics = metrics["rzeta_positive_excess_ratio"]
    lines = [
        "# Multi-link Glitch Detection and Rzeta Study",
        "",
        "Physics contract:",
        "- TDI generation: PyTDI G2 Sagnac `alpha2,beta2,gamma2,zeta21,zeta22,zeta23`.",
        "- Detection statistic: `D = J * sqrt(E_A + E_E)` from the current A/E WDM tube selection.",
        "- Rzeta statistic: per-channel positive zeta excess divided by per-channel positive ABC excess inside the A/E union tube.",
        "- Multi-link glitches are sums of source-only single-link lisaglitch responses; this assumes linear additive readout-glitch response.",
        "",
        "Outputs:",
        f"- Summary CSV: `{summary_csv}`",
        f"- Metrics JSON: `{metrics_json}`",
        f"- Plot: `{plot_path}`",
        "",
        "Current run metrics:",
        f"- MBHB benchmark D range: `{format_range(d_metrics['mbhb_range'])}`",
        f"- Glitch D range: `{format_range(d_metrics['glitch_range'])}`",
        f"- MBHB benchmark Rzeta range: `{format_range(r_metrics['mbhb_range'])}`",
        f"- Glitch Rzeta range: `{format_range(r_metrics['glitch_range'])}`",
        f"- Fraction of all glitches above local MBHB-max Rzeta threshold: `{r_metrics['fraction_glitches_above_internal_mbhb_max']:.3g}`",
        f"- Fraction of multi-link glitches above local MBHB-max Rzeta threshold: `{r_metrics['fraction_multi_link_above_internal_mbhb_max']:.3g}`",
        f"- Fraction of all glitches above locked broad midpoint Rzeta threshold: `{r_metrics['fraction_glitches_above_locked_broad_simple_threshold']:.3g}`",
        "",
        "Important caveat:",
        "- This is an exploratory threshold stress test, not a final FAR calibration. The local MBHB set contains three week-1 injections, and the locked broad benchmark remains small.",
    ]
    output.write_text("\n".join(lines) + "\n")
    return output


def format_range(stats: dict[str, object]) -> str:
    return f"{stats['min']:.4g} to {stats['max']:.4g} (median {stats['median']:.4g}, n={stats['n']})"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


if __name__ == "__main__":
    main()
