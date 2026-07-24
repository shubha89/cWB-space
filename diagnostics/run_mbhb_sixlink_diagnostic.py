from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import (  # noqa: E402
    LINK_LABELS,
    LisaLinkSimulationConfig,
    LisaLinkSimulationPaths,
    load_npz_waveform,
    read_orbit_time_range,
    run_lisa_link_simulation,
)


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
WAVEFORM_FILE = PROJECT_ROOT / "lisa_link_sim" / "1e5-wave" / "inputs" / "my_waveform_1e5.npz"
REPORT_FILE = WORK_DIR / "outputs" / "mbhb_sixlink_diagnostic_report.json"
INSTRUMENT_BALANCE_FILE = WORK_DIR / "outputs" / "instrument_observable_balance.csv"
WAVEFORM_BOUNDARY_PLOT = WORK_DIR / "outputs" / "waveform_boundary_diagnostic.png"
WAVEFORM_TAPER_PREVIEW = WORK_DIR / "outputs" / "waveform_taper_preview.npz"
TAPER_SECONDS = 10_000.0


def finite_stats(values: np.ndarray) -> dict[str, float | int | bool]:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    return {
        "size": int(arr.size),
        "finite": bool(np.all(finite)),
        "min": float(np.nanmin(arr)) if arr.size else np.nan,
        "max": float(np.nanmax(arr)) if arr.size else np.nan,
        "mean": float(np.nanmean(arr)) if arr.size else np.nan,
        "std": float(np.nanstd(arr)) if arr.size else np.nan,
        "absmax": float(np.nanmax(np.abs(arr))) if arr.size else np.nan,
    }


def contiguous_support(t: np.ndarray, amp: np.ndarray, fraction: float) -> dict[str, float | int | bool]:
    peak = float(np.nanmax(amp))
    if not np.isfinite(peak) or peak <= 0.0:
        return {
            "has_support": False,
            "threshold_fraction": fraction,
            "threshold": np.nan,
            "start_s": np.nan,
            "stop_s": np.nan,
            "duration_s": np.nan,
            "peak_time_s": np.nan,
            "peak_value": peak,
        }
    idx = np.flatnonzero(amp >= fraction * peak)
    peak_idx = int(np.nanargmax(amp))
    return {
        "has_support": bool(idx.size > 0),
        "threshold_fraction": fraction,
        "threshold": float(fraction * peak),
        "start_s": float(t[idx[0]]) if idx.size else np.nan,
        "stop_s": float(t[idx[-1]]) if idx.size else np.nan,
        "duration_s": float(t[idx[-1]] - t[idx[0]]) if idx.size else np.nan,
        "peak_time_s": float(t[peak_idx]),
        "peak_value": peak,
    }


def read_hdf5_dataset_shapes(path: Path) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    with h5py.File(path, "r") as hdf:
        def visitor(name: str, obj: h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                shapes[name] = list(obj.shape)

        hdf.visititems(visitor)
    return shapes


def edge_taper(n: int, dt: float, taper_seconds: float) -> np.ndarray:
    taper_size = int(round(taper_seconds / dt))
    taper_size = max(0, min(taper_size, n // 2))
    window = np.ones(n, dtype=float)
    if taper_size:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper_size, endpoint=False)))
        window[:taper_size] = ramp
        window[-taper_size:] = ramp[::-1]
    return window


def write_waveform_boundary_diagnostic(t: np.ndarray, hp: np.ndarray, hc: np.ndarray) -> dict[str, object]:
    amp = np.sqrt(hp**2 + hc**2)
    dt = float(np.median(np.diff(t)))
    taper = edge_taper(len(t), dt, TAPER_SECONDS)
    hp_tapered = hp * taper
    hc_tapered = hc * taper
    amp_tapered = np.sqrt(hp_tapered**2 + hc_tapered**2)

    WAVEFORM_TAPER_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        WAVEFORM_TAPER_PREVIEW,
        t=t,
        hplus_tapered=hp_tapered,
        hcross_tapered=hc_tapered,
        taper=taper,
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False, constrained_layout=True)
    head_seconds = min(20_000.0, float(t[-1] - t[0]))
    head = t <= t[0] + head_seconds
    axes[0].plot(t[head], amp[head], label="current", lw=1.0)
    axes[0].plot(t[head], amp_tapered[head], label=f"{TAPER_SECONDS:g}s edge taper", lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("strain amplitude")
    axes[0].set_title("Waveform start boundary")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    peak_idx = int(np.nanargmax(amp))
    peak_center = t[peak_idx]
    peak_half_width = min(8_000.0, 0.5 * float(t[-1] - t[0]))
    near_peak = (t >= peak_center - peak_half_width) & (t <= peak_center + peak_half_width)
    axes[1].plot(t[near_peak], amp[near_peak], label="current", lw=1.0)
    axes[1].plot(t[near_peak], amp_tapered[near_peak], label=f"{TAPER_SECONDS:g}s edge taper", lw=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time in science segment [s]")
    axes[1].set_ylabel("strain amplitude")
    axes[1].set_title("Peak region")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(WAVEFORM_BOUNDARY_PLOT, dpi=160)
    plt.close(fig)

    peak = float(np.nanmax(amp))
    tapered_peak = float(np.nanmax(amp_tapered))
    return {
        "taper_seconds": TAPER_SECONDS,
        "plot": str(WAVEFORM_BOUNDARY_PLOT),
        "taper_preview_npz": str(WAVEFORM_TAPER_PREVIEW),
        "current_start_amp_over_peak": float(amp[0] / peak) if peak > 0.0 else np.nan,
        "tapered_start_amp_over_peak": float(amp_tapered[0] / tapered_peak) if tapered_peak > 0.0 else np.nan,
        "current_peak_amp": peak,
        "tapered_peak_amp": tapered_peak,
        "tapered_peak_over_current_peak": float(tapered_peak / peak) if peak > 0.0 else np.nan,
    }


def instrument_observable_balance(path: Path, burn_in_size: int) -> tuple[pd.DataFrame, dict[str, object]]:
    candidate_groups = [
        "debug/sci_carrier_fluctuations",
        "debug/ref_carrier_fluctuations",
        "debug/tmi_carrier_fluctuations",
        "debug/sci_usb_fluctuations",
        "debug/ref_usb_fluctuations",
        "debug/tmi_usb_fluctuations",
        "sci_carriers",
        "ref_carriers",
        "tmi_carriers",
        "sci_usbs",
        "ref_usbs",
        "tmi_usbs",
        "mprs",
    ]
    rows = []
    with h5py.File(path, "r") as hdf:
        for group in candidate_groups:
            if group not in hdf:
                continue
            stds = []
            for label in LINK_LABELS:
                key = f"{group}/{label}"
                if key not in hdf:
                    continue
                arr = np.asarray(hdf[key][burn_in_size:], dtype=float)
                stats = finite_stats(arr)
                rows.append(
                    {
                        "group": group,
                        "link": label,
                        "std": stats["std"],
                        "absmax": stats["absmax"],
                        "mean": stats["mean"],
                        "finite": stats["finite"],
                    }
                )
                stds.append(float(stats["std"]))
            positive = np.asarray([v for v in stds if np.isfinite(v) and v > 0.0], dtype=float)
            if positive.size:
                median = float(np.median(positive))
                min_std = float(np.min(positive))
                max_std = float(np.max(positive))
                balanced = bool(min_std > 1e-6 * median and max_std / min_std < 1e4)
            else:
                median = min_std = max_std = np.nan
                balanced = False
            rows.append(
                {
                    "group": group,
                    "link": "__group_summary__",
                    "std": median,
                    "absmax": max_std / min_std if np.isfinite(min_std) and min_std > 0.0 else np.nan,
                    "mean": np.nan,
                    "finite": balanced,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(INSTRUMENT_BALANCE_FILE, index=False)

    summaries = df[df["link"] == "__group_summary__"].copy()
    balanced_groups = summaries[summaries["finite"].astype(bool)]["group"].tolist()
    return df, {
        "balance_table": str(INSTRUMENT_BALANCE_FILE),
        "balanced_candidate_groups": balanced_groups,
        "current_saved_noisy_group": "debug/sci_carrier_fluctuations",
        "recommended_balanced_diagnostic_group": "debug/tmi_carrier_fluctuations"
        if "debug/tmi_carrier_fluctuations" in balanced_groups
        else (balanced_groups[0] if balanced_groups else None),
        "summary_note": "std is computed after burn-in; group summary absmax is max(std)/min(std).",
    }


def build_report(paths: LisaLinkSimulationPaths, diagnostics: dict[str, float | int]) -> dict[str, object]:
    aligned = np.load(paths.aligned_waveform_npz)
    wave_t = aligned["t"]
    hp = aligned["hplus"]
    hc = aligned["hcross"]
    wave_amp = np.sqrt(hp**2 + hc**2)
    boundary_diagnostic = write_waveform_boundary_diagnostic(wave_t, hp, hc)

    orbit = read_orbit_time_range(paths.orbits_h5)
    gw_links = pd.read_csv(paths.gw_links_csv)
    noisy_links = pd.read_csv(paths.noisy_links_csv)
    noisy_tdi = pd.read_csv(paths.noisy_tdi_csv)
    gw_tdi = pd.read_csv(paths.gw_tdi_csv)

    gw_science = gw_links[gw_links["is_science"].astype(bool)].reset_index(drop=True)
    link_stats = {
        f"gw_{lab}_hz": finite_stats(gw_science[f"gw_{lab}_hz"].to_numpy(dtype=float))
        for lab in LINK_LABELS
    }
    noisy_stats = {
        f"noisy_{lab}": finite_stats(noisy_links[f"noisy_{lab}"].to_numpy(dtype=float))
        for lab in LINK_LABELS
    }
    burn_in_size = int(diagnostics.get("burn_in_size", noisy_links["instrument_sample"].iloc[0]))
    _, observable_balance = instrument_observable_balance(paths.instrument_h5, burn_in_size)

    checks = {
        "aligned_waveform_samples_match_science": bool(len(wave_t) == diagnostics["science_samples"]),
        "gw_science_rows_match_noisy_rows": bool(len(gw_science) == len(noisy_links)),
        "tdi_rows_match_science": bool(len(noisy_tdi) == diagnostics["tdi_samples"] == len(gw_tdi)),
        "gw_link_times_match_noisy_times": bool(
            len(gw_science) == len(noisy_links)
            and np.allclose(gw_science["t_lisa"].to_numpy(), noisy_links["t_lisa"].to_numpy())
        ),
        "orbit_covers_simulated_lisa_times": bool(
            orbit.t0 <= float(gw_links["t_lisa"].iloc[0])
            and float(gw_links["t_lisa"].iloc[-1]) <= orbit.t1
        ),
    }
    noisy_std = np.asarray([stats["std"] for stats in noisy_stats.values()], dtype=float)
    finite_positive_std = noisy_std[np.isfinite(noisy_std) & (noisy_std > 0.0)]
    median_noisy_std = float(np.median(finite_positive_std)) if finite_positive_std.size else np.nan
    weak_noisy_links = [
        label
        for label, stats in noisy_stats.items()
        if np.isfinite(median_noisy_std) and float(stats["std"]) < 1e-6 * median_noisy_std
    ]
    start_amp_over_peak = float(wave_amp[0] / np.max(wave_amp)) if np.max(wave_amp) > 0 else np.nan
    stop_amp_over_peak = float(wave_amp[-1] / np.max(wave_amp)) if np.max(wave_amp) > 0 else np.nan
    warnings = []
    if start_amp_over_peak > 1e-3:
        warnings.append(
            "Aligned waveform is nonzero at the science-segment start; this can create boundary/edge power."
        )
    if weak_noisy_links:
        warnings.append(
            "Some raw debug sci_carrier_fluctuations links are nearly zero relative to the others: "
            + ", ".join(weak_noisy_links)
        )

    return {
        "waveform_file": str(WAVEFORM_FILE),
        "work_dir": str(WORK_DIR),
        "diagnostics": diagnostics,
        "checks": checks,
        "warnings": warnings,
        "raw_noisy_link_balance": {
            "median_positive_std_hz": median_noisy_std,
            "weak_links": weak_noisy_links,
        },
        "instrument_observable_balance": observable_balance,
        "pytdi_measurement_convention": {
            "gw_only_tdi_input": "Data.from_gws sets carrier and sideband SCI beatnotes to the six GW link responses; other beatnotes are zero.",
            "instrument_tdi_input": "Data.from_instrument uses carrier SCI, TMI, REF beatnotes, sideband SCI/REF beatnotes, and measured pseudoranges.",
            "diagnostic_conclusion": (
                "The saved six_links_gw_only.csv is a true six-link GW response. "
                "The saved six_links_noisy.csv is only a raw debug sci_carrier_fluctuations view, "
                "not the complete noisy six-link input used by PyTDI."
            ),
        },
        "waveform_boundary_diagnostic": boundary_diagnostic,
        "aligned_waveform": {
            "hplus": finite_stats(hp),
            "hcross": finite_stats(hc),
            "amplitude": finite_stats(wave_amp),
            "support_above_1e-3_peak": contiguous_support(wave_t, wave_amp, 1e-3),
            "support_above_1e-2_peak": contiguous_support(wave_t, wave_amp, 1e-2),
            "start_amp_over_peak": start_amp_over_peak,
            "stop_amp_over_peak": stop_amp_over_peak,
        },
        "orbit": {
            "t0": orbit.t0,
            "t1": orbit.t1,
            "dt": orbit.dt,
            "size": orbit.size,
            "span": orbit.span,
        },
        "gw_links_science": {
            "rows": int(len(gw_science)),
            "t_lisa_start": float(gw_science["t_lisa"].iloc[0]),
            "t_lisa_stop": float(gw_science["t_lisa"].iloc[-1]),
            "t_waveform_start": float(gw_science["t_waveform"].iloc[0]),
            "t_waveform_stop": float(gw_science["t_waveform"].iloc[-1]),
            "per_link": link_stats,
        },
        "noisy_links_science": {
            "rows": int(len(noisy_links)),
            "per_link": noisy_stats,
        },
        "tdi": {
            "noisy_rows": int(len(noisy_tdi)),
            "gw_rows": int(len(gw_tdi)),
            "noisy_columns": list(noisy_tdi.columns),
            "gw_columns": list(gw_tdi.columns),
        },
        "hdf5_shapes": {
            "gws": read_hdf5_dataset_shapes(paths.gws_h5),
            "instrument": read_hdf5_dataset_shapes(paths.instrument_h5),
        },
        "outputs": {
            "aligned_waveform": str(paths.aligned_waveform_npz),
            "orbits": str(paths.orbits_h5),
            "gws": str(paths.gws_h5),
            "instrument": str(paths.instrument_h5),
            "gw_links": str(paths.gw_links_csv),
            "noisy_links": str(paths.noisy_links_csv),
            "noisy_tdi": str(paths.noisy_tdi_csv),
            "gw_tdi": str(paths.gw_tdi_csv),
            "plots": str(paths.plot_dir),
        },
    }


def main() -> None:
    parser = ArgumentParser(description="Run or summarize the MBHB six-link diagnostic.")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip simulation and rebuild the JSON report from existing diagnostic outputs.",
    )
    args = parser.parse_args()

    if not WAVEFORM_FILE.exists():
        raise FileNotFoundError(f"Missing MBHB waveform file: {WAVEFORM_FILE}")

    t, hplus, hcross = load_npz_waveform(WAVEFORM_FILE)
    config = LisaLinkSimulationConfig(
        dt=0.1,
        science_duration=150_000.0,
        burn_in_seconds=600.0,
        center_waveform_peak=True,
        edge_taper_seconds=TAPER_SECONDS,
        ra=np.pi / 4.0,
        dec=0.3,
        central_freq_hz=2.816e14,
        orbit_name="esa-trailing",
        orbit_dt=600.0,
        orbit_duration_factor=5.0,
        seed=12345,
        frequency_low_hz=1e-5,
        frequency_high_hz=1e-1,
        tdi_generation=2,
        interpolation_order=31,
        tdi_unit="frequency",
    )

    if args.reuse_existing:
        paths = LisaLinkSimulationPaths.from_work_dir(WORK_DIR)
        required = [
            paths.aligned_waveform_npz,
            paths.orbits_h5,
            paths.gws_h5,
            paths.instrument_h5,
            paths.gw_links_csv,
            paths.noisy_links_csv,
            paths.noisy_tdi_csv,
            paths.gw_tdi_csv,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Cannot reuse existing diagnostic outputs; missing: " + ", ".join(missing))
        noisy_tdi = pd.read_csv(paths.noisy_tdi_csv, usecols=["X2", "A2"])
        gw_tdi = pd.read_csv(paths.gw_tdi_csv, usecols=["X2_gw_hz", "A2_gw_hz"])
        diagnostics = {
            "science_samples": int(round(config.science_duration / config.dt)),
            "science_dt": config.dt,
            "science_fft_df_hz": 1.0 / config.science_duration,
            "waveform_peak_time": config.science_duration / 2.0,
            "waveform_max_strain": float(np.max(np.sqrt(hplus**2 + hcross**2))),
            "time_frame": "TCB seconds since LISA epoch",
            "sky_frame": "ICRS right ascension/declination",
            "link_index_convention": "ij means receiver i, emitter j",
            "gw_dataset": config.gw_dataset,
            "orbit_dataset": config.orbit_dataset,
            "gw_link_unit": "dimensionless fractional frequency y_ij",
            "gw_link_hz_unit": "Hz after multiplying by central_freq_hz",
            "tdi_unit": config.tdi_unit,
            "tdi_samples": int(len(noisy_tdi)),
            "noisy_x_std_hz": float(np.std(noisy_tdi["X2"].to_numpy(dtype=float))),
            "noisy_a_std_hz": float(np.std(noisy_tdi["A2"].to_numpy(dtype=float))),
            "gw_x_max_hz": float(np.max(np.abs(gw_tdi["X2_gw_hz"].to_numpy(dtype=float)))),
            "gw_a_max_hz": float(np.max(np.abs(gw_tdi["A2_gw_hz"].to_numpy(dtype=float)))),
        }
        products_paths = paths
    else:
        products = run_lisa_link_simulation(
            t,
            hplus,
            hcross,
            work_dir=WORK_DIR,
            config=config,
            make_plots=True,
        )
        diagnostics = dict(products.diagnostics)
        products_paths = products.paths

    report = build_report(products_paths, diagnostics)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report["checks"], indent=2))
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    print(f"report = {REPORT_FILE}")
    print(f"plots  = {products_paths.plot_dir}")


if __name__ == "__main__":
    main()
