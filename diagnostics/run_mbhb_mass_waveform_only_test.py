from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clean_wdm import WDMResolution, compute_wdm_transform, downsample_to_target_dt, require_wdm_dependencies
from scripts.lisa_link_simulation import LisaLinkSimulationConfig, LisaLinkSimulationPaths
from scripts.six_link_generation import compute_gw_links, prepare_science_waveform
from scripts.tdi_computation import compute_tdi_from_gws


BASE_WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
BASE_OUTPUT_DIR = BASE_WORK_DIR / "outputs"
BASE_ORBITS = BASE_WORK_DIR / "inputs" / "orbits.h5"
BASE_NOISE_TDI = BASE_OUTPUT_DIR / "tdi_noise_only_seed12345_XYZ_AET.csv"
MAIN_OUTPUT_DIR = BASE_OUTPUT_DIR
MAIN_PLOT_DIR = BASE_OUTPUT_DIR / "plots"

RESOLUTIONS = {
    "512": WDMResolution("dt2_M512", 2.0, 512),
    "4096": WDMResolution("dt2_M4096", 2.0, 4096),
}


def tag_from_mass(total_mass: float) -> str:
    exponent = int(round(np.log10(total_mass)))
    if np.isclose(total_mass, 10.0**exponent):
        return f"1e{exponent}"
    return f"{total_mass:.3g}".replace("+", "").replace(".", "p")


def generate_phenom_waveform(
    total_mass: float,
    *,
    distance_mpc: float,
    f_min_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    from phenomxpy import IMRPhenomTHM

    params: dict[str, float | int] = {
        "eta": 0.24,
        "total_mass": float(total_mass),
        "distance_mpc": float(distance_mpc),
        "s1": 0.3,
        "s2": -0.2,
        "f_min_hz": float(f_min_hz),
        "iota": 0.7,
        "phi_ref": 0.0,
        "raw_dt_s": 1.0,
        "raw_half_duration_s": 300_000.0,
    }
    times = np.arange(-params["raw_half_duration_s"], params["raw_half_duration_s"], params["raw_dt_s"])
    model = IMRPhenomTHM(
        eta=params["eta"],
        total_mass=params["total_mass"],
        distance=params["distance_mpc"],
        s1=params["s1"],
        s2=params["s2"],
        f_min=params["f_min_hz"],
        iota=params["iota"],
        phi_ref=params["phi_ref"],
    )
    out = model.compute_polarizations(times)
    if len(out) == 2:
        hplus, hcross = out
        t = times
    elif len(out) == 3:
        arrays = [np.asarray(item) for item in out]
        if arrays[0].shape == times.shape and np.all(np.diff(arrays[0]) > 0):
            t, hplus, hcross = arrays
        elif arrays[2].shape == times.shape and np.all(np.diff(arrays[2]) > 0):
            hplus, hcross, t = arrays
        else:
            t = times
            hplus, hcross = arrays[:2]
    else:
        raise ValueError(f"Unexpected phenomxpy output length: {len(out)}")
    t = np.asarray(t, dtype=float)
    hplus = np.asarray(hplus, dtype=float)
    hcross = np.asarray(hcross, dtype=float)
    finite = np.isfinite(t) & np.isfinite(hplus) & np.isfinite(hcross)
    t = t[finite]
    hplus = hplus[finite]
    hcross = hcross[finite]
    amp = np.sqrt(hplus**2 + hcross**2)
    peak_idx = int(np.argmax(amp))
    params.update(
        {
            "n_raw_samples": int(len(t)),
            "raw_peak_time_s": float(t[peak_idx]),
            "raw_max_strain": float(np.max(amp)),
        }
    )
    return t, hplus, hcross, params


def plot_waveform(t: np.ndarray, hp: np.ndarray, hc: np.ndarray, output: Path, total_mass: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    amp = np.sqrt(hp**2 + hc**2)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(t / 86400.0, amp, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Raw waveform time [days]")
    ax.set_ylabel("strain amplitude")
    ax.set_title(rf"IMRPhenomTHM MBHB waveform, $M={total_mass:.0e}\ M_\odot$")
    ax.grid(True, which="both", alpha=0.2)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def combine_noise_and_gw_tdi(paths: LisaLinkSimulationPaths) -> Path:
    noise = pd.read_csv(BASE_NOISE_TDI)
    gw = pd.read_csv(paths.gw_tdi_csv)
    if len(noise) != len(gw) or not np.allclose(noise["t_lisa"], gw["t_lisa"], rtol=0.0, atol=1e-6):
        raise ValueError("New GW-only TDI is not aligned with the reused noise-only TDI.")

    out = noise[["sample", "instrument_sample", "t_waveform", "t_lisa"]].copy()
    for ch in ("X", "Y", "Z", "A", "E", "T"):
        out[f"{ch}2"] = noise[f"{ch}2"].to_numpy(dtype=float) + gw[f"{ch}2_gw_hz"].to_numpy(dtype=float)
    paths.noisy_tdi_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.noisy_tdi_csv, index=False)
    return paths.noisy_tdi_csv


def write_links_for_loader(paths: LisaLinkSimulationPaths) -> None:
    gw_links = pd.read_csv(paths.gw_links_csv)
    science = gw_links[gw_links["is_science"].astype(bool)].reset_index(drop=True)
    noisy_links = pd.DataFrame(
        {
            "sample": np.arange(len(science)),
            "t_lisa": science["t_lisa"].to_numpy(dtype=float),
        }
    )
    for label in ("12", "23", "31", "13", "32", "21"):
        noisy_links[f"noisy_{label}"] = science[f"gw_{label}_hz"].to_numpy(dtype=float)
    noisy_links.to_csv(paths.noisy_links_csv, index=False)


def locate_signal(t_lisa: np.ndarray, gw: pd.DataFrame, *, threshold_fraction: float = 1e-3) -> dict[str, float]:
    power = (
        gw["A2_gw_hz"].to_numpy(dtype=float) ** 2
        + gw["E2_gw_hz"].to_numpy(dtype=float) ** 2
        + gw["T2_gw_hz"].to_numpy(dtype=float) ** 2
    )
    peak_idx = int(np.argmax(power))
    keep = np.flatnonzero(power >= threshold_fraction * float(power[peak_idx]))
    return {
        "signal_peak_lisa": float(t_lisa[peak_idx]),
        "signal_peak_rel": float(t_lisa[peak_idx] - t_lisa[0]),
        "signal_window_start_lisa": float(t_lisa[keep[0]]),
        "signal_window_stop_lisa": float(t_lisa[keep[-1]]),
        "signal_window_start_rel": float(t_lisa[keep[0]] - t_lisa[0]),
        "signal_window_stop_rel": float(t_lisa[keep[-1]] - t_lisa[0]),
    }


def compute_twhitened_aet_maps(
    paths: LisaLinkSimulationPaths,
    *,
    tag: str,
    resolutions: list[WDMResolution],
    report: dict[str, object],
) -> list[dict[str, str]]:
    resample_poly, WDM = require_wdm_dependencies()
    noise = pd.read_csv(BASE_NOISE_TDI, usecols=["t_lisa", "T2"])
    noisy = pd.read_csv(paths.noisy_tdi_csv)
    gw = pd.read_csv(paths.gw_tdi_csv)
    t_lisa = noisy["t_lisa"].to_numpy(dtype=float)
    if not np.allclose(noise["t_lisa"].to_numpy(dtype=float), t_lisa, rtol=0.0, atol=1e-6):
        raise ValueError("Noise TDI is not aligned with new noisy TDI.")
    native_dt = float(np.median(np.diff(t_lisa)))
    signal = locate_signal(t_lisa, gw)

    outputs = []
    for resolution in resolutions:
        wdm = WDM(M=resolution.M, K=32, beta_order=6, precision=6, backend="numba")
        t_noise_ds, dt_ds, _ = downsample_to_target_dt(
            noise["T2"].to_numpy(dtype=float),
            native_dt,
            resolution.target_dt,
            resample_poly,
        )
        t_coeff, times_wdm, freqs_wdm, _, _ = compute_wdm_transform(
            wdm,
            t_noise_ds,
            sample_rate=1.0 / dt_ds,
            t0=float(t_lisa[0]),
            original_t1=float(t_lisa[-1]),
        )
        t_floor = np.nanmedian(np.abs(t_coeff) ** 2, axis=1)
        t_floor = np.where(np.isfinite(t_floor) & (t_floor > 0.0), t_floor, 1e-300)

        payload: dict[str, np.ndarray] = {
            "signal": np.asarray(
                [
                    signal["signal_peak_lisa"],
                    signal["signal_peak_rel"],
                    signal["signal_window_start_lisa"],
                    signal["signal_window_stop_lisa"],
                    signal["signal_window_start_rel"],
                    signal["signal_window_stop_rel"],
                ],
                dtype=float,
            ),
            "native_t_lisa_start": np.asarray(float(t_lisa[0])),
            "native_dt_s": np.asarray(native_dt),
            f"T_noise_{resolution.label}_floor": t_floor.astype(np.float32),
            f"T_noise_{resolution.label}_times_rel": (times_wdm - t_lisa[0]).astype(np.float64),
            f"T_noise_{resolution.label}_freqs_hz": freqs_wdm.astype(np.float64),
        }
        rows = []
        sources = {
            "noisy": {"A": noisy["A2"].to_numpy(dtype=float), "E": noisy["E2"].to_numpy(dtype=float)},
            "gw_only": {"A": gw["A2_gw_hz"].to_numpy(dtype=float), "E": gw["E2_gw_hz"].to_numpy(dtype=float)},
        }
        for source, channels in sources.items():
            for channel, series in channels.items():
                x_ds, x_dt, _ = downsample_to_target_dt(series, native_dt, resolution.target_dt, resample_poly)
                coeff, x_times, x_freqs, _, _ = compute_wdm_transform(
                    wdm,
                    x_ds,
                    sample_rate=1.0 / x_dt,
                    t0=float(t_lisa[0]),
                    original_t1=float(t_lisa[-1]),
                )
                if coeff.shape != t_coeff.shape:
                    raise ValueError("A/E WDM map shape does not match T-noise map shape.")
                raw_power = np.abs(coeff) ** 2
                twhitened_power = raw_power / t_floor[:, None]
                key = f"{source}_{channel}_{resolution.label}"
                payload[f"{key}_twhitened_real"] = np.real(coeff / np.sqrt(t_floor[:, None])).astype(np.float32)
                payload[f"{key}_twhitened_imag"] = np.imag(coeff / np.sqrt(t_floor[:, None])).astype(np.float32)
                payload[f"{key}_twhitened_power"] = twhitened_power.astype(np.float32)
                payload[f"{key}_times_rel"] = (x_times - t_lisa[0]).astype(np.float64)
                payload[f"{key}_freqs_hz"] = x_freqs.astype(np.float64)
                band = (x_freqs >= 1e-5) & (x_freqs <= 1e-1)
                band_power = twhitened_power[band, :]
                peak = np.unravel_index(int(np.argmax(band_power)), band_power.shape)
                rows.append(
                    {
                        "source": source,
                        "channel": channel,
                        "resolution": resolution.label,
                        "max_twhitened_power": float(np.max(band_power)),
                        "peak_time_rel": float((x_times - t_lisa[0])[peak[1]]),
                        "peak_freq_hz": float(x_freqs[band][peak[0]]),
                        "wdm_df_hz": float(x_freqs[1] - x_freqs[0]) if len(x_freqs) > 1 else np.nan,
                        "wdm_dt_s": float(x_times[1] - x_times[0]) if len(x_times) > 1 else np.nan,
                    }
                )

        MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MAIN_PLOT_DIR.mkdir(parents=True, exist_ok=True)
        middle = "" if resolution.label == "dt2_M512" else f"_{resolution.label}"
        maps_npz = MAIN_OUTPUT_DIR / f"twhitened_wdm_mbhb_{tag}{middle}_aet_maps.npz"
        stats_csv = MAIN_OUTPUT_DIR / f"twhitened_wdm_mbhb_{tag}{middle}_aet_stats.csv"
        np.savez_compressed(maps_npz, **payload)
        pd.DataFrame(rows).to_csv(stats_csv, index=False)
        outputs.append({"resolution": resolution.label, "maps_npz": str(maps_npz), "stats_csv": str(stats_csv)})

    report["twhitened_wdm"] = {
        "outputs": outputs,
        "signal": signal,
    }
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MBHB waveform-only branch and T-whitened A/E WDM maps.")
    parser.add_argument("--total-mass", type=float, required=True, help="Detector-frame total mass in solar masses.")
    parser.add_argument("--tag", default=None, help="Output tag, e.g. 1e6. Defaults to a tag derived from total mass.")
    parser.add_argument("--distance-mpc", type=float, default=6230.0, help="Luminosity distance in Mpc.")
    parser.add_argument("--f-min-hz", type=float, default=1.0e-4, help="Starting frequency passed to phenomxpy.")
    parser.add_argument("--wdm-ms", nargs="+", choices=sorted(RESOLUTIONS), default=["512", "4096"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag or tag_from_mass(args.total_mass)
    work_dir = PROJECT_ROOT / "diagnostics" / f"mbhb_sixlinks_{tag}"
    output_dir = work_dir / "outputs"
    paths = LisaLinkSimulationPaths.from_work_dir(work_dir)
    paths.make_dirs()
    if not BASE_ORBITS.exists():
        raise FileNotFoundError(f"Missing base orbit file: {BASE_ORBITS}")
    if not BASE_NOISE_TDI.exists():
        raise FileNotFoundError(f"Missing reusable noise-only TDI: {BASE_NOISE_TDI}")
    paths.orbits_h5.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_ORBITS, paths.orbits_h5)

    t_raw, hp_raw, hc_raw, waveform_meta = generate_phenom_waveform(
        args.total_mass,
        distance_mpc=args.distance_mpc,
        f_min_hz=args.f_min_hz,
    )
    paths.input_dir.mkdir(parents=True, exist_ok=True)
    raw_waveform_npz = paths.input_dir / f"my_waveform_{tag}_raw.npz"
    np.savez_compressed(
        raw_waveform_npz,
        t=t_raw,
        hplus=hp_raw,
        hcross=hc_raw,
        total_mass=np.asarray(args.total_mass),
        distance_mpc=np.asarray(args.distance_mpc),
        f_min_hz=np.asarray(args.f_min_hz),
    )
    waveform_meta_path = output_dir / f"my_waveform_{tag}_metadata.json"
    waveform_meta_path.parent.mkdir(parents=True, exist_ok=True)
    waveform_meta_path.write_text(json.dumps(waveform_meta, indent=2) + "\n", encoding="utf-8")
    plot_waveform(t_raw, hp_raw, hc_raw, output_dir / f"my_waveform_{tag}_preview.png", args.total_mass)

    config = LisaLinkSimulationConfig(
        dt=0.1,
        science_duration=150_000.0,
        burn_in_seconds=600.0,
        center_waveform_peak=True,
        edge_taper_seconds=10_000.0,
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
    prepared = prepare_science_waveform(
        t_raw,
        hp_raw,
        hc_raw,
        config,
        output_file=paths.aligned_waveform_npz,
        plot_file=output_dir / f"01_aligned_waveform_{tag}.png",
    )
    gw_diag = compute_gw_links(prepared, config, paths, make_plots=False)
    gw_tdi_diag = compute_tdi_from_gws(config, paths, make_plots=False)
    noisy_tdi = combine_noise_and_gw_tdi(paths)
    write_links_for_loader(paths)

    report: dict[str, object] = {
        "method": "MBHB waveform-only branch; existing seed12345 realistic noise-only TDI reused, no lisainstrument noise regeneration",
        "tag": tag,
        "work_dir": str(work_dir),
        "waveform": {
            **waveform_meta,
            "raw_waveform_npz": str(raw_waveform_npz),
            "aligned_waveform_npz": str(paths.aligned_waveform_npz),
            "metadata_json": str(waveform_meta_path),
        },
        "config": {
            "dt_s": config.dt,
            "science_duration_s": config.science_duration,
            "burn_in_seconds": config.burn_in_seconds,
            "frequency_low_hz": config.frequency_low_hz,
            "frequency_high_hz": config.frequency_high_hz,
            "tdi_generation": config.tdi_generation,
            "tdi_unit": config.tdi_unit,
            "reused_noise_tdi": str(BASE_NOISE_TDI),
            "reused_orbits": str(BASE_ORBITS),
        },
        "gw_links": gw_diag,
        "gw_tdi": gw_tdi_diag,
        "outputs": {
            "gw_links_csv": str(paths.gw_links_csv),
            "gw_tdi_csv": str(paths.gw_tdi_csv),
            "noisy_tdi_csv": str(noisy_tdi),
        },
    }
    resolutions = [RESOLUTIONS[item] for item in args.wdm_ms]
    compute_twhitened_aet_maps(paths, tag=tag, resolutions=resolutions, report=report)
    report_path = output_dir / f"mbhb_{tag}_waveform_only_test_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report = {report_path}")


if __name__ == "__main__":
    main()
