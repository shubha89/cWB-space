"""Example: compute coronagraphic kappa from pyTDI Sagnac variables.

This is a diagnostic bridge, not yet the production kappa path. It uses pyTDI
for alpha/beta/gamma and reuses the existing static-geometry coronagraphic
projection A(f, sky). The projection is formally matched to first-generation
static Sagnac, so ALPHA1/BETA1/GAMMA1 is the default comparison target.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from pytdi import Data
import pytdi.sagnac

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cluster.config import load_pipeline_config  # noqa: E402
from coronagraphic_tdi import (  # noqa: E402
    LISAGeometry,
    _freq_to_time_plus,
    _time_to_freq_plus,
    build_sagnac_from_links,
    coronagraphic_kappa,
)
from scripts.six_link_generation import LINK_LABELS, LisaLinkSimulationConfig, LisaLinkSimulationPaths  # noqa: E402
from scripts.tdi_computation import apply_frequency_cut, get_delay_derivatives_if_available  # noqa: E402

C_M_PER_S = 299_792_458.0
LISA_ARM_M = 2.5e9


def build_lisa_config(config, row: pd.Series) -> LisaLinkSimulationConfig:
    sim = config.simulation
    return LisaLinkSimulationConfig(
        dt=sim.dt_s,
        science_duration=sim.science_duration_s,
        burn_in_seconds=sim.burn_in_seconds,
        center_waveform_peak=True,
        ra=float(row["ra"]),
        dec=float(row["dec"]),
        central_freq_hz=float(sim.central_freq_hz),
        orbit_dt=sim.orbit_dt_s,
        orbit_duration_factor=sim.orbit_duration_factor,
        seed=sim.seed,
        frequency_low_hz=sim.frequency_low_hz,
        frequency_high_hz=sim.frequency_high_hz,
        tdi_generation=sim.tdi_generation,
        interpolation_order=sim.interpolation_order,
        tdi_unit="frequency",
    )


def crop_to_science(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    paths: LisaLinkSimulationPaths,
    config: LisaLinkSimulationConfig,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray, int]:
    meta = pd.read_csv(paths.gw_links_csv, usecols=["t_waveform", "t_lisa", "is_science"])
    t_lisa_meta = meta["t_lisa"].to_numpy(dtype=float)
    t_waveform_meta = meta["t_waveform"].to_numpy(dtype=float)
    is_science = meta["is_science"].astype(bool).to_numpy()
    burn_in_size = int(np.argmax(is_science)) if not is_science[0] else 0
    start = burn_in_size + config.extra_skipped + config.edge_crop
    n = min([len(arr) for arr in arrays] + [len(t_lisa_meta)])
    stop = n - config.edge_crop
    if stop <= start:
        raise ValueError("Not enough samples after burn-in and edge crop.")
    cropped = tuple(arr[start:stop] for arr in arrays)
    n_out = min([len(arr) for arr in cropped] + [stop - start])
    return (
        tuple(arr[:n_out] for arr in cropped),
        t_waveform_meta[start : start + n_out],
        t_lisa_meta[start : start + n_out],
        start,
    )


def evaluate_pytdi_sagnac(
    *,
    paths: LisaLinkSimulationPaths,
    config: LisaLinkSimulationConfig,
    source: str,
    generation: int,
    max_samples: int | None,
    eta_combinations: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if generation not in {1, 2}:
        raise ValueError("generation must be 1 or 2.")
    if source == "noisy":
        data = Data.from_instrument(str(paths.instrument_h5), signals="fluctuations", skipped=0)
        scale = 1.0
    elif source == "gw_only":
        data = Data.from_gws(
            str(paths.gws_h5),
            str(paths.orbits_h5),
            skipped=0,
            gw_dataset=config.gw_dataset,
            orbit_dataset=config.orbit_dataset,
        )
        scale = float(config.central_freq_hz)
    else:
        raise ValueError("source must be 'noisy' or 'gw_only'.")

    build_kwargs = {"order": config.interpolation_order}
    delay_derivatives = get_delay_derivatives_if_available(data)
    if delay_derivatives is not None:
        build_kwargs["delay_derivatives"] = delay_derivatives

    suffix = "_ETA" if eta_combinations else ""
    alpha_comb = getattr(pytdi.sagnac, f"ALPHA{generation}{suffix}")
    beta_comb = getattr(pytdi.sagnac, f"BETA{generation}{suffix}")
    gamma_comb = getattr(pytdi.sagnac, f"GAMMA{generation}{suffix}")

    if eta_combinations:
        missing_eta = sorted(
            measurement
            for measurement in ("eta_12", "eta_23", "eta_31", "eta_13", "eta_32", "eta_21")
            if measurement not in data.measurements
        )
        if missing_eta:
            available = ", ".join(sorted(data.measurements.keys()))
            raise KeyError(
                "pyTDI *_ETA Sagnac combinations require eta_ij measurements, "
                f"but these are missing: {missing_eta}. Available measurements: {available}"
            )

    alpha = alpha_comb.build(data.delays, data.fs, **build_kwargs)(data.measurements, unit=config.tdi_unit) * scale
    beta = beta_comb.build(data.delays, data.fs, **build_kwargs)(data.measurements, unit=config.tdi_unit) * scale
    gamma = gamma_comb.build(data.delays, data.fs, **build_kwargs)(data.measurements, unit=config.tdi_unit) * scale
    (alpha, beta, gamma), t_waveform, _, _ = crop_to_science((alpha, beta, gamma), paths, config)
    dt = float(np.median(np.diff(t_waveform)))
    alpha, beta, gamma = [
        apply_frequency_cut(arr, dt, config.frequency_low_hz, config.frequency_high_hz)
        for arr in (alpha, beta, gamma)
    ]
    if max_samples is not None and int(max_samples) > 0 and len(t_waveform) > int(max_samples):
        n_keep = int(max_samples)
        center = len(t_waveform) // 2
        start = max(0, center - n_keep // 2)
        stop = start + n_keep
        t_waveform = t_waveform[start:stop]
        alpha = alpha[start:stop]
        beta = beta[start:stop]
        gamma = gamma[start:stop]
    freqs = np.fft.fftfreq(len(t_waveform), d=dt)
    return t_waveform, freqs, _time_to_freq_plus(alpha), _time_to_freq_plus(beta), _time_to_freq_plus(gamma)


def static_sagnac_from_link_csv(
    *,
    paths: LisaLinkSimulationPaths,
    source: str,
    t_reference: np.ndarray,
    geometry: LISAGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source == "noisy":
        df = pd.read_csv(paths.noisy_links_csv)
        cols = {link: f"noisy_{link}" for link in LINK_LABELS}
    else:
        df = pd.read_csv(paths.gw_links_csv)
        df = df[df["is_science"].astype(bool)].reset_index(drop=True)
        cols = {link: f"gw_{link}_hz" for link in LINK_LABELS}

    mask = (df["t_waveform"] >= float(t_reference[0])) & (df["t_waveform"] <= float(t_reference[-1]))
    chunk = df.loc[mask].reset_index(drop=True)
    n = min(len(chunk), len(t_reference))
    if n < 1024:
        raise ValueError("Could not match enough link samples to the pyTDI Sagnac chunk.")
    chunk = chunk.iloc[:n]
    t_reference = t_reference[:n]
    dt = float(np.median(np.diff(t_reference)))
    freqs = np.fft.fftfreq(n, d=dt)
    y_f = {link: _time_to_freq_plus(chunk[col].to_numpy(dtype=float)) for link, col in cols.items()}
    return (
        build_sagnac_from_links("alpha", freqs, y_f, geometry),
        build_sagnac_from_links("beta", freqs, y_f, geometry),
        build_sagnac_from_links("gamma", freqs, y_f, geometry),
    )


def real_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    xr = np.real(x)
    yr = np.real(y)
    ok = np.isfinite(xr) & np.isfinite(yr)
    if int(np.sum(ok)) < 2 or np.std(xr[ok]) == 0.0 or np.std(yr[ok]) == 0.0:
        return None
    return float(np.corrcoef(xr[ok], yr[ok])[0, 1])


def make_plots(out_dir: Path, t: np.ndarray, kappa_pytdi_t: np.ndarray, kappa_static_t: np.ndarray, freqs: np.ndarray, kappa_pytdi_f: np.ndarray) -> list[str]:
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, np.real(kappa_pytdi_t), lw=0.8, label="pyTDI Sagnac kappa")
    ax.plot(t[: len(kappa_static_t)], np.real(kappa_static_t), lw=0.8, alpha=0.7, label="static-link Sagnac kappa")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("kappa [Hz-like units]")
    ax.set_title("Kappa time series at injected sky direction")
    ax.legend(loc="best")
    fig.tight_layout()
    path = plot_dir / "pytdi_vs_static_kappa_time.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(str(path))

    positive = freqs > 0.0
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.loglog(freqs[positive], np.abs(kappa_pytdi_f[positive]) ** 2)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|kappa|^2")
    ax.set_title("pyTDI-Sagnac kappa spectrum")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "pytdi_kappa_spectrum.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute example kappa from pyTDI Sagnac alpha/beta/gamma.")
    parser.add_argument("--config", default="test-run/config.yaml")
    parser.add_argument("--catalog", default="test-run/catalog.csv")
    parser.add_argument("--runs-root", default="test-run/simulations")
    parser.add_argument("--source-id", default="smbhb_m1e6")
    parser.add_argument("--source", choices=("noisy", "gw_only"), default="noisy")
    parser.add_argument("--generation", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--eta-combinations",
        action="store_true",
        help="Use pyTDI ALPHA*_ETA/BETA*_ETA/GAMMA*_ETA instead of ALPHA*/BETA*/GAMMA*.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200_000,
        help="Center-crop the pyTDI Sagnac time series before kappa/CSV output. Use 0 for full length.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    catalog = pd.read_csv(args.catalog)
    selected = catalog[catalog["source_id"].astype(str) == str(args.source_id)]
    if selected.empty:
        raise ValueError(f"No catalog row found for source_id={args.source_id!r}.")
    row = selected.iloc[0]
    lisa_config = build_lisa_config(config, row)
    paths = LisaLinkSimulationPaths.from_work_dir(Path(args.runs_root) / str(args.source_id))
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.runs_root) / str(args.source_id) / "pytdi_sagnac_kappa_example"
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = LISAGeometry.equilateral(LISA_ARM_M / C_M_PER_S)
    t, freqs, alpha_f, beta_f, gamma_f = evaluate_pytdi_sagnac(
        paths=paths,
        config=lisa_config,
        source=args.source,
        generation=args.generation,
        max_samples=None if int(args.max_samples) <= 0 else int(args.max_samples),
        eta_combinations=bool(args.eta_combinations),
    )
    kappa_pytdi_f = coronagraphic_kappa(
        freqs,
        float(row["dec"]),
        float(row["ra"]),
        geometry,
        alpha_f,
        beta_f,
        gamma_f,
    )
    kappa_pytdi_t = _freq_to_time_plus(kappa_pytdi_f)

    static_alpha_f, static_beta_f, static_gamma_f = static_sagnac_from_link_csv(
        paths=paths,
        source=args.source,
        t_reference=t,
        geometry=geometry,
    )
    n = min(len(kappa_pytdi_f), len(static_alpha_f))
    kappa_static_f = coronagraphic_kappa(
        freqs[:n],
        float(row["dec"]),
        float(row["ra"]),
        geometry,
        static_alpha_f[:n],
        static_beta_f[:n],
        static_gamma_f[:n],
    )
    kappa_static_t = _freq_to_time_plus(kappa_static_f)

    out_csv = output_dir / "pytdi_sagnac_kappa_timeseries.csv"
    pd.DataFrame(
        {
            "t_waveform": t[:n],
            "kappa_pytdi_real": np.real(kappa_pytdi_t[:n]),
            "kappa_pytdi_imag": np.imag(kappa_pytdi_t[:n]),
            "kappa_static_real": np.real(kappa_static_t[:n]),
            "kappa_static_imag": np.imag(kappa_static_t[:n]),
        }
    ).to_csv(out_csv, index=False)

    plot_paths = [] if args.no_plots else make_plots(output_dir, t[:n], kappa_pytdi_t[:n], kappa_static_t[:n], freqs[:n], kappa_pytdi_f[:n])
    summary = {
        "config": asdict(lisa_config),
        "source_id": str(args.source_id),
        "source": args.source,
        "pytdi_sagnac_generation": int(args.generation),
        "pytdi_eta_combinations": bool(args.eta_combinations),
        "sky_ra": float(row["ra"]),
        "sky_dec": float(row["dec"]),
        "n_samples": int(n),
        "dt_s": float(np.median(np.diff(t[:n]))),
        "kappa_time_csv": str(out_csv),
        "plots": plot_paths,
        "pytdi_static_real_correlation": real_correlation(kappa_pytdi_t[:n], kappa_static_t[:n]),
        "pytdi_rms": float(np.sqrt(np.nanmean(np.real(kappa_pytdi_t[:n]) ** 2))),
        "static_rms": float(np.sqrt(np.nanmean(np.real(kappa_static_t[:n]) ** 2))),
        "note": (
            "The kappa projection uses the current static first-generation Sagnac response model. "
            "Use generation=1 for the closest convention check; generation=2 is exploratory."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
