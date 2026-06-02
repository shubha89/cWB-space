"""Off-source PSD estimation and time-domain whitening for A/E channels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PSDConfig


def _require_scipy_signal():
    try:
        from scipy.signal import welch
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        raise ModuleNotFoundError("PSD estimation requires scipy.") from exc
    return welch


def tdi_columns(tdi_generation: int = 2) -> tuple[str, str]:
    return f"A{tdi_generation}", f"E{tdi_generation}"


def load_signal_window(work_dir: str | Path, *, tdi_generation: int = 2, threshold_fraction: float = 1.0e-3) -> tuple[float, float, float]:
    """Find signal start, stop, and peak in LISA time for a simulated work dir."""

    work = Path(work_dir)
    metadata_file = work / "outputs" / "1e5_plot_and_wdm_metadata.json"
    if metadata_file.exists():
        raw = json.loads(metadata_file.read_text(encoding="utf-8"))
        return float(raw["signal_start_lisa"]), float(raw["signal_stop_lisa"]), float(raw["signal_peak_lisa"])

    maps_file = work / "outputs" / "multires_wdm_links_xyz_aet_maps.npz"
    if maps_file.exists():
        data = np.load(maps_file)
        if "signal_window_lisa" in data and "signal_peak_lisa" in data:
            window = np.asarray(data["signal_window_lisa"], dtype=float)
            return float(window[0]), float(window[1]), float(data["signal_peak_lisa"])

    gw_tdi = work / "outputs" / "tdi_gw_only_XYZ_AET.csv"
    if not gw_tdi.exists():
        raise FileNotFoundError(f"Cannot locate signal metadata or GW-only TDI file under {work}.")
    a_col = f"A{tdi_generation}_gw_hz"
    e_col = f"E{tdi_generation}_gw_hz"
    df = pd.read_csv(gw_tdi, usecols=["t_lisa", a_col, e_col])
    power = df[a_col].to_numpy(dtype=float) ** 2 + df[e_col].to_numpy(dtype=float) ** 2
    peak_idx = int(np.nanargmax(power))
    threshold = threshold_fraction * float(np.nanmax(power))
    idx = np.flatnonzero(power >= threshold)
    if len(idx) == 0:
        raise ValueError(f"Could not locate signal interval from {gw_tdi}.")
    t_lisa = df["t_lisa"].to_numpy(dtype=float)
    return float(t_lisa[idx[0]]), float(t_lisa[idx[-1]]), float(t_lisa[peak_idx])


def estimate_noise_psd_from_work_dir(
    work_dir: str | Path,
    *,
    config: PSDConfig,
    tdi_generation: int = 2,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Estimate A/E noise PSD from off-source samples in one simulation."""

    welch = _require_scipy_signal()
    work = Path(work_dir)
    tdi_csv = work / "outputs" / "tdi_michelson_XYZ_AET.csv"
    if not tdi_csv.exists():
        raise FileNotFoundError(f"Missing noisy TDI CSV: {tdi_csv}")

    a_col, e_col = tdi_columns(tdi_generation)
    df = pd.read_csv(tdi_csv, usecols=["t_lisa", a_col, e_col])
    t_lisa = df["t_lisa"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t_lisa)))
    signal_start, signal_stop, _ = load_signal_window(work, tdi_generation=tdi_generation)
    offsource = (
        (t_lisa < signal_start - config.offsource_guard_seconds)
        | (t_lisa > signal_stop + config.offsource_guard_seconds)
    )
    if int(np.sum(offsource)) < 8:
        raise ValueError("Too few off-source samples for PSD estimation.")

    nperseg = min(int(config.nperseg), int(np.sum(offsource)))
    fs = 1.0 / dt
    freqs, psd_a = welch(df.loc[offsource, a_col].to_numpy(dtype=float), fs=fs, nperseg=nperseg)
    freqs_e, psd_e = welch(df.loc[offsource, e_col].to_numpy(dtype=float), fs=fs, nperseg=nperseg)
    if not np.allclose(freqs, freqs_e):
        raise ValueError("A and E Welch frequency grids differ.")
    psd_a = np.maximum(psd_a, config.min_psd)
    psd_e = np.maximum(psd_e, config.min_psd)
    out = pd.DataFrame({"frequency_hz": freqs, "S_A": psd_a, "S_E": psd_e})

    target = Path(output_csv or config.output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return out


def load_noise_psd(path: str | Path) -> pd.DataFrame:
    psd = pd.read_csv(path)
    required = {"frequency_hz", "S_A", "S_E"}
    missing = required.difference(psd.columns)
    if missing:
        raise ValueError(f"PSD CSV missing columns: {sorted(missing)}")
    return psd


def whiten_time_series(x: np.ndarray, dt: float, freqs: np.ndarray, psd: np.ndarray, *, min_psd: float = 1.0e-300) -> np.ndarray:
    """Whiten a real time series with an interpolated one-sided PSD."""

    x = np.asarray(x, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    centered = x - np.nanmedian(x)
    fft_freqs = np.fft.rfftfreq(len(centered), d=dt)
    interp_psd = np.interp(fft_freqs, freqs, psd, left=psd[0], right=psd[-1])
    interp_psd = np.maximum(interp_psd, min_psd)
    spectrum = np.fft.rfft(centered)
    whitened = np.fft.irfft(spectrum / np.sqrt(interp_psd), n=len(centered))
    scale = np.nanstd(whitened)
    if np.isfinite(scale) and scale > 0.0:
        whitened = whitened / scale
    return whitened.astype(float)
