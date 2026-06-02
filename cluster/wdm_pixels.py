"""Build fixed-size A/E WDM patches for clustering."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import PSDConfig, WDMMapConfig
from .psd import load_signal_window, tdi_columns, whiten_time_series


def _require_wdm_dependencies():
    try:
        from scipy.ndimage import zoom
        from scipy.signal import resample_poly
        from wdm_wavelet.wdm import WDM
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        raise ModuleNotFoundError(
            "WDM patch extraction requires scipy and wdm-wavelet. "
            "Activate the lisa-tf environment from env.yml."
        ) from exc
    return zoom, resample_poly, WDM


def _downsample(x: np.ndarray, dt: float, target_dt: float, resample_poly) -> tuple[np.ndarray, float]:
    factor = int(round(target_dt / dt))
    if factor <= 1:
        return np.asarray(x, dtype=float), dt
    if not np.isclose(factor * dt, target_dt, rtol=1.0e-6, atol=1.0e-9):
        raise ValueError("target_dt_s must be an integer multiple of the TDI sample spacing.")
    return resample_poly(np.asarray(x, dtype=float), up=1, down=factor), factor * dt


def _pad_for_wdm(x: np.ndarray, wdm) -> tuple[np.ndarray, int]:
    min_len = max(len(x), 2 * int(wdm.m_H) + 1)
    if len(x) >= min_len:
        return x, 0
    pad_right = min_len - len(x)
    return np.pad(x, (0, pad_right), mode="constant"), pad_right


def _wdm_power(x: np.ndarray, dt: float, *, t0: float, config: WDMMapConfig):
    _, _, WDM = _require_wdm_dependencies()
    wdm = WDM(M=config.M, K=config.K, beta_order=config.beta_order, precision=config.precision, backend=config.backend)
    x_pad, _ = _pad_for_wdm(x, wdm)
    tf = wdm.t2w(x_pad, sample_rate=1.0 / dt, t0=t0)
    power = np.abs(tf.data) ** 2
    freqs = np.arange(tf.n_freq, dtype=float) * tf.df
    times = tf.t0 + (np.arange(tf.n_time, dtype=float) + 0.5) * tf.dt
    return power, times, freqs


def _resize_map(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    zoom, _, _ = _require_wdm_dependencies()
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot resize an empty WDM patch.")
    factors = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    return zoom(arr, factors, order=1)


def _normalize_patch(power: np.ndarray, *, config: WDMMapConfig) -> np.ndarray:
    x = np.maximum(np.asarray(power, dtype=float), 0.0)
    if config.log_scale:
        x = np.log1p(x)
    median = np.nanmedian(x)
    q25, q75 = np.nanpercentile(x, [25, 75])
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 0.0:
        scale = np.nanstd(x)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    x = (x - median) / scale
    x = np.clip(x, -config.robust_clip, config.robust_clip)
    return np.nan_to_num(x, nan=0.0, posinf=config.robust_clip, neginf=-config.robust_clip).astype(np.float32)


def _crop_to_signal(power: np.ndarray, times: np.ndarray, freqs: np.ndarray, peak_lisa: float, config: WDMMapConfig) -> np.ndarray:
    freq_mask = (freqs >= config.freq_min_hz) & (freqs <= config.freq_max_hz)
    half = 0.5 * config.patch_duration_seconds
    time_mask = (times >= peak_lisa - half) & (times <= peak_lisa + half)
    if not np.any(freq_mask):
        raise ValueError("No WDM frequency bins remain after freq_min_hz/freq_max_hz selection.")
    if not np.any(time_mask):
        nearest = int(np.argmin(np.abs(times - peak_lisa)))
        left = max(0, nearest - max(1, config.patch_time_bins // 2))
        right = min(len(times), left + max(2, config.patch_time_bins))
        time_mask = np.zeros_like(times, dtype=bool)
        time_mask[left:right] = True
    cropped = power[np.ix_(freq_mask, time_mask)]
    return _resize_map(cropped, (config.patch_freq_bins, config.patch_time_bins))


def build_patch_from_work_dir(
    work_dir: str | Path,
    *,
    psd: pd.DataFrame,
    wdm_config: WDMMapConfig,
    psd_config: PSDConfig,
    tdi_generation: int = 2,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Build one two-channel A/E WDM patch from a completed simulation."""

    _, resample_poly, _ = _require_wdm_dependencies()
    work = Path(work_dir)
    tdi_csv = work / "outputs" / "tdi_michelson_XYZ_AET.csv"
    if not tdi_csv.exists():
        raise FileNotFoundError(f"Missing noisy TDI CSV: {tdi_csv}")

    a_col, e_col = tdi_columns(tdi_generation)
    df = pd.read_csv(tdi_csv, usecols=["t_lisa", a_col, e_col])
    t_lisa = df["t_lisa"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t_lisa)))
    _, _, signal_peak = load_signal_window(work, tdi_generation=tdi_generation)

    psd_freqs = psd["frequency_hz"].to_numpy(dtype=float)
    channel_patches = []
    for channel, column, psd_column in (("A", a_col, "S_A"), ("E", e_col, "S_E")):
        white = whiten_time_series(
            df[column].to_numpy(dtype=float),
            dt,
            psd_freqs,
            psd[psd_column].to_numpy(dtype=float),
            min_psd=psd_config.min_psd,
        )
        white_ds, dt_wdm = _downsample(white, dt, wdm_config.target_dt_s, resample_poly)
        power, times, freqs = _wdm_power(white_ds, dt_wdm, t0=float(t_lisa[0]), config=wdm_config)
        patch = _crop_to_signal(power, times, freqs, signal_peak, wdm_config)
        channel_patches.append(_normalize_patch(patch, config=wdm_config))

    tensor = np.stack(channel_patches, axis=0).astype(np.float32)
    metadata = {
        "work_dir": str(work),
        "source_id": work.name,
        "signal_peak_lisa": float(signal_peak),
        "dt_s": float(dt),
        "target_dt_s": float(wdm_config.target_dt_s),
        "wdm_M": int(wdm_config.M),
        "patch_freq_bins": int(wdm_config.patch_freq_bins),
        "patch_time_bins": int(wdm_config.patch_time_bins),
    }
    return tensor, metadata


def discover_work_dirs(runs_root: str | Path) -> list[Path]:
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {root}")
    return sorted(path for path in root.iterdir() if (path / "outputs" / "tdi_michelson_XYZ_AET.csv").exists())


def build_dataset(
    work_dirs: Sequence[str | Path],
    *,
    psd: pd.DataFrame,
    wdm_config: WDMMapConfig,
    psd_config: PSDConfig,
    output_npz: str | Path,
    metadata_csv: str | Path,
    catalog: pd.DataFrame | None = None,
    tdi_generation: int = 2,
    cleanup_after_build: bool = False,
) -> tuple[Path, Path]:
    """Build and save the A/E WDM patch dataset."""

    if cleanup_after_build:
        from .cleanup import cleanup_work_dir

    patches = []
    rows: list[dict[str, float | str]] = []
    catalog_by_source: Mapping[str, dict] = {}
    if catalog is not None and "source_id" in catalog.columns:
        catalog_by_source = {str(row["source_id"]): dict(row) for _, row in catalog.iterrows()}

    for work_dir in work_dirs:
        patch, metadata = build_patch_from_work_dir(
            work_dir,
            psd=psd,
            wdm_config=wdm_config,
            psd_config=psd_config,
            tdi_generation=tdi_generation,
        )
        source_id = str(metadata["source_id"])
        if source_id in catalog_by_source:
            metadata.update(catalog_by_source[source_id])
        patches.append(patch)
        rows.append(metadata)
        if cleanup_after_build:
            cleanup_work_dir(work_dir, apply=True, require_diagnostics=True)

    if not patches:
        raise ValueError("No WDM patches were built.")

    output_path = Path(output_npz)
    metadata_path = Path(metadata_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.stack(patches, axis=0).astype(np.float32)
    source_ids = np.asarray([str(row["source_id"]) for row in rows])
    np.savez_compressed(output_path, x=x, source_id=source_ids)
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return output_path, metadata_path


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "x" not in data:
        raise ValueError(f"Dataset NPZ {path} does not contain array 'x'.")
    source_ids = np.asarray(data["source_id"]) if "source_id" in data else np.arange(len(data["x"])).astype(str)
    return np.asarray(data["x"], dtype=np.float32), source_ids
