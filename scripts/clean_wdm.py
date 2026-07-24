"""Clean WDM time-frequency transforms for LISA link-simulation products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


LINK_CHANNELS: tuple[str, ...] = ("12", "23", "31", "13", "32", "21")
XYZ_CHANNELS: tuple[str, ...] = ("X", "Y", "Z")
AET_CHANNELS: tuple[str, ...] = ("A", "E", "T")
CHANNEL_GROUPS: Mapping[str, tuple[str, ...]] = {
    "links": LINK_CHANNELS,
    "XYZ": XYZ_CHANNELS,
    "AET": AET_CHANNELS,
}


@dataclass(frozen=True)
class WDMResolution:
    label: str
    target_dt: float
    M: int


@dataclass(frozen=True)
class CleanWDMConfig:
    """Configuration for WDM post-processing."""

    tdi_suffix: str = "2"
    resolutions: Sequence[WDMResolution] = (
        WDMResolution("dt1_M256", 1.0, 256),
        WDMResolution("dt1_M512", 1.0, 512),
        WDMResolution("dt1_M1024", 1.0, 1024),
        WDMResolution("dt2_M2048", 2.0, 2048),
    )
    sources: Sequence[str] = ("noisy", "gw_only")
    groups: Sequence[str] = ("links", "XYZ", "AET")
    channels: Mapping[str, Sequence[str]] | None = None
    low_freq_hz: float = 1e-5
    high_freq_hz: float = 1e-1
    wdm_k: int = 32
    beta_order: int = 6
    precision: int = 6
    backend: str = "numba"
    signal_threshold_fraction: float = 1e-3
    offsource_guard_seconds: float = 500.0
    min_noise_floor: float = 1e-300
    save_power: bool = True
    save_normalized_power: bool = True


@dataclass(frozen=True)
class CleanWDMPaths:
    work_dir: Path
    output_dir: Path
    plot_dir: Path
    noisy_links_csv: Path
    gw_links_csv: Path
    noisy_tdi_csv: Path
    gw_tdi_csv: Path
    stats_csv: Path
    maps_npz: Path

    @classmethod
    def from_work_dir(cls, work_dir: str | Path, *, prefix: str = "clean_wdm") -> "CleanWDMPaths":
        root = Path(work_dir)
        output_dir = root / "outputs"
        plot_dir = output_dir / "plots"
        return cls(
            work_dir=root,
            output_dir=output_dir,
            plot_dir=plot_dir,
            noisy_links_csv=output_dir / "six_links_noisy.csv",
            gw_links_csv=output_dir / "six_links_gw_only.csv",
            noisy_tdi_csv=output_dir / "tdi_michelson_XYZ_AET.csv",
            gw_tdi_csv=output_dir / "tdi_gw_only_XYZ_AET.csv",
            stats_csv=output_dir / f"{prefix}_stats.csv",
            maps_npz=output_dir / f"{prefix}_maps.npz",
        )

    def make_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CleanWDMProducts:
    paths: CleanWDMPaths
    stats: pd.DataFrame
    diagnostics: Mapping[str, float | int | str]


def require_wdm_dependencies():
    try:
        from scipy.signal import resample_poly
        from wdm_wavelet.wdm import WDM
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError("Install scipy and wdm-wavelet to run clean WDM transforms.") from exc
    return resample_poly, WDM


def load_simulation_channels(
    paths: CleanWDMPaths,
    *,
    tdi_suffix: str,
) -> tuple[np.ndarray, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Load aligned noisy and GW-only links, XYZ, and AET channels."""

    link_cols_noisy = ["t_lisa", *[f"noisy_{ch}" for ch in LINK_CHANNELS]]
    link_cols_gw = ["t_lisa", "is_science", *[f"gw_{ch}_hz" for ch in LINK_CHANNELS]]
    tdi_cols_noisy = ["t_lisa", *[f"{ch}{tdi_suffix}" for ch in (*XYZ_CHANNELS, *AET_CHANNELS)]]
    tdi_cols_gw = ["t_lisa", *[f"{ch}{tdi_suffix}_gw_hz" for ch in (*XYZ_CHANNELS, *AET_CHANNELS)]]

    links_noisy = pd.read_csv(paths.noisy_links_csv, usecols=link_cols_noisy)
    links_gw = pd.read_csv(paths.gw_links_csv, usecols=link_cols_gw)
    links_gw = links_gw[links_gw["is_science"].astype(bool)].reset_index(drop=True)
    tdi_noisy = pd.read_csv(paths.noisy_tdi_csv, usecols=tdi_cols_noisy)
    tdi_gw = pd.read_csv(paths.gw_tdi_csv, usecols=tdi_cols_gw)

    t_lisa = links_noisy["t_lisa"].to_numpy(dtype=float)
    for name, frame in (("links_gw", links_gw), ("tdi_noisy", tdi_noisy), ("tdi_gw", tdi_gw)):
        if len(frame) != len(t_lisa):
            raise ValueError(f"{name} length {len(frame)} does not match noisy links length {len(t_lisa)}.")
        if not np.allclose(frame["t_lisa"].to_numpy(dtype=float), t_lisa, rtol=0.0, atol=1e-6):
            raise ValueError(f"{name} is not aligned to the noisy link time grid.")

    data = {
        "noisy": {
            "links": {ch: links_noisy[f"noisy_{ch}"].to_numpy(dtype=float) for ch in LINK_CHANNELS},
            "XYZ": {ch: tdi_noisy[f"{ch}{tdi_suffix}"].to_numpy(dtype=float) for ch in XYZ_CHANNELS},
            "AET": {ch: tdi_noisy[f"{ch}{tdi_suffix}"].to_numpy(dtype=float) for ch in AET_CHANNELS},
        },
        "gw_only": {
            "links": {ch: links_gw[f"gw_{ch}_hz"].to_numpy(dtype=float) for ch in LINK_CHANNELS},
            "XYZ": {ch: tdi_gw[f"{ch}{tdi_suffix}_gw_hz"].to_numpy(dtype=float) for ch in XYZ_CHANNELS},
            "AET": {ch: tdi_gw[f"{ch}{tdi_suffix}_gw_hz"].to_numpy(dtype=float) for ch in AET_CHANNELS},
        },
    }
    return t_lisa, data


def locate_signal_window(
    t_lisa: np.ndarray,
    gw_aet: Mapping[str, np.ndarray],
    *,
    threshold_fraction: float,
) -> dict[str, float]:
    power = np.zeros_like(t_lisa, dtype=float)
    for channel in AET_CHANNELS:
        power += np.asarray(gw_aet[channel], dtype=float) ** 2
    peak_idx = int(np.argmax(power))
    peak = float(power[peak_idx])
    if peak <= 0.0:
        raise ValueError("GW-only AET power is zero; cannot locate signal window.")
    keep = np.flatnonzero(power >= threshold_fraction * peak)
    return {
        "signal_peak_lisa": float(t_lisa[peak_idx]),
        "signal_peak_rel": float(t_lisa[peak_idx] - t_lisa[0]),
        "signal_window_start_lisa": float(t_lisa[keep[0]]),
        "signal_window_stop_lisa": float(t_lisa[keep[-1]]),
        "signal_window_start_rel": float(t_lisa[keep[0]] - t_lisa[0]),
        "signal_window_stop_rel": float(t_lisa[keep[-1]] - t_lisa[0]),
    }


def downsample_to_target_dt(
    x: np.ndarray,
    native_dt: float,
    target_dt: float,
    resample_poly,
) -> tuple[np.ndarray, float, int]:
    factor = int(round(target_dt / native_dt))
    if factor < 1 or not np.isclose(factor * native_dt, target_dt, rtol=0.0, atol=1e-6):
        raise ValueError("target_dt must be an integer multiple of the native sample spacing.")
    if factor == 1:
        return np.asarray(x, dtype=float), native_dt, factor
    return resample_poly(np.asarray(x, dtype=float), up=1, down=factor), factor * native_dt, factor


def pad_for_wdm(x: np.ndarray, wdm: Any) -> tuple[np.ndarray, int]:
    min_len = 2 * int(wdm.m_H) + 1
    if len(x) >= min_len:
        return x, 0
    pad_right = min_len - len(x)
    return np.pad(x, (0, pad_right), mode="constant"), pad_right


def compute_wdm_transform(
    wdm: Any,
    x: np.ndarray,
    *,
    sample_rate: float,
    t0: float,
    original_t1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, int]:
    x_pad, pad_right = pad_for_wdm(np.asarray(x, dtype=float), wdm)
    tf = wdm.t2w(x_pad, sample_rate=sample_rate, t0=t0)
    coeffs = np.asarray(tf.data, dtype=np.complex128)
    freqs = np.arange(tf.n_freq, dtype=float) * tf.df
    times_lisa = tf.t0 + (np.arange(tf.n_time, dtype=float) + 0.5) * tf.dt
    valid_time = times_lisa <= original_t1
    return coeffs[:, valid_time], times_lisa[valid_time], freqs, tf, pad_right


def compute_wdm_power(
    wdm: Any,
    x: np.ndarray,
    *,
    sample_rate: float,
    t0: float,
    original_t1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, int]:
    coeffs, times_lisa, freqs, tf, pad_right = compute_wdm_transform(
        wdm,
        x,
        sample_rate=sample_rate,
        t0=t0,
        original_t1=original_t1,
    )
    return np.abs(coeffs) ** 2, times_lisa, freqs, tf, pad_right


def selected_channels(config: CleanWDMConfig, group: str) -> Sequence[str]:
    if group not in CHANNEL_GROUPS:
        raise ValueError(f"Unknown channel group {group!r}.")
    if config.channels is not None and group in config.channels:
        return config.channels[group]
    return CHANNEL_GROUPS[group]


def summarize_wdm_map(
    *,
    source: str,
    group: str,
    channel: str,
    resolution: WDMResolution,
    power: np.ndarray,
    normalized_power: np.ndarray,
    times_lisa: np.ndarray,
    freqs_hz: np.ndarray,
    tf: Any,
    signal: Mapping[str, float],
    pad_right: int,
    downsample_factor: int,
    config: CleanWDMConfig,
) -> dict[str, float | int | str | bool]:
    signal_t = (
        (times_lisa >= signal["signal_window_start_lisa"])
        & (times_lisa <= signal["signal_window_stop_lisa"])
    )
    offsource_t = (
        (times_lisa < signal["signal_window_start_lisa"] - config.offsource_guard_seconds)
        | (times_lisa > signal["signal_window_stop_lisa"] + config.offsource_guard_seconds)
    )
    band_f = (freqs_hz >= config.low_freq_hz) & (freqs_hz <= config.high_freq_hz)
    signal_band = normalized_power[np.ix_(band_f, signal_t)]
    offsource_band = normalized_power[np.ix_(band_f, offsource_t)]
    band_freqs = freqs_hz[band_f]
    band_power = power[band_f, :]
    max_idx = np.unravel_index(int(np.nanargmax(band_power)), band_power.shape) if band_power.size else (0, 0)
    return {
        "source": source,
        "group": group,
        "channel": channel,
        "resolution": resolution.label,
        "target_dt_s": float(resolution.target_dt),
        "wdm_M": int(resolution.M),
        "wdm_df_hz": float(tf.df),
        "wdm_dt_s": float(tf.dt),
        "first_positive_freq_hz": float(tf.df),
        "reaches_low_freq_cut": bool(tf.df <= config.low_freq_hz),
        "n_time_bins": int(len(times_lisa)),
        "n_freq_bins": int(len(freqs_hz)),
        "n_band_freq_bins": int(np.sum(band_f)),
        "n_signal_time_bins": int(np.sum(signal_t)),
        "n_offsource_time_bins": int(np.sum(offsource_t)),
        "downsample_factor": int(downsample_factor),
        "pad_right_samples": int(pad_right),
        "signal_norm_power_mean": float(np.nanmean(signal_band)) if signal_band.size else np.nan,
        "signal_norm_power_median": float(np.nanmedian(signal_band)) if signal_band.size else np.nan,
        "signal_norm_power_max": float(np.nanmax(signal_band)) if signal_band.size else np.nan,
        "offsource_norm_power_mean": float(np.nanmean(offsource_band)) if offsource_band.size else np.nan,
        "offsource_norm_power_median": float(np.nanmedian(offsource_band)) if offsource_band.size else np.nan,
        "signal_over_offsource_mean": (
            float(np.nanmean(signal_band) / np.nanmean(offsource_band))
            if signal_band.size and offsource_band.size
            else np.nan
        ),
        "band_raw_power_max": float(np.nanmax(band_power)) if band_power.size else np.nan,
        "band_raw_power_peak_time_rel": (
            float(times_lisa[max_idx[1]] - times_lisa[0]) if np.any(band_f) else np.nan
        ),
        "band_raw_power_peak_freq_hz": float(band_freqs[max_idx[0]]) if np.any(band_f) else np.nan,
    }


def run_clean_wdm(
    work_dir: str | Path,
    *,
    config: CleanWDMConfig | None = None,
    output_prefix: str = "clean_wdm",
) -> CleanWDMProducts:
    """Run clean WDM maps and per-map summary statistics."""

    if config is None:
        config = CleanWDMConfig()
    resample_poly, WDM = require_wdm_dependencies()
    paths = CleanWDMPaths.from_work_dir(work_dir, prefix=output_prefix)
    paths.make_dirs()

    t_lisa, data = load_simulation_channels(paths, tdi_suffix=config.tdi_suffix)
    native_dt = float(np.median(np.diff(t_lisa)))
    signal = locate_signal_window(
        t_lisa,
        data["gw_only"]["AET"],
        threshold_fraction=config.signal_threshold_fraction,
    )

    rows: list[dict[str, float | int | str | bool]] = []
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
    }

    for resolution in config.resolutions:
        wdm = WDM(
            M=resolution.M,
            K=config.wdm_k,
            beta_order=config.beta_order,
            precision=config.precision,
            backend=config.backend,
        )
        for source in config.sources:
            for group in config.groups:
                for channel in selected_channels(config, group):
                    x_ds, dt_ds, downsample_factor = downsample_to_target_dt(
                        data[source][group][channel],
                        native_dt,
                        resolution.target_dt,
                        resample_poly,
                    )
                    power, times_wdm, freqs_wdm, tf, pad_right = compute_wdm_power(
                        wdm,
                        x_ds,
                        sample_rate=1.0 / dt_ds,
                        t0=float(t_lisa[0]),
                        original_t1=float(t_lisa[-1]),
                    )
                    offsource_t = (
                        (times_wdm < signal["signal_window_start_lisa"] - config.offsource_guard_seconds)
                        | (times_wdm > signal["signal_window_stop_lisa"] + config.offsource_guard_seconds)
                    )
                    noise_floor = np.nanmedian(power[:, offsource_t], axis=1)
                    noise_floor = np.where(
                        np.isfinite(noise_floor) & (noise_floor > 0.0),
                        noise_floor,
                        config.min_noise_floor,
                    )
                    normalized_power = power / noise_floor[:, None]
                    rows.append(
                        summarize_wdm_map(
                            source=source,
                            group=group,
                            channel=channel,
                            resolution=resolution,
                            power=power,
                            normalized_power=normalized_power,
                            times_lisa=times_wdm,
                            freqs_hz=freqs_wdm,
                            tf=tf,
                            signal=signal,
                            pad_right=pad_right,
                            downsample_factor=downsample_factor,
                            config=config,
                        )
                    )

                    prefix = f"{source}_{group}_{channel}_{resolution.label}"
                    if config.save_power:
                        payload[f"{prefix}_power"] = power.astype(np.float32)
                    if config.save_normalized_power:
                        payload[f"{prefix}_normalized_power"] = normalized_power.astype(np.float32)
                    payload[f"{prefix}_noise_floor"] = noise_floor.astype(np.float32)
                    payload[f"{prefix}_times_rel"] = (times_wdm - t_lisa[0]).astype(np.float64)
                    payload[f"{prefix}_freqs_hz"] = freqs_wdm.astype(np.float64)

    stats = pd.DataFrame(rows)
    stats.to_csv(paths.stats_csv, index=False)
    np.savez_compressed(paths.maps_npz, **payload)

    diagnostics = {
        "native_samples": int(len(t_lisa)),
        "native_dt_s": native_dt,
        "native_duration_s": float(t_lisa[-1] - t_lisa[0]),
        "n_rows": int(len(stats)),
        **signal,
    }
    return CleanWDMProducts(paths=paths, stats=stats, diagnostics=diagnostics)


__all__ = [
    "AET_CHANNELS",
    "CHANNEL_GROUPS",
    "CleanWDMConfig",
    "CleanWDMPaths",
    "CleanWDMProducts",
    "LINK_CHANNELS",
    "WDMResolution",
    "XYZ_CHANNELS",
    "load_simulation_channels",
    "locate_signal_window",
    "compute_wdm_transform",
    "run_clean_wdm",
]
