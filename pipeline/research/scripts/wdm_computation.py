"""Compute WDM maps and T-channel weighted A/E products."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .plotting_utils import (
    _plot_multires_grid,
    _plot_t_weighted_grid,
    _save_and_plot_best_pixels,
    _to_float32_npz,
)



LINK_LABELS: tuple[str, ...] = ("12", "23", "31", "13", "32", "21")
XYZ_LABELS: tuple[str, ...] = ("X", "Y", "Z")
AET_LABELS: tuple[str, ...] = ("A", "E", "T")
GROUP_CHANNELS: Mapping[str, tuple[str, ...]] = {
    "links": LINK_LABELS,
    "XYZ": XYZ_LABELS,
    "AET": AET_LABELS,
}


@dataclass(frozen=True)
class WDMResolution:
    """One WDM resolution setting."""

    label: str
    target_dt: float
    M: int


DEFAULT_WDM_RESOLUTIONS: tuple[WDMResolution, ...] = (
    WDMResolution("dt1_M64", 1.0, 64),
    WDMResolution("dt1_M128", 1.0, 128),
    WDMResolution("dt1_M256", 1.0, 256),
    WDMResolution("dt1_M512", 1.0, 512),
    WDMResolution("dt1_M1024", 1.0, 1024),
    WDMResolution("dt1_M2048", 1.0, 2048),
    WDMResolution("dt1_M4096", 1.0, 4096),
    WDMResolution("dt2_M1024", 2.0, 1024),
    WDMResolution("dt2_M2048", 2.0, 2048),
    WDMResolution("dt2_M4096", 2.0, 4096),
    WDMResolution("dt5_M2048", 5.0, 2048),
    WDMResolution("dt5_M4096", 5.0, 4096),
    WDMResolution("dt10_M2048", 10.0, 2048),
    WDMResolution("dt10_M4096", 10.0, 4096),
    WDMResolution("dt10_M8192", 10.0, 8192),
)


@dataclass(frozen=True)
class TDIWDMPipelineConfig:
    """Settings for TDI/WDM post-processing."""

    tdi_suffix: str = "2"
    resolutions: Sequence[WDMResolution] = DEFAULT_WDM_RESOLUTIONS
    wdm_k: int = 32
    wdm_beta_order: int = 6
    wdm_precision: int = 6
    wdm_backend: str = "numba"
    low_freq_cut: float = 1e-5
    high_freq_cut: float = 1e-2
    signal_threshold_fraction: float = 1e-3
    offsource_guard_seconds: float = 500.0
    min_noise_floor: float = 1e-300
    save_raw_power: bool = True
    save_normalized_power: bool = True
    make_grid_plots: bool = True
    make_t_weighted_plots: bool = True
    make_best_pixel_plots: bool = True
    best_pixel_time_bins: int = 240
    best_pixel_freq_bins: int = 180


@dataclass(frozen=True)
class TDIWDMPipelinePaths:
    """Filesystem layout for WDM post-processing products."""

    work_dir: Path
    output_dir: Path
    plot_dir: Path
    noisy_links_csv: Path
    gw_links_csv: Path
    noisy_tdi_csv: Path
    gw_tdi_csv: Path
    multires_stats_csv: Path
    multires_maps_npz: Path
    t_weighted_stats_csv: Path
    t_weighted_maps_npz: Path
    bestpixel_stats_csv: Path
    bestpixel_maps_npz: Path

    @classmethod
    def from_work_dir(cls, work_dir: str | Path) -> "TDIWDMPipelinePaths":
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
            multires_stats_csv=output_dir / "multires_wdm_links_xyz_aet_stats.csv",
            multires_maps_npz=output_dir / "multires_wdm_links_xyz_aet_maps.npz",
            t_weighted_stats_csv=output_dir / "multires_wdm_AE_Twhiten_stats.csv",
            t_weighted_maps_npz=output_dir / "multires_wdm_AE_Twhiten_maps.npz",
            bestpixel_stats_csv=output_dir / "multires_wdm_bestpixel_stats.csv",
            bestpixel_maps_npz=output_dir / "multires_wdm_bestpixel_maps.npz",
        )

    def make_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TDIWDMPipelineProducts:
    """Returned products from a completed TDI/WDM run."""

    paths: TDIWDMPipelinePaths
    wdm_summary: pd.DataFrame
    t_weighted_summary: pd.DataFrame
    bestpixel_summary: pd.DataFrame
    diagnostics: Mapping[str, float | int]
    wdm_products: Mapping[tuple[str, str, str, str], Mapping[str, Any]] = field(repr=False)
    t_weighted_products: Mapping[str, np.ndarray] = field(repr=False)


def _require_wdm_dependencies():
    try:
        from scipy.signal import resample_poly
        from wdm_wavelet.wdm import WDM
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        raise ModuleNotFoundError(
            "TDI/WDM post-processing requires scipy and wdm-wavelet. "
            "Install the project environment from env.yml, or run: pip install scipy wdm-wavelet"
        ) from exc
    return resample_poly, WDM


def _load_pipeline_products(
    paths: TDIWDMPipelinePaths,
    suffix: str,
) -> tuple[np.ndarray, dict[str, dict[str, dict[str, np.ndarray]]]]:
    links_noisy_cols = ["sample", "t_waveform", "t_lisa", *[f"noisy_{lab}" for lab in LINK_LABELS]]
    links_gw_cols = ["sample", "t_waveform", "t_lisa", "is_science", *[f"gw_{lab}_hz" for lab in LINK_LABELS]]
    tdi_noisy_cols = ["sample", "t_lisa", *[f"{ch}{suffix}" for ch in (*XYZ_LABELS, *AET_LABELS)]]
    tdi_gw_cols = ["sample", "t_lisa", *[f"{ch}{suffix}_gw_hz" for ch in (*XYZ_LABELS, *AET_LABELS)]]

    links_noisy = pd.read_csv(paths.noisy_links_csv, usecols=links_noisy_cols)
    links_gw = pd.read_csv(paths.gw_links_csv, usecols=links_gw_cols)
    links_gw = links_gw[links_gw["is_science"].astype(bool)].reset_index(drop=True)
    tdi_noisy = pd.read_csv(paths.noisy_tdi_csv, usecols=tdi_noisy_cols)
    tdi_gw = pd.read_csv(paths.gw_tdi_csv, usecols=tdi_gw_cols)

    t_lisa = links_noisy["t_lisa"].to_numpy(dtype=float)
    for name, df in (("links_gw", links_gw), ("tdi_noisy", tdi_noisy), ("tdi_gw", tdi_gw)):
        if len(df) != len(t_lisa):
            raise ValueError(f"{name} length {len(df)} does not match six-link data length {len(t_lisa)}.")
        if not np.allclose(df["t_lisa"].to_numpy(dtype=float), t_lisa, rtol=0.0, atol=1e-6):
            raise ValueError(f"{name} is not aligned on the same t_lisa grid.")

    data = {
        "noisy": {
            "links": {lab: links_noisy[f"noisy_{lab}"].to_numpy(dtype=float) for lab in LINK_LABELS},
            "XYZ": {ch: tdi_noisy[f"{ch}{suffix}"].to_numpy(dtype=float) for ch in XYZ_LABELS},
            "AET": {ch: tdi_noisy[f"{ch}{suffix}"].to_numpy(dtype=float) for ch in AET_LABELS},
        },
        "gw_only": {
            "links": {lab: links_gw[f"gw_{lab}_hz"].to_numpy(dtype=float) for lab in LINK_LABELS},
            "XYZ": {ch: tdi_gw[f"{ch}{suffix}_gw_hz"].to_numpy(dtype=float) for ch in XYZ_LABELS},
            "AET": {ch: tdi_gw[f"{ch}{suffix}_gw_hz"].to_numpy(dtype=float) for ch in AET_LABELS},
        },
    }
    return t_lisa, data


def _locate_signal(
    t_lisa: np.ndarray,
    data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    threshold_fraction: float,
) -> dict[str, float | tuple[float, float]]:
    gw_aet_power = sum(data["gw_only"]["AET"][ch] ** 2 for ch in AET_LABELS)
    signal_peak_idx = int(np.argmax(gw_aet_power))
    signal_t_lisa = float(t_lisa[signal_peak_idx])
    threshold = threshold_fraction * float(np.nanmax(gw_aet_power))
    signal_indices = np.flatnonzero(gw_aet_power >= threshold)
    if len(signal_indices) == 0:
        raise ValueError("Could not locate a GW signal interval from GW-only AET power.")
    signal_window_lisa = (float(t_lisa[signal_indices[0]]), float(t_lisa[signal_indices[-1]]))
    return {
        "signal_peak_lisa": signal_t_lisa,
        "signal_peak_rel": signal_t_lisa - float(t_lisa[0]),
        "signal_window_lisa": signal_window_lisa,
        "signal_window_rel": (signal_window_lisa[0] - float(t_lisa[0]), signal_window_lisa[1] - float(t_lisa[0])),
    }


def _downsample_for_wdm(x: np.ndarray, dt: float, target_dt: float, resample_poly) -> tuple[np.ndarray, float]:
    factor = int(round(target_dt / dt))
    if not np.isclose(factor * dt, target_dt):
        raise ValueError("target_dt must be an integer multiple of the native dt.")
    x = np.asarray(x, dtype=float)
    if factor <= 1:
        return x, dt
    return resample_poly(x, up=1, down=factor), factor * dt


def _pad_for_wdm_if_needed(x: np.ndarray, wdm: Any) -> tuple[np.ndarray, int]:
    min_len = max(len(x), 2 * int(wdm.m_H) + 1)
    if len(x) >= min_len:
        return x, 0
    pad_right = min_len - len(x)
    return np.pad(x, (0, pad_right), mode="constant"), pad_right


def _wdm_power_map(wdm: Any, x: np.ndarray, sample_rate: float, t0: float, original_t1: float):
    x_padded, pad_right = _pad_for_wdm_if_needed(x, wdm)
    tf = wdm.t2w(x_padded, sample_rate=sample_rate, t0=t0)
    power = np.abs(tf.data) ** 2
    freqs = np.arange(tf.n_freq, dtype=float) * tf.df
    times_lisa = tf.t0 + (np.arange(tf.n_time, dtype=float) + 0.5) * tf.dt

    valid_time = times_lisa <= original_t1
    power = power[:, valid_time]
    times_lisa = times_lisa[valid_time]
    return power, times_lisa, freqs, tf, pad_right


def _summarize_map(
    source_name: str,
    group: str,
    channel: str,
    resolution: WDMResolution,
    normalized_power: np.ndarray,
    times_lisa_wdm: np.ndarray,
    freqs_wdm: np.ndarray,
    tf: Any,
    pad_right: int,
    signal_window_lisa: tuple[float, float],
    config: TDIWDMPipelineConfig,
) -> dict[str, float | int | str | bool]:
    signal_tf_mask = (times_lisa_wdm >= signal_window_lisa[0]) & (times_lisa_wdm <= signal_window_lisa[1])
    offsource_tf_mask = (
        (times_lisa_wdm < signal_window_lisa[0] - config.offsource_guard_seconds)
        | (times_lisa_wdm > signal_window_lisa[1] + config.offsource_guard_seconds)
    )
    freq_tf_mask = (freqs_wdm >= config.low_freq_cut) & (freqs_wdm <= config.high_freq_cut)
    signal_band = normalized_power[np.ix_(freq_tf_mask, signal_tf_mask)]
    offsource_band = normalized_power[np.ix_(freq_tf_mask, offsource_tf_mask)]
    return {
        "source": source_name,
        "group": group,
        "channel": channel,
        "resolution": resolution.label,
        "target_dt_s": resolution.target_dt,
        "wdm_M": resolution.M,
        "wdm_df_hz": float(tf.df),
        "first_positive_freq_hz": float(tf.df),
        "reaches_1e-5_hz": bool(tf.df <= config.low_freq_cut),
        "wdm_dt_tf_s": float(tf.dt),
        "n_time_layers": int(len(times_lisa_wdm)),
        "n_frequency_layers": int(np.sum(freq_tf_mask)),
        "n_signal_time_layers": int(np.sum(signal_tf_mask)),
        "signal_tf_pixels": int(signal_band.size),
        "pad_right_samples": int(pad_right),
        "signal_norm_power_mean": float(np.nanmean(signal_band)) if signal_band.size else np.nan,
        "signal_norm_power_median": float(np.nanmedian(signal_band)) if signal_band.size else np.nan,
        "signal_norm_power_max": float(np.nanmax(signal_band)) if signal_band.size else np.nan,
        "offsource_norm_power_mean": float(np.nanmean(offsource_band)) if offsource_band.size else np.nan,
        "signal_over_offsource_mean": (
            float(np.nanmean(signal_band) / np.nanmean(offsource_band))
            if signal_band.size and offsource_band.size
            else np.nan
        ),
    }

def run_tdi_wdm_pipeline(
    work_dir: str | Path,
    *,
    config: TDIWDMPipelineConfig | None = None,
) -> TDIWDMPipelineProducts:
    """Run multi-resolution WDM and T-channel weighted A/E maps.

    Args:
        work_dir:
            Directory containing the ``outputs/`` folder created by
            ``run_lisa_link_simulation``.
        config:
            WDM and plotting settings.
    """

    if config is None:
        config = TDIWDMPipelineConfig()
    if not isinstance(config, TDIWDMPipelineConfig):
        raise TypeError("config must be a TDIWDMPipelineConfig instance.")

    resample_poly, WDM = _require_wdm_dependencies()
    paths = TDIWDMPipelinePaths.from_work_dir(work_dir)
    paths.make_dirs()

    t_lisa, product_data = _load_pipeline_products(paths, config.tdi_suffix)
    t_rel = t_lisa - t_lisa[0]
    dt = float(np.median(np.diff(t_lisa)))
    signal = _locate_signal(t_lisa, product_data, config.signal_threshold_fraction)
    signal_window_lisa = signal["signal_window_lisa"]
    if not isinstance(signal_window_lisa, tuple):
        raise RuntimeError("Internal signal window type mismatch.")

    wdm_products: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    summary_rows = []
    npz_payload: dict[str, np.ndarray] = {
        "signal_window_lisa": np.asarray(signal_window_lisa, dtype=float),
        "signal_window_rel": np.asarray(signal["signal_window_rel"], dtype=float),
        "signal_peak_lisa": np.asarray(signal["signal_peak_lisa"], dtype=float),
        "signal_peak_rel": np.asarray(signal["signal_peak_rel"], dtype=float),
        "resolution_labels": np.asarray([resolution.label for resolution in config.resolutions]),
        "resolution_target_dt_s": np.asarray([resolution.target_dt for resolution in config.resolutions], dtype=float),
        "resolution_M": np.asarray([resolution.M for resolution in config.resolutions], dtype=int),
    }

    for resolution in config.resolutions:
        wdm = WDM(
            M=resolution.M,
            K=config.wdm_k,
            beta_order=config.wdm_beta_order,
            precision=config.wdm_precision,
            backend=config.wdm_backend,
        )
        for source_name in ("noisy", "gw_only"):
            for group, channels in GROUP_CHANNELS.items():
                for ch in channels:
                    x_ds, dt_wdm = _downsample_for_wdm(
                        product_data[source_name][group][ch], dt, resolution.target_dt, resample_poly
                    )
                    power, times_lisa_wdm, freqs_wdm, tf, pad_right = _wdm_power_map(
                        wdm, x_ds, sample_rate=1.0 / dt_wdm, t0=t_lisa[0], original_t1=t_lisa[-1]
                    )
                    offsource_tf_mask = (
                        (times_lisa_wdm < signal_window_lisa[0] - config.offsource_guard_seconds)
                        | (times_lisa_wdm > signal_window_lisa[1] + config.offsource_guard_seconds)
                    )
                    noise_floor = np.nanmedian(power[:, offsource_tf_mask], axis=1)
                    noise_floor = np.where(
                        np.isfinite(noise_floor) & (noise_floor > 0), noise_floor, config.min_noise_floor
                    )
                    normalized_power = power / noise_floor[:, None]

                    key = (source_name, group, ch, resolution.label)
                    wdm_products[key] = {
                        "power": power,
                        "normalized_power": normalized_power,
                        "noise_floor": noise_floor,
                        "times_lisa": times_lisa_wdm,
                        "times_rel": times_lisa_wdm - t_lisa[0],
                        "freqs_hz": freqs_wdm,
                        "tf": tf,
                        "resolution": resolution,
                        "pad_right": pad_right,
                    }
                    summary_rows.append(
                        _summarize_map(
                            source_name,
                            group,
                            ch,
                            resolution,
                            normalized_power,
                            times_lisa_wdm,
                            freqs_wdm,
                            tf,
                            pad_right,
                            signal_window_lisa,
                            config,
                        )
                    )

                    prefix = f"{source_name}_{group}_{ch}_{resolution.label}"
                    if config.save_raw_power:
                        npz_payload[f"{prefix}_power"] = _to_float32_npz(power)
                    if config.save_normalized_power:
                        npz_payload[f"{prefix}_normalized_power"] = _to_float32_npz(normalized_power)
                    npz_payload[f"{prefix}_noise_floor"] = _to_float32_npz(noise_floor)
                    npz_payload[f"{prefix}_times_rel"] = (times_lisa_wdm - t_lisa[0]).astype(np.float64)
                    npz_payload[f"{prefix}_freqs_hz"] = freqs_wdm.astype(np.float64)

    wdm_summary_df = pd.DataFrame(summary_rows)
    wdm_summary_df.to_csv(paths.multires_stats_csv, index=False)
    np.savez_compressed(paths.multires_maps_npz, **npz_payload)

    t_weighted_rows = []
    t_weighted_payload: dict[str, np.ndarray] = {}
    for resolution in config.resolutions:
        t_prod = wdm_products[("noisy", "AET", "T", resolution.label)]
        t_noise_floor = np.asarray(t_prod["noise_floor"])
        for ch in ("A", "E"):
            prod = wdm_products[("noisy", "AET", ch, resolution.label)]
            t_weighted_power = np.asarray(prod["power"]) / np.maximum(t_noise_floor[:, None], config.min_noise_floor)
            times_lisa_wdm = np.asarray(prod["times_lisa"])
            freqs_wdm = np.asarray(prod["freqs_hz"])
            signal_tf_mask = (times_lisa_wdm >= signal_window_lisa[0]) & (times_lisa_wdm <= signal_window_lisa[1])
            offsource_tf_mask = (
                (times_lisa_wdm < signal_window_lisa[0] - config.offsource_guard_seconds)
                | (times_lisa_wdm > signal_window_lisa[1] + config.offsource_guard_seconds)
            )
            freq_tf_mask = (freqs_wdm >= config.low_freq_cut) & (freqs_wdm <= config.high_freq_cut)
            signal_band = t_weighted_power[np.ix_(freq_tf_mask, signal_tf_mask)]
            offsource_band = t_weighted_power[np.ix_(freq_tf_mask, offsource_tf_mask)]
            t_weighted_rows.append(
                {
                    "source": "noisy",
                    "group": "AET",
                    "channel": ch,
                    "weighting_proxy": "T",
                    "resolution": resolution.label,
                    "target_dt_s": resolution.target_dt,
                    "wdm_M": resolution.M,
                    "wdm_df_hz": float(prod["tf"].df),
                    "first_positive_freq_hz": float(prod["tf"].df),
                    "reaches_1e-5_hz": bool(prod["tf"].df <= config.low_freq_cut),
                    "wdm_dt_tf_s": float(prod["tf"].dt),
                    "n_frequency_layers": int(np.sum(freq_tf_mask)),
                    "n_signal_time_layers": int(np.sum(signal_tf_mask)),
                    "signal_tf_pixels": int(signal_band.size),
                    "t_weighted_signal_mean": float(np.nanmean(signal_band)) if signal_band.size else np.nan,
                    "t_weighted_signal_median": float(np.nanmedian(signal_band)) if signal_band.size else np.nan,
                    "t_weighted_signal_max": float(np.nanmax(signal_band)) if signal_band.size else np.nan,
                    "t_weighted_offsource_mean": float(np.nanmean(offsource_band)) if offsource_band.size else np.nan,
                    "t_weighted_signal_over_offsource_mean": (
                        float(np.nanmean(signal_band) / np.nanmean(offsource_band))
                        if signal_band.size and offsource_band.size
                        else np.nan
                    ),
                }
            )
            key = f"noisy_AET_{ch}_Twhiten_{resolution.label}"
            t_weighted_payload[f"{key}_power"] = _to_float32_npz(t_weighted_power)
            t_weighted_payload[f"{key}_times_rel"] = np.asarray(prod["times_rel"], dtype=np.float64)
            t_weighted_payload[f"{key}_freqs_hz"] = freqs_wdm.astype(np.float64)

    t_weighted_df = pd.DataFrame(t_weighted_rows)
    t_weighted_df.to_csv(paths.t_weighted_stats_csv, index=False)
    np.savez_compressed(paths.t_weighted_maps_npz, **t_weighted_payload)

    if config.make_grid_plots:
        for source_name in ("noisy", "gw_only"):
            for group in ("links", "XYZ", "AET"):
                _plot_multires_grid(source_name, group, wdm_products, paths, config)
    if config.make_t_weighted_plots:
        _plot_t_weighted_grid(wdm_products, t_weighted_payload, paths, config)

    bestpixel_df = _save_and_plot_best_pixels(wdm_products, t_weighted_payload, t_rel, paths, config)

    diagnostics = {
        "samples": int(len(t_lisa)),
        "dt": dt,
        "span_seconds": float(t_rel[-1] - t_rel[0]),
        "signal_peak_lisa": float(signal["signal_peak_lisa"]),
        "signal_peak_rel": float(signal["signal_peak_rel"]),
        "signal_window_start_lisa": float(signal_window_lisa[0]),
        "signal_window_stop_lisa": float(signal_window_lisa[1]),
        "signal_window_start_rel": float(signal["signal_window_rel"][0]),
        "signal_window_stop_rel": float(signal["signal_window_rel"][1]),
        "n_wdm_rows": int(len(wdm_summary_df)),
        "n_t_weighted_rows": int(len(t_weighted_df)),
        "n_bestpixel_rows": int(len(bestpixel_df)),
    }

    return TDIWDMPipelineProducts(
        paths=paths,
        wdm_summary=wdm_summary_df,
        t_weighted_summary=t_weighted_df,
        bestpixel_summary=bestpixel_df,
        diagnostics=diagnostics,
        wdm_products=wdm_products,
        t_weighted_products=t_weighted_payload,
    )
