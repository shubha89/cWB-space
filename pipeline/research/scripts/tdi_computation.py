"""Compute Michelson XYZ/AET and Sagnac TDI products with PyTDI."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from pytdi import Data
import pytdi.michelson
import pytdi.sagnac

from .six_link_generation import LisaLinkSimulationConfig, LisaLinkSimulationPaths
from .plotting_utils import plot_link_columns

def get_delay_derivatives_if_available(data: Data):
    if hasattr(data, "delay_derivatives"):
        return data.delay_derivatives
    if hasattr(data, "delay_derivs"):
        return data.delay_derivs
    if hasattr(data, "derivatives"):
        return data.derivatives
    return None


def apply_frequency_cut(arr: np.ndarray, dt: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Apply a simple FFT-domain bandpass cut."""

    freqs = np.fft.rfftfreq(len(arr), d=dt)
    spectrum = np.fft.rfft(arr)
    keep = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum[~keep] = 0.0
    return np.fft.irfft(spectrum, n=len(arr))


def _tdi_combinations(config: LisaLinkSimulationConfig):
    if config.tdi_generation == 2:
        return pytdi.michelson.X2, pytdi.michelson.Y2, pytdi.michelson.Z2, "2"
    if config.tdi_generation == 1:
        return pytdi.michelson.X1, pytdi.michelson.Y1, pytdi.michelson.Z1, "1"
    raise ValueError("tdi_generation must be 1 or 2.")


def _sagnac_combinations(config: LisaLinkSimulationConfig):
    if config.tdi_generation == 2:
        return (
            ("alpha2", pytdi.sagnac.ALPHA2),
            ("beta2", pytdi.sagnac.BETA2),
            ("gamma2", pytdi.sagnac.GAMMA2),
            ("zeta21", pytdi.sagnac.ZETA21),
            ("zeta22", pytdi.sagnac.ZETA22),
            ("zeta23", pytdi.sagnac.ZETA23),
        )
    if config.tdi_generation == 1:
        return (
            ("alpha1", pytdi.sagnac.ALPHA1),
            ("beta1", pytdi.sagnac.BETA1),
            ("gamma1", pytdi.sagnac.GAMMA1),
            ("zeta1", pytdi.sagnac.ZETA1),
        )
    raise ValueError("tdi_generation must be 1 or 2.")


def _build_kwargs(data: Data, config: LisaLinkSimulationConfig) -> dict[str, object]:
    build_kwargs: dict[str, object] = {"order": config.interpolation_order}
    delay_derivatives = get_delay_derivatives_if_available(data)
    if delay_derivatives is not None:
        build_kwargs["delay_derivatives"] = delay_derivatives
    return build_kwargs


def _crop_tdi_to_science(
    arrays: Sequence[np.ndarray],
    t_lisa_meta: np.ndarray,
    t_waveform_meta: np.ndarray,
    is_science: np.ndarray,
    config: LisaLinkSimulationConfig,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]:
    burn_in_size = int(np.argmax(is_science)) if not is_science[0] else 0
    start = burn_in_size + config.extra_skipped + config.edge_crop
    n = min([len(arr) for arr in arrays] + [len(t_lisa_meta)])
    stop = n - config.edge_crop
    if stop <= start:
        raise ValueError("Not enough samples after burn-in and edge crop.")

    cropped = [arr[start:stop] for arr in arrays]
    t_lisa = t_lisa_meta[start:stop]
    t_waveform = t_waveform_meta[start:stop]
    n_out = min([len(arr) for arr in cropped] + [len(t_lisa), len(t_waveform)])
    return [arr[:n_out] for arr in cropped], t_lisa[:n_out], t_waveform[:n_out], start


def _evaluate_michelson_tdi(data: Data, config: LisaLinkSimulationConfig) -> tuple[np.ndarray, ...]:
    x_comb, y_comb, z_comb, suffix = _tdi_combinations(config)
    build_kwargs = _build_kwargs(data, config)

    x_built = x_comb.build(data.delays, data.fs, **build_kwargs)
    y_built = y_comb.build(data.delays, data.fs, **build_kwargs)
    z_built = z_comb.build(data.delays, data.fs, **build_kwargs)

    x = x_built(data.measurements, unit=config.tdi_unit)
    y = y_built(data.measurements, unit=config.tdi_unit)
    z = z_built(data.measurements, unit=config.tdi_unit)
    a = (z - x) / np.sqrt(2.0)
    e = (x - 2.0 * y + z) / np.sqrt(6.0)
    t = (x + y + z) / np.sqrt(3.0)
    n = min(len(x), len(y), len(z), len(a), len(e), len(t))
    return x[:n], y[:n], z[:n], a[:n], e[:n], t[:n], suffix


def _evaluate_sagnac_tdi(data: Data, config: LisaLinkSimulationConfig) -> tuple[dict[str, np.ndarray], str]:
    build_kwargs = _build_kwargs(data, config)
    channels: dict[str, np.ndarray] = {}
    for name, combination in _sagnac_combinations(config):
        built = combination.build(data.delays, data.fs, **build_kwargs)
        channels[name] = built(data.measurements, unit=config.tdi_unit)
    n = min(len(arr) for arr in channels.values())
    return {name: arr[:n] for name, arr in channels.items()}, str(config.tdi_generation)


def compute_tdi_from_instrument(
    config: LisaLinkSimulationConfig,
    paths: LisaLinkSimulationPaths,
    *,
    make_plots: bool = True,
) -> dict[str, float | int]:
    """Compute noisy Michelson XYZ and orthogonal AET from lisainstrument output."""

    gw_meta = pd.read_csv(paths.gw_links_csv, usecols=["t_waveform", "t_lisa", "is_science"])
    t_lisa_meta = gw_meta["t_lisa"].to_numpy()
    t_waveform_meta = gw_meta["t_waveform"].to_numpy()
    is_science = gw_meta["is_science"].astype(bool).to_numpy()
    dt_tdi = float(np.median(np.diff(t_waveform_meta)))

    data = Data.from_instrument(str(paths.instrument_h5), signals="fluctuations", skipped=0)
    x, y, z, a, e, t, suffix = _evaluate_michelson_tdi(data, config)
    (x, y, z, a, e, t), t_lisa, t_waveform, start = _crop_tdi_to_science(
        [x, y, z, a, e, t], t_lisa_meta, t_waveform_meta, is_science, config
    )

    x, y, z, a, e, t = [
        apply_frequency_cut(arr, dt_tdi, config.frequency_low_hz, config.frequency_high_hz)
        for arr in (x, y, z, a, e, t)
    ]

    sample = np.arange(len(t_lisa))
    df = pd.DataFrame(
        {
            "sample": sample,
            "instrument_sample": sample + start,
            "t_waveform": t_waveform,
            "t_lisa": t_lisa,
            f"X{suffix}": x,
            f"Y{suffix}": y,
            f"Z{suffix}": z,
            f"A{suffix}": a,
            f"E{suffix}": e,
            f"T{suffix}": t,
        }
    )
    df.to_csv(paths.noisy_tdi_csv, index=False)

    if make_plots:
        plot_link_columns(
            t_lisa,
            {f"X{suffix}": x, f"Y{suffix}": y, f"Z{suffix}": z},
            f"Michelson TDI X{suffix}, Y{suffix}, Z{suffix}",
            "TDI frequency combination [Hz]",
            paths.noisy_tdi_xyz_plot,
        )
        plot_link_columns(
            t_lisa,
            {f"A{suffix}": a, f"E{suffix}": e, f"T{suffix}": t},
            f"Orthogonal TDI A{suffix}, E{suffix}, T{suffix}",
            "TDI frequency combination [Hz]",
            paths.noisy_tdi_aet_plot,
        )

    return {
        "tdi_samples": len(df),
        "tdi_start_index": int(start),
        "noisy_x_std_hz": float(np.std(x)),
        "noisy_a_std_hz": float(np.std(a)),
    }


def compute_sagnac_tdi_from_instrument(
    config: LisaLinkSimulationConfig,
    paths: LisaLinkSimulationPaths,
    *,
    make_plots: bool = True,
) -> dict[str, float | int]:
    """Compute noisy Sagnac TDI channels from the full lisainstrument output."""

    gw_meta = pd.read_csv(paths.gw_links_csv, usecols=["t_waveform", "t_lisa", "is_science"])
    t_lisa_meta = gw_meta["t_lisa"].to_numpy()
    t_waveform_meta = gw_meta["t_waveform"].to_numpy()
    is_science = gw_meta["is_science"].astype(bool).to_numpy()
    dt_tdi = float(np.median(np.diff(t_waveform_meta)))

    data = Data.from_instrument(str(paths.instrument_h5), signals="fluctuations", skipped=0)
    channels, suffix = _evaluate_sagnac_tdi(data, config)
    names = list(channels)
    cropped, t_lisa, t_waveform, start = _crop_tdi_to_science(
        [channels[name] for name in names], t_lisa_meta, t_waveform_meta, is_science, config
    )
    channels = {
        name: apply_frequency_cut(arr, dt_tdi, config.frequency_low_hz, config.frequency_high_hz)
        for name, arr in zip(names, cropped)
    }

    sample = np.arange(len(t_lisa))
    df = pd.DataFrame(
        {
            "sample": sample,
            "instrument_sample": sample + start,
            "t_waveform": t_waveform,
            "t_lisa": t_lisa,
            **channels,
        }
    )
    df.to_csv(paths.noisy_sagnac_tdi_csv, index=False)

    if make_plots:
        plot_link_columns(
            t_lisa,
            channels,
            f"Sagnac TDI generation {suffix}",
            "TDI frequency combination [Hz]",
            paths.noisy_tdi_sagnac_plot,
        )

    return {
        "sagnac_tdi_samples": len(df),
        "sagnac_tdi_start_index": int(start),
        **{f"noisy_{name}_std_hz": float(np.std(arr)) for name, arr in channels.items()},
    }


def compute_tdi_from_gws(
    config: LisaLinkSimulationConfig,
    paths: LisaLinkSimulationPaths,
    *,
    make_plots: bool = True,
) -> dict[str, float | int]:
    """Compute GW-only TDI and convert the result to Hz."""

    gw_meta = pd.read_csv(paths.gw_links_csv, usecols=["t_waveform", "t_lisa", "is_science"])
    t_lisa_meta = gw_meta["t_lisa"].to_numpy()
    t_waveform_meta = gw_meta["t_waveform"].to_numpy()
    is_science = gw_meta["is_science"].astype(bool).to_numpy()
    dt_tdi = float(np.median(np.diff(t_waveform_meta)))

    data = Data.from_gws(
        str(paths.gws_h5),
        str(paths.orbits_h5),
        skipped=0,
        gw_dataset=config.gw_dataset,
        orbit_dataset=config.orbit_dataset,
    )
    x, y, z, a, e, t, suffix = _evaluate_michelson_tdi(data, config)
    (x, y, z, a, e, t), t_lisa, t_waveform, start = _crop_tdi_to_science(
        [x, y, z, a, e, t], t_lisa_meta, t_waveform_meta, is_science, config
    )

    x, y, z, a, e, t = [
        apply_frequency_cut(arr, dt_tdi, config.frequency_low_hz, config.frequency_high_hz)
        for arr in (x, y, z, a, e, t)
    ]

    # Data.from_gws returns dimensionless fractional-frequency TDI.
    x, y, z, a, e, t = [config.central_freq_hz * arr for arr in (x, y, z, a, e, t)]

    sample = np.arange(len(t_lisa))
    df = pd.DataFrame(
        {
            "sample": sample,
            "source_sample": sample + start,
            "t_waveform": t_waveform,
            "t_lisa": t_lisa,
            f"X{suffix}_gw_hz": x,
            f"Y{suffix}_gw_hz": y,
            f"Z{suffix}_gw_hz": z,
            f"A{suffix}_gw_hz": a,
            f"E{suffix}_gw_hz": e,
            f"T{suffix}_gw_hz": t,
        }
    )
    df.to_csv(paths.gw_tdi_csv, index=False)

    if make_plots:
        plot_link_columns(
            t_lisa,
            {f"X{suffix}_gw_hz": x, f"Y{suffix}_gw_hz": y, f"Z{suffix}_gw_hz": z},
            f"GW-only Michelson TDI X{suffix}, Y{suffix}, Z{suffix} converted to Hz",
            "GW-only TDI frequency combination [Hz]",
            paths.gw_tdi_xyz_plot,
        )
        plot_link_columns(
            t_lisa,
            {f"A{suffix}_gw_hz": a, f"E{suffix}_gw_hz": e, f"T{suffix}_gw_hz": t},
            f"GW-only orthogonal TDI A{suffix}, E{suffix}, T{suffix} converted to Hz",
            "GW-only TDI frequency combination [Hz]",
            paths.gw_tdi_aet_plot,
        )

    return {
        "gw_tdi_samples": len(df),
        "gw_x_max_hz": float(np.max(np.abs(x))),
        "gw_a_max_hz": float(np.max(np.abs(a))),
    }


def compute_sagnac_tdi_from_gws(
    config: LisaLinkSimulationConfig,
    paths: LisaLinkSimulationPaths,
    *,
    make_plots: bool = True,
) -> dict[str, float | int]:
    """Compute GW-only Sagnac TDI channels and convert the result to Hz."""

    gw_meta = pd.read_csv(paths.gw_links_csv, usecols=["t_waveform", "t_lisa", "is_science"])
    t_lisa_meta = gw_meta["t_lisa"].to_numpy()
    t_waveform_meta = gw_meta["t_waveform"].to_numpy()
    is_science = gw_meta["is_science"].astype(bool).to_numpy()
    dt_tdi = float(np.median(np.diff(t_waveform_meta)))

    data = Data.from_gws(
        str(paths.gws_h5),
        str(paths.orbits_h5),
        skipped=0,
        gw_dataset=config.gw_dataset,
        orbit_dataset=config.orbit_dataset,
    )
    channels, suffix = _evaluate_sagnac_tdi(data, config)
    names = list(channels)
    cropped, t_lisa, t_waveform, start = _crop_tdi_to_science(
        [channels[name] for name in names], t_lisa_meta, t_waveform_meta, is_science, config
    )
    channels = {
        name: config.central_freq_hz
        * apply_frequency_cut(arr, dt_tdi, config.frequency_low_hz, config.frequency_high_hz)
        for name, arr in zip(names, cropped)
    }

    sample = np.arange(len(t_lisa))
    df = pd.DataFrame(
        {
            "sample": sample,
            "source_sample": sample + start,
            "t_waveform": t_waveform,
            "t_lisa": t_lisa,
            **{f"{name}_gw_hz": arr for name, arr in channels.items()},
        }
    )
    df.to_csv(paths.gw_sagnac_tdi_csv, index=False)

    if make_plots:
        plot_link_columns(
            t_lisa,
            {f"{name}_gw_hz": arr for name, arr in channels.items()},
            f"GW-only Sagnac TDI generation {suffix} converted to Hz",
            "GW-only TDI frequency combination [Hz]",
            paths.gw_tdi_sagnac_plot,
        )

    return {
        "gw_sagnac_tdi_samples": len(df),
        **{f"gw_{name}_max_hz": float(np.max(np.abs(arr))) for name, arr in channels.items()},
    }
