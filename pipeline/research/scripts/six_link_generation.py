"""Generate LISA six-link GW and noisy link data.

This module owns the waveform preparation, orbit generation, GW response, and
lisainstrument noisy-link stages. TDI and WDM post-processing live in separate
modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
from lisaorbits import OEMOrbits
from lisagwresponse import ReadStrain
from lisainstrument import Instrument

from .plotting_utils import plot_link_columns, plot_waveform

import pytdi.michelson


LINKS: tuple[int, ...] = (12, 23, 31, 13, 32, 21)
LINK_LABELS: tuple[str, ...] = ("12", "23", "31", "13", "32", "21")


@dataclass(frozen=True)
class LisaLinkSimulationConfig:
    """Numerical and physical settings for the LISA link simulation."""

    dt: float = 0.1
    science_duration: float = 100_000.0
    burn_in_seconds: float = 600.0
    center_waveform_peak: bool = True
    edge_taper_seconds: float = 0.0

    ra: float = np.pi / 4.0
    dec: float = 0.3
    central_freq_hz: float = 2.816e14

    orbit_name: str = "esa-trailing"
    orbit_dt: float = 600.0
    orbit_duration_factor: float = 5.0
    orbit_start_offset_seconds: float | None = None
    orbit_dataset: str = "tcb/ltt"
    gw_dataset: str = "tcb"

    seed: int = 12345
    frequency_low_hz: float = 1e-5
    frequency_high_hz: float = 1e-2
    tdi_generation: int = 2
    interpolation_order: int = 31
    tdi_unit: str = "frequency"
    extra_skipped: int = 0
    edge_crop: int = 0
    instrument_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LisaLinkSimulationPaths:
    """Filesystem layout used by the LISA link simulation."""

    work_dir: Path
    input_dir: Path
    output_dir: Path
    plot_dir: Path
    aligned_waveform_npz: Path
    orbits_h5: Path
    gws_h5: Path
    instrument_h5: Path
    gw_links_csv: Path
    noisy_links_csv: Path
    noisy_tdi_csv: Path
    gw_tdi_csv: Path
    noisy_sagnac_tdi_csv: Path
    gw_sagnac_tdi_csv: Path
    waveform_plot: Path
    gw_links_plot: Path
    noisy_links_plot: Path
    noisy_tdi_xyz_plot: Path
    noisy_tdi_aet_plot: Path
    gw_tdi_xyz_plot: Path
    gw_tdi_aet_plot: Path
    noisy_tdi_sagnac_plot: Path
    gw_tdi_sagnac_plot: Path
    overlay_xyz_plot: Path
    overlay_aet_plot: Path

    @classmethod
    def from_work_dir(cls, work_dir: str | Path) -> "LisaLinkSimulationPaths":
        root = Path(work_dir)
        input_dir = root / "inputs"
        output_dir = root / "outputs"
        plot_dir = output_dir / "plots"
        return cls(
            work_dir=root,
            input_dir=input_dir,
            output_dir=output_dir,
            plot_dir=plot_dir,
            aligned_waveform_npz=input_dir / "waveform_aligned.npz",
            orbits_h5=input_dir / "orbits.h5",
            gws_h5=output_dir / "gws.h5",
            instrument_h5=output_dir / "lisa_instrument_output.h5",
            gw_links_csv=output_dir / "six_links_gw_only.csv",
            noisy_links_csv=output_dir / "six_links_noisy.csv",
            noisy_tdi_csv=output_dir / "tdi_michelson_XYZ_AET.csv",
            gw_tdi_csv=output_dir / "tdi_gw_only_XYZ_AET.csv",
            noisy_sagnac_tdi_csv=output_dir / "tdi_sagnac_noisy.csv",
            gw_sagnac_tdi_csv=output_dir / "tdi_sagnac_gw_only.csv",
            waveform_plot=output_dir / "01_aligned_waveform_peak_centered.png",
            gw_links_plot=output_dir / "02_gw_only_six_links_peak_centered.png",
            noisy_links_plot=output_dir / "03_noisy_six_links.png",
            noisy_tdi_xyz_plot=plot_dir / "04_tdi_XYZ_orbit_time.png",
            noisy_tdi_aet_plot=plot_dir / "04_tdi_AET_orbit_time.png",
            gw_tdi_xyz_plot=plot_dir / "05_tdi_gw_only_XYZ.png",
            gw_tdi_aet_plot=plot_dir / "05_tdi_gw_only_AET.png",
            noisy_tdi_sagnac_plot=plot_dir / "04_tdi_sagnac_orbit_time.png",
            gw_tdi_sagnac_plot=plot_dir / "05_tdi_gw_only_sagnac.png",
            overlay_xyz_plot=plot_dir / "06_tdi_noisy_with_gw_signal_XYZ.png",
            overlay_aet_plot=plot_dir / "06_tdi_noisy_with_gw_signal_AET.png",
        )

    def make_dirs(self) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PreparedWaveform:
    t: np.ndarray
    tau: np.ndarray
    hplus: np.ndarray
    hcross: np.ndarray
    dt: float
    peak_index: int
    peak_time: float

    @property
    def duration(self) -> float:
        return self.dt * (len(self.t) - 1)


@dataclass(frozen=True)
class OrbitInfo:
    t0: float
    t1: float
    dt: float
    size: int
    span: float


def load_npz_waveform(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a waveform NPZ with arrays ``t``, ``hplus``, and ``hcross``."""

    data = np.load(path)
    return (
        np.asarray(data["t"], dtype=float),
        np.asarray(data["hplus"], dtype=float),
        np.asarray(data["hcross"], dtype=float),
    )


def _validate_waveform_arrays(
    t: Sequence[float] | np.ndarray,
    hplus: Sequence[float] | np.ndarray,
    hcross: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    hp_arr = np.asarray(hplus, dtype=float)
    hc_arr = np.asarray(hcross, dtype=float)

    if t_arr.ndim != 1 or hp_arr.ndim != 1 or hc_arr.ndim != 1:
        raise ValueError("t, hplus, and hcross must be one-dimensional arrays.")
    if not (len(t_arr) == len(hp_arr) == len(hc_arr)):
        raise ValueError("t, hplus, and hcross must have the same length.")
    if len(t_arr) < 2:
        raise ValueError("waveform arrays must contain at least two samples.")
    if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(hp_arr)) or not np.all(np.isfinite(hc_arr)):
        raise ValueError("waveform arrays must contain only finite values.")
    if not np.all(np.diff(t_arr) > 0.0):
        raise ValueError("waveform time array must be strictly increasing.")

    return t_arr, hp_arr, hc_arr


def prepare_science_waveform(
    t: Sequence[float] | np.ndarray,
    hplus: Sequence[float] | np.ndarray,
    hcross: Sequence[float] | np.ndarray,
    config: LisaLinkSimulationConfig,
    *,
    output_file: str | Path | None = None,
    plot_file: str | Path | None = None,
) -> PreparedWaveform:
    """Resample a user waveform onto the science grid used by the LISA link simulation.

    If ``config.center_waveform_peak`` is true, the largest strain-amplitude
    sample is placed at the center of the science segment. Values outside the
    supplied waveform support are zero-padded.
    """

    t_raw, hp_raw, hc_raw = _validate_waveform_arrays(t, hplus, hcross)
    n = int(round(config.science_duration / config.dt))
    if n < 2:
        raise ValueError("science_duration / dt must produce at least two samples.")

    amp_raw = np.sqrt(hp_raw**2 + hc_raw**2)
    peak_index_raw = int(np.argmax(amp_raw))
    peak_time_raw = float(t_raw[peak_index_raw])

    science_t = np.arange(n, dtype=float) * config.dt
    if config.center_waveform_peak:
        interp_t = science_t - config.science_duration / 2.0
        source_t = t_raw - peak_time_raw
    else:
        interp_t = science_t
        source_t = t_raw - float(t_raw[0])

    hp = np.interp(interp_t, source_t, hp_raw, left=0.0, right=0.0)
    hc = np.interp(interp_t, source_t, hc_raw, left=0.0, right=0.0)
    if config.edge_taper_seconds > 0.0:
        taper_size = int(round(config.edge_taper_seconds / config.dt))
        taper_size = max(0, min(taper_size, n // 2))
        if taper_size:
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper_size, endpoint=False)))
            taper = np.ones(n, dtype=float)
            taper[:taper_size] = ramp
            taper[-taper_size:] = ramp[::-1]
            hp = hp * taper
            hc = hc * taper
    amp = np.sqrt(hp**2 + hc**2)
    peak_index = int(np.argmax(amp))

    prepared = PreparedWaveform(
        t=science_t,
        tau=interp_t,
        hplus=hp,
        hcross=hc,
        dt=config.dt,
        peak_index=peak_index,
        peak_time=float(science_t[peak_index]),
    )

    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_file, t=prepared.t, hplus=prepared.hplus, hcross=prepared.hcross)

    if plot_file is not None:
        plot_waveform(prepared, plot_file)

    return prepared


def read_orbit_time_range(orbits_file: str | Path) -> OrbitInfo:
    """Read the time range stored in a LISA orbit HDF5 file."""

    with h5py.File(orbits_file, "r") as hdf:
        t0 = float(hdf.attrs["t0"])
        dt = float(hdf.attrs["dt"])
        size = int(hdf.attrs["size"])
    t1 = t0 + dt * (size - 1)
    return OrbitInfo(t0=t0, t1=t1, dt=dt, size=size, span=t1 - t0)


def create_orbit_file(
    prepared: PreparedWaveform,
    config: LisaLinkSimulationConfig,
    output_file: str | Path,
) -> OrbitInfo:
    """Create a realistic LISA orbit file scaled to the science duration."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    requested_duration = config.orbit_duration_factor * prepared.duration
    size = int(np.ceil(requested_duration / config.orbit_dt)) + 1
    if size < 2:
        raise ValueError("orbit_duration_factor and orbit_dt produce too few orbit samples.")

    orbits = OEMOrbits.from_included(config.orbit_name, version="latest")
    if output_path.exists():
        output_path.unlink()
    orbits.write(str(output_path), t0="init", dt=config.orbit_dt, size=size, mode="x")
    return read_orbit_time_range(output_path)


def compute_gw_links(
    prepared: PreparedWaveform,
    config: LisaLinkSimulationConfig,
    paths: LisaLinkSimulationPaths,
    *,
    make_plots: bool = True,
) -> dict[str, float | int]:
    """Compute six one-way GW links and write a lisainstrument GW HDF5 file."""

    orbit = read_orbit_time_range(paths.orbits_h5)
    burn_in_size = int(np.ceil(config.burn_in_seconds / prepared.dt))
    n_waveform = len(prepared.t)
    n_total = burn_in_size + n_waveform
    required_span = prepared.dt * (n_total - 1)

    if required_span > orbit.span:
        raise ValueError(
            "Burn-in plus waveform duration is longer than the generated orbit file. "
            f"Need {required_span} s, orbit has {orbit.span} s."
        )

    if config.orbit_start_offset_seconds is None:
        orbit_start_lisa = orbit.t0 + 0.5 * (orbit.span - required_span)
    else:
        orbit_start_lisa = orbit.t0 + config.orbit_start_offset_seconds

    t_lisa = orbit_start_lisa + np.arange(n_total, dtype=float) * prepared.dt
    if t_lisa[0] < orbit.t0 or t_lisa[-1] > orbit.t1:
        raise ValueError("Burn-in plus waveform time grid lies outside orbit file time range.")

    science = np.arange(n_total) >= burn_in_size
    t_waveform_full = (np.arange(n_total, dtype=float) - burn_in_size) * prepared.dt
    hplus_full = np.zeros(n_total, dtype=float)
    hcross_full = np.zeros(n_total, dtype=float)
    hplus_full[science] = prepared.hplus
    hcross_full[science] = prepared.hcross

    source = ReadStrain(
        t_lisa,
        hplus_full,
        hcross_full,
        orbits=str(paths.orbits_h5),
        ra=config.ra,
        dec=config.dec,
    )

    y_gw = source.compute_gw_response(t_lisa, LINKS)
    y_gw_hz = config.central_freq_hz * y_gw

    df = pd.DataFrame(
        np.column_stack(
            [
                np.arange(n_total),
                t_waveform_full,
                t_lisa,
                science.astype(int),
                y_gw,
                y_gw_hz,
            ]
        ),
        columns=[
            "sample",
            "t_waveform",
            "t_lisa",
            "is_science",
            *[f"gw_{lab}" for lab in LINK_LABELS],
            *[f"gw_{lab}_hz" for lab in LINK_LABELS],
        ],
    )
    df.to_csv(paths.gw_links_csv, index=False)

    if paths.gws_h5.exists():
        paths.gws_h5.unlink()
    source.write(str(paths.gws_h5), t0=float(t_lisa[0]), dt=prepared.dt, size=n_total)

    if make_plots:
        plot_link_columns(
            df.loc[science, "t_lisa"].to_numpy(),
            {f"gw_{lab}_hz": df.loc[science, f"gw_{lab}_hz"].to_numpy() for lab in LINK_LABELS},
            "GW-only LISA six-link response converted to Hz",
            "GW-only carrier frequency fluctuation [Hz]",
            paths.gw_links_plot,
        )

    return {
        "burn_in_size": burn_in_size,
        "n_waveform": n_waveform,
        "n_total": n_total,
        "orbit_start_lisa": float(orbit_start_lisa),
        "t_lisa_min": float(t_lisa[0]),
        "t_lisa_max": float(t_lisa[-1]),
        "gw_link_max_hz": float(np.max(np.abs(y_gw_hz[science]))),
    }


def run_lisa_instrument(
    config: LisaLinkSimulationConfig,
    paths: LisaLinkSimulationPaths,
    *,
    make_plots: bool = True,
) -> dict[str, float | int]:
    """Run LISA Instrument and save science-region noisy six-link data."""

    timing = pd.read_csv(paths.gw_links_csv, usecols=["t_waveform", "t_lisa", "is_science"])
    t_waveform = timing["t_waveform"].to_numpy()
    t_lisa = timing["t_lisa"].to_numpy()
    is_science = timing["is_science"].astype(bool).to_numpy()
    burn_in_size = int(np.argmax(is_science)) if not is_science[0] else 0

    dt = float(np.median(np.diff(t_waveform)))
    n_total = len(t_lisa)
    t0_lisa = float(t_lisa[0])

    if paths.instrument_h5.exists():
        paths.instrument_h5.unlink()

    instrument_kwargs = {
        "noises_f_min_hz": config.frequency_low_hz,
        "clock_f_min_hz": config.frequency_low_hz,
        "chunk_gw_files": True,
        **dict(config.instrument_kwargs),
    }
    instrument = Instrument(
        size=n_total,
        dt=dt,
        t0=t0_lisa,
        orbits=str(paths.orbits_h5),
        gws=str(paths.gws_h5),
        orbit_dataset=config.orbit_dataset,
        central_freq=config.central_freq_hz,
        seed=config.seed,
        **instrument_kwargs,
    )

    if hasattr(instrument, "export_hdf5"):
        instrument.export_hdf5(path=str(paths.instrument_h5), overwrite=True)
    elif hasattr(instrument, "write"):
        instrument.write(str(paths.instrument_h5), overwrite=True)
    else:
        instrument.simulate()
        raise RuntimeError("Could not find Instrument.export_hdf5() or Instrument.write().")

    group = "debug/sci_carrier_fluctuations"
    with h5py.File(paths.instrument_h5, "r") as hdf:
        arrays = []
        for lab in LINK_LABELS:
            key = f"{group}/{lab}"
            if key not in hdf:
                raise KeyError(f"Missing dataset {key} in {paths.instrument_h5}")
            arrays.append(np.asarray(hdf[key]))

    m_total = min([len(a) for a in arrays] + [n_total])
    science_idx = np.flatnonzero(is_science[:m_total])
    df = pd.DataFrame(
        {
            "sample": np.arange(len(science_idx)),
            "instrument_sample": science_idx,
            "t_waveform": t_waveform[science_idx],
            "t_lisa": t_lisa[science_idx],
        }
    )
    for lab, arr in zip(LINK_LABELS, arrays):
        df[f"noisy_{lab}"] = arr[science_idx]
    df.to_csv(paths.noisy_links_csv, index=False)

    if make_plots:
        plot_link_columns(
            df["t_lisa"].to_numpy(),
            {f"noisy_{lab}": df[f"noisy_{lab}"].to_numpy() for lab in LINK_LABELS},
            "Noisy LISA six-link science carrier fluctuations",
            "noisy science carrier fluctuation [Hz]",
            paths.noisy_links_plot,
        )

    return {
        "instrument_samples": n_total,
        "science_samples": int(np.sum(is_science)),
        "burn_in_size": burn_in_size,
    }
