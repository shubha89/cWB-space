"""Memory-bounded utilities for ``generate_one_year_noisy_lisa.ipynb``.

Noise generation itself is left to the continuous LISA Instrument scheduler.
The TDI combinations and A/E/T normalization are those of the repository's
existing simulation, not an alternative noise or response model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

import h5py
import numpy as np
from scipy.signal import firwin, kaiserord, resample_poly

JULIAN_YEAR_SECONDS = 365.25 * 86_400.0
CHANNELS = ("X2", "Y2", "Z2", "A2", "E2", "T2")


def sample_count(duration_s: float, dt_s: float) -> int:
    """Require an integer sample count; use half-open time intervals."""
    if not math.isfinite(duration_s) or not math.isfinite(dt_s):
        raise ValueError("Duration and sample spacing must be finite.")
    if duration_s <= 0 or dt_s <= 0:
        raise ValueError("Duration and sample spacing must be positive.")
    ratio = duration_s / dt_s
    count = round(ratio)
    if count < 1 or not math.isclose(ratio, count, rel_tol=0, abs_tol=1e-7):
        raise ValueError(f"{duration_s} s is not an integer number of {dt_s} s samples.")
    return count


@dataclass(frozen=True)
class YearPlan:
    duration_s: float = JULIAN_YEAR_SECONDS
    measurement_dt_s: float = 0.25
    output_dt_s: float = 4.0
    guard_s: float = 2048.0
    tdi_block_s: float = 86_400.0
    passband_hz: float = 0.1
    stopband_attenuation_db: float = 100.0
    interpolation_order: int = 31

    def __post_init__(self):
        sample_count(self.duration_s, self.output_dt_s)
        sample_count(self.duration_s, self.measurement_dt_s)
        sample_count(self.guard_s, self.output_dt_s)
        sample_count(self.tdi_block_s, self.output_dt_s)
        sample_count(self.output_dt_s, self.measurement_dt_s)
        if not 0 < self.passband_hz < 0.5 / self.output_dt_s:
            raise ValueError("Passband must lie strictly below the output Nyquist frequency.")
        if self.stopband_attenuation_db < 60:
            raise ValueError("Use at least 60 dB of resampling stopband attenuation.")
        # The reused evaluator applies order 31 to the measurements as well as
        # to the nested delays. Keep that contract explicit.
        if self.interpolation_order != 31:
            raise ValueError("This notebook uses the repository's order-31 TDI evaluator.")

    @property
    def output_size(self) -> int:
        return sample_count(self.duration_s, self.output_dt_s)

    @property
    def science_size(self) -> int:
        return sample_count(self.duration_s, self.measurement_dt_s)

    @property
    def guard_size(self) -> int:
        return sample_count(self.guard_s, self.measurement_dt_s)

    @property
    def instrument_size(self) -> int:
        return self.science_size + 2 * self.guard_size

    @property
    def downsampling(self) -> int:
        return sample_count(self.output_dt_s, self.measurement_dt_s)


def antialias_filter(plan: YearPlan) -> np.ndarray:
    """Unity-DC FIR, passband to Nyquist transition, applied before decimation.

    resample_poly compensates this symmetric filter's group delay. Padding
    is still required at every processing boundary.
    """
    nyquist_in = 0.5 / plan.measurement_dt_s
    stop_hz = 0.5 / plan.output_dt_s
    taps, beta = kaiserord(
        plan.stopband_attenuation_db,
        (stop_hz - plan.passband_hz) / nyquist_in,
    )
    taps = max(3, taps | 1)  # odd length, integer-sample group delay
    return firwin(
        taps,
        0.5 * (plan.passband_hz + stop_hz),
        fs=1.0 / plan.measurement_dt_s,
        window=("kaiser", beta),
    )


def processing_margin(plan: YearPlan, fir: np.ndarray, max_delay_s: float = 12.0) -> int:
    """Conservative read margin in high-rate samples, aligned to decimation.

    Bound the eight-link nested X2/Y2/Z2 delay chains and their interpolation
    supports, then add the decimator's half filter support. The actual measured
    delays are checked against max_delay_s when reading each block.
    """
    from pytdi.michelson import X2, Y2, Z2

    depth = max(
        len(operator)
        for combination in (X2, Y2, Z2)
        for polynomial in combination.components.values()
        for _, operator in polynomial
    )
    tdi_support_s = (
        depth * max_delay_s
        + (depth + 2) * (plan.interpolation_order + 1) * plan.measurement_dt_s
    )
    total_samples = math.ceil(tdi_support_s / plan.measurement_dt_s) + (len(fir) - 1) // 2 + 2
    margin = math.ceil(total_samples / plan.downsampling) * plan.downsampling
    if margin > plan.guard_size:
        raise ValueError("Increase guard_s: delay/interpolation/filter support exceeds the guard.")
    return margin


def read_tdi_block(reader, start: int, stop: int):
    """Read a bounded slice using LISA Instrument's public file reader.

    This is the same fluctuations/MPR mapping used by Data.from_instrument.
    Uncorrected Michelson combinations consume SCI, REF and TMI carriers;
    clock sideband calibration is not performed by the existing evaluator.
    """
    from pytdi import Data

    measurements = {}
    delays = {}
    for mosa in Data.MOSAS:
        for prefix, field in (
            ("sci", "sci_carrier_fluctuations"),
            ("ref", "ref_carrier_fluctuations"),
            ("tmi", "tmi_carrier_fluctuations"),
        ):
            measurements[f"{prefix}_{mosa}"] = reader.read_by_name_and_mosa(field, mosa, start, stop)
        delay = reader.read_by_name_and_mosa("mprs", mosa, start, stop)
        if not np.isfinite(delay).all() or np.any(delay <= 0) or np.max(delay) > 12.0:
            raise ValueError("MPR outside (0, 12] s; the processing-margin bound must be revised.")
        delays[f"d_{mosa}"] = delay
    if any(not np.isfinite(v).all() for v in measurements.values()):
        raise ValueError("Non-finite instrument measurement; refusing to repair it silently.")
    data = Data(measurements, delays, float(reader.metadata.fs))
    data.compute_delay_derivatives()
    return data


def write_tdi_product(
    instrument_file: str | Path,
    output_file: str | Path,
    plan: YearPlan,
    *,
    central_freq_hz: float,
    progress=print,
) -> dict:
    """Evaluate existing PyTDI combinations in overlaps and retain one year.

    One continuous instrument realization is read in overlapping blocks. No
    noise is restarted at block boundaries. No per-block FFT bandpass is used.
    A failed product keeps its .partial.h5 name and INCOMPLETE status.
    """
    from lisainstrument.instru import sim_result_file
    from pipeline.research.scripts.six_link_generation import LisaLinkSimulationConfig
    from pipeline.research.scripts.tdi_computation import _evaluate_michelson_tdi

    output_file = Path(output_file)
    partial = output_file.with_suffix(".partial.h5")
    if output_file.exists() or partial.exists():
        raise FileExistsError(f"Choose a fresh output path: {output_file}")
    fir = antialias_filter(plan)
    margin = processing_margin(plan, fir)
    block_size = sample_count(plan.tdi_block_s, plan.measurement_dt_s)
    reader = sim_result_file(str(instrument_file))
    if not np.isclose(reader.metadata.dt, plan.measurement_dt_s, rtol=0, atol=1e-14):
        raise ValueError("Instrument sampling does not match plan.")
    if int(reader.metadata.size) != plan.instrument_size:
        raise ValueError("Instrument length does not match the padded science interval.")
    science_t0 = float(reader.metadata.t0) + plan.guard_s
    cfg = LisaLinkSimulationConfig(
        dt=plan.measurement_dt_s,
        science_duration=plan.duration_s,
        central_freq_hz=central_freq_hz,
        tdi_generation=2,
        interpolation_order=plan.interpolation_order,
        tdi_unit="frequency",
    )
    n_written = 0
    with h5py.File(partial, "x") as out:
        out.attrs.update(
            status="INCOMPLETE",
            dt=plan.output_dt_s,
            size=plan.output_size,
            duration_s=plan.duration_s,
            t0=science_t0,
            time_frame="TCB seconds since LISA epoch (repository convention)",
            channel_unit="Hz",
            central_freq_hz=central_freq_hz,
            aet_convention="A=(Z-X)/sqrt(2); E=(X-2Y+Z)/sqrt(6); T=(X+Y+Z)/sqrt(3)",
            tdi_generation=2,
            clock_sideband_calibration=False,
            passband_hz=plan.passband_hz,
            stopband_hz=0.5 / plan.output_dt_s,
            resampling_filter_taps=len(fir),
            resampling_attenuation_db=plan.stopband_attenuation_db,
            read_margin_highrate_samples=margin,
            instrument_file=str(Path(instrument_file).resolve()),
        )
        chunk = min(plan.output_size, 21600)
        for name in ("t", "t_lisa", *CHANNELS):
            ds = out.create_dataset(name, (plan.output_size,), dtype="f8", chunks=(chunk,), compression="lzf", shuffle=True)
            ds.attrs["unit"] = "s" if name in ("t", "t_lisa") else "Hz"
        out.create_dataset("resampling_fir", data=fir)
        out["t"].attrs["description"] = "Seconds since start of retained science interval; endpoint excluded."
        for offset in range(0, plan.science_size, block_size):
            science_start = plan.guard_size + offset
            science_stop = min(science_start + block_size, plan.guard_size + plan.science_size)
            read_start, read_stop = science_start - margin, science_stop + margin
            data = read_tdi_block(reader, read_start, read_stop)
            values = _evaluate_michelson_tdi(data, cfg)[:6]
            first = margin // plan.downsampling
            count = (science_stop - science_start) // plan.downsampling
            output_slice = slice(n_written, n_written + count)
            retained_xyz = []
            for name, raw in zip(CHANNELS[:3], values[:3]):
                if len(raw) != read_stop - read_start or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid TDI block in {name} at offset {offset}.")
                reduced = resample_poly(raw, 1, plan.downsampling, window=fir)
                retained = reduced[first:first + count]
                if len(retained) != count or not np.isfinite(retained).all():
                    raise ValueError(f"Invalid resampled block in {name}.")
                out[name][output_slice] = retained
                retained_xyz.append(retained)
            # A/E/T and identical linear filtering commute mathematically.
            # Form A/E/T after filtering to avoid roundoff from independently
            # filtering combinations with large out-of-band noise power.
            x, y, z = retained_xyz
            for name, retained in zip(CHANNELS[3:], (
                (z - x) / np.sqrt(2.0),
                (x - 2.0 * y + z) / np.sqrt(6.0),
                (x + y + z) / np.sqrt(3.0),
            )):
                out[name][output_slice] = retained
            times = np.arange(n_written, n_written + count, dtype=float) * plan.output_dt_s
            out["t"][output_slice] = times
            out["t_lisa"][output_slice] = science_t0 + times
            n_written += count
            out.attrs["written_samples"] = n_written
            out.flush()
            progress(f"TDI {n_written:,}/{plan.output_size:,} samples ({100*n_written/plan.output_size:.1f}%)")
        if n_written != plan.output_size:
            raise ValueError(f"Expected {plan.output_size} samples, wrote {n_written}.")
        out.attrs["status"] = "COMPLETE"
    partial.rename(output_file)
    return {"output_samples": n_written, "science_t0": science_t0, "fir_taps": len(fir), "read_margin_s": margin * plan.measurement_dt_s}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_pilot_to_full_record(
    instrument_file: str | Path,
    tdi_file: str | Path,
    plan: YearPlan,
    *,
    relative_tolerance: float = 1e-9,
) -> dict:
    """Cross-check the bounded reader/TDI route against the existing full reader.

    Deliberately restricted to short pilots: Data.from_instrument loads whole
    arrays into RAM. The A/E/T comparison also measures roundoff from commuting
    the linear filtering and A/E/T transforms, on an explicit relative-L2 scale.
    """
    from pytdi import Data
    from pipeline.research.scripts.six_link_generation import LisaLinkSimulationConfig
    from pipeline.research.scripts.tdi_computation import _evaluate_michelson_tdi

    if plan.instrument_size > 100_000:
        raise ValueError("Use this whole-record comparison only on a short pilot (<=100,000 input samples).")
    data = Data.from_instrument(str(instrument_file), signals="fluctuations", skipped=0)
    cfg = LisaLinkSimulationConfig(dt=plan.measurement_dt_s, tdi_generation=2, tdi_unit="frequency")
    reference = _evaluate_michelson_tdi(data, cfg)[:6]
    fir = antialias_filter(plan)
    first = plan.guard_size // plan.downsampling
    report = {}
    with h5py.File(tdi_file, "r") as output:
        for name, raw in zip(CHANNELS, reference):
            expected = resample_poly(raw, 1, plan.downsampling, window=fir)[first:first + plan.output_size]
            actual = output[name][:]
            residual = actual - expected
            scale = float(np.linalg.norm(expected))
            if scale == 0:
                raise ValueError(f"Zero reference norm in {name}; relative comparison is undefined.")
            error = float(np.linalg.norm(residual) / scale)
            report[name] = dict(relative_l2=error, max_abs_hz=float(np.max(np.abs(residual))))
            if not math.isfinite(error) or error > relative_tolerance:
                raise ValueError(f"Whole-record comparison failed in {name}: {error} > {relative_tolerance}.")
    return {"relative_l2_tolerance": relative_tolerance, "channels": report}


def write_binary_response(
    source,
    path: str | Path,
    *,
    t0: float,
    size: int,
    dt: float,
    block_size: int = 100_000,
) -> None:
    """Write the optional built-in GW response without a year-sized allocation.

    lisagwresponse 3.0.2 initializes its output with a full np.zeros(size, 6)
    even when write(chunks=...) is requested. Write ordinary small response
    files through its public API and copy them into a lazily allocated dataset.
    Every block uses the same source object, reference epoch and absolute TCB
    times; no waveform phase is restarted.
    """
    path = Path(path)
    partial = path.with_suffix(".partial.h5")
    piece = path.with_suffix(".block.h5")
    if any(p.exists() for p in (path, partial, piece)):
        raise FileExistsError(f"Choose a fresh GW response path: {path}")
    with h5py.File(partial, "x") as out:
        target = out.create_dataset("tcb/y", (size, 6), dtype="f8", chunks=(min(size, block_size), 6))
        for start in range(0, size, block_size):
            count = min(block_size, size - start)
            source.write(str(piece), dt=dt, size=count, t0=t0 + start * dt, mode="x", timeframe="tcb", chunks=block_size)
            with h5py.File(piece, "r") as block:
                if start == 0:
                    for key, value in block.attrs.items():
                        out.attrs[key] = value
                    out.attrs.update(t0=t0, size=size, duration=size * dt)
                values = block["tcb/y"][:]
                if values.shape != (count, 6) or not np.isfinite(values).all():
                    raise ValueError("Invalid optional GW response block.")
                target[start:start + count] = values
            piece.unlink()
    partial.rename(path)


def write_json(path: str | Path, payload: dict) -> None:
    """Replace a small JSON receipt atomically."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
