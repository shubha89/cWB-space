"""Waveform generation for sampled SMBHB parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import WaveformConfig


def _require_phenomxpy():
    try:
        from phenomxpy import IMRPhenomTHM
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        raise ModuleNotFoundError(
            "SMBHB waveform generation requires phenomxpy. "
            "Activate the lisa-tf environment or install phenomxpy."
        ) from exc
    return IMRPhenomTHM


def _construct_phenom_model(params: Mapping[str, float]):
    IMRPhenomTHM = _require_phenomxpy()
    kwargs = {
        "eta": float(params["eta"]),
        "total_mass": float(params["total_mass"]),
        "distance": float(params["distance_mpc"]),
        "s1": float(params.get("s1", 0.0)),
        "s2": float(params.get("s2", 0.0)),
        "f_min": float(params.get("f_min_hz", 1.0e-4)),
        "iota": float(params.get("iota", 0.0)),
        "phi_ref": float(params.get("phi_ref", 0.0)),
    }
    attempts = [
        kwargs,
        {**kwargs, "inclination": kwargs["iota"]},
        {key: value for key, value in kwargs.items() if key not in {"iota", "phi_ref"}},
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return IMRPhenomTHM(**attempt), attempt
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not construct IMRPhenomTHM for source {params.get('source_id', '<unknown>')}") from last_error


def _unpack_polarizations(out, input_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(out, (tuple, list)):
        raise TypeError(f"Expected tuple/list from compute_polarizations, got {type(out)}.")
    arrays = [np.asarray(item) for item in out]
    if len(arrays) == 2:
        hplus, hcross = arrays
        t_save = input_times
    elif len(arrays) == 3:
        if arrays[0].shape == input_times.shape and np.all(np.diff(arrays[0]) > 0.0):
            t_save, hplus, hcross = arrays
        elif arrays[2].shape == input_times.shape and np.all(np.diff(arrays[2]) > 0.0):
            hplus, hcross, t_save = arrays
        else:
            t_save = input_times
            hplus, hcross = arrays[:2]
    else:
        raise ValueError(f"Unexpected number of polarization outputs: {len(arrays)}.")
    return np.asarray(t_save, dtype=float), np.asarray(hplus, dtype=float), np.asarray(hcross, dtype=float)


def generate_waveform_for_source(
    source: pd.Series | Mapping[str, float],
    output_dir: str | Path,
    *,
    config: WaveformConfig,
    force: bool = False,
) -> tuple[Path, Path]:
    """Generate and save an SMBHB waveform NPZ plus JSON metadata."""

    source_dict = dict(source)
    source_id = str(source_dict["source_id"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    waveform_path = out_dir / f"{source_id}_waveform.npz"
    metadata_path = out_dir / f"{source_id}_waveform_metadata.json"
    if waveform_path.exists() and metadata_path.exists() and not force:
        return waveform_path, metadata_path

    times = np.arange(
        -float(config.raw_half_duration_s),
        float(config.raw_half_duration_s),
        float(config.raw_dt_s),
    )
    model, used_kwargs = _construct_phenom_model(source_dict)
    t_save, hplus, hcross = _unpack_polarizations(model.compute_polarizations(times), times)

    finite = np.isfinite(t_save) & np.isfinite(hplus) & np.isfinite(hcross)
    if not np.any(finite):
        raise ValueError(f"Waveform for {source_id} contains no finite samples.")
    t_save = t_save[finite]
    hplus = hplus[finite]
    hcross = hcross[finite]

    amp = np.sqrt(hplus**2 + hcross**2)
    peak_index = int(np.argmax(amp))
    threshold = 1.0e-3 * float(np.max(amp))
    signal_idx = np.flatnonzero(amp >= threshold)
    signal_window = (
        [float(t_save[signal_idx[0]]), float(t_save[signal_idx[-1]])]
        if len(signal_idx)
        else [None, None]
    )

    np.savez_compressed(
        waveform_path,
        t=t_save,
        hplus=hplus,
        hcross=hcross,
        source_id=np.asarray(source_id),
        total_mass=np.asarray(float(source_dict["total_mass"])),
        eta=np.asarray(float(source_dict["eta"])),
        distance_mpc=np.asarray(float(source_dict["distance_mpc"])),
    )
    metadata = {
        **{key: _json_scalar(value) for key, value in source_dict.items()},
        "raw_dt_s": float(config.raw_dt_s),
        "raw_half_duration_s": float(config.raw_half_duration_s),
        "n_samples": int(len(t_save)),
        "peak_time_s": float(t_save[peak_index]),
        "max_strain": float(np.max(amp)),
        "threshold_1e-3_window_s": signal_window,
        "constructor_kwargs": used_kwargs,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return waveform_path, metadata_path


def _json_scalar(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
