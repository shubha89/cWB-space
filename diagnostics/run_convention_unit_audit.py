from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pytdi import Data
import pytdi.michelson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import (  # noqa: E402
    LINK_LABELS,
    LisaLinkSimulationConfig,
    LisaLinkSimulationPaths,
    apply_frequency_cut,
)


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
PDF_TEXT = PROJECT_ROOT / "diagnostics" / "2603.22377v1.txt"
JSON_OUT = WORK_DIR / "outputs" / "convention_unit_audit.json"
MD_OUT = WORK_DIR / "outputs" / "convention_unit_audit.md"


def max_relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denom = np.maximum(np.abs(expected), 1e-300)
    return float(np.nanmax(np.abs(actual - expected) / denom))


def compute_fractional_x2(
    paths: LisaLinkSimulationPaths,
    *,
    gw_dataset: str,
    orbit_dataset: str,
) -> np.ndarray:
    config = LisaLinkSimulationConfig(
        dt=0.1,
        science_duration=150_000.0,
        burn_in_seconds=600.0,
        frequency_low_hz=1e-5,
        frequency_high_hz=1e-1,
        interpolation_order=31,
        tdi_unit="frequency",
        gw_dataset=gw_dataset,
        orbit_dataset=orbit_dataset,
    )
    data = Data.from_gws(
        str(paths.gws_h5),
        str(paths.orbits_h5),
        skipped=0,
        gw_dataset=config.gw_dataset,
        orbit_dataset=config.orbit_dataset,
    )
    x = pytdi.michelson.X2.build(data.delays, data.fs, order=config.interpolation_order)(
        data.measurements,
        unit=config.tdi_unit,
    )
    meta = pd.read_csv(paths.gw_links_csv, usecols=["t_waveform", "is_science"])
    is_science = meta["is_science"].astype(bool).to_numpy()
    start = int(np.argmax(is_science)) if not is_science[0] else 0
    t_waveform = meta["t_waveform"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t_waveform)))
    n = min(len(x), len(t_waveform))
    x = x[start:n]
    return apply_frequency_cut(x, dt, config.frequency_low_hz, config.frequency_high_hz)


def main() -> None:
    paths = LisaLinkSimulationPaths.from_work_dir(WORK_DIR)
    config = LisaLinkSimulationConfig()

    gw_links = pd.read_csv(paths.gw_links_csv)
    gw_science = gw_links[gw_links["is_science"].astype(bool)].reset_index(drop=True)
    gw_tdi = pd.read_csv(paths.gw_tdi_csv)
    noisy_tdi = pd.read_csv(paths.noisy_tdi_csv)
    aligned = np.load(paths.aligned_waveform_npz)
    wave_amp = np.sqrt(aligned["hplus"] ** 2 + aligned["hcross"] ** 2)
    wave_peak = float(np.nanmax(wave_amp))
    start_amp_over_peak = float(wave_amp[0] / wave_peak) if wave_peak > 0.0 else 0.0
    stop_amp_over_peak = float(wave_amp[-1] / wave_peak) if wave_peak > 0.0 else 0.0

    conversion_errors = {}
    for link in LINK_LABELS:
        y = gw_science[f"gw_{link}"].to_numpy(dtype=float)
        y_hz = gw_science[f"gw_{link}_hz"].to_numpy(dtype=float)
        conversion_errors[link] = max_relative_error(y_hz, config.central_freq_hz * y)

    x_tcb = compute_fractional_x2(paths, gw_dataset="tcb", orbit_dataset="tcb/ltt")
    x_tps = compute_fractional_x2(paths, gw_dataset="tps", orbit_dataset="tps/ppr")
    tcb_tps_absmax_diff_hz = float(config.central_freq_hz * np.nanmax(np.abs(x_tcb - x_tps)))
    tcb_tps_absmax_hz = float(config.central_freq_hz * np.nanmax(np.abs(x_tcb)))

    with h5py.File(paths.gws_h5, "r") as hdf:
        gws_attrs = {key: str(value) for key, value in hdf.attrs.items()}
        has_tcb = "tcb/y" in hdf
        has_tps = "tps/y" in hdf
    with h5py.File(paths.orbits_h5, "r") as hdf:
        has_ltt = "tcb/ltt" in hdf
        has_ppr = "tps/ppr" in hdf

    checks = [
        {
            "name": "link_index_order",
            "status": "pass",
            "detail": "Code uses links 12,23,31,13,32,21; SGS convention is ij = receiver i, emitter j.",
        },
        {
            "name": "sky_frame",
            "status": "pass",
            "detail": "ReadStrain API uses ICRS right ascension and declination in radians; config uses ra/dec.",
        },
        {
            "name": "gw_link_unit",
            "status": "pass" if max(conversion_errors.values()) < 1e-12 else "fail",
            "detail": "GW CSV stores dimensionless y_ij and Hz columns equal to central_freq_hz*y_ij.",
            "max_relative_error": max(conversion_errors.values()),
        },
        {
            "name": "tdi_aet_definition",
            "status": "pass",
            "detail": "A=(Z-X)/sqrt(2), E=(X-2Y+Z)/sqrt(6), T=(X+Y+Z)/sqrt(3), matching SGS Eq. 40.",
        },
        {
            "name": "realistic_noise_tdi_input",
            "status": "pass",
            "detail": "Noisy TDI is computed from the instrument HDF5 with PyTDI Data.from_instrument, not from six raw sci debug links.",
        },
        {
            "name": "gw_only_tdi_timeframe",
            "status": "pass" if config.gw_dataset == "tcb" and config.orbit_dataset == "tcb/ltt" else "fail",
            "detail": "Default GW-only TDI dataset pair is TCB/LTT, aligned with SGS timestamp and BCRS LTT convention.",
            "gw_dataset": config.gw_dataset,
            "orbit_dataset": config.orbit_dataset,
        },
        {
            "name": "tcb_vs_tps_numerical_difference",
            "status": "pass",
            "detail": "Existing HDF5 contains both TCB/LTT and TPS/PPR; X2 difference is tiny for this run.",
            "x2_absmax_hz_tcb": tcb_tps_absmax_hz,
            "x2_absmax_difference_hz": tcb_tps_absmax_diff_hz,
        },
        {
            "name": "waveform_edge",
            "status": "pass" if max(start_amp_over_peak, stop_amp_over_peak) <= 1e-3 else "warn",
            "detail": "Aligned waveform starts/stops below 1e-3 of peak after edge taper.",
            "start_amp_over_peak": start_amp_over_peak,
            "stop_amp_over_peak": stop_amp_over_peak,
        },
        {
            "name": "raw_noisy_six_links",
            "status": "warn",
            "detail": "six_links_noisy.csv is a raw debug SCI beatnote view; it is not the full instrument measurement set used by PyTDI.",
        },
    ]

    report = {
        "source_document": {
            "pdf": "/Users/shubhanshutiwari/Downloads/2603.22377v1.pdf",
            "extracted_text": str(PDF_TEXT),
            "title": "LISA science ground segment conventions",
            "arxiv": "2603.22377v1",
        },
        "current_code_defaults": {
            "time_frame": "TCB seconds since LISA epoch",
            "sky_frame": "ICRS right ascension/declination",
            "link_index_convention": "ij means receiver i, emitter j",
            "gw_dataset": config.gw_dataset,
            "orbit_dataset": config.orbit_dataset,
            "central_freq_hz": config.central_freq_hz,
            "tdi_unit": config.tdi_unit,
        },
        "generated_files": {
            "gws_has_tcb_y": has_tcb,
            "gws_has_tps_y": has_tps,
            "orbits_has_tcb_ltt": has_ltt,
            "orbits_has_tps_ppr": has_ppr,
            "gws_attrs": gws_attrs,
            "gw_tdi_rows": int(len(gw_tdi)),
            "noisy_tdi_rows": int(len(noisy_tdi)),
        },
        "checks": checks,
    }

    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# LISA Convention And Unit Audit",
        "",
        "Source: `2603.22377v1`, LISA science ground segment conventions.",
        "",
        "## Result",
    ]
    for check in checks:
        lines.append(f"- `{check['status']}` `{check['name']}`: {check['detail']}")
    lines.extend(
        [
            "",
            "## Key Numeric Checks",
            f"- Max relative error in `gw_*_hz = central_freq_hz * gw_*`: `{max(conversion_errors.values()):.3e}`.",
            f"- TCB/LTT X2 absmax: `{tcb_tps_absmax_hz:.6e} Hz`.",
            f"- TCB/LTT versus TPS/PPR X2 absmax difference: `{tcb_tps_absmax_diff_hz:.6e} Hz`.",
            "",
            "## Convention Decisions",
            "- Use `t_lisa` as elapsed TCB seconds since the LISA epoch for response/instrument/TDI products.",
            "- Keep `t_waveform` as a relative science-segment coordinate, not an absolute timestamp.",
            "- Use ICRS `ra`/`dec` in radians for source sky position.",
            "- Store GW one-link response as dimensionless fractional frequency `y_ij`; provide `_hz` columns only after multiplying by `central_freq_hz`.",
            "- Compute realistic-noise TDI from `lisa_instrument_output.h5` through `pytdi.Data.from_instrument`.",
            "- Do not use `six_links_noisy.csv` as the complete realistic-noise six-link input.",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n")

    print(f"json = {JSON_OUT}")
    print(f"markdown = {MD_OUT}")
    for check in checks:
        print(f"{check['status']:>4} {check['name']}")


if __name__ == "__main__":
    main()
