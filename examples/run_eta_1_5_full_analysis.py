"""Run a full 1e5-style analysis for configurable waveform parameters.

Outputs are written under:

    lisa_link_sim/<folder-name>/

Defaults reproduce the eta=0.15 run in ``lisa_link_sim/eta-1_5``. The
simulation settings match
``lisa_link_sim/1e5-data/run_1e5_lisa_link_simulation.py``. The WDM and peak
zoom plots reuse the existing 1e5 plotting scripts, redirected to the new
output directory.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import LisaLinkSimulationConfig, load_npz_waveform, run_lisa_link_simulation


RUN_FOLDER_NAME = "eta-1_5"
RUN_TAG = "eta_1_5"
RUN_ETA = 0.15
RUN_DISTANCE_MPC = 6230.0

WORK_DIR = PROJECT_ROOT / "lisa_link_sim" / RUN_FOLDER_NAME
INPUT_DIR = WORK_DIR / "inputs"
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

WAVEFORM_FILE = INPUT_DIR / f"my_waveform_{RUN_TAG}.npz"
WAVEFORM_METADATA_FILE = OUTPUT_DIR / f"my_waveform_{RUN_TAG}_metadata.json"
WAVEFORM_PLOT_FILE = OUTPUT_DIR / f"phenomxpy_waveform_{RUN_TAG}_preview.png"
SIMULATION_DIAGNOSTICS_FILE = OUTPUT_DIR / "simulation_diagnostics.json"


def configure_run(*, folder_name: str, eta: float, distance_mpc: float, tag: str | None = None) -> None:
    global RUN_FOLDER_NAME, RUN_TAG, RUN_ETA, RUN_DISTANCE_MPC
    global WORK_DIR, INPUT_DIR, OUTPUT_DIR, PLOT_DIR
    global WAVEFORM_FILE, WAVEFORM_METADATA_FILE, WAVEFORM_PLOT_FILE, SIMULATION_DIAGNOSTICS_FILE

    RUN_FOLDER_NAME = folder_name
    RUN_TAG = tag or folder_name.replace("-", "_")
    RUN_ETA = float(eta)
    RUN_DISTANCE_MPC = float(distance_mpc)

    WORK_DIR = PROJECT_ROOT / "lisa_link_sim" / RUN_FOLDER_NAME
    INPUT_DIR = WORK_DIR / "inputs"
    OUTPUT_DIR = WORK_DIR / "outputs"
    PLOT_DIR = OUTPUT_DIR / "plots"
    WAVEFORM_FILE = INPUT_DIR / f"my_waveform_{RUN_TAG}.npz"
    WAVEFORM_METADATA_FILE = OUTPUT_DIR / f"my_waveform_{RUN_TAG}_metadata.json"
    WAVEFORM_PLOT_FILE = OUTPUT_DIR / f"phenomxpy_waveform_{RUN_TAG}_preview.png"
    SIMULATION_DIAGNOSTICS_FILE = OUTPUT_DIR / "simulation_diagnostics.json"


def build_phenom_model():
    from phenomxpy import IMRPhenomTHM

    eta = RUN_ETA
    total_mass = 1.0e5
    distance = RUN_DISTANCE_MPC
    s1 = 0.3
    s2 = -0.2
    f_min = 1.0e-4
    iota = 0.7
    phi_ref = 0.0

    constructor_attempts = [
        dict(eta=eta, total_mass=total_mass, distance=distance, s1=s1, s2=s2, f_min=f_min, iota=iota, phi_ref=phi_ref),
        dict(
            eta=eta,
            total_mass=total_mass,
            distance=distance,
            s1=s1,
            s2=s2,
            f_min=f_min,
            inclination=iota,
            phi_ref=phi_ref,
        ),
        dict(eta=eta, total_mass=total_mass, distance=distance, s1=s1, s2=s2, f_min=f_min),
    ]
    last_error = None
    for kwargs in constructor_attempts:
        try:
            return IMRPhenomTHM(**kwargs), kwargs
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not construct IMRPhenomTHM for eta={RUN_ETA}") from last_error


def unpack_polarizations(out, input_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(out, (tuple, list)):
        raise TypeError(f"Expected tuple/list from compute_polarizations, got {type(out)}")
    arrays = [np.asarray(item) for item in out]
    if len(arrays) == 2:
        hplus, hcross = arrays
        t_save = input_times
    elif len(arrays) == 3:
        if arrays[0].shape == input_times.shape and np.all(np.diff(arrays[0]) > 0):
            t_save, hplus, hcross = arrays
        elif arrays[2].shape == input_times.shape and np.all(np.diff(arrays[2]) > 0):
            hplus, hcross, t_save = arrays
        else:
            t_save = input_times
            hplus, hcross = arrays[:2]
    else:
        raise ValueError(f"Unexpected number of polarization outputs: {len(arrays)}")

    t_save = np.asarray(t_save, dtype=float)
    hplus = np.asarray(hplus, dtype=float)
    hcross = np.asarray(hcross, dtype=float)
    if not (len(t_save) == len(hplus) == len(hcross)):
        raise ValueError("Unpacked waveform arrays do not have matching lengths")
    return t_save, hplus, hcross


def generate_waveform(*, force: bool = False) -> None:
    if WAVEFORM_FILE.exists() and WAVEFORM_METADATA_FILE.exists() and not force:
        print(f"waveform exists, skipping: {WAVEFORM_FILE}")
        return

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_dt = 1.0
    raw_half_duration = 300_000.0
    times = np.arange(-raw_half_duration, raw_half_duration, raw_dt)

    phen, used_kwargs = build_phenom_model()
    out = phen.compute_polarizations(times)
    t_save, hplus, hcross = unpack_polarizations(out, times)

    bad = ~np.isfinite(t_save) | ~np.isfinite(hplus) | ~np.isfinite(hcross)
    if np.any(bad):
        keep = ~bad
        t_save = t_save[keep]
        hplus = hplus[keep]
        hcross = hcross[keep]

    amp = np.sqrt(hplus**2 + hcross**2)
    peak_index = int(np.argmax(amp))
    peak_time = float(t_save[peak_index])
    threshold = 1e-3 * float(np.max(amp))
    idx = np.flatnonzero(amp >= threshold)
    signal_window = [float(t_save[idx[0]]), float(t_save[idx[-1]])] if len(idx) else [None, None]

    np.savez(
        WAVEFORM_FILE,
        t=t_save,
        hplus=hplus,
        hcross=hcross,
        eta=np.asarray(RUN_ETA),
        total_mass=np.asarray(1.0e5),
        distance_mpc=np.asarray(RUN_DISTANCE_MPC),
        f_min_hz=np.asarray(1.0e-4),
    )

    metadata = {
        "eta": RUN_ETA,
        "total_mass": 1.0e5,
        "distance_mpc": RUN_DISTANCE_MPC,
        "s1": 0.3,
        "s2": -0.2,
        "f_min_hz": 1.0e-4,
        "iota": 0.7,
        "phi_ref": 0.0,
        "raw_dt_s": raw_dt,
        "raw_half_duration_s": raw_half_duration,
        "n_samples": int(len(t_save)),
        "peak_time_s": peak_time,
        "max_strain": float(np.max(amp)),
        "threshold_1e-3_window_s": signal_window,
        "constructor_kwargs": used_kwargs,
    }
    WAVEFORM_METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    plt.figure(figsize=(12, 4))
    stride = max(1, len(t_save) // 30_000)
    plt.plot(t_save[::stride], hplus[::stride], lw=0.8, label="hplus")
    plt.plot(t_save[::stride], hcross[::stride], lw=0.8, label="hcross")
    plt.axvline(peak_time, color="k", ls="--", alpha=0.7, label="peak")
    plt.xlabel("time [s]")
    plt.ylabel("strain")
    plt.title(f"IMRPhenomTHM waveform preview, eta={RUN_ETA:g}, distance={RUN_DISTANCE_MPC:g} Mpc")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(WAVEFORM_PLOT_FILE, dpi=150)
    plt.close()

    print(f"saved waveform: {WAVEFORM_FILE}")
    print(f"saved metadata: {WAVEFORM_METADATA_FILE}")
    print(f"saved preview:  {WAVEFORM_PLOT_FILE}")


def run_link_tdi_pipeline(*, force: bool = False) -> None:
    if SIMULATION_DIAGNOSTICS_FILE.exists() and not force:
        print(f"simulation diagnostics exist, skipping: {SIMULATION_DIAGNOSTICS_FILE}")
        return

    t, hplus, hcross = load_npz_waveform(WAVEFORM_FILE)
    config = LisaLinkSimulationConfig(
        dt=0.1,
        science_duration=150_000.0,
        burn_in_seconds=600.0,
        center_waveform_peak=True,
        ra=np.pi / 4.0,
        dec=0.3,
        central_freq_hz=2.816e14,
        orbit_name="esa-trailing",
        orbit_dt=600.0,
        orbit_duration_factor=5.0,
        seed=12345,
        frequency_low_hz=1e-5,
        frequency_high_hz=1e-1,
        tdi_generation=2,
        interpolation_order=31,
        tdi_unit="frequency",
    )
    products = run_lisa_link_simulation(
        t,
        hplus,
        hcross,
        work_dir=WORK_DIR,
        config=config,
        make_plots=False,
    )
    SIMULATION_DIAGNOSTICS_FILE.write_text(json.dumps(dict(products.diagnostics), indent=2))
    print(f"saved diagnostics: {SIMULATION_DIAGNOSTICS_FILE}")


def run_1e5_style_wdm_and_zoom_plots(*, force: bool = False) -> None:
    final_zoom = PLOT_DIR / "12_1e5_multires_wdm_AE_Twhiten_peak_zoom.png"
    if final_zoom.exists() and not force:
        print(f"final zoom plot exists, skipping: {final_zoom}")
        return

    helper_dir = PROJECT_ROOT / "lisa_link_sim" / "1e5-data"
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))

    make_wdm = importlib.import_module("make_1e5_plots_and_wdm")
    make_wdm.WORK_DIR = WORK_DIR
    make_wdm.OUTPUT_DIR = OUTPUT_DIR
    make_wdm.PLOT_DIR = PLOT_DIR
    make_wdm.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    make_wdm.main()

    peak_marker = importlib.import_module("make_1e5_wdm_peak_marker_plot")
    peak_marker.WORK_DIR = WORK_DIR
    peak_marker.OUTPUT_DIR = OUTPUT_DIR
    peak_marker.PLOT_DIR = PLOT_DIR
    peak_marker.MAPS_FILE = OUTPUT_DIR / "1e5_wdm_multires_maps.npz"
    peak_marker.METADATA_FILE = OUTPUT_DIR / "1e5_plot_and_wdm_metadata.json"
    peak_marker.main()

    twhiten = importlib.import_module("make_1e5_wdm_twhiten_peak_marker_plot")
    twhiten.OUTPUT_DIR = OUTPUT_DIR
    twhiten.PLOT_DIR = PLOT_DIR
    twhiten.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    twhiten.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full 1e5-style LISA/WDM analysis under lisa_link_sim/<folder-name>.")
    parser.add_argument("--folder-name", default=RUN_FOLDER_NAME, help="Output folder name under lisa_link_sim/.")
    parser.add_argument("--tag", default=None, help="Filename tag. Defaults to folder-name with '-' replaced by '_'.")
    parser.add_argument("--eta", type=float, default=RUN_ETA, help="Symmetric mass ratio.")
    parser.add_argument("--distance-mpc", type=float, default=RUN_DISTANCE_MPC, help="Luminosity distance in Mpc.")
    parser.add_argument("--force", action="store_true", help="Recompute stages even when output files already exist.")
    args = parser.parse_args()

    configure_run(folder_name=args.folder_name, eta=args.eta, distance_mpc=args.distance_mpc, tag=args.tag)
    print(f"work_dir: {WORK_DIR}")
    print(f"eta: {RUN_ETA}")
    print(f"distance_mpc: {RUN_DISTANCE_MPC}")
    generate_waveform(force=args.force)
    run_link_tdi_pipeline(force=args.force)
    run_1e5_style_wdm_and_zoom_plots(force=args.force)
    print(f"analysis complete: {RUN_FOLDER_NAME}")


if __name__ == "__main__":
    main()
