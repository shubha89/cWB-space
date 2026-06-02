"""End-to-end two-SMBHB coronagraphic clustering prototype.

This script builds a controlled two-source test case:

1. Generate two SMBHB waveforms with different sky directions.
2. Inject their peaks at separated times on one science segment.
3. Compute each source's six-link GW response and sum the links.
4. Run lisainstrument once to add noise to the combined GW links.
5. Run multi-resolution coronagraphic kappa clustering on noisy six links.
6. Merge per-resolution clusters across WDM resolutions.
7. Compare coherent event clusters against source-specific GW-only WDM pixels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from lisagwresponse import ReadStrain
from lisainstrument import Instrument

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cluster.config import WaveformConfig
from cluster.parameters import component_masses
from cluster.waveforms import generate_waveform_for_source
from scripts.coronagraphic_multires_merge import CoronagraphicMergeConfig, run_merge
from scripts.six_link_generation import (
    LINKS,
    LINK_LABELS,
    LisaLinkSimulationConfig,
    LisaLinkSimulationPaths,
    PreparedWaveform,
    create_orbit_file,
    plot_link_columns,
    read_orbit_time_range,
)


@dataclass(frozen=True)
class TwoSMBHBEndToEndConfig:
    """Configuration for the two-SMBHB end-to-end prototype."""

    work_dir: str = "choronagraphic-cluster/runs/two_smbhb_demo/simulation"
    cluster_run_dir: str = "choronagraphic-cluster/runs/two_smbhb_demo/coronagraphic"
    science_duration_s: float = 120_000.0
    dt_s: float = 1.0
    burn_in_seconds: float = 600.0
    post_buffer_seconds: float = 600.0
    source_peak_times_s: tuple[float, float] = (35_000.0, 85_000.0)
    raw_half_duration_s: float = 180_000.0
    raw_dt_s: float = 1.0
    seed: int = 24680
    nside: int = 2
    chunk_duration_s: float = 120_000.0
    wdm_ms: tuple[int, ...] = (128, 256, 512)
    top_pixel_fraction: float = 0.10
    background_m: int = 512
    min_cluster_size: int = 5
    merge_min_cluster_size: int = 2
    merge_min_resolutions: int = 2
    source: str = "noisy"
    make_plots: bool = True
    compute_tdi: bool = False
    force: bool = False
    skip_simulation: bool = False
    skip_kappa: bool = False
    dry_run: bool = False


def _load_coronagraphic_module():
    module_path = PROJECT_ROOT / "choronagraphic-cluster" / "run_one_smbhb.py"
    spec = importlib.util.spec_from_file_location("choronagraphic_run_one_smbhb", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def default_two_source_catalog(config: TwoSMBHBEndToEndConfig) -> pd.DataFrame:
    """Return deterministic non-overlapping SMBHB source parameters."""

    rows = [
        {
            "source_id": "smbhb_A",
            "total_mass": 8.0e4,
            "eta": 0.22,
            "distance_mpc": 1.8e3,
            "s1": 0.25,
            "s2": -0.10,
            "f_min_hz": 1.0e-4,
            "iota": 0.45,
            "phi_ref": 1.2,
            "ra": 0.70,
            "dec": -0.55,
            "injected_peak_time_s": float(config.source_peak_times_s[0]),
        },
        {
            "source_id": "smbhb_B",
            "total_mass": 1.6e5,
            "eta": 0.20,
            "distance_mpc": 2.4e3,
            "s1": -0.15,
            "s2": 0.20,
            "f_min_hz": 1.0e-4,
            "iota": 1.10,
            "phi_ref": 4.1,
            "ra": 3.80,
            "dec": 0.62,
            "injected_peak_time_s": float(config.source_peak_times_s[1]),
        },
    ]
    for row in rows:
        m1, m2, q = component_masses(row["total_mass"], row["eta"])
        row["m1"] = m1
        row["m2"] = m2
        row["q"] = q
    return pd.DataFrame(rows)


def _load_waveform(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return np.asarray(data["t"], dtype=float), np.asarray(data["hplus"], dtype=float), np.asarray(data["hcross"], dtype=float)


def _place_waveform_on_science_grid(
    waveform_path: Path,
    metadata_path: Path,
    science_t: np.ndarray,
    peak_time_s: float,
) -> dict[str, object]:
    raw_t, raw_hp, raw_hc = _load_waveform(waveform_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_peak_s = float(metadata.get("peak_time_s", raw_t[int(np.argmax(np.sqrt(raw_hp**2 + raw_hc**2)))]))
    shifted_t = raw_t - raw_peak_s + float(peak_time_s)
    hp = np.interp(science_t, shifted_t, raw_hp, left=0.0, right=0.0)
    hc = np.interp(science_t, shifted_t, raw_hc, left=0.0, right=0.0)
    amp = np.sqrt(hp**2 + hc**2)
    max_amp = float(np.nanmax(amp))

    windows: dict[str, list[float | None]] = {}
    for frac in (1.0e-3, 1.0e-2, 1.0e-1):
        idx = np.flatnonzero(amp >= frac * max_amp) if max_amp > 0.0 else np.array([], dtype=int)
        key = f"window_ge_{frac:g}_max_s"
        windows[key] = [float(science_t[idx[0]]), float(science_t[idx[-1]])] if len(idx) else [None, None]

    return {
        "hplus": hp,
        "hcross": hc,
        "max_strain": max_amp,
        "prepared_peak_time_s": float(science_t[int(np.argmax(amp))]) if max_amp > 0.0 else None,
        **windows,
    }


def _build_dummy_prepared(science_t: np.ndarray, hplus_sum: np.ndarray, hcross_sum: np.ndarray, dt: float) -> PreparedWaveform:
    amp = np.sqrt(hplus_sum**2 + hcross_sum**2)
    peak_index = int(np.argmax(amp)) if np.any(amp > 0.0) else len(science_t) // 2
    return PreparedWaveform(
        t=science_t,
        tau=science_t,
        hplus=hplus_sum,
        hcross=hcross_sum,
        dt=float(dt),
        peak_index=peak_index,
        peak_time=float(science_t[peak_index]),
    )


def _simulation_config(config: TwoSMBHBEndToEndConfig) -> LisaLinkSimulationConfig:
    return LisaLinkSimulationConfig(
        dt=config.dt_s,
        science_duration=config.science_duration_s,
        burn_in_seconds=config.burn_in_seconds,
        center_waveform_peak=False,
        central_freq_hz=2.816e14,
        orbit_dt=600.0,
        orbit_duration_factor=5.0,
        seed=config.seed,
        frequency_low_hz=1.0e-5,
        frequency_high_hz=1.0e-1,
        tdi_generation=2,
        interpolation_order=31,
        tdi_unit="frequency",
        instrument_kwargs={"aafilter": None},
    )


def _run_noise_only_instrument_and_add_gw(
    lisa_config: LisaLinkSimulationConfig,
    paths: LisaLinkSimulationPaths,
    *,
    t_waveform: np.ndarray,
    t_lisa: np.ndarray,
    is_science: np.ndarray,
    y_sum_hz: np.ndarray,
    make_plots: bool,
) -> dict[str, float | int]:
    """Generate instrumental noise and add summed GW links in the CSV product.

    The current coronagraphic pipeline consumes ``six_links_noisy.csv`` directly.
    For a multi-source test this is more robust than asking lisainstrument to
    interpolate a multi-GW HDF5 file at the simulation boundaries.
    """

    if paths.instrument_h5.exists():
        paths.instrument_h5.unlink()

    dt = float(np.median(np.diff(t_waveform)))
    n_total = len(t_lisa)
    instrument_kwargs = {
        "noises_f_min_hz": lisa_config.frequency_low_hz,
        "clock_f_min_hz": lisa_config.frequency_low_hz,
        **dict(lisa_config.instrument_kwargs),
    }
    instrument = Instrument(
        size=n_total,
        dt=dt,
        t0=float(t_lisa[0]),
        orbits=str(paths.orbits_h5),
        gws=None,
        orbit_dataset=lisa_config.orbit_dataset,
        central_freq=lisa_config.central_freq_hz,
        seed=lisa_config.seed,
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
        noise_arrays = []
        for lab in LINK_LABELS:
            key = f"{group}/{lab}"
            if key not in hdf:
                raise KeyError(f"Missing dataset {key} in {paths.instrument_h5}")
            noise_arrays.append(np.asarray(hdf[key]))

    m_total = min([len(arr) for arr in noise_arrays] + [len(t_lisa), len(y_sum_hz)])
    science_idx = np.flatnonzero(is_science[:m_total])
    df = pd.DataFrame(
        {
            "sample": np.arange(len(science_idx)),
            "instrument_sample": science_idx,
            "t_waveform": t_waveform[science_idx],
            "t_lisa": t_lisa[science_idx],
        }
    )
    for j, (lab, noise) in enumerate(zip(LINK_LABELS, noise_arrays)):
        noise_science = noise[science_idx]
        gw_science = y_sum_hz[science_idx, j]
        df[f"noise_only_{lab}"] = noise_science
        df[f"gw_sum_{lab}_hz"] = gw_science
        df[f"noisy_{lab}"] = noise_science + gw_science
    df.to_csv(paths.noisy_links_csv, index=False)

    if make_plots:
        plot_link_columns(
            df["t_lisa"].to_numpy(),
            {f"noisy_{lab}": df[f"noisy_{lab}"].to_numpy() for lab in LINK_LABELS},
            "Two-SMBHB noisy six-link science carrier fluctuations",
            "noise + GW carrier fluctuation [Hz]",
            paths.noisy_links_plot,
        )

    return {
        "instrument_samples": int(n_total),
        "science_samples": int(len(science_idx)),
        "noise_added_to_gw_csv": 1,
    }


def simulate_two_smbhb_links(config: TwoSMBHBEndToEndConfig) -> dict[str, object]:
    work_dir = PROJECT_ROOT / config.work_dir
    paths = LisaLinkSimulationPaths.from_work_dir(work_dir)
    paths.make_dirs()

    if config.force and work_dir.exists():
        for path in [
            paths.gws_h5,
            paths.instrument_h5,
            paths.gw_links_csv,
            paths.noisy_links_csv,
            paths.noisy_tdi_csv,
            paths.gw_tdi_csv,
        ]:
            if path.exists():
                path.unlink()

    catalog = default_two_source_catalog(config)
    catalog_path = paths.input_dir / "two_smbhb_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    waveform_config = WaveformConfig(raw_dt_s=config.raw_dt_s, raw_half_duration_s=config.raw_half_duration_s)
    science_t = np.arange(int(round(config.science_duration_s / config.dt_s)), dtype=float) * config.dt_s

    placed_sources: list[dict[str, object]] = []
    hplus_sum = np.zeros_like(science_t)
    hcross_sum = np.zeros_like(science_t)
    for _, source in catalog.iterrows():
        waveform_path, metadata_path = generate_waveform_for_source(source, paths.input_dir, config=waveform_config, force=config.force)
        placed = _place_waveform_on_science_grid(
            waveform_path,
            metadata_path,
            science_t,
            peak_time_s=float(source["injected_peak_time_s"]),
        )
        hplus = np.asarray(placed["hplus"], dtype=float)
        hcross = np.asarray(placed["hcross"], dtype=float)
        hplus_sum += hplus
        hcross_sum += hcross
        placed_sources.append(
            {
                **source.to_dict(),
                "waveform_path": str(waveform_path),
                "waveform_metadata_path": str(metadata_path),
                "prepared_peak_time_s": placed["prepared_peak_time_s"],
                "max_strain": placed["max_strain"],
                "window_ge_1e-3_max_s": placed["window_ge_0.001_max_s"],
                "window_ge_1e-2_max_s": placed["window_ge_0.01_max_s"],
                "window_ge_1e-1_max_s": placed["window_ge_0.1_max_s"],
            }
        )
        np.savez_compressed(
            paths.input_dir / f"{source['source_id']}_prepared_on_science_grid.npz",
            t=science_t,
            hplus=hplus,
            hcross=hcross,
        )

    np.savez_compressed(paths.aligned_waveform_npz, t=science_t, hplus=hplus_sum, hcross=hcross_sum)

    lisa_config = _simulation_config(config)
    prepared = _build_dummy_prepared(science_t, hplus_sum, hcross_sum, lisa_config.dt)
    create_orbit_file(prepared, lisa_config, paths.orbits_h5)
    orbit = read_orbit_time_range(paths.orbits_h5)

    burn_in_size = int(np.ceil(lisa_config.burn_in_seconds / prepared.dt))
    n_waveform = len(prepared.t)
    post_buffer_size = int(np.ceil(config.post_buffer_seconds / prepared.dt))
    n_total = burn_in_size + n_waveform + post_buffer_size
    required_span = prepared.dt * (n_total - 1)
    orbit_start_lisa = orbit.t0 + 0.5 * (orbit.span - required_span)
    t_lisa = orbit_start_lisa + np.arange(n_total, dtype=float) * prepared.dt
    sample_index = np.arange(n_total)
    science = (sample_index >= burn_in_size) & (sample_index < burn_in_size + n_waveform)
    t_waveform_full = (np.arange(n_total, dtype=float) - burn_in_size) * prepared.dt

    if paths.gws_h5.exists():
        paths.gws_h5.unlink()

    y_sum = np.zeros((n_total, len(LINKS)), dtype=float)
    per_source_y: dict[str, np.ndarray] = {}
    for source, placed in zip(catalog.to_dict(orient="records"), placed_sources):
        prepared_path = paths.input_dir / f"{source['source_id']}_prepared_on_science_grid.npz"
        data = np.load(prepared_path)
        hplus_full = np.zeros(n_total, dtype=float)
        hcross_full = np.zeros(n_total, dtype=float)
        hplus_full[science] = np.asarray(data["hplus"], dtype=float)
        hcross_full[science] = np.asarray(data["hcross"], dtype=float)
        response = ReadStrain(
            t_lisa,
            hplus_full,
            hcross_full,
            orbits=str(paths.orbits_h5),
            ra=float(source["ra"]),
            dec=float(source["dec"]),
        )
        y = np.asarray(response.compute_gw_response(t_lisa, LINKS), dtype=float)
        if y.shape[0] != n_total:
            y = y.T
        if y.shape != y_sum.shape:
            raise ValueError(f"Unexpected GW response shape for {source['source_id']}: {y.shape}")
        y_sum += y
        per_source_y[str(source["source_id"])] = y
        response.write(str(paths.gws_h5), t0=float(t_lisa[0]), dt=prepared.dt, size=n_total, mode="a")

    y_sum_hz = lisa_config.central_freq_hz * y_sum
    df = pd.DataFrame(
        {
            "sample": np.arange(n_total),
            "t_waveform": t_waveform_full,
            "t_lisa": t_lisa,
            "is_science": science.astype(int),
        }
    )
    for j, lab in enumerate(LINK_LABELS):
        df[f"gw_{lab}"] = y_sum[:, j]
    for j, lab in enumerate(LINK_LABELS):
        df[f"gw_{lab}_hz"] = y_sum_hz[:, j]
    for source_id, y in per_source_y.items():
        for j, lab in enumerate(LINK_LABELS):
            df[f"gw_{source_id}_{lab}_hz"] = lisa_config.central_freq_hz * y[:, j]
    df.to_csv(paths.gw_links_csv, index=False)

    instrument_diag = _run_noise_only_instrument_and_add_gw(
        lisa_config,
        paths,
        t_waveform=t_waveform_full,
        t_lisa=t_lisa,
        is_science=science,
        y_sum_hz=y_sum_hz,
        make_plots=config.make_plots,
    )
    if config.compute_tdi:
        raise ValueError(
            "--compute-tdi is not supported in the current two-SMBHB runner because "
            "the instrument HDF5 is generated noise-only and the summed GW is added "
            "at the six-link CSV level for coronagraphic clustering."
        )

    metadata = {
        "config": asdict(config),
        "catalog_csv": str(catalog_path),
        "sources": placed_sources,
        # The coronagraphic prototype sees this as the first waveform metadata
        # file and centers the WDM chunk on the science segment midpoint.
        "peak_time_s": 0.0,
        "raw_half_duration_s": float(config.science_duration_s / 2.0),
        "dt_s": float(prepared.dt),
        "science_duration_s": float(config.science_duration_s),
        "science_samples": int(n_waveform),
        "burn_in_size": int(burn_in_size),
        "post_buffer_size": int(post_buffer_size),
        "n_total": int(n_total),
        "orbit_start_lisa": float(orbit_start_lisa),
        "t_lisa_min": float(t_lisa[0]),
        "t_lisa_max": float(t_lisa[-1]),
        "gw_link_max_hz": float(np.nanmax(np.abs(y_sum_hz[science]))),
        **instrument_diag,
    }
    (paths.input_dir / "00_two_smbhb_waveform_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (paths.output_dir / "two_smbhb_simulation_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if config.make_plots:
        plot_link_columns(
            df.loc[science, "t_lisa"].to_numpy(),
            {f"gw_{lab}_hz": df.loc[science, f"gw_{lab}_hz"].to_numpy() for lab in LINK_LABELS},
            "Two-SMBHB summed GW-only six-link response",
            "GW-only carrier frequency fluctuation [Hz]",
            paths.gw_links_plot,
        )

    return metadata


def run_kappa_pipeline(config: TwoSMBHBEndToEndConfig) -> dict[str, object]:
    run_dir = PROJECT_ROOT / config.cluster_run_dir
    if config.force and run_dir.exists():
        for path in [
            run_dir / "summary.json",
            run_dir / "candidate_pixels_all_resolutions.csv",
            run_dir / "pixel_clusters_all_resolutions.csv",
            run_dir / "merged_pixel_clusters.csv",
            run_dir / "multires_event_summary.csv",
        ]:
            if path.exists():
                path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "choronagraphic-cluster" / "run_one_smbhb.py"),
        "--work-dir",
        config.work_dir,
        "--output-dir",
        config.cluster_run_dir,
        "--source",
        config.source,
        "--nside",
        str(config.nside),
        "--chunk-duration-s",
        str(config.chunk_duration_s),
        "--wdm-ms",
        ",".join(str(value) for value in config.wdm_ms),
        "--top-pixel-fraction",
        str(config.top_pixel_fraction),
        "--min-cluster-size",
        str(config.min_cluster_size),
    ]
    if not config.make_plots:
        cmd.append("--no-plots")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    overlay_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "choronagraphic-cluster" / "plot_multires_wdm_overlay.py"),
        "--run-dir",
        config.cluster_run_dir,
        "--background-m",
        str(config.background_m),
    ]
    subprocess.run(overlay_cmd, cwd=PROJECT_ROOT, check=True)

    merge_summary = run_merge(
        CoronagraphicMergeConfig(
            run_dir=str(run_dir),
            min_cluster_size=config.merge_min_cluster_size,
            min_resolutions=config.merge_min_resolutions,
            make_plots=config.make_plots,
        )
    )
    return merge_summary


def _nearest_match_counts(
    event_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    *,
    time_tol_s: float,
    freq_tol_hz: float,
) -> int:
    if event_rows.empty or reference_rows.empty:
        return 0
    event_tf = event_rows[["time_s", "freq_hz"]].to_numpy(dtype=float)
    ref_tf = reference_rows[["time_s", "freq_hz"]].to_numpy(dtype=float)
    count = 0
    for time_s, freq_hz in event_tf:
        distance = np.sqrt(((ref_tf[:, 0] - time_s) / time_tol_s) ** 2 + ((ref_tf[:, 1] - freq_hz) / freq_tol_hz) ** 2)
        if np.nanmin(distance) <= 1.5:
            count += 1
    return count


def compare_gw_reference_to_clusters(config: TwoSMBHBEndToEndConfig) -> dict[str, object]:
    proto = _load_coronagraphic_module()
    run_dir = PROJECT_ROOT / config.cluster_run_dir
    work_dir = PROJECT_ROOT / config.work_dir
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run_config_data = dict(summary["config"])
    run_config_data["wdm_m"] = int(config.background_m)
    run_config = proto.RunConfig(**run_config_data)
    _, _, resample_poly, _, WDM = proto.require_dependencies()

    gw_df = pd.read_csv(work_dir / "outputs" / "six_links_gw_only.csv")
    science_df = gw_df[gw_df["is_science"].astype(bool)].reset_index(drop=True)
    t_raw = science_df["t_waveform"].to_numpy(dtype=float)
    center_s = float(t_raw[0]) + 0.5 * (float(t_raw[-1]) - float(t_raw[0]))

    merged_pixels = pd.read_csv(run_dir / "merged_pixel_clusters.csv")
    merged_pixels["event_cluster"] = merged_pixels["event_cluster"].astype(int)

    reference_rows: list[dict[str, object]] = []
    for source_idx, source_id in enumerate(["smbhb_A", "smbhb_B"]):
        links = {lab: science_df[f"gw_{source_id}_{lab}_hz"].to_numpy(dtype=float) for lab in LINK_LABELS}
        t, links_ds, dt = proto.crop_and_downsample_links(
            t_raw,
            links,
            center_s=center_s,
            duration_s=run_config.chunk_duration_s,
            target_dt_s=run_config.target_dt_s,
            resample_poly=resample_poly,
        )
        geometry = proto.LISAGeometry.equilateral(proto.LISA_ARM_M / proto.C_M_PER_S)
        michelson = proto.compute_michelson_response_from_links(
            links_ds,
            dt=dt,
            geometry=geometry,
            generation="second",
            channels="aet",
            output_domain="time",
        )
        power_a, freqs, times = proto.wdm_power(np.asarray(michelson["time"]["A"]), dt=dt, t0=float(t[0]), config=run_config, WDM=WDM)
        power_e, _, _ = proto.wdm_power(np.asarray(michelson["time"]["E"]), dt=dt, t0=float(t[0]), config=run_config, WDM=WDM)
        candidates = proto.select_candidate_pixels(
            power_a,
            power_e,
            freqs,
            times,
            top_pixels=None,
            top_pixel_fraction=config.top_pixel_fraction,
            freq_min_hz=run_config.freq_min_hz,
            freq_max_hz=run_config.freq_max_hz,
        )
        for row in candidates.to_dict(orient="records"):
            reference_rows.append({"source_id": source_id, "source_index": source_idx, **row})

    reference = pd.DataFrame(reference_rows)
    reference.to_csv(run_dir / "gw_signal_reference_pixels.csv", index=False)

    background_layer_summary = json.loads((run_dir / f"wdm_m_{config.background_m}" / "summary.json").read_text(encoding="utf-8"))
    wdm_shape = background_layer_summary["wdm_shape"]
    time_tol_s = float(config.background_m)
    freq_tol_hz = 1.0 / (2.0 * config.background_m * run_config.target_dt_s)
    comparison_rows = []
    for event_cluster in sorted(int(value) for value in merged_pixels["event_cluster"].unique() if int(value) >= 0):
        event_rows = merged_pixels[merged_pixels["event_cluster"] == event_cluster]
        row: dict[str, object] = {
            "event_cluster": event_cluster,
            "n_event_pixels": int(len(event_rows)),
            "n_resolutions": int(event_rows["wdm_m"].nunique()),
            "median_time_s": float(np.nanmedian(event_rows["time_s"])),
            "median_freq_hz": float(np.nanmedian(event_rows["freq_hz"])),
        }
        best_source = None
        best_count = -1
        for source_id in sorted(reference["source_id"].unique()):
            ref_rows = reference[reference["source_id"] == source_id]
            count = _nearest_match_counts(event_rows, ref_rows, time_tol_s=time_tol_s, freq_tol_hz=freq_tol_hz)
            row[f"matched_pixels_{source_id}"] = int(count)
            row[f"matched_fraction_{source_id}"] = float(count / max(1, len(event_rows)))
            if count > best_count:
                best_count = count
                best_source = source_id
        row["best_matching_source"] = best_source
        row["best_matched_pixels"] = int(best_count)
        row["best_matched_fraction"] = float(best_count / max(1, len(event_rows)))
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(run_dir / "event_source_comparison.csv", index=False)

    _plot_reference_comparison(run_dir, reference, merged_pixels, config, run_config, wdm_shape)
    summary_out = {
        "reference_pixels_csv": str(run_dir / "gw_signal_reference_pixels.csv"),
        "event_source_comparison_csv": str(run_dir / "event_source_comparison.csv"),
        "plot": str(run_dir / "plots" / "gw_reference_vs_coherent_events.png"),
        "background_wdm_shape": wdm_shape,
        "time_match_tolerance_s": time_tol_s,
        "freq_match_tolerance_hz": freq_tol_hz,
        "comparison": comparison.to_dict(orient="records"),
    }
    (run_dir / "gw_reference_comparison_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    return summary_out


def _plot_reference_comparison(
    run_dir: Path,
    reference: pd.DataFrame,
    merged_pixels: pd.DataFrame,
    config: TwoSMBHBEndToEndConfig,
    run_config,
    wdm_shape: list[int],
) -> None:
    import matplotlib.pyplot as plt

    background_path = run_dir / "plots" / "multires_coherent_overlay_on_wdm.png"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    coherent = merged_pixels[merged_pixels["event_cluster"] >= 0]
    if len(coherent):
        scatter = ax.scatter(
            coherent["time_s"],
            coherent["freq_hz"] * 1.0e3,
            c=coherent["event_cluster"],
            s=32,
            cmap="tab20",
            alpha=0.72,
            label="recovered coherent event pixels",
        )
        fig.colorbar(scatter, ax=ax, label="event_cluster")

    source_styles = {
        "smbhb_A": {"marker": "x", "color": "black"},
        "smbhb_B": {"marker": "+", "color": "#00e5ff"},
    }
    for source_id, rows in reference.groupby("source_id"):
        style = source_styles.get(source_id, {"marker": "x", "color": "white"})
        ax.scatter(
            rows["time_s"],
            rows["freq_hz"] * 1.0e3,
            s=42,
            marker=style["marker"],
            color=style["color"],
            linewidths=1.5,
            alpha=0.85,
            label=f"{source_id} GW-only top {config.top_pixel_fraction:.0%}",
        )

    ax.set_xlabel("time in selected chunk [s]")
    ax.set_ylabel("frequency [mHz]")
    ax.set_title("GW-only source pixels compared with recovered coherent kappa clusters")
    ax.set_ylim(0.0, min(120.0, run_config.freq_max_hz * 1.0e3))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(plot_dir / "gw_reference_vs_coherent_events.png", dpi=180)
    plt.close(fig)


def run_end_to_end(config: TwoSMBHBEndToEndConfig) -> dict[str, object]:
    if config.dry_run:
        return {
            "config": asdict(config),
            "catalog": default_two_source_catalog(config).to_dict(orient="records"),
            "message": "dry-run only; no files were generated.",
        }

    simulation_summary = None
    if not config.skip_simulation:
        simulation_summary = simulate_two_smbhb_links(config)

    merge_summary = None
    comparison_summary = None
    if not config.skip_kappa:
        merge_summary = run_kappa_pipeline(config)
        comparison_summary = compare_gw_reference_to_clusters(config)

    summary = {
        "config": asdict(config),
        "simulation_summary": simulation_summary,
        "merge_summary": merge_summary,
        "comparison_summary": comparison_summary,
    }
    run_dir = PROJECT_ROOT / config.cluster_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "two_smbhb_end_to_end_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_tuple_of_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_tuple_of_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> TwoSMBHBEndToEndConfig:
    parser = argparse.ArgumentParser(description="Run a two-SMBHB six-link/noisy WDM/kappa end-to-end prototype.")
    parser.add_argument("--work-dir", default=TwoSMBHBEndToEndConfig.work_dir)
    parser.add_argument("--cluster-run-dir", default=TwoSMBHBEndToEndConfig.cluster_run_dir)
    parser.add_argument("--science-duration-s", type=float, default=TwoSMBHBEndToEndConfig.science_duration_s)
    parser.add_argument("--dt-s", type=float, default=TwoSMBHBEndToEndConfig.dt_s)
    parser.add_argument("--burn-in-seconds", type=float, default=TwoSMBHBEndToEndConfig.burn_in_seconds)
    parser.add_argument("--post-buffer-seconds", type=float, default=TwoSMBHBEndToEndConfig.post_buffer_seconds)
    parser.add_argument("--source-peak-times-s", default="35000,85000")
    parser.add_argument("--raw-half-duration-s", type=float, default=TwoSMBHBEndToEndConfig.raw_half_duration_s)
    parser.add_argument("--seed", type=int, default=TwoSMBHBEndToEndConfig.seed)
    parser.add_argument("--nside", type=int, default=TwoSMBHBEndToEndConfig.nside)
    parser.add_argument("--chunk-duration-s", type=float, default=TwoSMBHBEndToEndConfig.chunk_duration_s)
    parser.add_argument("--wdm-ms", default="128,256,512")
    parser.add_argument("--top-pixel-fraction", type=float, default=TwoSMBHBEndToEndConfig.top_pixel_fraction)
    parser.add_argument("--background-m", type=int, default=TwoSMBHBEndToEndConfig.background_m)
    parser.add_argument("--min-cluster-size", type=int, default=TwoSMBHBEndToEndConfig.min_cluster_size)
    parser.add_argument("--merge-min-cluster-size", type=int, default=TwoSMBHBEndToEndConfig.merge_min_cluster_size)
    parser.add_argument("--merge-min-resolutions", type=int, default=TwoSMBHBEndToEndConfig.merge_min_resolutions)
    parser.add_argument("--source", choices=("noisy", "gw_only"), default=TwoSMBHBEndToEndConfig.source)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--compute-tdi", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-kappa", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    peak_times = _parse_tuple_of_floats(args.source_peak_times_s)
    if len(peak_times) != 2:
        raise ValueError("--source-peak-times-s must contain exactly two comma-separated values.")

    return TwoSMBHBEndToEndConfig(
        work_dir=args.work_dir,
        cluster_run_dir=args.cluster_run_dir,
        science_duration_s=args.science_duration_s,
        dt_s=args.dt_s,
        burn_in_seconds=args.burn_in_seconds,
        post_buffer_seconds=args.post_buffer_seconds,
        source_peak_times_s=(peak_times[0], peak_times[1]),
        raw_half_duration_s=args.raw_half_duration_s,
        seed=args.seed,
        nside=args.nside,
        chunk_duration_s=args.chunk_duration_s,
        wdm_ms=_parse_tuple_of_ints(args.wdm_ms),
        top_pixel_fraction=args.top_pixel_fraction,
        background_m=args.background_m,
        min_cluster_size=args.min_cluster_size,
        merge_min_cluster_size=args.merge_min_cluster_size,
        merge_min_resolutions=args.merge_min_resolutions,
        source=args.source,
        make_plots=not args.no_plots,
        compute_tdi=args.compute_tdi,
        force=args.force,
        skip_simulation=args.skip_simulation,
        skip_kappa=args.skip_kappa,
        dry_run=args.dry_run,
    )


def main() -> None:
    summary = run_end_to_end(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
