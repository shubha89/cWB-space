"""Prototype coronagraphic-TDI pixel clustering for one SMBHB simulation.

The prototype:
- selects high-power A/E WDM pixels from one source,
- scans coronagraphic kappa over a HEALPix sky grid,
- assigns each selected WDM pixel to its best-null sky direction,
- clusters pixels by the resulting sky-response features.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coronagraphic_tdi import (  # noqa: E402
    LISAGeometry,
    _freq_to_time_plus,
    _time_to_freq_plus,
    build_sagnac_from_links,
    coronagraphic_kappa,
)

LINK_LABELS = ("12", "23", "31", "13", "32", "21")
C_M_PER_S = 299_792_458.0
LISA_ARM_M = 2.5e9


@dataclass(frozen=True)
class RunConfig:
    work_dir: str
    output_dir: str
    source: str
    nside: int
    chunk_duration_s: float
    target_dt_s: float
    wdm_m: int
    wdm_k: int
    wdm_beta_order: int
    wdm_precision: int
    wdm_backend: str
    top_pixels: int
    freq_min_hz: float
    freq_max_hz: float
    min_cluster_size: int
    min_samples: int | None
    make_plots: bool
    wdm_ms: tuple[int, ...] | None = None
    top_pixel_fraction: float | None = None
    selection_normalization: str = "raw"
    offsource_edge_fraction: float = 0.20
    wdm_aet_source: str = "pytdi"
    run_hdbscan: bool = True
    selection_start_guard_s: float = 0.0
    selection_end_guard_s: float = 0.0


def require_dependencies():
    try:
        import healpy as hp
        import hdbscan
        from scipy.signal import resample_poly
        from sklearn.preprocessing import StandardScaler
        from wdm_wavelet.wdm import WDM
    except Exception as exc:  # pragma: no cover - depends on environment.
        raise ModuleNotFoundError(
            "This prototype requires healpy, hdbscan, scipy, scikit-learn, and wdm-wavelet "
            "inside the lisa-tf environment."
        ) from exc
    return hp, hdbscan, resample_poly, StandardScaler, WDM


def load_source_metadata(work_dir: Path) -> dict:
    metadata_files = sorted((work_dir / "inputs").glob("*_waveform_metadata.json"))
    if not metadata_files:
        return {}
    return json.loads(metadata_files[0].read_text(encoding="utf-8"))


def load_links(work_dir: Path, source: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if source == "gw_only":
        path = work_dir / "outputs" / "six_links_gw_only.csv"
        cols = ["t_waveform", "is_science", *[f"gw_{link}_hz" for link in LINK_LABELS]]
        df = pd.read_csv(path, usecols=cols)
        df = df[df["is_science"].astype(bool)].reset_index(drop=True)
        links = {link: df[f"gw_{link}_hz"].to_numpy(dtype=float) for link in LINK_LABELS}
    elif source == "noisy":
        path = work_dir / "outputs" / "six_links_noisy.csv"
        cols = ["t_waveform", *[f"noisy_{link}" for link in LINK_LABELS]]
        df = pd.read_csv(path, usecols=cols)
        links = {link: df[f"noisy_{link}"].to_numpy(dtype=float) for link in LINK_LABELS}
    else:
        raise ValueError("source must be 'gw_only' or 'noisy'.")
    return df["t_waveform"].to_numpy(dtype=float), links


def load_pytdi_aet(work_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source == "gw_only":
        path = work_dir / "outputs" / "tdi_gw_only_XYZ_AET.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing pyTDI GW-only A/E file: {path}")
        df = pd.read_csv(path)
        a_col = next((col for col in df.columns if col.startswith("A") and col.endswith("_gw_hz")), None)
        e_col = next((col for col in df.columns if col.startswith("E") and col.endswith("_gw_hz")), None)
    elif source == "noisy":
        path = work_dir / "outputs" / "tdi_michelson_XYZ_AET.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing pyTDI noisy A/E file: {path}")
        df = pd.read_csv(path)
        a_col = next((col for col in df.columns if col.startswith("A") and not col.endswith("_gw_hz")), None)
        e_col = next((col for col in df.columns if col.startswith("E") and not col.endswith("_gw_hz")), None)
    else:
        raise ValueError("source must be 'gw_only' or 'noisy'.")

    if a_col is None or e_col is None:
        raise ValueError(f"Could not identify A/E columns in {path}.")
    return (
        df["t_waveform"].to_numpy(dtype=float),
        df[a_col].to_numpy(dtype=float),
        df[e_col].to_numpy(dtype=float),
    )


def crop_and_downsample_links(
    t: np.ndarray,
    links: dict[str, np.ndarray],
    *,
    center_s: float,
    duration_s: float,
    target_dt_s: float,
    resample_poly,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    dt = float(np.median(np.diff(t)))
    start_s = max(float(t[0]), center_s - 0.5 * duration_s)
    stop_s = min(float(t[-1]), center_s + 0.5 * duration_s)
    mask = (t >= start_s) & (t <= stop_s)
    if int(np.sum(mask)) < 1024:
        raise ValueError("Selected chunk is too short for the WDM/coronagraphic scan.")

    t_chunk = t[mask]
    links_chunk = {link: values[mask] for link, values in links.items()}
    factor = int(round(target_dt_s / dt))
    if factor > 1:
        if not np.isclose(factor * dt, target_dt_s, rtol=1.0e-6, atol=1.0e-9):
            raise ValueError("target_dt_s must be an integer multiple of the source sample spacing.")
        links_ds = {link: resample_poly(values, up=1, down=factor) for link, values in links_chunk.items()}
        n = min(len(values) for values in links_ds.values())
        links_ds = {link: values[:n] for link, values in links_ds.items()}
        t_ds = t_chunk[0] + np.arange(n, dtype=float) * factor * dt
        return t_ds, links_ds, factor * dt
    return t_chunk, links_chunk, dt


def crop_and_downsample_aet(
    t: np.ndarray,
    a_t: np.ndarray,
    e_t: np.ndarray,
    *,
    center_s: float,
    duration_s: float,
    target_dt_s: float,
    resample_poly,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dt = float(np.median(np.diff(t)))
    start_s = max(float(t[0]), center_s - 0.5 * duration_s)
    stop_s = min(float(t[-1]), center_s + 0.5 * duration_s)
    mask = (t >= start_s) & (t <= stop_s)
    if int(np.sum(mask)) < 1024:
        raise ValueError("Selected pyTDI A/E chunk is too short for WDM selection.")

    t_chunk = t[mask]
    a_chunk = np.asarray(a_t[mask], dtype=float)
    e_chunk = np.asarray(e_t[mask], dtype=float)
    factor = int(round(target_dt_s / dt))
    if factor > 1:
        if not np.isclose(factor * dt, target_dt_s, rtol=1.0e-6, atol=1.0e-9):
            raise ValueError("target_dt_s must be an integer multiple of the pyTDI sample spacing.")
        a_ds = resample_poly(a_chunk, up=1, down=factor)
        e_ds = resample_poly(e_chunk, up=1, down=factor)
        n = min(len(a_ds), len(e_ds))
        t_ds = t_chunk[0] + np.arange(n, dtype=float) * factor * dt
        return t_ds, a_ds[:n], e_ds[:n], factor * dt
    return t_chunk, a_chunk, e_chunk, dt


def wdm_power(x: np.ndarray, *, dt: float, t0: float, config: RunConfig, WDM):
    wdm = WDM(
        M=config.wdm_m,
        K=config.wdm_k,
        beta_order=config.wdm_beta_order,
        precision=config.wdm_precision,
        backend=config.wdm_backend,
    )
    min_len = max(len(x), 2 * int(wdm.m_H) + 1)
    if len(x) < min_len:
        x = np.pad(x, (0, min_len - len(x)), mode="constant")
    tf = wdm.t2w(np.asarray(np.real(x), dtype=float), sample_rate=1.0 / dt, t0=t0)
    power = np.abs(tf.data) ** 2
    freqs = np.arange(tf.n_freq, dtype=float) * tf.df
    times = tf.t0 + (np.arange(tf.n_time, dtype=float) + 0.5) * tf.dt
    return power, freqs, times


def select_candidate_pixels(
    power_a: np.ndarray,
    power_e: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    top_pixels: int | None,
    top_pixel_fraction: float | None,
    freq_min_hz: float,
    freq_max_hz: float,
    selection_normalization: str,
    offsource_edge_fraction: float,
    selection_start_guard_s: float,
    selection_end_guard_s: float,
) -> pd.DataFrame:
    ae_power = np.asarray(power_a) + np.asarray(power_e)
    selection_power, background = normalize_selection_power(
        ae_power,
        mode=selection_normalization,
        offsource_edge_fraction=offsource_edge_fraction,
    )
    freq_mask = (freqs >= freq_min_hz) & (freqs <= freq_max_hz)
    if not np.any(freq_mask):
        raise ValueError("No WDM frequency bins survive the requested frequency range.")
    time_mask = np.ones_like(times, dtype=bool)
    if selection_start_guard_s > 0.0:
        time_mask &= times >= float(times[0]) + float(selection_start_guard_s)
    if selection_end_guard_s > 0.0:
        time_mask &= times <= float(times[-1]) - float(selection_end_guard_s)
    if not np.any(time_mask):
        raise ValueError("No WDM time bins survive the requested selection time guards.")
    valid = np.zeros_like(ae_power, dtype=bool)
    valid[np.ix_(freq_mask, time_mask)] = np.isfinite(selection_power[np.ix_(freq_mask, time_mask)])
    coords = np.argwhere(valid)
    values = selection_power[valid]
    if len(values) == 0:
        raise ValueError("No finite WDM pixels survive the requested frequency range.")
    if top_pixel_fraction is not None:
        if not (0.0 < top_pixel_fraction <= 1.0):
            raise ValueError("top_pixel_fraction must be in the interval (0, 1].")
        n_keep = int(np.ceil(float(top_pixel_fraction) * len(values)))
    else:
        if top_pixels is None:
            raise ValueError("Either top_pixels or top_pixel_fraction must be provided.")
        n_keep = int(top_pixels)
    n_keep = max(1, min(n_keep, len(values)))
    order = np.argsort(values)[-n_keep:][::-1]
    rows = []
    for pixel_id, idx in enumerate(order):
        f_idx, t_idx = coords[idx]
        rows.append(
            {
                "pixel_id": pixel_id,
                "selection_rank": pixel_id + 1,
                "freq_idx": int(f_idx),
                "time_idx": int(t_idx),
                "freq_hz": float(freqs[f_idx]),
                "time_s": float(times[t_idx]),
                "ae_power": float(ae_power[f_idx, t_idx]),
                "selection_power": float(selection_power[f_idx, t_idx]),
                "selection_background": float(background[f_idx]),
                "selection_normalization": selection_normalization,
                "selection_start_guard_s": float(selection_start_guard_s),
                "selection_end_guard_s": float(selection_end_guard_s),
                "n_valid_pixels": int(len(values)),
                "selected_fraction": float(n_keep / len(values)),
            }
        )
    return pd.DataFrame(rows)


def normalize_selection_power(
    ae_power: np.ndarray,
    *,
    mode: str,
    offsource_edge_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the WDM image used for candidate ranking and its frequency background."""

    power = np.asarray(ae_power, dtype=float)
    eps = 1.0e-300
    if mode == "raw":
        return power.copy(), np.ones(power.shape[0], dtype=float)

    if mode not in {"frequency-median", "offsource-frequency-median"}:
        raise ValueError("selection_normalization must be raw, frequency-median, or offsource-frequency-median.")

    if mode == "offsource-frequency-median":
        if not (0.0 < float(offsource_edge_fraction) <= 0.5):
            raise ValueError("offsource_edge_fraction must be in (0, 0.5].")
        n_time = power.shape[1]
        n_edge = max(1, int(round(float(offsource_edge_fraction) * n_time)))
        bg_values = np.concatenate([power[:, :n_edge], power[:, -n_edge:]], axis=1)
    else:
        bg_values = power

    background = np.nanmedian(bg_values, axis=1)
    background = np.where(np.isfinite(background) & (background > 0.0), background, eps)
    return power / background[:, None], background


def sky_grid(nside: int, hp) -> pd.DataFrame:
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix, dtype=int)
    theta, phi = hp.pix2ang(nside, ipix)
    dec = 0.5 * np.pi - theta
    ra = phi
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return pd.DataFrame({"healpix": ipix, "ra": ra, "dec": dec, "x": x, "y": y, "z": z})


def angular_separation_deg(ra1: float, dec1: float, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    cos_sep = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))


def scan_sky_for_pixels(
    sagnac_frequency_series: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    t: np.ndarray,
    dt: float,
    candidates: pd.DataFrame,
    sky: pd.DataFrame,
    geometry: LISAGeometry,
    config: RunConfig,
    WDM,
) -> tuple[np.ndarray, pd.DataFrame]:
    freqs_fft, alpha_f, beta_f, gamma_f = sagnac_frequency_series
    coords = candidates[["freq_idx", "time_idx"]].to_numpy(dtype=int)
    responses = np.full((len(candidates), len(sky)), np.nan, dtype=np.float64)
    for j, row in sky.iterrows():
        try:
            kappa_f = coronagraphic_kappa(
                freqs_fft,
                float(row["dec"]),
                float(row["ra"]),
                geometry,
                alpha_f,
                beta_f,
                gamma_f,
            )
            kappa_t = _freq_to_time_plus(kappa_f)
            power, _, _ = wdm_power(kappa_t, dt=dt, t0=float(t[0]), config=config, WDM=WDM)
            responses[:, j] = power[coords[:, 0], coords[:, 1]]
        except Exception:
            responses[:, j] = np.nan

    feature_rows = []
    eps = 1.0e-300
    for i, cand in candidates.iterrows():
        values = responses[i]
        finite = np.isfinite(values)
        if not np.any(finite):
            best_idx = -1
            best = np.nan
            median = np.nan
            contrast = np.nan
            sharpness = np.nan
        else:
            finite_values = values[finite]
            finite_indices = np.flatnonzero(finite)
            local_best = int(np.nanargmin(finite_values))
            best_idx = int(finite_indices[local_best])
            best = float(values[best_idx])
            median = float(np.nanmedian(finite_values))
            p10 = float(np.nanpercentile(finite_values, 10.0))
            contrast = median / max(best, eps)
            sharpness = p10 / max(best, eps)
        sky_row = sky.iloc[best_idx] if best_idx >= 0 else None
        feature_rows.append(
            {
                **cand.to_dict(),
                "best_healpix": int(sky_row["healpix"]) if sky_row is not None else -1,
                "best_ra": float(sky_row["ra"]) if sky_row is not None else np.nan,
                "best_dec": float(sky_row["dec"]) if sky_row is not None else np.nan,
                "best_x": float(sky_row["x"]) if sky_row is not None else np.nan,
                "best_y": float(sky_row["y"]) if sky_row is not None else np.nan,
                "best_z": float(sky_row["z"]) if sky_row is not None else np.nan,
                "kappa_min_power": best,
                "kappa_median_power": median,
                "sky_contrast": contrast,
                "sky_sharpness": sharpness,
            }
        )
    return responses, pd.DataFrame(feature_rows)


def build_sagnac_frequency_series(
    links: dict[str, np.ndarray],
    t: np.ndarray,
    dt: float,
    geometry: LISAGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freqs_fft = np.fft.fftfreq(len(t), d=dt)
    y_f = {link: _time_to_freq_plus(values) for link, values in links.items()}
    alpha_f = build_sagnac_from_links("alpha", freqs_fft, y_f, geometry)
    beta_f = build_sagnac_from_links("beta", freqs_fft, y_f, geometry)
    gamma_f = build_sagnac_from_links("gamma", freqs_fft, y_f, geometry)
    return freqs_fft, alpha_f, beta_f, gamma_f


def cluster_pixel_features(features: pd.DataFrame, *, min_cluster_size: int, min_samples: int | None, hdbscan, StandardScaler):
    feature_cols = ["best_x", "best_y", "best_z", "kappa_min_power", "sky_contrast", "sky_sharpness"]
    x = features[feature_cols].to_numpy(dtype=float).copy()
    x[:, 3:] = np.log10(np.maximum(x[:, 3:], 1.0e-300))
    valid = np.all(np.isfinite(x), axis=1)
    labels = np.full(len(features), -1, dtype=int)
    probabilities = np.zeros(len(features), dtype=float)
    if int(np.sum(valid)) >= max(2, min_cluster_size):
        scaled = StandardScaler().fit_transform(x[valid])
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels_valid = clusterer.fit_predict(scaled)
        labels[valid] = labels_valid
        probabilities[valid] = getattr(clusterer, "probabilities_", np.zeros(np.sum(valid)))

    method = "hdbscan"
    if np.all(labels < 0):
        method = "best_healpix_fallback"
        labels = np.full(len(features), -1, dtype=int)
        probabilities = np.zeros(len(features), dtype=float)
        cluster_id = 0
        counts = features["best_healpix"].value_counts()
        for healpix, count in counts.items():
            if int(healpix) < 0 or int(count) < min_cluster_size:
                continue
            mask = features["best_healpix"].to_numpy(dtype=int) == int(healpix)
            labels[mask] = cluster_id
            probabilities[mask] = 1.0
            cluster_id += 1

    out = features.copy()
    out["cluster"] = labels
    out["cluster_probability"] = probabilities
    out["cluster_method"] = method
    return out


def make_plots(features: pd.DataFrame, sky: pd.DataFrame, source_meta: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(features["time_s"], features["freq_hz"], c=features["cluster"], s=35, cmap="tab20")
    plt.xlabel("time in selected chunk [s]")
    plt.ylabel("frequency [Hz]")
    plt.title("Selected WDM pixels colored by coronagraphic-sky cluster")
    plt.colorbar(scatter, label="cluster")
    plt.tight_layout()
    plt.savefig(plot_dir / "wdm_pixels_by_cluster.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.scatter(np.degrees(features["best_ra"]), np.degrees(features["best_dec"]), c=features["cluster"], s=35, cmap="tab20")
    if "ra" in source_meta and "dec" in source_meta:
        plt.scatter([np.degrees(source_meta["ra"])], [np.degrees(source_meta["dec"])], marker="*", s=180, c="black", label="injected")
        plt.legend(loc="best")
    plt.xlabel("best RA [deg]")
    plt.ylabel("best Dec [deg]")
    plt.title("Best coronagraphic-null sky direction per WDM pixel")
    plt.tight_layout()
    plt.savefig(plot_dir / "best_sky_by_cluster.png", dpi=160)
    plt.close()


def active_wdm_ms(config: RunConfig) -> tuple[int, ...]:
    if config.wdm_ms is None:
        return (int(config.wdm_m),)
    values = tuple(int(value) for value in config.wdm_ms)
    if not values:
        raise ValueError("At least one WDM resolution must be requested.")
    if any(value <= 0 for value in values):
        raise ValueError("All WDM M values must be positive integers.")
    return values


def run_wdm_layer(
    *,
    config: RunConfig,
    a_t: np.ndarray,
    e_t: np.ndarray,
    selection_t: np.ndarray,
    selection_dt: float,
    sagnac_frequency_series: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    t: np.ndarray,
    dt: float,
    geometry: LISAGeometry,
    metadata: dict,
    sky: pd.DataFrame,
    output_dir: Path,
    hdbscan,
    StandardScaler,
    WDM,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    power_a, wdm_freqs, wdm_times = wdm_power(
        a_t,
        dt=selection_dt,
        t0=float(selection_t[0]),
        config=config,
        WDM=WDM,
    )
    power_e, _, _ = wdm_power(
        e_t,
        dt=selection_dt,
        t0=float(selection_t[0]),
        config=config,
        WDM=WDM,
    )
    candidates = select_candidate_pixels(
        power_a,
        power_e,
        wdm_freqs,
        wdm_times,
        top_pixels=config.top_pixels,
        top_pixel_fraction=config.top_pixel_fraction,
        freq_min_hz=config.freq_min_hz,
        freq_max_hz=config.freq_max_hz,
        selection_normalization=config.selection_normalization,
        offsource_edge_fraction=config.offsource_edge_fraction,
        selection_start_guard_s=config.selection_start_guard_s,
        selection_end_guard_s=config.selection_end_guard_s,
    )
    candidates.insert(0, "wdm_m", int(config.wdm_m))

    responses, features = scan_sky_for_pixels(
        sagnac_frequency_series,
        t,
        dt,
        candidates,
        sky,
        geometry,
        config,
        WDM,
    )

    if config.run_hdbscan:
        clustered = cluster_pixel_features(
            features,
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            hdbscan=hdbscan,
            StandardScaler=StandardScaler,
        )
    else:
        clustered = features.copy()
        clustered["cluster"] = -1
        clustered["cluster_probability"] = 0.0
        clustered["cluster_method"] = "not_run"

    if "ra" in metadata and "dec" in metadata:
        clustered["true_sky_sep_deg"] = angular_separation_deg(
            float(metadata["ra"]),
            float(metadata["dec"]),
            clustered["best_ra"].to_numpy(dtype=float),
            clustered["best_dec"].to_numpy(dtype=float),
        )

    candidates.to_csv(output_dir / "candidate_pixels.csv", index=False)
    sky.to_csv(output_dir / "healpix_sky_grid.csv", index=False)
    features.to_csv(output_dir / "kappa_features.csv", index=False)
    clustered.to_csv(output_dir / "pixel_clusters.csv", index=False)
    np.savez_compressed(output_dir / "kappa_sky_responses.npz", responses=responses)

    cluster_counts = clustered["cluster"].value_counts().sort_index().to_dict()
    summary = {
        "config": asdict(config),
        "wdm_m": int(config.wdm_m),
        "wdm_shape": [int(power_a.shape[0]), int(power_a.shape[1])],
        "n_valid_pixels": int(candidates["n_valid_pixels"].iloc[0]) if len(candidates) else 0,
        "selected_fraction": float(candidates["selected_fraction"].iloc[0]) if len(candidates) else None,
        "n_candidates": int(len(candidates)),
        "n_sky_directions": int(len(sky)),
        "chunk_start_s": float(t[0]),
        "chunk_stop_s": float(t[-1]),
        "dt_s": float(dt),
        "selection_chunk_start_s": float(selection_t[0]),
        "selection_chunk_stop_s": float(selection_t[-1]),
        "selection_dt_s": float(selection_dt),
        "wdm_aet_source": "pytdi",
        "run_hdbscan": bool(config.run_hdbscan),
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
        "source_ra": metadata.get("ra"),
        "source_dec": metadata.get("dec"),
        "median_true_sky_sep_deg": (
            float(np.nanmedian(clustered["true_sky_sep_deg"]))
            if "true_sky_sep_deg" in clustered
            else None
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if config.make_plots:
        make_plots(clustered, sky, metadata, output_dir)
    return summary, candidates, clustered


def run(config: RunConfig) -> dict:
    hp, hdbscan, resample_poly, StandardScaler, WDM = require_dependencies()
    if config.wdm_aet_source != "pytdi":
        raise ValueError("This pipeline only supports pyTDI-computed A/E for WDM pixel selection.")
    work_dir = Path(config.work_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wdm_ms = active_wdm_ms(config)
    multi_resolution = len(wdm_ms) > 1

    metadata = load_source_metadata(work_dir)
    t_raw, link_raw = load_links(work_dir, config.source)
    center_s = float(metadata.get("raw_half_duration_s", 150_000.0))
    if "peak_time_s" in metadata:
        # Simulation pipeline centers the waveform peak in the science segment.
        center_s = float(t_raw[0]) + 0.5 * (float(t_raw[-1]) - float(t_raw[0]))

    t, links, dt = crop_and_downsample_links(
        t_raw,
        link_raw,
        center_s=center_s,
        duration_s=config.chunk_duration_s,
        target_dt_s=config.target_dt_s,
        resample_poly=resample_poly,
    )

    geometry = LISAGeometry.equilateral(LISA_ARM_M / C_M_PER_S)
    t_aet_raw, pytdi_a_raw, pytdi_e_raw = load_pytdi_aet(work_dir, config.source)
    selection_t, selection_a_t, selection_e_t, selection_dt = crop_and_downsample_aet(
        t_aet_raw,
        pytdi_a_raw,
        pytdi_e_raw,
        center_s=center_s,
        duration_s=config.chunk_duration_s,
        target_dt_s=config.target_dt_s,
        resample_poly=resample_poly,
    )

    sagnac_frequency_series = build_sagnac_frequency_series(links, t, dt, geometry)
    sky = sky_grid(config.nside, hp)
    sky.to_csv(output_dir / "healpix_sky_grid.csv", index=False)

    layer_summaries = []
    all_candidates = []
    all_features = []
    all_clustered = []
    for wdm_m in wdm_ms:
        layer_config = replace(config, wdm_m=int(wdm_m))
        layer_output_dir = output_dir / f"wdm_m_{wdm_m}" if multi_resolution else output_dir
        layer_summary, candidates, clustered = run_wdm_layer(
            config=layer_config,
            a_t=selection_a_t,
            e_t=selection_e_t,
            selection_t=selection_t,
            selection_dt=selection_dt,
            sagnac_frequency_series=sagnac_frequency_series,
            t=t,
            dt=dt,
            geometry=geometry,
            metadata=metadata,
            sky=sky,
            output_dir=layer_output_dir,
            hdbscan=hdbscan,
            StandardScaler=StandardScaler,
            WDM=WDM,
        )
        layer_summaries.append(layer_summary)
        all_candidates.append(candidates)
        all_features.append(pd.read_csv(layer_output_dir / "kappa_features.csv"))
        all_clustered.append(clustered)

    if multi_resolution:
        pd.concat(all_candidates, ignore_index=True).to_csv(output_dir / "candidate_pixels_all_resolutions.csv", index=False)
        pd.concat(all_features, ignore_index=True).to_csv(output_dir / "kappa_features_all_resolutions.csv", index=False)
        pd.concat(all_clustered, ignore_index=True).to_csv(output_dir / "pixel_clusters_all_resolutions.csv", index=False)
        summary = {
            "config": asdict(config),
            "mode": "multi_resolution",
            "wdm_ms": [int(value) for value in wdm_ms],
            "n_resolutions": int(len(wdm_ms)),
            "n_sky_directions": int(len(sky)),
            "chunk_start_s": float(t[0]),
            "chunk_stop_s": float(t[-1]),
            "dt_s": float(dt),
            "selection_chunk_start_s": float(selection_t[0]),
            "selection_chunk_stop_s": float(selection_t[-1]),
            "selection_dt_s": float(selection_dt),
            "wdm_aet_source": "pytdi",
            "run_hdbscan": bool(config.run_hdbscan),
            "resolutions": layer_summaries,
            "source_ra": metadata.get("ra"),
            "source_dec": metadata.get("dec"),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        summary = layer_summaries[0]
    return summary


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run one-SMBHB coronagraphic pixel clustering prototype.")
    parser.add_argument("--work-dir", default="cluster/runs/simulations/smbhb_00000")
    parser.add_argument("--output-dir", default="choronagraphic-cluster/runs/smbhb_00000")
    parser.add_argument("--source", choices=("gw_only", "noisy"), default="gw_only")
    parser.add_argument("--nside", type=int, default=4)
    parser.add_argument("--chunk-duration-s", type=float, default=32768.0)
    parser.add_argument("--target-dt-s", type=float, default=1.0)
    parser.add_argument("--wdm-m", type=int, default=512)
    parser.add_argument(
        "--wdm-ms",
        default=None,
        help="Comma-separated WDM M values for multi-resolution mode, for example '128,256,512'.",
    )
    parser.add_argument("--wdm-k", type=int, default=32)
    parser.add_argument("--wdm-beta-order", type=int, default=6)
    parser.add_argument("--wdm-precision", type=int, default=6)
    parser.add_argument("--wdm-backend", default="numba")
    parser.add_argument("--top-pixels", type=int, default=80)
    parser.add_argument(
        "--top-pixel-fraction",
        type=float,
        default=None,
        help="Select this fraction of valid pixels per WDM resolution. Overrides --top-pixels when set.",
    )
    parser.add_argument("--freq-min-hz", type=float, default=1.0e-4)
    parser.add_argument("--freq-max-hz", type=float, default=0.1)
    parser.add_argument(
        "--selection-normalization",
        choices=("raw", "frequency-median", "offsource-frequency-median"),
        default="raw",
        help="Normalize A+E WDM power before selecting pixels. Kappa is still evaluated at the selected coordinates.",
    )
    parser.add_argument(
        "--offsource-edge-fraction",
        type=float,
        default=0.20,
        help="Fraction of WDM time bins on each edge used by offsource-frequency-median normalization.",
    )
    parser.add_argument(
        "--skip-hdbscan",
        action="store_true",
        help="Write candidate pixels and kappa_features.csv, but leave HDBSCAN clustering for scripts/coronagraphic_hdbscan.py.",
    )
    parser.add_argument(
        "--selection-start-guard-s",
        type=float,
        default=0.0,
        help="Exclude WDM pixels this many seconds from the start of the selected chunk.",
    )
    parser.add_argument(
        "--selection-end-guard-s",
        type=float,
        default=0.0,
        help="Exclude WDM pixels this many seconds from the end of the selected chunk.",
    )
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    wdm_ms = None
    if args.wdm_ms:
        wdm_ms = tuple(int(value.strip()) for value in args.wdm_ms.split(",") if value.strip())
    return RunConfig(
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        source=args.source,
        nside=args.nside,
        chunk_duration_s=args.chunk_duration_s,
        target_dt_s=args.target_dt_s,
        wdm_m=args.wdm_m,
        wdm_k=args.wdm_k,
        wdm_beta_order=args.wdm_beta_order,
        wdm_precision=args.wdm_precision,
        wdm_backend=args.wdm_backend,
        top_pixels=args.top_pixels,
        freq_min_hz=args.freq_min_hz,
        freq_max_hz=args.freq_max_hz,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        make_plots=not args.no_plots,
        wdm_ms=wdm_ms,
        top_pixel_fraction=args.top_pixel_fraction,
        selection_normalization=args.selection_normalization,
        offsource_edge_fraction=args.offsource_edge_fraction,
        wdm_aet_source="pytdi",
        run_hdbscan=not args.skip_hdbscan,
        selection_start_guard_s=args.selection_start_guard_s,
        selection_end_guard_s=args.selection_end_guard_s,
    )


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
