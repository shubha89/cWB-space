"""Seeded time-frequency event clustering for blind LISA WDM searches."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clean_wdm import WDMResolution, compute_wdm_transform, require_wdm_dependencies


@dataclass(frozen=True)
class SeededChirpEventConfig:
    low_hz: float = 1.0e-5
    high_hz: float = 1.0e-1
    seed_quantile: float = 0.99
    grow_quantile: float = 0.95
    max_freq_jump_bins: int = 4
    same_time_connect_bins: int = 1
    min_pixels: int = 8
    min_time_bins: int = 2
    freq_tolerance_bins: float = 1.0
    max_events: int = 8


def compute_self_ae_median_wdm(
    t_lisa: np.ndarray,
    a_channel: np.ndarray,
    e_channel: np.ndarray,
    *,
    resolution: WDMResolution,
) -> dict[str, np.ndarray]:
    """Compute self-normalized A/E WDM maps with a per-frequency median floor."""

    _, WDM = require_wdm_dependencies()
    wdm = WDM(M=resolution.M, K=32, beta_order=6, precision=6, backend="numba")
    t_lisa = np.asarray(t_lisa, dtype=float)
    native_dt = float(np.median(np.diff(t_lisa)))
    coeffs = {}
    times_ref = None
    freqs_ref = None
    tf_ref = None
    for label, values in (("A", a_channel), ("E", e_channel)):
        coeff, times, freqs, tf, _ = compute_wdm_transform(
            wdm,
            np.asarray(values, dtype=float),
            sample_rate=1.0 / native_dt,
            t0=float(t_lisa[0]),
            original_t1=float(t_lisa[-1]),
        )
        if times_ref is None:
            times_ref = times
            freqs_ref = freqs
            tf_ref = tf
        elif not np.array_equal(times_ref, times) or not np.array_equal(freqs_ref, freqs):
            raise ValueError(f"WDM grid mismatch for channel {label}.")
        coeffs[label] = coeff

    a_raw = np.abs(coeffs["A"]) ** 2
    e_raw = np.abs(coeffs["E"]) ** 2
    a_floor = clean_floor(np.nanmedian(a_raw, axis=1))
    e_floor = clean_floor(np.nanmedian(e_raw, axis=1))
    a_coeff = coeffs["A"] / np.sqrt(a_floor[:, None])
    e_coeff = coeffs["E"] / np.sqrt(e_floor[:, None])
    a_power = np.abs(a_coeff) ** 2
    e_power = np.abs(e_coeff) ** 2
    return {
        "times_rel_s": np.asarray(times_ref, dtype=float) - float(t_lisa[0]),
        "times_day": (np.asarray(times_ref, dtype=float) - float(t_lisa[0])) / 86_400.0,
        "freqs_hz": np.asarray(freqs_ref, dtype=float),
        "A_coeff": a_coeff,
        "E_coeff": e_coeff,
        "A_power": a_power,
        "E_power": e_power,
        "AE_power": a_power + e_power,
        "A_floor": a_floor,
        "E_floor": e_floor,
        "wdm_df_hz": np.asarray(float(tf_ref.df)),
        "wdm_dt_s": np.asarray(float(tf_ref.dt)),
        "native_dt_s": np.asarray(native_dt),
    }


def clean_floor(floor: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(floor) & (floor > 0.0), floor, 1.0e-300)


def find_seeded_chirp_events(
    maps: dict[str, np.ndarray],
    *,
    config: SeededChirpEventConfig,
    start_day: float,
    stop_day: float,
    label: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Find seeded, locally connected bright-pixel events in a time window."""

    times = np.asarray(maps["times_day"], dtype=float)
    freqs = np.asarray(maps["freqs_hz"], dtype=float)
    score = np.asarray(maps["AE_power"], dtype=float)
    band = (freqs >= config.low_hz) & (freqs <= config.high_hz)
    time = (times >= start_day) & (times <= stop_day)
    band_idx = np.flatnonzero(band)
    time_idx = np.flatnonzero(time)
    if len(band_idx) == 0 or len(time_idx) == 0:
        raise ValueError(f"Empty search region for {label}.")

    local_score = score[np.ix_(band_idx, time_idx)]
    finite = np.isfinite(local_score)
    seed_threshold = float(np.nanquantile(local_score[finite], config.seed_quantile))
    grow_threshold = float(np.nanquantile(local_score[finite], config.grow_quantile))
    grow_mask = finite & (local_score >= grow_threshold)
    seed_mask = finite & (local_score >= seed_threshold)
    seed_order = np.argsort(local_score[seed_mask].ravel())[::-1]
    seed_coords = list(zip(*np.nonzero(seed_mask)))
    seed_coords = [seed_coords[int(i)] for i in seed_order]

    visited = np.zeros_like(grow_mask, dtype=bool)
    events = []
    masks: dict[str, np.ndarray] = {}
    candidate_index = 0
    for seed in seed_coords:
        if visited[seed] or not grow_mask[seed]:
            continue
        component = grow_component(seed, grow_mask, visited, config)
        if int(np.sum(component)) < config.min_pixels:
            continue
        if count_time_bins(component) < config.min_time_bins:
            continue
        candidate_index += 1
        event_id = f"{label}_event{candidate_index:02d}"
        full_mask = np.zeros_like(score, dtype=bool)
        rows, cols = np.nonzero(component)
        full_mask[band_idx[rows], time_idx[cols]] = True
        event = summarize_event(
            event_id,
            label,
            full_mask,
            maps,
            seed_threshold=seed_threshold,
            grow_threshold=grow_threshold,
            search_start_day=start_day,
            search_stop_day=stop_day,
            freq_tolerance_bins=config.freq_tolerance_bins,
        )
        events.append(event)
        masks[event_id] = full_mask

    if not events:
        return pd.DataFrame(), {}
    frame = pd.DataFrame(events)
    frame = frame.sort_values("event_statistic", ascending=False).head(config.max_events).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    masks = {event_id: masks[event_id] for event_id in frame["event_id"]}
    return frame, masks


def grow_component(
    seed: tuple[int, int],
    grow_mask: np.ndarray,
    visited: np.ndarray,
    config: SeededChirpEventConfig,
) -> np.ndarray:
    n_freq, n_time = grow_mask.shape
    component = np.zeros_like(grow_mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque([seed])
    visited[seed] = True
    component[seed] = True
    same_time_offsets = [offset for offset in range(-config.same_time_connect_bins, config.same_time_connect_bins + 1) if offset != 0]
    time_offsets = (-1, 1)
    while queue:
        fi, ti = queue.popleft()
        for dfi in same_time_offsets:
            maybe_push(fi + dfi, ti, grow_mask, visited, component, queue, n_freq, n_time)
        for dti in time_offsets:
            nti = ti + dti
            for dfi in range(-config.max_freq_jump_bins, config.max_freq_jump_bins + 1):
                maybe_push(fi + dfi, nti, grow_mask, visited, component, queue, n_freq, n_time)
    return component


def maybe_push(
    fi: int,
    ti: int,
    grow_mask: np.ndarray,
    visited: np.ndarray,
    component: np.ndarray,
    queue: deque[tuple[int, int]],
    n_freq: int,
    n_time: int,
) -> None:
    if fi < 0 or fi >= n_freq or ti < 0 or ti >= n_time:
        return
    if visited[fi, ti] or not grow_mask[fi, ti]:
        return
    visited[fi, ti] = True
    component[fi, ti] = True
    queue.append((fi, ti))


def count_time_bins(mask: np.ndarray) -> int:
    return int(np.sum(np.any(mask, axis=0)))


def summarize_event(
    event_id: str,
    label: str,
    mask: np.ndarray,
    maps: dict[str, np.ndarray],
    *,
    seed_threshold: float,
    grow_threshold: float,
    search_start_day: float,
    search_stop_day: float,
    freq_tolerance_bins: float,
) -> dict[str, object]:
    freqs = np.asarray(maps["freqs_hz"], dtype=float)
    times = np.asarray(maps["times_day"], dtype=float)
    score = np.asarray(maps["AE_power"], dtype=float)
    a_power = np.asarray(maps["A_power"], dtype=float)
    e_power = np.asarray(maps["E_power"], dtype=float)
    a_coeff = np.asarray(maps["A_coeff"], dtype=np.complex128)
    e_coeff = np.asarray(maps["E_coeff"], dtype=np.complex128)
    fi, ti = np.nonzero(mask)
    values = score[fi, ti]
    peak_local = int(np.nanargmax(values))
    peak_fi = int(fi[peak_local])
    peak_ti = int(ti[peak_local])
    a_sum = float(np.nansum(a_power[fi, ti]))
    e_sum = float(np.nansum(e_power[fi, ti]))
    ae_sum = float(np.nansum(values))
    balance = 4.0 * a_sum * e_sum / (a_sum + e_sum) ** 2 if a_sum + e_sum > 0.0 else np.nan
    coherence = complex_coherence(a_coeff[fi, ti], e_coeff[fi, ti])
    chirp = chirp_metrics(mask, score, freqs, times, float(maps["wdm_df_hz"]), freq_tolerance_bins=freq_tolerance_bins)
    event_statistic = float(np.sqrt(max(ae_sum, 0.0)) * np.sqrt(max(balance, 0.0)) * (0.5 + 0.5 * chirp["monotonic_fraction"]))
    return {
        "event_id": event_id,
        "search_label": label,
        "rank": np.nan,
        "n_pixels": int(len(fi)),
        "n_time_bins": chirp["n_time_bins"],
        "seed_threshold": seed_threshold,
        "grow_threshold": grow_threshold,
        "search_start_day": search_start_day,
        "search_stop_day": search_stop_day,
        "start_day": float(np.min(times[ti])),
        "stop_day": float(np.max(times[ti])),
        "duration_days": float(np.max(times[ti]) - np.min(times[ti]) + float(maps["wdm_dt_s"]) / 86_400.0),
        "freq_min_hz": float(np.min(freqs[fi])),
        "freq_max_hz": float(np.max(freqs[fi])),
        "bandwidth_hz": float(np.max(freqs[fi]) - np.min(freqs[fi])),
        "peak_time_day": float(times[peak_ti]),
        "peak_freq_hz": float(freqs[peak_fi]),
        "peak_score": float(score[peak_fi, peak_ti]),
        "ae_total_power": ae_sum,
        "a_total_power": a_sum,
        "e_total_power": e_sum,
        "a_power_fraction": a_sum / ae_sum if ae_sum > 0.0 else np.nan,
        "ae_balance": balance,
        "ae_complex_coherence": coherence,
        "monotonic_fraction": chirp["monotonic_fraction"],
        "upward_fraction": chirp["upward_fraction"],
        "centroid_start_freq_hz": chirp["centroid_start_freq_hz"],
        "centroid_stop_freq_hz": chirp["centroid_stop_freq_hz"],
        "centroid_slope_hz_per_day": chirp["centroid_slope_hz_per_day"],
        "event_statistic": event_statistic,
        "touches_search_edge": bool(np.isclose(np.min(times[ti]), search_start_day) or np.isclose(np.max(times[ti]), search_stop_day)),
    }


def complex_coherence(a: np.ndarray, e: np.ndarray) -> float:
    denom = float(np.sqrt(np.nansum(np.abs(a) ** 2) * np.nansum(np.abs(e) ** 2)))
    if denom <= 0.0:
        return np.nan
    return float(np.abs(np.nansum(a * np.conj(e))) / denom)


def chirp_metrics(
    mask: np.ndarray,
    score: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    df_hz: float,
    *,
    freq_tolerance_bins: float,
) -> dict[str, float | int]:
    _, time_indices = np.nonzero(mask)
    unique_times = np.unique(time_indices)
    centroids = []
    centroid_times = []
    for ti in unique_times:
        fi = np.flatnonzero(mask[:, ti])
        weights = np.asarray(score[fi, ti], dtype=float)
        weight_sum = float(np.nansum(weights))
        if weight_sum <= 0.0:
            continue
        centroids.append(float(np.nansum(freqs[fi] * weights) / weight_sum))
        centroid_times.append(float(times[ti]))
    if len(centroids) < 2:
        return {
            "n_time_bins": int(len(centroids)),
            "monotonic_fraction": 0.0,
            "upward_fraction": 0.0,
            "centroid_start_freq_hz": centroids[0] if centroids else np.nan,
            "centroid_stop_freq_hz": centroids[0] if centroids else np.nan,
            "centroid_slope_hz_per_day": np.nan,
        }
    centroids_arr = np.asarray(centroids, dtype=float)
    diffs = np.diff(centroids_arr)
    tolerance = float(df_hz) * float(freq_tolerance_bins)
    duration = centroid_times[-1] - centroid_times[0]
    return {
        "n_time_bins": int(len(centroids_arr)),
        "monotonic_fraction": float(np.mean(diffs >= -tolerance)),
        "upward_fraction": float(np.mean(diffs > 0.0)),
        "centroid_start_freq_hz": float(centroids_arr[0]),
        "centroid_stop_freq_hz": float(centroids_arr[-1]),
        "centroid_slope_hz_per_day": float((centroids_arr[-1] - centroids_arr[0]) / duration) if duration > 0.0 else np.nan,
    }
