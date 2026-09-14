from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.blind_event_clustering import (
    SeededChirpEventConfig,
    compute_self_ae_median_wdm,
    find_seeded_chirp_events,
)
from scripts.clean_wdm import WDMResolution


INPUT_DIR = PROJECT_ROOT / "diagnostics" / "simulated_2week_mbhb_then_glitch_highrate" / "outputs"
PRODUCTS_NPZ = INPUT_DIR / "two_week_mbhb_then_glitch_tdi.npz"
MBHB_CATALOG = INPUT_DIR / "mbhb_catalog.csv"
RUN_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_snr_event_statistic_study"
OUTPUT_DIR = RUN_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

LOW_HZ = 1.0e-5
HIGH_HZ = 1.0e-1
DAY = 86_400.0
PSD_NPERSEG = 262_144
SOURCE_INDEX = 2
SOURCE_PREFIX = f"mbhb{SOURCE_INDEX}"
AMPLITUDE_SCALES = (0.35, 0.50, 0.65, 0.80, 1.00)
SEARCH_HALF_WINDOW_DAYS = 0.35
NOISE_WINDOW_CENTERS_DAYS = (0.85, 1.55, 6.35, 8.25, 10.0, 12.9)
RANDOM_NOISE_EVENTS = 12
RANDOM_SEED = 20260730
RESOLUTION = WDMResolution("dt4_M512", 4.0, 512)
EVENT_CONFIG = SeededChirpEventConfig(
    seed_quantile=0.99,
    grow_quantile=0.95,
    max_freq_jump_bins=4,
    same_time_connect_bins=1,
    min_pixels=8,
    min_time_bins=2,
    max_events=8,
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(PRODUCTS_NPZ)
    catalog = pd.read_csv(MBHB_CATALOG)
    source = catalog.iloc[SOURCE_INDEX - 1]
    t_lisa = np.asarray(data["t_lisa"], dtype=float)
    dt = float(np.nanmedian(np.diff(t_lisa)))

    unit_rho = optimal_ae_snr(data, dt, scale=1.0)
    noise_background, random_noise_events = compute_noise_background(data, t_lisa)
    rho_noise = float(noise_background["rho_event"].median())

    rows = []
    for scale in AMPLITUDE_SCALES:
        observed_a = np.asarray(data["noise_A2"], dtype=float) + scale * np.asarray(data[f"{SOURCE_PREFIX}_A2"], dtype=float)
        observed_e = np.asarray(data["noise_E2"], dtype=float) + scale * np.asarray(data[f"{SOURCE_PREFIX}_E2"], dtype=float)
        maps = compute_self_ae_median_wdm(t_lisa, observed_a, observed_e, resolution=RESOLUTION)
        start_day = max(0.0, float(source.response_peak_day) - SEARCH_HALF_WINDOW_DAYS)
        stop_day = min(float(np.nanmax(data["time_day"])), float(source.response_peak_day) + SEARCH_HALF_WINDOW_DAYS)
        events, _ = find_seeded_chirp_events(
            maps,
            config=EVENT_CONFIG,
            start_day=start_day,
            stop_day=stop_day,
            label=f"{source.case_id}_scale{scale:.2f}",
        )
        if events.empty:
            rows.append(empty_scale_row(source, scale, unit_rho, rho_noise))
            continue
        selected = events.sort_values("event_statistic", ascending=False).iloc[0]
        rho_event = float(selected.event_statistic)
        rows.append(
            {
                "source_case_id": str(source.case_id),
                "case_id": f"{source.case_id}_scale{scale:.2f}",
                "amplitude_scale": float(scale),
                "effective_distance_mpc": float(source.distance_mpc) / float(scale),
                "total_mass": float(source.total_mass),
                "q": float(source.q),
                "ra": float(source.ra),
                "dec": float(source.dec),
                "response_peak_day": float(source.response_peak_day),
                "optimal_snr_ae": float(scale * unit_rho),
                "rho_event": rho_event,
                "rho_noise": rho_noise,
                "rho_ratio": rho_event / rho_noise,
                "event_rank": int(selected["rank"]),
                "event_n_pixels": int(selected.n_pixels),
                "event_peak_time_day": float(selected.peak_time_day),
                "event_peak_freq_hz": float(selected.peak_freq_hz),
                "event_peak_time_error_days": float(selected.peak_time_day - float(source.response_peak_day)),
                "event_ae_balance": float(selected.ae_balance),
                "event_ae_complex_coherence": float(selected.ae_complex_coherence),
                "event_monotonic_fraction": float(selected.monotonic_fraction),
            }
        )

    results = pd.DataFrame(rows)
    results = results.sort_values("optimal_snr_ae").reset_index(drop=True)
    results.to_csv(OUTPUT_DIR / "mbhb_snr_event_statistic.csv", index=False)
    noise_background.to_csv(OUTPUT_DIR / "rho_noise_windows.csv", index=False)
    random_noise_events.to_csv(OUTPUT_DIR / "random_noise_events.csv", index=False)
    plot_results(results, rho_noise)
    write_report(results, noise_background, random_noise_events, source, unit_rho, rho_noise, dt)
    print(results.to_string(index=False))
    print(OUTPUT_DIR / "mbhb_snr_event_statistic.csv")
    print(PLOT_DIR / "mbhb_snr_event_statistic.png")


def one_sided_psd(noise: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(PSD_NPERSEG, len(noise))
    freqs, psd = welch(
        np.asarray(noise, dtype=float),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        average="median",
    )
    good = np.isfinite(freqs) & np.isfinite(psd) & (psd > 0.0)
    return freqs[good], psd[good]


def interpolate_psd(psd_freqs: np.ndarray, psd: np.ndarray, target_freqs: np.ndarray) -> np.ndarray:
    interpolator = interp1d(
        psd_freqs,
        np.log(psd),
        kind="linear",
        bounds_error=False,
        fill_value=(np.log(psd[0]), np.log(psd[-1])),
    )
    return np.exp(interpolator(target_freqs))


def channel_rho_squared(signal: np.ndarray, noise: np.ndarray, dt: float) -> float:
    fs = 1.0 / dt
    psd_freqs, psd = one_sided_psd(noise, fs)
    freqs = np.fft.rfftfreq(len(signal), dt)
    signal_ft = dt * np.fft.rfft(np.asarray(signal, dtype=float))
    psd_on_bins = interpolate_psd(psd_freqs, psd, freqs)
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ) & np.isfinite(psd_on_bins) & (psd_on_bins > 0.0)
    df = 1.0 / (len(signal) * dt)
    integrand = 4.0 * np.real(np.conj(signal_ft[band]) * signal_ft[band]) / psd_on_bins[band]
    return float(np.sum(integrand) * df)


def optimal_ae_snr(data: np.lib.npyio.NpzFile, dt: float, *, scale: float) -> float:
    rho2 = 0.0
    for channel in ("A", "E"):
        signal = scale * np.asarray(data[f"{SOURCE_PREFIX}_{channel}2"], dtype=float)
        noise = np.asarray(data[f"noise_{channel}2"], dtype=float)
        rho2 += channel_rho_squared(signal, noise, dt)
    return float(np.sqrt(max(rho2, 0.0)))


def compute_noise_background(data: np.lib.npyio.NpzFile, t_lisa: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    maps = compute_self_ae_median_wdm(
        t_lisa,
        np.asarray(data["noise_A2"], dtype=float),
        np.asarray(data["noise_E2"], dtype=float),
        resolution=RESOLUTION,
    )
    loudest_rows = []
    candidate_rows = []
    for center_day in NOISE_WINDOW_CENTERS_DAYS:
        events, _ = find_seeded_chirp_events(
            maps,
            config=EVENT_CONFIG,
            start_day=max(0.0, center_day - SEARCH_HALF_WINDOW_DAYS),
            stop_day=min(float(np.nanmax(data["time_day"])), center_day + SEARCH_HALF_WINDOW_DAYS),
            label=f"noise_window_day{center_day:.2f}",
        )
        if events.empty:
            continue
        selected = events.sort_values("event_statistic", ascending=False).iloc[0]
        loudest_rows.append(
            {
                "center_day": float(center_day),
                "rho_event": float(selected.event_statistic),
                "event_ae_complex_coherence": float(selected.ae_complex_coherence),
                "event_n_pixels": int(selected.n_pixels),
                "event_peak_time_day": float(selected.peak_time_day),
                "event_peak_freq_hz": float(selected.peak_freq_hz),
            }
        )
        for _, event in events.iterrows():
            candidate_rows.append(
                {
                    "center_day": float(center_day),
                    "candidate_rank": int(event["rank"]),
                    "rho_event": float(event.event_statistic),
                    "event_ae_complex_coherence": float(event.ae_complex_coherence),
                    "event_n_pixels": int(event.n_pixels),
                    "event_peak_time_day": float(event.peak_time_day),
                    "event_peak_freq_hz": float(event.peak_freq_hz),
                }
            )
    if not loudest_rows:
        raise RuntimeError("No noise-window events found for the rho-noise estimate.")
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        raise RuntimeError("No noise candidate events found for random-noise scatter.")
    random_events = candidates.sample(
        n=min(RANDOM_NOISE_EVENTS, len(candidates)),
        random_state=RANDOM_SEED,
    ).sort_values(["center_day", "candidate_rank"]).reset_index(drop=True)
    return pd.DataFrame(loudest_rows), random_events


def empty_scale_row(
    source: pd.Series,
    scale: float,
    unit_rho: float,
    rho_noise: float,
) -> dict[str, object]:
    return {
        "source_case_id": str(source.case_id),
        "case_id": f"{source.case_id}_scale{scale:.2f}",
        "amplitude_scale": float(scale),
        "effective_distance_mpc": float(source.distance_mpc) / float(scale),
        "total_mass": float(source.total_mass),
        "q": float(source.q),
        "ra": float(source.ra),
        "dec": float(source.dec),
        "response_peak_day": float(source.response_peak_day),
        "optimal_snr_ae": float(scale * unit_rho),
        "rho_event": np.nan,
        "rho_noise": rho_noise,
        "rho_ratio": np.nan,
        "event_rank": np.nan,
        "event_n_pixels": 0,
        "event_peak_time_day": np.nan,
        "event_peak_freq_hz": np.nan,
        "event_peak_time_error_days": np.nan,
        "event_ae_balance": np.nan,
        "event_ae_complex_coherence": np.nan,
        "event_monotonic_fraction": np.nan,
    }


def plot_results(results: pd.DataFrame, rho_noise: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    event_line = ax.plot(
        results["optimal_snr_ae"],
        results["rho_event"],
        marker="o",
        lw=1.5,
        color="#174a7c",
        label=r"$\rho_{\rm event}$",
    )
    ax2 = ax.twinx()
    for _, row in results.iterrows():
        ax.text(
            float(row["optimal_snr_ae"]) * 1.01,
            float(row["rho_event"]) + 1.0,
            f"{row['amplitude_scale']:.2f}",
            fontsize=7.4,
            color="#174a7c",
        )
    ax.set_xlabel(r"optimal A/E SNR, $\rho_{\rm opt}$")
    ax.set_ylabel(r"$\rho_{\rm event}$")
    ax2.set_ylabel(r"$R_\rho$")
    y_min = max(0.0, float(results["rho_event"].min()) * 0.88)
    y_max = float(results["rho_event"].max()) * 1.10
    ax.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min / rho_noise, y_max / rho_noise)
    ax.set_title("Distance-scaled MBHB recovery")
    ax.grid(True, which="both", alpha=0.16)
    ax.legend(event_line, [event_line[0].get_label()], frameon=False, fontsize=8, loc="upper left")
    fig.savefig(PLOT_DIR / "mbhb_snr_event_statistic.pdf", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "mbhb_snr_event_statistic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    results: pd.DataFrame,
    noise_background: pd.DataFrame,
    random_noise_events: pd.DataFrame,
    source: pd.Series,
    unit_rho: float,
    rho_noise: float,
    dt: float,
) -> None:
    report = {
        "input_products": str(PRODUCTS_NPZ),
        "source_case_id": str(source.case_id),
        "source_index": SOURCE_INDEX,
        "method": "Amplitude-scale one saved MBHB link-level response before adding it to the saved realistic-noise A/E TDI streams.",
        "optimal_snr_definition": "Sum of A and E one-sided TDI inner products over 1e-5 <= f <= 1e-1 Hz using Welch PSDs from the saved noise-only streams.",
        "rho_noise_definition": "Median loudest rho_event from same-duration off-source noise windows processed with the same WDM and event-clustering settings.",
        "rho_ratio_definition": "rho_event / rho_noise.",
        "amplitude_scales": list(AMPLITUDE_SCALES),
        "search_half_window_days": SEARCH_HALF_WINDOW_DAYS,
        "noise_window_centers_days": list(NOISE_WINDOW_CENTERS_DAYS),
        "random_noise_events": RANDOM_NOISE_EVENTS,
        "random_seed": RANDOM_SEED,
        "rho_noise": rho_noise,
        "n_noise_windows": int(len(noise_background)),
        "n_random_noise_events": int(len(random_noise_events)),
        "unit_scale_optimal_snr_ae": unit_rho,
        "dt_s": dt,
        "monotonic_rho_event": bool(np.all(np.diff(results["rho_event"].to_numpy(dtype=float)) > 0.0)),
        "monotonic_rho_ratio": bool(np.all(np.diff(results["rho_ratio"].to_numpy(dtype=float)) > 0.0)),
        "outputs": {
            "results_csv": str(OUTPUT_DIR / "mbhb_snr_event_statistic.csv"),
            "noise_windows_csv": str(OUTPUT_DIR / "rho_noise_windows.csv"),
            "random_noise_events_csv": str(OUTPUT_DIR / "random_noise_events.csv"),
            "plot_pdf": str(PLOT_DIR / "mbhb_snr_event_statistic.pdf"),
            "plot_png": str(PLOT_DIR / "mbhb_snr_event_statistic.png"),
        },
    }
    (OUTPUT_DIR / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
