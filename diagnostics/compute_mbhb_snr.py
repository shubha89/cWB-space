from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
SNR_JSON = OUTPUT_DIR / "mbhb_snr_report.json"
SNR_CSV = OUTPUT_DIR / "mbhb_snr_by_channel.csv"
SNR_PLOT = PLOT_DIR / "07_mbhb_snr_integrand_AE.png"
INDEPENDENT_NOISE_TDI = OUTPUT_DIR / "tdi_noise_only_XYZ_AET.csv"
SAME_SEED_NOISE_TDI = OUTPUT_DIR / "tdi_noise_only_seed12345_XYZ_AET.csv"
BABAK_SNR_PDF = PROJECT_ROOT / "docs" / "babak_hewitson_petiteau_lisa_sensitivity_snr_2021.pdf"
BABAK_SNR_TXT = PROJECT_ROOT / "docs" / "babak_hewitson_petiteau_lisa_sensitivity_snr_2021.txt"
SNR_CONVENTION_REFERENCE = {
    "document": "Babak, Hewitson, Petiteau, LISA Sensitivity and SNR Calculations, LISA-LCST-SGS-TN-001, Issue 1 Revision 0, 2021-08-04",
    "arxiv": "https://arxiv.org/abs/2108.01167",
    "local_pdf": str(BABAK_SNR_PDF),
    "local_text_extract": str(BABAK_SNR_TXT),
    "equation_1": "rho^2 = 4 Re int_0^fmax X_tilde(f) X_tilde*(f) / S_n(f) df for deterministic detector-output/TDI signal X.",
    "equation_65": "rho^2 = 4 Re int_0^fmax h_tilde(f) h_tilde*(f) / S_n(f) df for the LIGO-sense matched-filter form.",
    "lisa_combination": "For explicit TDI channels, total rho^2 is the sum of the per-channel rho_i^2 using each channel PSD.",
}

LOW_FREQ_HZ = 1e-5
HIGH_FREQ_HZ = 1e-1
CHANNELS = ("A", "E", "T")
PRIMARY_CHANNELS = ("A", "E")
TDI_SUFFIX = "2"
PSD_NPERSEG = 262_144


def luminosity_distance_to_redshift(distance_mpc: float) -> dict[str, float | str]:
    try:
        from astropy.cosmology import Planck18 as cosmo
        from astropy.cosmology import z_at_value
        import astropy.units as u

        z = float(z_at_value(cosmo.luminosity_distance, distance_mpc * u.Mpc))
        return {
            "redshift": z,
            "cosmology": "astropy.cosmology.Planck18",
            "distance_mpc": distance_mpc,
        }
    except Exception as exc:
        # Hubble-law fallback is not accurate at z~1; mark it as a rough diagnostic only.
        c_km_s = 299_792.458
        h0_km_s_mpc = 67.66
        return {
            "redshift": distance_mpc * h0_km_s_mpc / c_km_s,
            "cosmology": f"linear Hubble-law fallback ({type(exc).__name__}: {exc})",
            "distance_mpc": distance_mpc,
        }


def one_sided_psd_from_residual(residual: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the one-sided PSD S_n(f) used by the Babak et al. convention."""
    n = len(residual)
    nperseg = min(PSD_NPERSEG, n)
    noverlap = nperseg // 2
    freqs, psd = welch(
        residual,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        average="median",
    )
    psd = np.where(np.isfinite(psd) & (psd > 0.0), psd, np.nan)
    return freqs, psd


def one_sided_psd_from_offsource(series: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the one-sided PSD S_n(f) from samples outside the signal window."""
    n = len(series)
    nperseg = min(PSD_NPERSEG, n)
    noverlap = nperseg // 2
    freqs, psd = welch(
        series,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        average="median",
    )
    psd = np.where(np.isfinite(psd) & (psd > 0.0), psd, np.nan)
    return freqs, psd


def interpolate_psd(psd_freqs: np.ndarray, psd: np.ndarray, target_freqs: np.ndarray) -> np.ndarray:
    good = np.isfinite(psd_freqs) & np.isfinite(psd) & (psd > 0.0)
    if np.count_nonzero(good) < 2:
        raise ValueError("Need at least two finite PSD samples for interpolation.")
    interp = interp1d(
        psd_freqs[good],
        np.log(psd[good]),
        kind="linear",
        bounds_error=False,
        fill_value=(np.log(psd[good][0]), np.log(psd[good][-1])),
    )
    return np.exp(interp(target_freqs))


def inner_product(
    a: np.ndarray,
    b: np.ndarray,
    dt: float,
    psd_freqs: np.ndarray,
    psd: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Babak et al. one-sided detector-output/TDI inner product.

    The discrete calculation uses continuous-transform units:
    a_tilde = dt * rfft(a), df = 1 / (N * dt). The returned value
    approximates 4 Re int a_tilde*(f) b_tilde(f) / S_n(f) df for one
    explicit TDI channel. The optimal SNR is sqrt((h|h)); independent
    channels are combined by summing their rho_i^2.
    """
    n = min(len(a), len(b))
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    freqs = np.fft.rfftfreq(n, dt)
    a_ft = dt * np.fft.rfft(a)
    b_ft = dt * np.fft.rfft(b)
    psd_on_bins = interpolate_psd(psd_freqs, psd, freqs)
    band = (freqs >= LOW_FREQ_HZ) & (freqs <= HIGH_FREQ_HZ) & np.isfinite(psd_on_bins) & (psd_on_bins > 0.0)
    df = 1.0 / (n * dt)
    integrand = np.zeros_like(freqs, dtype=float)
    integrand[band] = 4.0 * np.real(np.conj(a_ft[band]) * b_ft[band]) / psd_on_bins[band]
    value = float(np.sum(integrand[band]) * df)
    return value, freqs, integrand


def signal_window(t: np.ndarray, channels: dict[str, np.ndarray], threshold_fraction: float = 1e-3) -> dict[str, float]:
    power = sum(channels[ch] ** 2 for ch in CHANNELS)
    peak_idx = int(np.nanargmax(power))
    peak = float(power[peak_idx])
    idx = np.flatnonzero(power >= threshold_fraction * peak)
    return {
        "threshold_fraction": threshold_fraction,
        "start_s": float(t[idx[0]]) if idx.size else np.nan,
        "stop_s": float(t[idx[-1]]) if idx.size else np.nan,
        "duration_s": float(t[idx[-1]] - t[idx[0]]) if idx.size else np.nan,
        "peak_s": float(t[peak_idx]),
    }


def template_comparison(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    comparison = {}
    for ch in CHANNELS:
        ref = np.asarray(reference[ch], dtype=float)
        cand = np.asarray(candidate[ch], dtype=float)
        denom = float(np.sqrt(np.sum(ref**2) * np.sum(cand**2)))
        comparison[ch] = {
            "time_domain_normalized_dot": float(np.sum(ref * cand) / denom) if denom > 0.0 else np.nan,
            "rms_ratio_candidate_over_reference": float(np.std(cand) / np.std(ref)) if np.std(ref) > 0.0 else np.nan,
            "max_abs_difference_hz": float(np.max(np.abs(cand - ref))),
        }
    return comparison


def main() -> None:
    metadata_file = PROJECT_ROOT / "lisa_link_sim" / "1e5-wave" / "outputs" / "my_waveform_1e5_metadata.json"
    metadata = json.loads(metadata_file.read_text())
    distance_info = luminosity_distance_to_redshift(float(metadata["distance_mpc"]))

    noisy = pd.read_csv(OUTPUT_DIR / "tdi_michelson_XYZ_AET.csv")
    gw = pd.read_csv(OUTPUT_DIR / "tdi_gw_only_XYZ_AET.csv")
    noise_only = pd.read_csv(INDEPENDENT_NOISE_TDI)
    same_seed_noise = pd.read_csv(SAME_SEED_NOISE_TDI) if SAME_SEED_NOISE_TDI.exists() else None
    n = min(len(noisy), len(gw), len(noise_only), len(same_seed_noise) if same_seed_noise is not None else len(noisy))
    noisy = noisy.iloc[:n].reset_index(drop=True)
    gw = gw.iloc[:n].reset_index(drop=True)
    noise_only = noise_only.iloc[:n].reset_index(drop=True)
    if same_seed_noise is not None:
        same_seed_noise = same_seed_noise.iloc[:n].reset_index(drop=True)

    t = noisy["t_waveform"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    exported_signal_channels = {ch: gw[f"{ch}{TDI_SUFFIX}_gw_hz"].to_numpy(dtype=float) for ch in CHANNELS}
    data_channels = {ch: noisy[f"{ch}{TDI_SUFFIX}"].to_numpy(dtype=float) for ch in CHANNELS}
    if same_seed_noise is not None:
        same_seed_noise_channels = {ch: same_seed_noise[f"{ch}{TDI_SUFFIX}"].to_numpy(dtype=float) for ch in CHANNELS}
        signal_channels = {ch: data_channels[ch] - same_seed_noise_channels[ch] for ch in CHANNELS}
        template_source = "instrument_isolated_same_seed_noise_subtraction"
        template_diagnostic = template_comparison(exported_signal_channels, signal_channels)
    else:
        same_seed_noise_channels = None
        signal_channels = exported_signal_channels
        template_source = "exported_gw_only_tdi"
        template_diagnostic = None
    residual_channels = {ch: data_channels[ch] - signal_channels[ch] for ch in CHANNELS}
    noise_only_channels = {ch: noise_only[f"{ch}{TDI_SUFFIX}"].to_numpy(dtype=float) for ch in CHANNELS}
    synthetic_channels = {ch: signal_channels[ch] + noise_only_channels[ch] for ch in CHANNELS}
    window = signal_window(t, signal_channels)
    offsource_guard_s = 500.0
    offsource_mask = (t < window["start_s"] - offsource_guard_s) | (t > window["stop_s"] + offsource_guard_s)
    if np.count_nonzero(offsource_mask) < 10_000:
        raise ValueError("Not enough off-source samples for a noise PSD estimate.")

    rows = []
    totals = {}
    integrand_for_plot: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for ch in CHANNELS:
        psd_sources = {
            "noise_only": one_sided_psd_from_residual(noise_only_channels[ch], fs),
            "residual": one_sided_psd_from_residual(residual_channels[ch], fs),
            "offsource": one_sided_psd_from_offsource(data_channels[ch][offsource_mask], fs),
        }
        for psd_method, (psd_freqs, psd) in psd_sources.items():
            hh, freqs, hh_integrand = inner_product(signal_channels[ch], signal_channels[ch], dt, psd_freqs, psd)
            actual_dh, _, _ = inner_product(data_channels[ch], signal_channels[ch], dt, psd_freqs, psd)
            actual_nh = actual_dh - hh
            synthetic_dh, _, _ = inner_product(synthetic_channels[ch], signal_channels[ch], dt, psd_freqs, psd)
            synthetic_nh, _, _ = inner_product(noise_only_channels[ch], signal_channels[ch], dt, psd_freqs, psd)
            optimal = float(np.sqrt(max(hh, 0.0)))
            actual_recovered = float(actual_dh / optimal) if optimal > 0.0 else np.nan
            actual_noise_projection = float(actual_nh / optimal) if optimal > 0.0 else np.nan
            synthetic_recovered = float(synthetic_dh / optimal) if optimal > 0.0 else np.nan
            synthetic_noise_projection = float(synthetic_nh / optimal) if optimal > 0.0 else np.nan
            rows.append(
                {
                    "channel": ch,
                    "psd_method": psd_method,
                    "optimal_snr": optimal,
                    "actual_recovered_matched_filter_snr": actual_recovered,
                    "actual_noise_projection_snr": actual_noise_projection,
                    "synthetic_recovered_matched_filter_snr": synthetic_recovered,
                    "synthetic_noise_projection_snr": synthetic_noise_projection,
                    "signal_inner_product": float(hh),
                    "residual_std_hz": float(np.std(residual_channels[ch])),
                    "same_seed_noise_std_hz": (
                        float(np.std(same_seed_noise_channels[ch])) if same_seed_noise_channels is not None else np.nan
                    ),
                    "noise_only_std_hz": float(np.std(noise_only_channels[ch])),
                    "offsource_std_hz": float(np.std(data_channels[ch][offsource_mask])),
                    "signal_absmax_hz": float(np.max(np.abs(signal_channels[ch]))),
                    "psd_median_hz2_per_hz": float(
                        np.nanmedian(psd[(psd_freqs >= LOW_FREQ_HZ) & (psd_freqs <= HIGH_FREQ_HZ)])
                    ),
                }
            )
            totals.setdefault(
                psd_method,
                {
                    "AE_hh": 0.0,
                    "AE_actual_dh": 0.0,
                    "AE_actual_nh": 0.0,
                    "AE_synthetic_dh": 0.0,
                    "AE_synthetic_nh": 0.0,
                    "AET_hh": 0.0,
                    "AET_actual_dh": 0.0,
                    "AET_actual_nh": 0.0,
                    "AET_synthetic_dh": 0.0,
                    "AET_synthetic_nh": 0.0,
                },
            )
            if ch in PRIMARY_CHANNELS:
                totals[psd_method]["AE_hh"] += hh
                totals[psd_method]["AE_actual_dh"] += actual_dh
                totals[psd_method]["AE_actual_nh"] += actual_nh
                totals[psd_method]["AE_synthetic_dh"] += synthetic_dh
                totals[psd_method]["AE_synthetic_nh"] += synthetic_nh
                if psd_method == "noise_only":
                    integrand_for_plot[ch] = (
                        freqs,
                        np.cumsum(hh_integrand) * (1.0 / (len(signal_channels[ch]) * dt)),
                    )
            totals[psd_method]["AET_hh"] += hh
            totals[psd_method]["AET_actual_dh"] += actual_dh
            totals[psd_method]["AET_actual_nh"] += actual_nh
            totals[psd_method]["AET_synthetic_dh"] += synthetic_dh
            totals[psd_method]["AET_synthetic_nh"] += synthetic_nh

    combined: dict[str, float] = {}
    for psd_method, vals in totals.items():
        optimal_ae = float(np.sqrt(max(vals["AE_hh"], 0.0)))
        optimal_aet = float(np.sqrt(max(vals["AET_hh"], 0.0)))
        combined[f"{psd_method}_AE_optimal_snr"] = optimal_ae
        combined[f"{psd_method}_AE_actual_recovered_matched_filter_snr"] = (
            float(vals["AE_actual_dh"] / optimal_ae) if optimal_ae > 0.0 else np.nan
        )
        combined[f"{psd_method}_AE_actual_noise_projection_snr"] = (
            float(vals["AE_actual_nh"] / optimal_ae) if optimal_ae > 0.0 else np.nan
        )
        combined[f"{psd_method}_AE_synthetic_recovered_matched_filter_snr"] = (
            float(vals["AE_synthetic_dh"] / optimal_ae) if optimal_ae > 0.0 else np.nan
        )
        combined[f"{psd_method}_AE_synthetic_noise_projection_snr"] = (
            float(vals["AE_synthetic_nh"] / optimal_ae) if optimal_ae > 0.0 else np.nan
        )
        combined[f"{psd_method}_AET_optimal_snr"] = optimal_aet
        combined[f"{psd_method}_AET_actual_recovered_matched_filter_snr"] = (
            float(vals["AET_actual_dh"] / optimal_aet) if optimal_aet > 0.0 else np.nan
        )
        combined[f"{psd_method}_AET_actual_noise_projection_snr"] = (
            float(vals["AET_actual_nh"] / optimal_aet) if optimal_aet > 0.0 else np.nan
        )
        combined[f"{psd_method}_AET_synthetic_recovered_matched_filter_snr"] = (
            float(vals["AET_synthetic_dh"] / optimal_aet) if optimal_aet > 0.0 else np.nan
        )
        combined[f"{psd_method}_AET_synthetic_noise_projection_snr"] = (
            float(vals["AET_synthetic_nh"] / optimal_aet) if optimal_aet > 0.0 else np.nan
        )

    snr_df = pd.DataFrame(rows)
    snr_df.to_csv(SNR_CSV, index=False)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for ch, (freqs, cumulative_snr2) in integrand_for_plot.items():
        band = (freqs >= LOW_FREQ_HZ) & (freqs <= HIGH_FREQ_HZ)
        cumulative = np.maximum(cumulative_snr2[band], 0.0)
        plt.plot(freqs[band], np.sqrt(cumulative), label=f"{ch} cumulative")
    plt.xscale("log")
    plt.xlabel("frequency [Hz]")
    plt.ylabel("cumulative optimal SNR")
    plt.title("MBHB z~1 diagnostic: cumulative A/E SNR")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SNR_PLOT, dpi=160)
    plt.close()

    report = {
        "source": {
            "waveform_metadata": str(metadata_file),
            "distance_mpc": float(metadata["distance_mpc"]),
            "estimated_redshift": distance_info,
            "total_mass": float(metadata["total_mass"]),
            "eta": float(metadata["eta"]),
            "f_min_hz": float(metadata["f_min_hz"]),
        },
        "snr_method": {
            "convention_reference": SNR_CONVENTION_REFERENCE,
            "definition": "Babak et al. one-sided detector-output/TDI convention: (a|b) = 4 Re int_0^fmax a_tilde*(f) b_tilde(f) / S_n(f) df.",
            "optimal_snr_definition": "rho_opt = sqrt((h|h)) for the TDI-domain signal h and the matching one-sided channel PSD S_n.",
            "recovered_snr_definition": "rho_rec = (d|h) / sqrt((h|h)) for data d = h + n.",
            "channel_combination": "AE and AET totals are sqrt(sum_i (h_i|h_i)); recovered projections sum (d_i|h_i) before dividing by the combined optimal SNR.",
            "channels_primary": list(PRIMARY_CHANNELS),
            "template_source": template_source,
            "independent_noise_tdi": str(INDEPENDENT_NOISE_TDI),
            "same_seed_noise_tdi": str(SAME_SEED_NOISE_TDI) if same_seed_noise is not None else None,
            "noise_estimate": "primary estimate uses an independent realistic noise-only instrument run; residual and off-source estimates are retained as diagnostics",
            "psd_convention": "one-sided PSD from scipy.signal.welch(..., scaling='density') in the same Hz-valued TDI channel units as the signal",
            "fourier_convention": "continuous Fourier approximation a_tilde[k] = dt * rfft(a)[k], df = 1 / (N * dt)",
            "primary_expected_snr_key": "noise_only_AE_optimal_snr",
            "primary_sanity_check_key": "noise_only_AE_synthetic_recovered_matched_filter_snr",
            "actual_noisy_recovered_note": "The primary actual-data recovery uses the GW component isolated from the same instrument HDF5 path by subtracting a same-seed noise-only TDI run. The exported GW-only TDI is retained only as a template-comparison diagnostic.",
            "low_freq_hz": LOW_FREQ_HZ,
            "high_freq_hz": HIGH_FREQ_HZ,
            "n_samples": int(n),
            "dt_s": dt,
            "fft_df_hz": float(1.0 / (n * dt)),
            "psd_welch_nperseg": int(min(PSD_NPERSEG, n)),
            "psd_welch_df_hz": float(fs / min(PSD_NPERSEG, n)),
            "offsource_guard_s": offsource_guard_s,
            "offsource_samples": int(np.count_nonzero(offsource_mask)),
        },
        "template_diagnostic_against_exported_gw_only_tdi": template_diagnostic,
        "signal_window": window,
        "combined": combined,
        "by_channel_csv": str(SNR_CSV),
        "cumulative_snr_plot": str(SNR_PLOT),
    }
    SNR_JSON.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report["source"], indent=2))
    print(json.dumps(report["combined"], indent=2))
    print(snr_df.to_string(index=False))
    print(f"report = {SNR_JSON}")
    print(f"plot   = {SNR_PLOT}")


if __name__ == "__main__":
    main()
