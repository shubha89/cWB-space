from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from wdm_wavelet.wdm import WDM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WORK_DIR = PROJECT_ROOT / "lisa_link_sim" / "1e5-data"
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LINK_LABELS = ["12", "23", "31", "13", "32", "21"]
XYZ_LABELS = ["X", "Y", "Z"]
AET_LABELS = ["A", "E", "T"]

LOW_FREQ_CUT = 1e-5
HIGH_FREQ_CUT = 1e-1
SIGNAL_THRESHOLD_FRACTION = 1e-3
OFFSOURCE_GUARD_SECONDS = 500.0
MIN_NOISE_FLOOR = 1e-300
EDGE_TAPER_SECONDS = 10_000.0
ANALYSIS_EDGE_GUARD_SECONDS = 10_000.0

WDM_CONFIGS = [
    {"label": "dt1_M128", "target_dt": 1.0, "M": 128},
    {"label": "dt1_M256", "target_dt": 1.0, "M": 256},
    {"label": "dt1_M512", "target_dt": 1.0, "M": 512},
    {"label": "dt1_M1024", "target_dt": 1.0, "M": 1024},
    {"label": "dt2_M2048", "target_dt": 2.0, "M": 2048},
    {"label": "dt5_M4096", "target_dt": 5.0, "M": 4096},
    {"label": "dt10_M8192", "target_dt": 10.0, "M": 8192},
]
WDM_K = 32
WDM_BETA_ORDER = 6
WDM_PRECISION = 6
WDM_BACKEND = "numba"


def robust_absmax(arr: np.ndarray) -> float:
    finite = np.isfinite(arr)
    return float(np.nanmax(np.abs(arr[finite]))) if np.any(finite) else 0.0


def plot_channels(t_rel: np.ndarray, channels: dict[str, np.ndarray], title: str, ylabel: str, outfile: Path) -> None:
    plt.figure(figsize=(13, 5))
    for name, arr in channels.items():
        plt.plot(t_rel, arr, lw=0.7, label=name)
    plt.xlabel("time in science segment [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(ncol=3, loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"saved {outfile}")


def signal_window_from_gw_aet(gw_tdi: pd.DataFrame) -> tuple[float, float, float]:
    t = gw_tdi["t_lisa"].to_numpy(dtype=float)
    power = sum(gw_tdi[f"{ch}2_gw_hz"].to_numpy(dtype=float) ** 2 for ch in AET_LABELS)
    peak_idx = int(np.argmax(power))
    threshold = SIGNAL_THRESHOLD_FRACTION * float(np.max(power))
    idx = np.flatnonzero(power >= threshold)
    return float(t[idx[0]]), float(t[idx[-1]]), float(t[peak_idx])


def make_basic_plots() -> tuple[float, float, float]:
    noisy_links = pd.read_csv(OUTPUT_DIR / "six_links_noisy.csv")
    gw_links = pd.read_csv(OUTPUT_DIR / "six_links_gw_only.csv")
    gw_links = gw_links[gw_links["is_science"].astype(bool)].reset_index(drop=True)
    noisy_tdi = pd.read_csv(OUTPUT_DIR / "tdi_michelson_XYZ_AET.csv")
    gw_tdi = pd.read_csv(OUTPUT_DIR / "tdi_gw_only_XYZ_AET.csv")

    t_lisa = noisy_tdi["t_lisa"].to_numpy(dtype=float)
    t_rel = t_lisa - t_lisa[0]
    signal_start, signal_stop, signal_peak = signal_window_from_gw_aet(gw_tdi)

    plot_channels(
        noisy_links["t_lisa"].to_numpy(dtype=float) - noisy_links["t_lisa"].iloc[0],
        {f"noisy_{lab}": noisy_links[f"noisy_{lab}"].to_numpy(dtype=float) for lab in LINK_LABELS},
        "1e5 data: noisy six links",
        "carrier fluctuation [Hz]",
        PLOT_DIR / "01_1e5_noisy_six_links.png",
    )
    plot_channels(
        gw_links["t_lisa"].to_numpy(dtype=float) - gw_links["t_lisa"].iloc[0],
        {f"gw_{lab}_hz": gw_links[f"gw_{lab}_hz"].to_numpy(dtype=float) for lab in LINK_LABELS},
        "1e5 data: GW-only six links",
        "GW link response [Hz]",
        PLOT_DIR / "02_1e5_gw_only_six_links.png",
    )
    plot_channels(
        t_rel,
        {f"{ch}2": noisy_tdi[f"{ch}2"].to_numpy(dtype=float) for ch in XYZ_LABELS},
        "1e5 data: noisy Michelson XYZ",
        "TDI [Hz]",
        PLOT_DIR / "03_1e5_noisy_XYZ.png",
    )
    plot_channels(
        t_rel,
        {f"{ch}2": noisy_tdi[f"{ch}2"].to_numpy(dtype=float) for ch in AET_LABELS},
        "1e5 data: noisy AET",
        "TDI [Hz]",
        PLOT_DIR / "04_1e5_noisy_AET.png",
    )
    plot_channels(
        t_rel,
        {f"{ch}2_gw_hz": gw_tdi[f"{ch}2_gw_hz"].to_numpy(dtype=float) for ch in XYZ_LABELS},
        "1e5 data: GW-only Michelson XYZ",
        "GW-only TDI [Hz]",
        PLOT_DIR / "05_1e5_gw_only_XYZ.png",
    )
    plot_channels(
        t_rel,
        {f"{ch}2_gw_hz": gw_tdi[f"{ch}2_gw_hz"].to_numpy(dtype=float) for ch in AET_LABELS},
        "1e5 data: GW-only AET",
        "GW-only TDI [Hz]",
        PLOT_DIR / "06_1e5_gw_only_AET.png",
    )

    # Overlay AET GW-only on noisy AET with a visual scale.
    plt.figure(figsize=(13, 5))
    noisy = {f"{ch}2": noisy_tdi[f"{ch}2"].to_numpy(dtype=float) for ch in AET_LABELS}
    signal = {f"{ch}2_gw_hz": gw_tdi[f"{ch}2_gw_hz"].to_numpy(dtype=float) for ch in AET_LABELS}
    noisy_max = max(robust_absmax(v) for v in noisy.values())
    signal_max = max(robust_absmax(v) for v in signal.values())
    scale = 1.0 if noisy_max == 0 or signal_max == 0 else 0.35 * noisy_max / signal_max
    for name, arr in noisy.items():
        plt.plot(t_rel, arr, lw=0.7, alpha=0.55, label=f"{name} noisy+GW")
    for name, arr in signal.items():
        plt.plot(t_rel, scale * arr, lw=1.0, label=f"{name} GW x {scale:.2e}")
    plt.axvspan(signal_start - t_lisa[0], signal_stop - t_lisa[0], color="gold", alpha=0.18, label="GW interval")
    plt.axvline(signal_peak - t_lisa[0], color="k", ls="--", alpha=0.7, label="GW peak")
    plt.xlabel("time in science segment [s]")
    plt.ylabel("TDI [Hz]")
    plt.title("1e5 data: noisy AET with GW-only overlay")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    out = PLOT_DIR / "07_1e5_noisy_AET_with_gw_overlay.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")

    return signal_start, signal_stop, signal_peak


def edge_taper(x: np.ndarray, dt: float, taper_seconds: float = EDGE_TAPER_SECONDS) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n_taper = int(round(taper_seconds / dt))
    if n_taper <= 1 or 2 * n_taper >= len(x):
        return x.copy()
    window = np.ones(len(x), dtype=float)
    phase = np.arange(n_taper, dtype=float) / n_taper
    ramp = 0.5 * (1.0 - np.cos(np.pi * phase))
    window[:n_taper] = ramp
    window[-n_taper:] = ramp[::-1]
    return x * window


def downsample_for_wdm(x: np.ndarray, dt: float, target_dt: float) -> tuple[np.ndarray, float]:
    factor = int(round(target_dt / dt))
    if not np.isclose(factor * dt, target_dt):
        raise ValueError("target_dt must be an integer multiple of native dt")
    if factor <= 1:
        return np.asarray(x, dtype=float), dt
    return resample_poly(np.asarray(x, dtype=float), up=1, down=factor), factor * dt


def pad_for_wdm(x: np.ndarray, wdm: WDM) -> tuple[np.ndarray, int]:
    min_len = max(len(x), 2 * int(wdm.m_H) + 1)
    if len(x) >= min_len:
        return x, 0
    return np.pad(x, (0, min_len - len(x)), mode="constant"), min_len - len(x)


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    edges = np.empty(len(values) + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.maximum(edges, np.nextafter(0.0, 1.0))


def robust_limits(images: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([im[np.isfinite(im)] for im in images if np.any(np.isfinite(im))])
    lo, hi = np.nanpercentile(vals, [2.0, 98.0])
    return float(lo), float(hi)


def wdm_transform_channel(x: np.ndarray, t0: float, t1: float, native_dt: float, config: dict) -> dict[str, np.ndarray | float]:
    x_tapered = edge_taper(x, native_dt)
    x_ds, dt_wdm = downsample_for_wdm(x_tapered, native_dt, config["target_dt"])
    wdm = WDM(M=config["M"], K=WDM_K, beta_order=WDM_BETA_ORDER, precision=WDM_PRECISION, backend=WDM_BACKEND)
    x_pad, pad_right = pad_for_wdm(x_ds, wdm)
    tf = wdm.t2w(x_pad, sample_rate=1.0 / dt_wdm, t0=t0)
    power = np.abs(tf.data) ** 2
    freqs = np.arange(tf.n_freq, dtype=float) * tf.df
    times = tf.t0 + (np.arange(tf.n_time, dtype=float) + 0.5) * tf.dt
    valid = times <= t1
    return {
        "power": power[:, valid],
        "freqs": freqs,
        "times": times[valid],
        "tf_dt": float(tf.dt),
        "df": float(tf.df),
        "pad_right": int(pad_right),
    }


def make_wdm_maps() -> None:
    noisy_tdi = pd.read_csv(OUTPUT_DIR / "tdi_michelson_XYZ_AET.csv")
    gw_tdi = pd.read_csv(OUTPUT_DIR / "tdi_gw_only_XYZ_AET.csv")
    t_lisa = noisy_tdi["t_lisa"].to_numpy(dtype=float)
    t0 = float(t_lisa[0])
    t1 = float(t_lisa[-1])
    t_rel0 = t0
    native_dt = float(np.median(np.diff(t_lisa)))
    signal_start, signal_stop, signal_peak = signal_window_from_gw_aet(gw_tdi)

    datasets = {
        "noisy_AET": {ch: noisy_tdi[f"{ch}2"].to_numpy(dtype=float) for ch in AET_LABELS},
        "gw_only_AET": {ch: gw_tdi[f"{ch}2_gw_hz"].to_numpy(dtype=float) for ch in AET_LABELS},
        "noisy_XYZ": {ch: noisy_tdi[f"{ch}2"].to_numpy(dtype=float) for ch in XYZ_LABELS},
        "gw_only_XYZ": {ch: gw_tdi[f"{ch}2_gw_hz"].to_numpy(dtype=float) for ch in XYZ_LABELS},
    }

    summary = []
    npz_payload: dict[str, np.ndarray] = {
        "signal_window_rel": np.asarray([signal_start - t0, signal_stop - t0]),
        "signal_peak_rel": np.asarray(signal_peak - t0),
    }

    for dataset_name, channels in datasets.items():
        products = {}
        for config in WDM_CONFIGS:
            label = config["label"]
            for ch, arr in channels.items():
                tr = wdm_transform_channel(arr, t0, t1, native_dt, config)
                off_mask = (
                    (tr["times"] < signal_start - OFFSOURCE_GUARD_SECONDS)
                    | (tr["times"] > signal_stop + OFFSOURCE_GUARD_SECONDS)
                )
                floor = np.nanmedian(tr["power"][:, off_mask], axis=1)
                floor = np.where(np.isfinite(floor) & (floor > 0), floor, MIN_NOISE_FLOOR)
                norm = tr["power"] / floor[:, None]
                products[(label, ch)] = {**tr, "normalized_power": norm, "noise_floor": floor}
                fmask = (tr["freqs"] >= LOW_FREQ_CUT) & (tr["freqs"] <= HIGH_FREQ_CUT)
                smask = (tr["times"] >= signal_start) & (tr["times"] <= signal_stop)
                band = norm[np.ix_(fmask, smask)]
                summary.append(
                    {
                        "dataset": dataset_name,
                        "channel": ch,
                        "resolution": label,
                        "target_dt_s": config["target_dt"],
                        "wdm_M": config["M"],
                        "wdm_df_hz": tr["df"],
                        "wdm_dt_tf_s": tr["tf_dt"],
                        "reaches_1e-5_hz": bool(tr["df"] <= LOW_FREQ_CUT),
                        "signal_norm_power_mean": float(np.nanmean(band)) if band.size else np.nan,
                        "signal_norm_power_max": float(np.nanmax(band)) if band.size else np.nan,
                    }
                )
                key = f"{dataset_name}_{ch}_{label}"
                npz_payload[f"{key}_normalized_power"] = norm.astype(np.float32)
                npz_payload[f"{key}_freqs_hz"] = tr["freqs"].astype(np.float64)
                npz_payload[f"{key}_times_rel"] = (tr["times"] - t_rel0).astype(np.float64)

        # Grid plot: rows resolutions, columns channels.
        labels = [cfg["label"] for cfg in WDM_CONFIGS]
        ch_names = list(channels)
        images = []
        prepared = {}
        for r, label in enumerate(labels):
            for c, ch in enumerate(ch_names):
                prod = products[(label, ch)]
                fmask = (prod["freqs"] > 0) & (prod["freqs"] <= HIGH_FREQ_CUT)
                image = np.log10(prod["normalized_power"][fmask] + 1e-30)
                prepared[(r, c)] = (prod, fmask, image)
                images.append(image)
        vmin, vmax = robust_limits(images)
        fig, axes = plt.subplots(
            len(labels),
            len(ch_names),
            figsize=(3.5 * len(ch_names), max(1.25 * len(labels), 8)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(len(labels), len(ch_names))
        last = None
        for r, label in enumerate(labels):
            for c, ch in enumerate(ch_names):
                ax = axes[r, c]
                prod, fmask, image = prepared[(r, c)]
                time_edges = centers_to_edges(prod["times"] - t0)
                freq_edges = centers_to_edges(prod["freqs"][fmask])
                last = ax.pcolormesh(time_edges, freq_edges, image, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
                ax.set_yscale("log")
                ax.set_ylim(LOW_FREQ_CUT, HIGH_FREQ_CUT)
                if r == 0:
                    ax.set_title(ch)
                if c == 0:
                    ax.set_ylabel(label, fontsize=8)
                if r == len(labels) - 1:
                    ax.set_xlabel("time [s]")
        fig.colorbar(last, ax=axes, label=r"$\log_{10}$ off-source-normalized WDM power", extend="both")
        fig.suptitle(f"1e5 data: multi-resolution WDM {dataset_name}")
        out = PLOT_DIR / f"08_1e5_multires_wdm_{dataset_name}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"saved {out}")

    summary_df = pd.DataFrame(summary)
    stats_file = OUTPUT_DIR / "1e5_wdm_multires_stats.csv"
    maps_file = OUTPUT_DIR / "1e5_wdm_multires_maps.npz"
    summary_df.to_csv(stats_file, index=False)
    np.savez_compressed(maps_file, **npz_payload)
    print(f"saved {stats_file}")
    print(f"saved {maps_file}")


def main() -> None:
    signal_start, signal_stop, signal_peak = make_basic_plots()
    make_wdm_maps()
    info = {
        "signal_start_lisa": signal_start,
        "signal_stop_lisa": signal_stop,
        "signal_peak_lisa": signal_peak,
        "low_freq_cut_hz": LOW_FREQ_CUT,
        "high_freq_cut_hz": HIGH_FREQ_CUT,
        "wdm_configs": WDM_CONFIGS,
        "edge_taper_seconds": EDGE_TAPER_SECONDS,
        "analysis_edge_guard_seconds": ANALYSIS_EDGE_GUARD_SECONDS,
    }
    path = OUTPUT_DIR / "1e5_plot_and_wdm_metadata.json"
    path.write_text(json.dumps(info, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
