"""Make representative slide plots for the SMBHB WDM clustering pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHORO_DIR = PROJECT_ROOT / "choronagraphic-cluster"
if str(CHORO_DIR) not in sys.path:
    sys.path.insert(0, str(CHORO_DIR))

import plot_multires_wdm_overlay as overlay  # noqa: E402
import run_one_smbhb as proto  # noqa: E402


LINK_LABELS = ("12", "23", "31", "13", "32", "21")
COLORS = {
    "12": "#1f77b4",
    "23": "#ff7f0e",
    "31": "#2ca02c",
    "13": "#d62728",
    "32": "#9467bd",
    "21": "#8c564b",
    "A": "#005f73",
    "E": "#ca6702",
    "T": "#6a4c93",
}


def robust_ylim(values: np.ndarray, percentile: float = 99.5, pad: float = 1.15) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lim = float(np.nanpercentile(np.abs(finite), percentile))
    if lim <= 0.0:
        lim = float(np.nanmax(np.abs(finite))) or 1.0
    return -pad * lim, pad * lim


def chunk_bounds(summary_path: Path) -> tuple[float, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary["chunk_start_s"]), float(summary["chunk_stop_s"])


def gw_signal_window(work_dir: Path, start_s: float, stop_s: float, threshold_fraction: float = 1.0e-3) -> tuple[float, float, float]:
    cols = ["t_waveform", "is_science", *[f"gw_{link}_hz" for link in LINK_LABELS]]
    gw = pd.read_csv(work_dir / "outputs" / "six_links_gw_only.csv", usecols=cols)
    gw = gw[gw["is_science"].astype(bool)]
    gw = gw[(gw["t_waveform"] >= start_s) & (gw["t_waveform"] <= stop_s)].reset_index(drop=True)
    if gw.empty:
        return start_s, stop_s, 0.5 * (start_s + stop_s)
    power = sum(gw[f"gw_{link}_hz"].to_numpy(dtype=float) ** 2 for link in LINK_LABELS)
    peak_idx = int(np.nanargmax(power))
    threshold = threshold_fraction * float(np.nanmax(power))
    active = np.flatnonzero(power >= threshold)
    if active.size == 0:
        return start_s, stop_s, float(gw["t_waveform"].iloc[peak_idx])
    return (
        float(gw["t_waveform"].iloc[int(active[0])]),
        float(gw["t_waveform"].iloc[int(active[-1])]),
        float(gw["t_waveform"].iloc[peak_idx]),
    )


def plot_six_links(work_dir: Path, out_dir: Path, start_s: float, stop_s: float) -> Path:
    df = pd.read_csv(work_dir / "outputs" / "six_links_noisy.csv")
    df = df[(df["t_waveform"] >= start_s) & (df["t_waveform"] <= stop_s)].reset_index(drop=True)
    stride = max(1, len(df) // 6000)
    df = df.iloc[::stride].reset_index(drop=True)
    time = df["t_waveform"].to_numpy(dtype=float) - start_s

    fig, axes = plt.subplots(3, 2, figsize=(13, 7), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, link in zip(axes, LINK_LABELS):
        y = df[f"noisy_{link}"].to_numpy(dtype=float)
        ax.plot(time, y, color=COLORS[link], lw=0.8)
        ax.set_title(f"Link {link}", fontsize=10)
        ax.set_ylabel("Hz")
        ax.set_ylim(*robust_ylim(y))
        ax.grid(alpha=0.25)
    for ax in axes[-2:]:
        ax.set_xlabel("time since chunk start [s]")
    fig.suptitle("1. LISA six-link noisy carrier fluctuations", fontsize=15)
    out = out_dir / "01_lisa_six_links_noisy.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_noisy_links_with_gw_highlight(work_dir: Path, out_dir: Path, start_s: float, stop_s: float) -> Path:
    noisy = pd.read_csv(work_dir / "outputs" / "six_links_noisy.csv")
    noisy = noisy[(noisy["t_waveform"] >= start_s) & (noisy["t_waveform"] <= stop_s)].reset_index(drop=True)
    gw_cols = ["t_waveform", "is_science", *[f"gw_{link}_hz" for link in LINK_LABELS]]
    gw = pd.read_csv(work_dir / "outputs" / "six_links_gw_only.csv", usecols=gw_cols)
    gw = gw[gw["is_science"].astype(bool)]
    gw = gw[(gw["t_waveform"] >= start_s) & (gw["t_waveform"] <= stop_s)].reset_index(drop=True)

    n = min(len(noisy), len(gw))
    noisy = noisy.iloc[:n].reset_index(drop=True)
    gw = gw.iloc[:n].reset_index(drop=True)
    signal_start, signal_stop, signal_peak = gw_signal_window(work_dir, start_s, stop_s)

    stride = max(1, n // 6000)
    noisy = noisy.iloc[::stride].reset_index(drop=True)
    gw = gw.iloc[::stride].reset_index(drop=True)
    time = noisy["t_waveform"].to_numpy(dtype=float) - start_s

    fig, axes = plt.subplots(3, 2, figsize=(13, 7), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, link in zip(axes, LINK_LABELS):
        y = noisy[f"noisy_{link}"].to_numpy(dtype=float)
        g = gw[f"gw_{link}_hz"].to_numpy(dtype=float)
        y_lim = robust_ylim(y)
        y_scale = max(abs(y_lim[0]), abs(y_lim[1]))
        g_scale = float(np.nanpercentile(np.abs(g[np.isfinite(g)]), 99.5)) if np.any(np.isfinite(g)) else 0.0
        scale = 0.35 * y_scale / g_scale if g_scale > 0.0 else 1.0

        ax.axvspan(signal_start - start_s, signal_stop - start_s, color="#ffd166", alpha=0.22, label="GW-active interval")
        ax.axvline(signal_peak - start_s, color="black", ls="--", lw=0.9, alpha=0.8)
        ax.plot(time, y, color=COLORS[link], lw=0.75, label="noisy link")
        ax.plot(time, scale * g, color="black", lw=1.0, alpha=0.9, label=f"GW-only x {scale:.1e}")
        ax.set_title(f"Link {link}", fontsize=10)
        ax.set_ylabel("Hz")
        ax.set_ylim(y_lim)
        ax.grid(alpha=0.25)
    for ax in axes[-2:]:
        ax.set_xlabel("time since chunk start [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("1b. Noisy LISA links with injected GW response highlighted", fontsize=15, y=0.98)
    out = out_dir / "01b_noisy_links_with_gw_signal_highlighted.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_aet_time(work_dir: Path, out_dir: Path, start_s: float, stop_s: float) -> Path:
    df = pd.read_csv(work_dir / "outputs" / "tdi_michelson_XYZ_AET.csv")
    df = df[(df["t_waveform"] >= start_s) & (df["t_waveform"] <= stop_s)].reset_index(drop=True)
    stride = max(1, len(df) // 8000)
    df = df.iloc[::stride].reset_index(drop=True)
    time = df["t_waveform"].to_numpy(dtype=float) - start_s

    channels = {"A": "A2", "E": "E2", "T": "T2"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for ax, (label, col) in zip(axes, channels.items()):
        y = df[col].to_numpy(dtype=float)
        ax.plot(time, y, color=COLORS[label], lw=0.8)
        ax.set_ylabel(f"{label} [Hz]")
        ax.set_ylim(*robust_ylim(y))
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time since chunk start [s]")
    fig.suptitle("2. Second-generation pyTDI A/E/T channels in time domain", fontsize=15)
    out = out_dir / "02_second_generation_AET_time_domain.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def load_selection_timeseries(run_dir: Path, background_m: int):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config_data = dict(summary["config"])
    config_data["wdm_m"] = int(background_m)
    config = proto.RunConfig(**config_data)
    _, _, resample_poly, _, WDM = proto.require_dependencies()

    work_dir = PROJECT_ROOT / config.work_dir
    metadata = proto.load_source_metadata(work_dir)
    t_raw, _ = proto.load_links(work_dir, config.source)
    center_s = float(metadata.get("raw_half_duration_s", 150_000.0))
    if "peak_time_s" in metadata:
        center_s = float(t_raw[0]) + 0.5 * (float(t_raw[-1]) - float(t_raw[0]))

    if config.wdm_aet_source == "pytdi":
        t_aet_raw, a_raw, e_raw = proto.load_pytdi_aet(work_dir, config.source)
        t_sel, a_t, e_t, dt_sel = proto.crop_and_downsample_aet(
            t_aet_raw,
            a_raw,
            e_raw,
            center_s=center_s,
            duration_s=config.chunk_duration_s,
            target_dt_s=config.target_dt_s,
            resample_poly=resample_poly,
        )
    else:
        raise ValueError("Slide plots expect pyTDI A/E selection products.")
    return config, WDM, t_sel, a_t, e_t, dt_sel


def wdm_raw_and_weighted(run_dir: Path, wdm_m: int):
    config, WDM, t_sel, a_t, e_t, dt_sel = load_selection_timeseries(run_dir, wdm_m)
    power_a, freqs, times = proto.wdm_power(a_t, dt=dt_sel, t0=float(t_sel[0]), config=config, WDM=WDM)
    power_e, _, _ = proto.wdm_power(e_t, dt=dt_sel, t0=float(t_sel[0]), config=config, WDM=WDM)
    raw = power_a + power_e
    weighted, background = proto.normalize_selection_power(
        raw,
        mode=config.selection_normalization,
        offsource_edge_fraction=config.offsource_edge_fraction,
    )
    return raw, weighted, background, freqs, times, float(t_sel[0])


def load_tdi_group_timeseries(
    run_dir: Path,
    wdm_m: int,
    *,
    source: str,
    group: str,
):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config_data = dict(summary["config"])
    config_data["wdm_m"] = int(wdm_m)
    config = proto.RunConfig(**config_data)
    _, _, resample_poly, _, WDM = proto.require_dependencies()

    work_dir = PROJECT_ROOT / config.work_dir
    metadata = proto.load_source_metadata(work_dir)
    t_raw, _ = proto.load_links(work_dir, config.source)
    center_s = float(metadata.get("raw_half_duration_s", 150_000.0))
    if "peak_time_s" in metadata:
        center_s = float(t_raw[0]) + 0.5 * (float(t_raw[-1]) - float(t_raw[0]))

    if source == "noisy":
        path = work_dir / "outputs" / "tdi_michelson_XYZ_AET.csv"
        suffix = ""
    elif source == "gw_only":
        path = work_dir / "outputs" / "tdi_gw_only_XYZ_AET.csv"
        suffix = "_gw_hz"
    else:
        raise ValueError("source must be 'noisy' or 'gw_only'.")

    channels = {
        "AET": ("A2", "E2", "T2"),
        "AE": ("A2", "E2"),
        "XYZ": ("X2", "Y2", "Z2"),
    }[group]
    cols = ["t_waveform", *[f"{channel}{suffix}" for channel in channels]]
    df = pd.read_csv(path, usecols=cols)
    arrays = [df[f"{channel}{suffix}"].to_numpy(dtype=float) for channel in channels]
    t = df["t_waveform"].to_numpy(dtype=float)

    ds = []
    for arr in arrays:
        t_sel, arr_ds, _, dt_sel = proto.crop_and_downsample_aet(
            t,
            arr,
            np.zeros_like(arr),
            center_s=center_s,
            duration_s=config.chunk_duration_s,
            target_dt_s=config.target_dt_s,
            resample_poly=resample_poly,
        )
        ds.append(arr_ds)
    return config, WDM, t_sel, ds, dt_sel


def wdm_group_power(run_dir: Path, wdm_m: int, *, source: str, group: str):
    config, WDM, t_sel, arrays, dt_sel = load_tdi_group_timeseries(run_dir, wdm_m, source=source, group=group)
    total = None
    freqs = None
    times = None
    for arr in arrays:
        power, freqs, times = proto.wdm_power(arr, dt=dt_sel, t0=float(t_sel[0]), config=config, WDM=WDM)
        total = power if total is None else total + power
    if total is None or freqs is None or times is None:
        raise RuntimeError("No TDI channels loaded for WDM group power.")
    return total, freqs, times, float(t_sel[0])


def image_extent(times: np.ndarray, freqs: np.ndarray, chunk_start_s: float) -> list[float]:
    return overlay.image_extent(times, freqs, chunk_start_s)


def plot_multires_wdm(run_dir: Path, out_dir: Path, min_freq_mhz: float, max_freq_mhz: float, trim_start_s: float) -> Path:
    ms = [128, 256, 512]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True, constrained_layout=True)
    last = None
    for ax, m in zip(axes, ms):
        raw, _, _, freqs, times, chunk_start = wdm_raw_and_weighted(run_dir, m)
        image = np.log10(np.maximum(raw, 1.0e-300))
        extent = image_extent(times, freqs, chunk_start)
        last = ax.imshow(image, origin="lower", aspect="auto", cmap="magma", extent=extent)
        ax.set_title(f"WDM M={m}")
        ax.set_xlabel("time since chunk start [s]")
        ax.set_xlim(extent[0] + trim_start_s, extent[1])
        ax.set_ylim(min_freq_mhz, max_freq_mhz)
        ax.grid(color="white", alpha=0.15, linewidth=0.6)
    axes[0].set_ylabel("frequency [mHz]")
    fig.colorbar(last, ax=axes, label="log10 raw A+E WDM power")
    fig.suptitle("3. Multi-resolution WDM time-frequency maps", fontsize=15)
    out = out_dir / "03_multiresolution_WDM_raw_power.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_multires_wdm_broad_linear(
    run_dir: Path,
    out_dir: Path,
    min_freq_mhz: float,
    max_freq_mhz: float,
    trim_start_s: float,
    *,
    source: str = "noisy",
    group: str = "AE",
    filename: str = "03b_broad_multiresolution_WDM_linear_power.png",
    title: str = "3b. Broad multi-resolution WDM maps with linear power scale",
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.5,
) -> Path:
    ms = [128, 256, 512]
    images = []
    products = []
    for m in ms:
        raw, freqs, times, chunk_start = wdm_group_power(run_dir, m, source=source, group=group)
        extent = image_extent(times, freqs, chunk_start)
        fmask = (freqs * 1.0e3 >= min_freq_mhz) & (freqs * 1.0e3 <= max_freq_mhz)
        sub = raw[fmask]
        finite = sub[np.isfinite(sub)]
        images.append(finite)
        products.append((m, raw, freqs, times, chunk_start, extent))

    finite_all = np.concatenate([vals for vals in images if vals.size]) if any(vals.size for vals in images) else np.array([0.0, 1.0])
    vmin = float(np.nanpercentile(finite_all, vmin_percentile))
    vmax = float(np.nanpercentile(finite_all, vmax_percentile))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite_all)) if finite_all.size else 1.0
        vmin = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True, constrained_layout=True)
    last = None
    for ax, (m, raw, freqs, times, chunk_start, extent) in zip(axes, products):
        last = ax.imshow(raw, origin="lower", aspect="auto", cmap="magma", extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(f"WDM M={m}")
        ax.set_xlabel("time since chunk start [s]")
        ax.set_xlim(extent[0] + trim_start_s, extent[1])
        ax.set_ylim(min_freq_mhz, max_freq_mhz)
        ax.grid(color="white", alpha=0.15, linewidth=0.6)
    axes[0].set_ylabel("frequency [mHz]")
    fig.colorbar(
        last,
        ax=axes,
        label=f"{source} {group} WDM power (linear scale, p{vmin_percentile:g}-p{vmax_percentile:g} clipped)",
    )
    fig.suptitle(title, fontsize=15)
    out = out_dir / filename
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_noise_weighted_wdm(
    run_dir: Path,
    out_dir: Path,
    min_freq_mhz: float,
    max_freq_mhz: float,
    trim_start_s: float,
) -> Path:
    _, weighted, _, freqs, times, chunk_start = wdm_raw_and_weighted(run_dir, 512)
    image = np.log10(np.maximum(weighted, 1.0e-300))
    extent = image_extent(times, freqs, chunk_start)
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    im = ax.imshow(image, origin="lower", aspect="auto", cmap="magma", extent=extent)
    fig.colorbar(im, ax=ax, label="log10 off-source-normalized A+E WDM power")
    ax.set_xlabel("time since chunk start [s]")
    ax.set_ylabel("frequency [mHz]")
    ax.set_xlim(extent[0] + trim_start_s, extent[1])
    ax.set_ylim(min_freq_mhz, max_freq_mhz)
    ax.grid(color="white", alpha=0.15, linewidth=0.6)
    ax.set_title("4. Off-source noise-weighted WDM pixel map (M=512)")
    out = out_dir / "04_offsource_noise_weighted_WDM.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_top_pixels(
    run_dir: Path,
    out_dir: Path,
    min_freq_mhz: float,
    max_freq_mhz: float,
    trim_start_s: float,
) -> Path:
    _, weighted, _, freqs, times, chunk_start = wdm_raw_and_weighted(run_dir, 512)
    image = np.log10(np.maximum(weighted, 1.0e-300))
    extent = image_extent(times, freqs, chunk_start)
    candidates = pd.read_csv(run_dir / "candidate_pixels_all_resolutions.csv")
    candidates = candidates.copy()
    candidates["time_offset_s"] = candidates["time_s"].astype(float) - chunk_start
    candidates["freq_mhz"] = candidates["freq_hz"].astype(float) * 1.0e3

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    im = ax.imshow(image, origin="lower", aspect="auto", cmap="magma", extent=extent)
    fig.colorbar(im, ax=ax, label="log10 off-source-normalized A+E WDM power")
    markers = {128: "o", 256: "s", 512: "^"}
    colors = {128: "#00e5ff", 256: "#ffea00", 512: "#ff4d6d"}
    for m in [128, 256, 512]:
        rows = candidates[candidates["wdm_m"].astype(int) == m]
        ax.scatter(
            rows["time_offset_s"],
            rows["freq_mhz"],
            s=30,
            marker=markers[m],
            facecolor=colors[m],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
            label=f"M={m} top 10%",
        )
    ax.set_xlabel("time since chunk start [s]")
    ax.set_ylabel("frequency [mHz]")
    ax.set_xlim(extent[0] + trim_start_s, extent[1])
    ax.set_ylim(min_freq_mhz, max_freq_mhz)
    ax.grid(color="white", alpha=0.15, linewidth=0.6)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("5. Top 10% WDM-noise-weighted pixels selected for clustering")
    out = out_dir / "05_top10_selected_WDM_pixels.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create representative plots for pipeline slide bullets.")
    parser.add_argument("--source-id", default="smbhb_m1e6")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-dir", default="test-run/slide_plots")
    parser.add_argument("--min-freq-mhz", type=float, default=0.01)
    parser.add_argument("--max-freq-mhz", type=float, default=10.0)
    parser.add_argument("--broad-min-freq-mhz", type=float, default=0.0)
    parser.add_argument("--broad-max-freq-mhz", type=float, default=100.0)
    parser.add_argument("--trim-start-s", type=float, default=4096.0)
    args = parser.parse_args()

    source_id = args.source_id
    work_dir = PROJECT_ROOT / (args.work_dir or f"test-run/simulations/{source_id}")
    run_dir = PROJECT_ROOT / (
        args.run_dir or f"test-run/coronagraphic/{source_id}_noisy_pytdi_aet_offsource_wdmnorm"
    )
    out_dir = PROJECT_ROOT / args.out_dir / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    start_s, stop_s = chunk_bounds(run_dir / "summary.json")
    outputs = [
        plot_six_links(work_dir, out_dir, start_s, stop_s),
        plot_noisy_links_with_gw_highlight(work_dir, out_dir, start_s, stop_s),
        plot_aet_time(work_dir, out_dir, start_s, stop_s),
        plot_multires_wdm(run_dir, out_dir, args.min_freq_mhz, args.max_freq_mhz, args.trim_start_s),
        plot_multires_wdm_broad_linear(
            run_dir,
            out_dir,
            args.broad_min_freq_mhz,
            args.broad_max_freq_mhz,
            args.trim_start_s,
            source="noisy",
            group="AE",
            filename="03b_broad_multiresolution_WDM_linear_power_noisy_AE.png",
            title="3b. Broad WDM map: noisy A/E (GW+noise), linear power scale",
        ),
        plot_multires_wdm_broad_linear(
            run_dir,
            out_dir,
            args.broad_min_freq_mhz,
            args.broad_max_freq_mhz,
            args.trim_start_s,
            source="noisy",
            group="AE",
            filename="03b2_broad_multiresolution_WDM_linear_power_noisy_AE_noise_visible.png",
            title="3b2. Broad WDM map: noisy A/E with noise floor visible",
            vmin_percentile=0.5,
            vmax_percentile=95.0,
        ),
        plot_multires_wdm_broad_linear(
            run_dir,
            out_dir,
            args.broad_min_freq_mhz,
            args.broad_max_freq_mhz,
            args.trim_start_s,
            source="noisy",
            group="XYZ",
            filename="03c_broad_multiresolution_WDM_linear_power_noisy_XYZ.png",
            title="3c. Broad WDM map: noisy X/Y/Z, linear power scale",
        ),
        plot_multires_wdm_broad_linear(
            run_dir,
            out_dir,
            args.broad_min_freq_mhz,
            args.broad_max_freq_mhz,
            args.trim_start_s,
            source="gw_only",
            group="AE",
            filename="03d_broad_multiresolution_WDM_linear_power_gw_only_AE.png",
            title="3d. Broad WDM map: GW-only A/E, linear power scale",
        ),
        plot_multires_wdm_broad_linear(
            run_dir,
            out_dir,
            args.broad_min_freq_mhz,
            args.broad_max_freq_mhz,
            args.trim_start_s,
            source="gw_only",
            group="XYZ",
            filename="03e_broad_multiresolution_WDM_linear_power_gw_only_XYZ.png",
            title="3e. Broad WDM map: GW-only X/Y/Z, linear power scale",
        ),
        plot_noise_weighted_wdm(run_dir, out_dir, args.min_freq_mhz, args.max_freq_mhz, args.trim_start_s),
        plot_top_pixels(run_dir, out_dir, args.min_freq_mhz, args.max_freq_mhz, args.trim_start_s),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
