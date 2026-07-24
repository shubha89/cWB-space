"""Overlay multi-resolution coronagraphic clusters on one WDM power map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_one_smbhb as proto  # noqa: E402


def reconstruct_background(run_dir: Path, background_m: int | None):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config_data = dict(summary["config"])
    if background_m is None:
        background_m = max(int(value) for value in summary.get("wdm_ms", [config_data["wdm_m"]]))
    config_data["wdm_m"] = int(background_m)
    config = proto.RunConfig(**config_data)
    _, _, resample_poly, _, WDM = proto.require_dependencies()

    work_dir = PROJECT_ROOT / config.work_dir
    metadata = proto.load_source_metadata(work_dir)
    t_raw, link_raw = proto.load_links(work_dir, config.source)
    center_s = float(metadata.get("raw_half_duration_s", 150_000.0))
    if "peak_time_s" in metadata:
        center_s = float(t_raw[0]) + 0.5 * (float(t_raw[-1]) - float(t_raw[0]))
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

    power_a, freqs, times = proto.wdm_power(a_t, dt=dt_sel, t0=float(t_sel[0]), config=config, WDM=WDM)
    power_e, _, _ = proto.wdm_power(e_t, dt=dt_sel, t0=float(t_sel[0]), config=config, WDM=WDM)
    ae_power = power_a + power_e
    background, _ = proto.normalize_selection_power(
        ae_power,
        mode=config.selection_normalization,
        offsource_edge_fraction=config.offsource_edge_fraction,
    )
    return background, freqs, times, int(background_m), float(t_sel[0])


def image_extent(times: np.ndarray, freqs: np.ndarray, chunk_start_s: float) -> list[float]:
    time_step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    freq_step = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 1.0
    return [
        float(times[0] - 0.5 * time_step - chunk_start_s),
        float(times[-1] + 0.5 * time_step - chunk_start_s),
        float((freqs[0] - 0.5 * freq_step) * 1.0e3),
        float((freqs[-1] + 0.5 * freq_step) * 1.0e3),
    ]


def background_colorbar_label(run_dir: Path) -> str:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    normalization = summary.get("config", {}).get("selection_normalization", "raw")
    if normalization != "raw":
        return f"log10({normalization} A+E WDM selection power)"
    return "log10(A+E WDM power)"


def plot_wdm_only(
    run_dir: Path,
    background_m: int | None,
    min_freq_mhz: float | None,
    max_freq_mhz: float | None,
    trim_start_s: float,
    trim_end_s: float,
    log_freq_axis: bool,
) -> Path:
    ae_power, freqs, times, resolved_background_m, chunk_start_s = reconstruct_background(run_dir, background_m)
    background = np.log10(np.maximum(ae_power, 1.0e-300))
    extent = image_extent(times, freqs, chunk_start_s)

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if trim_start_s > 0.0 or trim_end_s > 0.0:
        suffix = f"_trim_start_{trim_start_s:g}s_end_{trim_end_s:g}s"
    if min_freq_mhz is not None or max_freq_mhz is not None:
        min_label = "auto" if min_freq_mhz is None else f"{min_freq_mhz:g}"
        max_label = "auto" if max_freq_mhz is None else f"{max_freq_mhz:g}"
        suffix = f"{suffix}_freq_{min_label}_to_{max_label}mHz"
    if log_freq_axis:
        suffix = f"{suffix}_logfreq"
    out_path = plot_dir / f"wdm_only_m_{resolved_background_m}{suffix}.png"

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(background, origin="lower", aspect="auto", cmap="magma", extent=extent)
    fig.colorbar(im, ax=ax, label=background_colorbar_label(run_dir))
    ax.set_xlabel("time since chunk start [s]")
    ax.set_ylabel("frequency [mHz]")
    ax.set_title(f"WDM-only A+E map (M={resolved_background_m})")
    ax.set_xlim(extent[0] + float(trim_start_s), extent[1] - float(trim_end_s))
    if log_freq_axis:
        positive = freqs[freqs > 0.0] * 1.0e3
        ax.set_yscale("log")
        ax.set_ylim(
            min_freq_mhz if min_freq_mhz is not None else float(np.nanmin(positive)),
            max_freq_mhz if max_freq_mhz is not None else 110.0,
        )
    else:
        ax.set_ylim(min_freq_mhz if min_freq_mhz is not None else 0.0, max_freq_mhz if max_freq_mhz is not None else 110.0)
    ax.grid(color="white", alpha=0.18, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_multires_overlay(
    run_dir: Path,
    background_m: int | None,
    min_freq_mhz: float | None,
    max_freq_mhz: float | None,
    trim_start_s: float = 0.0,
    trim_end_s: float = 0.0,
    log_freq_axis: bool = False,
) -> tuple[Path, Path]:
    clusters_path = run_dir / "pixel_clusters_all_resolutions.csv"
    if not clusters_path.exists():
        raise FileNotFoundError(f"Missing multi-resolution clusters file: {clusters_path}")
    clusters = pd.read_csv(clusters_path)
    ae_power, freqs, times, resolved_background_m, chunk_start_s = reconstruct_background(run_dir, background_m)
    background = np.log10(np.maximum(ae_power, 1.0e-300))
    colorbar_label = background_colorbar_label(run_dir)
    extent = image_extent(times, freqs, chunk_start_s)

    clusters = clusters.copy()
    clusters["wdm_m"] = clusters["wdm_m"].astype(int)
    clusters["time_offset_s"] = clusters["time_s"] - chunk_start_s
    clusters["freq_mhz"] = clusters["freq_hz"] * 1.0e3

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    wdm_ms = sorted(clusters["wdm_m"].unique())
    palette = plt.get_cmap("tab10")
    colors = {wdm_m: palette(i % palette.N) for i, wdm_m in enumerate(wdm_ms)}
    markers = {128: "o", 256: "s", 512: "^", 1024: "D"}

    suffix = ""
    if trim_start_s > 0.0 or trim_end_s > 0.0:
        suffix = f"_trim_start_{trim_start_s:g}s_end_{trim_end_s:g}s"
    if min_freq_mhz is not None or max_freq_mhz is not None:
        min_label = "auto" if min_freq_mhz is None else f"{min_freq_mhz:g}"
        max_label = "auto" if max_freq_mhz is None else f"{max_freq_mhz:g}"
        suffix = f"{suffix}_freq_{min_label}_to_{max_label}mHz"
    if log_freq_axis:
        suffix = f"{suffix}_logfreq"
    all_path = plot_dir / f"multires_selected_overlay_on_wdm{suffix}.png"
    coherent_path = plot_dir / f"multires_coherent_overlay_on_wdm{suffix}.png"

    def setup_axis(title: str):
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(background, origin="lower", aspect="auto", cmap="magma", extent=extent)
        fig.colorbar(im, ax=ax, label=colorbar_label)
        ax.set_xlabel("time since chunk start [s]")
        ax.set_ylabel("frequency [mHz]")
        ax.set_title(title)
        ax.set_xlim(extent[0] + float(trim_start_s), extent[1] - float(trim_end_s))
        y_max = max_freq_mhz if max_freq_mhz is not None else max(110.0, float(np.nanmax(clusters["freq_mhz"])) * 1.15)
        if log_freq_axis:
            positive = freqs[freqs > 0.0] * 1.0e3
            ax.set_yscale("log")
            ax.set_ylim(min_freq_mhz if min_freq_mhz is not None else float(np.nanmin(positive)), y_max)
        else:
            ax.set_ylim(min_freq_mhz if min_freq_mhz is not None else 0.0, y_max)
        ax.grid(color="white", alpha=0.18, linewidth=0.7)
        return fig, ax

    fig, ax = setup_axis(f"Multi-resolution selected pixels on WDM background (background M={resolved_background_m})")
    noise = clusters[clusters["cluster"] < 0]
    if len(noise):
        ax.scatter(
            noise["time_offset_s"],
            noise["freq_mhz"],
            s=22,
            marker="x",
            color="#e6e6e6",
            linewidths=0.8,
            alpha=0.65,
            label=f"noise / unclustered (n={len(noise)})",
        )
    for wdm_m in wdm_ms:
        rows = clusters[(clusters["wdm_m"] == wdm_m) & (clusters["cluster"] >= 0)]
        if len(rows) == 0:
            continue
        ax.scatter(
            rows["time_offset_s"],
            rows["freq_mhz"],
            s=54,
            marker=markers.get(int(wdm_m), "o"),
            facecolors=colors[int(wdm_m)],
            edgecolors="white",
            linewidths=0.9,
            alpha=0.88,
            label=f"M={wdm_m} coherent (n={len(rows)})",
        )
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(all_path, dpi=180)
    plt.close(fig)

    coherent = clusters[clusters["cluster"] >= 0]
    fig, ax = setup_axis(f"Multi-resolution coherent pixels on WDM background (background M={resolved_background_m})")
    for wdm_m in wdm_ms:
        rows = coherent[coherent["wdm_m"] == wdm_m]
        if len(rows) == 0:
            continue
        ax.scatter(
            rows["time_offset_s"],
            rows["freq_mhz"],
            s=62,
            marker=markers.get(int(wdm_m), "o"),
            facecolors=colors[int(wdm_m)],
            edgecolors="white",
            linewidths=1.0,
            alpha=0.9,
            label=f"M={wdm_m} (n={len(rows)})",
        )
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(coherent_path, dpi=180)
    plt.close(fig)

    return all_path, coherent_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay multi-resolution coronagraphic pixels on a WDM map.")
    parser.add_argument("--run-dir", default="choronagraphic-cluster/runs/smbhb_00000_multires_test")
    parser.add_argument("--background-m", type=int, default=None)
    parser.add_argument("--min-freq-mhz", type=float, default=None)
    parser.add_argument("--max-freq-mhz", type=float, default=None)
    parser.add_argument("--trim-start-s", type=float, default=0.0, help="Hide this many seconds from the WDM map start.")
    parser.add_argument("--trim-end-s", type=float, default=0.0, help="Hide this many seconds from the WDM map end.")
    parser.add_argument("--log-freq-axis", action="store_true", help="Use a logarithmic frequency axis.")
    parser.add_argument("--wdm-only", action="store_true", help="Write only the WDM background map with no pixel overlay.")
    args = parser.parse_args()
    if args.wdm_only:
        print(
            plot_wdm_only(
                PROJECT_ROOT / args.run_dir,
                background_m=args.background_m,
                min_freq_mhz=args.min_freq_mhz,
                max_freq_mhz=args.max_freq_mhz,
                trim_start_s=args.trim_start_s,
                trim_end_s=args.trim_end_s,
                log_freq_axis=args.log_freq_axis,
            )
        )
        return
    all_path, coherent_path = plot_multires_overlay(
        PROJECT_ROOT / args.run_dir,
        background_m=args.background_m,
        min_freq_mhz=args.min_freq_mhz,
        max_freq_mhz=args.max_freq_mhz,
        trim_start_s=args.trim_start_s,
        trim_end_s=args.trim_end_s,
        log_freq_axis=args.log_freq_axis,
    )
    print(all_path)
    print(coherent_path)


if __name__ == "__main__":
    main()
