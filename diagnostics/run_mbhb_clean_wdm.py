from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clean_wdm import CleanWDMConfig, WDMResolution, run_clean_wdm


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"


RESOLUTION_SETS = {
    "smoke": (
        WDMResolution("dt1_M512", 1.0, 512),
        WDMResolution("dt2_M2048", 2.0, 2048),
    ),
    "default": (
        WDMResolution("dt1_M256", 1.0, 256),
        WDMResolution("dt1_M512", 1.0, 512),
        WDMResolution("dt1_M1024", 1.0, 1024),
        WDMResolution("dt2_M2048", 2.0, 2048),
    ),
    "expanded": (
        WDMResolution("dt1_M256", 1.0, 256),
        WDMResolution("dt1_M512", 1.0, 512),
        WDMResolution("dt1_M1024", 1.0, 1024),
        WDMResolution("dt1_M2048", 1.0, 2048),
        WDMResolution("dt2_M2048", 2.0, 2048),
        WDMResolution("dt2_M4096", 2.0, 4096),
        WDMResolution("dt5_M4096", 5.0, 4096),
    ),
}


def plot_aet_overview(npz_path: Path, stats, output: Path, *, source: str, resolution: str, normalized: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    maps = np.load(npz_path)
    suffix = "normalized_power" if normalized else "power"
    signal = np.asarray(maps["signal"], dtype=float)
    signal_start_days = signal[4] / 86400.0
    signal_stop_days = signal[5] / 86400.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, constrained_layout=True)
    panels = []
    for channel in ("A", "E", "T"):
        prefix = f"{source}_AET_{channel}_{resolution}"
        power = np.asarray(maps[f"{prefix}_{suffix}"], dtype=float)
        times = np.asarray(maps[f"{prefix}_times_rel"], dtype=float) / 86400.0
        freqs = np.asarray(maps[f"{prefix}_freqs_hz"], dtype=float)
        band = (freqs >= 1e-5) & (freqs <= 1e-1)
        values = power[band, :]
        panels.append(values)
        image = axes[len(panels) - 1].pcolormesh(
            times,
            freqs[band],
            np.log10(np.maximum(values, 1e-300)),
            shading="auto",
            cmap="magma",
        )
        axes[len(panels) - 1].set_yscale("log")
        axes[len(panels) - 1].set_title(f"{source} {channel}")
        axes[len(panels) - 1].set_xlabel("Time [days]")
        axes[len(panels) - 1].axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.16, lw=0)
        axes[len(panels) - 1].grid(True, which="both", alpha=0.15)
    axes[0].set_ylabel("Frequency [Hz]")

    signal_rows = stats[(stats["source"] == source) & (stats["group"] == "AET") & (stats["resolution"] == resolution)]
    if len(signal_rows):
        peak_times = signal_rows.set_index("channel")["band_raw_power_peak_time_rel"].to_dict()
        peak_freqs = signal_rows.set_index("channel")["band_raw_power_peak_freq_hz"].to_dict()
        for ax, channel in zip(axes, ("A", "E", "T")):
            if channel in peak_times:
                ax.scatter(float(peak_times[channel]) / 86400.0, float(peak_freqs[channel]), marker="*", s=90, color="cyan")

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.84, pad=0.02)
    cbar.set_label("log10 normalized power" if normalized else "log10 power")
    fig.suptitle(f"Clean WDM AET overview: {source}, {resolution}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Run clean WDM transforms for the MBHB diagnostic products.")
    parser.add_argument("--resolution-set", choices=tuple(RESOLUTION_SETS), default="smoke")
    parser.add_argument("--groups", nargs="+", default=["AET"], choices=["links", "XYZ", "AET"])
    parser.add_argument("--sources", nargs="+", default=["noisy", "gw_only"], choices=["noisy", "gw_only"])
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    prefix = args.prefix or f"clean_wdm_{args.resolution_set}_{'_'.join(args.groups).lower()}"
    config = CleanWDMConfig(
        resolutions=RESOLUTION_SETS[args.resolution_set],
        groups=tuple(args.groups),
        sources=tuple(args.sources),
    )
    products = run_clean_wdm(WORK_DIR, config=config, output_prefix=prefix)

    plots = {}
    if not args.no_plot and "AET" in args.groups:
        resolution = config.resolutions[0].label
        source = "noisy" if "noisy" in args.sources else args.sources[0]
        plot_path = products.paths.plot_dir / f"{prefix}_aet_{source}_{resolution}.png"
        plot_aet_overview(
            products.paths.maps_npz,
            products.stats,
            plot_path,
            source=source,
            resolution=resolution,
            normalized=True,
        )
        plots["aet_overview"] = str(plot_path)

    report = {
        "config": {
            "resolution_set": args.resolution_set,
            "resolutions": [
                {"label": item.label, "target_dt": item.target_dt, "M": item.M}
                for item in config.resolutions
            ],
            "groups": list(config.groups),
            "sources": list(config.sources),
            "low_freq_hz": config.low_freq_hz,
            "high_freq_hz": config.high_freq_hz,
        },
        "diagnostics": dict(products.diagnostics),
        "outputs": {
            "stats_csv": str(products.paths.stats_csv),
            "maps_npz": str(products.paths.maps_npz),
            "plots": plots,
        },
    }
    report_path = products.paths.output_dir / f"{prefix}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report = {report_path}")


if __name__ == "__main__":
    main()
