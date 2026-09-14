from __future__ import annotations

from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.blind_event_clustering import (
    SeededChirpEventConfig,
    compute_self_ae_median_wdm,
    find_seeded_chirp_events,
)
from scripts.clean_wdm import WDMResolution


SECONDS_PER_DAY = 86_400.0
INPUT_HDF = Path("data/LDC2_sangria_blind_v2.h5")
OUTPUT_DIR = Path("data/sangria_full_mbhb_search")


def xyz_to_aet(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (z - x) / np.sqrt(2.0), (x - 2.0 * y + z) / np.sqrt(6.0), (x + y + z) / np.sqrt(3.0)


def load_window(start_index: int, stop_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(INPUT_HDF, "r") as h5:
        arr = h5["obs/tdi"][start_index:stop_index, 0]
    a, e, _ = xyz_to_aet(arr["X"].astype(float), arr["Y"].astype(float), arr["Z"].astype(float))
    return arr["t"].astype(float), a, e


def chunk_ranges(n_samples: int, dt: float, chunk_days: float) -> list[tuple[int, int, int]]:
    chunk_samples = int(round(chunk_days * SECONDS_PER_DAY / dt))
    ranges = []
    for chunk_number, start in enumerate(range(0, n_samples, chunk_samples), start=1):
        stop = min(n_samples, start + chunk_samples)
        if stop - start > 2:
            ranges.append((chunk_number, start, stop))
    return ranges


def make_candidate_plots(events: pd.DataFrame, outfile: Path) -> None:
    if events.empty:
        return
    top = events.sort_values("event_statistic", ascending=False).head(40).copy()
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=True, constrained_layout=True)

    sc = axes[0].scatter(
        top["peak_time_day"],
        top["peak_freq_hz"],
        c=np.log10(np.maximum(top["event_statistic"], 1.0e-300)),
        s=20 + 2.0 * np.sqrt(np.maximum(top["n_pixels"], 1)),
        cmap="viridis",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("peak frequency [Hz]")
    axes[0].set_title("Top Sangria self-AE WDM events")
    fig.colorbar(sc, ax=axes[0], label=r"$\log_{10}$ event statistic")

    axes[1].scatter(top["peak_time_day"], top["ae_complex_coherence"], c="#0072B2", s=28)
    axes[1].set_ylabel("A/E coherence")
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].scatter(top["peak_time_day"], top["monotonic_fraction"], c="#D55E00", s=28)
    axes[2].set_ylabel("chirp monotonic fraction")
    axes[2].set_xlabel("Sangria time [day]")
    axes[2].set_ylim(-0.05, 1.05)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resolution = WDMResolution("dt5_M512", 5.0, 512)
    config = SeededChirpEventConfig(
        low_hz=1.0e-5,
        high_hz=2.0e-2,
        seed_quantile=0.99,
        grow_quantile=0.95,
        max_freq_jump_bins=4,
        same_time_connect_bins=1,
        min_pixels=8,
        min_time_bins=2,
        freq_tolerance_bins=1.0,
        max_events=12,
    )

    with h5py.File(INPUT_HDF, "r") as h5:
        dset = h5["obs/tdi"]
        n_samples = int(dset.shape[0])
        dt = float(dset.attrs["dt"])

    rows = []
    for chunk_number, start, stop in chunk_ranges(n_samples, dt, chunk_days=30.0):
        t, a, e = load_window(start, stop)
        start_day = float(t[0] / SECONDS_PER_DAY)
        stop_day = float(t[-1] / SECONDS_PER_DAY)
        print(f"chunk {chunk_number:02d}: days {start_day:.2f}-{stop_day:.2f}, samples={len(t)}", flush=True)
        maps = compute_self_ae_median_wdm(t, a, e, resolution=resolution)
        event_frame, _ = find_seeded_chirp_events(
            maps,
            config=config,
            start_day=0.5,
            stop_day=(stop_day - start_day) - 0.5,
            label=f"sangria_chunk{chunk_number:02d}",
        )
        if event_frame.empty:
            continue
        for col in ["start_day", "stop_day", "peak_time_day", "search_start_day", "search_stop_day"]:
            event_frame[col] = event_frame[col] + start_day
        event_frame["chunk"] = chunk_number
        event_frame["chunk_start_day"] = start_day
        event_frame["chunk_stop_day"] = stop_day
        rows.append(event_frame)

    if rows:
        events = pd.concat(rows, ignore_index=True)
        events = events.sort_values("event_statistic", ascending=False).reset_index(drop=True)
        events["global_rank"] = np.arange(1, len(events) + 1)
    else:
        events = pd.DataFrame()

    events.to_csv(OUTPUT_DIR / "sangria_full_mbhb_candidate_events.csv", index=False)
    make_candidate_plots(events, OUTPUT_DIR / "plots" / "sangria_full_mbhb_candidate_summary.png")

    summary_cols = [
        "global_rank",
        "chunk",
        "rank",
        "peak_time_day",
        "peak_freq_hz",
        "start_day",
        "stop_day",
        "freq_min_hz",
        "freq_max_hz",
        "n_pixels",
        "n_time_bins",
        "ae_total_power",
        "ae_balance",
        "ae_complex_coherence",
        "monotonic_fraction",
        "event_statistic",
    ]
    print("\nTop candidates:")
    print(events.head(30)[summary_cols].to_string(index=False) if not events.empty else "no events found")
    print(f"\nsaved outputs under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
