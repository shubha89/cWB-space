from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.run_mbhb_aet_chirp_track_morphology import robust_frequency_score, monotonic_chirp_path, tube_from_path


OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
CASES = [
    ("1e4", r"$10^4$"),
    ("1e4p5", r"$10^{4.5}$"),
    ("1e5", r"$10^5$"),
    ("1e6", r"$10^6$"),
    ("1e7", r"$10^7$"),
    ("1e7p5", r"$10^{7.5}$"),
    ("1e8", r"$10^8$"),
    ("1e8p5", r"$10^{8.5}$"),
]
SOURCES = ("gw_only", "noisy")
RESOLUTION_FOR_PART = {"low": "dt2_M4096", "high": "dt2_M512"}
SPLIT_HZ = 4.882812427240425e-4
LOW_HZ = 1e-5
HIGH_HZ = 1e-1
PART_SETTINGS = {
    "low": {"low_hz": LOW_HZ, "high_hz": SPLIT_HZ, "smooth_time_bins": 1, "max_jump_bins": 2, "tube_width_bins": 0},
    "high": {"low_hz": SPLIT_HZ, "high_hz": HIGH_HZ, "smooth_time_bins": 3, "max_jump_bins": 10, "tube_width_bins": 2},
}


def parse_args() -> object:
    parser = ArgumentParser(description="Diagnose separate A/E tube-cluster overlap and coherence.")
    parser.add_argument(
        "--case-set",
        choices=("baseline", "double_distance"),
        default="baseline",
        help="Which mass tag set to analyze.",
    )
    parser.add_argument("--output-suffix", default=None, help="Suffix inserted before output extension.")
    return parser.parse_args()


def selected_cases(case_set: str) -> list[tuple[str, str]]:
    if case_set == "baseline":
        return CASES
    return [(f"{tag}_d2", label) for tag, label in CASES]


def map_path(tag: str, resolution: str) -> Path:
    if tag == "1e5":
        return OUTPUT_DIR / "twhitened_wdm_pycwb_mra_full_aet_maps.npz"
    if resolution == "dt2_M512":
        return OUTPUT_DIR / f"twhitened_wdm_mbhb_{tag}_aet_maps.npz"
    return OUTPUT_DIR / f"twhitened_wdm_mbhb_{tag}_{resolution}_aet_maps.npz"


def load_resolution(tag: str, resolution: str) -> np.lib.npyio.NpzFile:
    path = map_path(tag, resolution)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def load_complex(data: np.lib.npyio.NpzFile, *, source: str, channel: str, resolution: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"{source}_{channel}_{resolution}"
    real = np.asarray(data[f"{prefix}_twhitened_real"], dtype=float)
    imag = np.asarray(data[f"{prefix}_twhitened_imag"], dtype=float)
    power = np.asarray(data[f"{prefix}_twhitened_power"], dtype=float)
    times = np.asarray(data[f"{prefix}_times_rel"], dtype=float)
    freqs = np.asarray(data[f"{prefix}_freqs_hz"], dtype=float)
    return real + 1j * imag, power, times, freqs


def time_overlap_mask(times: np.ndarray, signal: np.ndarray) -> np.ndarray:
    half_width = 0.5 * float(np.nanmedian(np.diff(times))) if len(times) > 1 else 0.0
    return (times + half_width >= signal[4]) & (times - half_width <= signal[5])


def channel_cluster(
    data: np.lib.npyio.NpzFile,
    *,
    source: str,
    channel: str,
    part: str,
) -> dict[str, object]:
    resolution = RESOLUTION_FOR_PART[part]
    coeff, power_full, times, freqs_full = load_complex(data, source=source, channel=channel, resolution=resolution)
    settings = PART_SETTINGS[part]
    band = (freqs_full >= settings["low_hz"]) & (freqs_full < settings["high_hz"])
    coeff = coeff[band, :]
    power = power_full[band, :]
    freqs = freqs_full[band]
    score = robust_frequency_score(power, z_floor=0.35, smooth_size=(3, int(settings["smooth_time_bins"])))
    signal = np.asarray(data["signal"], dtype=float)
    time_mask = time_overlap_mask(times, signal)
    anchor_power = np.where(time_mask[None, :], power, -np.inf)
    if not np.any(np.isfinite(anchor_power)):
        anchor_power = power
    end_cell = tuple(int(v) for v in np.unravel_index(int(np.nanargmax(anchor_power)), anchor_power.shape))
    path, path_score, _ = monotonic_chirp_path(
        score,
        max_jump_bins=int(settings["max_jump_bins"]),
        step_penalty=0.03,
        jump_penalty=0.03 if part == "low" else 0.004,
        end_cell=end_cell,
    )
    tube = tube_from_path(path, score.shape, width_bins=int(settings["tube_width_bins"]))
    return {
        "coeff": coeff,
        "power": power,
        "score": np.maximum(score, 0.0),
        "times": times,
        "freqs": freqs,
        "path": path,
        "tube": tube,
        "path_score": float(path_score),
        "anchor_cell": end_cell,
    }


def coherence(a: np.ndarray, e: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan
    av = a[mask]
    ev = e[mask]
    a_energy = float(np.sum(np.abs(av) ** 2))
    e_energy = float(np.sum(np.abs(ev) ** 2))
    cross = np.sum(np.conjugate(av) * ev)
    denom = np.sqrt(max(a_energy * e_energy, 1e-300))
    return float(np.abs(cross) / denom), float(np.angle(cross)), a_energy, e_energy


def analyze_case(tag: str, source: str) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    by_part: dict[str, dict[str, object]] = {}
    rows = []
    for part in ("low", "high"):
        resolution = RESOLUTION_FOR_PART[part]
        data = load_resolution(tag, resolution)
        a = channel_cluster(data, source=source, channel="A", part=part)
        e = channel_cluster(data, source=source, channel="E", part=part)
        a_tube = np.asarray(a["tube"], dtype=bool)
        e_tube = np.asarray(e["tube"], dtype=bool)
        intersection = a_tube & e_tube
        union = a_tube | e_tube
        a_coeff = np.asarray(a["coeff"], dtype=np.complex128)
        e_coeff = np.asarray(e["coeff"], dtype=np.complex128)
        c_inter, phase_inter, a_energy_inter, e_energy_inter = coherence(a_coeff, e_coeff, intersection)
        c_union, phase_union, a_energy_union, e_energy_union = coherence(a_coeff, e_coeff, union)
        by_part[part] = {"data": data, "A": a, "E": e, "intersection": intersection, "union": union}
        rows.append(
            {
                "tag": tag,
                "source": source,
                "part": part,
                "n_a_tube": int(np.sum(a_tube)),
                "n_e_tube": int(np.sum(e_tube)),
                "n_intersection": int(np.sum(intersection)),
                "n_union": int(np.sum(union)),
                "jaccard": float(np.sum(intersection) / np.sum(union)) if np.any(union) else np.nan,
                "a_sum_score": float(np.sum(np.asarray(a["score"])[a_tube])) if np.any(a_tube) else 0.0,
                "e_sum_score": float(np.sum(np.asarray(e["score"])[e_tube])) if np.any(e_tube) else 0.0,
                "a_sum_power": float(np.sum(np.asarray(a["power"])[a_tube])) if np.any(a_tube) else 0.0,
                "e_sum_power": float(np.sum(np.asarray(e["power"])[e_tube])) if np.any(e_tube) else 0.0,
                "intersection_coherence": c_inter,
                "intersection_phase_rad": phase_inter,
                "intersection_a_power": a_energy_inter,
                "intersection_e_power": e_energy_inter,
                "union_coherence": c_union,
                "union_phase_rad": phase_union,
                "union_a_power": a_energy_union,
                "union_e_power": e_energy_union,
            }
        )
    return by_part, rows


def collect(source: str, cases: list[tuple[str, str]]) -> tuple[dict[str, dict[str, dict[str, object]]], pd.DataFrame]:
    products = {}
    rows = []
    for tag, _ in cases:
        products[tag], case_rows = analyze_case(tag, source)
        rows.extend(case_rows)
    df = pd.DataFrame(rows)
    combined_rows = []
    for tag, group in df.groupby("tag", sort=False):
        inter = int(group["n_intersection"].sum())
        union = int(group["n_union"].sum())
        finite_inter = np.isfinite(group["intersection_coherence"].to_numpy(dtype=float))
        finite_union = np.isfinite(group["union_coherence"].to_numpy(dtype=float))
        inter_weights = np.maximum(group["n_intersection"].to_numpy(dtype=float), 1.0)
        union_weights = np.maximum(group["n_union"].to_numpy(dtype=float), 1.0)
        combined_rows.append(
            {
                "tag": tag,
                "source": source,
                "part": "combined_low_high",
                "n_a_tube": int(group["n_a_tube"].sum()),
                "n_e_tube": int(group["n_e_tube"].sum()),
                "n_intersection": inter,
                "n_union": union,
                "jaccard": float(inter / union) if union else np.nan,
                "a_sum_score": float(group["a_sum_score"].sum()),
                "e_sum_score": float(group["e_sum_score"].sum()),
                "a_sum_power": float(group["a_sum_power"].sum()),
                "e_sum_power": float(group["e_sum_power"].sum()),
                "intersection_coherence": (
                    float(np.average(group["intersection_coherence"].to_numpy(dtype=float)[finite_inter], weights=inter_weights[finite_inter]))
                    if np.any(finite_inter)
                    else np.nan
                ),
                "union_coherence": (
                    float(np.average(group["union_coherence"].to_numpy(dtype=float)[finite_union], weights=union_weights[finite_union]))
                    if np.any(finite_union)
                    else np.nan
                ),
            }
        )
    df = pd.concat([df, pd.DataFrame(combined_rows)], ignore_index=True)
    return products, df


def norm_for(products: dict[str, dict[str, dict[str, object]]]) -> PowerNorm:
    values = []
    for case in products.values():
        for part in ("low", "high"):
            for channel in ("A", "E"):
                score = np.asarray(case[part][channel]["score"], dtype=float)
                values.append(score[np.isfinite(score) & (score > 0)])
    all_values = np.concatenate(values)
    return PowerNorm(gamma=0.55, vmin=0.0, vmax=float(np.nanpercentile(all_values, 99.5)))


def plot_source(
    source: str,
    products: dict[str, dict[str, dict[str, object]]],
    summary: pd.DataFrame,
    cases: list[tuple[str, str]],
    *,
    output_suffix: str,
) -> Path:
    output = PLOT_DIR / f"twhitened_wdm_mbhb_mass_sweep_{source}_ae_separate_cluster_correlation{output_suffix}.png"
    norm = norm_for(products)
    fig, axes = plt.subplots(3, len(cases), figsize=(21.5, 9.0), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for col, (tag, title) in enumerate(cases):
        signal = np.asarray(products[tag]["high"]["data"]["signal"], dtype=float)
        for row, channel in enumerate(("A", "E")):
            ax = axes[row, col]
            for part in ("low", "high"):
                cluster = products[tag][part][channel]
                times = np.asarray(cluster["times"], dtype=float) / 86400.0
                freqs = np.asarray(cluster["freqs"], dtype=float)
                score = np.where(np.asarray(cluster["tube"], dtype=bool), np.asarray(cluster["score"], dtype=float), np.nan)
                image = ax.pcolormesh(times, freqs, score, shading="auto", cmap="viridis", norm=norm)
                path = np.asarray(cluster["path"], dtype=int)
                if len(path):
                    ax.plot(times[path[:, 1]], freqs[path[:, 0]], color="white", lw=0.7, alpha=0.9)
            ax.axvspan(signal[4] / 86400.0, signal[5] / 86400.0, color="cyan", alpha=0.12, lw=0)
            ax.axhline(SPLIT_HZ, color="white", lw=0.55, alpha=0.7)
            if row == 0:
                ax.set_title(rf"$M={title}\ M_\odot$", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{channel} tube\nFrequency [Hz]")
        ax = axes[2, col]
        for part in ("low", "high"):
            cluster = products[tag][part]["A"]
            times = np.asarray(cluster["times"], dtype=float) / 86400.0
            freqs = np.asarray(cluster["freqs"], dtype=float)
            overlap = np.where(np.asarray(products[tag][part]["intersection"], dtype=bool), 1.0, np.nan)
            ax.pcolormesh(times, freqs, overlap, shading="auto", cmap="Greys", vmin=0.0, vmax=1.0)
        combined = summary[(summary["tag"] == tag) & (summary["part"] == "combined_low_high")].iloc[0]
        ax.text(
            0.03,
            0.94,
            f"J={combined['jaccard']:.2f}\nC={combined['union_coherence']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )
        ax.axvspan(signal[4] / 86400.0, signal[5] / 86400.0, color="cyan", alpha=0.12, lw=0)
        ax.axhline(SPLIT_HZ, color="white", lw=0.55, alpha=0.7)
        if col == 0:
            ax.set_ylabel("A/E tube overlap\nFrequency [Hz]")
        ax.set_xlabel("Time [days]")
        for ax2 in axes[:, col]:
            ax2.set_yscale("log")
            ax2.set_ylim(1e-5, 1e-1)
            ax2.set_xlim(0.0, 150000.0 / 86400.0)
            ax2.grid(True, which="both", alpha=0.11)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes[:2, :].ravel().tolist(), shrink=0.86, pad=0.01)
        cbar.set_label("channel-specific tube score")
    fig.suptitle(f"{source}: separate A/E tube clusters and cluster correlation", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    cases = selected_cases(args.case_set)
    output_suffix = args.output_suffix
    if output_suffix is None:
        output_suffix = "" if args.case_set == "baseline" else "_double_distance"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    outputs = []
    for source in SOURCES:
        products, summary = collect(source, cases)
        outputs.append(plot_source(source, products, summary, cases, output_suffix=output_suffix))
        all_rows.append(summary)
    summary_path = OUTPUT_DIR / f"twhitened_wdm_mbhb_mass_sweep_ae_separate_cluster_correlation{output_suffix}.csv"
    pd.concat(all_rows, ignore_index=True).to_csv(summary_path, index=False)
    for output in outputs:
        print(output)
    print(summary_path)


if __name__ == "__main__":
    main()
