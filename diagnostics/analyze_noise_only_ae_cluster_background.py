from __future__ import annotations

import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.run_mbhb_aet_chirp_track_morphology import robust_frequency_score, monotonic_chirp_path, tube_from_path
from scripts.clean_wdm import WDMResolution, compute_wdm_transform, downsample_to_target_dt, require_wdm_dependencies


OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks" / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
LOW_HZ = 1e-5
HIGH_HZ = 1e-1
SPLIT_HZ = 4.882812427240425e-4
RESOLUTIONS = {
    "low": WDMResolution("dt2_M4096", 2.0, 4096),
    "high": WDMResolution("dt2_M512", 2.0, 512),
}
PART_SETTINGS = {
    "low": {"low_hz": LOW_HZ, "high_hz": SPLIT_HZ, "smooth_time_bins": 1, "max_jump_bins": 2, "tube_width_bins": 0},
    "high": {"low_hz": SPLIT_HZ, "high_hz": HIGH_HZ, "smooth_time_bins": 3, "max_jump_bins": 10, "tube_width_bins": 2},
}


@dataclass(frozen=True)
class NoiseInstance:
    seed: int
    tdi_csv: Path


def tdi_path(seed: int) -> Path:
    if seed == 54321:
        return OUTPUT_DIR / "tdi_noise_only_XYZ_AET.csv"
    return OUTPUT_DIR / f"tdi_noise_only_seed{seed}_XYZ_AET.csv"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Blind A/E tube-cluster J/C/energy diagnostics for noise-only TDI instances.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345, 54321, 11111, 22222, 33333])
    parser.add_argument("--output-prefix", default="noise_only_ae_cluster_background")
    return parser.parse_args()


def load_instance(seed: int) -> NoiseInstance:
    path = tdi_path(seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing noise-only TDI for seed {seed}: {path}")
    return NoiseInstance(seed=seed, tdi_csv=path)


def compute_maps(instance: NoiseInstance) -> dict[str, dict[str, np.ndarray]]:
    resample_poly, WDM = require_wdm_dependencies()
    df = pd.read_csv(instance.tdi_csv, usecols=["t_lisa", "A2", "E2", "T2"])
    t_lisa = df["t_lisa"].to_numpy(dtype=float)
    native_dt = float(np.median(np.diff(t_lisa)))
    products: dict[str, dict[str, np.ndarray]] = {}
    for part, resolution in RESOLUTIONS.items():
        wdm = WDM(M=resolution.M, K=32, beta_order=6, precision=6, backend="numba")
        t_ds, dt_ds, _ = downsample_to_target_dt(df["T2"].to_numpy(dtype=float), native_dt, resolution.target_dt, resample_poly)
        t_coeff, times, freqs, _, _ = compute_wdm_transform(
            wdm,
            t_ds,
            sample_rate=1.0 / dt_ds,
            t0=float(t_lisa[0]),
            original_t1=float(t_lisa[-1]),
        )
        floor = np.nanmedian(np.abs(t_coeff) ** 2, axis=1)
        floor = np.where(np.isfinite(floor) & (floor > 0.0), floor, 1e-300)
        products[part] = {
            "times_rel": times - t_lisa[0],
            "freqs_hz": freqs,
            "floor": floor,
        }
        for channel in ("A", "E"):
            x_ds, x_dt, _ = downsample_to_target_dt(df[f"{channel}2"].to_numpy(dtype=float), native_dt, resolution.target_dt, resample_poly)
            coeff, x_times, x_freqs, _, _ = compute_wdm_transform(
                wdm,
                x_ds,
                sample_rate=1.0 / x_dt,
                t0=float(t_lisa[0]),
                original_t1=float(t_lisa[-1]),
            )
            if coeff.shape != t_coeff.shape or not np.array_equal(x_times, times) or not np.array_equal(x_freqs, freqs):
                raise ValueError(f"WDM shape/time/frequency mismatch for seed {instance.seed}, {channel}, {resolution.label}.")
            whitened = coeff / np.sqrt(floor[:, None])
            products[part][f"{channel}_coeff"] = whitened
            products[part][f"{channel}_power"] = np.abs(whitened) ** 2
    return products


def channel_cluster(products: dict[str, np.ndarray], *, channel: str, part: str) -> dict[str, object]:
    settings = PART_SETTINGS[part]
    freqs = np.asarray(products["freqs_hz"], dtype=float)
    times = np.asarray(products["times_rel"], dtype=float)
    coeff_full = np.asarray(products[f"{channel}_coeff"], dtype=np.complex128)
    power_full = np.asarray(products[f"{channel}_power"], dtype=float)
    band = (freqs >= settings["low_hz"]) & (freqs < settings["high_hz"])
    coeff = coeff_full[band, :]
    power = power_full[band, :]
    freqs_band = freqs[band]
    score = robust_frequency_score(power, z_floor=0.35, smooth_size=(3, int(settings["smooth_time_bins"])))
    score = np.maximum(score, 0.0)
    end_cell = tuple(int(v) for v in np.unravel_index(int(np.nanargmax(power)), power.shape))
    path, path_score, _ = monotonic_chirp_path(
        score,
        max_jump_bins=int(settings["max_jump_bins"]),
        step_penalty=0.03,
        jump_penalty=0.03 if part == "low" else 0.004,
        end_cell=end_cell,
    )
    tube = tube_from_path(path, score.shape, width_bins=int(settings["tube_width_bins"]))
    return {
        "times": times,
        "freqs": freqs_band,
        "coeff": coeff,
        "power": power,
        "score": score,
        "path": path,
        "tube": tube,
        "path_score": float(path_score),
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


def analyze_instance(instance: NoiseInstance) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    maps = compute_maps(instance)
    products: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for part in ("low", "high"):
        a = channel_cluster(maps[part], channel="A", part=part)
        e = channel_cluster(maps[part], channel="E", part=part)
        a_tube = np.asarray(a["tube"], dtype=bool)
        e_tube = np.asarray(e["tube"], dtype=bool)
        intersection = a_tube & e_tube
        union = a_tube | e_tube
        c_inter, phase_inter, inter_a_power, inter_e_power = coherence(
            np.asarray(a["coeff"], dtype=np.complex128),
            np.asarray(e["coeff"], dtype=np.complex128),
            intersection,
        )
        c_union, phase_union, union_a_power, union_e_power = coherence(
            np.asarray(a["coeff"], dtype=np.complex128),
            np.asarray(e["coeff"], dtype=np.complex128),
            union,
        )
        products[part] = {"A": a, "E": e, "intersection": intersection, "union": union}
        rows.append(
            {
                "seed": instance.seed,
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
                "intersection_a_power": inter_a_power,
                "intersection_e_power": inter_e_power,
                "union_coherence": c_union,
                "union_phase_rad": phase_union,
                "union_a_power": union_a_power,
                "union_e_power": union_e_power,
            }
        )
    return products, rows


def append_combined_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    combined_rows = []
    for seed, group in df.groupby("seed", sort=False):
        inter = int(group["n_intersection"].sum())
        union = int(group["n_union"].sum())
        finite_inter = np.isfinite(group["intersection_coherence"].to_numpy(dtype=float))
        finite_union = np.isfinite(group["union_coherence"].to_numpy(dtype=float))
        inter_weights = np.maximum(group["n_intersection"].to_numpy(dtype=float), 1.0)
        union_weights = np.maximum(group["n_union"].to_numpy(dtype=float), 1.0)
        combined_rows.append(
            {
                "seed": int(seed),
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
    df["ae_total_power"] = df["a_sum_power"] + df["e_sum_power"]
    df["a_power_fraction"] = df["a_sum_power"] / df["ae_total_power"].replace(0.0, np.nan)
    df["e_over_a_power"] = df["e_sum_power"] / df["a_sum_power"].replace(0.0, np.nan)
    df["log10_ae_total_power"] = np.log10(np.maximum(df["ae_total_power"], 1e-300))
    return df


def score_norm(all_products: dict[int, dict[str, dict[str, object]]]) -> PowerNorm:
    values = []
    for products in all_products.values():
        for part in ("low", "high"):
            for channel in ("A", "E"):
                score = np.asarray(products[part][channel]["score"], dtype=float)
                values.append(score[np.isfinite(score) & (score > 0.0)])
    all_values = np.concatenate(values)
    return PowerNorm(gamma=0.55, vmin=0.0, vmax=float(np.nanpercentile(all_values, 99.5)))


def plot_instances(all_products: dict[int, dict[str, dict[str, object]]], summary: pd.DataFrame, output_prefix: str) -> Path:
    seeds = list(all_products)
    norm = score_norm(all_products)
    output = PLOT_DIR / f"{output_prefix}_tube_maps.png"
    fig, axes = plt.subplots(3, len(seeds), figsize=(max(12.0, 3.0 * len(seeds)), 8.6), sharex=True, sharey=True, constrained_layout=True)
    if len(seeds) == 1:
        axes = axes[:, None]
    image = None
    for col, seed in enumerate(seeds):
        products = all_products[seed]
        for row, channel in enumerate(("A", "E")):
            ax = axes[row, col]
            for part in ("low", "high"):
                cluster = products[part][channel]
                times = np.asarray(cluster["times"], dtype=float) / 86400.0
                freqs = np.asarray(cluster["freqs"], dtype=float)
                score = np.where(np.asarray(cluster["tube"], dtype=bool), np.asarray(cluster["score"], dtype=float), np.nan)
                image = ax.pcolormesh(times, freqs, score, shading="auto", cmap="viridis", norm=norm)
                path = np.asarray(cluster["path"], dtype=int)
                if len(path):
                    ax.plot(times[path[:, 1]], freqs[path[:, 0]], color="white", lw=0.75, alpha=0.9)
            if row == 0:
                ax.set_title(f"seed {seed}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{channel} tube\nFrequency [Hz]")
        ax = axes[2, col]
        for part in ("low", "high"):
            cluster = products[part]["A"]
            times = np.asarray(cluster["times"], dtype=float) / 86400.0
            freqs = np.asarray(cluster["freqs"], dtype=float)
            overlap = np.where(np.asarray(products[part]["intersection"], dtype=bool), 1.0, np.nan)
            ax.pcolormesh(times, freqs, overlap, shading="auto", cmap="Greys", vmin=0.0, vmax=1.0)
        row = summary[(summary["seed"] == seed) & (summary["part"] == "combined_low_high")].iloc[0]
        ax.text(
            0.03,
            0.94,
            f"J={row['jaccard']:.2f}\nC={row['union_coherence']:.2f}\nlogE={row['log10_ae_total_power']:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )
        if col == 0:
            ax.set_ylabel("A/E overlap\nFrequency [Hz]")
        ax.set_xlabel("Time [days]")
        for ax2 in axes[:, col]:
            ax2.axhline(SPLIT_HZ, color="white", lw=0.6, alpha=0.7)
            ax2.set_yscale("log")
            ax2.set_ylim(1e-5, 1e-1)
            ax2.set_xlim(0.0, 150000.0 / 86400.0)
            ax2.grid(True, which="both", alpha=0.11)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes[:2, :].ravel().tolist(), shrink=0.86, pad=0.01)
        cbar.set_label("noise-only channel-specific tube score")
    fig.suptitle("Noise-only blind A/E tube clusters and overlap", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def plot_summary(summary: pd.DataFrame, output_prefix: str) -> Path:
    combined = summary[summary["part"] == "combined_low_high"].copy()
    output = PLOT_DIR / f"{output_prefix}_jc_energy_summary.png"
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), constrained_layout=True)
    x = np.arange(len(combined))
    labels = [str(seed) for seed in combined["seed"]]
    fields = [
        ("jaccard", "J"),
        ("union_coherence", "C"),
        ("log10_ae_total_power", r"$\log_{10}(E_A+E_E)$"),
        ("a_power_fraction", "A energy fraction"),
    ]
    for ax, (field, title) in zip(axes, fields):
        ax.bar(x, combined[field], color="#3b82f6")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[3].set_ylim(0, 1)
    fig.suptitle("Noise-only blind event statistics", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    instances = [load_instance(seed) for seed in args.seeds]
    all_products = {}
    rows = []
    for instance in instances:
        print(f"analyzing noise-only seed {instance.seed}: {instance.tdi_csv}", flush=True)
        products, instance_rows = analyze_instance(instance)
        all_products[instance.seed] = products
        rows.extend(instance_rows)
    summary = append_combined_rows(rows)
    summary_csv = OUTPUT_DIR / f"{args.output_prefix}_summary.csv"
    summary.to_csv(summary_csv, index=False)
    tube_plot = plot_instances(all_products, summary, args.output_prefix)
    stat_plot = plot_summary(summary, args.output_prefix)
    print(summary_csv)
    print(tube_plot)
    print(stat_plot)


if __name__ == "__main__":
    main()
