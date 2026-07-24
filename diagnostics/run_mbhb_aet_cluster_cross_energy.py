from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
OUTPUT_DIR = WORK_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
LOW_HZ = 1e-5
HIGH_HZ = 1e-1


def load_complex_channel(maps: np.lib.npyio.NpzFile, *, source: str, channel: str, resolution: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"{source}_{channel}_{resolution}"
    real = np.asarray(maps[f"{prefix}_twhitened_real"], dtype=float)
    imag = np.asarray(maps[f"{prefix}_twhitened_imag"], dtype=float)
    times = np.asarray(maps[f"{prefix}_times_rel"], dtype=float)
    freqs = np.asarray(maps[f"{prefix}_freqs_hz"], dtype=float)
    return real + 1j * imag, times, freqs


def top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    if not (0.0 < fraction <= 1.0):
        raise ValueError("top fraction must be in (0, 1].")
    finite = np.isfinite(values)
    threshold = np.quantile(values[finite], 1.0 - fraction)
    return finite & (values >= threshold)


def cluster_cross_energy(
    maps: np.lib.npyio.NpzFile,
    *,
    source: str,
    resolution: str,
    top_fraction: float,
    mask_mode: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    a, times, freqs = load_complex_channel(maps, source=source, channel="A", resolution=resolution)
    e, e_times, e_freqs = load_complex_channel(maps, source=source, channel="E", resolution=resolution)
    if a.shape != e.shape or not np.array_equal(times, e_times) or not np.array_equal(freqs, e_freqs):
        raise ValueError(f"A and E maps are not aligned for {source} {resolution}.")

    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    band_freq_indices = np.flatnonzero(band)
    a_band = a[band, :]
    e_band = e[band, :]
    a_power = np.abs(a_band) ** 2
    e_power = np.abs(e_band) ** 2
    ae_power = a_power + e_power

    if mask_mode == "ae_energy":
        cluster_mask = top_fraction_mask(ae_power, top_fraction)
    elif mask_mode == "union":
        cluster_mask = top_fraction_mask(a_power, top_fraction) | top_fraction_mask(e_power, top_fraction)
    elif mask_mode == "common":
        cluster_mask = top_fraction_mask(a_power, top_fraction) & top_fraction_mask(e_power, top_fraction)
    else:
        raise ValueError(f"Unknown mask mode: {mask_mode}")

    labels, n_labels = label(cluster_mask, structure=np.ones((3, 3), dtype=bool))
    coherence_map = np.full(labels.shape, np.nan, dtype=float)
    cluster_energy_map = np.full(labels.shape, np.nan, dtype=float)
    coherent_energy_map = np.full(labels.shape, np.nan, dtype=float)

    signal = np.asarray(maps["signal"], dtype=float)
    signal_start_rel = float(signal[4])
    signal_stop_rel = float(signal[5])
    dt = float(np.median(np.diff(times))) if len(times) > 1 else np.nan
    df = float(np.median(np.diff(freqs))) if len(freqs) > 1 else np.nan

    rows = []
    for cluster_id in range(1, n_labels + 1):
        where = labels == cluster_id
        if not np.any(where):
            continue
        fi_local, ti = np.nonzero(where)
        aw = a_band[where]
        ew = e_band[where]
        a_sum = float(np.sum(np.abs(aw) ** 2))
        e_sum = float(np.sum(np.abs(ew) ** 2))
        cross = np.sum(np.conjugate(aw) * ew)
        denom = np.sqrt(max(a_sum * e_sum, 1e-300))
        coherence = float(np.clip(np.abs(cross) / denom, 0.0, 1.0))
        energy = float(a_sum + e_sum)
        coherent_energy = float(energy * coherence)
        freq_indices = band_freq_indices[fi_local]
        peak_local = int(np.argmax(ae_power[where]))
        peak_fi_local = int(fi_local[peak_local])
        peak_ti = int(ti[peak_local])

        coherence_map[where] = coherence
        cluster_energy_map[where] = energy
        coherent_energy_map[where] = coherent_energy

        time_start = float(times[np.min(ti)] - 0.5 * dt)
        time_stop = float(times[np.max(ti)] + 0.5 * dt)
        rows.append(
            {
                "source": source,
                "resolution": resolution,
                "mask_mode": mask_mode,
                "top_fraction": float(top_fraction),
                "cluster_id": int(cluster_id),
                "n_pixels": int(np.sum(where)),
                "time_start_rel": time_start,
                "time_stop_rel": time_stop,
                "freq_min_hz": float(freqs[np.min(freq_indices)]),
                "freq_max_hz": float(freqs[np.max(freq_indices)]),
                "peak_time_rel": float(times[peak_ti]),
                "peak_freq_hz": float(freqs[band_freq_indices[peak_fi_local]]),
                "peak_ae_twhitened_power": float(ae_power[peak_fi_local, peak_ti]),
                "a_energy": a_sum,
                "e_energy": e_sum,
                "cluster_energy": energy,
                "cross_energy_real": float(np.real(cross)),
                "cross_energy_imag": float(np.imag(cross)),
                "cross_energy_abs": float(np.abs(cross)),
                "cross_energy_phase_rad": float(np.angle(cross)),
                "cluster_ae_coherence": coherence,
                "coherent_cluster_energy": coherent_energy,
                "overlaps_signal_window": bool(time_stop >= signal_start_rel and time_start <= signal_stop_rel),
            }
        )

    clusters = pd.DataFrame(rows)
    if not clusters.empty:
        clusters["energy_rank"] = clusters["cluster_energy"].rank(method="first", ascending=False).astype(int)
        clusters["coherent_energy_rank"] = clusters["coherent_cluster_energy"].rank(method="first", ascending=False).astype(int)
        clusters = clusters.sort_values("coherent_cluster_energy", ascending=False).reset_index(drop=True)

    products = {
        "times_rel": times,
        "freqs_hz": freqs,
        "band": band,
        "a_power_band": a_power,
        "e_power_band": e_power,
        "ae_power_band": ae_power,
        "cluster_mask_band": cluster_mask,
        "cluster_labels_band": labels.astype(np.int32),
        "cluster_coherence_band": coherence_map,
        "cluster_energy_band": cluster_energy_map,
        "coherent_cluster_energy_band": coherent_energy_map,
    }
    return clusters, products


def plot_cluster_products(
    output: Path,
    *,
    products_by_source: dict[str, dict[str, np.ndarray]],
    signal: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = list(products_by_source)
    fig, axes = plt.subplots(
        len(sources),
        3,
        figsize=(15.2, max(4.2, 3.7 * len(sources))),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(sources) == 1:
        axes = np.asarray([axes])

    signal_start_days = float(signal[4]) / 86400.0
    signal_stop_days = float(signal[5]) / 86400.0
    images = []
    for row_axes, source in zip(axes, sources):
        products = products_by_source[source]
        times_days = products["times_rel"] / 86400.0
        freqs = products["freqs_hz"][products["band"]]
        panels = [
            ("pixel energy", np.log10(np.maximum(products["ae_power_band"], 1e-300)), "magma", None, None),
            ("cluster coherence", products["cluster_coherence_band"], "viridis", 0.0, 1.0),
            ("coherent cluster energy", np.log10(np.maximum(products["coherent_cluster_energy_band"], 1e-300)), "magma", None, None),
        ]
        for ax, (title, values, cmap, vmin, vmax) in zip(row_axes, panels):
            image = ax.pcolormesh(times_days, freqs, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            images.append((title, image))
            ax.set_title(title)
            ax.set_yscale("log")
            ax.axvspan(signal_start_days, signal_stop_days, color="cyan", alpha=0.13, lw=0)
            ax.grid(True, which="both", alpha=0.14)
            ax.set_xlabel("Time [days]")
        row_axes[0].set_ylabel(f"{source}\nFrequency [Hz]")

    cbar1 = fig.colorbar(images[-3][1], ax=axes[:, 0].tolist(), shrink=0.78, pad=0.012)
    cbar1.set_label(r"$\log_{10}(|A_T|^2 + |E_T|^2)$")
    cbar2 = fig.colorbar(images[-2][1], ax=axes[:, 1].tolist(), shrink=0.78, pad=0.012)
    cbar2.set_label(r"$| \sum A_T^*E_T | / \sqrt{\sum |A_T|^2 \sum |E_T|^2}$")
    cbar3 = fig.colorbar(images[-1][1], ax=axes[:, 2].tolist(), shrink=0.78, pad=0.012)
    cbar3.set_label(r"$\log_{10}(E_\mathrm{cluster}\rho_{AE})$")
    fig.suptitle("Cluster-level complex A/E cross-energy coherence")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Cluster-level complex A/E cross-energy coherence on T-whitened WDM maps.")
    parser.add_argument("--input-prefix", default="twhitened_wdm_pycwb_mra_full_aet")
    parser.add_argument("--resolution", default="dt2_M512")
    parser.add_argument("--sources", nargs="+", choices=("noisy", "gw_only"), default=["noisy", "gw_only"])
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--mask-mode", choices=("ae_energy", "union", "common"), default="ae_energy")
    args = parser.parse_args()

    input_npz = OUTPUT_DIR / f"{args.input_prefix}_maps.npz"
    if not input_npz.exists():
        raise FileNotFoundError(f"Missing complex T-whitened map file: {input_npz}")

    maps = np.load(input_npz)
    signal = np.asarray(maps["signal"], dtype=float)
    frames = []
    products_by_source = {}
    for source in args.sources:
        clusters, products = cluster_cross_energy(
            maps,
            source=source,
            resolution=args.resolution,
            top_fraction=float(args.top_fraction),
            mask_mode=str(args.mask_mode),
        )
        frames.append(clusters)
        products_by_source[source] = products

    cluster_table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tag = f"{args.input_prefix}_{args.resolution}_{args.mask_mode}_top{int(round(100 * args.top_fraction)):02d}_cluster_cross_energy"
    clusters_csv = OUTPUT_DIR / f"{tag}_clusters.csv"
    products_npz = OUTPUT_DIR / f"{tag}_maps.npz"
    plot_png = PLOT_DIR / f"{tag}.png"
    report_json = OUTPUT_DIR / f"{tag}_report.json"

    cluster_table.to_csv(clusters_csv, index=False)
    payload: dict[str, np.ndarray] = {"signal": signal}
    for source, products in products_by_source.items():
        for name, value in products.items():
            payload[f"{source}_{name}"] = value
    np.savez_compressed(products_npz, **payload)
    plot_cluster_products(plot_png, products_by_source=products_by_source, signal=signal)

    summary = {}
    for source in args.sources:
        subset = cluster_table[cluster_table["source"] == source].copy()
        if subset.empty:
            summary[source] = {"n_clusters": 0}
            continue
        top_energy = subset.sort_values("cluster_energy", ascending=False).iloc[0]
        top_coherent = subset.sort_values("coherent_cluster_energy", ascending=False).iloc[0]
        summary[source] = {
            "n_clusters": int(len(subset)),
            "n_mask_pixels": int(np.sum(products_by_source[source]["cluster_mask_band"])),
            "top_plain_energy_cluster_id": int(top_energy["cluster_id"]),
            "top_plain_energy_overlaps_signal": bool(top_energy["overlaps_signal_window"]),
            "top_plain_energy": float(top_energy["cluster_energy"]),
            "top_plain_energy_coherence": float(top_energy["cluster_ae_coherence"]),
            "top_coherent_cluster_id": int(top_coherent["cluster_id"]),
            "top_coherent_overlaps_signal": bool(top_coherent["overlaps_signal_window"]),
            "top_coherent_energy": float(top_coherent["coherent_cluster_energy"]),
            "top_coherent_plain_energy": float(top_coherent["cluster_energy"]),
            "top_coherent_coherence": float(top_coherent["cluster_ae_coherence"]),
        }

    report = {
        "method": "connected TF clusters scored by complex cross-energy Z_AE=sum conj(A_T) E_T",
        "input_npz": str(input_npz),
        "resolution": args.resolution,
        "top_fraction": float(args.top_fraction),
        "mask_mode": str(args.mask_mode),
        "sources": list(args.sources),
        "cluster_statistic": {
            "cluster_energy": "sum |A_T|^2 + sum |E_T|^2",
            "cluster_ae_coherence": "|sum conj(A_T) E_T| / sqrt(sum |A_T|^2 sum |E_T|^2)",
            "coherent_cluster_energy": "cluster_energy * cluster_ae_coherence",
        },
        "summary": summary,
        "outputs": {
            "clusters_csv": str(clusters_csv),
            "products_npz": str(products_npz),
            "plot": str(plot_png),
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report = {report_json}")


if __name__ == "__main__":
    main()
