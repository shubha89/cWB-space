from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = PROJECT_ROOT / "docs" / "paper" / "cwb_space_prd"
FIG_DIR = PAPER_DIR / "figures"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.blind_event_clustering import SeededChirpEventConfig, find_seeded_chirp_events

TWO_WEEK_DIR = PROJECT_ROOT / "diagnostics" / "simulated_2week_mbhb_then_glitch_highrate" / "outputs"
GENERIC_DIR = PROJECT_ROOT / "diagnostics" / "generic_burst_response_validation" / "outputs"
SANGRIA_FULL_DIR = PROJECT_ROOT / "data" / "sangria_full_mbhb_search"
SANGRIA_FOCUSED_DIR = PROJECT_ROOT / "data" / "sangria_focused_mbhb_candidate"
SNR_STUDY_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_snr_event_statistic_study" / "outputs"

LOW_HZ = 1.0e-5
WDM_HIGH_HZ = 2.0e-2
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str, default: float = np.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def band_indices(freqs: np.ndarray, high_hz: float = WDM_HIGH_HZ) -> np.ndarray:
    return np.flatnonzero((freqs >= LOW_HZ) & (freqs <= high_hz))


def available_low_hz(freqs: np.ndarray, high_hz: float = WDM_HIGH_HZ) -> float:
    idx = band_indices(freqs, high_hz=high_hz)
    if len(idx) == 0:
        return LOW_HZ
    positive = np.asarray(freqs[idx], dtype=float)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return LOW_HZ
    return float(np.nanmin(positive))


def log_power_limits(power: np.ndarray, lo: float = 5.0, hi: float = 99.5) -> tuple[float, float]:
    values = np.log10(np.maximum(power, 1.0e-12))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    return tuple(np.nanpercentile(finite, [lo, hi]))


def positive_log_limits(values: np.ndarray, lo: float = 0.50, hi: float = 0.995) -> tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values) & (values > 0.0)], dtype=float)
    if finite.size == 0:
        return 1.0e-6, 1.0
    vmin = float(np.nanquantile(finite, lo))
    vmax = float(np.nanquantile(finite, hi))
    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = float(np.nanmin(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = 10.0 * vmin
    return vmin, vmax


def plot_wdm(
    ax: plt.Axes,
    times: np.ndarray,
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    title: str,
    xlim: tuple[float, float] | None = None,
    cmap: str = "magma",
    high_hz: float = WDM_HIGH_HZ,
) -> matplotlib.cm.ScalarMappable:
    idx = band_indices(freqs, high_hz=high_hz)
    shown = np.log10(np.maximum(power[idx, :], 1.0e-12))
    vmin, vmax = log_power_limits(power[idx, :])
    image = ax.pcolormesh(times, freqs[idx], shown, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yscale("log")
    ax.set_ylim(available_low_hz(freqs, high_hz=high_hz), high_hz)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.set_ylabel("frequency [Hz]")
    ax.grid(True, which="both", alpha=0.12)
    return image


def make_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.75))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    def add_box(
        key: str,
        text: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        face: str,
        edge: str,
        fontsize: float = 8.5,
    ) -> tuple[float, float, float, float]:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.15,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2.0,
            y + h / 2.0,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.15,
            color="#17202a",
            transform=ax.transAxes,
        )
        return x, y, w, h

    def add_tag(text: str, x: float, y: float, *, color: str) -> None:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=5.9,
            color=color,
            transform=ax.transAxes,
            bbox={
                "boxstyle": "round,pad=0.18,rounding_size=0.12",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.8,
            },
        )

    boxes = {
        "source": add_box(
            "source",
            "Simulator input\nstrain waveform\nor generic burst",
            0.08,
            0.82,
            0.26,
            0.10,
            face="#d9ecff",
            edge="#2f6f9f",
        ),
        "response": add_box(
            "response",
            "LISA response\nsix GW links",
            0.08,
            0.66,
            0.26,
            0.10,
            face="#d9ecff",
            edge="#2f6f9f",
        ),
        "noise": add_box(
            "noise",
            "Instrument model\nrealistic noisy links",
            0.08,
            0.50,
            0.26,
            0.10,
            face="#d9ecff",
            edge="#2f6f9f",
        ),
        "real": add_box(
            "real",
            "Real LISA input\ncalibrated one-way\nlink data",
            0.66,
            0.65,
            0.25,
            0.12,
            face="#dff3df",
            edge="#4f8a52",
        ),
        "links": add_box(
            "links",
            "Common link data\n$(12,23,31,13,32,21)$",
            0.39,
            0.515,
            0.25,
            0.10,
            face="#f5f1df",
            edge="#9b7b2f",
        ),
        "tdi": add_box(
            "tdi",
            "TDI observables\n$X,Y,Z$; $A,E,T$;\n$\\alpha,\\beta,\\gamma,\\zeta$",
            0.36,
            0.34,
            0.28,
            0.12,
            face="#eadff5",
            edge="#745196",
        ),
        "ae": add_box(
            "ae",
            "A/E WDM maps\nself-median\nwhitening",
            0.055,
            0.18,
            0.235,
            0.11,
            face="#fce3d8",
            edge="#b55d3b",
        ),
        "cluster": add_box(
            "cluster",
            "Bright-pixel\nevent clustering",
            0.385,
            0.18,
            0.23,
            0.11,
            face="#fff0cc",
            edge="#c9901f",
        ),
        "zeta": add_box(
            "zeta",
            "Sagnac/zeta WDM\npositive-excess\nratio",
            0.705,
            0.18,
            0.235,
            0.11,
            face="#fce3d8",
            edge="#b55d3b",
        ),
        "events": add_box(
            "events",
            "Event catalog\ndetection statistic\nand glitch classifier",
            0.36,
            0.025,
            0.26,
            0.10,
            face="#ffe6f0",
            edge="#a83d66",
        ),
    }

    def center(key: str, side: str) -> tuple[float, float]:
        x, y, w, h = boxes[key]
        if side == "top":
            return x + w / 2.0, y + h
        if side == "bottom":
            return x + w / 2.0, y
        if side == "left":
            return x, y + h / 2.0
        if side == "right":
            return x + w, y + h / 2.0
        raise ValueError(side)

    def add_arrow(start: tuple[float, float], stop: tuple[float, float], *, curve: str = "arc3,rad=0.0") -> None:
        ax.annotate(
            "",
            xy=stop,
            xytext=start,
            xycoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "->",
                "lw": 1.25,
                "color": "#283747",
                "shrinkA": 5,
                "shrinkB": 5,
                "connectionstyle": curve,
            },
        )
    add_arrow(center("source", "bottom"), center("response", "top"))
    add_arrow(center("response", "bottom"), center("noise", "top"))
    add_arrow(center("noise", "right"), center("links", "left"))
    add_arrow(center("real", "bottom"), center("links", "right"))
    add_arrow(center("links", "bottom"), center("tdi", "top"))
    add_arrow(center("tdi", "left"), center("ae", "top"))
    add_arrow(center("tdi", "right"), center("zeta", "top"))
    add_arrow(center("ae", "right"), center("cluster", "left"))
    add_arrow(center("cluster", "bottom"), center("events", "top"))
    add_arrow(center("zeta", "bottom"), center("events", "right"), curve="arc3,rad=-0.22")
    add_tag("phenomxpy", 0.27, 0.812, color="#2f6f9f")
    add_tag("LISA Orbits", 0.37, 0.727, color="#2f6f9f")
    add_tag("GW Response", 0.37, 0.692, color="#2f6f9f")
    add_tag("LISA Instrument", 0.205, 0.477, color="#2f6f9f")
    add_tag("PyTDI", 0.68, 0.402, color="#745196")
    add_tag("wdm-wavelet", 0.29, 0.306, color="#b55d3b")
    add_tag("wdm-wavelet", 0.835, 0.306, color="#b55d3b")
    ax.text(
        0.5,
        0.965,
        "cWB-space accepts either simulated link products or calibrated real one-way LISA links.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#283747",
        transform=ax.transAxes,
    )
    save_figure(fig, "fig01_pipeline_overview")


def validation_windows() -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    for row in read_csv_rows(TWO_WEEK_DIR / "mbhb_week1_zeta_wdm_summary.csv"):
        target = f(row, "response_peak_day")
        half_width = f(row, "local_half_window_days")
        windows.append(
            {
                "case_id": row["case_id"],
                "target_day": target,
                "start_day": max(0.0, target - half_width),
                "stop_day": min(14.0, target + half_width),
            }
        )
    for row in read_csv_rows(TWO_WEEK_DIR / "glitch_week2_zeta_wdm_summary.csv"):
        target = f(row, "injection_peak_day")
        windows.append(
            {
                "case_id": row["case_id"],
                "target_day": target,
                "start_day": max(0.0, f(row, "local_start_day")),
                "stop_day": min(14.0, f(row, "local_stop_day")),
            }
        )
    return windows


def load_zeta_fraction(stem: str) -> dict[str, np.ndarray]:
    data = np.load(TWO_WEEK_DIR / f"{stem}_zeta_wdm_maps.npz")
    abc = np.asarray(data["abc_positive_excess"], dtype=float)
    zeta = np.asarray(data["zeta_positive_excess"], dtype=float)
    total = abc + zeta
    fraction = np.divide(zeta, total, out=np.full_like(zeta, np.nan), where=total > 0.0)
    return {
        "times_day": np.asarray(data["times_day"], dtype=float),
        "freqs_hz": np.asarray(data["freqs_hz"], dtype=float),
        "fraction": fraction,
    }


def fill_week_fraction(full_times: np.ndarray, case: dict[str, np.ndarray], target: np.ndarray) -> None:
    case_times = np.asarray(case["times_day"], dtype=float)
    indices = np.searchsorted(case_times, full_times)
    indices = np.clip(indices, 1, len(case_times) - 1)
    left = indices - 1
    choose_left = np.abs(case_times[left] - full_times) <= np.abs(case_times[indices] - full_times)
    nearest = np.where(choose_left, left, indices)
    tolerance_days = 1.5 * np.nanmedian(np.diff(case_times))
    matched = np.abs(case_times[nearest] - full_times) <= tolerance_days
    target[:, matched] = np.asarray(case["fraction"], dtype=float)[:, nearest[matched]]


def combined_zeta_fraction(times: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    out = np.full((len(freqs), len(times)), np.nan)
    for stem in ("mbhb_week1", "glitch_week2"):
        case = load_zeta_fraction(stem)
        if len(case["freqs_hz"]) != len(freqs) or not np.allclose(case["freqs_hz"], freqs, rtol=0.0, atol=1.0e-15):
            raise ValueError(f"Zeta WDM frequency grid mismatch for {stem}.")
        fill_week_fraction(times, case, out)
    return out


def selected_validation_clusters(maps: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    config = SeededChirpEventConfig(
        low_hz=LOW_HZ,
        high_hz=WDM_HIGH_HZ,
        seed_quantile=0.99,
        grow_quantile=0.95,
        max_freq_jump_bins=4,
        same_time_connect_bins=1,
        min_pixels=4,
        min_time_bins=1,
        max_events=8,
    )
    score = np.asarray(maps["AE_power"], dtype=float)
    cluster_mask = np.zeros_like(score, dtype=bool)
    rho_map = np.full_like(score, np.nan)
    selected: list[dict[str, object]] = []
    for window in validation_windows():
        events, masks = find_seeded_chirp_events(
            maps,
            config=config,
            start_day=float(window["start_day"]),
            stop_day=float(window["stop_day"]),
            label=str(window["case_id"]),
        )
        if events.empty:
            continue
        time_error = np.abs(events["peak_time_day"].to_numpy(dtype=float) - float(window["target_day"]))
        case_mask = np.zeros_like(score, dtype=bool)
        rho_max = np.nan
        for _, event in events.iterrows():
            mask = np.asarray(masks[str(event["event_id"])], dtype=bool)
            rho = float(event["event_statistic"])
            case_mask |= mask
            rho_map[mask] = np.fmax(np.nan_to_num(rho_map[mask], nan=-np.inf), rho)
            rho_max = float(np.nanmax([rho_max, rho]))
        cluster_mask |= case_mask
        selected.append(
            {
                "case_id": str(window["case_id"]),
                "event_id": str(events.iloc[0]["event_id"]),
                "rho_event": rho_max,
                "target_time_error_days": float(np.nanmin(time_error)),
            }
        )
    return cluster_mask, rho_map, selected


def make_primary_validation_summary() -> None:
    data = np.load(TWO_WEEK_DIR / "seeded_chirp_event_test_self_ae_maps.npz")
    times = np.asarray(data["times_day"], dtype=float)
    freqs = np.asarray(data["freqs_hz"], dtype=float)
    maps = {
        "times_day": times,
        "freqs_hz": freqs,
        "A_power": np.asarray(data["A_power"], dtype=float),
        "E_power": np.asarray(data["E_power"], dtype=float),
        "AE_power": np.asarray(data["AE_power"], dtype=float),
        "A_coeff": np.asarray(data["A_coeff"], dtype=np.complex128),
        "E_coeff": np.asarray(data["E_coeff"], dtype=np.complex128),
        "wdm_df_hz": np.asarray(float(np.nanmedian(np.diff(freqs)))),
        "wdm_dt_s": np.asarray(float(np.nanmedian(np.diff(times))) * 86_400.0),
    }
    cluster_mask, rho_map, _ = selected_validation_clusters(maps)
    zeta_fraction = combined_zeta_fraction(times, freqs)
    idx = band_indices(freqs)

    ae_values = maps["AE_power"][idx, :]
    ae_vmin, ae_vmax = positive_log_limits(ae_values)
    rho_values = np.ma.masked_invalid(rho_map[idx, :])
    rho_finite = np.asarray(rho_values.compressed(), dtype=float)
    rho_vmin = float(np.nanmin(rho_finite)) if rho_finite.size else 1.0
    rho_vmax = float(np.nanmax(rho_finite)) if rho_finite.size else 10.0
    zeta_values = np.ma.masked_where(~cluster_mask[idx, :], zeta_fraction[idx, :])

    y_low = available_low_hz(freqs)
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 4.75), sharex=True, constrained_layout=True, gridspec_kw={"hspace": 0.015})
    panels = [
        ("Self-normalized A/E WDM power", ae_values, "cividis", LogNorm(vmin=ae_vmin, vmax=ae_vmax), r"$P_A+P_E$"),
        ("Detection statistic on A/E clusters", rho_values, "magma", LogNorm(vmin=max(rho_vmin, 1.0e-6), vmax=rho_vmax), r"$\rho_{\rm event}$"),
        ("Zeta fraction on A/E clusters", zeta_values, "viridis", None, r"$E_\zeta/(E_{\alpha\beta\gamma}+E_\zeta)$"),
    ]
    for ax, (title, values, cmap_name, norm, label) in zip(axes, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("white")
        kwargs = {"shading": "auto", "cmap": cmap}
        if norm is None:
            kwargs.update({"vmin": 0.0, "vmax": 1.0})
        else:
            kwargs["norm"] = norm
        image = ax.pcolormesh(times, freqs[idx], values, **kwargs)
        ax.set_yscale("log")
        ax.set_ylim(y_low, WDM_HIGH_HZ)
        ax.set_xlim(0.0, 14.0)
        ax.set_title(title, fontsize=10, pad=2)
        ax.set_ylabel("frequency [Hz]", fontsize=9)
        ax.tick_params(labelsize=8)
        colorbar = fig.colorbar(image, ax=ax, pad=0.01, shrink=0.88)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label(label, fontsize=9)
    axes[-1].set_xlabel("time from start [days]", fontsize=10)
    save_figure(fig, "fig03_primary_validation_summary")


def make_mbhb_snr_event_statistic() -> None:
    rows = read_csv_rows(SNR_STUDY_DIR / "mbhb_snr_event_statistic.csv")
    snr = np.array([f(row, "optimal_snr_ae") for row in rows])
    rho_event = np.array([f(row, "rho_event") for row in rows])
    scale = np.array([f(row, "amplitude_scale") for row in rows])
    rho_noise = f(rows[0], "rho_noise")

    order = np.argsort(snr)
    snr = snr[order]
    rho_event = rho_event[order]
    scale = scale[order]

    fig, ax = plt.subplots(figsize=(6.5, 3.7), constrained_layout=True)
    event_line = ax.plot(snr, rho_event, color="#174a7c", lw=1.45, marker="o", ms=5.2, label=r"$\rho_{\rm event}$")
    ax2 = ax.twinx()
    for xval, yval, sval in zip(snr, rho_event, scale):
        ax.text(xval + 4.0, yval + 1.0, f"{sval:.2f}", fontsize=7.0, color="#174a7c")
    ax.set_xlabel(r"optimal A/E SNR, $\rho_{\rm opt}$")
    ax.set_ylabel(r"$\rho_{\rm event}$")
    ax2.set_ylabel(r"$R_\rho=\rho_{\rm event}/\rho_{\rm noise}$")
    y_min = max(0.0, float(np.nanmin(rho_event)) * 0.88)
    y_max = float(np.nanmax(rho_event)) * 1.10
    ax.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min / rho_noise, y_max / rho_noise)
    ax.set_title("Controlled MBHB signal-strength check")
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(event_line, [event_line[0].get_label()], frameon=False, fontsize=8, loc="upper left")
    save_figure(fig, "fig07_mbhb_snr_rho_event")


def make_ae_coherence_mbhb_noise() -> None:
    mbhb_rows = read_csv_rows(SNR_STUDY_DIR / "mbhb_snr_event_statistic.csv")
    noise_rows = read_csv_rows(SNR_STUDY_DIR / "random_noise_events.csv")

    mbhb_rho = np.array([f(row, "rho_event") for row in mbhb_rows])
    mbhb_coh = np.array([f(row, "event_ae_complex_coherence") for row in mbhb_rows])
    mbhb_scale = np.array([f(row, "amplitude_scale") for row in mbhb_rows])
    noise_rho = np.array([f(row, "rho_event") for row in noise_rows])
    noise_coh = np.array([f(row, "event_ae_complex_coherence") for row in noise_rows])

    fig, ax = plt.subplots(figsize=(6.5, 3.7), constrained_layout=True)
    ax.scatter(noise_rho, noise_coh, marker="x", s=48, lw=1.4, color="#8a8f98", label="noise candidates")
    ax.scatter(mbhb_rho, mbhb_coh, marker="o", s=48, color="#174a7c", edgecolor="#0b2239", linewidth=0.35, label="scaled MBHB")
    for xval, yval, sval in zip(mbhb_rho, mbhb_coh, mbhb_scale):
        ax.text(xval + 1.5, yval + 0.006, f"{sval:.2f}", fontsize=7.0, color="#174a7c")
    ax.set_xlabel(r"$\rho_{\rm event}$")
    ax.set_ylabel(r"$C_{\rm AE}$")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("A/E coherence of MBHB and noise events")
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save_figure(fig, "fig08_ae_coherence_mbhb_noise")


def make_generic_burst_response() -> None:
    maps = np.load(GENERIC_DIR / "response_burst_wdm_maps.npz")
    masks = np.load(GENERIC_DIR / "response_burst_event_masks.npz")
    events = read_csv_rows(GENERIC_DIR / "response_burst_candidate_events.csv")
    times = np.asarray(maps["times_day"], dtype=float)
    freqs = np.asarray(maps["freqs_hz"], dtype=float)
    idx = band_indices(freqs)

    cluster_mask = np.zeros_like(np.asarray(maps["AE_power"], dtype=float), dtype=bool)
    rho_map = np.full_like(np.asarray(maps["AE_power"], dtype=float), np.nan)
    for row in events:
        if not row.get("matched_case_id", ""):
            continue
        event_id = row["event_id"]
        if event_id not in masks.files:
            continue
        mask = np.asarray(masks[event_id], dtype=bool)
        rho = f(row, "event_statistic")
        cluster_mask |= mask
        rho_map[mask] = rho

    ae_values = np.asarray(maps["AE_power"], dtype=float)[idx, :]
    ae_vmin, ae_vmax = positive_log_limits(ae_values)
    rho_values = np.ma.masked_invalid(rho_map[idx, :])
    rho_finite = np.asarray(rho_values.compressed(), dtype=float)
    rho_vmin = float(np.nanmin(rho_finite)) if rho_finite.size else 1.0
    rho_vmax = float(np.nanmax(rho_finite)) if rho_finite.size else 10.0
    zeta_values = np.ma.masked_where(~cluster_mask[idx, :], np.asarray(maps["zeta_fraction"], dtype=float)[idx, :])

    y_low = available_low_hz(freqs)
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 4.75), sharex=True, constrained_layout=True, gridspec_kw={"hspace": 0.015})
    panels = [
        ("Self-normalized A/E WDM power", ae_values, "cividis", LogNorm(vmin=ae_vmin, vmax=ae_vmax), r"$P_A+P_E$"),
        ("Detection statistic on A/E clusters", rho_values, "magma", LogNorm(vmin=max(rho_vmin, 1.0e-6), vmax=rho_vmax), r"$\rho_{\rm event}$"),
        ("Zeta fraction on A/E clusters", zeta_values, "viridis", None, r"$E_\zeta/(E_{\alpha\beta\gamma}+E_\zeta)$"),
    ]
    for ax, (title, values, cmap_name, norm, label) in zip(axes, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("white")
        kwargs = {"shading": "auto", "cmap": cmap}
        if norm is None:
            kwargs.update({"vmin": 0.0, "vmax": 1.0})
        else:
            kwargs["norm"] = norm
        image = ax.pcolormesh(times, freqs[idx], values, **kwargs)
        ax.set_yscale("log")
        ax.set_ylim(y_low, WDM_HIGH_HZ)
        ax.set_xlim(0.0, 14.0)
        ax.set_title(title, fontsize=10, pad=2)
        ax.set_ylabel("frequency [Hz]", fontsize=9)
        ax.tick_params(labelsize=8)
        colorbar = fig.colorbar(image, ax=ax, pad=0.01, shrink=0.88)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label(label, fontsize=9)
    axes[-1].set_xlabel("time from start [days]", fontsize=10)
    save_figure(fig, "fig05_generic_burst_summary")


def make_sangria_demo() -> None:
    events = read_csv_rows(SANGRIA_FULL_DIR / "sangria_full_mbhb_candidate_events.csv")
    maps = np.load(SANGRIA_FOCUSED_DIR / "focused_sangria_candidate_wdm_maps.npz")
    masks = np.load(SANGRIA_FOCUSED_DIR / "focused_sangria_candidate_event_masks.npz")

    top = sorted(events, key=lambda row: f(row, "global_rank"))[:10]
    fig, axes = plt.subplots(1, 2, figsize=(6.95, 3.35), constrained_layout=True)
    scatter = axes[0].scatter(
        [f(row, "peak_time_day") for row in top],
        [f(row, "peak_freq_hz") for row in top],
        c=[f(row, "event_statistic") for row in top],
        s=[32.0 + 0.16 * f(row, "event_statistic") for row in top],
        cmap="plasma",
        edgecolor="#222222",
        linewidth=0.35,
    )
    for row in top[:3]:
        axes[0].text(f(row, "peak_time_day") + 2.0, f(row, "peak_freq_hz"), str(int(f(row, "global_rank"))), fontsize=7)
    axes[0].set_yscale("log")
    axes[0].set_ylim(available_low_hz(maps["freqs_hz"]), WDM_HIGH_HZ)
    axes[0].set_xlabel("time [days]")
    axes[0].set_ylabel("peak frequency [Hz]")
    axes[0].set_title("Sangria A/E candidates")
    axes[0].grid(True, which="both", alpha=0.14)
    fig.colorbar(scatter, ax=axes[0], pad=0.01, label=r"$\rho_{\rm event}$")

    focused_times = maps["times_day"] + float(maps["t_abs_start_day"])
    image = plot_wdm(
        axes[1],
        focused_times,
        maps["freqs_hz"],
        maps["AE_power"],
        title="Focused candidate map",
        high_hz=WDM_HIGH_HZ,
    )
    for key in masks.files:
        if key.endswith("event01"):
            contour = masks[key].astype(float)
            idx = band_indices(maps["freqs_hz"])
            axes[1].contour(focused_times, maps["freqs_hz"][idx], contour[idx, :], levels=[0.5], colors="cyan", linewidths=0.8)
            break
    axes[1].set_xlabel("time [days]")
    fig.colorbar(image, ax=axes[1], pad=0.01, label=r"$\log_{10}$ A+E power")
    save_figure(fig, "fig06_sangria_demo")


def main() -> None:
    make_pipeline_overview()
    make_primary_validation_summary()
    make_mbhb_snr_event_statistic()
    make_ae_coherence_mbhb_noise()
    make_generic_burst_response()
    make_sangria_demo()


if __name__ == "__main__":
    main()
