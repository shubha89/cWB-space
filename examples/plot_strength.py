"""Replot paper Figures 2 and 3 from the stored numerical results."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SNR_STUDY_DIR = ROOT / "data" / "strength"
FIG_DIR = ROOT / "outputs"

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--data-dir", type=Path, default=SNR_STUDY_DIR, help="Directory of original strength-study CSV products")
    args = parser.parse_args()
    FIG_DIR = args.output_dir
    SNR_STUDY_DIR = args.data_dir
    make_mbhb_snr_event_statistic()
    make_ae_coherence_mbhb_noise()
