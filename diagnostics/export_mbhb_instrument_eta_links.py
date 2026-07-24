from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pytdi import Data
from pytdi.intervar import compute_etas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lisa_link_simulation import LINK_LABELS, LisaLinkSimulationPaths, apply_frequency_cut  # noqa: E402


WORK_DIR = PROJECT_ROOT / "diagnostics" / "mbhb_sixlinks"
OUTPUT_DIR = WORK_DIR / "outputs"
ETA_LINKS_CSV = OUTPUT_DIR / "six_eta_links_noisy.csv"
ETA_REPORT_JSON = OUTPUT_DIR / "six_eta_links_noisy_report.json"

LOW_FREQ_HZ = 1e-5
HIGH_FREQ_HZ = 1e-1
INTERPOLATION_ORDER = 31
DELAY_ORDER = 5


def stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "absmax": float(np.nanmax(np.abs(arr))),
    }


def main() -> None:
    paths = LisaLinkSimulationPaths.from_work_dir(WORK_DIR)
    timing = pd.read_csv(paths.gw_links_csv, usecols=["sample", "t_waveform", "t_lisa", "is_science"])
    is_science = timing["is_science"].astype(bool).to_numpy()
    burn_in_size = int(np.argmax(is_science)) if not is_science[0] else 0
    science_idx = np.flatnonzero(is_science)
    dt = float(np.median(np.diff(timing["t_waveform"].to_numpy(dtype=float))))

    data = Data.from_instrument(str(paths.instrument_h5), signals="fluctuations", skipped=0)
    etas = compute_etas(data, order=INTERPOLATION_ORDER, delay_order=DELAY_ORDER, unit="frequency")

    n = min([len(arr) for arr in etas.values()] + [len(timing)])
    science_idx = science_idx[science_idx < n]

    df = pd.DataFrame(
        {
            "sample": np.arange(len(science_idx)),
            "instrument_sample": science_idx,
            "t_waveform": timing["t_waveform"].to_numpy(dtype=float)[science_idx],
            "t_lisa": timing["t_lisa"].to_numpy(dtype=float)[science_idx],
        }
    )
    report: dict[str, object] = {
        "source_instrument": str(paths.instrument_h5),
        "output_csv": str(ETA_LINKS_CSV),
        "low_freq_hz": LOW_FREQ_HZ,
        "high_freq_hz": HIGH_FREQ_HZ,
        "burn_in_size": burn_in_size,
        "samples": int(len(science_idx)),
        "columns": [],
        "per_eta": {},
    }

    for link in LINK_LABELS:
        raw = np.asarray(etas[f"eta_{link}"], dtype=float)[science_idx]
        bandpassed = apply_frequency_cut(raw, dt, LOW_FREQ_HZ, HIGH_FREQ_HZ)
        col = f"eta_{link}"
        df[col] = bandpassed
        report["columns"].append(col)
        report["per_eta"][col] = stats(bandpassed)

    stds = np.asarray([report["per_eta"][f"eta_{link}"]["std"] for link in LINK_LABELS], dtype=float)
    report["std_balance"] = {
        "median_std_hz": float(np.median(stds)),
        "min_std_hz": float(np.min(stds)),
        "max_std_hz": float(np.max(stds)),
        "max_over_min": float(np.max(stds) / np.min(stds)) if np.min(stds) > 0.0 else np.nan,
    }

    ETA_LINKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ETA_LINKS_CSV, index=False)
    ETA_REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report["std_balance"], indent=2))
    print(f"eta links = {ETA_LINKS_CSV}")
    print(f"report    = {ETA_REPORT_JSON}")


if __name__ == "__main__":
    main()
