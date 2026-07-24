"""Run HDBSCAN on coronagraphic kappa feature tables.

This script is intentionally separate from the kappa scan. It consumes
``kappa_features.csv`` written by ``choronagraphic-cluster/run_one_smbhb.py``
and rewrites ``pixel_clusters.csv`` without recomputing WDM or kappa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHORO_DIR = PROJECT_ROOT / "choronagraphic-cluster"
if str(CHORO_DIR) not in sys.path:
    sys.path.insert(0, str(CHORO_DIR))

import run_one_smbhb as proto  # noqa: E402


def _require_hdbscan_dependencies():
    try:
        import hdbscan
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        raise ModuleNotFoundError(
            "Coronagraphic HDBSCAN requires hdbscan and scikit-learn in the lisa-tf environment."
        ) from exc
    return hdbscan, StandardScaler


def _attach_truth_separation(features: pd.DataFrame, summary: dict) -> pd.DataFrame:
    source_ra = summary.get("source_ra")
    source_dec = summary.get("source_dec")
    if source_ra is None or source_dec is None:
        return features
    out = features.copy()
    out["true_sky_sep_deg"] = proto.angular_separation_deg(
        float(source_ra),
        float(source_dec),
        out["best_ra"].to_numpy(dtype=float),
        out["best_dec"].to_numpy(dtype=float),
    )
    return out


def cluster_features_file(
    features_path: Path,
    output_path: Path,
    *,
    min_cluster_size: int,
    min_samples: int | None,
    summary: dict,
) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Missing kappa feature table: {features_path}")

    hdbscan, StandardScaler = _require_hdbscan_dependencies()
    features = pd.read_csv(features_path)
    clustered = proto.cluster_pixel_features(
        features,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        hdbscan=hdbscan,
        StandardScaler=StandardScaler,
    )
    clustered = _attach_truth_separation(clustered, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clustered.to_csv(output_path, index=False)
    return clustered


def _layer_dirs(run_dir: Path, summary: dict) -> list[Path]:
    if summary.get("mode") == "multi_resolution":
        return [run_dir / f"wdm_m_{int(wdm_m)}" for wdm_m in summary.get("wdm_ms", [])]
    return [run_dir]


def run_hdbscan(
    run_dir: Path,
    *,
    min_cluster_size: int,
    min_samples: int | None,
) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing coronagraphic summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    all_clustered: list[pd.DataFrame] = []
    layer_outputs: list[str] = []
    for layer_dir in _layer_dirs(run_dir, summary):
        clustered = cluster_features_file(
            layer_dir / "kappa_features.csv",
            layer_dir / "pixel_clusters.csv",
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            summary=summary,
        )
        all_clustered.append(clustered)
        layer_outputs.append(str(layer_dir / "pixel_clusters.csv"))

        layer_summary_path = layer_dir / "summary.json"
        if layer_summary_path.exists():
            layer_summary = json.loads(layer_summary_path.read_text(encoding="utf-8"))
            counts = clustered["cluster"].value_counts().sort_index().to_dict()
            layer_summary["run_hdbscan"] = True
            layer_summary["cluster_counts"] = {str(k): int(v) for k, v in counts.items()}
            layer_summary["median_true_sky_sep_deg"] = (
                float(np.nanmedian(clustered["true_sky_sep_deg"]))
                if "true_sky_sep_deg" in clustered
                else None
            )
            layer_summary_path.write_text(json.dumps(layer_summary, indent=2), encoding="utf-8")

    if summary.get("mode") == "multi_resolution":
        combined = pd.concat(all_clustered, ignore_index=True)
        combined_path = run_dir / "pixel_clusters_all_resolutions.csv"
        combined.to_csv(combined_path, index=False)
        layer_outputs.append(str(combined_path))

    counts = pd.concat(all_clustered, ignore_index=True)["cluster"].value_counts().sort_index().to_dict()
    summary["run_hdbscan"] = True
    summary["hdbscan_min_cluster_size"] = int(min_cluster_size)
    summary["hdbscan_min_samples"] = None if min_samples is None else int(min_samples)
    summary["cluster_counts"] = {str(k): int(v) for k, v in counts.items()}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "outputs": layer_outputs,
        "cluster_counts": summary["cluster_counts"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HDBSCAN on existing coronagraphic kappa features.")
    parser.add_argument("--run-dir", required=True, help="Coronagraphic run directory containing kappa_features.csv files.")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_hdbscan(
        Path(args.run_dir),
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
