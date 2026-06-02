"""Cluster evaluation and interpretation summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clustering import load_embeddings


def _silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    clustered = labels >= 0
    unique = np.unique(labels[clustered])
    if len(unique) < 2 or int(np.sum(clustered)) <= len(unique):
        return None
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None
    return float(silhouette_score(StandardScaler().fit_transform(embeddings[clustered]), labels[clustered]))


def evaluate_clusters(
    *,
    embeddings_npz: str | Path,
    labels_csv: str | Path,
    metadata_csv: str | Path,
    metrics_json: str | Path,
    summary_csv: str | Path,
) -> tuple[Path, Path]:
    """Write cluster quality metrics and physical-parameter summaries."""

    embeddings, source_ids = load_embeddings(embeddings_npz)
    labels = pd.read_csv(labels_csv)
    metadata = pd.read_csv(metadata_csv)
    joined = labels.merge(metadata, on="source_id", how="left")
    label_array = joined["cluster"].to_numpy(dtype=int)

    n_noise = int(np.sum(label_array < 0))
    clusters = sorted(int(label) for label in np.unique(label_array) if label >= 0)
    metrics = {
        "n_samples": int(len(label_array)),
        "n_clusters": int(len(clusters)),
        "n_noise": n_noise,
        "noise_fraction": float(n_noise / max(1, len(label_array))),
        "silhouette_non_noise": _silhouette_score(embeddings, label_array),
    }

    summary_rows = []
    numeric_columns = [
        column
        for column in ("total_mass", "m1", "m2", "q", "eta", "distance_mpc", "signal_peak_lisa")
        if column in joined.columns
    ]
    for cluster_id, group in joined.groupby("cluster", dropna=False):
        row = {"cluster": int(cluster_id), "n": int(len(group))}
        for column in numeric_columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_median"] = float(np.nanmedian(values))
            row[f"{column}_min"] = float(np.nanmin(values))
            row[f"{column}_max"] = float(np.nanmax(values))
        summary_rows.append(row)

    metrics_path = Path(metrics_json)
    summary_path = Path(summary_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).sort_values("cluster").to_csv(summary_path, index=False)
    return metrics_path, summary_path
