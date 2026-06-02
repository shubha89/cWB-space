"""HDBSCAN clustering for autoencoder embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _require_clustering_deps():
    try:
        import hdbscan
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on caller environment.
        raise ModuleNotFoundError(
            "HDBSCAN clustering requires hdbscan and scikit-learn. "
            "Update the lisa-tf environment from env.yml."
        ) from exc
    return hdbscan, StandardScaler


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "embeddings" not in data:
        raise ValueError(f"Embeddings NPZ {path} does not contain array 'embeddings'.")
    source_ids = np.asarray(data["source_id"]) if "source_id" in data else np.arange(len(data["embeddings"])).astype(str)
    return np.asarray(data["embeddings"], dtype=float), source_ids


def run_hdbscan(
    embeddings_npz: str | Path,
    *,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
    labels_csv: str | Path,
) -> Path:
    """Cluster latent embeddings with HDBSCAN and save labels."""

    hdbscan, StandardScaler = _require_clustering_deps()
    embeddings, source_ids = load_embeddings(embeddings_npz)
    scaled = StandardScaler().fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
    labels = clusterer.fit_predict(scaled)
    probabilities = getattr(clusterer, "probabilities_", np.full(len(labels), np.nan))
    out = pd.DataFrame({"source_id": source_ids.astype(str), "cluster": labels.astype(int), "cluster_probability": probabilities})
    labels_path = Path(labels_csv)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(labels_path, index=False)
    return labels_path
