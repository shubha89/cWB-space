"""Lightweight checks for the cluster package."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PSDConfig, SMBHBParameterRange, WDMMapConfig
from .parameters import component_masses, sample_smbhb_catalog
from .psd import whiten_time_series
from .wdm_pixels import _normalize_patch, load_dataset


def run_smoke_checks(*, with_ml: bool = False) -> None:
    ranges = SMBHBParameterRange(total_mass_min=1.0e4, total_mass_max=1.0e8, eta_min=1.5, eta_max=2.5)
    catalog = sample_smbhb_catalog(8, ranges, seed=7)
    if catalog["total_mass"].min() < 1.0e4 or catalog["total_mass"].max() > 1.0e8:
        raise AssertionError("Sampled total masses are outside the requested bounds.")
    if catalog["eta"].min() < 0.15 or catalog["eta"].max() > 0.25:
        raise AssertionError("Sampled eta values are outside [0.15, 0.25].")
    m1, m2, q = component_masses(1.0e5, 0.25)
    if not np.isclose(m1, m2) or not np.isclose(q, 1.0):
        raise AssertionError("Equal-mass eta=0.25 conversion failed.")

    dt = 0.25
    t = np.arange(2048) * dt
    x = np.sin(2.0 * np.pi * 0.03 * t) + 0.1 * np.sin(2.0 * np.pi * 0.1 * t)
    freqs = np.linspace(0.0, 1.0 / (2.0 * dt), 128)
    psd = np.ones_like(freqs)
    white = whiten_time_series(x, dt, freqs, psd, min_psd=PSDConfig().min_psd)
    if not np.all(np.isfinite(white)) or not np.isclose(np.std(white), 1.0, atol=1.0e-6):
        raise AssertionError("Whitening did not produce finite unit-scale output.")

    patch = _normalize_patch(np.abs(np.outer(np.linspace(1, 2, 20), np.linspace(1, 3, 30))), config=WDMMapConfig())
    if patch.shape != (20, 30) or patch.dtype != np.float32:
        raise AssertionError("Patch normalization returned an unexpected shape or dtype.")

    with tempfile.TemporaryDirectory(prefix="cluster_smoke_") as tmp:
        path = Path(tmp) / "dataset.npz"
        source_ids = np.asarray(["a", "b", "c"])
        samples = np.stack([np.full((2, 32, 32), idx, dtype=np.float32) for idx in range(3)], axis=0)
        np.savez_compressed(path, x=samples, source_id=source_ids)
        loaded, loaded_ids = load_dataset(path)
        if loaded.shape != (3, 2, 32, 32) or list(loaded_ids.astype(str)) != ["a", "b", "c"]:
            raise AssertionError("Dataset loader smoke check failed.")

        if with_ml:
            _run_optional_ml_check(Path(tmp), samples, source_ids)

    print("cluster smoke checks passed")


def _run_optional_ml_check(tmp: Path, samples: np.ndarray, source_ids: np.ndarray) -> None:
    try:
        from .autoencoder import train_autoencoder
        from .clustering import run_hdbscan
    except ModuleNotFoundError as exc:
        print(f"ML smoke check skipped: {exc}")
        return

    dataset_path = tmp / "dataset.npz"
    embeddings_path = tmp / "embeddings.npz"
    labels_path = tmp / "labels.csv"
    np.savez_compressed(dataset_path, x=samples, source_id=source_ids)
    try:
        train_autoencoder(
            dataset_path,
            latent_dim=4,
            epochs=1,
            batch_size=2,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            validation_fraction=0.0,
            seed=5,
            model_path=tmp / "model.pt",
            embeddings_npz=embeddings_path,
            history_csv=tmp / "history.csv",
        )
        run_hdbscan(embeddings_path, min_cluster_size=2, min_samples=None, metric="euclidean", labels_csv=labels_path)
    except ModuleNotFoundError as exc:
        print(f"ML smoke check skipped: {exc}")
        return
    labels = pd.read_csv(labels_path)
    if "cluster" not in labels.columns:
        raise AssertionError("HDBSCAN output missing cluster column.")


if __name__ == "__main__":
    run_smoke_checks()
