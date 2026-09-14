"""SMBHB parameter sampling utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SMBHBParameterRange


def mass_ratio_from_eta(eta: float) -> float:
    """Return q=m2/m1<=1 for a symmetric mass ratio eta."""

    eta = float(eta)
    if np.isclose(eta, 0.25):
        return 1.0
    if not (0.0 < eta < 0.25):
        raise ValueError("eta must be in (0, 0.25].")
    discriminant = np.sqrt(max(0.0, 1.0 - 4.0 * eta))
    return float((1.0 - 2.0 * eta - discriminant) / (2.0 * eta))


def component_masses(total_mass: float, eta: float) -> tuple[float, float, float]:
    """Return m1, m2, q from total mass and eta."""

    q = mass_ratio_from_eta(eta)
    m1 = total_mass / (1.0 + q)
    m2 = total_mass - m1
    return float(m1), float(m2), float(q)


def sample_smbhb_catalog(
    n_sources: int,
    ranges: SMBHBParameterRange,
    *,
    seed: int = 12345,
) -> pd.DataFrame:
    """Sample a reproducible SMBHB parameter catalog."""

    if n_sources <= 0:
        raise ValueError("n_sources must be positive.")
    ranges.validate()
    eta_min, eta_max = ranges.eta_bounds
    rng = np.random.default_rng(seed)

    log_m = rng.uniform(np.log10(ranges.total_mass_min), np.log10(ranges.total_mass_max), size=n_sources)
    total_mass = 10.0**log_m
    eta = rng.uniform(eta_min, eta_max, size=n_sources)
    log_d = rng.uniform(np.log10(ranges.distance_mpc_min), np.log10(ranges.distance_mpc_max), size=n_sources)
    distance_mpc = 10.0**log_d

    rows = []
    for idx in range(n_sources):
        m1, m2, q = component_masses(total_mass[idx], eta[idx])
        rows.append(
            {
                "source_id": f"smbhb_{idx:05d}",
                "total_mass": float(total_mass[idx]),
                "m1": m1,
                "m2": m2,
                "q": q,
                "eta": float(eta[idx]),
                "distance_mpc": float(distance_mpc[idx]),
                "s1": float(rng.uniform(ranges.spin1_min, ranges.spin1_max)),
                "s2": float(rng.uniform(ranges.spin2_min, ranges.spin2_max)),
                "f_min_hz": float(ranges.f_min_hz),
                "iota": float(rng.uniform(ranges.iota_min, ranges.iota_max)),
                "phi_ref": float(rng.uniform(ranges.phi_ref_min, ranges.phi_ref_max)),
                "ra": float(rng.uniform(ranges.ra_min, ranges.ra_max)),
                "dec": float(rng.uniform(ranges.dec_min, ranges.dec_max)),
            }
        )
    return pd.DataFrame(rows)


def write_catalog(catalog: pd.DataFrame, output_csv: str | Path) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False)
    return output_path


def read_catalog(path: str | Path) -> pd.DataFrame:
    catalog = pd.read_csv(path)
    required = {"source_id", "total_mass", "eta", "distance_mpc", "s1", "s2", "f_min_hz", "iota", "phi_ref", "ra", "dec"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")
    return catalog
