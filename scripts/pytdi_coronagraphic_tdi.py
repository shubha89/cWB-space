"""Coronagraphic TDI built from PyTDI Sagnac channels.

This module implements the frequency-domain construction in
Costa Barroso et al. 2024, arXiv:2406.00190, using PyTDI-computed
Sagnac channels as the data vector. The paper's derivation is for
first-generation Sagnac generators, so the default input channels are
``alpha1``, ``beta1``, and ``gamma1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd


C_M_PER_S = 299_792_458.0
SAGNAC_NAMES = ("alpha", "beta", "gamma")
LINKS = ("12", "23", "31", "13", "32", "21")


@dataclass(frozen=True)
class StaticLISAGeometry:
    """Static geometry snapshot in light-seconds."""

    x: Mapping[int, np.ndarray]
    L: Mapping[str, float]
    n: Mapping[str, np.ndarray]
    t_lisa_reference: float
    source: str


def _as_1d_float(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    return arr


def _as_1d_complex(values: Sequence[complex] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    return arr.astype(np.complex128, copy=False)


def _time_to_freq_plus(x_time: np.ndarray) -> np.ndarray:
    """Fourier transform with positive-sign convention used by the coefficients."""

    return np.fft.ifft(x_time) * x_time.size


def _freq_to_time_plus(x_freq: np.ndarray) -> np.ndarray:
    """Inverse transform for ``_time_to_freq_plus``."""

    return np.fft.fft(x_freq) / x_freq.size


def geometry_from_orbit_file(
    orbit_h5: str | Path,
    *,
    t_lisa_reference: float | None = None,
    center: bool = True,
    c_m_per_s: float = C_M_PER_S,
) -> StaticLISAGeometry:
    """Freeze the simulated LISA orbit into a static geometry snapshot.

    The orbit file stores spacecraft positions in meters in ``tcb/x`` with
    shape ``(time, spacecraft, coordinate)``. Internally this function converts
    them to light-seconds. If ``center`` is true, the constellation barycenter is
    subtracted before building the response coefficients.
    """

    path = Path(orbit_h5)
    with h5py.File(path, "r") as hdf:
        t0 = float(hdf.attrs["t0"])
        dt = float(hdf.attrs["dt"])
        x_samples = np.asarray(hdf["tcb/x"], dtype=float)

    times = t0 + dt * np.arange(x_samples.shape[0], dtype=float)
    if t_lisa_reference is None:
        t_lisa_reference = float(times[len(times) // 2])
    if t_lisa_reference < times[0] or t_lisa_reference > times[-1]:
        raise ValueError("t_lisa_reference lies outside the orbit file time range.")

    positions_m = np.empty((3, 3), dtype=float)
    for sc in range(3):
        for coord in range(3):
            positions_m[sc, coord] = np.interp(t_lisa_reference, times, x_samples[:, sc, coord])
    if center:
        positions_m = positions_m - np.mean(positions_m, axis=0, keepdims=True)

    positions = {idx + 1: positions_m[idx] / c_m_per_s for idx in range(3)}
    return geometry_from_positions(
        positions[1],
        positions[2],
        positions[3],
        t_lisa_reference=float(t_lisa_reference),
        source=str(path),
    )


def geometry_from_positions(
    x1: Sequence[float],
    x2: Sequence[float],
    x3: Sequence[float],
    *,
    t_lisa_reference: float = np.nan,
    source: str = "positions",
) -> StaticLISAGeometry:
    """Build static LISA geometry from spacecraft positions in light-seconds."""

    x = {
        1: np.asarray(x1, dtype=float),
        2: np.asarray(x2, dtype=float),
        3: np.asarray(x3, dtype=float),
    }
    for idx, vec in x.items():
        if vec.shape != (3,):
            raise ValueError(f"x{idx} must be a 3-vector.")

    lengths: dict[str, float] = {}
    directions: dict[str, np.ndarray] = {}
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            if i == j:
                continue
            link = f"{i}{j}"
            delta = x[i] - x[j]
            length = float(np.linalg.norm(delta))
            if length <= 0.0:
                raise ValueError(f"Degenerate geometry for link {link}.")
            lengths[link] = length
            directions[link] = delta / length
    return StaticLISAGeometry(x=x, L=lengths, n=directions, t_lisa_reference=t_lisa_reference, source=source)


def polarization_basis(beta: float, lamb: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return propagation vector and polarization basis for sky position.

    ``beta`` is latitude and ``lamb`` is longitude in the same frame as the
    spacecraft positions. The returned ``k`` points from source to detector.
    """

    cb = np.cos(beta)
    sb = np.sin(beta)
    cl = np.cos(lamb)
    sl = np.sin(lamb)
    k = np.array([-cb * cl, -cb * sl, -sb], dtype=float)
    u = np.array([sl, -cl, 0.0], dtype=float)
    v = np.array([-sb * cl, -sb * sl, cb], dtype=float)
    return k, u, v


def sky_to_unit_source_vector(beta: float, lamb: float) -> np.ndarray:
    """Unit vector from observer to source for latitude/longitude coordinates."""

    return np.array([np.cos(beta) * np.cos(lamb), np.cos(beta) * np.sin(lamb), np.sin(beta)], dtype=float)


def unit_source_vector_to_sky(vector: Sequence[float]) -> tuple[float, float]:
    """Convert an observer-to-source unit vector to latitude and longitude."""

    vec = np.asarray(vector, dtype=float)
    if vec.shape != (3,):
        raise ValueError("vector must be a 3-vector.")
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError("vector norm must be positive.")
    vec = vec / norm
    beta = float(np.arcsin(np.clip(vec[2], -1.0, 1.0)))
    lamb = float(np.mod(np.arctan2(vec[1], vec[0]), 2.0 * np.pi))
    return beta, lamb


def antenna_patterns(beta: float, lamb: float, geometry: StaticLISAGeometry) -> tuple[dict[str, float], dict[str, float]]:
    _, u, v = polarization_basis(beta, lamb)
    xi_plus: dict[str, float] = {}
    xi_cross: dict[str, float] = {}
    for link, n_vec in geometry.n.items():
        u_dot = float(np.dot(u, n_vec))
        v_dot = float(np.dot(v, n_vec))
        xi_plus[link] = u_dot * u_dot - v_dot * v_dot
        xi_cross[link] = 2.0 * u_dot * v_dot
    return xi_plus, xi_cross


def _link_wave_coeff(link: str, omega: np.ndarray, k_vec: np.ndarray, geometry: StaticLISAGeometry) -> np.ndarray:
    i = int(link[0])
    j = int(link[1])
    n_ij = geometry.n[link]
    denom = 2.0 * (1.0 - float(np.dot(k_vec, n_ij)))
    if np.isclose(denom, 0.0):
        raise ValueError(f"Singular link response for link {link}: denominator is close to zero.")
    phase_emit = np.exp(1j * omega * (geometry.L[link] + float(np.dot(k_vec, geometry.x[j]))))
    phase_recv = np.exp(1j * omega * float(np.dot(k_vec, geometry.x[i])))
    return (phase_emit - phase_recv) / denom


def _delay_phase(delays: Iterable[str], omega: np.ndarray, geometry: StaticLISAGeometry) -> np.ndarray:
    delay_sum = 0.0
    for link in delays:
        delay_sum += geometry.L[link]
    return np.exp(1j * omega * delay_sum)


SAGNAC_TERMS: Mapping[str, tuple[tuple[str, int, tuple[str, ...]], ...]] = {
    "alpha": (
        ("12", +1, ()),
        ("23", +1, ("12",)),
        ("31", +1, ("12", "23")),
        ("13", -1, ()),
        ("32", -1, ("13",)),
        ("21", -1, ("13", "32")),
    ),
    "beta": (
        ("23", +1, ()),
        ("31", +1, ("23",)),
        ("12", +1, ("23", "31")),
        ("21", -1, ()),
        ("13", -1, ("21",)),
        ("32", -1, ("21", "13")),
    ),
    "gamma": (
        ("31", +1, ()),
        ("12", +1, ("31",)),
        ("23", +1, ("31", "12")),
        ("32", -1, ()),
        ("21", -1, ("32",)),
        ("13", -1, ("32", "21")),
    ),
}


def sagnac_polarization_coeff(
    tdi_name: str,
    freqs_hz: Sequence[float] | np.ndarray,
    beta: float,
    lamb: float,
    geometry: StaticLISAGeometry,
    *,
    pol: str,
) -> np.ndarray:
    """Return the Sagnac response coefficient for one polarization."""

    name = tdi_name.lower()
    if name not in SAGNAC_TERMS:
        raise ValueError(f"Unknown Sagnac variable {tdi_name!r}.")
    if pol not in {"plus", "cross"}:
        raise ValueError("pol must be 'plus' or 'cross'.")

    freqs = _as_1d_float(freqs_hz)
    omega = 2.0 * np.pi * freqs
    k_vec, _, _ = polarization_basis(beta, lamb)
    xi_plus, xi_cross = antenna_patterns(beta, lamb, geometry)
    xi = xi_plus if pol == "plus" else xi_cross

    out = np.zeros_like(omega, dtype=np.complex128)
    for link, sign, delays in SAGNAC_TERMS[name]:
        out += (
            float(sign)
            * _delay_phase(delays, omega, geometry)
            * _link_wave_coeff(link, omega, k_vec, geometry)
            * xi[link]
        )
    return out


def projector_vectors(
    freqs_hz: Sequence[float] | np.ndarray,
    beta: float,
    lamb: float,
    geometry: StaticLISAGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the plus and cross projector vectors with shape ``(3, Nf)``."""

    p_plus = np.vstack(
        [sagnac_polarization_coeff(name, freqs_hz, beta, lamb, geometry, pol="plus") for name in SAGNAC_NAMES]
    )
    p_cross = np.vstack(
        [sagnac_polarization_coeff(name, freqs_hz, beta, lamb, geometry, pol="cross") for name in SAGNAC_NAMES]
    )
    return p_plus, p_cross


def coronagraphic_vector(
    freqs_hz: Sequence[float] | np.ndarray,
    beta: float,
    lamb: float,
    geometry: StaticLISAGeometry,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Compute the coronagraphic coefficient vector ``A = P_plus x P_cross``."""

    p_plus, p_cross = projector_vectors(freqs_hz, beta, lamb, geometry)
    avec = np.vstack(
        [
            p_plus[1] * p_cross[2] - p_plus[2] * p_cross[1],
            p_plus[2] * p_cross[0] - p_plus[0] * p_cross[2],
            p_plus[0] * p_cross[1] - p_plus[1] * p_cross[0],
        ]
    )
    if normalize:
        norm = np.sqrt(np.sum(np.abs(avec) ** 2, axis=0))
        good = norm > 0.0
        avec[:, good] = avec[:, good] / norm[good]
    return avec


def compute_kappa_frequency(
    freqs_hz: Sequence[float] | np.ndarray,
    alpha_tilde: Sequence[complex] | np.ndarray,
    beta_tilde: Sequence[complex] | np.ndarray,
    gamma_tilde: Sequence[complex] | np.ndarray,
    *,
    beta: float,
    lamb: float,
    geometry: StaticLISAGeometry,
    normalize_vector: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``kappa_tilde = A . (alpha, beta, gamma)``."""

    freqs = _as_1d_float(freqs_hz)
    alpha = _as_1d_complex(alpha_tilde)
    beta_ch = _as_1d_complex(beta_tilde)
    gamma = _as_1d_complex(gamma_tilde)
    if not (freqs.shape == alpha.shape == beta_ch.shape == gamma.shape):
        raise ValueError("Frequency and Sagnac arrays must have matching shapes.")
    avec = coronagraphic_vector(freqs, beta, lamb, geometry, normalize=normalize_vector)
    kappa = avec[0] * alpha + avec[1] * beta_ch + avec[2] * gamma
    return kappa, avec


def compute_kappa_from_sagnac_time(
    alpha: Sequence[complex] | np.ndarray,
    beta_channel: Sequence[complex] | np.ndarray,
    gamma: Sequence[complex] | np.ndarray,
    *,
    dt: float,
    beta: float,
    lamb: float,
    geometry: StaticLISAGeometry,
    normalize_vector: bool = False,
) -> dict[str, np.ndarray]:
    """Compute coronagraphic TDI from time-domain PyTDI Sagnac channels."""

    alpha_t = _as_1d_complex(alpha)
    beta_t = _as_1d_complex(beta_channel)
    gamma_t = _as_1d_complex(gamma)
    if not (alpha_t.shape == beta_t.shape == gamma_t.shape):
        raise ValueError("alpha, beta, and gamma time arrays must match.")
    freqs = np.fft.fftfreq(alpha_t.size, d=dt)
    alpha_f = _time_to_freq_plus(alpha_t)
    beta_f = _time_to_freq_plus(beta_t)
    gamma_f = _time_to_freq_plus(gamma_t)
    kappa_f, avec = compute_kappa_frequency(
        freqs,
        alpha_f,
        beta_f,
        gamma_f,
        beta=beta,
        lamb=lamb,
        geometry=geometry,
        normalize_vector=normalize_vector,
    )
    return {
        "freqs_hz": freqs,
        "alpha_tilde": alpha_f,
        "beta_tilde": beta_f,
        "gamma_tilde": gamma_f,
        "kappa_tilde": kappa_f,
        "kappa": _freq_to_time_plus(kappa_f),
        "A": avec,
    }


def load_pytdi_sagnac_csv(
    path: str | Path,
    *,
    generation: int = 1,
    suffix: str = "",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load PyTDI Sagnac CSV columns written by ``tdi_computation``."""

    p = Path(path)
    cols = [
        "t_waveform",
        "t_lisa",
        f"alpha{generation}{suffix}",
        f"beta{generation}{suffix}",
        f"gamma{generation}{suffix}",
    ]
    df = pd.read_csv(p, usecols=cols)
    return (
        df["t_waveform"].to_numpy(dtype=float),
        df["t_lisa"].to_numpy(dtype=float),
        {
            "alpha": df[f"alpha{generation}{suffix}"].to_numpy(dtype=float),
            "beta": df[f"beta{generation}{suffix}"].to_numpy(dtype=float),
            "gamma": df[f"gamma{generation}{suffix}"].to_numpy(dtype=float),
        },
    )


def write_kappa_time_csv(
    path: str | Path,
    *,
    t_waveform: np.ndarray,
    t_lisa: np.ndarray,
    results: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    """Write complex kappa time series for one or more sky positions."""

    data: dict[str, np.ndarray] = {
        "sample": np.arange(len(t_waveform)),
        "t_waveform": t_waveform,
        "t_lisa": t_lisa,
    }
    for name, result in results.items():
        kappa = np.asarray(result["kappa"], dtype=np.complex128)
        data[f"kappa_{name}_real"] = np.real(kappa)
        data[f"kappa_{name}_imag"] = np.imag(kappa)
        data[f"kappa_{name}_abs"] = np.abs(kappa)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)


def kappa_summary(
    kappa: Sequence[complex] | np.ndarray,
    freqs_hz: np.ndarray,
    band: tuple[float, float],
) -> dict[str, float]:
    arr = _as_1d_complex(kappa)
    band_mask = (np.abs(freqs_hz) >= band[0]) & (np.abs(freqs_hz) <= band[1])
    spectrum = _time_to_freq_plus(arr)
    return {
        "time_abs_rms": float(np.sqrt(np.mean(np.abs(arr) ** 2))),
        "time_abs_max": float(np.max(np.abs(arr))),
        "freq_band_abs_rms": float(np.sqrt(np.mean(np.abs(spectrum[band_mask]) ** 2))) if np.any(band_mask) else np.nan,
        "freq_band_abs_max": float(np.max(np.abs(spectrum[band_mask]))) if np.any(band_mask) else np.nan,
    }
