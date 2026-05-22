"""Coronagraphic TDI implementation for static LISA geometry.

This module implements the frequency-domain construction described in:
R. Costa Barroso et al. (2024), arXiv:2406.00190v1.

Key relation:
    kappa_tilde(f, beta, lambda) = A(f, beta, lambda) . D(f)
where:
    D(f) = (alpha_tilde, beta_tilde, gamma_tilde)
and A is built from projector vectors P_+ and P_x:
    A = P_+ x P_x

Assumptions:
- First-generation TDI (static constellation).
- Spacecraft positions are static in the chosen frame.
- Positions are provided in light-seconds by default (c = 1 convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


Link = str
_LINK_KEYS: Tuple[Link, ...] = ("12", "23", "31", "13", "32", "21")


def _as_1d_complex(values: Sequence[complex] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")
    return arr.astype(np.complex128, copy=False)


def _as_1d_float(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")
    return arr


def _validate_link_map(link_data: Mapping[Link, Sequence[complex] | np.ndarray]) -> Dict[Link, np.ndarray]:
    out: Dict[Link, np.ndarray] = {}
    ref_shape: Tuple[int, ...] | None = None
    for link in _LINK_KEYS:
        if link not in link_data:
            raise ValueError(f"Missing link data for key '{link}'.")
        arr = _as_1d_complex(link_data[link])
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(f"All link arrays must have identical shape. '{link}' has {arr.shape}, expected {ref_shape}.")
        out[link] = arr
    return out


def _compute_chunk_indices(
    n_samples: int,
    dt: float,
    *,
    chunk_start_s: float | None = None,
    chunk_duration_s: float | None = None,
    chunk_start_idx: int | None = None,
    chunk_end_idx: int | None = None,
) -> Tuple[int, int]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    if chunk_start_idx is not None or chunk_end_idx is not None:
        start = 0 if chunk_start_idx is None else int(chunk_start_idx)
        end = n_samples if chunk_end_idx is None else int(chunk_end_idx)
    else:
        start = 0 if chunk_start_s is None else int(np.floor(chunk_start_s / dt))
        if chunk_duration_s is None:
            end = n_samples
        else:
            end = start + int(np.floor(chunk_duration_s / dt))

    start = max(0, start)
    end = min(n_samples, end)
    if end <= start:
        raise ValueError(f"Invalid chunk bounds: start={start}, end={end}.")
    return start, end


def _slice_links_time(
    link_data_time: Mapping[Link, Sequence[complex] | np.ndarray],
    *,
    dt: float,
    chunk_start_s: float | None = None,
    chunk_duration_s: float | None = None,
    chunk_start_idx: int | None = None,
    chunk_end_idx: int | None = None,
) -> Tuple[Dict[Link, np.ndarray], np.ndarray]:
    y = _validate_link_map(link_data_time)
    n = y[_LINK_KEYS[0]].size
    i0, i1 = _compute_chunk_indices(
        n,
        dt,
        chunk_start_s=chunk_start_s,
        chunk_duration_s=chunk_duration_s,
        chunk_start_idx=chunk_start_idx,
        chunk_end_idx=chunk_end_idx,
    )
    t = np.arange(i0, i1, dtype=float) * dt
    return {k: v[i0:i1] for k, v in y.items()}, t


def _time_to_freq_plus(x_time: np.ndarray) -> np.ndarray:
    """Fourier transform with +iωt convention, matching this module's delays."""
    n = x_time.size
    return np.fft.ifft(x_time) * n


def _freq_to_time_plus(x_freq: np.ndarray) -> np.ndarray:
    """Inverse of _time_to_freq_plus."""
    n = x_freq.size
    return np.fft.fft(x_freq) / n


def _select_output_mode(output_domain: str) -> Tuple[bool, bool]:
    mode = output_domain.lower()
    if mode == "time":
        return True, False
    if mode == "frequency":
        return False, True
    if mode == "both":
        return True, True
    raise ValueError("output_domain must be one of {'time', 'frequency', 'both'}.")


@dataclass(frozen=True)
class LISAGeometry:
    """Static LISA geometry in light-seconds.

    Attributes:
        x: Spacecraft positions indexed by 1, 2, 3. Values are 3-vectors.
        L: Light-travel times L_ij (seconds) indexed as "ij".
        n: Unit vectors n_ij from spacecraft j to i, indexed as "ij".
    """

    x: Mapping[int, np.ndarray]
    L: Mapping[Link, float]
    n: Mapping[Link, np.ndarray]

    @staticmethod
    def _validate_positions(x: Mapping[int, np.ndarray]) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for idx in (1, 2, 3):
            if idx not in x:
                raise ValueError(f"Missing spacecraft position for index {idx}.")
            vec = np.asarray(x[idx], dtype=float)
            if vec.shape != (3,):
                raise ValueError(f"x[{idx}] must be a 3-vector.")
            out[idx] = vec
        return out

    @classmethod
    def from_positions(
        cls,
        x1: Sequence[float],
        x2: Sequence[float],
        x3: Sequence[float],
        *,
        c_m_per_s: float | None = None,
    ) -> "LISAGeometry":
        """Build geometry from static spacecraft positions.

        Args:
            x1, x2, x3:
                Spacecraft positions.
                - If ``c_m_per_s`` is None, values are interpreted as light-seconds.
                - If ``c_m_per_s`` is given, values are interpreted as meters and
                  converted internally to light-seconds by dividing by c.
            c_m_per_s:
                Speed of light in m/s if positions are in meters.
        """
        x_raw = {1: np.asarray(x1, dtype=float), 2: np.asarray(x2, dtype=float), 3: np.asarray(x3, dtype=float)}
        x = cls._validate_positions(x_raw)

        if c_m_per_s is not None:
            if c_m_per_s <= 0:
                raise ValueError("c_m_per_s must be positive.")
            x = {k: v / c_m_per_s for k, v in x.items()}

        L: Dict[Link, float] = {}
        n: Dict[Link, np.ndarray] = {}
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if i == j:
                    continue
                link = f"{i}{j}"
                delta = x[i] - x[j]
                length = float(np.linalg.norm(delta))
                if length == 0.0:
                    raise ValueError(f"Degenerate geometry: spacecraft {i} and {j} overlap.")
                L[link] = length
                n[link] = delta / length

        return cls(x=x, L=L, n=n)

    @classmethod
    def equilateral(
        cls,
        arm_length_light_seconds: float,
        *,
        rotation_rad: float = 0.0,
    ) -> "LISAGeometry":
        """Construct an equilateral static constellation in the x-y plane."""
        if arm_length_light_seconds <= 0:
            raise ValueError("arm_length_light_seconds must be positive.")

        r = arm_length_light_seconds / np.sqrt(3.0)
        angles = rotation_rad + np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        positions = [np.array([r * np.cos(a), r * np.sin(a), 0.0]) for a in angles]
        return cls.from_positions(positions[0], positions[1], positions[2])


# First-generation Sagnac combinations.
# Each entry is (link_ij, sign, delays_to_apply_before_link).
_SAGNAC_TERMS: Dict[str, Tuple[Tuple[Link, int, Tuple[Link, ...]], ...]] = {
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

# Michelson X channel centered on spacecraft 1.
# First generation (X) and second generation correction (X1 - X) in the
# operator convention Dij x(t) = x(t - Lij).
#
# First-generation expression:
#   X = y13 + D13 y31 + D131 y12 + D1312 y21
#     - [y12 + D12 y21 + D121 y13 + D1213 y31]
#
# Second-generation expression used here follows:
#   X1 = X + D13121 y12 + D131212 y21 + D1312121 y13 + D13121213 y31
#         - [D12131 y13 + D121313 y31 + D1213131 y12 + D12131312 y21]
_MICHELSON_X_FIRST_TERMS: Tuple[Tuple[Link, int, Tuple[Link, ...]], ...] = (
    ("13", +1, ()),
    ("31", +1, ("13",)),
    ("12", +1, ("13", "31")),
    ("21", +1, ("13", "31", "12")),
    ("12", -1, ()),
    ("21", -1, ("12",)),
    ("13", -1, ("12", "21")),
    ("31", -1, ("12", "21", "13")),
)

_MICHELSON_X_SECOND_EXTRA_TERMS: Tuple[Tuple[Link, int, Tuple[Link, ...]], ...] = (
    ("12", +1, ("13", "31", "21")),
    ("21", +1, ("13", "31", "21", "12")),
    ("13", +1, ("13", "31", "21", "12", "21")),
    ("31", +1, ("13", "31", "21", "12", "21", "13")),
    ("13", -1, ("12", "21", "31")),
    ("31", -1, ("12", "21", "31", "13")),
    ("12", -1, ("12", "21", "31", "13", "31")),
    ("21", -1, ("12", "21", "31", "13", "31", "12")),
)


def polarization_basis(beta: float, lamb: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (k, u, v) basis vectors from Eq. (A1-A3)."""
    cb = np.cos(beta)
    sb = np.sin(beta)
    cl = np.cos(lamb)
    sl = np.sin(lamb)

    k = np.array([-cb * cl, -cb * sl, -sb], dtype=float)
    u = np.array([sl, -cl, 0.0], dtype=float)
    v = np.array([-sb * cl, -sb * sl, cb], dtype=float)
    return k, u, v


def antenna_patterns(
    beta: float,
    lamb: float,
    geometry: LISAGeometry,
) -> Tuple[Dict[Link, float], Dict[Link, float]]:
    """Compute xi_plus and xi_cross for all directed links."""
    _, u, v = polarization_basis(beta, lamb)

    xi_plus: Dict[Link, float] = {}
    xi_cross: Dict[Link, float] = {}
    for link, n_vec in geometry.n.items():
        u_dot = float(np.dot(u, n_vec))
        v_dot = float(np.dot(v, n_vec))
        xi_plus[link] = u_dot * u_dot - v_dot * v_dot
        xi_cross[link] = 2.0 * u_dot * v_dot
    return xi_plus, xi_cross


def _link_wave_coeff(
    link: Link,
    omega: np.ndarray,
    k_vec: np.ndarray,
    geometry: LISAGeometry,
) -> np.ndarray:
    i = int(link[0])
    j = int(link[1])

    x_i = geometry.x[i]
    x_j = geometry.x[j]
    L_ij = geometry.L[link]
    n_ij = geometry.n[link]

    denom = 2.0 * (1.0 - float(np.dot(k_vec, n_ij)))
    if np.isclose(denom, 0.0):
        raise ValueError(f"Singular link response for link {link}: denominator ~ 0.")

    phase_emit = np.exp(1j * omega * (L_ij + float(np.dot(k_vec, x_j))))
    phase_recv = np.exp(1j * omega * float(np.dot(k_vec, x_i)))
    return (phase_emit - phase_recv) / denom


def _delay_phase(
    delays: Iterable[Link],
    omega: np.ndarray,
    geometry: LISAGeometry,
) -> np.ndarray:
    delay_sum = 0.0
    for link in delays:
        delay_sum += geometry.L[link]
    return np.exp(1j * omega * delay_sum)


def _map_sc_index(idx: int, shift: int) -> int:
    return ((idx - 1 + shift) % 3) + 1


def _permute_link(link: Link, shift: int) -> Link:
    i = _map_sc_index(int(link[0]), shift)
    j = _map_sc_index(int(link[1]), shift)
    return f"{i}{j}"


def _permute_terms(
    terms: Tuple[Tuple[Link, int, Tuple[Link, ...]], ...],
    shift: int,
) -> Tuple[Tuple[Link, int, Tuple[Link, ...]], ...]:
    out = []
    for link, sign, delays in terms:
        out.append((_permute_link(link, shift), sign, tuple(_permute_link(d, shift) for d in delays)))
    return tuple(out)


def _build_linear_delay_combination(
    terms: Tuple[Tuple[Link, int, Tuple[Link, ...]], ...],
    freqs_hz: Sequence[float] | np.ndarray,
    y_tilde: Mapping[Link, Sequence[complex] | np.ndarray],
    geometry: LISAGeometry,
) -> np.ndarray:
    freqs = _as_1d_float(freqs_hz)
    omega = 2.0 * np.pi * freqs
    out = np.zeros_like(omega, dtype=np.complex128)
    for link, sign, delays in terms:
        if link not in y_tilde:
            raise ValueError(f"Missing y_tilde for link {link}.")
        y_arr = _as_1d_complex(y_tilde[link])
        if y_arr.shape != out.shape:
            raise ValueError(f"y_tilde[{link}] has shape {y_arr.shape}, expected {out.shape}.")
        out += float(sign) * _delay_phase(delays, omega, geometry) * y_arr
    return out


def sagnac_polarization_coeff(
    tdi_name: str,
    freqs_hz: Sequence[float] | np.ndarray,
    beta: float,
    lamb: float,
    geometry: LISAGeometry,
    pol: str,
) -> np.ndarray:
    """Return alpha_+, beta_+, gamma_+ or alpha_x, beta_x, gamma_x.

    This computes Eq. (A18)-style coefficients for one Sagnac variable.
    """
    name = tdi_name.lower()
    if name not in _SAGNAC_TERMS:
        raise ValueError(f"Unknown TDI variable '{tdi_name}'.")
    if pol not in {"plus", "cross"}:
        raise ValueError("pol must be 'plus' or 'cross'.")

    freqs = _as_1d_float(freqs_hz)
    omega = 2.0 * np.pi * freqs
    k_vec, _, _ = polarization_basis(beta, lamb)
    xi_plus, xi_cross = antenna_patterns(beta, lamb, geometry)
    xi = xi_plus if pol == "plus" else xi_cross

    out = np.zeros_like(omega, dtype=np.complex128)
    for link, sign, delays in _SAGNAC_TERMS[name]:
        coeff = _link_wave_coeff(link, omega, k_vec, geometry)
        phase = _delay_phase(delays, omega, geometry)
        out += float(sign) * phase * coeff * xi[link]
    return out


def build_michelson_xyz_first_generation(
    freqs_hz: Sequence[float] | np.ndarray,
    y_tilde: Mapping[Link, Sequence[complex] | np.ndarray],
    geometry: LISAGeometry,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build first-generation Michelson channels (X, Y, Z) from link data."""
    X = _build_linear_delay_combination(_MICHELSON_X_FIRST_TERMS, freqs_hz, y_tilde, geometry)
    Y = _build_linear_delay_combination(_permute_terms(_MICHELSON_X_FIRST_TERMS, shift=1), freqs_hz, y_tilde, geometry)
    Z = _build_linear_delay_combination(_permute_terms(_MICHELSON_X_FIRST_TERMS, shift=2), freqs_hz, y_tilde, geometry)
    return X, Y, Z


def build_michelson_xyz_second_generation(
    freqs_hz: Sequence[float] | np.ndarray,
    y_tilde: Mapping[Link, Sequence[complex] | np.ndarray],
    geometry: LISAGeometry,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build second-generation Michelson channels (X1, Y1, Z1) from link data.

    Naming note:
    - In some literature, the second-generation Michelson channel centered on
      spacecraft 1 is denoted X2. In many LISA pipelines it is denoted X1.
    - This function returns the second-generation set as (X1, Y1, Z1).
    """
    X = _build_linear_delay_combination(_MICHELSON_X_FIRST_TERMS, freqs_hz, y_tilde, geometry)
    X += _build_linear_delay_combination(_MICHELSON_X_SECOND_EXTRA_TERMS, freqs_hz, y_tilde, geometry)

    y_terms_first = _permute_terms(_MICHELSON_X_FIRST_TERMS, shift=1)
    y_terms_extra = _permute_terms(_MICHELSON_X_SECOND_EXTRA_TERMS, shift=1)
    Y = _build_linear_delay_combination(y_terms_first, freqs_hz, y_tilde, geometry)
    Y += _build_linear_delay_combination(y_terms_extra, freqs_hz, y_tilde, geometry)

    z_terms_first = _permute_terms(_MICHELSON_X_FIRST_TERMS, shift=2)
    z_terms_extra = _permute_terms(_MICHELSON_X_SECOND_EXTRA_TERMS, shift=2)
    Z = _build_linear_delay_combination(z_terms_first, freqs_hz, y_tilde, geometry)
    Z += _build_linear_delay_combination(z_terms_extra, freqs_hz, y_tilde, geometry)
    return X, Y, Z


def projector_vectors(
    freqs_hz: Sequence[float] | np.ndarray,
    beta: float,
    lamb: float,
    geometry: LISAGeometry,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return P_plus and P_cross arrays of shape (3, Nf)."""
    p_plus = np.vstack(
        [
            sagnac_polarization_coeff("alpha", freqs_hz, beta, lamb, geometry, pol="plus"),
            sagnac_polarization_coeff("beta", freqs_hz, beta, lamb, geometry, pol="plus"),
            sagnac_polarization_coeff("gamma", freqs_hz, beta, lamb, geometry, pol="plus"),
        ]
    )
    p_cross = np.vstack(
        [
            sagnac_polarization_coeff("alpha", freqs_hz, beta, lamb, geometry, pol="cross"),
            sagnac_polarization_coeff("beta", freqs_hz, beta, lamb, geometry, pol="cross"),
            sagnac_polarization_coeff("gamma", freqs_hz, beta, lamb, geometry, pol="cross"),
        ]
    )
    return p_plus, p_cross


def coronagraphic_A_vector(
    freqs_hz: Sequence[float] | np.ndarray,
    beta: float,
    lamb: float,
    geometry: LISAGeometry,
) -> np.ndarray:
    """Compute A = P_plus x P_cross (Eq. 11), shape (3, Nf)."""
    p_plus, p_cross = projector_vectors(freqs_hz, beta, lamb, geometry)

    a0 = p_plus[1] * p_cross[2] - p_plus[2] * p_cross[1]
    a1 = p_plus[2] * p_cross[0] - p_plus[0] * p_cross[2]
    a2 = p_plus[0] * p_cross[1] - p_plus[1] * p_cross[0]
    return np.vstack([a0, a1, a2])


def build_sagnac_from_links(
    tdi_name: str,
    freqs_hz: Sequence[float] | np.ndarray,
    y_tilde: Mapping[Link, Sequence[complex] | np.ndarray],
    geometry: LISAGeometry,
) -> np.ndarray:
    """Build alpha_tilde, beta_tilde or gamma_tilde from link data y_ij.

    Args:
        tdi_name:
            One of {"alpha", "beta", "gamma"}.
        freqs_hz:
            Frequency bins.
        y_tilde:
            Mapping with entries for directed links:
            {"12","23","31","13","32","21"}.
    """
    name = tdi_name.lower()
    if name not in _SAGNAC_TERMS:
        raise ValueError(f"Unknown TDI variable '{tdi_name}'.")

    freqs = _as_1d_float(freqs_hz)
    omega = 2.0 * np.pi * freqs
    out = np.zeros_like(omega, dtype=np.complex128)

    for link, sign, delays in _SAGNAC_TERMS[name]:
        if link not in y_tilde:
            raise ValueError(f"Missing y_tilde for link {link}.")
        y_arr = _as_1d_complex(y_tilde[link])
        if y_arr.shape != out.shape:
            raise ValueError(f"y_tilde[{link}] has shape {y_arr.shape}, expected {out.shape}.")
        phase = _delay_phase(delays, omega, geometry)
        out += float(sign) * phase * y_arr
    return out


def coronagraphic_kappa(
    freqs_hz: Sequence[float] | np.ndarray,
    beta: float,
    lamb: float,
    geometry: LISAGeometry,
    alpha_tilde: Sequence[complex] | np.ndarray,
    beta_tilde: Sequence[complex] | np.ndarray,
    gamma_tilde: Sequence[complex] | np.ndarray,
) -> np.ndarray:
    """Compute coronagraphic kappa_tilde(f, beta, lambda) from Eq. (12)."""
    A = coronagraphic_A_vector(freqs_hz, beta, lamb, geometry)

    a = _as_1d_complex(alpha_tilde)
    b = _as_1d_complex(beta_tilde)
    g = _as_1d_complex(gamma_tilde)

    if not (a.shape == b.shape == g.shape == A.shape[1:]):
        raise ValueError(
            f"Data arrays must all match frequency shape {(A.shape[1],)}; got {a.shape}, {b.shape}, {g.shape}."
        )

    return A[0] * a + A[1] * b + A[2] * g


def xyz_to_aet(
    X_tilde: Sequence[complex] | np.ndarray,
    Y_tilde: Sequence[complex] | np.ndarray,
    Z_tilde: Sequence[complex] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform Michelson XYZ channels into orthogonal AET channels.

    This linear transform is the standard orthonormal basis:
      A = (Z - X) / sqrt(2)
      E = (X - 2Y + Z) / sqrt(6)
      T = (X + Y + Z) / sqrt(3)

    Notes:
    - A/E/T are not a TDI generation by themselves. They are linear
      combinations of some XYZ set (which may be first- or second-generation).
    """
    X = _as_1d_complex(X_tilde)
    Y = _as_1d_complex(Y_tilde)
    Z = _as_1d_complex(Z_tilde)
    if not (X.shape == Y.shape == Z.shape):
        raise ValueError(f"XYZ shapes must match; got {X.shape}, {Y.shape}, {Z.shape}.")

    A = (Z - X) / np.sqrt(2.0)
    E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
    T = (X + Y + Z) / np.sqrt(3.0)
    return A, E, T


def aet_to_xyz(
    A_tilde: Sequence[complex] | np.ndarray,
    E_tilde: Sequence[complex] | np.ndarray,
    T_tilde: Sequence[complex] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse transform from orthogonal AET channels to Michelson XYZ."""
    A = _as_1d_complex(A_tilde)
    E = _as_1d_complex(E_tilde)
    T = _as_1d_complex(T_tilde)
    if not (A.shape == E.shape == T.shape):
        raise ValueError(f"AET shapes must match; got {A.shape}, {E.shape}, {T.shape}.")

    X = -A / np.sqrt(2.0) + E / np.sqrt(6.0) + T / np.sqrt(3.0)
    Y = -(2.0 / np.sqrt(6.0)) * E + T / np.sqrt(3.0)
    Z = A / np.sqrt(2.0) + E / np.sqrt(6.0) + T / np.sqrt(3.0)
    return X, Y, Z


def compute_michelson_response_from_links(
    link_data_time: Mapping[Link, Sequence[complex] | np.ndarray],
    *,
    dt: float,
    geometry: LISAGeometry,
    generation: str = "second",
    channels: str = "both",
    output_domain: str = "both",
    chunk_start_s: float | None = None,
    chunk_duration_s: float | None = None,
    chunk_start_idx: int | None = None,
    chunk_end_idx: int | None = None,
) -> Dict[str, object]:
    """Compute Michelson response from link data with chunking and domain options.

    Args:
        link_data_time:
            Time-domain link data for keys {"12","23","31","13","32","21"}.
        dt:
            Sampling cadence in seconds.
        geometry:
            Static LISA geometry.
        generation:
            "first" for (X,Y,Z), "second" for (X1,Y1,Z1)-style channels.
        channels:
            "xyz", "aet", or "both".
        output_domain:
            "time", "frequency", or "both".
        chunk_start_s, chunk_duration_s, chunk_start_idx, chunk_end_idx:
            Optional chunk selection.

    Returns:
        Dictionary with:
        - "time_s": sampled time vector for selected chunk.
        - "freqs_hz": frequency bins for selected chunk.
        - optional "time": {"X","Y","Z"} and/or {"A","E","T"}
        - optional "frequency": {"X","Y","Z"} and/or {"A","E","T"}
    """
    need_time, need_freq = _select_output_mode(output_domain)
    channel_mode = channels.lower()
    if channel_mode not in {"xyz", "aet", "both"}:
        raise ValueError("channels must be one of {'xyz', 'aet', 'both'}.")

    y_chunk, t = _slice_links_time(
        link_data_time,
        dt=dt,
        chunk_start_s=chunk_start_s,
        chunk_duration_s=chunk_duration_s,
        chunk_start_idx=chunk_start_idx,
        chunk_end_idx=chunk_end_idx,
    )

    # Compute in frequency domain, then optionally transform to time domain.
    y_f = {k: _time_to_freq_plus(v) for k, v in y_chunk.items()}
    freqs_hz = np.fft.fftfreq(t.size, d=dt)

    gen = generation.lower()
    if gen == "first":
        Xf, Yf, Zf = build_michelson_xyz_first_generation(freqs_hz, y_f, geometry)
    elif gen == "second":
        Xf, Yf, Zf = build_michelson_xyz_second_generation(freqs_hz, y_f, geometry)
    else:
        raise ValueError("generation must be one of {'first', 'second'}.")

    out: Dict[str, object] = {"time_s": t, "freqs_hz": freqs_hz}
    if need_freq:
        freq_dict: Dict[str, np.ndarray] = {}
        if channel_mode in {"xyz", "both"}:
            freq_dict.update({"X": Xf, "Y": Yf, "Z": Zf})
        if channel_mode in {"aet", "both"}:
            Af, Ef, Tf = xyz_to_aet(Xf, Yf, Zf)
            freq_dict.update({"A": Af, "E": Ef, "T": Tf})
        out["frequency"] = freq_dict

    if need_time:
        Xt = _freq_to_time_plus(Xf)
        Yt = _freq_to_time_plus(Yf)
        Zt = _freq_to_time_plus(Zf)
        time_dict: Dict[str, np.ndarray] = {}
        if channel_mode in {"xyz", "both"}:
            time_dict.update({"X": Xt, "Y": Yt, "Z": Zt})
        if channel_mode in {"aet", "both"}:
            At, Et, Tt = xyz_to_aet(Xt, Yt, Zt)
            time_dict.update({"A": At, "E": Et, "T": Tt})
        out["time"] = time_dict
    return out


def compute_sagnac_and_coronagraphic_from_links(
    link_data_time: Mapping[Link, Sequence[complex] | np.ndarray],
    *,
    dt: float,
    geometry: LISAGeometry,
    beta: float,
    lamb: float,
    output_domain: str = "both",
    chunk_start_s: float | None = None,
    chunk_duration_s: float | None = None,
    chunk_start_idx: int | None = None,
    chunk_end_idx: int | None = None,
) -> Dict[str, object]:
    """Compute Sagnac (alpha,beta,gamma) and coronagraphic kappa from link data.

    Frequency-domain outputs are native; time-domain outputs are returned via
    inverse transform and are suitable for downstream TF decomposition.
    """
    need_time, need_freq = _select_output_mode(output_domain)
    y_chunk, t = _slice_links_time(
        link_data_time,
        dt=dt,
        chunk_start_s=chunk_start_s,
        chunk_duration_s=chunk_duration_s,
        chunk_start_idx=chunk_start_idx,
        chunk_end_idx=chunk_end_idx,
    )
    y_f = {k: _time_to_freq_plus(v) for k, v in y_chunk.items()}
    freqs_hz = np.fft.fftfreq(t.size, d=dt)

    alpha_f = build_sagnac_from_links("alpha", freqs_hz, y_f, geometry)
    beta_f = build_sagnac_from_links("beta", freqs_hz, y_f, geometry)
    gamma_f = build_sagnac_from_links("gamma", freqs_hz, y_f, geometry)
    kappa_f = coronagraphic_kappa(freqs_hz, beta, lamb, geometry, alpha_f, beta_f, gamma_f)

    out: Dict[str, object] = {"time_s": t, "freqs_hz": freqs_hz}
    if need_freq:
        out["frequency"] = {
            "alpha": alpha_f,
            "beta": beta_f,
            "gamma": gamma_f,
            "kappa": kappa_f,
        }
    if need_time:
        out["time"] = {
            "alpha": _freq_to_time_plus(alpha_f),
            "beta": _freq_to_time_plus(beta_f),
            "gamma": _freq_to_time_plus(gamma_f),
            "kappa": _freq_to_time_plus(kappa_f),
        }
    return out


def _resolve_wdm_class(wdm_class: type | None = None) -> type:
    if wdm_class is not None:
        return wdm_class
    try:
        from wdm_wavelet.wdm import WDM as _WDM  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without dependency.
        raise ModuleNotFoundError(
            "wdm-wavelet is required for WDM TF maps. "
            "Install with: pip install wdm-wavelet"
        ) from exc
    return _WDM


def compute_wdm_tf_map_for_series(
    series_time: Mapping[str, Sequence[complex] | np.ndarray],
    *,
    wdm_args: Sequence[Any] = (32, 64, 6, 10),
    wdm_kwargs: Mapping[str, Any] | None = None,
    reconstruct_time: bool = False,
    enforce_real_input: bool = True,
    imag_tol: float = 1e-10,
    strict_real: bool = False,
    wdm_class: type | None = None,
) -> Dict[str, object]:
    """Apply WDM transform (t2w) on multiple time-series channels.

    Args:
        series_time:
            Mapping from channel name to 1D time-domain array.
        wdm_args, wdm_kwargs:
            Arguments used to construct ``WDM(*wdm_args, **wdm_kwargs)``.
        reconstruct_time:
            If True, also run inverse transform ``w2t``.
        enforce_real_input:
            WDM is typically used on real-valued input. If True, reject channels
            with |imag| > imag_tol only when ``strict_real=True``.
            Otherwise, the real part is used.
        imag_tol:
            Tolerance used with ``enforce_real_input``.
        strict_real:
            If True and ``enforce_real_input`` is True, raise when imaginary
            part exceeds ``imag_tol``. If False, always project to real.
        wdm_class:
            Optional class injection for testing. Defaults to
            ``wdm_wavelet.wdm.WDM``.
    """
    kwargs = {} if wdm_kwargs is None else dict(wdm_kwargs)
    wdm = _resolve_wdm_class(wdm_class)(*wdm_args, **kwargs)

    tf_maps: Dict[str, Any] = {}
    recon_time: Dict[str, np.ndarray] = {}
    for key, values in series_time.items():
        x = _as_1d_complex(values)
        if enforce_real_input:
            max_imag = float(np.max(np.abs(np.imag(x))))
            if strict_real and max_imag > imag_tol:
                raise ValueError(
                    f"Channel '{key}' is complex in time domain (max imag={max_imag:.3e}). "
                    "Set strict_real=False or enforce_real_input=False to override."
                )
            x_in: np.ndarray = np.real(x)
        else:
            x_in = x

        tf = wdm.t2w(x_in)
        tf_maps[key] = tf
        if reconstruct_time:
            recon_time[key] = np.asarray(wdm.w2t(tf))

    out: Dict[str, object] = {"tf_maps": tf_maps}
    if reconstruct_time:
        out["reconstructed_time"] = recon_time
    return out


def compute_wdm_tf_for_aet_and_kappa_from_links(
    link_data_time: Mapping[Link, Sequence[complex] | np.ndarray],
    *,
    dt: float,
    geometry: LISAGeometry,
    beta: float,
    lamb: float,
    generation: str = "second",
    chunk_start_s: float | None = None,
    chunk_duration_s: float | None = None,
    chunk_start_idx: int | None = None,
    chunk_end_idx: int | None = None,
    include_frequency_domain_tdi: bool = True,
    wdm_args: Sequence[Any] = (32, 64, 6, 10),
    wdm_kwargs: Mapping[str, Any] | None = None,
    reconstruct_time: bool = False,
    enforce_real_input: bool = True,
    imag_tol: float = 1e-10,
    strict_real: bool = False,
    wdm_class: type | None = None,
) -> Dict[str, object]:
    """Compute chunked A/E/T and kappa, then apply WDM TF transform.

    Returns:
        - ``time_s``
        - ``tdi_time``: dict with ``A,E,T,kappa`` arrays
        - ``wdm_tf``: dict with transformed map objects
        - optionally ``tdi_frequency`` if include_frequency_domain_tdi=True
        - optionally ``wdm_reconstructed_time`` if reconstruct_time=True
    """
    tdi_domain = "both" if include_frequency_domain_tdi else "time"
    m = compute_michelson_response_from_links(
        link_data_time,
        dt=dt,
        geometry=geometry,
        generation=generation,
        channels="aet",
        output_domain=tdi_domain,
        chunk_start_s=chunk_start_s,
        chunk_duration_s=chunk_duration_s,
        chunk_start_idx=chunk_start_idx,
        chunk_end_idx=chunk_end_idx,
    )
    s = compute_sagnac_and_coronagraphic_from_links(
        link_data_time,
        dt=dt,
        geometry=geometry,
        beta=beta,
        lamb=lamb,
        output_domain=tdi_domain,
        chunk_start_s=chunk_start_s,
        chunk_duration_s=chunk_duration_s,
        chunk_start_idx=chunk_start_idx,
        chunk_end_idx=chunk_end_idx,
    )

    t_m = np.asarray(m["time_s"])
    t_s = np.asarray(s["time_s"])
    if t_m.shape != t_s.shape or not np.allclose(t_m, t_s):
        raise RuntimeError("Internal mismatch: Michelson and Sagnac chunking returned inconsistent time vectors.")

    tdi_time = {
        "A": np.asarray(m["time"]["A"]),
        "E": np.asarray(m["time"]["E"]),
        "T": np.asarray(m["time"]["T"]),
        "kappa": np.asarray(s["time"]["kappa"]),
    }
    wdm_out = compute_wdm_tf_map_for_series(
        tdi_time,
        wdm_args=wdm_args,
        wdm_kwargs=wdm_kwargs,
        reconstruct_time=reconstruct_time,
        enforce_real_input=enforce_real_input,
        imag_tol=imag_tol,
        strict_real=strict_real,
        wdm_class=wdm_class,
    )

    out: Dict[str, object] = {
        "time_s": t_m,
        "tdi_time": tdi_time,
        "wdm_tf": wdm_out["tf_maps"],
    }
    if include_frequency_domain_tdi:
        out["freqs_hz"] = np.asarray(m["freqs_hz"])
        out["tdi_frequency"] = {
            "A": np.asarray(m["frequency"]["A"]),
            "E": np.asarray(m["frequency"]["E"]),
            "T": np.asarray(m["frequency"]["T"]),
            "kappa": np.asarray(s["frequency"]["kappa"]),
        }
    if reconstruct_time:
        out["wdm_reconstructed_time"] = wdm_out["reconstructed_time"]
    return out


# Alias for user's spelling.
choronagraphic_kappa = coronagraphic_kappa
