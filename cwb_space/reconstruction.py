"""Reconstruction primitives copied from the published reconstruction-window validation driver.

Use raw WDM coefficients, never normalized search coefficients, for inversion.
The caller supplies the existing seed mask, observed normalized A/E power,
original grid, study band and guarded quantile domain. Grid equality between
search and reconstruction transforms must be checked by the caller.
"""
from collections import deque
import numpy as np

def inverse_quadrature_average(wdm, transform):
    """Existing raw inverse convention; does not guarantee exact round trips.

Validated research usage used M512/K256/beta6/precision6/numba. K32 inverses
failed earlier controls. Finite K256 residuals and boundaries remain relevant.
"""
    return .5 * (np.asarray(wdm.w2t(transform).value)
                 + np.asarray(wdm.w2tQ(transform).value))

def grow(original, eligible):
    """Published adjacency: ±1 frequency same-time; ±4 at adjacent time."""
    result = original.copy()
    nf, nt = result.shape
    queue = deque(zip(*np.nonzero(result)))
    offsets = [(-1, 0), (1, 0)] + [(df, dt) for dt in [-1, 1] for df in range(-4, 5)]
    while queue:
        f, t = queue.popleft()
        for df, dt in offsets:
            ff, tt = f + df, t + dt
            if 0 <= ff < nf and 0 <= tt < nt and eligible[ff, tt] and not result[ff, tt]:
                result[ff, tt] = True
                queue.append((ff, tt))
    return result

def select_reconstruction_window(power, original, band, guard):
    """Two-bin margin/95th-percentile growth on supplied observed powers.

power is the sum of separately full-valid-record median-normalized A/E
K256 powers. band and guard are Boolean frequency/time arrays supplied from
that study's approved configuration, not inferred from injected truth.
"""
    occupied = np.flatnonzero(original.any(axis=0))
    first, last = int(occupied[0]), int(occupied[-1])
    left, right = max(0, first - 2), min(power.shape[1] - 1, last + 2)
    assert not np.any(original[~band])
    threshold = float(np.quantile(power[np.ix_(band, guard)], .95))
    window = np.zeros_like(original)
    window[np.ix_(band, np.arange(left, right + 1))] = True
    eligible = (power >= threshold) & window
    expanded = grow(original, eligible)
    assert np.all(expanded[original])
    assert not np.any((expanded & ~original) & ~eligible)
    return expanded, threshold, window
