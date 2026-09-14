"""Observed-only event response, as used in the paper's response study.

Inputs are unnormalized observed monitor-channel WDM powers on a common grid.
No noise-only realization or injection truth is accepted by this function.
"""
import numpy as np
import pandas as pd

ABC_CHANNELS = ('alpha2', 'beta2', 'gamma2')
ZETA_CHANNELS = ('zeta21', 'zeta22', 'zeta23')

def observed_background_ratio(powers, masks):
    """Return event ratios, per-channel sums, and frequency-wise median floors.

All retained masks form the exclusion union. Each frequency uses all remaining
valid time samples; negative unit-subtracted normalized powers are clipped.
Undefined ratios are flagged, never regularized. Empty event collections return
empty tables and no floors, matching the no-candidate workflow.
"""
    if not masks:
        return pd.DataFrame(), pd.DataFrame(), {}
    union = np.logical_or.reduce(list(masks.values()))
    if union.ndim != 2 or not np.all((~union).sum(axis=1) > 0):
        raise ValueError('Every included frequency needs off-event samples.')
    sums, floors = [], {}
    for channel in (*ABC_CHANNELS, *ZETA_CHANNELS):
        q = np.asarray(powers[channel], dtype=float)
        if q.shape != union.shape or not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError('Finite nonnegative powers on the mask grid required.')
        floor = np.array([np.median(v[~m]) for v, m in zip(q, union)])
        if not np.all(np.isfinite(floor) & (floor > 0)):
            raise ValueError('Nonpositive or undefined observed background.')
        floors[channel] = floor
        excess = np.maximum(q / floor[:, None] - 1, 0)
        for event_id, mask in masks.items():
            sums.append(dict(event_id=event_id, channel=channel,
                             value=float(excess[mask].sum())))
    rows = []
    for event_id in masks:
        selected = [r for r in sums if r['event_id'] == event_id]
        numerator = sum(r['value'] for r in selected if r['channel'] in ZETA_CHANNELS)
        denominator = sum(r['value'] for r in selected if r['channel'] in ABC_CHANNELS)
        rows.append(dict(event_id=event_id, numerator=numerator, denominator=denominator,
                         ratio=numerator / denominator if denominator > 0 else np.nan,
                         status='OK' if denominator > 0 else 'ZERO_DENOMINATOR'))
    return pd.DataFrame(rows), pd.DataFrame(sums), floors
