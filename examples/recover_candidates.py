"""Reproduce one saved observed-map search and response calculation.

Run from the repository root: python examples/recover_candidates.py
Requires numpy and pandas only. No WDM transform or instrument run occurs.
"""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cwb_space import SeededChirpEventConfig, find_seeded_chirp_events, observed_background_ratio

DATA = Path(__file__).resolve().parents[1] / 'data' / 'analysis'

def recover(data=DATA):
    metadata = json.loads((data / 'manifest.json').read_text())
    maps = dict(np.load(data / 'search_maps.npz'))
    events, masks = find_seeded_chirp_events(
        maps, config=SeededChirpEventConfig(**metadata['configuration']),
        start_day=metadata['start_day'], stop_day=metadata['stop_day'],
        label=metadata['record'])
    monitor = np.load(data / 'monitor_powers.npz')
    used = monitor['used_frequency_indices']
    powers = {key.removesuffix('_observed_power'): monitor[key]
              for key in monitor.files if key.endswith('_observed_power')}
    ratios, sums, floors = observed_background_ratio(
        powers, {key: mask[used] for key, mask in masks.items()})
    return events, masks, ratios, sums, floors

if __name__ == '__main__':
    events, _, ratios, _, _ = recover()
    print(events[['event_id', 'event_statistic', 'ae_complex_coherence']].to_string(index=False))
    print(ratios.to_string(index=False))
