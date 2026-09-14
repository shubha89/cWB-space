"""Saved observed-power replay of reconstruction-window growth, without a new transform."""
from pathlib import Path
import sys, unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cwb_space.reconstruction import select_reconstruction_window

class ReconstructionSelection(unittest.TestCase):
    def test_original_approved_growth(self):
        p = Path(__file__).resolve().parents[1] / 'data/analysis/reconstruction_selection.npz'
        z = np.load(p)
        band = (z['grid_freqs_hz'] >= 1e-5) & (z['grid_freqs_hz'] <= .02)
        guard = (z['grid_times_day'] >= .5) & (z['grid_times_day'] <= 13.5)
        mask, _, window = select_reconstruction_window(z['normalized_power'], z['original_mask'], band, guard)
        np.testing.assert_array_equal(mask, z['expanded_mask'])
        self.assertTrue(np.all(mask[z['original_mask']]))
        self.assertFalse(np.any((mask & ~z['original_mask']) & ~window))

if __name__ == '__main__':
    unittest.main()
