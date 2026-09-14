"""Regression against original research masks, all event scalars, and ratios."""
from pathlib import Path
import sys, unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from examples.recover_candidates import recover, DATA
from cwb_space import observed_background_ratio

class SavedAnalysis(unittest.TestCase):
    def test_original_search_and_response(self):
        events, masks, ratios, _, _ = recover()
        expected = pd.read_csv(DATA / 'search.csv')
        self.assertEqual(events.event_id.tolist(), expected.event_id.tolist())
        for column in expected:
            if pd.api.types.is_numeric_dtype(expected[column]):
                np.testing.assert_allclose(events[column], expected[column], rtol=1e-12, atol=1e-14, equal_nan=True)
            else:
                self.assertEqual(events[column].tolist(), expected[column].tolist())
        saved = np.load(DATA / 'masks.npz')
        self.assertEqual(set(masks), set(saved.files))
        for key in masks:
            np.testing.assert_array_equal(masks[key], saved[key])
        expected_r = pd.read_csv(DATA / 'expected_response.csv').set_index('event_id')
        for row in ratios.itertuples():
            ref = expected_r.loc[row.event_id]
            np.testing.assert_allclose([row.numerator, row.denominator, row.ratio],
                                      [ref.off_subtracted_N, ref.off_subtracted_D, ref.off_subtracted_R], rtol=1e-12)

    def test_zero_denominator_and_exclusion(self):
        mask = np.array([[True, False, False]])
        powers = {c: np.array([[1., 1., 1.]]) for c in ['alpha2','beta2','gamma2','zeta21','zeta22','zeta23']}
        powers['zeta21'][0, 0] = 100
        ratios, _, floors = observed_background_ratio(powers, {'candidate': mask})
        self.assertEqual(ratios.iloc[0].status, 'ZERO_DENOMINATOR')
        self.assertTrue(np.isnan(ratios.iloc[0].ratio))
        self.assertEqual(floors['zeta21'][0], 1.)
        with self.assertRaises(ValueError):
            observed_background_ratio(powers, {'candidate': np.ones((1, 3), bool)})

if __name__ == '__main__':
    unittest.main()
