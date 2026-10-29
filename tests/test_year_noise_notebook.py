"""Sampling and antialiasing contracts for the year-generation notebook.

These tests do not run a production instrument simulation or validate a mission
noise floor. The notebook's short instrument pilot exercises the connected route.
"""

from pathlib import Path
import importlib.util
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAVE_NUMERICS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "h5py"))


@unittest.skipUnless(HAVE_NUMERICS, "Install numpy, scipy and h5py for resampling checks.")
class YearNoiseSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "noise only generation"))
        import numpy as np
        from scipy.signal import freqz, resample_poly
        import lisa_year_helpers as helpers

        cls.np = np
        cls.freqz = staticmethod(freqz)
        cls.resample = staticmethod(resample_poly)
        cls.helpers = helpers

    def test_exact_year_and_reject_fractional_samples(self):
        plan = self.helpers.YearPlan()
        self.assertEqual(plan.output_size, 7_889_400)
        self.assertEqual(plan.output_size * plan.output_dt_s, 31_557_600.0)
        self.assertEqual(plan.instrument_size, plan.science_size + 2 * plan.guard_size)
        for kwargs in ({"duration_s": 4097.0}, {"output_dt_s": 4.1}, {"tdi_block_s": 513.0}):
            with self.assertRaises(ValueError):
                self.helpers.YearPlan(**kwargs)

    def test_filter_passband_and_alias_rejection(self):
        plan = self.helpers.YearPlan()
        fir = self.helpers.antialias_filter(plan)
        freq, transfer = self.freqz(fir, worN=131072, fs=1 / plan.measurement_dt_s)
        pass_error = self.np.max(self.np.abs(self.np.abs(transfer[freq <= plan.passband_hz]) - 1))
        stop_max = self.np.max(self.np.abs(transfer[freq >= 0.5 / plan.output_dt_s]))
        self.assertLess(pass_error, 1e-4)
        self.assertLess(stop_max, 2e-5)
        times = self.np.arange(65536) * plan.measurement_dt_s
        # A 0.2 Hz tone would alias to 0.05 Hz at the output cadence if sliced.
        tone = self.np.sin(2 * self.np.pi * 0.2 * times)
        filtered = self.resample(tone, 1, plan.downsampling, window=fir)
        self.assertLess(self.np.std(filtered[100:-100]) / self.np.sqrt(0.5), 2e-5)

    def test_overlapping_decimation_matches_continuous_record(self):
        np = self.np
        plan = self.helpers.YearPlan(duration_s=4096.0, tdi_block_s=512.0)
        fir = self.helpers.antialias_filter(plan)
        # Deliberately use colored noise and tones on both sides of Nyquist.
        rng = np.random.default_rng(7812)
        times = np.arange(plan.instrument_size) * plan.measurement_dt_s
        raw = np.cumsum(rng.normal(size=plan.instrument_size))
        raw += 2 * np.sin(2 * np.pi * 0.03 * times) + np.sin(2 * np.pi * 0.2 * times)
        ratio = plan.downsampling
        margin = ((len(fir) - 1) // 2 // ratio + 1) * ratio
        full = self.resample(raw, 1, ratio, window=fir)
        expected = full[plan.guard_size // ratio:(plan.guard_size + plan.science_size) // ratio]
        blocks = []
        block_size = self.helpers.sample_count(plan.tdi_block_s, plan.measurement_dt_s)
        for offset in range(0, plan.science_size, block_size):
            first = plan.guard_size + offset
            last = min(first + block_size, plan.guard_size + plan.science_size)
            block = self.resample(raw[first - margin:last + margin], 1, ratio, window=fir)
            blocks.append(block[margin // ratio:margin // ratio + (last - first) // ratio])
        np.testing.assert_allclose(np.concatenate(blocks), expected, rtol=1e-13, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
