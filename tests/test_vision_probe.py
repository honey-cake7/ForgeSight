"""Guard measurement integrity and bank sizing; no model accuracy simulations."""

import math
import unittest

from forgesight.vision_probe import estimated_bank_size, timing_summary


class VisionProbeTests(unittest.TestCase):
    def test_percentiles_use_observed_samples_and_interpolation(self):
        summary = timing_summary(list(range(1, 21)))
        self.assertEqual(summary["p50_ms"], 10.5)
        self.assertAlmostEqual(summary["p95_ms"], 19.05)
        self.assertEqual(summary["samples"], 20)

    def test_invalid_or_insufficient_timings_cannot_be_reported(self):
        for values in ([1.0] * 19, [1.0] * 19 + [0], [1.0] * 19 + [math.nan], [1.0] * 19 + [math.inf]):
            with self.assertRaisesRegex(ValueError, "positive finite"):
                timing_summary(values)

    def test_bank_estimate_scales_with_count_and_resolution(self):
        self.assertEqual(estimated_bank_size(176, 32 * 32, 0.1), 18022)
        self.assertEqual(estimated_bank_size(176, 48 * 48, 0.1), 40550)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            estimated_bank_size(176, 1024, 0)


if __name__ == "__main__":
    unittest.main()
