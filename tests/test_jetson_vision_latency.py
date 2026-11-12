"""Percentile labels in Jetson evidence use a reproducible calculation."""

import unittest

from scripts.jetson_offline_vision_latency import percentile


class LatencyPercentileTests(unittest.TestCase):
    def test_interpolated_p50_p95(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], .5), 3)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], .95), 4.8)


if __name__ == "__main__":
    unittest.main()
