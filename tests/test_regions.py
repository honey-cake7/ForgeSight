import unittest

import numpy as np

from forgesight.regions import response_regions


class RegionGeometryTests(unittest.TestCase):
    def test_area_box_and_diagonal_connectivity(self):
        values = np.zeros((12, 15))
        values[2:4, 3:6] = 1
        values[4, 6] = 1  # Touches at corner, belongs to same 8-connected region.
        result = response_regions(values, .5)
        self.assertEqual(result['total_area_px'], 7)
        self.assertEqual(len(result['regions']), 1)
        region = result['regions'][0]
        self.assertEqual(region['bbox_xywh_px'], [3, 2, 4, 3])
        self.assertEqual(region['bbox_diagonal_px'], 5.)
        self.assertFalse(result['pixel_threshold_validated'])

    def test_empty_support_equality_and_bad_maps(self):
        self.assertEqual(response_regions(np.ones((4, 4)), 1.)['regions'], [])
        for values in [np.array([]), np.zeros((2,)), np.array([[float('nan')]])]:
            with self.assertRaises(ValueError):
                response_regions(values, .5)


if __name__ == '__main__':
    unittest.main()
