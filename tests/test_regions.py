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

    def test_largest_response_gets_provisional_nut_area_and_other_regions_remain_visible(self):
        values = np.zeros((700, 700), dtype=np.float32)
        values[340:360, 340:360] = 1
        values[340:350, 510:520] = 1
        result = response_regions(values, .5, mvtec_nut_layout=True)
        self.assertEqual(result['total_area_px'], 500)
        self.assertEqual(result['additional_region_count'], 1)
        self.assertEqual(result['regions'][0]['area_px'], 400)
        self.assertEqual(result['regions'][0]['approximate_nut_area'], 'central opening')
        self.assertEqual(result['regions'][1]['approximate_nut_area'], 'annular face')
        self.assertFalse(result['pixel_threshold_validated'])
        self.assertEqual(result['area_reference'], 'full_image_not_nut_or_physical_damage')

    def test_broad_response_and_unvalidated_camera_have_no_specific_nut_area(self):
        values = np.ones((700, 700), dtype=np.float32)
        broad = response_regions(values, .5, mvtec_nut_layout=True)
        self.assertIn('undetermined', broad['regions'][0]['approximate_nut_area'])
        camera = response_regions(np.ones((480, 640), dtype=np.float32), .5)
        self.assertNotIn('approximate_nut_area', camera['regions'][0])

    def test_response_crossing_adjacent_nut_areas_names_both(self):
        values = np.zeros((700, 700), dtype=np.float32)
        values[460:480, 340:360] = 1
        region = response_regions(values, .5, mvtec_nut_layout=True)['regions'][0]
        self.assertEqual(region['approximate_nut_area'], 'inner rim / annular face')


if __name__ == '__main__':
    unittest.main()
