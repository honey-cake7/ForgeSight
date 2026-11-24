"""Screw geometry remains separate from the existing metal-nut classifier."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from forgesight.defect_type import assess_response, load_profile, predict_type
from forgesight.regions import response_regions


class ScrewDefectTypeTests(unittest.TestCase):
    def sample(self, x=35):
        pixels = np.full((160, 160, 3), 220, np.uint8)
        pixels[48:112, 20:48] = 55
        pixels[66:94, 48:136] = 75
        values = np.zeros((160, 160), np.float32)
        values[72:88, x:x + 8] = .8
        return Image.fromarray(pixels), values

    def profile(self):
        from forgesight.defect_type import SCREW_FEATURES, SCREW_TYPES
        return {'schema_version': 1, 'profile_id': 'screw-test', 'kind': 'screw_decision_tree',
                'category': 'screw', 'feature_names': list(SCREW_FEATURES),
                'labels': list(SCREW_TYPES), 'model_sha256': 'a' * 64, 'image_threshold': .5,
                'tree': {'feature': 'peak_axial', 'threshold': .4,
                         'left': {'label': 'scratch_head'}, 'right': {'label': 'thread_side'}}}

    def test_head_and_threads_follow_object_axis_after_rotation(self):
        from forgesight.defect_type import screw_features, SCREW_FEATURES
        head, values = self.sample()
        geometry = response_regions(values, .5)
        a = screw_features(head, values, geometry)
        self.assertEqual(set(a), set(SCREW_FEATURES))
        rotated = head.transpose(Image.Transpose.ROTATE_180)
        rotated_map = np.rot90(values, 2).copy()
        b = screw_features(rotated, rotated_map, response_regions(rotated_map, .5))
        self.assertLess(a['peak_axial'], .4)
        self.assertAlmostEqual(a['peak_axial'], b['peak_axial'], delta=.02)
        threaded, thread_map = self.sample(110)
        c = screw_features(threaded, thread_map, response_regions(thread_map, .5))
        self.assertGreater(c['peak_axial'], .6)
        self.assertGreater(c['thread_response_fraction'], a['thread_response_fraction'])

    def test_empty_or_misaligned_evidence_abstains(self):
        from forgesight.defect_type import screw_features
        image, values = self.sample()
        self.assertIsNone(screw_features(image, values, response_regions(values, 1)))
        blank = Image.new('RGB', image.size, 'white')
        self.assertIsNone(screw_features(blank, values, response_regions(values, .5)))
        self.assertIsNone(screw_features(image, values[:80], response_regions(values[:80], .5)))

    def test_profile_schema_bounds_labels_features_and_tree_depth(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            profile = self.profile()
            path.write_text(json.dumps(profile))
            exported = load_profile(path)
            features = dict.fromkeys(exported['feature_names'], 0.)
            self.assertEqual(predict_type(features, exported), 'scratch_head')
            features['peak_axial'] = 1.
            self.assertEqual(predict_type(features, exported), 'thread_side')
            features['peak_axial'] = float('nan')
            self.assertEqual(predict_type(features, exported), 'unknown')
            for field, value in [('category', 'metal_nut'), ('labels', ['bent_tip']),
                                 ('feature_names', ['peak_axial'])]:
                bad = dict(profile, **{field: value})
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_profile(path)
            bad = deepcopy(profile)
            bad['tree']['left']['label'] = 'bent_tip'
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                load_profile(path)

    def test_assessment_is_opt_in_hash_bound_and_does_not_change_grade(self):
        image, values = self.sample()
        profile = self.profile()
        record = {'source': {'mode': 'SAVED_IMAGE_REPLAY'}, 'candidate': 'ANOMALY_CANDIDATE',
                  'model': {'export_sha256': 'a' * 64, 'image_threshold': .5},
                  'geometry': response_regions(values, .5), 'timings': {},
                  'decision': {'grade': 'REVIEW'}, 'predicted_defect_type': 'unknown'}
        original = deepcopy(record)
        assess_response(record, image, values, profile)
        self.assertEqual(record['assessment']['defect_type'], 'scratch_head')
        self.assertEqual(record['decision'], original['decision'])
        self.assertEqual(record['predicted_defect_type'], 'unknown')
        record['model']['export_sha256'] = 'b' * 64
        assess_response(record, image, values, profile)
        self.assertEqual(record['assessment']['defect_type'], 'unknown')
        record['source']['mode'] = 'LIVE_CAMERA'
        record.pop('assessment')
        assess_response(record, image, values, profile)
        self.assertNotIn('assessment', record)

    def test_geometry_threshold_must_match_the_detector_and_classifier(self):
        image, values = self.sample()
        profile = self.profile()
        record = {'source': {'mode': 'SAVED_IMAGE_REPLAY'}, 'candidate': 'ANOMALY_CANDIDATE',
                  'model': {'export_sha256': 'a' * 64, 'image_threshold': .5},
                  'geometry': response_regions(values, .4), 'timings': {}}
        assess_response(record, image, values, profile)
        self.assertEqual(record['assessment']['defect_type'], 'unknown')
        self.assertIsNone(record['assessment']['features'])

    def test_json_tree_uses_the_same_float32_inputs_as_sklearn(self):
        from forgesight.defect_type import SCREW_FEATURES
        profile = self.profile()
        profile['tree']['threshold'] = .41025350987911224
        features = dict.fromkeys(SCREW_FEATURES, 0.)
        features['peak_axial'] = profile['tree']['threshold']
        self.assertEqual(predict_type(features, profile), 'thread_side')
        features['peak_axial'] = 1e100
        self.assertEqual(predict_type(features, profile), 'unknown')


if __name__ == '__main__':
    unittest.main()
