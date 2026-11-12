"""Synthetic policy fixtures exercise rules, never claim detector/classifier accuracy."""

from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from forgesight.grade import grade_record, validate_policy


class GradeTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(Path('configs/forgegrade_v1.json').read_text())
        p = self.policy
        self.record = {'status': 'ANALYZED', 'source': {'size_px': [700, 700]},
                       'model': {k: p[k] for k in ('checkpoint_sha256', 'score_policy_sha256', 'image_threshold')},
                       'score': .1, 'candidate': 'NORMAL_CANDIDATE',
                       'predicted_defect_type': 'unknown', 'type_status': 'NO_VALIDATED_SUBTYPE_CLASSIFIER',
                       'geometry': {'source_size_px': [700, 700], 'units': 'pixels',
                                    'threshold': p['image_threshold'], 'total_area_px': 0, 'regions': [],
                                    'measurement': 'thresholded_model_response_not_physical_damage',
                                    'pixel_threshold_validated': False, 'connectivity': 8, 'morphology': 'none'}}

    def anomaly(self, x=10, y=10, width=10, height=10):
        self.record.update(score=.6, candidate='ANOMALY_CANDIDATE', predicted_defect_type='scratch',
                           type_status='VALIDATED_SAME_DOMAIN_RELIABLE')
        self.record['geometry'].update(total_area_px=width*height,
            regions=[{'area_px': width*height, 'bbox_xywh_px': [x, y, width, height],
                      'bbox_diagonal_px': math.hypot(width, height)}])

    def grade(self):
        return grade_record(self.record, self.policy)

    def test_repeatability_and_no_llm_override(self):
        before = deepcopy(self.record)
        a = self.grade()
        self.record['llama_advice'] = {'grade': 'REJECT'}
        self.assertEqual(a, self.grade())
        self.assertEqual(a['grade'], 'PASS')
        self.assertFalse(a['physical_scale_calibrated'])
        self.assertFalse(a['industrial_certified'])
        self.assertEqual(a['units'], 'pixels')
        self.record.pop('llama_advice')
        self.assertEqual(self.record, before)

    def test_size_and_length_rules_on_explicit_policy_fixtures(self):
        self.anomaly()
        self.assertEqual(self.grade()['grade'], 'REVIEW')
        self.anomaly(width=70, height=70)  # Exactly 1% of full 700x700 frame.
        self.assertEqual(self.grade()['grade'], 'REJECT')
        self.anomaly(width=140, height=1)  # Small area, long extent.
        self.assertEqual(self.grade()['grade'], 'REJECT')

    def test_location_and_policy_fingerprint(self):
        self.anomaly()
        original = self.grade()
        self.policy['critical_zones'] = [{'name': 'TEST_ONLY_ZONE', 'bbox_xywh_px': [100, 100, 30, 30]}]
        self.policy['policy_id'] = 'test-zone-policy'
        self.assertEqual(self.grade()['grade'], 'REVIEW')
        self.anomaly(x=105, y=105)
        graded = self.grade()
        self.assertEqual(graded['grade'], 'REJECT')
        self.assertNotEqual(original['policy_sha256'], graded['policy_sha256'])
        self.assertEqual(graded['evidence']['critical_zone_bbox_overlaps'], ['TEST_ONLY_ZONE'])

    def test_unknown_unseen_low_confidence_never_auto_rejected(self):
        self.anomaly(width=200, height=200)
        for name, status in [('unknown', 'NO_VALIDATED_SUBTYPE_CLASSIFIER'),
                             ('unseen', 'VALIDATED_SAME_DOMAIN_RELIABLE'), ('scratch', 'LOW_CONFIDENCE')]:
            with self.subTest(name=name, status=status):
                self.record.update(predicted_defect_type=name, type_status=status)
                self.assertEqual(self.grade()['grade'], 'REVIEW')

    def test_errors_invalid_geometry_and_nonfinite_score(self):
        self.record['status'] = 'ERROR'
        self.assertEqual(self.grade()['grade'], 'REVIEW')
        self.record['status'] = 'ANALYZED'
        self.record['score'] = float('nan')
        self.assertEqual(self.grade()['reason'], 'INVALID_SCORE')
        self.record['score'] = .1
        self.record['geometry']['total_area_px'] = -1
        self.assertEqual(self.grade()['grade'], 'REVIEW')
        self.anomaly()
        self.record['geometry']['regions'][0]['bbox_diagonal_px'] = 3.
        self.assertEqual(self.grade()['grade'], 'REVIEW')

    def test_threshold_equality_and_review_band(self):
        self.record['score'] = self.policy['image_threshold']
        self.assertEqual(self.grade()['grade'], 'REVIEW')
        self.record['score'] = self.policy['image_threshold'] * .9
        self.assertEqual(self.grade()['grade'], 'REVIEW')
        self.record['model']['score_policy_sha256'] = 'wrong'
        self.assertEqual(self.grade()['reason'], 'MODEL_OR_SCORE_POLICY_MISMATCH')

    def test_no_mm_certification_or_invalid_zone(self):
        for key, value in [('units', 'mm'), ('industrial_certified', True), ('reject_response_area_fraction', float('nan'))]:
            p = deepcopy(self.policy); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_policy(p)
        self.policy['critical_zones'] = [{'name': 'bad', 'bbox_xywh_px': [699, 0, 2, 3]}]
        with self.assertRaises(ValueError):
            validate_policy(self.policy)

    def test_camera_domain_cannot_inherit_benchmark_pass_or_reject(self):
        self.record['source']['mode'] = 'LIVE_CAMERA'
        self.assertEqual(self.grade()['reason'], 'CAMERA_DOMAIN_UNVALIDATED')
        self.assertEqual(self.grade()['grade'], 'REVIEW')
        self.anomaly(width=200, height=200)
        self.assertEqual(self.grade()['reason'], 'CAMERA_DOMAIN_UNVALIDATED')
        self.assertEqual(self.grade()['grade'], 'REVIEW')
        self.record['source']['mode'] = 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC'
        self.assertEqual(self.grade()['reason'], 'CAMERA_DOMAIN_UNVALIDATED')
        self.assertIn('no physical part was inspected', self.grade()['operator_action'])

    def test_user_upload_cannot_inherit_benchmark_pass_or_reject(self):
        self.record['source']['mode'] = 'USER_UPLOADED_IMAGE_DIAGNOSTIC'
        self.assertEqual(self.grade()['reason'], 'UPLOADED_IMAGE_DOMAIN_UNVALIDATED')
        self.assertEqual(self.grade()['grade'], 'REVIEW')
        self.anomaly(width=200, height=200)
        self.assertEqual(self.grade()['reason'], 'UPLOADED_IMAGE_DOMAIN_UNVALIDATED')
        self.assertEqual(self.grade()['grade'], 'REVIEW')


if __name__ == '__main__':
    unittest.main()
