"""v4 camera-advisory contract tests: scoped G3 entry, required unvalidated-domain
disclosure, and cross-citation refusals in both directions."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from forgesight.assist_contract import (CAMERA_SOURCE_MODES, DEFAULT_GUIDANCE,
                                        parse_advisory, prepare_advisory)
from forgesight.grade import grade_record


def saved_record():
    return deepcopy(json.loads(Path('manifests/operator_app_verification_v1.json').read_text())
                    ['browser_proof']['inspections'][1]['record'])


def camera_record():
    record = saved_record()
    record['source']['mode'] = sorted(CAMERA_SOURCE_MODES)[0]
    record['source']['physical_part'] = False
    record['decision'] = grade_record(record, record['decision']['policy_snapshot'])
    assert record['decision']['reason'] == 'CAMERA_DOMAIN_UNVALIDATED'
    assert record['decision']['severity'] == 'UNASSESSED'
    return record


CAMERA_NOTE = ("An anomaly is flagged; type and cause are unknown [E1]. "
               "The live camera frame is unvalidated, not a verified physical part [E1]. "
               "Severity HIGH, prototype pixel-response severity, not physical-damage severity. "
               "Anomaly score 0.682809 vs threshold 0.227766. "
               "Response area 9015 px across 2 regions; largest region width 139 px height 116 px "
               "at x 512 y 361 in the right area of the frame. "
               "Hold for human review; recapture with steady lighting and alignment; compare the "
               "highlighted region with a known-good part [G3].")


class CameraAdvisoryContractTests(unittest.TestCase):
    def setUp(self):
        self.guidance = json.loads(DEFAULT_GUIDANCE.read_text())
        self.camera_context = prepare_advisory(camera_record(), self.guidance)
        self.saved_context = prepare_advisory(saved_record(), self.guidance)

    def test_camera_record_gets_scoped_g3_entry_and_disclosure_line(self):
        prompt = self.camera_context['prompt_text']
        self.assertIn('[G3] Prototype checklist:', prompt)
        self.assertIn('camera domain unvalidated, not a verified physical part', prompt)
        self.assertIn('recapture with steady lighting and alignment', prompt)
        self.assertIn('State the camera domain is unvalidated.', self.camera_context['system_text'])
        self.assertEqual(self.camera_context['model_received_fields'][0], 'camera_domain_unvalidated')
        self.assertIn('G3', self.camera_context['evidence'])
        self.assertNotIn('G1', self.camera_context['evidence'])

    def test_saved_image_record_still_gets_g1_without_camera_line(self):
        prompt = self.saved_context['prompt_text']
        self.assertIn('[G1] Prototype checklist:', prompt)
        self.assertNotIn('camera', prompt)
        self.assertNotIn('camera_domain_unvalidated', self.saved_context['model_received_fields'])

    def test_legal_camera_note_is_accepted(self):
        result = parse_advisory(CAMERA_NOTE, self.camera_context)
        self.assertEqual(result['authoritative_grade'], 'REVIEW')
        self.assertEqual(result['status'], 'GENERATED_REQUIRES_HUMAN_REVIEW')

    def test_camera_note_citing_g1_is_refused(self):
        with self.assertRaises(ValueError):
            parse_advisory(CAMERA_NOTE.replace('[G3]', '[G1]'), self.camera_context)

    def test_camera_note_without_unvalidated_disclosure_is_refused(self):
        omitted = CAMERA_NOTE.replace(
            'The live camera frame is unvalidated, not a verified physical part [E1]. ', '')
        with self.assertRaises(ValueError):
            parse_advisory(omitted, self.camera_context)

    def test_saved_image_note_citing_g3_is_refused(self):
        cross = ("An anomaly is flagged; type and cause are unknown [E1]. "
                 "Severity HIGH, prototype pixel-response severity, not physical-damage severity. "
                 "Anomaly score 0.682809 vs threshold 0.227766. "
                 "Response area 9015 px across 2 regions; largest region width 139 px height 116 px "
                 "at x 512 y 361 in the right area of the frame. "
                 "Hold for human review; recapture with steady lighting and alignment; compare the "
                 "highlighted region with a known-good part [G3].")
        with self.assertRaises(ValueError):
            parse_advisory(cross, self.saved_context)

    def test_camera_deterministic_grade_is_never_overridden(self):
        # The advisory never upgrades the UNASSESSED deterministic severity.
        self.assertEqual(self.camera_context['authoritative_decision']['severity'], 'UNASSESSED')
        result = parse_advisory(CAMERA_NOTE, self.camera_context)
        self.assertEqual(result['authoritative_grade'], 'REVIEW')
        self.assertEqual(self.camera_context['authoritative_decision']['severity'], 'UNASSESSED')


if __name__ == '__main__':
    unittest.main()
