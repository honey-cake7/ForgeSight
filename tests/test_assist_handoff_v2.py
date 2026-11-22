"""Structured hand-off v2: verified evidence reaches the prompt, bounded composition out."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from forgesight.assist_contract import (DEFAULT_GUIDANCE, OUTPUT_POLICY_VERSION, parse_advisory,
                                        prepare_advisory)
from forgesight.severity import SEVERITY_MAPPING_VERSION


VALID = ("An anomaly is flagged; type and cause are unknown [E1]. "
         "Severity HIGH, prototype pixel-response severity, not physical-damage severity. "
         "Anomaly score 0.682809 vs threshold 0.227766. "
         "Response area 9015 px across 2 regions; largest region width 139 px height 116 px "
         "at x 512 y 361 in the right area of the frame. "
         "Hold for human review; check clarity and lighting; compare the highlighted "
         "region with a known-good part [G1].")


class HandoffV2Tests(unittest.TestCase):
    def setUp(self):
        self.record = deepcopy(json.loads(Path('manifests/operator_app_verification_v1.json').read_text())
                               ['browser_proof']['inspections'][1]['record'])
        self.guidance = json.loads(DEFAULT_GUIDANCE.read_text())
        self.context = prepare_advisory(self.record, self.guidance)

    def test_prompt_carries_verified_evidence(self):
        prompt = self.context['prompt_text']
        for fragment in ('0.682809', '0.227766', '9015', '1.840', '2 regions',
                         '139', '116', '512', '361', 'Severity HIGH',
                         'not physical damage', 'not a probability'):
            self.assertIn(fragment, prompt)
        self.assertTrue(self.context['score_and_geometry_in_prompt'])
        self.assertFalse(self.context['image_supplied_to_llama'])
        self.assertNotIn(self.record['inspection_id'], prompt)

    def test_evidence_records_severity_and_location(self):
        evidence = self.context['evidence']['E1']
        severity = evidence['pixel_response_severity']
        self.assertEqual(severity['severity'], 'HIGH')
        self.assertEqual(severity['mapping_version'], SEVERITY_MAPPING_VERSION)
        self.assertIn('not physical-damage severity', severity['qualifier'])
        self.assertEqual(evidence['largest_region']['location'], 'right')
        # The deterministic decision severity field is preserved separately.
        self.assertEqual(evidence['severity'], self.record['decision']['severity'])

    def test_composed_note_accepted_and_reordered(self):
        self.assertEqual(parse_advisory(VALID, self.context)['status'], 'GENERATED_REQUIRES_HUMAN_REVIEW')
        reordered = ("Hold for human review; check clarity and lighting; compare the "
                     "highlighted region with a known-good part [G1]. Severity HIGH. "
                     "Response area 9015 px. An anomaly is flagged; type and cause are unknown [E1].")
        self.assertEqual(parse_advisory(reordered, self.context)['output_policy_version'],
                         OUTPUT_POLICY_VERSION)

    def test_unsupplied_or_rounded_measurements_rejected(self):
        for bad in (VALID.replace('9015', '9000'), VALID.replace('0.682809', '0.683'),
                    VALID.replace('9015 px', '9015.0 px')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_advisory(bad, self.context)

    def test_severity_omitted_or_contradicted_rejected(self):
        with self.assertRaises(ValueError):
            parse_advisory(VALID.replace('Severity HIGH, prototype pixel-response severity, '
                                         'not physical-damage severity. ', ''), self.context)
        with self.assertRaises(ValueError):
            parse_advisory(VALID.replace('Severity HIGH', 'Severity LOW'), self.context)

    def test_no_measurement_at_all_rejected(self):
        bare = ("An anomaly is flagged; type and cause are unknown [E1]. Severity HIGH. "
                "Hold for human review; check clarity and lighting; compare the "
                "highlighted region with a known-good part [G1].")
        with self.assertRaises(ValueError):
            parse_advisory(bare, self.context)

    def test_cause_or_type_assertion_rejected(self):
        for claim in ('The cause is lighting.', 'The type is scratch.', 'The cause was lighting.'):
            with self.subTest(claim=claim), self.assertRaises(ValueError):
                parse_advisory(VALID + ' ' + claim, self.context)

    def test_unknown_type_and_cause_still_required(self):
        with self.assertRaises(ValueError):
            parse_advisory(VALID.replace('type and cause are unknown', 'type is clear'), self.context)

    def test_cause_assertion_via_alternate_copulas_rejected(self):
        # The guard must not depend on the copula "is/was/are": any clause that
        # concludes cause/type with anything but unknown is an assertion.
        for claim in ('The cause remains lighting.', 'The cause should be lighting.',
                      'The cause has been lighting.', 'The cause detected lighting.',
                      'Lighting is the cause.'):
            with self.subTest(claim=claim), self.assertRaises(ValueError):
                parse_advisory(VALID + ' ' + claim, self.context)

    def test_unknown_cause_statement_variants_accepted(self):
        ok = VALID.replace('type and cause are unknown', 'type and cause remain unknown')
        self.assertEqual(parse_advisory(ok, self.context)['status'],
                         'GENERATED_REQUIRES_HUMAN_REVIEW')

    def test_defect_wording_outside_compact_prompt_is_refused(self):
        # The v4 compact prompt carries 'type and cause unknown' without the word
        # 'defect', so a note adding it asserts a word the evidence never supplied.
        with self.assertRaises(ValueError):
            parse_advisory(VALID.replace('type and cause are unknown',
                                         'defect type and cause unknown'), self.context)

    def test_supplied_value_on_wrong_quantity_rejected(self):
        # 9015 is the supplied response area; it may not be presented as the score.
        with self.assertRaises(ValueError):
            parse_advisory(VALID.replace('Anomaly score 0.682809', 'Anomaly score 9015'),
                           self.context)

    def test_unit_injection_rejected(self):
        # A percent sign changes the quantity: area is pixels, the score is unitless.
        with self.assertRaises(ValueError):
            parse_advisory(VALID.replace('Response area 9015 px',
                                         'Response area 9015% of the frame'), self.context)
        with self.assertRaises(ValueError):
            parse_advisory(VALID.replace('Anomaly score 0.682809',
                                         'Anomaly score 0.682809%'), self.context)

    def test_checklist_scrambled_rejected(self):
        # Same words, different order: the procedure meaning is not preserved.
        scrambled = VALID.replace(
            'Hold for human review; check clarity and lighting; compare the '
            'highlighted region with a known-good part',
            'Compare lighting with a known-good part; check image clarity and the '
            'highlighted region; hold for human review')
        with self.assertRaises(ValueError):
            parse_advisory(scrambled, self.context)

    def test_checklist_restatement_with_interleaved_connectors_accepted(self):
        interleaved = VALID.replace('check image clarity', 'please check image clarity')
        self.assertEqual(parse_advisory(interleaved, self.context)['status'],
                         'GENERATED_REQUIRES_HUMAN_REVIEW')

    def test_v1_guidance_is_no_longer_recognized(self):
        old = json.loads(Path('guidance/prototype_inspection_v1.json').read_text())
        with self.assertRaises(ValueError):
            prepare_advisory(self.record, old)


if __name__ == '__main__':
    unittest.main()
