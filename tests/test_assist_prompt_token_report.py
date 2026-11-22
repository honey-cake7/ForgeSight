"""Host tests for the read-only prompt token report, using a stub counter.
The pinned-tokenizer counts themselves are device evidence; this pins plumbing."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from forgesight.assist_contract import DEFAULT_GUIDANCE
from forgesight.assist_prompt_token_report import camera_variant, measure_case

PROFILE = {'max_seq_length': 100_000, 'max_context_length': 200_000,
           'model_id': 'stub'}


def fixture_record():
    return deepcopy(json.loads(Path('manifests/operator_app_verification_v1.json').read_text())
                    ['browser_proof']['inspections'][1]['record'])


class TokenReportTests(unittest.TestCase):
    def setUp(self):
        self.guidance = json.loads(DEFAULT_GUIDANCE.read_text())
        self.count = len  # deterministic stub, not a tokenizer

    def test_eligible_row_reports_consistent_token_math(self):
        row = measure_case('case', fixture_record(), None, self.guidance, self.count, PROFILE)
        self.assertEqual(row['eligibility'], 'ELIGIBLE')
        self.assertEqual(row['input_kind'], 'RECORD_FILE')
        tokens = row['tokens']
        self.assertEqual(tokens['full_formatted'],
                         tokens['prompt_text'] + tokens['system_text'] + tokens['template_overhead'])
        self.assertGreater(tokens['template_overhead'], 0)
        self.assertTrue(row['budget']['prefill_ok'])
        self.assertTrue(row['budget']['context_ok'])
        self.assertEqual(row['budget']['max_new_tokens'], 64)

    def test_prefill_bound_is_strict(self):
        tight = dict(PROFILE, max_seq_length=1)
        row = measure_case('case', fixture_record(), None, self.guidance, self.count, tight)
        self.assertFalse(row['budget']['prefill_ok'])

    def test_refused_record_is_reported_not_raised(self):
        record = fixture_record()
        record['decision']['grade'] = 'PASS'
        row = measure_case('case', record, None, self.guidance, self.count, PROFILE)
        self.assertEqual(row['eligibility'], 'UNSUPPORTED_TYPE_OR_GRADE')
        self.assertNotIn('tokens', row)

    def test_camera_variant_is_regraded_and_labeled_synthetic(self):
        variant = camera_variant(fixture_record())
        self.assertEqual(variant['decision']['reason'], 'CAMERA_DOMAIN_UNVALIDATED')
        row = measure_case('cam', variant, None, self.guidance, self.count, PROFILE, synthetic=True)
        self.assertEqual(row['eligibility'], 'ELIGIBLE')
        self.assertEqual(row['input_kind'], 'SYNTHETIC_MEASUREMENT_DERIVED')
        self.assertEqual(row['guide_id'], 'G3')
        # The camera prompt carries the disclosure line, so it must be larger.
        saved = measure_case('saved', fixture_record(), None, self.guidance, self.count, PROFILE)
        self.assertGreater(row['tokens']['full_formatted'], saved['tokens']['full_formatted'])

    def test_record_hash_is_reported_when_path_given(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'record.json'
            path.write_text(json.dumps(fixture_record()))
            row = measure_case('case', json.loads(path.read_text()), path,
                               self.guidance, self.count, PROFILE)
            import hashlib
            self.assertEqual(row['record_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
