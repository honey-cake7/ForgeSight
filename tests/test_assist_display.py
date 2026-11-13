"""Synthetic report fixtures check binding; they are not model proof."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from forgesight.assist_contract import canonical_hash
from forgesight.assist_display import advice_for


class AdviceDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = json.loads(Path('manifests/graded_inspection_replay_v1.json').read_text())['records'][0]
        self.folder = self.root / self.record['inspection_id']
        self.report = {
            'inspection_id': self.record['inspection_id'],
            'inspection_sha256': canonical_hash(self.record),
            'decision_sha256': canonical_hash(self.record['decision']),
            'authoritative_grade': self.record['decision']['grade'],
            'source_mode': self.record['source']['mode'],
            'physical_part_inspected': False,
            'inspection_and_sqlite_unchanged': True,
            'status': 'JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW',
            'native': {'status': 'JETSON_EXECUTORCH_LLAMA_GENERATION_VERIFIED'},
            'advisory': {
                'inspection_id': self.record['inspection_id'],
                'inspection_sha256': canonical_hash(self.record),
                'decision_sha256': canonical_hash(self.record['decision']),
                'authoritative_grade': self.record['decision']['grade'],
                'quality_grade_authority': 'DETERMINISTIC_RULES_ONLY',
                'jetson_verified': True,
                'source_mode': self.record['source']['mode'],
                'guidance_approval': 'PROTOTYPE_NOT_FACTORY_APPROVED',
                'response': {'generated_note': '[E1] Fixture note [G1]', 'evidence_ids': ['E1', 'G1']}}}

    def write_report(self, report):
        self.folder.mkdir(exist_ok=True)
        (self.folder / 'advice_report.json').write_text(json.dumps(report))

    def test_separate_states_and_unchanged_grade(self):
        original = deepcopy(self.record)
        self.assertEqual(advice_for(self.record, None)['status'], 'NOT_CONNECTED')
        self.assertEqual(advice_for(self.record, self.root)['status'], 'NOT_REQUESTED')
        self.folder.mkdir()
        self.assertEqual(advice_for(self.record, self.root)['status'], 'PENDING')
        self.write_report(self.report)
        shown = advice_for(self.record, self.root)
        self.assertEqual(shown['status'], 'GENERATED_REQUIRES_HUMAN_REVIEW')
        self.assertEqual(shown['advisory']['note'], '[E1] Fixture note [G1]')
        self.assertEqual(self.record, original)

    def test_persistent_worker_provenance_displays(self):
        """Regression: a successful warm worker report must display; before
        the fix the native dict carried no worker status and every warm note
        showed ADVICE_UNAVAILABLE."""
        report = deepcopy(self.report)
        report['llama_runtime_mode'] = 'PERSISTENT_WORKER'
        report['native'] = {'status': 'GENERATED_BY_PERSISTENT_WORKER',
                            'worker_mode': 'PERSISTENT_PROCESS',
                            'worker_started_utc': '2026-11-14T00:00:00Z'}
        self.write_report(report)
        shown = advice_for(self.record, self.root)
        self.assertEqual(shown['status'], 'GENERATED_REQUIRES_HUMAN_REVIEW')
        self.assertEqual(shown['advisory']['note'], '[E1] Fixture note [G1]')

    def test_unknown_native_status_withheld(self):
        report = deepcopy(self.report)
        report['native'] = {'status': 'SOME_FUTURE_UNREVIEWED_MODE'}
        self.write_report(report)
        self.assertEqual(advice_for(self.record, self.root)['status'], 'ADVICE_UNAVAILABLE')

    def test_mismatch_and_unavailable_withhold_note(self):
        for changed in ('inspection_sha256', 'decision_sha256', 'authoritative_grade', 'native', 'status'):
            with self.subTest(changed=changed):
                report = deepcopy(self.report)
                report[changed] = 'wrong'
                self.write_report(report)
                result = advice_for(self.record, self.root)
                self.assertEqual(result['status'], 'ADVICE_UNAVAILABLE')
                self.assertIsNone(result['advisory'])
        (self.folder / 'advice_report.json').write_text('{invalid')
        self.assertEqual(advice_for(self.record, self.root)['status'], 'ADVICE_UNAVAILABLE')
        report = deepcopy(self.report)
        report['advisory']['response']['generated_note'] = 'Unsupported advice without citations'
        self.write_report(report)
        self.assertEqual(advice_for(self.record, self.root)['status'], 'ADVICE_UNAVAILABLE')
        report = deepcopy(self.report)
        report['advisory']['authoritative_grade'] = 'REVIEW'
        self.write_report(report)
        self.assertEqual(advice_for(self.record, self.root)['status'], 'ADVICE_UNAVAILABLE')


if __name__ == '__main__':
    unittest.main()
