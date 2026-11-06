"""Storage/transaction fixtures are not new model or industrial acceptance evidence."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from uuid import uuid4

from forgesight.grade import grade_record
from forgesight.inspection import CaptureGate, inspect_one_image
from forgesight.storage import (APPLICATION_ID, InspectionStore, PartAlreadyInspected,
                                RecordConflict, new_correction)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database = self.root/'history.sqlite3'
        self.store = InspectionStore(self.database)
        self.records = json.loads(Path('manifests/graded_inspection_replay_v1.json').read_text())['records']

    def fixture(self, index=0):
        return deepcopy(self.records[index])

    def test_version_integrity_and_real_record_metadata_roundtrip(self):
        expected = self.fixture()
        self.assertTrue(self.store.save_inspection(expected))
        self.assertEqual(self.store.get_inspection(expected['inspection_id']), expected)
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute('PRAGMA user_version').fetchone()[0], 1)
            self.assertEqual(connection.execute('PRAGMA application_id').fetchone()[0], APPLICATION_ID)
        checks = self.store.integrity()
        self.assertEqual(checks['integrity_check'], ['ok'])
        self.assertEqual(checks['foreign_key_check'], [])
        self.assertTrue(checks['foreign_keys_enabled'])
        self.assertEqual(checks['synchronous'], 2)

    def test_exact_retry_reordered_keys_and_conflicting_content(self):
        original = self.fixture()
        self.store.save_inspection(original)
        self.assertFalse(self.store.save_inspection(dict(reversed(list(original.items())))))
        changed = deepcopy(original)
        changed['timings']['test_change'] = 1
        with self.assertRaises(RecordConflict):
            self.store.save_inspection(changed)
        self.assertEqual(self.store.get_inspection(original['inspection_id']), original)
        self.assertEqual(self.store.summary()['inspection_count'], 1)

    def test_part_identity_survives_restart_and_identical_images_are_distinct(self):
        first, other = self.fixture(), self.fixture(2)
        self.assertEqual(first['source']['sha256'], other['source']['sha256'])
        self.store.save_inspection(first)
        reopened = InspectionStore(self.database, create=False)
        gate = CaptureGate(already_recorded=reopened.has_part)
        with self.assertRaises(ValueError):
            gate.arm(first['part_id'])
        gate.arm(other['part_id'])
        reopened.save_inspection(other)
        duplicate = deepcopy(first)
        duplicate['inspection_id'] = str(uuid4())
        with self.assertRaises(PartAlreadyInspected):
            reopened.save_inspection(duplicate)
        self.assertEqual(reopened.summary()['distinct_part_count'], 2)

    def test_all_grades_errors_filters_and_stable_pagination(self):
        for row in self.records:
            self.store.save_inspection(row)
        synthetic = self.fixture(1)
        synthetic.update(inspection_id=str(uuid4()), part_id='SYNTHETIC_REJECT_STORAGE_FIXTURE',
                         predicted_defect_type='bent', type_status='VALIDATED_SAME_DOMAIN_RELIABLE')
        synthetic['source']['mode'] = 'SYNTHETIC_STORAGE_RULE_FIXTURE'
        synthetic['decision'] = grade_record(synthetic, synthetic['decision']['policy_snapshot'])
        self.assertEqual(synthetic['decision']['grade'], 'REJECT')
        self.store.save_inspection(synthetic)
        summary = self.store.summary()
        self.assertEqual(summary['original_grade_counts'], {'PASS': 2, 'REVIEW': 2, 'REJECT': 1})
        self.assertEqual(summary['processing_status_counts'], {'ANALYZED': 4, 'ERROR': 1})
        self.assertEqual(len(self.store.history(grade='REVIEW')), 2)
        self.assertEqual(len(self.store.history(status='ERROR')), 1)
        self.assertEqual(self.store.history(limit=2) + self.store.history(limit=3, offset=2), self.store.history())
        self.assertEqual(len(self.store.history(part_id=self.records[0]['part_id'])), 1)
        self.assertEqual(self.store.history(part_id="' OR 1=1 --"), [])
        self.assertEqual(self.store.get_inspection(str(uuid4())), None)
        for kwargs in [{'limit': 0}, {'offset': -1}, {'grade': 'ANY'}, {'status': 'OK'}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.store.history(**kwargs)

    def test_corrections_are_unverified_idempotent_and_preserve_counts(self):
        original = self.fixture()
        self.store.save_inspection(original)
        note = new_correction(original['inspection_id'], 'TEST_OPERATOR', 'Synthetic storage test claim; not a real factory finding.', 'REVIEW')
        self.assertTrue(self.store.save_correction(note))
        self.assertFalse(self.store.save_correction(note))
        changed = deepcopy(note); changed['reason'] = 'Changed claim'
        with self.assertRaises(RecordConflict):
            self.store.save_correction(changed)
        self.assertEqual(self.store.corrections(original['inspection_id']), [note])
        self.assertEqual(self.store.get_inspection(original['inspection_id']), original)
        self.assertEqual(self.store.summary()['original_grade_counts']['PASS'], 1)
        self.assertEqual(self.store.summary()['unverified_correction_count'], 1)
        bad = deepcopy(note); bad['correction_id'] = str(uuid4()); bad['verification_status'] = 'CONFIRMED'
        with self.assertRaises(ValueError):
            self.store.save_correction(bad)
        orphan = new_correction(str(uuid4()), 'TEST', 'Orphan claim')
        with self.assertRaises(ValueError):
            self.store.save_correction(orphan)

    def test_rollback_and_immutability_guards(self):
        record = self.fixture()
        self.store.save_inspection(record)
        with sqlite3.connect(self.database) as connection:
            for sql in ['UPDATE inspections SET grade = ?', 'DELETE FROM inspections']:
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(sql, ('REJECT',) if '?' in sql else ())
        conflicting = deepcopy(record); conflicting['inspection_id'] = str(uuid4())
        with self.assertRaises(PartAlreadyInspected):
            self.store.save_inspection(conflicting)
        self.assertEqual(self.store.summary()['inspection_count'], 1)
        self.assertEqual(self.store.get_inspection(record['inspection_id']), record)

    def test_reject_invalid_metadata_before_any_partial_write(self):
        cases = []
        for key, value in [('schema_version', 2), ('score', float('nan')), ('inspection_id', 'bad'),
                           ('part_id', ' padded '), ('timestamp_utc', '2026-11-06T12:00:00')]:
            row = self.fixture(); row[key] = value; cases.append(row)
        row = self.fixture(); row['decision']['policy_sha256'] = '0'*64; cases.append(row)
        row = self.fixture(3); row['decision']['grade'] = 'PASS'; cases.append(row)
        for row in cases:
            with self.subTest(part=row['part_id']), self.assertRaises(ValueError):
                self.store.save_inspection(row)
        self.assertEqual(self.store.summary()['inspection_count'], 0)

    def test_refuse_foreign_unversioned_and_future_databases_without_changing_them(self):
        for kind in ['foreign', 'future', 'unversioned']:
            path = self.root/(kind+'.sqlite3')
            with sqlite3.connect(path) as connection:
                connection.execute('CREATE TABLE private_data(value TEXT)')
                connection.execute("INSERT INTO private_data VALUES ('preserve')")
                if kind == 'future':
                    connection.execute(f'PRAGMA application_id={APPLICATION_ID}')
                    connection.execute('PRAGMA user_version=99')
            before = path.read_bytes()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                InspectionStore(path)
            self.assertEqual(path.read_bytes(), before)

    def test_separate_process_reads_committed_history_and_missing_path_fails(self):
        record = self.fixture()
        self.store.save_inspection(record)
        command = [sys.executable, '-B', '-m', 'forgesight.storage', '--db', str(self.database), 'summary']
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['inspection_count'], 1)
        missing = self.root/'missing.sqlite3'
        result = subprocess.run(command[:5]+[str(missing), 'summary'], capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(missing.exists())

    def test_two_connections_retry_same_record_once(self):
        other = InspectionStore(self.database, create=False)
        record = self.fixture()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(store.save_inspection, deepcopy(record)) for store in (self.store, other)]
            self.assertEqual(sorted(f.result() for f in futures), [False, True])
        self.assertEqual(self.store.summary()['inspection_count'], 1)

    def test_json_survives_failed_database_write_and_can_be_retried(self):
        reference = self.records[0]['model']
        predictor = SimpleNamespace(reference={'calibrated_checkpoint': {'sha256': reference['checkpoint_sha256']},
                                              'policy_file_sha256': reference['score_policy_sha256']},
                                    policy={'threshold': reference['image_threshold']}, device='test')
        def fail(record):
            raise sqlite3.OperationalError('TEST: database unavailable')
        failed_store = SimpleNamespace(has_part=lambda part: False, save_inspection=fail)
        with self.assertRaisesRegex(RuntimeError, 'JSON is preserved'):
            inspect_one_image(self.root/'missing.png', 'recoverable-error', predictor, self.root/'out', store=failed_store)
        files = list((self.root/'out').glob('*.json'))
        self.assertEqual(len(files), 1)
        record = json.loads(files[0].read_text())
        self.assertEqual(record['status'], 'ERROR')
        self.assertEqual(record['decision']['grade'], 'REVIEW')
        self.store.save_inspection(record)
        with self.assertRaises(PartAlreadyInspected):
            inspect_one_image(self.root/'missing.png', 'recoverable-error', predictor, self.root/'out', store=self.store)
        self.assertEqual(len(list((self.root/'out').glob('*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
