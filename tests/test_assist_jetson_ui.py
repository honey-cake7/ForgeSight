"""Replay queue state tests; fake workers are not device inference proof."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from forgesight.assist_jetson_ui import ReplayAdviceService
from forgesight.storage import InspectionStore


class ReplayAdviceServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = json.loads(Path('manifests/graded_inspection_replay_v1.json').read_text())['records'][0]
        self.record['source']['physical_part'] = False
        self.record_path = self.root / 'record.json'
        self.record_path.write_text(json.dumps(self.record))
        self.database = self.root / 'inspections.sqlite3'
        InspectionStore(self.database).save_inspection(self.record)

    def test_pending_busy_and_failure_keep_original_grade(self):
        entered, release = Event(), Event()
        def timeout_worker(_output):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test worker deadline')
            raise TimeoutError('simulated timeout')
        service = ReplayAdviceService(self.record_path, self.database,
                                      self.root / 'advice', timeout_worker)
        try:
            self.assertEqual(service.state()['advice']['status'], 'NOT_REQUESTED')
            self.assertEqual(service.request(), 'PENDING')
            self.assertTrue(entered.wait(1))
            self.assertEqual(service.request(), 'BUSY')
            self.assertEqual(service.state()['advice']['status'], 'PENDING')
            self.assertEqual(service.state()['inspection']['grade'], self.record['decision']['grade'])
            release.set()
            self.assertEqual(service.future.result(timeout=2)['status'], 'ADVICE_UNAVAILABLE')
            self.assertEqual(service.state()['advice']['status'], 'ADVICE_UNAVAILABLE')
            self.assertEqual(service.request(), 'ALREADY_REQUESTED')
            self.assertEqual(InspectionStore(self.database, create=False).get_inspection(self.record['inspection_id']), self.record)
        finally:
            release.set()
            service.close()


if __name__ == '__main__':
    unittest.main()
