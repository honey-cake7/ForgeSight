"""API contracts use saved real records; genuine browser/model proof is separate."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from threading import Event
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch
import hashlib

from PIL import Image

from fastapi.testclient import TestClient

from forgesight.operator_app import create_app
from forgesight.storage import InspectionStore


class OperatorAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database = self.root/'history.sqlite3'
        self.record = json.loads(Path('manifests/graded_inspection_replay_v1.json').read_text())['records'][0]

    def client(self, factory=None, advice_root=None):
        if factory is None:
            def factory(device):
                raise RuntimeError('TEST: model intentionally unavailable')
        return TestClient(create_app(database=self.database, output=self.root/'output', data_root=self.root/'missing_data',
                                    advice_root=advice_root,
                                    predictor_factory=factory), base_url='http://127.0.0.1')

    def test_separate_advice_endpoint_does_not_change_inspection(self):
        InspectionStore(self.database).save_inspection(self.record)
        path = '/api/inspections/' + self.record['inspection_id'] + '/advice'
        with self.client() as client:
            self.assertEqual(client.get(path).json()['status'], 'NOT_CONNECTED')
        advice_root = self.root/'advice'
        with self.client(advice_root=advice_root) as client:
            self.assertEqual(client.get(path).json()['status'], 'NOT_REQUESTED')
            (advice_root/self.record['inspection_id']).mkdir(parents=True)
            self.assertEqual(client.get(path).json()['status'], 'PENDING')
            self.assertEqual(client.get('/api/inspections/'+self.record['inspection_id']).json()['record'], self.record)

    def metadata_only_predictor(self, device):
        # No detector method or generated prediction: missing-file error path only.
        model = self.record['model']
        return SimpleNamespace(reference={'calibrated_checkpoint': {'sha256': model['checkpoint_sha256']},
                                          'policy_file_sha256': model['score_policy_sha256']},
                               policy={'threshold': model['image_threshold']}, device=device)

    def test_empty_ui_configuration_and_no_holdout_catalog(self):
        with self.client() as client:
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.get('/assets/app.js').status_code, 200)
            self.assertEqual(client.get('/api/latest').json(), {'inspection': None})
            self.assertEqual(client.get('/api/history').json()['items'], [])
            config = client.get('/api/config').json()
            self.assertEqual(len(config['samples']), 5)
            self.assertEqual(config['predicted_defect_type'], 'unknown')
            self.assertFalse(config['camera_connected'])
            self.assertEqual(client.post('/api/config', json={}).status_code, 405)
            self.assertEqual(client.get('/docs').status_code, 404)

    def test_unavailable_model_keeps_history_visible_and_refuses_inspection(self):
        InspectionStore(self.database).save_inspection(self.record)
        with self.client() as client:
            self.assertEqual(client.get('/api/health').status_code, 503)
            self.assertEqual(client.get('/api/history').json()['items'][0]['record'], self.record)
            response = client.post('/api/inspect', json={'part_id':'new', 'sample_id':'normal-fit-000'})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(client.get('/api/summary').json()['inspection_count'], 1)

    def test_missing_image_is_real_saved_error_review_and_duplicate_is_blocked(self):
        with self.client(self.metadata_only_predictor) as client:
            payload = {'part_id':'API-ERROR-001', 'sample_id':'normal-fit-000'}
            response = client.post('/api/inspect', json=payload)
            self.assertEqual(response.status_code, 201, response.text)
            saved = response.json()['record']
            self.assertEqual(saved['status'], 'ERROR')
            self.assertEqual(saved['error']['type'], 'FileNotFoundError')
            self.assertEqual(saved['decision']['grade'], 'REVIEW')
            self.assertEqual(client.get('/api/latest').json()['inspection']['record'], saved)
            self.assertEqual(client.post('/api/inspect', json=payload).status_code, 409)
            self.assertEqual(client.get('/api/summary').json()['inspection_count'], 1)
        with self.client(self.metadata_only_predictor) as reopened:
            self.assertEqual(reopened.get('/api/latest').json()['inspection']['record'], saved)
            self.assertEqual(reopened.post('/api/inspect', json=payload).status_code, 409)

    def test_invalid_requests_cross_origin_and_path_parameters(self):
        with self.client(self.metadata_only_predictor) as client:
            for payload in [{'part_id':' spaced ', 'sample_id':'normal-fit-000'},
                            {'part_id':'x', 'sample_id':'../../.env'},
                            {'part_id':'x', 'sample_id':'test/scratch/000.png'},
                            {'part_id':'x', 'sample_id':'normal-fit-000', 'threshold':0}]:
                with self.subTest(payload=payload):
                    self.assertEqual(client.post('/api/inspect', json=payload).status_code, 422)
            self.assertEqual(client.post('/api/inspect', json={'part_id':'x','sample_id':'normal-fit-000'},
                                         headers={'origin':'https://example.invalid'}).status_code, 403)
            self.assertEqual(client.get('/api/inspections/not-a-uuid').status_code, 422)
            self.assertEqual(client.get('/api/history?limit=0').status_code, 422)
            self.assertEqual(client.get('/api/history?grade=ANY').status_code, 422)
            self.assertEqual(client.get('/api/summary').json()['inspection_count'], 0)

    def test_history_pagination_and_no_arbitrary_file_serving(self):
        store = InspectionStore(self.database)
        rows = json.loads(Path('manifests/graded_inspection_replay_v1.json').read_text())['records']
        for record in rows:
            store.save_inspection(record)
        with self.client() as client:
            page = client.get('/api/history?limit=2').json()
            self.assertEqual(len(page['items']), 2)
            self.assertTrue(page['pagination']['has_next'])
            self.assertEqual(len(client.get('/api/history?grade=PASS').json()['items']), 2)
            self.assertEqual(len(client.get('/api/history?status=ERROR').json()['items']), 1)
            response = client.get('/api/inspections/'+rows[0]['inspection_id']+'/source')
            self.assertEqual(response.status_code, 404)  # Outside the test data root.
            self.assertEqual(client.get('/api/inspections/'+rows[0]['inspection_id']+'/score_map').status_code, 422)

    def test_media_checksum_and_missing_image_are_explicit(self):
        # Synthetic image fixture tests HTTP/file integrity, not detector accuracy.
        data = self.root/'missing_data'; data.mkdir()
        image_path = data/'fixture.png'
        Image.new('RGB',(8,8),'white').save(image_path)
        record = deepcopy(self.record)
        record['source'].update(path=str(image_path),sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                                mode='SYNTHETIC_MEDIA_TEST_FIXTURE')
        InspectionStore(self.database).save_inspection(record)
        url='/api/inspections/'+record['inspection_id']+'/source'
        with self.client() as client:
            response=client.get(url)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.content,image_path.read_bytes())
            Image.new('RGB',(8,8),'black').save(image_path)
            self.assertEqual(client.get(url).status_code,409)
            image_path.unlink()
            self.assertEqual(client.get(url).status_code,404)

    def test_busy_service_refuses_second_attempt_without_queueing_inference(self):
        entered, release = Event(), Event()
        def unavailable(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('TEST timed out')
            raise OSError('TEST output failure; no detector success')
        with self.client(self.metadata_only_predictor) as client, patch('forgesight.operator_app.inspect_one_image',unavailable):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending=pool.submit(client.post,'/api/inspect',json={'part_id':'busy-A','sample_id':'normal-fit-000'})
                self.assertTrue(entered.wait(5))
                try:
                    response=client.post('/api/inspect',json={'part_id':'busy-B','sample_id':'development-1'})
                    self.assertEqual(response.status_code,409)
                    self.assertIn('running',response.json()['detail'])
                finally:
                    release.set()
                self.assertEqual(pending.result(timeout=5).status_code,503)
            self.assertEqual(client.get('/api/summary').json()['inspection_count'],0)


if __name__ == '__main__':
    unittest.main()
