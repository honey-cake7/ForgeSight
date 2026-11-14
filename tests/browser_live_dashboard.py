"""Browser-only contract/failure tests. Fixtures do not prove model inference.

Run explicitly with the isolated Playwright QA environment; not part of the
device's runtime dependencies or ordinary unittest discovery.
"""
import base64
import copy
import json
from pathlib import Path
import unittest
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright, expect

WEB = Path(__file__).resolve().parents[1] / 'forgesight/web'
PIXEL = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5S8AAAAASUVORK5CYII=')
RESULT = dict(inspection_id='fixture-1', part_id='FIXTURE-001', status='ANALYZED',
              source_mode='USER_DECLARED_MVTEC_METAL_NUT_UPLOAD', grade='REJECT',
              score=0.68, threshold=0.23, candidate='ANOMALY_CANDIDATE',
              model_forward_ms=1600, response_area_px=10, source_area_px=100,
              response_regions=[dict(area_px=10, approximate_nut_area='annular face')],
              action='Fixture action', reason='FIXTURE_ONLY')


class DashboardBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page(viewport={'width': 1366, 'height': 768})
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.state = dict(camera=dict(ready=True, message='Live C270 preview'), busy=False,
                          latest=copy.deepcopy(RESULT), history=[], manual_reviews=[],
                          advice=dict(status='NOT_REQUESTED', advisory=None))
        self.posts = []
        self.pending = []
        self.hold_posts = False
        self.post_error = None
        self.bad_media = set()
        self.page.route('**/*', self.route)
        self.page.goto('http://127.0.0.1:18888/')
        expect(self.page.locator('#connection-status')).to_have_text('Station connected')

    def tearDown(self):
        self.page.close()
        self.assertEqual(self.errors, [])

    def route(self, route):
        path = urlsplit(route.request.url).path
        assets = {'/': ('jetson_live.html', 'text/html'), '/app.js': ('jetson_live.js', 'application/javascript'), '/style.css': ('jetson_live.css', 'text/css')}
        if path in assets:
            name, mime = assets[path]
            route.fulfill(content_type=mime, body=(WEB / name).read_bytes())
        elif route.request.method == 'POST':
            self.posts.append((path, route.request.post_data_buffer, route.request.headers))
            if self.hold_posts:
                self.pending.append(route)
            else:
                route.fulfill(status=409 if self.post_error else 202, content_type='application/json',
                              body=json.dumps({'error': self.post_error} if self.post_error else {'status': 'PROCESSING'}))
        elif path == '/api/state':
            route.fulfill(content_type='application/json', body=json.dumps(self.state))
        elif path in self.bad_media:
            route.fulfill(status=404, body='Missing test image')
        elif path in ('/api/frame.jpg', '/api/overlay.png', '/api/source.png'):
            route.fulfill(content_type='image/png', body=PIXEL)
        else:
            route.fulfill(status=404)

    def refresh(self):
        self.page.evaluate('refreshState()')

    def test_camera_failure_disables_capture_but_upload_remains_available(self):
        expect(self.page.locator('#camera-status')).to_have_text('LIVE')
        self.state['camera'] = dict(ready=False, message='Camera could not open')
        self.refresh()
        expect(self.page.locator('#camera-status')).to_have_text('ERROR')
        expect(self.page.locator('#preview')).to_be_hidden()
        expect(self.page.locator('#inspect')).to_be_disabled()
        self.page.locator('#diagnostic-file').set_input_files(dict(name='fixture.png', mimeType='image/png', buffer=PIXEL))
        self.page.locator('#diagnostic-id').fill('FIXTURE-UPLOAD')
        expect(self.page.locator('#diagnostic-submit')).to_be_enabled()
        self.state['camera'] = dict(ready=False, message='Starting camera')
        self.refresh()
        expect(self.page.locator('#camera-status')).to_have_text('CONNECTING')

    def test_processing_and_error_clear_previous_result_and_images(self):
        expect(self.page.locator('#overlay')).to_be_visible()
        self.state['busy'] = True
        self.state['latest'] = dict(status='PROCESSING', part_id='FIXTURE-002', source_mode='USER_UPLOADED_IMAGE_DIAGNOSTIC')
        self.refresh()
        expect(self.page.locator('#result-content')).to_be_hidden()
        expect(self.page.locator('#overlay')).to_be_hidden()
        expect(self.page.locator('#source-link')).to_be_hidden()
        self.state['busy'] = False
        self.state['latest'] = dict(status='ERROR', part_id='FIXTURE-002', error='Fixture failed')
        self.refresh()
        expect(self.page.locator('#grade')).to_have_text('ERROR')
        expect(self.page.locator('#overlay')).to_be_hidden()
        expect(self.page.locator('#review-submit')).to_be_disabled()
        self.state['latest'] = None
        self.refresh()
        expect(self.page.locator('#inspection-status')).to_have_text('AWAITING IMAGE')

    def test_grades_and_region_are_rendered_from_server(self):
        for index, grade in enumerate(['PASS', 'REVIEW', 'REJECT']):
            self.state['latest'].update(inspection_id=f'fixture-{index}', grade=grade)
            self.refresh()
            expect(self.page.locator('#grade')).to_have_text(grade)
            expect(self.page.locator('#grade')).to_have_class('grade ' + grade.lower())
        expect(self.page.locator('#region-summary')).to_have_text('annular face')
        expect(self.page.locator('#area')).to_have_text('10 px · 10.0%')
        expect(self.page.locator('#action')).to_have_text('Fixture action')

    def test_capture_payload_and_duplicate_submit_lock(self):
        expect(self.page.locator('#camera-status')).to_have_text('LIVE')
        self.page.locator('#part-id').fill('SCREEN-QA')
        self.page.locator('#confirmed').check()
        self.hold_posts = True
        self.page.locator('#inspect').click()
        expect(self.page.locator('#inspect')).to_be_disabled()
        self.page.evaluate("document.getElementById('capture-form').dispatchEvent(new Event('submit', {cancelable:true}))")
        self.refresh()
        self.assertEqual(len(self.posts), 1)
        path, body, _ = self.posts[0]
        self.assertEqual(path, '/api/inspect')
        self.assertEqual(json.loads(body), dict(part_id='SCREEN-QA', confirmed=True, source_kind='screen_photo'))
        self.state['busy'] = True
        self.state['latest'] = dict(status='PROCESSING', part_id='SCREEN-QA', source_mode='LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC')
        self.pending.pop().fulfill(status=202, content_type='application/json', body='{"status":"PROCESSING"}')
        expect(self.page.locator('#inspection-status')).to_have_text('PROCESSING')
        expect(self.page.locator('#confirmed')).not_to_be_checked()

    def test_upload_sends_bytes_and_declaration_and_keeps_rejection_visible(self):
        self.page.locator('#diagnostic-file').set_input_files(dict(name='fixture.png', mimeType='image/png', buffer=PIXEL))
        self.page.locator('#diagnostic-id').fill('UPLOAD-QA')
        self.page.locator('#mvtec-benchmark').check()
        self.post_error = 'This upload ID is already recorded; enter a new one'
        self.page.locator('#diagnostic-submit').click()
        expect(self.page.locator('#diagnostic-status')).to_contain_text('already recorded')
        self.refresh()
        expect(self.page.locator('#diagnostic-status')).to_contain_text('already recorded')
        path, body, headers = self.posts[0]
        self.assertEqual(path, '/api/upload-diagnostic')
        self.assertEqual(body, PIXEL)
        self.assertEqual(headers['x-mvtec-benchmark'], 'yes')
        self.assertEqual(headers['x-upload-id'], 'UPLOAD-QA')
        expect(self.page.locator('#grade')).to_have_text('REJECT')

    def test_review_draft_is_cleared_when_another_inspection_arrives(self):
        self.page.locator('#review-actor').fill('QA')
        self.page.locator('#review-outcome').select_option('visible_issue')
        self.page.locator('#review-reason').fill('Old inspection draft')
        expect(self.page.locator('#review-submit')).to_be_enabled()
        self.state['latest']['inspection_id'] = 'fixture-new'
        self.refresh()
        expect(self.page.locator('#review-reason')).to_have_value('')
        expect(self.page.locator('#review-outcome')).to_have_value('')
        expect(self.page.locator('#review-submit')).to_be_disabled()

    def test_advice_pending_and_failure_do_not_change_grade(self):
        self.state['advice']['status'] = 'PENDING'
        self.refresh()
        expect(self.page.locator('#advice-request')).to_be_disabled()
        expect(self.page.locator('#grade')).to_have_text('REJECT')
        self.state['advice']['status'] = 'ADVICE_UNAVAILABLE'
        self.refresh()
        expect(self.page.locator('#advice-status')).to_contain_text('unavailable')
        expect(self.page.locator('#advice-note')).to_be_hidden()
        expect(self.page.locator('#grade')).to_have_text('REJECT')

    def test_image_failure_has_placeholder_and_retains_grade(self):
        self.bad_media.add('/api/overlay.png')
        self.state['latest']['inspection_id'] = 'fixture-new'
        self.refresh()
        expect(self.page.locator('#overlay-empty')).to_contain_text('could not load')
        expect(self.page.locator('#overlay')).to_be_hidden()
        expect(self.page.locator('#grade')).to_have_text('REJECT')
        self.page.locator('#show-source').click()
        expect(self.page.locator('#source-image')).to_be_visible()


if __name__ == '__main__':
    unittest.main()
