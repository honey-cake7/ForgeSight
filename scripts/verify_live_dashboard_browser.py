"""Browser verification for the live dashboard; device proof requires a real API.

By default only reads existing results and checks layout. --png creates one
explicitly identified benchmark inspection and one unverified QA review, then
requests real Llama advice. Use only with a known development-preview PNG.
Browser-only fault scenarios intercept API responses; they are not model proof.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8770')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--png', type=Path)
    parser.add_argument('--part-id')
    args = parser.parse_args()
    if urlsplit(args.url).hostname not in ('127.0.0.1', 'localhost'):
        parser.error('Use a loopback URL, optionally through SSH.')
    if args.png and not args.part_id:
        parser.error('--png requires a unique --part-id')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Browser presentation QA; inference only when --png is supplied',
              'viewports': [], 'page_errors': [], 'external_requests': [], 'console_errors': []}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1366, 'height': 768}, device_scale_factor=1)
        page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        page.on('console', lambda msg: report['console_errors'].append(msg.text) if msg.type == 'error' else None)
        page.on('request', lambda req: report['external_requests'].append(req.url)
                if urlsplit(req.url).netloc != urlsplit(args.url).netloc else None)
        page.goto(args.url, wait_until='domcontentloaded')
        expect(page.locator('#connection-status')).to_have_text('Station connected', timeout=15000)
        state = page.request.get(args.url + '/api/state').json()
        report['displayed_inspection'] = state['latest']
        report['displayed_advice'] = state.get('advice')
        report['displayed_manual_reviews'] = state.get('manual_reviews', [])
        if state['camera']['ready']:
            expect(page.locator('#camera-status')).to_have_text('LIVE', timeout=10000)
            expect(page.locator('#preview')).to_be_visible()
            report['preview_size'] = page.locator('#preview').evaluate('(img) => [img.naturalWidth, img.naturalHeight]')
        expect(page.get_by_role('button', name='Start Camera', exact=True)).to_be_disabled()
        expect(page.get_by_role('button', name='Stop Camera', exact=True)).to_be_disabled()
        report['camera_control_limit_visible'] = page.locator('#camera-control-help').is_visible()

        if args.png:
            page.locator('#diagnostic-file').set_input_files(str(args.png))
            page.locator('#diagnostic-id').fill(args.part_id)
            page.locator('#mvtec-benchmark').check()
            with page.expect_response(lambda r: r.url.endswith('/api/upload-diagnostic')) as upload:
                page.locator('#diagnostic-submit').click()
            assert upload.value.status == 202, upload.value.text()
            expect(page.locator('#result-part')).to_have_text(args.part_id, timeout=90000)
            expect(page.locator('#inspection-status')).to_have_text('SAVED', timeout=90000)
            state = page.request.get(args.url + '/api/state').json()
            report['real_inspection'] = state['latest']
            assert state['latest']['part_id'] == args.part_id
            assert state['latest']['status'] == 'ANALYZED'
            expect(page.locator('#overlay')).to_be_visible()
            original_grade = page.locator('#grade').inner_text()
            page.locator('#show-source').click()
            expect(page.locator('#source-image')).to_be_visible()
            page.locator('#review-actor').fill('Dashboard-QA')
            page.locator('#review-outcome').select_option('uncertain_or_recapture')
            page.locator('#review-reason').fill('Interface verification on a development image only; no physical part reviewed.')
            with page.expect_response(lambda r: r.url.endswith('/api/manual-review')) as review:
                page.locator('#review-submit').click()
            assert review.value.status == 201, review.value.text()
            expect(page.locator('#review-status')).to_contain_text('Original model grade is unchanged')
            expect(page.locator('#review-history')).to_contain_text('Dashboard-QA')
            expect(page.locator('#grade')).to_have_text(original_grade)
            report['real_manual_review'] = review.value.json()
            if state['latest']['candidate'] == 'ANOMALY_CANDIDATE':
                expect(page.locator('#advice-request')).to_be_enabled()
                with page.expect_response(lambda r: r.url.endswith('/api/advice')) as advice:
                    page.locator('#advice-request').click()
                assert advice.value.status == 202, advice.value.text()
                expect(page.locator('#advice-note')).to_be_visible(timeout=150000)
                report['real_advice'] = page.request.get(args.url + '/api/state').json()['advice']
                expect(page.locator('#grade')).to_have_text(original_grade)
            # Duplicate IDs must show the real backend rejection without erasing the result.
            page.locator('#diagnostic-submit').click()
            expect(page.locator('#diagnostic-status')).to_contain_text('already recorded', timeout=10000)
            expect(page.locator('#grade')).to_have_text(original_grade)
            report['duplicate_upload_rejected'] = True

        if state['latest'] and state['latest']['status'] == 'ANALYZED':
            if page.locator('#show-overlay').get_attribute('aria-pressed') == 'true':
                expect(page.locator('#overlay')).to_be_visible()
            page.locator('#show-source').click()
            expect(page.locator('#source-image')).to_be_visible()
            page.locator('#show-overlay').focus()
            page.keyboard.press('Enter')
            expect(page.locator('#overlay')).to_be_visible()
            report['original_overlay_keyboard_switch'] = True
        for width, height in [(1366, 768), (1920, 1080), (1024, 768), (768, 1024), (390, 844)]:
            page.set_viewport_size({'width': width, 'height': height})
            page.evaluate('window.scrollTo(0, 0)')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'Overflow at {width}'
            camera = page.locator('.camera-panel').bounding_box()
            overlay = page.locator('.overlay-panel').bounding_box()
            assert abs(camera['y'] - overlay['y']) < 2 if width > 700 else overlay['y'] > camera['y']
            assert page.locator('#preview').evaluate('(img) => getComputedStyle(img).objectFit') == 'contain'
            page.screenshot(path=str(args.output / f'dashboard_{width}.png'), full_page=True)
            report['viewports'].append([width, height])
        page.reload(wait_until='domcontentloaded')
        expect(page.locator('#connection-status')).to_have_text('Station connected')
        if state['latest'] and state['latest']['status'] == 'ANALYZED':
            expect(page.locator('#result-part')).to_have_text(state['latest']['part_id'])
            expect(page.locator('#history')).to_contain_text(state['latest']['part_id'])
        report['refresh_and_history_verified'] = True
        report['normal_path_console_errors'] = [error for error in report['console_errors']
            if not (report.get('duplicate_upload_rejected') and '409 (Conflict)' in error)]
        assert not report['normal_path_console_errors'], report['normal_path_console_errors']

        # Only this browser sees the intercepted outage; the Jetson keeps running.
        page.route('**/api/state', lambda route: route.abort())
        expect(page.locator('#connection-status')).to_have_text('Station disconnected', timeout=10000)
        for control in ['inspect', 'diagnostic-submit', 'review-submit', 'advice-request']:
            expect(page.locator('#' + control)).to_be_disabled()
        expect(page.locator('#preview')).to_be_hidden()
        expect(page.locator('#connection-error')).to_be_visible()
        page.unroute('**/api/state')
        expect(page.locator('#connection-status')).to_have_text('Station connected', timeout=10000)
        report['browser_only_connection_loss_recovery'] = True
        assert not report['page_errors'], report['page_errors']
        assert not report['external_requests'], report['external_requests']
        report['browser_version'] = browser.version
        browser.close()
    (args.output / 'browser_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
