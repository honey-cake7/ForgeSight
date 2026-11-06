"""Windows headless-browser proof against the real local API, never a detector mock."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--restart-only', action='store_true')
    parser.add_argument('--resume-existing', action='store_true', help='Resume layout QA on records saved by a previous interrupted proof; no new inference.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.url.rstrip('/')
    if urlsplit(base).hostname not in ('127.0.0.1', 'localhost'):
        parser.error('This proof is for a loopback server only.')
    errors, remote_requests, saved = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width':1440, 'height':1120}, device_scale_factor=1, timezone_id='Asia/Kolkata')
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: remote_requests.append(request.url)
                if urlsplit(request.url).scheme in ('http', 'https') and urlsplit(request.url).hostname not in ('127.0.0.1','localhost') else None)
        page.goto(base, wait_until='networkidle')
        expect(page.locator('#health-label')).to_have_text('Local GPU model ready', timeout=60000)

        def get(path):
            result = page.request.get(base+path)
            assert result.ok, (path, result.status, result.text())
            return result.json()

        if args.restart_only:
            summary = get('/api/summary')
            assert summary['inspection_count'] == 2
            assert summary['original_grade_counts'] == {'PASS':1,'REVIEW':1,'REJECT':0}
            expect(page.locator('#result-heading')).to_have_text('UI-PROOF-B')
            expect(page.locator('#grade')).to_have_text('REVIEW')
            expect(page.locator('#total-count')).to_have_text('2')
            expect(page.locator('#overlay-image')).to_be_visible()
            page.screenshot(path=str(args.output/'history_after_server_restart.png'), full_page=True)
            report={'scope':'FRESH_BROWSER_AFTER_REAL_SERVER_RESTART', 'summary':summary,
                    'latest':get('/api/latest'), 'page_errors':errors, 'external_requests':remote_requests}
        else:
            if args.resume_existing:
                assert get('/api/summary')['inspection_count'] == 2
                saved = sorted(get('/api/history')['items'], key=lambda row: row['record']['part_id'])
                assert [row['record']['part_id'] for row in saved] == ['UI-PROOF-A','UI-PROOF-B']
                (args.output/'recovered_actual_records.json').write_text(json.dumps(saved,indent=2)+'\n')
                page.locator('#part-id').fill('UI-PROOF-B')
            else:
                assert get('/api/summary')['inspection_count'] == 0, 'Use a fresh proof database; never overwrite prior proof.'
                expect(page.locator('#history-empty')).to_be_visible()
            for part, sample, grade in ([] if args.resume_existing else [('UI-PROOF-A','normal-fit-000','PASS'), ('UI-PROOF-B','development-1','REVIEW')]):
                page.locator('#part-id').fill(part)
                page.locator('#sample-id').select_option(sample)
                started=time.perf_counter()
                with page.expect_response(lambda r:r.url==base+'/api/inspect' and r.request.method=='POST') as response:
                    page.get_by_role('button',name='Inspect part',exact=True).click()
                result=response.value
                assert result.status==201,result.text()
                payload=result.json(); saved.append(payload)
                (args.output/'actual_click_responses.json').write_text(json.dumps(saved,indent=2)+'\n')
                expect(page.locator('#result-heading')).to_have_text(part)
                expect(page.locator('#grade')).to_contain_text(grade)
                expect(page.locator('#overlay-image')).to_be_visible()
                expect(page.locator('#inspect-progress')).to_contain_text('Saved one inspection')
                payload['browser_click_to_saved_display_ms']=(time.perf_counter()-started)*1000
                (args.output/'actual_click_responses.json').write_text(json.dumps(saved,indent=2)+'\n')
                assert payload['record']['decision']['grade']==grade
                assert payload['record']['predicted_defect_type']=='unknown'
            summary=get('/api/summary')
            assert summary['inspection_count']==2
            with page.expect_response(lambda r:r.url==base+'/api/inspect') as duplicate:
                page.get_by_role('button',name='Inspect part',exact=True).click()
            assert duplicate.value.status==409
            expect(page.locator('#message')).to_contain_text('already has a saved inspection')
            assert get('/api/summary')['inspection_count']==2
            page.reload(wait_until='networkidle')
            expect(page.locator('#result-heading')).to_have_text('UI-PROOF-B')
            expect(page.locator('#overlay-image')).to_be_visible()
            assert get('/api/summary')['inspection_count']==2
            page.locator('#grade-filter').select_option('PASS')
            expect(page.locator('#history-body tr')).to_have_count(1)
            page.get_by_role('button',name='View inspection UI-PROOF-A',exact=True).click()
            expect(page.locator('#grade')).to_contain_text('PASS')
            page.locator('#grade-filter').select_option('')
            expect(page.locator('#history-body tr')).to_have_count(2)
            page.get_by_role('button',name='View inspection UI-PROOF-B',exact=True).click()
            expect(page.locator('#overlay-image')).to_be_visible()
            page.screenshot(path=str(args.output/'operator_desktop.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844})
            page.screenshot(path=str(args.output/'operator_mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile page overflows viewport'
            config=get('/api/config')
            assert len(config['samples'])==5 and config['mode']=='SAVED_IMAGE_REPLAY'
            previous=json.loads(Path('manifests/graded_inspection_replay_v1.json').read_text())['records']
            for current,old in zip(saved,previous[:2]):
                r=current['record']
                assert (r['source']['sha256'],r['score'],r['geometry'])==(old['source']['sha256'],old['score'],old['geometry'])
            report={'scope':'REAL_BROWSER_CLICK_REAL_GPU_MODEL_AND_SQLITE_DESKTOP_REPLAY',
                    'browser':browser.version,'inspections':saved,'summary':summary,
                    'duplicate_post_status':duplicate.value.status,'refresh_preserved_records':True,
                    'history_filter_and_view_worked':True,'mobile_width_px':390,'mobile_horizontal_overflow':False,
                    'page_errors':errors,'external_requests':remote_requests,
                    'same_image_scores_geometry_as_stage5':True,'new_holdout_inference':0,
                    'resumed_after_layout_failure':args.resume_existing,
                    'inspection_trigger_note': 'Original browser run clicked Inspect twice and saved these records before a mobile-layout assertion failed; this run verifies the existing records without new inference.' if args.resume_existing else 'Both records created by Inspect clicks in this run.',
                    'jetson_verified':False,'physical_camera_verified':False,'llama_advice_connected':False}
        assert not errors,errors
        assert not remote_requests,remote_requests
        for path in args.output.glob('*.png'):
            report.setdefault('screenshots',{})[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output/'browser_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('inspections','latest')},indent=2))
        browser.close()


if __name__=='__main__':
    main()
