"""Local saved-image operator API; real vision, deterministic grades and SQLite."""

import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import Lock
import time
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from forgesight.data_manifest import DEFAULT_ROOT, read_manifest
from forgesight.grade import DEFAULT_POLICY
from forgesight.inspection import CaptureGate, inspect_one_image
from forgesight.storage import DEFAULT_DATABASE, InspectionStore, PartAlreadyInspected


WEB_ROOT = Path(__file__).with_name('web')
SAMPLE_PATHS = (
    ('normal-fit-000', 'Normal reference (fitting example)', 'train/good/000.png'),
    ('development-1', 'Development photo 1', 'test/bent/017.png'),
    ('development-2', 'Development photo 2', 'test/color/009.png'),
    ('development-3', 'Development photo 3', 'test/flip/012.png'),
    ('development-4', 'Development photo 4', 'test/scratch/020.png'),
)


class InspectRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    part_id: str = Field(min_length=1, max_length=128)
    sample_id: str = Field(min_length=1, max_length=64)

    @field_validator('part_id')
    @classmethod
    def valid_part(cls, value):
        if not value.strip() or value != value.strip():
            raise ValueError('Enter a part ID without leading or trailing spaces.')
        return value


def _real_predictor(device):
    from forgesight.vision_inference import CalibratedPredictor
    return CalibratedPredictor(device=device)


def create_app(*, database=DEFAULT_DATABASE, output=Path('artifacts/runtime/operator'),
               data_root=DEFAULT_ROOT, device='cuda', predictor_factory=_real_predictor):
    """The injectable factory supports contract tests; the launcher always uses the real model."""
    output, data_root = Path(output).resolve(), Path(data_root).resolve()
    policy = json.loads(DEFAULT_POLICY.read_text())
    manifest = read_manifest(Path('manifests/metal_nut_v1.json'))
    allowed = {row['path'] for row in manifest['samples'] if row['role'] in ('fit', 'development_preview')}
    if any(path not in allowed for _, _, path in SAMPLE_PATHS):
        raise ValueError('Operator catalog may only expose frozen fit/development examples.')
    catalog = {sample: {'id': sample, 'label': label, 'path': data_root/path}
               for sample, label, path in SAMPLE_PATHS}
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = InspectionStore(database)
        app.state.gate = CaptureGate(already_recorded=app.state.store.has_part)
        app.state.predictor, app.state.model_error = None, None
        try:
            app.state.predictor = predictor_factory(device)
        except Exception as exc:
            app.state.model_error = f'{type(exc).__name__}: {exc}'
        yield
        app.state.predictor = None

    app = FastAPI(title='ForgeSight Local Operator', version='0.1.0', lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost'])

    @app.middleware('http')
    async def local_requests(request, call_next):
        if request.method == 'POST':
            origin = request.headers.get('origin')
            if origin and origin != str(request.base_url).rstrip('/'):
                return JSONResponse({'detail': 'Use the local operator page for inspection requests.'}, status_code=403)
            if int(request.headers.get('content-length', '0')) > 4096:
                return JSONResponse({'detail': 'Inspection requests contain IDs only; body too large.'}, status_code=413)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'"
        return response

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request, exc):
        return JSONResponse({'detail': 'Local inspection storage is unavailable. No successful save is claimed.'}, status_code=503)

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(WEB_ROOT/'index.html', media_type='text/html')

    app.mount('/assets', StaticFiles(directory=WEB_ROOT), name='assets')

    def envelope(record):
        if record is None:
            return None
        identifier = record['inspection_id']
        return {'record': record, 'media': {
            'source': f'/api/inspections/{identifier}/source' if record['source']['sha256'] else None,
            'overlay': f'/api/inspections/{identifier}/overlay' if record['artifacts'].get('response_overlay') else None}}

    def lookup(identifier):
        record = app.state.store.get_inspection(str(identifier))
        if record is None:
            raise HTTPException(404, 'Inspection not found.')
        return record

    @app.get('/api/health')
    def health():
        app.state.store.summary()
        ready = app.state.predictor is not None
        return JSONResponse({'status': 'ready' if ready else 'degraded', 'mode': 'SAVED_IMAGE_REPLAY',
                             'model_ready': ready, 'model_error': app.state.model_error,
                             'runtime_device': device, 'jetson_verified': False,
                             'advice_status': 'NOT_CONNECTED_STAGE_9_PENDING'}, status_code=200 if ready else 503)

    @app.get('/api/config')
    def config():
        return {'mode': 'SAVED_IMAGE_REPLAY', 'camera_connected': False,
                'samples': [{'id': row['id'], 'label': row['label']} for row in catalog.values()],
                'model': {'name': 'EfficientAD-S', 'precision': 'fp32', 'input_size_px': [256, 256],
                          'source_size_px': [700, 700], 'device': device},
                'policy': policy, 'predicted_defect_type': 'unknown',
                'limits': {'industrial_certified': False, 'physical_scale_calibrated': False,
                           'known_missed_test_defects': 2, 'automatic_type_based_reject_verified': False}}

    @app.get('/api/summary')
    def summary():
        return app.state.store.summary()

    @app.get('/api/history')
    def history(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                grade: Literal['PASS', 'REVIEW', 'REJECT'] | None = None,
                status: Literal['ANALYZED', 'ERROR'] | None = None):
        rows = app.state.store.history(limit=limit+1, offset=offset, grade=grade, status=status)
        return {'items': [envelope(r) for r in rows[:limit]],
                'pagination': {'limit': limit, 'offset': offset, 'has_next': len(rows) > limit}}

    @app.get('/api/latest')
    def latest():
        rows = app.state.store.history(limit=1)
        return {'inspection': envelope(rows[0]) if rows else None}

    @app.get('/api/inspections/{inspection_id}')
    def get_inspection(inspection_id: UUID):
        return envelope(lookup(inspection_id))

    @app.get('/api/inspections/{inspection_id}/{kind}')
    def media(inspection_id: UUID, kind: Literal['source', 'overlay']):
        record = lookup(inspection_id)
        if kind == 'source':
            path, expected = Path(record['source']['path']).resolve(), record['source']['sha256']
        else:
            path = Path(record['artifacts'].get('response_overlay', '')).resolve()
            expected = record['artifacts'].get('response_overlay_sha256')
        roots = (data_root, Path('artifacts').resolve())
        if not any(path.is_relative_to(root) for root in roots) or path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            raise HTTPException(404, 'No permitted image reference.')
        if not path.is_file():
            raise HTTPException(404, 'Saved image is unavailable.')
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise HTTPException(409, 'Saved image changed; original inspection evidence does not match.')
        return FileResponse(path, media_type='image/png' if path.suffix.lower() == '.png' else 'image/jpeg')

    @app.post('/api/inspect', status_code=201)
    def inspect(body: InspectRequest, request: Request):
        if body.sample_id not in catalog:
            raise HTTPException(422, 'Choose a listed development photo. Holdout images are unavailable.')
        if app.state.predictor is None:
            raise HTTPException(503, 'The vision model is unavailable; no inspection was run. Check server startup output.')
        if not lock.acquire(blocking=False):
            raise HTTPException(409, 'An inspection is running. Wait for it to finish.')
        started = time.perf_counter()
        try:
            if app.state.store.has_part(body.part_id):
                raise HTTPException(409, 'This part already has a saved inspection. Open it from history; do not rename the same part to count it again.')
            try:
                app.state.gate.arm(body.part_id)
                record = app.state.gate.on_frame(catalog[body.sample_id]['path'],
                    lambda path, part: inspect_one_image(path, part, app.state.predictor, output/'records', store=app.state.store))
            except (ValueError, PartAlreadyInspected) as exc:
                raise HTTPException(409, str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
            except OSError as exc:
                raise HTTPException(503, 'Inspection output could not be saved; no successful inspection is claimed.') from exc
            result = envelope(asdict(record))
            result['request_handler_ms'] = (time.perf_counter()-started)*1000
            result['request_timing_scope'] = 'handler through JSON/SQLite commit; excludes startup, HTTP transfer and browser rendering'
            return result
        finally:
            lock.release()

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--output', type=Path, default=Path('artifacts/runtime/operator'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(database=args.database, output=args.output, device=args.device),
                host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__':
    main()
