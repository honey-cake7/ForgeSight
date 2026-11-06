"""One declared part, one real inference attempt, one structured JSON record."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
from uuid import uuid4

from PIL import Image, ImageDraw

from forgesight.regions import response_regions


@dataclass
class InspectionRecord:
    schema_version: int
    inspection_id: str
    part_id: str
    timestamp_utc: str
    source: dict
    model: dict
    status: str
    score: float | None
    candidate: str | None
    predicted_defect_type: str
    type_status: str
    geometry: dict | None
    decision: dict | None
    advice_status: str
    error: dict | None
    artifacts: dict
    timings: dict


def validate_part_id(part_id: str):
    if not isinstance(part_id, str) or not part_id.strip() or len(part_id) > 128:
        raise ValueError('A nonempty operator-declared part ID of at most 128 characters is required.')


def inspect_one_image(path: Path, part_id: str, predictor, output: Path,
                      grade_policy: Path = Path('configs/forgegrade_v1.json'), *, store=None) -> InspectionRecord:
    """An invalid image yields an ERROR record. Output-write errors propagate."""
    validate_part_id(part_id)
    if store is not None and store.has_part(part_id):
        from forgesight.storage import PartAlreadyInspected
        raise PartAlreadyInspected('Part ID already recorded; no new inference or record created.')
    from forgesight.grade import grade_record, validate_policy
    policy = json.loads(grade_policy.read_text())
    validate_policy(policy)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record = InspectionRecord(
        schema_version=1, inspection_id=str(uuid4()), part_id=part_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source={'path': str(path), 'mode': 'SAVED_IMAGE_REPLAY', 'sha256': None, 'size_px': None},
        model={'name': 'EfficientAD-S', 'input_size_px': [256, 256], 'precision': 'fp32',
               'checkpoint_sha256': predictor.reference['calibrated_checkpoint']['sha256'],
               'score_policy_sha256': predictor.reference['policy_file_sha256'],
               'image_threshold': predictor.policy['threshold'], 'runtime_device': predictor.device,
               'jetson_verified': False},
        status='ERROR', score=None, candidate=None, predicted_defect_type='unknown',
        type_status='NO_VALIDATED_SUBTYPE_CLASSIFIER', geometry=None, decision=None,
        advice_status='NOT_REQUESTED_STAGE_9_PENDING', error=None, artifacts={}, timings={})
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('Image file exceeds the 32 MiB input bound.')
        encoded = path.read_bytes()
        record.source['sha256'] = hashlib.sha256(encoded).hexdigest()
        with Image.open(BytesIO(encoded)) as image:
            if image.mode != 'RGB' or image.size != (700, 700) or image.format not in ('PNG', 'JPEG'):
                raise ValueError('This benchmark profile requires a 700x700 RGB PNG/JPEG image.')
            image.load()
            record.source['size_px'] = list(image.size)
            prediction = predictor.predict(image)
            record.score, record.candidate = prediction['score'], prediction['candidate']
            record.geometry = response_regions(prediction['map_source'], predictor.policy['threshold'])
            import numpy as np
            map_path = output / f'{record.inspection_id}.npz'
            np.savez_compressed(map_path, map_source=prediction['map_source'], map_model=prediction['map_model'])
            record.artifacts['score_map'] = str(map_path)
            record.artifacts['score_map_sha256'] = hashlib.sha256(map_path.read_bytes()).hexdigest()
            # Plain red response extent and boxes: no invented segmentation/type.
            support = Image.fromarray((prediction['map_source'] > predictor.policy['threshold']).astype('uint8') * 110)
            overlay = Image.composite(Image.new('RGB', image.size, 'red'), image, support)
            draw = ImageDraw.Draw(overlay)
            for region in record.geometry['regions']:
                x, y, width, height = region['bbox_xywh_px']
                draw.rectangle([x, y, x + width - 1, y + height - 1], outline='yellow', width=2)
            overlay_path = output / f'{record.inspection_id}_response.png'
            overlay.save(overlay_path)
            record.artifacts['response_overlay'] = str(overlay_path)
            record.timings = {k: prediction[k] for k in ('model_forward_ms', 'predict_image_ms', 'timing_scope')}
            record.status = 'ANALYZED'
    except Exception as exc:
        record.status = 'ERROR'
        record.error = {'type': type(exc).__name__, 'message': str(exc)}
    record.decision = grade_record(asdict(record), policy)
    record.timings['inspection_before_record_write_ms'] = (time.perf_counter() - started) * 1000
    record.timings['scope'] = 'decode/hash, prediction, regions, grading, map and overlay writes; excludes bundle/policy load, duplicate check, JSON/SQLite writes and Llama'
    record_path = output / f'{record.inspection_id}.json'
    with record_path.open('x') as stream:
        json.dump(asdict(record), stream, indent=2, allow_nan=False)
    if store is not None:
        try:
            store.save_inspection(asdict(record))
        except Exception as exc:
            raise RuntimeError(f'SQLite save failed; inspection JSON is preserved for retry at {record_path}') from exc
    return record


class CaptureGate:
    """Session-local explicit re-arm gate; never infer physical identity from pixels."""
    def __init__(self, *, already_recorded=None):
        self.part_id = None
        self.latched = False
        self.seen_ids = set()
        self.suppressed_frames = 0
        self.already_recorded = already_recorded

    def arm(self, part_id: str):
        validate_part_id(part_id)
        if self.part_id is not None and not self.latched:
            raise ValueError('Current part has not been inspected yet.')
        if part_id in self.seen_ids:
            raise ValueError('Part ID already used in this session; declare a distinct new part.')
        if self.already_recorded is not None and self.already_recorded(part_id):
            raise ValueError('Part ID already recorded in persistent history; declare a distinct new part.')
        self.part_id, self.latched = part_id, False
        self.seen_ids.add(part_id)

    def on_frame(self, path: Path, inspect):
        if self.part_id is None:
            raise ValueError('Declare a new part before accepting a frame.')
        if self.latched:
            self.suppressed_frames += 1
            return None
        # Latch before inference: failures and repeated frames cannot create extra events.
        self.latched = True
        return inspect(path, self.part_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--part-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    from forgesight.storage import DEFAULT_DATABASE, InspectionStore
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    store = InspectionStore(args.database)
    if store.has_part(args.part_id):
        parser.error('Part ID already recorded; no new model load or inspection attempted.')
    from forgesight.vision_inference import CalibratedPredictor
    predictor = CalibratedPredictor(device=args.device)
    record = inspect_one_image(args.image, args.part_id, predictor, args.output, store=store)
    print(json.dumps(asdict(record), indent=2))
    if record.status == 'ERROR':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
