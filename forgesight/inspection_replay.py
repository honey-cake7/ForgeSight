"""Proof of the part gate with real development-image inference, not accuracy testing."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from forgesight.edge_assets import fingerprint
from forgesight.inspection import CaptureGate, inspect_one_image
from forgesight.vision_inference import CalibratedPredictor
from forgesight.storage import InspectionStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--database', type=Path, help='Defaults to inspections.sqlite3 within the new output directory.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    database = args.database or args.output/'inspections.sqlite3'
    store = InspectionStore(database)
    predictor, gate = CalibratedPredictor(), CaptureGate(already_recorded=store.has_part)
    root = Path('data/mvtec_ad/metal_nut')
    cases = [('replay-A', root/'train/good/000.png', 3),
             ('replay-B', root/'test/bent/017.png', 2),
             ('replay-C', root/'train/good/000.png', 2),
             ('replay-invalid', args.output/'intentionally_missing.png', 2)]
    records, frames = [], 0
    for part_id, path, repetitions in cases:
        gate.arm(part_id)
        for _ in range(repetitions):
            frames += 1
            record = gate.on_frame(path, lambda p, part: inspect_one_image(p, part, predictor, args.output/'records', store=store))
            if record is not None:
                records.append(asdict(record))
    assert len(records) == 4 and gate.suppressed_frames == 5 and frames == 9
    assert [r['status'] for r in records] == ['ANALYZED'] * 3 + ['ERROR']
    assert records[0]['source']['sha256'] == records[2]['source']['sha256']
    assert records[0]['inspection_id'] != records[2]['inspection_id']
    persisted = InspectionStore(database, create=False).summary()
    assert persisted['inspection_count'] == 4 and persisted['processing_status_counts']['ERROR'] == 1
    summary = {'schema_version': 1, 'scope': 'DEVELOPMENT_SAVED_IMAGE_REPLAY_NOT_INDEPENDENT_ACCURACY',
               'frames': frames, 'distinct_declared_parts': len(records),
               'suppressed_frames': gate.suppressed_frames,
               'physical_camera_identity_verified': False, 'persistence': 'SQLITE_COMMITTED',
               'database': str(database), 'database_summary': persisted,
               'records': records, 'source_sha256': fingerprint(Path(__file__))[1]}
    (args.output/'replay_report.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'frames': frames, 'records': len(records), 'suppressed_frames': gate.suppressed_frames}, indent=2))


if __name__ == '__main__':
    main()
