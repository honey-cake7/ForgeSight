"""Verify durable real replay records after the writer exits; no model inference."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from uuid import uuid4

from forgesight.edge_assets import fingerprint
from forgesight.inspection import CaptureGate
from forgesight.storage import InspectionStore, PartAlreadyInspected, RecordConflict, new_correction


def run_reader(database, action, *args):
    result = subprocess.run([sys.executable, '-B', '-m', 'forgesight.storage',
                             '--db', str(database), action, *args],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    replay = json.loads(args.replay.read_text())
    database = Path(replay['database'])
    store = InspectionStore(database, create=False)
    records = replay['records']
    assert replay['frames'] == 9 and len(records) == 4 and replay['suppressed_frames'] == 5
    assert [r['decision']['grade'] for r in records] == ['PASS', 'REVIEW', 'PASS', 'REVIEW']

    # The writer process has exited. A new interpreter retrieves the original full records.
    after_writer_exit = run_reader(database, 'summary')
    history = run_reader(database, 'history', '--limit', '100')
    assert {r['inspection_id']: r for r in history} == {r['inspection_id']: r for r in records}
    assert after_writer_exit['inspection_count'] == after_writer_exit['distinct_part_count'] == 4
    assert after_writer_exit['original_grade_counts'] == {'PASS': 2, 'REVIEW': 2, 'REJECT': 0}
    assert after_writer_exit['processing_status_counts'] == {'ANALYZED': 3, 'ERROR': 1}
    (args.output/'history_after_restart.json').write_text(json.dumps(history, indent=2)+'\n')

    # Exact JSON retries do not create new rows or modify originals.
    inserted_again = [store.save_inspection(record) for record in records]
    assert inserted_again == [False]*4
    conflicts = []
    changed = deepcopy(records[0]); changed['timings']['TEST_CONFLICT'] = 1
    duplicate_part = deepcopy(records[0]); duplicate_part['inspection_id'] = str(uuid4())
    for candidate, error_type in [(changed, RecordConflict), (duplicate_part, PartAlreadyInspected)]:
        try:
            store.save_inspection(candidate)
        except error_type as exc:
            conflicts.append(type(exc).__name__)
        else:
            raise AssertionError('Conflicting inspection was not refused.')
    gate = CaptureGate(already_recorded=store.has_part)
    try:
        gate.arm(records[0]['part_id'])
    except ValueError:
        gate_blocked_after_restart = True
    else:
        raise AssertionError('Already recorded part re-armed after restart.')

    # The actual CLI refuses a duplicate before loading even the model bundle.
    duplicate = subprocess.run([sys.executable, '-B', '-m', 'forgesight.inspection',
                               records[0]['source']['path'], '--part-id', records[0]['part_id'],
                               '--database', str(database), '--output', str(args.output/'duplicate_must_not_exist')],
                              capture_output=True, text=True, timeout=30)
    assert duplicate.returncode == 2 and 'no new model load' in duplicate.stderr
    assert not (args.output/'duplicate_must_not_exist').exists()
    (args.output/'duplicate_cli_refused.txt').write_text(duplicate.stderr)

    # Demonstrate storage of a proposal, explicitly not an actual operator correction.
    note = new_correction(records[0]['inspection_id'], 'STORAGE_PROOF_DEMONSTRATION',
                          'Synthetic storage demonstration: request review. No actual operator judgment or confirmed finding.', 'REVIEW')
    note['context'] = 'SYNTHETIC_STORAGE_TEST_CLAIM'
    assert store.save_correction(note)
    assert store.save_correction(note) is False
    assert run_reader(database, 'corrections', records[0]['inspection_id']) == [note]
    assert run_reader(database, 'get', records[0]['inspection_id']) == records[0]
    after_note = run_reader(database, 'summary')
    assert after_note['inspection_count'] == 4 and after_note['original_grade_counts'] == after_writer_exit['original_grade_counts']
    assert after_note['unverified_correction_count'] == 1
    (args.output/'demonstration_correction.json').write_text(json.dumps(note, indent=2)+'\n')

    previous = json.loads(Path('manifests/graded_inspection_replay_v1.json').read_text())['records']
    for current, old in zip(records[:3], previous[:3]):
        assert current['source']['sha256'] == old['source']['sha256']
        assert (current['score'], current['candidate'], current['geometry']) == (old['score'], old['candidate'], old['geometry'])
    checks = store.integrity()
    assert checks['integrity_check'] == ['ok'] and checks['foreign_key_check'] == [] and checks['foreign_keys_enabled']
    size, digest = fingerprint(database)
    report = {'schema_version': 1, 'scope': 'DESKTOP_SAVED_IMAGE_SQLITE_REPLAY_NOT_CAMERA_OR_JETSON_PROOF',
              'database': {'path': str(database), 'bytes': size, 'sha256_at_verification': digest,
                           'sqlite_version': sqlite3.sqlite_version},
              'replay_report_path': str(args.replay), 'replay_report_sha256': fingerprint(args.replay)[1],
              'replay_frames': replay['frames'], 'suppressed_repeats': replay['suppressed_frames'],
              'summary_after_writer_process_exit': after_writer_exit, 'summary_after_demonstration_claim': after_note,
              'exact_record_roundtrip_after_restart': True, 'idempotent_retries': len(inserted_again),
              'refused_conflicts': conflicts, 'persistent_gate_blocked': gate_blocked_after_restart,
              'duplicate_cli_returncode': duplicate.returncode, 'duplicate_cli_started_inference': False,
              'same_image_scores_geometry_as_stage5': True,
              'correction_scope': 'ONE_SYNTHETIC_UNVERIFIED_STORAGE_DEMONSTRATION; ORIGINAL_DECISION_UNCHANGED',
              'reject_storage_scope': 'UNIT_FIXTURE_ONLY; NO_REAL_AUTOMATIC_REJECT_CLAIM',
              'integrity': checks, 'new_holdout_model_forwards': 0,
              'physical_camera_identity_verified': False, 'jetson_verified': False,
              'power_loss_or_process_kill_recovery_verified': False,
              'source_sha256': fingerprint(Path(__file__))[1]}
    (args.output/'storage_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
