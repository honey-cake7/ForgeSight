"""Save grading evidence from a real replay and existing holdout caches; no inference."""

import argparse
from collections import Counter
import json
from pathlib import Path

from forgesight.edge_assets import fingerprint
from forgesight.grade import DEFAULT_POLICY, grade_record, policy_digest
from forgesight.regions import response_regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, default=Path('artifacts/core/graded_replay_v1/replay_report.json'))
    parser.add_argument('--holdout', type=Path, default=Path('artifacts/vision/holdout_evaluation_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    policy = json.loads(DEFAULT_POLICY.read_text())
    replay = json.loads(args.replay.read_text())
    grades = []
    for record in replay['records']:
        result = grade_record(record, policy)
        assert result == grade_record(record, policy) == record['decision']
        grades.append(result['grade'])
    assert grades == ['PASS', 'REVIEW', 'PASS', 'REVIEW']
    assert len({r['inspection_id'] for r in replay['records']}) == 4

    import numpy as np
    context = json.loads((args.holdout/'protocol.json').read_text())
    rows = [json.loads(line) for line in (args.holdout/'predictions.jsonl').read_text().splitlines()]
    # Binding to the committed, previously evaluated predictions: no new model calls.
    evaluated = json.loads(Path('manifests/efficientad_s_evaluation_v1.json').read_text())
    if fingerprint(args.holdout/'predictions.jsonl')[1] != evaluated['predictions_sha256']:
        raise ValueError('Holdout predictions differ from committed evaluation.')
    diagnostics = []
    for row in rows:
        map_path = args.holdout/row['map_file']
        if fingerprint(map_path)[1] != row['map_file_sha256']:
            raise ValueError('Cached map changed.')
        with np.load(map_path, allow_pickle=False) as saved:
            geometry = response_regions(saved['map_source'], context['threshold'])
        # Evaluation evidence only: no fabricated inspection/physical part IDs.
        record = {'status': 'ANALYZED', 'source': {'size_px': row['source_size']},
                  'model': {'checkpoint_sha256': context['checkpoint_sha256'],
                            'score_policy_sha256': context['policy_sha256'], 'image_threshold': context['threshold']},
                  'score': row['score'], 'candidate': row['candidate'], 'geometry': geometry,
                  'predicted_defect_type': 'unknown', 'type_status': 'NO_VALIDATED_SUBTYPE_CLASSIFIER'}
        decision = grade_record(record, policy)
        diagnostics.append({'source_path': row['source_path'], 'ground_truth_anomalous': row['truth_anomalous'],
                            'grade': decision['grade'], 'reason': decision['reason']})
    counts = {name: dict(Counter(r['grade'] for r in diagnostics if r['ground_truth_anomalous'] == value))
              for name, value in [('ground_truth_good', 0), ('ground_truth_anomalous', 1)]}
    report = {'schema_version': 1, 'policy_id': policy['policy_id'], 'policy_sha256': policy_digest(policy),
              'policy_file_sha256': fingerprint(DEFAULT_POLICY)[1],
              'replay_report_path': str(args.replay), 'replay_report_sha256': fingerprint(args.replay)[1],
              'replay_frames': replay['frames'], 'replay_records': len(grades),
              'replay_suppressed_frames': replay['suppressed_frames'], 'replay_grades': grades,
              'cached_holdout_scope': 'POST_EVALUATION_RULE_DIAGNOSTIC_NOT_FRESH_INDEPENDENT_GRADE_ACCEPTANCE',
              'new_holdout_model_forwards': 0, 'diagnostic_rows_are_new_part_events': False,
              'cached_holdout_counts': counts,
              'known_defects_provisionally_passed': [r['source_path'] for r in diagnostics if r['ground_truth_anomalous'] and r['grade'] == 'PASS'],
              'automatic_type_based_reject_verified': False,
              'jetson_verified': False, 'physical_scale_calibrated': False, 'industrial_certified': False}
    (args.output/'cached_holdout_grades.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    (args.output/'grade_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
