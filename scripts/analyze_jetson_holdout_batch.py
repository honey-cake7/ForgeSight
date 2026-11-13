"""Check a Jetson score-only batch against frozen image identities and PC scores."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def by_path(items):
    indexed = {row['source_path']: row for row in items}
    if len(indexed) != len(items):
        raise ValueError('Duplicate image predictions')
    return indexed


def compare(device, reference):
    old = by_path(rows(reference))
    if set(old) != set(device):
        raise ValueError('PC reference image set differs from the Jetson batch')
    if any('source_sha256' in old[path] and old[path]['source_sha256'] != row['source_sha256']
           for path, row in device.items()):
        raise ValueError('PC and Jetson source image hashes differ')
    return {'reference_sha256': sha256(reference),
            'max_absolute_score_difference': max(abs(row['score'] - old[path]['score'])
                                                 for path, row in device.items()),
            'candidate_disagreements': sorted(path for path, row in device.items()
                                               if row['candidate'] != old[path]['candidate'])}


def analyze(args):
    report = json.loads(args.report.read_text())
    if (report.get('status') != 'FROZEN_MVTEC_BATCH_SCORE_EVALUATED'
            or report.get('jetson_verified') is not True
            or report.get('images') != 111
            or report.get('predictions_sha256') != sha256(args.predictions)
            or report.get('manifest_sha256') != sha256(args.manifest)):
        raise ValueError('Jetson report, image count, or input fingerprints mismatch')
    manifest = json.loads(args.manifest.read_text())
    holdout = {row['path']: row for row in manifest['samples'] if row['role'] == 'final_holdout'}
    device = by_path(rows(args.predictions))
    if len(holdout) != 111 or set(device) != set(holdout):
        raise ValueError('Incomplete or different final-holdout image set')
    threshold, review_start = report['threshold'], report['review_start']
    counts = Counter()
    grades = Counter()
    by_class = defaultdict(Counter)
    mistakes, reviews = [], []
    for path, row in sorted(device.items()):
        source = holdout[path]
        if (row['source_sha256'] != source['sha256'] or row['truth_type'] != source['label']
                or row['truth_anomalous'] != int(source['label'] != 'good')
                or not math.isfinite(row['score']) or row['candidate'] !=
                ('ANOMALY_CANDIDATE' if row['score'] > threshold else 'NORMAL_CANDIDATE')):
            raise ValueError('Image identity, truth folder, score, or candidate mismatch: ' + path)
        flagged = row['score'] > threshold
        grade = 'REJECT' if flagged else 'REVIEW' if row['score'] >= review_start else 'PASS'
        outcome = ('tp' if source['label'] != 'good' else 'fp') if flagged else (
            'fn' if source['label'] != 'good' else 'tn')
        counts[outcome] += 1
        grades[grade] += 1
        by_class[source['label']][outcome] += 1
        by_class[source['label']]['images'] += 1
        detail = {'source_path': path, 'truth_folder': source['label'],
                  'score': row['score'], 'candidate': row['candidate'], 'benchmark_grade': grade}
        if outcome in ('fp', 'fn'):
            mistakes.append(detail)
        if grade == 'REVIEW':
            reviews.append(detail)
    if (dict(counts) != report['confusion'] or dict(grades) != report['benchmark_grade_counts']
            or {row['source_path'] for row in mistakes} != {row['source_path'] for row in report['errors']}):
        raise ValueError('Report totals do not match the verified per-image rows')
    result = {'scope': '111 MVTec metal_nut final-holdout images, score only; no physical parts',
              'jetson_report_sha256': sha256(args.report),
              'jetson_predictions_sha256': sha256(args.predictions),
              'images': len(device), 'confusion': report['confusion'],
              'benchmark_grade_counts': report['benchmark_grade_counts'],
              'by_truth_folder': {name: dict(by_class[name]) for name in sorted(by_class)},
              'mistakes': mistakes, 'review_band_images': reviews,
              'pc_batch_parity': compare(device, args.pc_batch),
              'pc_original_parity': compare(device, args.pc_original)}
    if args.output.exists():
        raise FileExistsError('Choose a new analysis output file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pc-batch', type=Path, required=True)
    parser.add_argument('--pc-original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    result = analyze(parser.parse_args())
    print(json.dumps({key: result[key] for key in
                      ('images', 'confusion', 'benchmark_grade_counts', 'by_truth_folder',
                       'mistakes', 'review_band_images', 'pc_batch_parity', 'pc_original_parity')}, indent=2))


if __name__ == '__main__':
    main()
