"""Deterministic, pixel-only prototype decisions; no generative model input."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


DEFAULT_POLICY = Path('configs/forgegrade_v1.json')


def policy_digest(policy: dict) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_policy(policy: dict):
    if (policy.get('schema_version') != 1 or not policy.get('policy_id')
            or policy.get('units') != 'pixels' or policy.get('scope') != 'UNCERTIFIED_BENCHMARK_SAVED_IMAGE_PROTOTYPE'
            or policy.get('physical_scale_calibrated') is not False
            or policy.get('industrial_certified') is not False
            or policy.get('pixel_threshold_validated') is not False):
        raise ValueError('Expected a versioned, uncertified, pixel-only prototype policy.')
    width, height = policy['source_size_px']
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError('Invalid policy source size.')
    for key, lower, upper in [('image_threshold', 0, float('inf')),
                              ('normal_review_margin_fraction', 0, 1),
                              ('reject_response_area_fraction', 0, 1),
                              ('reject_bbox_diagonal_px', 0, math.hypot(width, height))]:
        value = policy[key]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not lower < value <= upper:
            raise ValueError(f'Invalid policy cutoff: {key}')
    if (policy['required_type_status'] != 'VALIDATED_SAME_DOMAIN_RELIABLE'
            or not policy['accepted_types'] or 'unknown' in policy['accepted_types']):
        raise ValueError('Unknown types cannot be authorized for automatic rejection.')
    for zone in policy['critical_zones']:
        x, y, zw, zh = zone['bbox_xywh_px']
        if (not zone['name'] or any(type(v) is not int for v in (x, y, zw, zh))
                or x < 0 or y < 0 or zw <= 0 or zh <= 0 or x + zw > width or y + zh > height):
            raise ValueError('Critical zone must be a named in-frame pixel rectangle.')
    policy_digest(policy)


def _validated_geometry(record: dict, policy: dict):
    geometry = record['geometry']
    width, height = policy['source_size_px']
    if (record['source']['size_px'] != [width, height] or geometry['source_size_px'] != [width, height]
            or geometry['units'] != 'pixels' or geometry['threshold'] != policy['image_threshold']
            or geometry['measurement'] != 'thresholded_model_response_not_physical_damage'
            or geometry['pixel_threshold_validated'] is not False
            or geometry['connectivity'] != 8 or geometry['morphology'] != 'none'):
        raise ValueError('Unmatched geometry profile.')
    total = geometry['total_area_px']
    if type(total) is not int or not 0 <= total <= width * height:
        raise ValueError('Invalid response area.')
    summed = 0
    for region in geometry['regions']:
        x, y, rw, rh = region['bbox_xywh_px']
        area, length = region['area_px'], region['bbox_diagonal_px']
        if (any(type(v) is not int for v in (x, y, rw, rh, area))
                or min(x, y) < 0 or min(rw, rh, area) <= 0
                or x + rw > width or y + rh > height or area > rw * rh
                or not math.isfinite(length) or not math.isclose(length, math.hypot(rw, rh), rel_tol=1e-9)):
            raise ValueError('Invalid region extent.')
        summed += area
    if summed != total:
        raise ValueError('Region and total areas disagree.')
    return geometry


def _overlap(a, b):
    x, y, w, h = a
    xx, yy, ww, hh = b
    return x < xx + ww and xx < x + w and y < yy + hh and yy < y + h


def grade_record(record: dict, policy: dict) -> dict:
    """No clock/randomness/LLM; identical record and policy yield identical decision."""
    validate_policy(policy)
    def decision(grade, reason, severity, action, **evidence):
        return {'grade': grade, 'reason': reason, 'severity': severity,
                'operator_action': action, 'policy_id': policy['policy_id'],
                'policy_sha256': policy_digest(policy), 'policy_snapshot': deepcopy(policy),
                'units': 'pixels', 'industrial_certified': False,
                'physical_scale_calibrated': False, 'evidence': evidence,
                'authority': 'DETERMINISTIC_RULES_ONLY'}
    def review(reason, **evidence):
        return decision('REVIEW', reason, 'UNASSESSED',
                        'Hold this part for human inspection; check image quality and the highlighted response. No factory cause is inferred.', **evidence)

    if record.get('status') != 'ANALYZED':
        return review('INSPECTION_ERROR_OR_INCOMPLETE')
    if record.get('source', {}).get('mode') in ('LIVE_CAMERA', 'LIVE_CAMERA_DIAGNOSTIC_NO_PART'):
        return review('CAMERA_DOMAIN_UNVALIDATED',
                      score=record.get('score'),
                      response_area_px=(record.get('geometry') or {}).get('total_area_px'))
    try:
        model, score = record['model'], record['score']
        if (model['checkpoint_sha256'] != policy['checkpoint_sha256']
                or model['score_policy_sha256'] != policy['score_policy_sha256']
                or model['image_threshold'] != policy['image_threshold']):
            return review('MODEL_OR_SCORE_POLICY_MISMATCH')
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            return review('INVALID_SCORE')
        geometry = _validated_geometry(record, policy)
        anomaly = score > policy['image_threshold']
        if record['candidate'] != ('ANOMALY_CANDIDATE' if anomaly else 'NORMAL_CANDIDATE'):
            return review('INCONSISTENT_CANDIDATE')
        if not anomaly:
            if geometry['total_area_px']:
                return review('INCONSISTENT_NORMAL_LOCALIZATION')
            if score >= policy['image_threshold'] * (1 - policy['normal_review_margin_fraction']):
                return review('NEAR_IMAGE_THRESHOLD_NOT_A_CONFIDENCE_PROBABILITY')
            return decision('PASS', 'NORMAL_CANDIDATE_OUTSIDE_REVIEW_BAND', 'NO_FLAGGED_RESPONSE',
                            'No anomaly flagged under this benchmark policy. PASS is provisional; missed scratches are documented.',
                            score=score, response_area_px=0)
        if (record['predicted_defect_type'] not in policy['accepted_types']
                or record['type_status'] != policy['required_type_status']):
            return review('UNKNOWN_UNSEEN_OR_UNRELIABLE_TYPE', response_area_px=geometry['total_area_px'])
        if not geometry['regions']:
            return review('ANOMALY_WITHOUT_SOURCE_GRID_LOCALIZATION')
        area_fraction = geometry['total_area_px'] / (policy['source_size_px'][0] * policy['source_size_px'][1])
        longest = max(r['bbox_diagonal_px'] for r in geometry['regions'])
        zones = [z['name'] for z in policy['critical_zones']
                 if any(_overlap(r['bbox_xywh_px'], z['bbox_xywh_px']) for r in geometry['regions'])]
        if zones or area_fraction >= policy['reject_response_area_fraction'] or longest >= policy['reject_bbox_diagonal_px']:
            return decision('REJECT', 'PROTOTYPE_RESPONSE_GEOMETRY_RULE', 'HIGH_RESPONSE_EXTENT_OR_ZONE',
                            'Segregate for manual confirmation against an approved part specification; this prototype is not a certified rejection standard.',
                            response_area_fraction=area_fraction, bbox_diagonal_px=longest,
                            critical_zone_bbox_overlaps=zones)
        return review('KNOWN_RESPONSE_BELOW_PROVISIONAL_REJECT_LIMITS', response_area_fraction=area_fraction)
    except (KeyError, TypeError, ValueError, OverflowError):
        return review('INVALID_OR_INCOMPLETE_EVIDENCE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    record['decision'] = grade_record(record, json.loads(args.policy.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(record['decision'], indent=2))


if __name__ == '__main__':
    main()
