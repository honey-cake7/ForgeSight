"""Conservative screw benchmark grading and bounded manual-inspection choices."""
import math
import re

from forgesight.assist_contract import canonical_hash
from forgesight.defect_type import SCREW_TYPES

GUIDANCE_DOCUMENT_SHA256 = '11138245b406db3c88d24f8754ca013057c76ef50b48a7ea128a4b0b57ac065f'


def validate_guidance(guidance):
    if canonical_hash(guidance) != GUIDANCE_DOCUMENT_SHA256:
        raise ValueError('Screw guidance content differs from the reviewed prototype document.')
    if (guidance.get('schema_version') != 1 or guidance.get('plugin_id') != 'screw'
            or guidance.get('approval_status') != 'PROTOTYPE_PENDING_SITE_QC_APPROVAL'
            or guidance.get('authorizes_release_or_repair') is not False
            or set(guidance.get('checks', {})) != set(SCREW_TYPES) | {'unknown'}):
        raise ValueError('Unrecognized screw inspection guidance.')
    texts = [guidance.get('fallback')]
    for choices in guidance['checks'].values():
        if not isinstance(choices, dict) or set(choices) != {'A', 'B'}:
            raise ValueError('Exactly two reviewed choices are required.')
        texts.extend(choices.values())
    for text in texts:
        if (not isinstance(text, str) or not text.strip() or len(text) > 160
                or '<|' in text or '|>' in text or any(ord(c) < 32 for c in text)):
            raise ValueError('Guidance must be bounded plain inspection text.')


def advice_request(defect_type, guidance):
    validate_guidance(guidance)
    if defect_type not in guidance['checks']:
        raise ValueError('Unsupported screw defect label.')
    choices = guidance['checks'][defect_type]
    return {'prompt_text': f'Possible screw defect: {defect_type.replace("_", " ")}. Type is provisional. '
                           f'Part remains for manual review.\nA. {choices["A"]}\nB. {choices["B"]}\n'
                           'Choose the best first inspection check. Answer A or B only.',
            'system_text': 'Choose a supplied inspection check. Reply with A or B only.',
            'max_new_tokens': 2}


def resolve_choice(text, defect_type, guidance):
    advice_request(defect_type, guidance)
    # Fail closed on added actions; a model only chooses already-reviewed text.
    match = re.fullmatch(r'\s*([AB])[.!]?\s*', text) if isinstance(text, str) else None
    choice = match.group(1) if match else None
    return {'status': 'REVIEWED_CHOICE' if choice else 'FALLBACK_MANUAL_INSPECTION',
            'choice': choice, 'text': guidance['checks'][defect_type][choice] if choice else guidance['fallback'],
            'guidance_sha256': canonical_hash(guidance), 'approval_status': guidance['approval_status'],
            'quality_grade_authority': 'DETERMINISTIC_RULES_ONLY', 'part_release_authorized': False,
            'type_is_provisional': True, 'root_cause': 'UNKNOWN'}


def grade_screw(score, policy, *, benchmark=False, model=None):
    if (type(benchmark) is not bool or policy.get('schema_version') != 1 or policy.get('plugin_id') != 'screw'
            or policy.get('comparison') != 'strict_greater_than'
            or policy.get('normal_outside_margin_grade') != 'PASS' or policy.get('near_threshold_grade') != 'REVIEW'
            or policy.get('anomaly_grade') != 'REVIEW' or policy.get('physical_part_grade') != 'REVIEW'
            or policy.get('automatic_physical_part_release') is not False
            or policy.get('industrial_certified') is not False
            or policy.get('physical_scale_calibrated') is not False):
        raise ValueError('Unrecognized screw prototype grade policy.')
    threshold, margin = policy['image_threshold'], policy['normal_review_margin_fraction']
    if (type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0
            or type(margin) not in (int, float) or not 0 < margin < 1):
        raise ValueError('Invalid screw grade cutoffs.')
    grade, reason = 'REVIEW', 'PHYSICAL_CAMERA_DOMAIN_UNVALIDATED'
    if type(score) not in (int, float) or not math.isfinite(score):
        reason = 'INVALID_SCORE'
    elif benchmark and (not isinstance(model, dict)
                        or model.get('export_sha256') != policy.get('model_sha256')
                        or model.get('image_threshold') != threshold):
        reason = 'MODEL_OR_THRESHOLD_MISMATCH'
    elif benchmark:
        if score > threshold:
            reason = 'ANOMALY_REQUIRES_MANUAL_INSPECTION'
        elif score >= threshold * (1 - margin):
            reason = 'NEAR_THRESHOLD_NUMERICAL_MARGIN_NOT_CONFIDENCE'
        else:
            grade, reason = 'PASS', 'BENCHMARK_NORMAL_CANDIDATE_ONLY'
    return {'grade': grade, 'reason': reason, 'policy_id': policy['policy_id'],
            'policy_sha256': canonical_hash(policy), 'authority': 'DETERMINISTIC_RULES_ONLY',
            'part_release_authorized': False, 'industrial_certified': False}
