"""Compact advisory evidence and output guards; deterministic grades remain authoritative."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from forgesight.grade import grade_record


DEFAULT_GUIDANCE = Path('guidance/prototype_inspection_v1.json')
PROMPT_VERSION = 'forgeassist-system-cited-note-v1'
OUTPUT_POLICY_VERSION = 'forgeassist-reviewed-note-forms-v2'
# Canonical document hash, independent of JSON whitespace/Windows line endings.
# A changed procedure requires deliberate review/versioning, not just the same ID.
GUIDANCE_DOCUMENT_SHA256 = '60238ff09a2453786e81e1a9fd1e905d53ca3b5c5ad7c368c8a2972fe31c7825'


class AdvisoryRefusal(ValueError):
    """An explicit host refusal before requesting any language-model execution."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def canonical_hash(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def prepare_advisory(record: dict, guidance: dict) -> dict:
    try:
        return _prepare_advisory(record, guidance)
    except AdvisoryRefusal:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AdvisoryRefusal('INVALID_OR_INCOMPLETE_EVIDENCE',
                              'Inspection or guidance metadata is invalid or incomplete.') from exc


def _prepare_advisory(record: dict, guidance: dict) -> dict:
    """Only trusted numerical/enumerated facts enter the prompt; free-text IDs/notes do not."""
    if not isinstance(record, dict) or not record or not isinstance(guidance, dict):
        raise AdvisoryRefusal('INVALID_OR_INCOMPLETE_EVIDENCE', 'Expected inspection and guidance objects.')
    if record.get('status') != 'ANALYZED':
        raise AdvisoryRefusal('INSPECTION_NOT_ANALYZED', 'No defect advice is requested for a failed or incomplete inspection.')
    if record.get('candidate') == 'NORMAL_CANDIDATE':
        raise AdvisoryRefusal('UNSUPPORTED_NORMAL_CANDIDATE', 'The normal result keeps its deterministic action; no Llama call is needed.')
    if record.get('candidate') != 'ANOMALY_CANDIDATE':
        raise AdvisoryRefusal('UNSUPPORTED_CANDIDATE', 'Unrecognized anomaly evidence; advice is withheld.')
    decision = record['decision']
    if (decision['grade'] != 'REVIEW' or record['predicted_defect_type'] != 'unknown'
            or record['type_status'] != 'NO_VALIDATED_SUBTYPE_CLASSIFIER'):
        raise AdvisoryRefusal('UNSUPPORTED_TYPE_OR_GRADE', 'Expected the current unknown-type REVIEW prototype, not a different grading/type policy.')
    if grade_record(record, decision['policy_snapshot']) != decision:
        raise AdvisoryRefusal('INCONSISTENT_DECISION', 'Decision does not match its recorded deterministic evidence/policy.')
    if decision['reason'] != 'UNKNOWN_UNSEEN_OR_UNRELIABLE_TYPE':
        raise AdvisoryRefusal('INVALID_EVIDENCE_FOR_ADVICE', 'Invalid evidence cannot be presented as an analyzed defect event.')
    if canonical_hash(guidance) != GUIDANCE_DOCUMENT_SHA256:
        raise AdvisoryRefusal('UNRECOGNIZED_GUIDANCE', 'Guidance content differs from the reviewed prototype document; advice is withheld.')
    entries = [entry for entry in guidance['entries'] if entry['candidate'] == record['candidate'] and entry['grade'] == decision['grade']]
    if len(entries) != 1 or entries[0]['id'] != 'G1':
        raise ValueError('No single permitted guidance entry matches this inspection.')
    entry = entries[0]
    if not isinstance(entry['text'], str) or not entry['text'].strip() or len(entry['text']) > 160 or '<|' in entry['text'] or '|>' in entry['text']:
        raise ValueError('Prototype guidance is empty, excessive or contains chat control markers.')
    area = record['geometry']['total_area_px']
    score = record['score']
    if type(area) is not int or area <= 0 or type(score) not in (int, float) or not math.isfinite(score):
        raise ValueError('Expected finite model evidence and a positive response area.')
    # E1 aliases the full immutable record outside the tight model prefill budget.
    # Numbers are response pixels, not physical damage or probabilities.
    prompt = (
        '[E1] Anomaly flagged; grade REVIEW; defect type and cause unknown.\n'
        f'[G1] Prototype checklist: {entry["text"]}'
    )
    return {'schema_version': 1, 'prompt_version': PROMPT_VERSION, 'output_policy_version': OUTPUT_POLICY_VERSION,
            'inspection_id': record['inspection_id'], 'inspection_sha256': canonical_hash(record),
            'authoritative_decision': deepcopy(decision), 'decision_sha256': canonical_hash(decision),
            'evidence': {'E1': {'inspection_id': record['inspection_id'], 'record_sha256': canonical_hash(record),
                               'score': score, 'image_threshold': record['model']['image_threshold'],
                               'candidate': record['candidate'], 'predicted_defect_type': 'unknown',
                               'severity': decision['severity'], 'total_response_area_px': area,
                               'measurement': 'model_response_not_verified_physical_damage'},
                         'G1': {'document_id': guidance['document_id'], 'document_sha256': canonical_hash(guidance),
                                'approval_status': guidance['approval_status'], 'text': entry['text']}},
            'recurrence': 'NOT_SUPPLIED_NO_PATTERN_ANALYSIS', 'prompt_text': prompt,
            'system_text': 'You are an inspection assistant. Reply with only one short operator instruction. State unknown type and cause. Use only supplied facts. Cite [E1][G1].',
            'model_received_fields': ['anomaly_flag', 'fixed_REVIEW_grade', 'unknown_type_and_cause', 'prototype_G1_text'],
            'score_and_geometry_in_prompt': False,
            'image_supplied_to_llama': False, 'quality_grade_authority': 'DETERMINISTIC_RULES_ONLY'}


def parse_advisory(raw: str, context: dict) -> dict:
    """The host owns JSON; the model supplies a short cited note, never a grade."""
    cleaned = raw.strip()
    if cleaned.endswith('<|eot_id|>'):
        cleaned = cleaned[:-len('<|eot_id|>')].strip()
    if '<|' in cleaned or '|>' in cleaned:
        raise ValueError('Unexpected generated control markers.')
    if not 1 <= len(cleaned) <= 600 or any(mark in cleaned for mark in ('{', '}', '```')):
        raise ValueError('Expected a bounded plain-text advisory, not JSON or formatting instructions.')
    citations = re.findall(r'\[([^\]]*)\]', cleaned)
    if set(citations) != {'E1', 'G1'}:
        raise ValueError('Advisory cites unknown or missing evidence.')
    text = re.sub(r'\[(E1|G1)\]', '', cleaned)
    if re.search(r'\d', text):
        raise ValueError('The compact note is not permitted to invent measurements or confidence.')
    words = re.findall(r'[a-z]+', text.lower())
    if not 10 <= len(words) <= 50:
        raise ValueError('Expected a concise 10..50-word advisory.')
    permitted = set(re.findall(r'[a-z]+', context['prompt_text'].lower())) | {
        'anomaly', 'flagged', 'type', 'cause', 'unknown', 'grade', 'review', 'defect',
        'an', 'a', 'the', 'is', 'are', 'was', 'has', 'been', 'in', 'this', 'its', 'of',
        'with', 'to', 'part', 'requires', 'needs', 'so', 'remain', 'remains', 'please',
        'should', 'be', 'on', 'detected', 'unusual', 'appearance', 'further'}
    if not set(words) <= permitted:
        raise ValueError('Generated note includes words/facts outside the permitted evidence/checklist.')
    if not {'unknown', 'type', 'cause', 'review'} <= set(words) or not set(words) & {'clarity', 'lighting'}:
        raise ValueError('Required uncertainty, fixed review or image-verification step is missing.')
    # Vocabulary alone accepts "the cause is lighting" using only source words.
    # This deliberately narrow profile admits two reviewed complete statements,
    # allowing whitespace/case/citation placement only, rather than arbitrary
    # recombinations. A new wording/procedure needs a separately reviewed policy.
    normalized = ' '.join(text.lower().split())
    reviewed_forms = {
        'anomaly flagged; grade review; defect type and cause unknown. '
        'prototype checklist: hold for human review; check image clarity and lighting.',
        'an anomaly is flagged; type and cause are unknown. '
        'hold for human review and check image clarity and lighting.',
    }
    # Citations may precede a final period or follow it.
    normalized = normalized.replace(' .', '.')
    if normalized not in reviewed_forms:
        raise ValueError('Generated note is outside the reviewed statement forms; advice is withheld.')
    return {'schema_version': 1, 'status': 'GENERATED_REQUIRES_HUMAN_REVIEW',
            'inspection_id': context['inspection_id'], 'inspection_sha256': context['inspection_sha256'],
            'authoritative_grade': context['authoritative_decision']['grade'],
            'decision_sha256': context['decision_sha256'],
            'response': {'generated_note': cleaned, 'evidence_ids': ['E1', 'G1'],
                         'possible_explanation': None,
                         'possible_explanation_origin': 'HOST_WITHHELD_NO_CAUSAL_EVIDENCE'},
            'guidance_approval': 'PROTOTYPE_NOT_FACTORY_APPROVED',
            'output_policy_version': OUTPUT_POLICY_VERSION,
            'validation_scope': 'HOST_REVIEWED_STATEMENT_FORMS_NOT_GENERAL_REASONING_OR_DIAGNOSIS',
            'quality_grade_authority': 'DETERMINISTIC_RULES_ONLY',
            'model_saw_image': False, 'jetson_verified': False}
