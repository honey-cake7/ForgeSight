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


def canonical_hash(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def prepare_advisory(record: dict, guidance: dict) -> dict:
    """Only trusted numerical/enumerated facts enter the prompt; free-text IDs/notes do not."""
    if record.get('status') != 'ANALYZED' or record.get('candidate') != 'ANOMALY_CANDIDATE':
        raise ValueError('This first advisory step requires an analyzed anomaly; no inference is requested for incomplete evidence.')
    decision = record['decision']
    if (decision['grade'] != 'REVIEW' or record['predicted_defect_type'] != 'unknown'
            or record['type_status'] != 'NO_VALIDATED_SUBTYPE_CLASSIFIER'):
        raise ValueError('Expected the current unknown-type REVIEW prototype, not a different grading/type policy.')
    if grade_record(record, decision['policy_snapshot']) != decision:
        raise ValueError('Decision does not match its recorded deterministic evidence/policy.')
    if (guidance.get('schema_version') != 1 or guidance.get('approval_status') != 'PROTOTYPE_NOT_FACTORY_APPROVED'
            or guidance.get('document_id') != 'forgesight-prototype-image-review-v1'):
        raise ValueError('Expected explicitly labelled, versioned prototype guidance.')
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
    return {'schema_version': 1, 'prompt_version': PROMPT_VERSION,
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
    return {'schema_version': 1, 'status': 'GENERATED_REQUIRES_HUMAN_REVIEW',
            'inspection_id': context['inspection_id'], 'inspection_sha256': context['inspection_sha256'],
            'authoritative_grade': context['authoritative_decision']['grade'],
            'decision_sha256': context['decision_sha256'],
            'response': {'generated_note': cleaned, 'evidence_ids': ['E1', 'G1'],
                         'possible_explanation': None,
                         'possible_explanation_origin': 'HOST_WITHHELD_NO_CAUSAL_EVIDENCE'},
            'guidance_approval': 'PROTOTYPE_NOT_FACTORY_APPROVED',
            'validation_scope': 'HOST_SCHEMA_CITATION_AND_RESTRICTED_VOCABULARY_NOT_SEMANTIC_PROOF',
            'quality_grade_authority': 'DETERMINISTIC_RULES_ONLY',
            'model_saw_image': False, 'jetson_verified': False}
