"""Compact advisory evidence and output guards; deterministic grades remain authoritative."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from forgesight.assist_vision import (VISION_GUIDE_ID, VISION_OUTPUT_POLICY, VISION_PROMPT_VERSION,
                                     vision_context, vision_note)
from forgesight.grade import grade_record
from forgesight.severity import severity_assessment


DEFAULT_GUIDANCE = Path('guidance/prototype_inspection_v3.json')
PROMPT_VERSION = 'forgeassist-system-cited-note-v4'
OUTPUT_POLICY_VERSION = 'forgeassist-evidence-composed-note-v6'
# Canonical document hash, independent of JSON whitespace/Windows line endings.
# A changed procedure requires deliberate review/versioning, not just the same ID.
GUIDANCE_DOCUMENT_SHA256 = '9ecdbcc80cb93d8e3b853a4bcb590110ab65b473c4c7d221b280fa89cbca6744'

# Grammatical glue permitted in a composed note in addition to words that
# appear in the prompt itself. No negators, release/adjust/repair verbs or
# causal vocabulary here: a note cannot assemble an action or cause the
# supplied evidence does not contain.
CONNECTOR_WORDS = {
    'an', 'a', 'the', 'is', 'are', 'was', 'has', 'been', 'in', 'this', 'its',
    'of', 'with', 'to', 'part', 'requires', 'needs', 'so', 'remain', 'remains',
    'please', 'should', 'be', 'on', 'detected', 'unusual', 'appearance',
    'further', 'and', 'by', 'for', 'region', 'regions',
}

SEVERITY_WORDS = {'none', 'low', 'medium', 'high'}

# Camera-sourced records carry the deterministic REVIEW grade with reason
# CAMERA_DOMAIN_UNVALIDATED; their advice must state that unvalidated domain.
CAMERA_SOURCE_MODES = frozenset(
    {'LIVE_CAMERA', 'LIVE_CAMERA_DIAGNOSTIC_NO_PART', 'LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC'})


def _guide_id_for(source_mode, grade):
    """Expected guidance entry from the validated record source and grade only.

    Shared by prepare and parse so both sides always agree; the model never
    influences which entry is expected.
    """
    if grade == 'REJECT':
        return 'G2'
    if source_mode in CAMERA_SOURCE_MODES:
        return 'G3'
    return 'G1'


def _entry_scope_matches(entry, source_mode):
    modes = entry.get('source_modes')
    if modes:
        return source_mode in modes
    return source_mode not in CAMERA_SOURCE_MODES


class AdvisoryRefusal(ValueError):
    """An explicit host refusal before requesting any language-model execution."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def canonical_hash(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _format_fixed6(value):
    return f'{value:.6f}'


def _location_words(bbox, size):
    """Host-computed coarse location; the model never derives its own."""
    x, y, w, h = bbox
    width, height = size
    cx, cy = x + w / 2, y + h / 2
    x_word = 'left' if cx < width / 3 else 'right' if cx >= 2 * width / 3 else 'central'
    y_word = 'upper' if cy < height / 3 else 'lower' if cy >= 2 * height / 3 else 'central'
    if x_word == 'central' and y_word == 'central':
        return 'central'
    if x_word == 'central':
        return y_word
    if y_word == 'central':
        return x_word
    return f'{y_word} {x_word}'


LABEL_WORDS = {'score', 'threshold', 'area', 'width', 'height', 'x', 'y', 'percent'}
UNIT_WORDS = {'px', 'region', 'regions'}


def _value_mentions(text):
    """(label, value, unit) triples for every numeric token.

    The unit comes from the token itself (a trailing %) or the word right
    after it (px, region, regions); the label is the nearest preceding
    non-connector label word (score, threshold, area, ...). A supplied value
    attached to a different quantity or unit in the note than in the prompt
    does not match, so area pixels cannot be restated as a score and no unit
    can be added or dropped.
    """
    tokens = re.findall(r'[a-z]+|\d+(?:\.\d+)?%?', text.lower())
    mentions = set()
    for index, token in enumerate(tokens):
        if not token[0].isdigit():
            continue
        value = token.rstrip('%')
        unit = '%' if token.endswith('%') else ''
        if not unit and index + 1 < len(tokens) and tokens[index + 1] in UNIT_WORDS:
            unit = tokens[index + 1]
        label = None
        back = index - 1
        while back >= 0:
            word = tokens[back]
            if word in CONNECTOR_WORDS:
                back -= 1
                continue
            label = word if word in LABEL_WORDS else None
            break
        mentions.add((label, value, unit))
    return mentions


def prepare_advisory(record: dict, guidance: dict, *, mode: str = 'composed') -> dict:
    try:
        return _prepare_advisory(record, guidance, mode=mode)
    except AdvisoryRefusal:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AdvisoryRefusal('INVALID_OR_INCOMPLETE_EVIDENCE',
                              'Inspection or guidance metadata is invalid or incomplete.') from exc


def _prepare_advisory(record: dict, guidance: dict, *, mode: str = 'composed') -> dict:
    """Only trusted numerical/enumerated facts enter the prompt; free-text IDs/notes do not."""
    if mode not in ('composed', 'vision_choice'):
        raise ValueError('Unknown advisory mode.')
    if not isinstance(record, dict) or not record or not isinstance(guidance, dict):
        raise AdvisoryRefusal('INVALID_OR_INCOMPLETE_EVIDENCE', 'Expected inspection and guidance objects.')
    if record.get('status') != 'ANALYZED':
        raise AdvisoryRefusal('INSPECTION_NOT_ANALYZED', 'No defect advice is requested for a failed or incomplete inspection.')
    if record.get('candidate') == 'NORMAL_CANDIDATE':
        raise AdvisoryRefusal('UNSUPPORTED_NORMAL_CANDIDATE', 'The normal result keeps its deterministic action; no Llama call is needed.')
    if record.get('candidate') != 'ANOMALY_CANDIDATE':
        raise AdvisoryRefusal('UNSUPPORTED_CANDIDATE', 'Unrecognized anomaly evidence; advice is withheld.')
    decision = record['decision']
    source_mode = record.get('source', {}).get('mode')
    if (decision['grade'] not in ('REVIEW', 'REJECT')
            or (decision['grade'] == 'REJECT'
                and source_mode != 'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD')
            or record['predicted_defect_type'] != 'unknown'
            or record['type_status'] != 'NO_VALIDATED_SUBTYPE_CLASSIFIER'):
        raise AdvisoryRefusal('UNSUPPORTED_TYPE_OR_GRADE', 'The grade or defect type is outside the reviewed advisory scope.')
    if grade_record(record, decision['policy_snapshot']) != decision:
        raise AdvisoryRefusal('INCONSISTENT_DECISION', 'Decision does not match its recorded deterministic evidence/policy.')
    camera = (decision['grade'] == 'REVIEW'
              and decision['reason'] == 'CAMERA_DOMAIN_UNVALIDATED'
              and source_mode in CAMERA_SOURCE_MODES)
    benchmark = (source_mode == 'USER_DECLARED_MVTEC_METAL_NUT_UPLOAD'
                 and record['source'].get('physical_part') is False
                 and record['source'].get('dataset_origin_verified') is False
                 and record['source'].get('size_px') == [700, 700]
                 and decision['grade'] == 'REJECT'
                 and decision['reason'] == 'BENCHMARK_SCORE_ABOVE_IMAGE_THRESHOLD')
    supported_reason = (benchmark or camera or
                        (decision['grade'] == 'REVIEW' and
                         (decision['reason'] == 'UNKNOWN_UNSEEN_OR_UNRELIABLE_TYPE'
                          or (source_mode == 'USER_UPLOADED_IMAGE_DIAGNOSTIC'
                              and record['source'].get('physical_part') is False
                              and decision['reason'] == 'UPLOADED_IMAGE_DOMAIN_UNVALIDATED'))))
    if not supported_reason:
        raise AdvisoryRefusal('INVALID_EVIDENCE_FOR_ADVICE', 'Invalid evidence cannot be presented as an analyzed defect event.')
    if canonical_hash(guidance) != GUIDANCE_DOCUMENT_SHA256:
        raise AdvisoryRefusal('UNRECOGNIZED_GUIDANCE', 'Guidance content differs from the reviewed prototype document; advice is withheld.')
    if mode == 'vision_choice':
        return _prepare_vision_advisory(record, guidance, source_mode, decision)
    if record.get('assessment') is not None:
        raise AdvisoryRefusal('UNSUPPORTED_ASSESSED_RECORD',
                              'Record carries a provisional classifier assessment; request the vision-choice advisory mode explicitly.')
    guide_id = _guide_id_for(source_mode, decision['grade'])
    entries = [entry for entry in guidance['entries']
               if entry['candidate'] == record['candidate'] and entry['grade'] == decision['grade']
               and _entry_scope_matches(entry, source_mode)]
    if len(entries) != 1 or entries[0]['id'] != guide_id:
        raise ValueError('No single permitted guidance entry matches this inspection.')
    entry = entries[0]
    if not isinstance(entry['text'], str) or not entry['text'].strip() or len(entry['text']) > 160 or '<|' in entry['text'] or '|>' in entry['text']:
        raise ValueError('Prototype guidance is empty, excessive or contains chat control markers.')
    geometry = record['geometry']
    area = geometry['total_area_px']
    score = record['score']
    if type(area) is not int or area <= 0 or type(score) not in (int, float) or not math.isfinite(score):
        raise ValueError('Expected finite model evidence and a positive response area.')
    severity = severity_assessment(record, policy=decision['policy_snapshot'])
    if severity['severity'] == 'NONE':
        raise AdvisoryRefusal('INVALID_EVIDENCE_FOR_ADVICE', 'An anomaly without flagged response evidence cannot be presented for advice.')
    regions = geometry['regions']
    largest = max(regions, key=lambda region: region['area_px'])
    width, height = geometry['source_size_px']
    area_percent = f'{area / (width * height) * 100:.3f}'
    score_text = _format_fixed6(score)
    threshold_text = _format_fixed6(record['model']['image_threshold'])
    location = _location_words(largest['bbox_xywh_px'], geometry['source_size_px'])
    # E1 aliases the full immutable record outside the tight model prefill budget.
    # Numbers are response pixels, not physical damage or probabilities.
    camera_line = ('[E1] Source: live camera frame; camera domain unvalidated, not a verified physical part.\n'
                   if camera else '')
    prompt = (
        ('[E1] Anomaly flagged; provisional benchmark grade REJECT; type and cause unknown.\n'
         if benchmark else '[E1] Anomaly flagged; grade REVIEW; type and cause unknown.\n')
        + camera_line
        + f"[E1] Severity {severity['severity']}, pixel-response, not physical damage.\n"
        + f'[E1] Score {score_text} vs threshold {threshold_text}; not a probability.\n'
        + f'[E1] Response area {area} px, {area_percent}% of frame, across {len(regions)} '
        + ('region' if len(regions) == 1 else 'regions')
        + (f'; largest region width {largest["bbox_xywh_px"][2]} px height {largest["bbox_xywh_px"][3]} px '
           f'at x {largest["bbox_xywh_px"][0]} y {largest["bbox_xywh_px"][1]}, {location}.\n')
        + f'[{guide_id}] Prototype checklist: {entry["text"]}'
    )
    return {'schema_version': 1, 'prompt_version': PROMPT_VERSION, 'output_policy_version': OUTPUT_POLICY_VERSION,
            'source_mode': source_mode,
            'inspection_id': record['inspection_id'], 'inspection_sha256': canonical_hash(record),
            'authoritative_decision': deepcopy(decision), 'decision_sha256': canonical_hash(decision),
            'evidence': {'E1': {'inspection_id': record['inspection_id'], 'record_sha256': canonical_hash(record),
                               'score': score, 'image_threshold': record['model']['image_threshold'],
                               'candidate': record['candidate'], 'predicted_defect_type': 'unknown',
                               'severity': decision['severity'],
                               'pixel_response_severity': severity,
                               'total_response_area_px': area,
                               'region_count': len(regions),
                               'largest_region': {'bbox_xywh_px': largest['bbox_xywh_px'],
                                                  'area_px': largest['area_px'],
                                                  'location': location},
                               'measurement': 'model_response_not_verified_physical_damage'},
                         guide_id: {'document_id': guidance['document_id'], 'document_sha256': canonical_hash(guidance),
                                    'approval_status': guidance['approval_status'], 'text': entry['text']}},
            'recurrence': 'NOT_SUPPLIED_NO_PATTERN_ANALYSIS', 'prompt_text': prompt,
            'system_text': ('Inspection assistant. Give one short advisory from the supplied facts and '
                            'values exactly; never invent measurements, types, causes or actions. State '
                            'unknown type and cause, severity, and the checklist. Cite [E1][' + guide_id + '].'
                            + (' State the camera domain is unvalidated.' if camera else '')),
            'model_received_fields': (['camera_domain_unvalidated'] if camera else [])
                                     + ['anomaly_flag', 'fixed_' + decision['grade'] + '_grade',
                                        'unknown_type_and_cause', 'pixel_response_severity',
                                        'score_vs_threshold', 'response_area_fraction_and_region_count',
                                        'largest_region_bbox_and_location', 'prototype_' + guide_id + '_text'],
            'score_and_geometry_in_prompt': True,
            'image_supplied_to_llama': False, 'quality_grade_authority': 'DETERMINISTIC_RULES_ONLY'}


def _prepare_vision_advisory(record, guidance, source_mode, decision):
    """Reviewed A/B check choice for records with a valid provisional benchmark assessment."""
    vision = vision_context(record)
    if vision is None:
        raise AdvisoryRefusal('UNSUPPORTED_VISION_EVIDENCE',
                              'The vision-choice advisory requires a valid provisional benchmark assessment.')
    facts, choices, prompt = vision
    return {'schema_version': 1, 'prompt_version': VISION_PROMPT_VERSION,
            'output_policy_version': VISION_OUTPUT_POLICY,
            'source_mode': source_mode,
            'inspection_id': record['inspection_id'], 'inspection_sha256': canonical_hash(record),
            'authoritative_decision': deepcopy(decision), 'decision_sha256': canonical_hash(decision),
            'evidence': {'E1': {'inspection_id': record['inspection_id'], 'record_sha256': canonical_hash(record),
                                'candidate': record['candidate'], 'predicted_defect_type': 'unknown',
                                'vision_assessment': facts},
                         VISION_GUIDE_ID: {'document_id': VISION_OUTPUT_POLICY,
                                           'approval_status': guidance['approval_status'],
                                           'actions': choices}},
            'prompt_text': prompt,
            'system_text': 'Choose the best first image inspection check. Reply with A or B only.',
            'model_received_fields': ['provisional_type', 'provisional_severity', 'approximate_location',
                                      'response_extent', 'reviewed_action_choices'],
            'score_and_geometry_in_prompt': False, 'geometry_in_prompt': True, 'max_new_tokens': 2,
            'image_supplied_to_llama': False, 'quality_grade_authority': 'DETERMINISTIC_RULES_ONLY'}


def _base_advisory(context, note, evidence_ids):
    return {'schema_version': 1, 'status': 'GENERATED_REQUIRES_HUMAN_REVIEW',
            'inspection_id': context['inspection_id'], 'inspection_sha256': context['inspection_sha256'],
            'authoritative_grade': context['authoritative_decision']['grade'],
            'decision_sha256': context['decision_sha256'],
            'response': {'generated_note': note, 'evidence_ids': evidence_ids,
                         'possible_explanation': None,
                         'possible_explanation_origin': 'HOST_WITHHELD_NO_CAUSAL_EVIDENCE'},
            'guidance_approval': 'PROTOTYPE_NOT_FACTORY_APPROVED',
            'output_policy_version': context['output_policy_version'],
            'validation_scope': 'HOST_COMPOSED_EVIDENCE_STATEMENTS_NOT_GENERAL_REASONING_OR_DIAGNOSIS',
            'quality_grade_authority': 'DETERMINISTIC_RULES_ONLY',
            'model_saw_image': False, 'jetson_verified': False}


def parse_advisory(raw: str, context: dict) -> dict:
    """The host owns JSON; the model composes a short cited note, never a grade.

    Composition is bounded: every value mention must match a supplied
    quantity, value and unit exactly, every content word must come from the
    prompt or the connector whitelist, the checklist must be restated in
    order, the supplied severity must be named without contradiction, and
    unknown type/cause stays explicit.
    """
    cleaned = raw.strip()
    if cleaned.endswith('<|eot_id|>'):
        cleaned = cleaned[:-len('<|eot_id|>')].strip()
    if '<|' in cleaned or '|>' in cleaned:
        raise ValueError('Unexpected generated control markers.')
    if context['output_policy_version'] == VISION_OUTPUT_POLICY:
        if not re.fullmatch(r'[AB]\.?', cleaned):
            raise ValueError('Expected one reviewed vision action choice.')
        choice = cleaned[0]
        actions = context['evidence'][VISION_GUIDE_ID]['actions']
        advisory = _base_advisory(context, vision_note(choice, actions), ['E1', VISION_GUIDE_ID])
        advisory['response']['note_origin'] = 'LLAMA_SELECTED_REVIEWED_CHECK'
        advisory['response']['model_action_choice'] = choice
        return advisory
    if not 1 <= len(cleaned) <= 800 or any(mark in cleaned for mark in ('{', '}', '```')):
        raise ValueError('Expected a bounded plain-text advisory, not JSON or formatting instructions.')
    citations = re.findall(r'\[([^\]]*)\]', cleaned)
    guide_id = _guide_id_for(context['source_mode'], context['authoritative_decision']['grade'])
    if set(citations) != {'E1', guide_id}:
        raise ValueError('Advisory cites unknown or missing evidence.')
    text = re.sub(r'\[(E1|G1|G2|G3)\]', '', cleaned)
    prompt_mentions = _value_mentions(context['prompt_text'])
    note_mentions = _value_mentions(text)
    if not note_mentions:
        raise ValueError('The note cites none of the supplied measurements or values.')
    if not note_mentions <= prompt_mentions:
        raise ValueError('The note includes measurements or values that were not supplied, '
                         'or attaches a supplied value to a different quantity or unit.')
    prompt_words = set(re.findall(r'[a-z]+', context['prompt_text'].lower()))
    permitted = prompt_words | CONNECTOR_WORDS
    words = re.findall(r'[a-z]+', text.lower())
    if not 10 <= len(words) <= 90:
        raise ValueError('Expected a concise 10..90-word advisory.')
    if not set(words) <= permitted:
        raise ValueError('Generated note includes words/facts outside the permitted evidence/checklist.')
    word_set = set(words)
    if not {'unknown', 'type', 'cause'} <= word_set:
        raise ValueError('Required uncertainty statement is missing.')
    # Every clause mentioning cause or type may conclude only "unknown",
    # whatever the copula: clause-local content words beyond connectors and
    # cause/type itself are limited to the unknown statement, so "the cause
    # remains lighting" or "lighting is the cause" cannot assert a cause the
    # evidence does not contain.
    for clause in re.split(r'[.;!?\n]+', text.lower()):
        clause_words = re.findall(r'[a-z]+', clause)
        if not ({'cause', 'type'} & set(clause_words)):
            continue
        content = [word for word in clause_words
                   if word not in CONNECTOR_WORDS and word not in ('cause', 'type')]
        if 'unknown' not in content or not set(content) <= {'unknown', 'defect'}:
            raise ValueError('Generated note asserts a cause or type the evidence does not contain.')
    severity = context['evidence']['E1']['pixel_response_severity']['severity'].lower()
    if severity not in word_set or (SEVERITY_WORDS - {severity}) & word_set:
        raise ValueError('The note omits or contradicts the supplied severity.')
    if not word_set & {'score', 'area'}:
        raise ValueError('The note references none of the supplied size, location or score evidence.')
    if guide_id == 'G3' and not {'camera', 'unvalidated'} <= word_set:
        raise ValueError('The advisory omits the required unvalidated-camera-domain statement.')
    # The checklist is a procedure: its words must be restated in order
    # (connectors may be interleaved), not merely present as a word set whose
    # reordering could invert the instruction.
    guidance_seq = re.findall(r'[a-z]+', context['evidence'][guide_id]['text'].lower())
    note_iter = iter(words)
    if not all(any(word == needed for word in note_iter) for needed in guidance_seq):
        raise ValueError('The required checklist step is missing or restated out of order.')
    normalized = ' '.join(text.lower().split()).replace(' .', '.')
    if context['authoritative_decision']['grade'] == 'REJECT' and 'grade review' in normalized:
        raise ValueError('Generated note contradicts the benchmark grade.')
    if context['authoritative_decision']['grade'] == 'REVIEW' and ('benchmark' in word_set or 'grade reject' in normalized):
        raise ValueError('Generated note contradicts the review grade.')
    return _base_advisory(context, cleaned, ['E1', guide_id])
