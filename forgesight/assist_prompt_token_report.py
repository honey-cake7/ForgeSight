"""Read-only pinned-tokenizer prompt size report for advisory variants.

Counts tokens with the deployed tokenizer (CppTiktoken over model/tokenizer.model)
for the exact chat-formatted prompts the composed advisory would send, and checks
them against the deployed edge_llama.json bounds. It never loads or runs the model
and never writes outside the requested --output JSON file.

Usage (in the private runtime venv on the device):
    python3 -m forgesight.assist_prompt_token_report \
        --llama-root "$HOME/.local/share/forgesight/llama_cpu_v1" \
        --record <record.json> [--record <record2.json> ...] \
        [--derive-camera-variant <record.json>] --output token_report.json

--derive-camera-variant synthesizes a live-camera measurement variant from a saved
record (source mode swap plus deterministic regrade) so the camera prompt shape can
be measured before any physical camera session exists. Such rows are labeled
SYNTHETIC_MEASUREMENT_DERIVED and are size evidence only, not inspection evidence.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from forgesight.assist_contract import (CAMERA_SOURCE_MODES, DEFAULT_GUIDANCE,
                                        AdvisoryRefusal, prepare_advisory)
from forgesight.edge_jetson_smoke import chat_prompt
from forgesight.grade import grade_record

MAX_NEW_TOKENS = 64  # composed advisory generation cap, matching the deployed config


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def camera_variant(record):
    variant = deepcopy(record)
    variant['source']['mode'] = sorted(CAMERA_SOURCE_MODES)[0]
    variant['source']['physical_part'] = False
    variant['decision'] = grade_record(variant, variant['decision']['policy_snapshot'])
    return variant


def measure_case(name, record, source_path, guidance, token_count, profile, synthetic=False):
    row = {'name': name, 'record_sha256': _sha256(source_path) if source_path else None,
           'input_kind': 'SYNTHETIC_MEASUREMENT_DERIVED' if synthetic else 'RECORD_FILE'}
    try:
        context = prepare_advisory(record, guidance)
    except AdvisoryRefusal as exc:
        row['eligibility'] = exc.code
        return row
    except (KeyError, ValueError) as exc:
        row['eligibility'] = 'INVALID_FIXTURE: ' + str(exc)[:120]
        return row
    formatted = chat_prompt(context['prompt_text'], context['system_text'])
    full = token_count(formatted)
    row.update({
        'eligibility': 'ELIGIBLE',
        'prompt_version': context['prompt_version'],
        'source_mode': context['source_mode'],
        'guide_id': next(key for key in context['evidence'] if key != 'E1'),
        'tokens': {'full_formatted': full,
                   'prompt_text': token_count(context['prompt_text']),
                   'system_text': token_count(context['system_text']),
                   'template_overhead': full - token_count(context['prompt_text'])
                                        - token_count(context['system_text'])},
        'budget': {'max_seq_length': profile['max_seq_length'],
                   'max_context_length': profile['max_context_length'],
                   'max_new_tokens': MAX_NEW_TOKENS,
                   'prefill_ok': 0 < full < profile['max_seq_length'],
                   'context_ok': full + MAX_NEW_TOKENS <= profile['max_context_length']},
    })
    return row


def load_token_counter(llama_root):
    from pytorch_tokenizers import CppTiktoken
    tokenizer = CppTiktoken()
    tokenizer.load(str(Path(llama_root) / 'model/tokenizer.model'))
    return lambda text: len(tokenizer.encode(text, 0, 0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--llama-root', type=Path, required=True)
    parser.add_argument('--record', type=Path, action='append', default=[])
    parser.add_argument('--derive-camera-variant', type=Path, action='append', default=[])
    parser.add_argument('--guidance', type=Path, default=DEFAULT_GUIDANCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.record and not args.derive_camera_variant:
        parser.error('Provide at least one --record or --derive-camera-variant.')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing report; choose a fresh path.')
    root = args.llama_root.resolve()
    profile_path = root / 'model/edge_llama.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    guidance = json.loads(args.guidance.read_text(encoding='utf-8'))
    token_count = load_token_counter(root)
    cases = []
    for path in args.record:
        cases.append(measure_case(path.stem, json.loads(path.read_text(encoding='utf-8')),
                                  path, guidance, token_count, profile))
    for path in args.derive_camera_variant:
        record = json.loads(path.read_text(encoding='utf-8'))
        cases.append(measure_case(path.stem + '__camera_variant', camera_variant(record),
                                  path, guidance, token_count, profile, synthetic=True))
    report = {'schema_version': 1,
              'scope': 'READ_ONLY_TOKEN_COUNT_NO_MODEL_EXECUTION',
              'tokenizer': {'path': str(root / 'model/tokenizer.model'),
                            'sha256': _sha256(root / 'model/tokenizer.model')},
              'profile': {'path': str(profile_path), 'sha256': _sha256(profile_path),
                          'model_id': profile.get('model_id'),
                          'max_seq_length': profile['max_seq_length'],
                          'max_context_length': profile['max_context_length']},
              'guidance': {'path': str(args.guidance), 'sha256': _sha256(args.guidance)},
              'cases': cases}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for row in cases:
        budget = row.get('budget') or {}
        print(f"{row['name']}: {row['eligibility']}"
              + (f" full={row['tokens']['full_formatted']} prefill_ok={budget['prefill_ok']}"
                 if row['eligibility'] == 'ELIGIBLE' else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
