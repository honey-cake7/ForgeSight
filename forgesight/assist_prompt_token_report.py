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

import re

# Guard-preserving trim variants, measured in the SAME step as the unmodified v4
# candidate so a deeper prompt is never a second device round trip. Each transform
# only removes optional clause text from the built prompt; a later contract change
# implementing a chosen variant must emit byte-identical text and update the parser
# pins. Disclaimer removal needs explicit auditor sign-off before any contract use.
VARIANT_TRANSFORMS = {
    'noPct': lambda text: re.sub(r', \d+\.\d+% of frame', '', text),
    'noLoc': lambda text: re.sub(r'; largest region [^\n]*', '', text),
    'noDisclaimer': lambda text: text.replace(', not physical damage', '').replace('; not a probability', ''),
}


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


def measure_case(name, record, source_path, guidance, token_count, profile, synthetic=False,
                 variant_tags=()):
    row = {'name': name, 'record_sha256': _sha256(source_path) if source_path else None,
           'input_kind': ('VARIANT_TRIM_MEASUREMENT' if variant_tags else
                          'SYNTHETIC_MEASUREMENT_DERIVED' if synthetic else 'RECORD_FILE')}
    if variant_tags:
        row['variant'] = list(variant_tags)
        if synthetic:
            row['derived_from'] = 'SYNTHETIC_MEASUREMENT_DERIVED'
    try:
        context = prepare_advisory(record, guidance)
    except AdvisoryRefusal as exc:
        row['eligibility'] = exc.code
        return row
    except (KeyError, ValueError) as exc:
        row['eligibility'] = 'INVALID_FIXTURE: ' + str(exc)[:120]
        return row
    prompt_text = context['prompt_text']
    for tag in variant_tags:
        prompt_text = VARIANT_TRANSFORMS[tag](prompt_text)
    formatted = chat_prompt(prompt_text, context['system_text'])
    full = token_count(formatted)
    row.update({
        'eligibility': 'ELIGIBLE',
        'prompt_version': context['prompt_version'],
        'source_mode': context['source_mode'],
        'guide_id': next(key for key in context['evidence'] if key != 'E1'),
        'tokens': {'full_formatted': full,
                   'prompt_text': token_count(prompt_text),
                   'system_text': token_count(context['system_text']),
                   'template_overhead': full - token_count(prompt_text)
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
    parser.add_argument('--variant', action='append', default=[],
                        help='Comma-separated trim tags (%s); each use adds one measured '
                             'variant row per case, labeled VARIANT_TRIM_MEASUREMENT.'
                             % ','.join(sorted(VARIANT_TRANSFORMS)))
    args = parser.parse_args(argv)
    variant_sets = []
    for spec in args.variant:
        tags = tuple(spec.split(','))
        unknown = [tag for tag in tags if tag not in VARIANT_TRANSFORMS]
        if unknown:
            parser.error('Unknown variant tag(s): ' + ','.join(unknown))
        variant_sets.append(tags)
    if not args.record and not args.derive_camera_variant:
        parser.error('Provide at least one --record or --derive-camera-variant.')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing report; choose a fresh path.')
    root = args.llama_root.resolve()
    profile_path = root / 'model/edge_llama.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    guidance = json.loads(args.guidance.read_text(encoding='utf-8'))
    token_count = load_token_counter(root)
    inputs = []
    for path in args.record:
        inputs.append((path.stem, json.loads(path.read_text(encoding='utf-8')), path, False))
    for path in args.derive_camera_variant:
        record = json.loads(path.read_text(encoding='utf-8'))
        inputs.append((path.stem + '__camera_variant', camera_variant(record), path, True))
    cases = []
    for name, record, path, synthetic in inputs:
        cases.append(measure_case(name, record, path, guidance, token_count, profile,
                                  synthetic=synthetic))
        for tags in variant_sets:
            cases.append(measure_case(name + '__variant_' + '-'.join(tags), record, path,
                                      guidance, token_count, profile, synthetic=synthetic,
                                      variant_tags=tags))
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
