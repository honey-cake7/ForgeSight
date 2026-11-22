#!/usr/bin/env python3
"""One-step read-only advisory prompt measurement for the native device.

Fetch and run from the ForgeSight folder (two short lines, nothing edited):

    git fetch origin pull/42/head
    git show FETCH_HEAD:forgesight_token_measure.py > /tmp/ftm.py && python3 /tmp/ftm.py

The script validates the private llama runtime root (readiness + pinned tokenizer
SHA-256 + pinned binding version), re-executes itself under the runtime venv python,
materializes the reviewed candidate package from FETCH_HEAD into a fresh temporary
directory (nothing user-owned is overwritten or deleted; the app checkout is never
touched), measures every prompt variant with the deployed tokenizer, and writes one
JSON report into a fresh timestamped folder in the current directory. It never loads
or runs the model and never installs anything.
"""

import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

EXPECTED_TOKENIZER_SHA256 = '82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55'
EXPECTED_BINDING = '1.6.0'
PACKAGE_FILES = ('forgesight/__init__.py', 'forgesight/assist_contract.py',
                 'forgesight/assist_vision.py', 'forgesight/grade.py',
                 'forgesight/severity.py', 'forgesight/defect_severity.py',
                 'forgesight/defect_type.py', 'forgesight/assist_prompt_token_report.py',
                 'forgesight/edge_jetson_smoke.py', 'guidance/prototype_inspection_v3.json')
RECORD_GLOBS = ('forgesight-feature-audits/*/*/records/*/result/record.json',
                'forgesight-feature-audits/*/*/records/*/record.json',
                '.local/share/forgesight/*/runs/*/record.json')
MAX_RECORDS = 8
# Guard-preserving trims: notes simply omit the removed detail. noDisclaimer is
# measured for research only and is never a recommended fix variant.
VIABLE_VARIANTS = (('noPct',), ('noLoc',), ('noPct', 'noLoc'))
RESEARCH_ONLY_VARIANTS = (('noDisclaimer',), ('noPct', 'noLoc', 'noDisclaimer'))


def fail(message):
    print('token measure: ' + message, file=sys.stderr)
    raise SystemExit(2)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_llama_root(home):
    base = home / '.local/share/forgesight'
    problems, ready = [], []
    for candidate in sorted(base.glob('*')):
        missing = [rel for rel in ('model/tokenizer.model', 'model/edge_llama.json',
                                   'model/llama3_2_1b_assets.json', 'venv/bin/python')
                   if not (candidate / rel).exists()]
        if missing:
            problems.append(f'{candidate}: missing {", ".join(missing)}')
        else:
            ready.append(candidate)
    if len(ready) != 1:
        fail('need exactly one ready llama runtime under %s; found %d. %s'
             % (base, len(ready), ' | '.join(problems) or 'no candidates'))
    root = ready[0]
    actual = sha256(root / 'model/tokenizer.model')
    if actual != EXPECTED_TOKENIZER_SHA256:
        fail('tokenizer.model sha256 %s does not match the pinned deployed asset %s'
             % (actual, EXPECTED_TOKENIZER_SHA256))
    return root


def reexec_into_venv(root):
    venv_python = root / 'venv/bin/python'
    if os.environ.get('_FORGESIGHT_MEASURE_REEXEC') == '1':
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    env = dict(os.environ, _FORGESIGHT_MEASURE_REEXEC='1', _FORGESIGHT_MEASURE_CWD=os.getcwd())
    os.execve(str(venv_python), [str(venv_python), str(Path(__file__).resolve())], env)


def check_binding():
    import importlib.metadata
    version = importlib.metadata.version('pytorch-tokenizers')
    if version != EXPECTED_BINDING:
        fail('pytorch-tokenizers %s found, need pinned %s' % (version, EXPECTED_BINDING))


def materialize_package(workdir):
    source_dir = Path(os.environ.get('_FORGESIGHT_MEASURE_CWD', os.getcwd()))
    try:
        commit = subprocess.run(['git', '-C', str(source_dir), 'rev-parse', 'FETCH_HEAD'],
                                check=True, capture_output=True, text=True,
                                timeout=30).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        fail('cannot resolve FETCH_HEAD in %s; run the fetch line first from the '
             'ForgeSight folder' % source_dir)
    files = []
    for relative in PACKAGE_FILES:
        try:
            content = subprocess.run(['git', '-C', str(source_dir), 'show', 'FETCH_HEAD:' + relative],
                                     check=True, capture_output=True, timeout=30).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            fail('cannot read %s from FETCH_HEAD in %s; run the fetch line first from '
                 'the ForgeSight folder' % (relative, source_dir))
        target = workdir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files.append({'path': relative, 'source': 'FETCH_HEAD:%s' % relative,
                      'sha256': hashlib.sha256(content).hexdigest()})
    return workdir, {'candidate_commit': commit, 'source_repo': str(source_dir),
                     'package_files': files}


def discover_records(home):
    found = []
    for pattern in RECORD_GLOBS:
        found.extend(glob.glob(str(home / pattern)))
    records = sorted((Path(p) for p in found), key=lambda p: p.stat().st_mtime, reverse=True)
    return records[:MAX_RECORDS]


def build_cases(records):
    from forgesight.assist_contract import AdvisoryRefusal, prepare_advisory
    from forgesight.assist_prompt_token_report import camera_variant
    from forgesight.grade import grade_record
    guidance = json.loads(Path('guidance/prototype_inspection_v3.json').read_text())
    cases, seen = [], set()
    for path in records:
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            context = prepare_advisory(record, guidance)
        except (AdvisoryRefusal, ValueError, KeyError):
            continue
        kind = ('real_reject' if context['authoritative_decision']['grade'] == 'REJECT'
                else 'real_review')
        if kind in seen:
            continue
        seen.add(kind)
        cases.append((kind, record, path, False))
    if 'real_review' not in seen:
        fail('no ELIGIBLE saved-image REVIEW record found under the audit space')
    input_kinds = {expected: 'MISSING_NOT_FOUND_ON_THIS_DEVICE' for expected in
                   ('real_review', 'real_reject')}
    for kind, _, path, _ in cases:
        input_kinds[kind] = str(path)
    review = next(record for kind, record, _, _ in cases if kind == 'real_review')
    review_path = next(path for kind, _, path, _ in cases if kind == 'real_review')
    cases.append(('synthetic_camera_variant', camera_variant(review), review_path, True))

    def regraded(mutate, name):
        from copy import deepcopy
        variant = deepcopy(review)
        mutate(variant)
        variant['decision'] = grade_record(variant, variant['decision']['policy_snapshot'])
        cases.append((name, variant, review_path, True))

    def many_regions(record):
        regions = record['geometry']['regions']
        base = regions[0]
        while len(regions) < 12:
            regions.append(dict(base))
        record['geometry']['total_area_px'] = sum(r['area_px'] for r in regions)

    def wide_digits(record):
        # Widest values the validated geometry profile permits (700x700 policy frame):
        # 5-digit area, 3-digit coordinates, diagonal consistent with the bbox.
        import math
        geometry = record['geometry']
        geometry['regions'][0]['area_px'] = 38009
        geometry['regions'][0]['bbox_xywh_px'] = [500, 300, 199, 191]
        geometry['regions'][0]['bbox_diagonal_px'] = math.hypot(199, 191)
        geometry['total_area_px'] = sum(r['area_px'] for r in geometry['regions'])

    regraded(many_regions, 'synthetic_multiregion_12')
    regraded(wide_digits, 'synthetic_wide_digits')
    for kind, _, path, _ in cases:
        if kind.startswith('synthetic_'):
            input_kinds[kind] = 'DERIVED_FROM:' + str(path)
    return cases, guidance, input_kinds


def main():
    home = Path.home()
    root = find_llama_root(home)
    reexec_into_venv(root)
    check_binding()
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_dir = Path(os.environ.get('_FORGESIGHT_MEASURE_CWD', os.getcwd())) / f'forgesight_token_measure_{stamp}'
    out_dir.mkdir(exist_ok=False)
    workdir = Path(tempfile.mkdtemp(prefix='forgesight_measure_'))
    _, provenance = materialize_package(workdir)
    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    for name in list(sys.modules):
        if name.startswith('forgesight'):
            del sys.modules[name]
    from forgesight.assist_prompt_token_report import (VARIANT_TRANSFORMS, load_token_counter,
                                                       measure_case)
    profile = json.loads((root / 'model/edge_llama.json').read_text(encoding='utf-8'))
    token_count = load_token_counter(root)
    records = discover_records(home)
    cases, guidance, input_kinds = build_cases(records)
    rows = []
    for name, record, path, synthetic in cases:
        rows.append(measure_case(name, record, path, guidance, token_count, profile,
                                 synthetic=synthetic))
        for tags in VIABLE_VARIANTS + RESEARCH_ONLY_VARIANTS:
            row = measure_case(name + '__' + '-'.join(tags), record, path, guidance,
                               token_count, profile, synthetic=synthetic, variant_tags=tags)
            if tags in RESEARCH_ONLY_VARIANTS:
                row['compliance'] = 'NONCOMPLIANT_RESEARCH_ONLY_NEVER_A_FIX_VARIANT'
            else:
                row['compliance'] = 'GUARD_PRESERVING_CANDIDATE'
            rows.append(row)
    viable = [r for r in rows if r.get('compliance') == 'GUARD_PRESERVING_CANDIDATE'
              and r.get('budget', {}).get('prefill_ok')]
    report = {'schema_version': 1, 'llama_root': str(root),
              'tokenizer_sha256': EXPECTED_TOKENIZER_SHA256,
              'binding': EXPECTED_BINDING, 'provenance': provenance,
              'input_kinds': input_kinds, 'cases': rows,
              'recommendation_scope': ('CANDIDATE_MEASURED_ON_LISTED_INPUTS_ONLY' if viable
                                       else 'NO_GUARD_PRESERVING_VARIANT_FITS_THESE_INPUTS')}
    (out_dir / 'token_report.json').write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        budget = row.get('budget') or {}
        status = row['eligibility'] if row['eligibility'] != 'ELIGIBLE' else (
            'full=%d prefill_ok=%s' % (row['tokens']['full_formatted'], budget.get('prefill_ok')))
        print('%-58s %s' % (row['name'], status))
    for kind, detail in sorted(input_kinds.items()):
        if detail.startswith('MISSING'):
            print('input kind %-16s %s' % (kind, detail))
    print('candidate: %s' % provenance['candidate_commit'])
    print('report: %s' % (out_dir / 'token_report.json'))


if __name__ == '__main__':
    main()
