"""Predeclared PC advisory/refusal checks; fixtures and native runs remain distinct."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from forgesight.assist_contract import (AdvisoryRefusal, OUTPUT_POLICY_VERSION,
                                        parse_advisory, prepare_advisory)
from forgesight.assist_once import DEFAULT_RUNTIME, advise_once
from forgesight.edge_assets import fingerprint


DEFAULT_PROTOCOL = Path('configs/forgeassist_scenarios_v1.json')


def write_json(path, document):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')


def scenario_inputs(case, sources, guidance):
    record = deepcopy(sources.get(case['record'], {}))
    guide = deepcopy(guidance)
    for change in case.get('changes', []):
        target = record if change['target'] == 'record' else guide
        for key in change['keys'][:-1]:
            target = target[key]
        if change.get('remove'):
            del target[change['keys'][-1]]
        else:
            target[change['keys'][-1]] = change['value']
    return record, guide


def evaluate(protocol_path, output, *, run_native=False, runtime_python=DEFAULT_RUNTIME):
    """Save protocol/provenance before any check; never edit source records or databases."""
    if output.exists():
        raise FileExistsError('Choose a fresh evaluation directory; existing evidence is preserved.')
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol['schema_version'] != 1 or protocol['protocol_id'] != 'forgeassist-refusal-scenarios-v1':
        raise ValueError('Unrecognized advisory evaluation protocol.')
    sources, protected = {}, {}
    for name, reference in protocol['record_sources'].items():
        path = Path(reference['path'])
        protected[str(path)] = fingerprint(path)[1]
        document = json.loads(path.read_text(encoding='utf-8'))
        for key in reference['keys']:
            document = document[key]
        sources[name] = document
    guidance_path = Path(protocol['guidance'])
    guidance = json.loads(guidance_path.read_text(encoding='utf-8'))
    protected[str(guidance_path)] = fingerprint(guidance_path)[1]
    output.mkdir(parents=True, exist_ok=False)
    (output/'protocol.json').write_bytes(protocol_bytes)
    declaration = {'created_utc': datetime.now(timezone.utc).isoformat(),
                   'protocol_sha256': fingerprint(protocol_path)[1],
                   'output_policy_version': OUTPUT_POLICY_VERSION,
                   'native_execution_requested': run_native,
                   'protected_source_sha256': protected,
                   'code_sha256': {p: fingerprint(Path(p))[1] for p in
                                   ('forgesight/assist_evaluate.py', 'forgesight/assist_contract.py',
                                    'forgesight/assist_once.py', 'forgesight/edge_pc_smoke.py')}}
    write_json(output/'declaration.json', declaration)
    report = {'schema_version': 1, 'scope': protocol['scope'], 'declaration': declaration,
              'eligibility_cases': [], 'output_fixture_cases': [], 'runtime_cases': [],
              'jetson_verified': False, 'offline_disconnection_verified': False,
              'general_advisory_usefulness_verified': False,
              'no_vision_inference_requested': True,
              'limitation': protocol['acceptance']['quality_limit']}
    contexts, saved_inputs = {}, {}
    for case in protocol['eligibility_cases']:
        record, guide = scenario_inputs(case, sources, guidance)
        case_root = output/case['id']
        case_root.mkdir()
        write_json(case_root/'record.json', record)
        write_json(case_root/'guidance.json', guide)
        saved_inputs[case['id']] = (case_root/'record.json', case_root/'guidance.json')
        row = {'id': case['id'], 'expected': case['expected'],
               'input_kind': 'MODIFIED_TEST_FIXTURE' if case.get('changes') or case['record'] == 'empty'
                             else 'EXISTING_REAL_RECORD', 'runtime_requested': False}
        try:
            context = prepare_advisory(record, guide)
            contexts[case['id']] = context
            row['actual'] = 'ELIGIBLE'
            if case.get('same_prompt_as'):
                reference = contexts[case['same_prompt_as']]
                row['excluded_metadata_prompt_unchanged'] = all(context[k] == reference[k]
                                                               for k in ('prompt_text', 'system_text'))
        except AdvisoryRefusal as exc:
            row['actual'] = exc.code
            # Run the real controller refusal path too: no subprocess should be requested.
            result = advise_once(case_root/'record.json', case_root/'refusal',
                                 guidance_path=case_root/'guidance.json', runtime_python=runtime_python)
            row['runtime_requested'] = result['runtime_requested']
            row['process'] = result['process']
            row['refusal_report_sha256'] = fingerprint(case_root/'refusal/advice_report.json')[1]
            row['unavailable_without_advisory'] = result['status'] == 'ADVICE_UNAVAILABLE' and result['advisory'] is None
            row['controller_refusal_matches'] = result['input_refusal']['code'] == exc.code
        row['matches_expectation'] = (row['actual'] == row['expected'] and not row['runtime_requested']
                                      and row.get('excluded_metadata_prompt_unchanged', True)
                                      and row.get('unavailable_without_advisory', True)
                                      and row.get('controller_refusal_matches', True))
        report['eligibility_cases'].append(row)
    for case in protocol['output_cases']:
        row = {'id': case['id'], 'scope': 'HANDWRITTEN_FIXTURE_NOT_LLAMA_OUTPUT',
               'expected': case['expected'], 'text': case['text']}
        try:
            parse_advisory(case['text'], contexts['real-anomaly'])
            row['actual'] = 'ACCEPT'
        except ValueError as exc:
            row['actual'] = 'REFUSE'
            row['validation_error'] = str(exc)
        row['matches_expectation'] = row['actual'] == row['expected']
        report['output_fixture_cases'].append(row)
    if run_native:
        for case in protocol['runtime_cases']:
            print(f"Checking {case['id']}...", flush=True)
            record_path, guide_path = saved_inputs[case['input_case']]
            runtime = output/'intentionally_missing_python' if case.get('runtime') == 'MISSING' else runtime_python
            case_root = output/case['id']
            result = advise_once(record_path, case_root, guidance_path=guide_path,
                                 runtime_python=runtime, timeout=case['timeout'],
                                 max_new_tokens=case['max_new_tokens'])
            native = result['worker_result']
            row = {'id': case['id'], 'expected': case['expected'], 'actual': result['status'],
                   'report_path': str(case_root/'advice_report.json'),
                   'report_sha256': fingerprint(case_root/'advice_report.json')[1],
                   'completed_native_generation': native is not None,
                   'process': result['process'], 'runtime_error': result['runtime_error'],
                   'validation_error': result['validation_error'],
                   'advisory': result['advisory'],
                   'original_inspection_bytes_unchanged': result['original_inspection_bytes_unchanged']}
            if native:
                row['native_observation'] = {key: native[key] for key in
                                             ('model_id', 'model_revision', 'artifact', 'tokenizer',
                                              'packages', 'host', 'prompt_token_count', 'generation_config',
                                              'generated_text', 'stats')}
            kind = case.get('failure_kind')
            failure_matches = (not kind or kind in ('runtime_error', 'validation_error') and bool(result[kind])
                               or kind == 'TIMEOUT' and result['process']['completion_reason'] == 'TIMEOUT')
            row['matches_expectation'] = (row['actual'] == row['expected'] and failure_matches
                                         and result['original_inspection_bytes_unchanged']
                                         and (row['actual'] != 'ADVICE_UNAVAILABLE' or result['advisory'] is None))
            report['runtime_cases'].append(row)
    all_rows = report['eligibility_cases'] + report['output_fixture_cases'] + report['runtime_cases']
    report['source_files_unchanged'] = all(fingerprint(Path(p))[1] == digest for p, digest in protected.items())
    report['protocol_unchanged'] = protocol_path.read_bytes() == protocol_bytes
    report['matched_expectations'] = sum(row['matches_expectation'] for row in all_rows)
    report['checks_executed'] = len(all_rows)
    report['completed_native_generations'] = sum(row['completed_native_generation'] for row in report['runtime_cases'])
    report['unique_native_input_prompts'] = len({(json.loads((Path(row['report_path']).parent/'evidence.json').read_text())['prompt_text'])
                                               for row in report['runtime_cases'] if row['completed_native_generation']})
    report['passed'] = (report['matched_expectations'] == len(all_rows)
                        and report['source_files_unchanged'] and report['protocol_unchanged'])
    write_json(output/'evaluation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-native', action='store_true', help='Opt into the existing PC-only ExecuTorch worker.')
    parser.add_argument('--runtime-python', type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()
    report = evaluate(args.protocol, args.output, run_native=args.run_native, runtime_python=args.runtime_python)
    print(json.dumps({key: report[key] for key in ('passed', 'matched_expectations', 'checks_executed',
                                                  'completed_native_generations', 'jetson_verified')}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
