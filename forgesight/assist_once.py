"""Stage 9.1: one saved inspection to real ExecuTorch advice, separate from grading/UI."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from forgesight.assist_contract import (DEFAULT_GUIDANCE, canonical_hash,
                                        parse_advisory, prepare_advisory)
from forgesight.edge_assets import fingerprint
from forgesight.edge_capture import capture_process


DEFAULT_RUNTIME = Path('/root/.cache/forgesight/et1.5.1/venv/bin/python')


def advise_once(record_path: Path, output: Path, *, guidance_path=DEFAULT_GUIDANCE,
                runtime_python=DEFAULT_RUNTIME, timeout=90.0, max_new_tokens=96) -> dict:
    if output.exists():
        raise FileExistsError('Choose a new advice output directory; previous evidence is preserved.')
    if not 0 < timeout <= 300 or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError('Use timeout 0..300 seconds and 1..128 generated tokens.')
    if record_path.stat().st_size > 1024*1024:
        raise ValueError('Inspection metadata exceeds 1 MiB.')
    before_bytes = record_path.read_bytes()
    record = json.loads(before_bytes)
    context = prepare_advisory(record, json.loads(guidance_path.read_text()))
    output.mkdir(parents=True, exist_ok=False)
    raw = output/'native'; raw.mkdir()
    prompt_path = output/'prompt.txt'
    prompt_path.write_text(context['prompt_text'], encoding='utf-8')
    (output/'evidence.json').write_text(json.dumps(context, indent=2)+'\n')
    (output/'input_inspection.json').write_bytes(before_bytes)
    request = {
        'model': str(Path('artifacts/edge/export_attempt_01/exported/llama3_2_1b_xnnpack.pte').resolve()),
        'tokenizer': str(Path('models/llama3_2_1b/original/tokenizer.model').resolve()),
        'profile': str(Path('configs/edge_llama.json').resolve()),
        'reference': str(Path('manifests/llama3_2_1b_export.json').resolve()),
        'assets': str(Path('manifests/llama3_2_1b_assets.json').resolve()),
        'prompt': str(prompt_path.resolve()), 'output': str(raw.resolve()), 'max_new_tokens': max_new_tokens}
    request['system_text'] = context['system_text']
    request_path = raw/'request.json'
    request_path.write_text(json.dumps(request, indent=2)+'\n')
    report = {'schema_version': 1, 'status': 'ADVICE_UNAVAILABLE',
              'scope': 'PC_SINGLE_INSPECTION_ADVISORY_NOT_JETSON_OR_PRODUCTION_PROOF',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'inspection_id': record['inspection_id'], 'inspection_file_sha256': fingerprint(record_path)[1],
              'inspection_sha256': context['inspection_sha256'],
              'decision_sha256': context['decision_sha256'], 'authoritative_grade': record['decision']['grade'],
              'guidance_file_sha256': fingerprint(guidance_path)[1],
              'guidance_approval': 'PROTOTYPE_NOT_FACTORY_APPROVED',
              'advisory': None, 'process': None, 'worker_result': None,
              'runtime_error': None, 'validation_error': None,
              'jetson_verified': False, 'offline_network_disconnection_verified': False,
              'ui_connected': False, 'background_queue_implemented': False,
              'raw_output_path': str(raw/'generated.txt')}
    command = ['env', 'OMP_NUM_THREADS=8', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
               str(runtime_python), '-B', '-m', 'forgesight.edge_pc_smoke', '--worker', str(request_path.resolve())]
    try:
        report['process'] = capture_process(command, raw, timeout)
        worker_file = raw/'worker_result.json'
        if (report['process']['returncode'] != 0 or report['process']['completion_reason'] != 'EXITED'
                or not worker_file.is_file()):
            report['runtime_error'] = f"Runtime did not complete successfully: {report['process']['completion_reason']}"
        else:
            native = json.loads(worker_file.read_text())
            report['worker_result'] = native
            try:
                report['advisory'] = parse_advisory(native['generated_text'], context)
                report['status'] = 'PC_ADVICE_GENERATED_REQUIRES_REVIEW'
            except (ValueError, TypeError, KeyError) as exc:
                report['validation_error'] = str(exc)
    except OSError as exc:
        report['runtime_error'] = f'{type(exc).__name__}: {exc}'
    if record_path.read_bytes() != before_bytes or canonical_hash(record['decision']) != context['decision_sha256']:
        raise RuntimeError('Original inspection changed; refusing to report a verified advice result.')
    report['original_inspection_bytes_unchanged'] = True
    report['source_sha256'] = {name: fingerprint(Path(name))[1] for name in
                              ('forgesight/assist_contract.py', 'forgesight/assist_once.py', 'forgesight/edge_pc_smoke.py')}
    with (output/'advice_report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guidance', type=Path, default=DEFAULT_GUIDANCE)
    parser.add_argument('--runtime-python', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    report = advise_once(args.record, args.output, guidance_path=args.guidance,
                         runtime_python=args.runtime_python, timeout=args.timeout)
    print(json.dumps({key: report[key] for key in ('status', 'authoritative_grade', 'advisory', 'runtime_error', 'validation_error')}, indent=2))
    if report['status'] != 'PC_ADVICE_GENERATED_REQUIRES_REVIEW':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
