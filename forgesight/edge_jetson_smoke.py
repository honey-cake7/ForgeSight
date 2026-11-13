"""Bounded Llama/ExecuTorch generation on an identified Jetson Orin Nano."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import site
import subprocess
import sys
import time


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(path, expected):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected['bytes']:
        raise ValueError('Missing or wrong-sized asset: ' + path.name)
    actual = sha256(path)
    if actual != expected['sha256']:
        raise ValueError('Asset checksum mismatch: ' + path.name)
    return {'bytes': path.stat().st_size, 'sha256': actual}


def chat_prompt(text, system_text=None):
    text = text.strip()
    if not text or len(text.encode('utf-8')) > 4096 or '<|' in text or '|>' in text:
        raise ValueError('Prompt must be bounded plain text without chat markers.')
    system = ''
    if system_text is not None:
        system_text = system_text.strip()
        if not system_text or len(system_text.encode('utf-8')) > 2048 or '<|' in system_text or '|>' in system_text:
            raise ValueError('System prompt must be bounded plain text without chat markers.')
        system = '<|start_header_id|>system<|end_header_id|>\n\n' + system_text + '<|eot_id|>'
    return ('<|begin_of_text|>' + system + '<|start_header_id|>user<|end_header_id|>\n\n'
            + text + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')


def validate_budget(prompt_tokens, new_tokens, profile):
    if not 1 <= new_tokens <= 64:
        raise ValueError('Initial Jetson smoke limit is 1-64 new tokens.')
    if not 0 < prompt_tokens < profile['max_seq_length']:
        raise ValueError('Prompt exceeds exported prefill bound.')
    if prompt_tokens + new_tokens > profile['max_context_length']:
        raise ValueError('Prompt plus answer exceeds exported context.')


def verify_target(root):
    root = Path(root).resolve()
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise ValueError('Actual Jetson Orin Nano ARM64 device required.')
    model = Path('/proc/device-tree/model').read_bytes().decode(errors='replace').strip('\0')
    if 'Jetson Orin Nano' not in model:
        raise ValueError('Actual Jetson Orin Nano ARM64 device required.')
    if (platform.python_version() != '3.12.14' or Path(sys.prefix).resolve() != root/'venv'
            or Path(sys.base_prefix).resolve() != root/'python' or site.ENABLE_USER_SITE is not False):
        raise ValueError('Use the checked private Jetson Python 3.12.14 venv.')
    marker = json.loads((root/'.forgesight-runtime.json').read_text())
    if marker.get('purpose') != 'FORGESIGHT_STAGE_2_9_CPU_RUNTIME' or marker.get('prefix') != str(root):
        raise ValueError('Runtime root is not the Stage 2.9 verified prefix.')
    return {'machine': platform.machine(), 'model': model, 'python': platform.python_version(),
            'venv': str(root/'venv'), 'system_python': subprocess.check_output(
                ['/usr/bin/python3', '--version'], text=True, timeout=5).strip()}


def verify_inputs(request):
    root = Path(request['root']).resolve()
    host = verify_target(root)
    # 'prompt' belongs to one smoke generation, not to asset verification:
    # callers that only verify and load (the persistent worker) have none yet.
    files = {name: Path(request[name]).resolve()
             for name in ('profile', 'export', 'assets', 'prompt', 'model', 'tokenizer')
             if name in request}
    if files['model'].parent != root/'model' or files['tokenizer'].parent != root/'model':
        raise ValueError('Model/tokenizer must stay in the private runtime model directory.')
    profile = json.loads(files['profile'].read_text())
    export = json.loads(files['export'].read_text())
    assets = json.loads(files['assets'].read_text())
    if (profile['model_id'] != export['model_id'] or profile['model_id'] != assets['model_id']
            or profile['model_revision'] != export['model_revision']
            or profile['model_revision'] != assets['model_revision']
            or profile['executorch_commit'] != export['executorch_commit']
            or profile['backend'] != 'xnnpack' or export['backend_requested'] != 'xnnpack'
            or export['quantization']['qmode'] != '8da4w'):
        raise ValueError('Model/export/profile provenance differs.')
    if sha256(files['profile']) != export['inputs_sha256']['configs/edge_llama.json']:
        raise ValueError('Profile differs from the exported model input.')
    if sha256(files['assets']) != export['inputs_sha256']['manifests/llama3_2_1b_assets.json']:
        raise ValueError('Tokenizer asset reference differs from export input.')
    model_fingerprint = fingerprint(files['model'], export['artifact'])
    tokenizer_fingerprint = fingerprint(files['tokenizer'], assets['files']['original/tokenizer.model'])
    return host, profile, files, model_fingerprint, tokenizer_fingerprint


def worker(request):
    host, profile, files, model_hash, tokenizer_hash = verify_inputs(request)
    if importlib.metadata.version('torch') != '2.14.0+cpu':
        raise ValueError('Unexpected Torch version.')
    if importlib.metadata.version('executorch') != '1.5.1+cpu':
        raise ValueError('Unexpected ExecuTorch version.')
    if importlib.metadata.version('pytorch-tokenizers') != '1.6.0':
        raise ValueError('Unexpected tokenizer binding version.')
    from executorch.runtime import Runtime
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
    import executorch
    import executorch.extension.llm.runner._llm_runner as native
    from pytorch_tokenizers import CppTiktoken
    import torch

    if torch.version.cuda is not None:
        raise ValueError('Expected the separate CPU Torch build.')
    threads = request['threads']
    torch.set_num_threads(threads)
    library_records = []
    for relative in ('lib/libexecutorch_kernels_quantized.so',
                     'extension/llm/custom_ops/libcustom_ops_aot_lib.so'):
        library = Path(next(iter(executorch.__path__))) / relative
        torch.ops.load_library(str(library))
        library_records.append({'relative_path': relative, 'sha256': sha256(library)})
    runtime = Runtime.get()
    required = ('quantized_decomposed::embedding_4bit.dtype_out',
                'llama::update_cache.out', 'llama::custom_sdpa.out')
    if not all(name in runtime.operator_registry.operator_names for name in required):
        raise ValueError('Required ExecuTorch operators did not register.')
    tokenizer = CppTiktoken()
    tokenizer.load(str(files['tokenizer']))
    prompt_text = files['prompt'].read_text(encoding='utf-8')
    system_text = Path(request['system_prompt']).read_text(encoding='utf-8') if request.get('system_prompt') else None
    formatted = chat_prompt(prompt_text, system_text)
    ids = tokenizer.encode(formatted, 0, 0)
    headers = 3 if system_text is not None else 2
    if (ids[0] != 128000 or ids.count(128000) != 1 or ids.count(128006) != headers
            or ids.count(128007) != headers or ids.count(128009) != headers-1):
        raise ValueError('Unexpected Llama chat tokenization.')
    validate_budget(len(ids), request['max_new_tokens'], profile)
    config = {'echo': False, 'temperature': 0.0, 'warming': False,
              'max_new_tokens': request['max_new_tokens'], 'seq_len': -1,
              'num_bos': 0, 'num_eos': 0}
    output = Path(request['output'])
    report = {'status': 'INPUTS_VERIFIED_GENERATION_PENDING', 'host': host,
              'model_id': profile['model_id'], 'model_revision': profile['model_revision'],
              'executorch_source_commit': profile['executorch_commit'],
              'backend_requested': profile['backend'], 'quantization': export_quantization(files['export']),
              'artifact': model_hash, 'tokenizer': tokenizer_hash,
              'native_runner_sha256': sha256(native.__file__), 'kernel_libraries': library_records,
              'packages': {name: importlib.metadata.version(name) for name in
                           ('torch', 'executorch', 'pytorch-tokenizers')},
              'prompt_text': prompt_text.strip(), 'prompt_token_count': len(ids),
              'system_prompt_sha256': sha256(request['system_prompt']) if system_text is not None else None,
              'generation_config': config, 'torch_threads': torch.get_num_threads(),
              'offline_status': 'NOT_VERIFIED_CONNECTED_WIFI', 'stats': []}
    (output/'inputs.json').write_text(json.dumps(report, indent=2)+'\n')
    started = time.monotonic()
    runner = TextLLMRunner(str(files['model']), str(files['tokenizer']))
    report['runner_constructor_seconds'] = time.monotonic() - started
    pieces = []
    with (output/'generated.txt').open('x', encoding='utf-8') as stream:
        def token_callback(piece):
            pieces.append(piece)
            stream.write(piece)
            stream.flush()

        def stats_callback(stats):
            report['stats'].append(json.loads(stats.to_json_string()))

        started = time.monotonic()
        runner.generate(formatted, GenerationConfig(**config), token_callback, stats_callback)
        report['generate_call_seconds'] = time.monotonic() - started
    raw_text = ''.join(pieces)
    if not raw_text.strip() or not report['stats']:
        raise ValueError('No model output/native statistics captured.')
    stats = report['stats'][-1]
    if stats['prompt_tokens'] != len(ids) or not 0 < stats['generated_tokens'] <= request['max_new_tokens']:
        raise ValueError('Native token statistics do not match request.')
    report.update({'status': 'JETSON_EXECUTORCH_LLAMA_GENERATION_VERIFIED',
                   'generated_text_raw': raw_text,
                   'generated_text_display': raw_text.replace('<|eot_id|>', '').strip(),
                   'callback_chunks': len(pieces), 'model_loaded': True})
    (output/'worker_result.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def export_quantization(path):
    return json.loads(Path(path).read_text())['quantization']


def capture(request, timeout):
    output = Path(request['output'])
    output.mkdir(parents=False, exist_ok=False)
    (output/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': str(request['threads']), 'PYTHONNOUSERSITE': '1',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'PYTHONDONTWRITEBYTECODE': '1'})
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(output/'request.json')]
    started = time.monotonic()
    peak = 0
    reason = 'EXITED'
    with (output/'stdout.log').open('xb') as stdout, (output/'stderr.log').open('xb') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                   cwd=Path(request['root']), start_new_session=True)
        while process.poll() is None:
            try:
                content = Path('/proc/{}/status'.format(process.pid)).read_text()
                match = re.search(r'^VmRSS:\s+(\d+)\s+kB$', content, re.MULTILINE)
                if match:
                    peak = max(peak, int(match.group(1)))
            except OSError:
                pass
            if time.monotonic() - started >= timeout:
                reason = 'TIMEOUT'
            if stdout.tell() + stderr.tell() > 2 * 1024 * 1024:
                reason = 'OUTPUT_LIMIT'
            if reason != 'EXITED':
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                break
            time.sleep(0.1)
    result_file = output/'worker_result.json'
    report = {'status': 'FAILED_NOT_VERIFIED', 'process_returncode': process.returncode,
              'completion_reason': reason, 'process_wall_seconds': time.monotonic()-started,
              'sampled_peak_child_rss_kib': peak,
              'memory_note': 'One child process sampled at 0.1 s; may miss peaks and excludes other processes.',
              'worker_result': None, 'offline_status': 'NOT_VERIFIED_CONNECTED_WIFI'}
    if reason == 'EXITED' and process.returncode == 0 and result_file.is_file():
        report['worker_result'] = json.loads(result_file.read_text())
        if report['worker_result']['status'] == 'JETSON_EXECUTORCH_LLAMA_GENERATION_VERIFIED':
            report['status'] = report['worker_result']['status']
    (output/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=False)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--prompt', type=Path)
    parser.add_argument('--system-prompt', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = worker(json.loads(args.worker.read_text()))
        print(result['status'])
        return
    required = ('root', 'model', 'tokenizer', 'profile', 'export', 'assets', 'prompt', 'output')
    if any(getattr(args, key) is None for key in required):
        parser.error('Root, model, tokenizer, profile, export, assets, prompt and output are required.')
    if not 1 <= args.threads <= 4 or not 1 <= args.max_new_tokens <= 64 or not 0 < args.timeout <= 300:
        parser.error('Use 1-4 threads, 1-64 tokens and timeout 0-300 seconds.')
    # Reject a wrong device/environment before creating any output directory.
    verify_target(args.root)
    request = {name: str(getattr(args, name).resolve()) for name in required}
    if args.system_prompt is not None:
        request['system_prompt'] = str(args.system_prompt.resolve())
    request.update({'threads': args.threads, 'max_new_tokens': args.max_new_tokens})
    report = capture(request, args.timeout)
    print(json.dumps({'status': report['status'], 'completion_reason': report['completion_reason'],
                      'process_wall_seconds': report['process_wall_seconds'],
                      'sampled_peak_child_rss_kib': report['sampled_peak_child_rss_kib'],
                      'generated_text': (report['worker_result'] or {}).get('generated_text_display')}, indent=2))
    if report['status'] != 'JETSON_EXECUTORCH_LLAMA_GENERATION_VERIFIED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
