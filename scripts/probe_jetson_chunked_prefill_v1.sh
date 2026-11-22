# ForgeSight chunked-prefill probe - native-chunked-prefill-v1-probe-1
#
# What this does (single shot, read-only):
#   Runs ONE generation on the DEPLOYED Llama chain (128/512 profile) with the
#   full 180-token v4 advisory prompt from the saved-review record, proving the
#   native ExecuTorch runner consumes the whole prompt via chunked prefill.
#   It checks the model/tokenizer hashes and profile first, checks free memory,
#   and refuses to run if an assist_edge_worker is already resident.
#   It writes only under a fresh /tmp directory, installs nothing, stops
#   nothing, and changes no app or model files.
#
# How to run (on the Jetson, one command, no edits):
#   bash probe_chunked_prefill_v1.sh
#
# Expected readout at the end: probe_result.json with
#   "status": "PROBE_PASS_CHUNKED_PREFILL_GENERATED",
#   "prompt_source": "REAL_V4_SAVED_REVIEW", "prompt_token_count": 180,
#   native_stats.prompt_tokens = 180, generated_tokens between 1 and 8,
#   and a nonempty generated_text_display.
#   Status PROBE_PASS_CHUNKED_PREFILL_CAPABILITY_ONLY_SYNTHETIC_PROMPT instead
#   means the real prompt could not be composed on the device and a synthetic
#   prompt ran: capability proven, NOT the real advisory. Any other status is
#   a finding.
#
set -u
OUT=$(mktemp -d /tmp/forgesight_chunked_prefill_probe_XXXXXXXXXX) || exit 1
cat > "$OUT/probe.py" <<'PROBE_PY_EOF'
#!/usr/bin/env python3
"""ForgeSight chunked-prefill isolated probe (native-chunked-prefill-v1).

Single-shot, read-only proof on the DEPLOYED Llama chain (profile 128/512):
the native ExecuTorch 1.5.1 runner consumes a full >128-token advisory
prompt via chunked prefill and generates, with native stats reporting the
full supplied prompt-token count and a nonempty completion.

Bounds: one generation, max_new_tokens=8, no writes outside the output
directory, no installs, no service/UI changes, no file swaps. Aborts
BEFORE loading the model if memory is short, a resident assist_edge_worker
is detected (its RSS would be real and the probe must not add a second
instance), or any pinned hash/profile value mismatches.

Environment overrides exist ONLY as host fixture-test seams; the shipped
command sets none of them: PROBE_LLAMA_ROOT, PROBE_CHECKOUT, PROBE_RECORD,
PROBE_MEMINFO, PROBE_PROC_ROOT, PROBE_EXPECT_MODEL_SHA256,
PROBE_EXPECT_TOKENIZER_SHA256. --preflight-only stops before model load.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

PROBE_VERSION = 'native-chunked-prefill-v1-probe-1'
LLAMA_ROOT = Path(os.environ.get('PROBE_LLAMA_ROOT',
                                 '/home/nvidia/.local/share/forgesight/llama_cpu_v1'))
MODEL = LLAMA_ROOT / 'model' / 'llama3_2_1b_xnnpack.pte'
TOKENIZER = LLAMA_ROOT / 'model' / 'tokenizer.model'
PROFILE = LLAMA_ROOT / 'model' / 'edge_llama.json'
EXPECT_MODEL_SHA256 = os.environ.get(
    'PROBE_EXPECT_MODEL_SHA256',
    'b82d0de5e0785269112d818be5a6c675225aab02c4e2a3b6c98d9e599da86739')
EXPECT_TOKENIZER_SHA256 = os.environ.get(
    'PROBE_EXPECT_TOKENIZER_SHA256',
    '82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55')
CHECKOUT = Path(os.environ.get(
    'PROBE_CHECKOUT', '/home/nvidia/forgesight-feature-audits/candidate-4988df3/checkout'))
RECORD = Path(os.environ.get(
    'PROBE_RECORD',
    '/home/nvidia/forgesight-feature-audits/pr42-96efa80/sample-inputs/records/'
    'saved-review/result/record.json'))
MEMINFO = Path(os.environ.get('PROBE_MEMINFO', '/proc/meminfo'))
PROC_ROOT = Path(os.environ.get('PROBE_PROC_ROOT', '/proc'))
MAX_NEW_TOKENS = 8
# KV cache, activations and runner workspace on top of the 765 MiB artifact.
MEM_HEADROOM_KIB = 1_310_720  # 1.25 GiB


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rss_kib():
    try:
        for line in (PROC_ROOT / 'self' / 'status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def mem_available_kib():
    for line in MEMINFO.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
    raise ValueError('MemAvailable missing from ' + str(MEMINFO))


def resident_worker_pids():
    pids = []
    for entry in PROC_ROOT.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmdline = (entry / 'cmdline').read_bytes()
        except (OSError, PermissionError):
            continue
        if b'assist_edge_worker' in cmdline:
            pids.append(int(entry.name))
    return sorted(pids)


def chat_prompt(text, system_text):
    # Verbatim copy of forgesight.edge_jetson_smoke.chat_prompt (bounded).
    text = text.strip()
    if not text or len(text.encode('utf-8')) > 4096 or '<|' in text or '|>' in text:
        raise ValueError('Prompt must be bounded plain text without chat markers.')
    system = ''
    if system_text is not None:
        system_text = system_text.strip()
        if not system_text or len(system_text.encode('utf-8')) > 2048 \
                or '<|' in system_text or '|>' in system_text:
            raise ValueError('System prompt must be bounded plain text without chat markers.')
        system = '<|start_header_id|>system<|end_header_id|>\n\n' + system_text + '<|eot_id|>'
    return ('<|begin_of_text|>' + system + '<|start_header_id|>user<|end_header_id|>\n\n'
            + text + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')


def preflight(report):
    problems = []
    report['python'] = sys.version.split()[0]
    packages = {}
    # Same pinned trust boundary as forgesight.edge_jetson_smoke.worker.
    expected = {'torch': '2.14.0+cpu', 'executorch': '1.5.1+cpu',
                'pytorch-tokenizers': '1.6.0'}
    for name, pin in expected.items():
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
        if packages[name] != pin:
            problems.append('unexpected %s version: %s (pinned %s)'
                            % (name, packages[name], pin))
    report['packages'] = packages
    pte_kib = MODEL.stat().st_size // 1024
    available = mem_available_kib()
    required = pte_kib + MEM_HEADROOM_KIB
    report['memory'] = {'pte_kib': pte_kib, 'mem_available_kib': available,
                        'required_kib': required}
    if available < required:
        problems.append('MemAvailable %d KiB below required %d KiB' % (available, required))
    workers = resident_worker_pids()
    report['resident_assist_edge_worker_pids'] = workers
    if workers:
        problems.append('resident assist_edge_worker detected (pids %s); not adding a '
                        'second instance' % workers)
    model_sha = sha256(MODEL)
    tokenizer_sha = sha256(TOKENIZER)
    report['artifact_sha256'] = model_sha
    report['tokenizer_sha256'] = tokenizer_sha
    if model_sha != EXPECT_MODEL_SHA256:
        problems.append('artifact sha256 mismatch')
    if tokenizer_sha != EXPECT_TOKENIZER_SHA256:
        problems.append('tokenizer sha256 mismatch')
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    report['profile'] = {'max_seq_length': profile.get('max_seq_length'),
                         'max_context_length': profile.get('max_context_length')}
    if profile.get('max_seq_length') != 128 or profile.get('max_context_length') != 512:
        problems.append('profile is not the deployed 128/512 chain')
    return problems, profile


def compose_prompt(report):
    try:
        sys.path.insert(0, str(CHECKOUT))
        from forgesight.assist_contract import prepare_advisory
        record = json.loads(RECORD.read_text(encoding='utf-8'))
        guidance = json.loads((CHECKOUT / 'guidance' / 'prototype_inspection_v3.json')
                              .read_text(encoding='utf-8'))
        context = prepare_advisory(record, guidance, mode='composed')
        report['prompt_source'] = 'REAL_V4_SAVED_REVIEW'
        report['checkout'] = str(CHECKOUT)
        report['record'] = str(RECORD)
        return context['prompt_text'], context['system_text']
    except Exception as error:
        report['prompt_source'] = 'SYNTHETIC_FALLBACK'
        report['prompt_fallback_reason'] = (type(error).__name__ + ': ' + str(error))[:300]
        text = ' '.join(['inspect the bracket surface for scratches and dents'] * 22)
        return (text,
                'Inspection assistant. Give one short advisory from the supplied facts and '
                'cite the evidence ids.')


def run_generation(report, formatted, prompt_token_count, output_dir):
    import torch
    from executorch.runtime import Runtime
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
    import executorch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    for relative in ('lib/libexecutorch_kernels_quantized.so',
                     'extension/llm/custom_ops/libcustom_ops_aot_lib.so'):
        library = Path(next(iter(executorch.__path__))) / relative
        torch.ops.load_library(str(library))
    runtime = Runtime.get()
    required = ('quantized_decomposed::embedding_4bit.dtype_out',
                'llama::update_cache.out', 'llama::custom_sdpa.out')
    if not all(name in runtime.operator_registry.operator_names for name in required):
        raise ValueError('Required ExecuTorch operators did not register.')
    report['rss_kib_before_load'] = rss_kib()
    started = time.monotonic()
    runner = TextLLMRunner(str(MODEL), str(TOKENIZER))
    report['runner_constructor_seconds'] = round(time.monotonic() - started, 3)
    report['rss_kib_after_load'] = rss_kib()
    # The application worker resets before EVERY request; mirror that here.
    report['runner_reset_called'] = False
    if hasattr(runner, 'reset'):
        runner.reset()
        report['runner_reset_called'] = True
    config = {'echo': False, 'temperature': 0.0, 'warming': False,
              'max_new_tokens': MAX_NEW_TOKENS, 'seq_len': -1, 'num_bos': 0, 'num_eos': 0}
    report['generation_config'] = config
    pieces = []
    stats = []

    def token_callback(piece):
        pieces.append(piece)

    def stats_callback(entry):
        stats.append(json.loads(entry.to_json_string()))

    started = time.monotonic()
    runner.generate(formatted, GenerationConfig(**config), token_callback, stats_callback)
    report['generate_call_seconds'] = round(time.monotonic() - started, 3)
    report['rss_kib_after_generate'] = rss_kib()
    raw = ''.join(pieces)
    # A nonempty check must look past special tokens: raw output that is only
    # special tokens is an empty continuation.
    display = re.sub(r'<\|.*?\|>', '', raw).strip()
    report['generated_text_display'] = display[:160]
    report['callback_chunks'] = len(pieces)
    report['native_stats'] = stats[-1] if stats else None
    problems = []
    if not stats:
        problems.append('no native statistics captured')
    else:
        final = stats[-1]
        if final.get('prompt_tokens') != prompt_token_count:
            problems.append('native prompt_tokens %s != supplied %d'
                            % (final.get('prompt_tokens'), prompt_token_count))
        if not 0 < final.get('generated_tokens', 0) <= MAX_NEW_TOKENS:
            problems.append('native generated_tokens out of bound: %s'
                            % (final.get('generated_tokens'),))
    if not display:
        problems.append('empty generation after removing special tokens')
    return problems


def main():
    output_dir = Path(sys.argv[1])
    preflight_only = '--preflight-only' in sys.argv[2:]
    report = {'probe_version': PROBE_VERSION, 'status': 'PROBE_ERROR',
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        return run(report, output_dir, preflight_only)
    except Exception as error:
        # No bare tracebacks: every failure path leaves a structured readout.
        report['status'] = 'PROBE_ERROR'
        report['problems'] = ['probe raised: ' + type(error).__name__ + ': '
                              + str(error)[:300]]
        (output_dir / 'probe_result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 1


def run(report, output_dir, preflight_only):
    report['status'] = 'PREFLIGHT_BLOCKED'
    problems, profile = preflight(report)
    if problems:
        report['problems'] = problems
        (output_dir / 'probe_result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 2
    try:
        prompt_text, system_text = compose_prompt(report)
        formatted = chat_prompt(prompt_text, system_text)
        from pytorch_tokenizers import CppTiktoken
        tokenizer = CppTiktoken()
        tokenizer.load(str(TOKENIZER))
        ids = tokenizer.encode(formatted, 0, 0)
        count = len(ids)
    except Exception as error:
        report['status'] = 'PROBE_ERROR'
        report['problems'] = ['prompt setup raised: ' + type(error).__name__ + ': '
                              + str(error)[:300]]
        (output_dir / 'probe_result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 1
    report['prompt_token_count'] = count
    report['prompt_sha256'] = hashlib.sha256(formatted.encode('utf-8')).hexdigest()
    problems = []
    if not count > profile['max_seq_length']:
        problems.append('prompt %d tokens does not exceed max_seq_length %d; '
                        'probe proves nothing' % (count, profile['max_seq_length']))
    if count + MAX_NEW_TOKENS > profile['max_context_length']:
        problems.append('prompt %d + %d new exceeds the context envelope %d'
                        % (count, MAX_NEW_TOKENS, profile['max_context_length']))
    if problems:
        report['status'] = 'PROMPT_UNSUITABLE'
        report['problems'] = problems
        (output_dir / 'probe_result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 2
    if preflight_only:
        report['status'] = 'PREFLIGHT_OK_MODEL_NOT_LOADED'
        (output_dir / 'probe_result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 0
    try:
        problems = run_generation(report, formatted, count, output_dir)
    except Exception as error:
        problems = ['generation raised: ' + type(error).__name__ + ': ' + str(error)[:300]]
        report['status'] = 'PROBE_ERROR'
    else:
        if problems:
            report['status'] = 'PROBE_FAIL'
        elif report['prompt_source'] == 'REAL_V4_SAVED_REVIEW':
            report['status'] = 'PROBE_PASS_CHUNKED_PREFILL_GENERATED'
        else:
            # Capability proven, but NOT the real advisory prompt: say so.
            report['status'] = 'PROBE_PASS_CHUNKED_PREFILL_CAPABILITY_ONLY_SYNTHETIC_PROMPT'
    if problems:
        report['problems'] = problems
    (output_dir / 'probe_result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
PROBE_PY_EOF
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
VENV_PY=/home/nvidia/.local/share/forgesight/llama_cpu_v1/venv/bin/python3.12
timeout --signal=TERM --kill-after=10 900 "$VENV_PY" "$OUT/probe.py" "$OUT"
rc=$?
echo "probe exit code: $rc (0=pass, 1=fail, 2=preflight blocked, 124=timeout)"
echo "readout: $OUT/probe_result.json"
