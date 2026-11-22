"""Bounded post-export native acceptance for one staged Llama/ExecuTorch chain.

Runs AFTER a re-exported artifact chain (profile, export manifest, PTE) has been
staged into the private Jetson runtime root. One command exercises the native
behavior that a JSON manifest alone cannot prove for the new chain:

- metadata: executes the artifact's own get_max_seq_len/get_max_context_len
  methods (edge_artifact only lists method names; execution was NOT_RUN) and
  binds them to the staged profile values;
- regression: the pinned smoke prompt still verifies, with the recorded
  baseline prompt-token count, on the new artifact;
- boundary: a prompt fitted to exactly max_context_length - 2 tokens
  generates with max_new_tokens=2, and one fitted to max_context_length - 1
  is refused by the application's budget guard BEFORE generation (the native
  runner's chunked prefill accepts prompts beyond max_seq_length; the
  context envelope is the contract);
- replay: real saved REVIEW/REJECT inspections with measured prompts above
  128 tokens verify through the trusted cold path, proving actual >128
  chunked prefill on the artifact under test (any chain whose context
  envelope covers the prompt plus answer can run them);
- camera: an optional camera inspection record verifies the same way, labeled
  by the record's own stored source mode; absent, it stays UNASSESSED;
- warm/cold: the persistent worker's startup/warmup evidence plus one warm
  advisory compared byte-for-byte against the cold path through the existing
  assist_worker_acceptance gate.

Scope honesty: this proves native prefill/generation behavior of ONE artifact
on ONE device. It does not validate physical part inspection, vision defect
detection, or real camera capture unless a genuinely camera-captured record
was supplied. RSS figures are sampled point/peak values, not peaks guaranteed.
Timings are measurements on this device, not performance promises.

The driver is a stdlib-only controller. Device steps run as children under
the private runtime venv interpreter (the same trust boundary as
edge_jetson_smoke): --child metadata and --child fit-prompt. Nothing outside
the fresh --output directory is created, and no repository manifest is edited.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import time

VERIFIED_SMOKE = 'JETSON_EXECUTORCH_LLAMA_GENERATION_VERIFIED'
VERIFIED_ADVICE = 'JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW'
REFUSAL_TEXT = 'Prompt plus answer exceeds exported context.'
# Boundary runs use this generation budget; the boundary pair derives from it.
BOUNDARY_MAX_NEW = 2
FAILED_STATUS = 'FAILED_NOT_VERIFIED'

BASE_BOUNDARY_PROMPT = ('Describe one lighting property that matters when '
                        'comparing two inspection images.')
# Padding words for exact-token fitting. The primary filler is expected to be
# one token each for the deployed tokenizer; the fine fillers are oscillation
# breakers in case a filler tokenizes to more than one token (parity trap).
FILLER_WORD = 'stable'
FINE_FILLERS = ('steady', 'even', 'bright', 'uniform', 'mild', 'plain')
MAX_FIT_ENCODES = 4096

METADATA_METHODS = ('get_max_seq_len', 'get_max_context_len')

PASSED = 'JETSON_NATIVE_ACCEPTANCE_PASSED'
FAILED = 'JETSON_NATIVE_ACCEPTANCE_FAILED'
INCOMPLETE = 'JETSON_NATIVE_ACCEPTANCE_INCOMPLETE_NOT_VERIFIED'


def _reject_non_finite(value):
    """json.loads parse_constant hook: NaN/Infinity is not evidence."""
    raise ValueError('Non-finite number in native output: ' + value)


def _load_json_strict(path):
    return json.loads(Path(path).read_text(), parse_constant=_reject_non_finite)


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Exact-token prompt fitting (pure; the tokenizer is injected)
# ---------------------------------------------------------------------------

def fit_prompt_to_target(target, encode, base_text=BASE_BOUNDARY_PROMPT,
                         filler=FILLER_WORD, fine_fillers=FINE_FILLERS,
                         max_encodes=MAX_FIT_ENCODES):
    """Return (text, token_count) whose chat-formatted token count == target.

    encode(text) must return the token count of the FINAL chat-formatted
    prompt for the supplied user text (forgesight.edge_jetson_smoke.chat_prompt
    plus the deployed tokenizer on the device; a stub in tests). The budget
    guard counts that formatted length, so the fit must be against it.

    Every step is MEASURED against the real tokenizer: greedy append of the
    candidate word whose re-measured count comes closest to the target without
    exceeding it, then a bounded single-word substitution search when every
    candidate overshoots (a BPE filler may tokenize to more than one token).
    Fails closed rather than ever returning an inexact prompt.
    """
    if type(target) is not int or not 8 <= target <= 1024:
        raise ValueError('Fit target must be an integer in 8..1024.')
    base = base_text.strip()
    if not base or len(base.encode('utf-8')) > 4096 or '<|' in base or '|>' in base:
        raise ValueError('Fit base prompt must be bounded plain text.')
    candidates = (filler,) + tuple(fine_fillers)
    words = []
    count = encode(base)
    if count > target:
        raise ValueError('Base prompt alone exceeds the fit target.')
    encodes = 0
    stalls = 0
    while count != target and encodes < max_encodes:
        best = None
        for word in candidates:
            encodes += 1
            measured = encode(' '.join([base] + words + [word]))
            if measured <= target and (best is None or measured > best[1]):
                best = (word, measured)
        if best is not None:
            words.append(best[0])
            count = best[1]
            stalls = 0
            continue
        # Every append overshoots: substitute one placed word for a different
        # candidate, keeping the count at or below the target. A substitution
        # that does not raise the count is a stall; a few stalls are allowed
        # to change the word mix, then the fit fails closed.
        substituted = False
        for index in range(len(words) - 1, -1, -1):
            for word in candidates:
                if word == words[index]:
                    continue
                encodes += 1
                trial = words[:index] + [word] + words[index + 1:]
                measured = encode(' '.join([base] + trial))
                if measured <= target:
                    if measured > count:
                        stalls = 0
                    else:
                        stalls += 1
                    words = trial
                    count = measured
                    substituted = True
                    break
            if substituted or encodes >= max_encodes:
                break
        if not substituted or stalls > 3:
            raise ValueError('Prompt fit did not converge on the exact token count.')
    if count != target:
        raise ValueError('Prompt fit did not converge on the exact token count.')
    text = ' '.join([base] + words)
    if encode(text) != target:
        raise ValueError('Prompt fit verification failed on the final text.')
    return text, target


# ---------------------------------------------------------------------------
# Pure check evaluators (host-tested; no device imports)
# ---------------------------------------------------------------------------

def check_metadata(report, profile):
    """Executed artifact metadata must match the staged profile exactly."""
    methods = report.get('methods') or {}
    detail = {'status': 'PASS', 'expected': {
        'get_max_seq_len': profile['max_seq_length'],
        'get_max_context_len': profile['max_context_length']},
        'executed': {}, 'routes': {}}
    problems = []
    for name in METADATA_METHODS:
        entry = methods.get(name) or {}
        value = entry.get('value')
        detail['executed'][name] = value
        detail['routes'][name] = entry.get('route')
        if value != detail['expected'][name]:
            problems.append('{}={!r}, expected {!r}'.format(
                name, value, detail['expected'][name]))
    if report.get('status') != 'METADATA_EXECUTED':
        problems.append('metadata child status: {!r}'.format(report.get('status')))
    if problems:
        detail['status'] = 'FAIL'
        detail['problems'] = problems
    return detail


def check_generation(result, expected_prompt_tokens=None):
    """A smoke generation must be natively verified; prompt count must match."""
    detail = {'status': 'PASS'}
    run = result.get('run')
    worker = (run or {}).get('worker_result') or {}
    observed = worker.get('prompt_token_count')
    detail['prompt_token_count'] = observed
    detail['generated_tokens'] = ((worker.get('stats') or [{}])[-1] or {}).get('generated_tokens')
    detail['process_wall_seconds'] = (run or {}).get('process_wall_seconds')
    detail['sampled_peak_child_rss_kib'] = (run or {}).get('sampled_peak_child_rss_kib')
    problems = []
    if result.get('returncode') != 0 or (run or {}).get('status') != VERIFIED_SMOKE:
        problems.append('generation not verified: status {!r}, returncode {!r}'.format(
            (run or {}).get('status'), result.get('returncode')))
    if expected_prompt_tokens is not None and observed != expected_prompt_tokens:
        problems.append('prompt token count {!r}, expected {!r}'.format(
            observed, expected_prompt_tokens))
    if problems:
        detail['status'] = 'FAIL'
        detail['problems'] = problems
    return detail


def check_refusal(result):
    """The prefill guard must refuse BEFORE any generation output exists."""
    run = result.get('run')
    detail = {'status': 'PASS',
              'completion_reason': (run or {}).get('completion_reason'),
              'generated_file_exists': bool(result.get('generated_exists'))}
    problems = []
    if result.get('returncode') == 0:
        problems.append('over-bound prompt was not refused')
    if run is None or run.get('status') != FAILED_STATUS:
        problems.append('expected a failed, unverified run record')
    if REFUSAL_TEXT not in (result.get('stderr_tail') or ''):
        problems.append('refusal reason did not name the exported context bound')
    if result.get('generated_exists'):
        problems.append('a generation output file exists for a refused prompt')
    if problems:
        detail['status'] = 'FAIL'
        detail['problems'] = problems
    return detail


def check_replay(result, min_prompt_tokens=None):
    """A saved-inspection advisory must verify; optionally prove >N prefill."""
    detail = {'status': 'PASS'}
    report = result.get('report') or {}
    native = report.get('native') or {}
    observed = native.get('prompt_tokens')
    detail['prompt_token_count'] = observed
    detail['controller_runtime_wall_ms'] = report.get('controller_runtime_wall_ms')
    detail['sampled_peak_child_rss_kib'] = native.get('sampled_peak_child_rss_kib')
    problems = []
    if result.get('returncode') != 0 or report.get('status') != VERIFIED_ADVICE:
        problems.append('advisory not verified: status {!r}, runtime_error {!r}'.format(
            report.get('status'), report.get('runtime_error')))
    if min_prompt_tokens is not None and not (
            type(observed) is int and observed > min_prompt_tokens):
        problems.append('prompt token count {!r} does not prove prefill above {}'.format(
            observed, min_prompt_tokens))
    if problems:
        detail['status'] = 'FAIL'
        detail['problems'] = problems
    return detail


def check_warm_cold(warm_report, cold_report, worker_evidence):
    """The existing warm-vs-cold adoption gate decides; evidence is recorded."""
    from forgesight.assist_worker_acceptance import compare_reports
    verdict = compare_reports(warm_report, cold_report)
    detail = {'status': 'PASS' if verdict['adoption_accepted'] else 'FAIL',
              'gate_checks': verdict['checks'],
              'worker_ready_wall_seconds': worker_evidence.get('ready_wall_seconds'),
              'worker_warmup_generate_ms': worker_evidence.get('warmup_generate_ms'),
              'warm_controller_runtime_wall_ms': warm_report.get('controller_runtime_wall_ms'),
              'cold_controller_runtime_wall_ms': cold_report.get('controller_runtime_wall_ms'),
              'worker_rss_samples_kib': worker_evidence.get('rss_samples_kib'),
              'rss_note': 'Point samples of the worker process VmRSS; may miss peaks.'}
    if not verdict['adoption_accepted']:
        detail['problems'] = ['warm/cold adoption gate rejected: '
                              + json.dumps(verdict['checks'], sort_keys=True)]
    return detail


def overall_status(checks):
    statuses = {entry['status'] for entry in checks.values()}
    if 'FAIL' in statuses:
        return FAILED
    if 'ERROR' in statuses:
        return INCOMPLETE
    return PASSED


# ---------------------------------------------------------------------------
# Production device executors (thin subprocess wiring; faked in tests)
# ---------------------------------------------------------------------------

class DeviceExecutors:
    """Runs each device step as a bounded subprocess; one fresh dir per step.

    Known limitation, deliberately not refactored here: subprocess.run with
    capture_output pipes and timeout terminates only the direct child, not
    the child's own process group. Every writer below is a known, bounded
    child (the smoke adapter writes its own bounded logs; the replay
    controller bounds its own children; the metadata/fit children print one
    small JSON document), so pipe-buffer deadlock is not expected; the
    persistent worker is the exception and IS managed as a process group via
    WorkerHandle. Broaden this only with measured evidence.
    """

    def __init__(self, repo, llama_root, timeout_seconds):
        self.repo = Path(repo).resolve()
        self.llama_root = Path(llama_root).resolve()
        self.model_dir = self.llama_root / 'model'
        self.venv_python = self.llama_root / 'venv/bin/python'
        self.timeout = timeout_seconds

    def _env(self):
        env = os.environ.copy()
        env.update({'PYTHONPATH': str(self.repo), 'PYTHONNOUSERSITE': '1',
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                    'PYTHONDONTWRITEBYTECODE': '1'})
        return env

    def run_child(self, child, extra, timeout=None):
        argv = [str(self.venv_python), '-B', '-m', 'forgesight.edge_native_acceptance',
                '--child', child] + [str(item) for item in extra]
        process = subprocess.run(argv, capture_output=True, text=True,
                                 cwd=self.repo, env=self._env(),
                                 timeout=timeout or self.timeout)
        payload = None
        if process.returncode == 0:
            payload = json.loads(process.stdout, parse_constant=_reject_non_finite)
        return {'returncode': process.returncode, 'report': payload,
                'stderr_tail': (process.stderr or '')[-2000:]}

    def run_smoke(self, prompt_path, max_new_tokens, out_dir):
        model = self.model_dir
        argv = [str(self.venv_python), '-B', str(model / 'edge_jetson_smoke.py'),
                '--root', str(self.llama_root),
                '--model', str(model / 'llama3_2_1b_xnnpack.pte'),
                '--tokenizer', str(model / 'tokenizer.model'),
                '--profile', str(model / 'edge_llama.json'),
                '--export', str(model / 'llama3_2_1b_export.json'),
                '--assets', str(model / 'llama3_2_1b_assets.json'),
                '--prompt', str(prompt_path), '--output', str(out_dir),
                '--threads', '2', '--max-new-tokens', str(max_new_tokens),
                '--timeout', str(self.timeout)]
        process = subprocess.run(argv, capture_output=True, text=True,
                                 cwd=self.llama_root, env=self._env(),
                                 timeout=self.timeout + 60)
        run = None
        run_path = Path(out_dir) / 'run.json'
        if run_path.is_file():
            run = _load_json_strict(run_path)
        return {'returncode': process.returncode, 'run': run,
                'generated_exists': (Path(out_dir) / 'generated.txt').is_file(),
                'stderr_tail': (process.stderr or '')[-2000:]}

    def run_replay(self, record, database, guidance, out_dir, worker_socket=None):
        argv = [str(self.venv_python), '-B', '-m', 'forgesight.assist_jetson_replay_once',
                '--record', str(record), '--database', str(database),
                '--guidance', str(guidance), '--llama-root', str(self.llama_root),
                '--output-dir', str(out_dir),
                # The replay controller caps its own budget at 240 seconds.
                '--llama-timeout-seconds', str(min(self.timeout, 240))]
        if worker_socket is not None:
            argv += ['--llama-worker-socket', str(worker_socket)]
        process = subprocess.run(argv, capture_output=True, text=True,
                                 cwd=self.repo, env=self._env(),
                                 timeout=self.timeout + 120)
        report = None
        report_path = Path(out_dir) / 'advice_report.json'
        if report_path.is_file():
            report = _load_json_strict(report_path)
        return {'returncode': process.returncode, 'report': report,
                'stderr_tail': (process.stderr or '')[-2000:]}

    def start_worker(self, socket_path, log_path):
        argv = [str(self.venv_python), '-B', '-m', 'forgesight.assist_edge_worker',
                '--socket', str(socket_path), '--llama-root', str(self.llama_root)]
        log = Path(log_path).open('xb')
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log,
                                   cwd=self.repo, env=self._env(),
                                   start_new_session=True)
        return WorkerHandle(process, log)

    @staticmethod
    def sample_rss_kib(pid):
        try:
            content = Path('/proc/{}/status'.format(pid)).read_text()
        except OSError:
            return None
        match = re.search(r'^VmRSS:\s+(\d+)\s+kB$', content, re.MULTILINE)
        return int(match.group(1)) if match else None


class WorkerHandle:
    """Reads the worker's READY line; stops the whole process group on close.

    The READY read is bounded end to end: a worker that stays alive but never
    prints (wedged before its warmup completes) must not hang the acceptance
    run, so the read polls with select() under one absolute deadline and any
    timeout or protocol violation stops and reaps the worker before raising.
    """

    MAX_READY_LINE_BYTES = 65536

    def __init__(self, process, log):
        self.process = process
        self.log = log
        self.ready = None

    def wait_ready(self, timeout_seconds):
        if not 0 < timeout_seconds <= 600:
            raise ValueError('Worker READY wait must be between 0 and 600 seconds.')
        deadline = time.monotonic() + timeout_seconds
        descriptor = self.process.stdout.fileno()
        line = b''
        while True:
            if self.process.poll() is not None:
                raise RuntimeError('Advisory worker exited before READY: {}'.format(
                    self.process.returncode))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise TimeoutError('Advisory worker did not report READY in time; '
                                   'the wedged worker was stopped and reaped.')
            readable, _writable, _error = select.select([descriptor], [], [],
                                                        min(remaining, 1.0))
            if not readable:
                continue
            chunk = os.read(descriptor, 65536)
            if not chunk:
                # EOF on stdout: the worker is gone or wedged silent; reap it
                # so the exit code is the honest reason (poll() can lag EOF).
                raise RuntimeError('Advisory worker exited before READY: {}'.format(
                    self.process.wait()))
            line += chunk
            if len(line) > self.MAX_READY_LINE_BYTES:
                self.stop()
                raise RuntimeError('Advisory worker READY line exceeded the size bound; '
                                   'the worker was stopped and reaped.')
            if b'\n' in line:
                line = line.split(b'\n', 1)[0]
                break
        payload = json.loads(line.decode('utf-8'), parse_constant=_reject_non_finite)
        if payload.get('status') != 'WORKER_READY':
            raise RuntimeError('Advisory worker first line was not WORKER_READY.')
        self.ready = payload
        return payload

    def stop(self):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except OSError:
                    pass
                self.process.wait()
        self.log.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_acceptance(config, executors):
    """Execute every acceptance step once; assemble the manifest.

    config keys: llama_root, database, guidance, review_record, reject_record,
    camera_record (optional), smoke_prompt, output_dir, timeout_seconds,
    expected_smoke_prompt_tokens. Each step gets one fresh subdirectory.
    """
    output = Path(config['output_dir'])
    checks = {}
    unassessed = []
    profile_path = Path(config['llama_root']) / 'model/edge_llama.json'
    profile = _load_json_strict(profile_path)
    profile_sha = sha256(profile_path)
    manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': INCOMPLETE, 'llama_root': str(config['llama_root']),
                'profile': {'path': str(profile_path), 'sha256': profile_sha,
                            'max_seq_length': profile['max_seq_length'],
                            'max_context_length': profile['max_context_length']},
                'artifact_sha256': None, 'tokenizer_sha256': None,
                'checks': checks, 'unassessed': unassessed, 'scope_notes': [
                    'Native prefill/generation acceptance for ONE artifact on ONE device.',
                    'No physical part inspection, vision detection, or offline network '
                    'disconnection is validated here.',
                    'RSS figures are sampled values and may miss peaks; timings are '
                    'measurements on this device, not performance promises.']}
    max_context = profile['max_context_length']

    def step_dir(name):
        path = output / name
        path.mkdir(parents=True, exist_ok=False)
        return path

    def record_error(name, error):
        checks[name] = {'status': 'ERROR',
                        'error': type(error).__name__ + ': ' + str(error)}

    # 1. Executed artifact metadata bound to the staged profile.
    try:
        result = executors.run_child('metadata', ['--root', config['llama_root']])
        report = result['report']
        if report is None:
            raise RuntimeError('metadata child failed: ' + result['stderr_tail'][-400:])
        manifest['artifact_sha256'] = (report.get('artifact') or {}).get('sha256')
        manifest['tokenizer_sha256'] = (report.get('tokenizer') or {}).get('sha256')
        checks['metadata_methods'] = check_metadata(report, profile)
    except Exception as error:
        record_error('metadata_methods', error)

    # 2. Smoke regression with the pinned prompt on the new artifact.
    try:
        result = executors.run_smoke(config['smoke_prompt'], 64, step_dir('regression_smoke'))
        checks['regression_smoke'] = check_generation(
            result, expected_prompt_tokens=config['expected_smoke_prompt_tokens'])
    except Exception as error:
        record_error('regression_smoke', error)

    # 3. Budget boundary: context-2 accepted, context-1 refused pre-generation
    #    (the guard refuses prompt + max_new_tokens > max_context_length).
    for label, target in (('boundary_accepted', max_context - BOUNDARY_MAX_NEW),
                          ('boundary_refused', max_context - BOUNDARY_MAX_NEW + 1)):
        try:
            directory = step_dir(label)
            fit = executors.run_child('fit-prompt', ['--root', config['llama_root'],
                                                     '--target', target,
                                                     '--output', directory / 'prompt.txt'])
            if fit['report'] is None:
                raise RuntimeError('fit child failed: ' + fit['stderr_tail'][-400:])
            result = executors.run_smoke(directory / 'prompt.txt', BOUNDARY_MAX_NEW,
                                         directory / 'run')
            if label == 'boundary_accepted':
                checks[label] = check_generation(result, expected_prompt_tokens=target)
            else:
                checks[label] = check_refusal(result)
            checks[label]['fit_target_tokens'] = target
        except Exception as error:
            record_error(label, error)

    # 4. Real saved >128-token inspections through the trusted cold path.
    cold_review_report = None
    for label, key in (('replay_review', 'review_record'), ('replay_reject', 'reject_record')):
        try:
            result = executors.run_replay(config[key], config['database'], config['guidance'],
                                          step_dir(label))
            checks[label] = check_replay(result, min_prompt_tokens=128)
            if label == 'replay_review' and checks[label]['status'] == 'PASS':
                cold_review_report = result['report']
        except Exception as error:
            record_error(label, error)

    # 5. Optional camera record, labeled by its own stored source mode.
    camera_record = config.get('camera_record')
    if camera_record is None:
        checks['camera_replay'] = {'status': 'UNASSESSED',
                                   'reason': 'No camera inspection record supplied.'}
        unassessed.append('camera_replay')
    else:
        try:
            record = _load_json_strict(camera_record)
            source_mode = (record.get('source') or {}).get('mode')
            result = executors.run_replay(camera_record, config['database'],
                                          config['guidance'], step_dir('camera_replay'))
            checks['camera_replay'] = check_replay(result)
            checks['camera_replay']['record_source_mode'] = source_mode
            checks['camera_replay']['camera_evidence'] = (
                'RECORDED_AS_DEVICE_CAMERA_CAPTURE' if source_mode and 'CAMERA' in source_mode.upper()
                else 'SYNTHETIC_OR_NON_CAMERA_RECORD_LABELED_BY_SOURCE_MODE')
            checks['camera_replay']['physical_part_inspected'] = bool(
                (record.get('source') or {}).get('physical_part'))
        except Exception as error:
            record_error('camera_replay', error)

    # 6. Warm worker vs trusted cold path for the same REVIEW inspection.
    if cold_review_report is None:
        checks['warm_cold'] = {'status': 'UNASSESSED',
                               'reason': 'Cold REVIEW replay did not produce a verified report.'}
        unassessed.append('warm_cold')
    else:
        worker = None
        try:
            directory = step_dir('warm_cold')
            socket_path = directory / 'advisory.sock'
            worker = executors.start_worker(socket_path, directory / 'worker.stderr.log')
            started = time.monotonic()
            ready = worker.wait_ready(config['timeout_seconds'])
            evidence = {'ready_wall_seconds': time.monotonic() - started,
                        'warmup_generate_ms': ready.get('warmup_generate_ms'),
                        'rss_samples_kib': [executors.sample_rss_kib(worker.process.pid)]}
            warm = executors.run_replay(config['review_record'], config['database'],
                                        config['guidance'], directory / 'warm_replay',
                                        worker_socket=socket_path)
            evidence['rss_samples_kib'].append(executors.sample_rss_kib(worker.process.pid))
            if warm['report'] is None:
                raise RuntimeError('warm replay failed: ' + warm['stderr_tail'][-400:])
            checks['warm_cold'] = check_warm_cold(warm['report'], cold_review_report, evidence)
        except Exception as error:
            record_error('warm_cold', error)
        finally:
            if worker is not None:
                worker.stop()

    manifest['status'] = overall_status(checks)
    payload = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
    manifest_path = output / 'acceptance.json'
    with manifest_path.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return manifest


# ---------------------------------------------------------------------------
# Device children (run under the private runtime venv interpreter)
# ---------------------------------------------------------------------------

def _runtime_spec(root):
    root = Path(root).resolve()
    model = root / 'model'
    return {'root': str(root), 'model': str(model / 'llama3_2_1b_xnnpack.pte'),
            'tokenizer': str(model / 'tokenizer.model'),
            'profile': str(model / 'edge_llama.json'),
            'export': str(model / 'llama3_2_1b_export.json'),
            'assets': str(model / 'llama3_2_1b_assets.json')}


def _execute_method(program, name):
    """Execute one no-input metadata method; return (route, plain value)."""
    for route in ('run_method', 'execute'):
        function = getattr(program, route, None)
        if function is None:
            continue
        try:
            outputs = function(name, [])
        except TypeError:
            outputs = function(name)
        values = []
        for item in outputs if isinstance(outputs, (list, tuple)) else (outputs,):
            values.append(item.tolist() if hasattr(item, 'tolist') else item)
        if len(values) != 1 or not isinstance(values[0], int):
            raise RuntimeError('Metadata method {} returned unexpected values {!r}'.format(
                name, values))
        return route, values[0]
    raise RuntimeError('The pinned runtime Program has no executable method route.')


def child_metadata(root):
    """Verify the staged chain, then execute the artifact's metadata methods."""
    from forgesight import edge_jetson_smoke as smoke
    import importlib.metadata
    host, profile, files, model_hash, tokenizer_hash = smoke.verify_inputs(_runtime_spec(root))
    version = importlib.metadata.version('executorch')
    if version.split('+', 1)[0] != profile['executorch_version']:
        raise ValueError('Use the pinned ExecuTorch environment.')
    from executorch.runtime import Runtime
    program = Runtime.get().load_program(str(files['model']))
    available = sorted(program.method_names)
    methods = {}
    for name in METADATA_METHODS:
        if name not in available:
            raise ValueError('Artifact is missing metadata method: ' + name)
        route, value = _execute_method(program, name)
        methods[name] = {'route': route, 'value': value}
    return {'status': 'METADATA_EXECUTED', 'host': host, 'artifact': model_hash,
            'tokenizer': tokenizer_hash, 'executorch_version': version,
            'profile': {'max_seq_length': profile['max_seq_length'],
                        'max_context_length': profile['max_context_length']},
            'methods_available': available, 'methods': methods,
            'scope': 'Metadata methods executed; no forward/inference run here.'}


def child_fit_prompt(root, target, output):
    """Fit a prompt to exactly target formatted tokens with the deployed tokenizer."""
    from forgesight import edge_jetson_smoke as smoke
    from pytorch_tokenizers import CppTiktoken
    _host, _profile, files, _model_hash, _tokenizer_hash = smoke.verify_inputs(_runtime_spec(root))
    tokenizer = CppTiktoken()
    tokenizer.load(str(files['tokenizer']))

    def encode(text):
        return len(tokenizer.encode(smoke.chat_prompt(text), 0, 0))

    text, count = fit_prompt_to_target(target, encode)
    output = Path(output)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(text)
    return {'status': 'PROMPT_FITTED', 'target': target, 'prompt_token_count': count,
            'prompt_sha256': sha256(output), 'prompt_bytes': output.stat().st_size}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', choices=('metadata', 'fit-prompt'), default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--root', type=Path, help='Private llama runtime root (child mode).')
    parser.add_argument('--target', type=int, help='Exact formatted prompt tokens (fit child).')
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh acceptance directory (driver) or prompt file (fit child).')
    parser.add_argument('--llama-root', type=Path, help='Private llama runtime root (driver).')
    parser.add_argument('--database', type=Path)
    parser.add_argument('--guidance', type=Path)
    parser.add_argument('--review-record', type=Path)
    parser.add_argument('--reject-record', type=Path)
    parser.add_argument('--camera-record', type=Path, default=None)
    parser.add_argument('--smoke-prompt', type=Path,
                        default=Path('configs/edge_smoke_prompt.txt'))
    parser.add_argument('--timeout-seconds', type=float, default=240)
    parser.add_argument('--expected-smoke-prompt-tokens', type=int, default=33)
    args = parser.parse_args()

    if args.child == 'metadata':
        if args.root is None:
            parser.error('The metadata child needs --root.')
        print(json.dumps(child_metadata(args.root), indent=2, allow_nan=False))
        return
    if args.child == 'fit-prompt':
        if args.root is None or args.target is None:
            parser.error('The fit child needs --root and --target.')
        print(json.dumps(child_fit_prompt(args.root, args.target, args.output),
                         indent=2, allow_nan=False))
        return

    missing = [name for name in ('llama_root', 'database', 'guidance',
                                 'review_record', 'reject_record')
               if getattr(args, name) is None]
    if missing:
        parser.error('Driver mode requires: ' + ', '.join('--' + name.replace('_', '-')
                                                          for name in missing))
    if args.output.exists():
        parser.error('Choose a new acceptance output directory.')
    if not 0 < args.timeout_seconds <= 300:
        parser.error('Per-generation timeout must be greater than zero and at most 300 seconds.')
    for name in ('review_record', 'reject_record', 'camera_record'):
        path = getattr(args, name)
        if path is not None and (not path.is_file() or path.stat().st_size > 1024 * 1024):
            parser.error('Inspection record missing or above the metadata size bound: ' + name)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {'llama_root': args.llama_root, 'database': args.database,
              'guidance': args.guidance, 'review_record': args.review_record,
              'reject_record': args.reject_record, 'camera_record': args.camera_record,
              'smoke_prompt': args.smoke_prompt, 'output_dir': args.output,
              'timeout_seconds': args.timeout_seconds,
              'expected_smoke_prompt_tokens': args.expected_smoke_prompt_tokens}
    executors = DeviceExecutors(Path.cwd(), args.llama_root, args.timeout_seconds)
    manifest = run_acceptance(config, executors)
    print(json.dumps({'status': manifest['status'],
                      'checks': {name: entry['status'] for name, entry in
                                 manifest['checks'].items()},
                      'unassessed': manifest['unassessed']}, indent=2))
    if manifest['status'] != PASSED:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
