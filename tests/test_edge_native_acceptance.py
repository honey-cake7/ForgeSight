"""Host tests for the post-export native acceptance driver.

All device interaction is faked: the check evaluators and the orchestration
logic run against synthetic reports, and the prompt fitter runs against stub
tokenizers. Real metadata execution, fitting with the deployed tokenizer, and
every generation happen only on the Jetson.
"""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest import mock

from forgesight.edge_jetson_smoke import chat_prompt
from forgesight import edge_native_acceptance as acceptance
from forgesight.edge_native_acceptance import (check_generation, check_metadata,
                                               check_refusal, check_replay,
                                               check_warm_cold, fit_prompt_to_target,
                                               overall_status, run_acceptance)

VERIFIED_SMOKE = acceptance.VERIFIED_SMOKE
VERIFIED_ADVICE = acceptance.VERIFIED_ADVICE
NOTE = ('[E1] Anomaly flagged; grade REVIEW. '
        '[G1] Prototype checklist: hold for human review.')
PROFILE = {'max_seq_length': 256, 'max_context_length': 512}


CHAT_OVERHEAD_TOKENS = 3  # fixed marker cost in the stub models


def stub_encode(text):
    """One whitespace-separated word is one token, plus chat overhead."""
    return CHAT_OVERHEAD_TOKENS + len(text.split())


def parity_encode(text):
    """' stable' costs two tokens; every other word costs one (parity trap)."""
    words = text.split()
    return (CHAT_OVERHEAD_TOKENS + len(words)
            + sum(1 for word in words if word == 'stable'))


def smoke_result(prompt_tokens=33, generated=40, returncode=0, status=VERIFIED_SMOKE):
    run = {'status': status, 'process_wall_seconds': 21.5,
           'sampled_peak_child_rss_kib': 4_200_000,
           'worker_result': {'status': VERIFIED_SMOKE,
                             'prompt_token_count': prompt_tokens,
                             'stats': [{'generated_tokens': generated,
                                        'prompt_tokens': prompt_tokens}]}}
    return {'returncode': returncode, 'run': run, 'generated_exists': returncode == 0,
            'stderr_tail': ''}


def advice_report(mode='COLD_SUBPROCESS', prompt_tokens=180, note=NOTE,
                  status=VERIFIED_ADVICE):
    native = {'model_sha256': 'm' * 64, 'tokenizer_sha256': 't' * 64,
              'prompt_tokens': prompt_tokens,
              'sampled_peak_child_rss_kib': 4_100_000}
    if mode == 'PERSISTENT_WORKER':
        native.update(worker_mode='PERSISTENT_PROCESS',
                      worker_started_utc='2026-11-22T00:00:00Z')
    return {'status': status, 'llama_runtime_mode': mode,
            'inspection_sha256': 'i' * 64, 'decision_sha256': 'd' * 64,
            'authoritative_grade': 'REVIEW', 'inspection_and_sqlite_unchanged': True,
            'controller_runtime_wall_ms': 21000.0 if mode == 'COLD_SUBPROCESS' else 3200.0,
            'native': native, 'advisory': {'response': {'generated_note': note}}}


def metadata_report(seq=256, context=512):
    return {'status': 'METADATA_EXECUTED',
            'artifact': {'sha256': 'a' * 64}, 'tokenizer': {'sha256': 't' * 64},
            'methods': {'get_max_seq_len': {'route': 'run_method', 'value': seq},
                        'get_max_context_len': {'route': 'run_method', 'value': context}}}


class FitTests(unittest.TestCase):
    def test_exact_fit_with_one_token_filler(self):
        for target in (60, 255, 256):
            with self.subTest(target=target):
                text, count = fit_prompt_to_target(target, stub_encode)
                self.assertEqual(count, target)
                self.assertEqual(stub_encode(text), target)
                # The fitted text stays a valid bounded chat user prompt.
                chat_prompt(text)

    def test_exact_fit_with_two_token_filler_uses_fine_breaker(self):
        # Base is 14; filler costs 2, so an odd target needs one 1-token word.
        text, count = fit_prompt_to_target(59, parity_encode)
        self.assertEqual(count, 59)
        self.assertEqual(parity_encode(text), 59)
        self.assertTrue(any(word in text.split() for word in acceptance.FINE_FILLERS))

    def test_fit_fails_closed_when_base_exceeds_target(self):
        with self.assertRaisesRegex(ValueError, 'Base prompt alone exceeds'):
            fit_prompt_to_target(8, stub_encode)

    def test_fit_fails_closed_without_additive_filler(self):
        with self.assertRaisesRegex(ValueError, 'did not converge'):
            fit_prompt_to_target(60, lambda text: 14, max_encodes=200)

    def test_fit_fails_closed_when_no_combination_exists(self):
        # Filler and every fine word cost 2 tokens; an odd need is unreachable.
        def even_encode(text):
            words = text.split()
            extras = sum(1 for word in words
                         if word in ('stable',) + acceptance.FINE_FILLERS)
            return CHAT_OVERHEAD_TOKENS + len(words) + extras
        with self.assertRaisesRegex(ValueError, 'did not converge'):
            fit_prompt_to_target(59, even_encode)

    def test_fit_rejects_bad_targets_and_bases(self):
        for target in (7, 1025, '255', 33.5):
            with self.subTest(target=target), self.assertRaises(ValueError):
                fit_prompt_to_target(target, stub_encode)
        with self.assertRaises(ValueError):
            fit_prompt_to_target(60, stub_encode, base_text='has <|eot_id|> marker')


class CheckTests(unittest.TestCase):
    def test_metadata_matches_profile(self):
        detail = check_metadata(metadata_report(), PROFILE)
        self.assertEqual(detail['status'], 'PASS')
        self.assertEqual(detail['executed']['get_max_seq_len'], 256)

    def test_metadata_mismatch_fails(self):
        detail = check_metadata(metadata_report(seq=128), PROFILE)
        self.assertEqual(detail['status'], 'FAIL')
        self.assertIn('get_max_seq_len', detail['problems'][0])

    def test_generation_verified_with_expected_count(self):
        self.assertEqual(check_generation(smoke_result(33), 33)['status'], 'PASS')

    def test_generation_wrong_count_fails(self):
        detail = check_generation(smoke_result(34), 33)
        self.assertEqual(detail['status'], 'FAIL')

    def test_generation_unverified_fails(self):
        detail = check_generation(smoke_result(returncode=1, status='FAILED_NOT_VERIFIED'))
        self.assertEqual(detail['status'], 'FAIL')

    def test_refusal_before_generation_passes(self):
        result = {'returncode': 1, 'run': {'status': 'FAILED_NOT_VERIFIED',
                                           'completion_reason': 'EXITED'},
                  'generated_exists': False,
                  'stderr_tail': 'ValueError: Prompt plus answer exceeds exported context.'}
        self.assertEqual(check_refusal(result)['status'], 'PASS')

    def test_refusal_fails_when_generation_happened(self):
        result = {'returncode': 1, 'run': {'status': 'FAILED_NOT_VERIFIED'},
                  'generated_exists': True,
                  'stderr_tail': 'ValueError: Prompt plus answer exceeds exported context.'}
        self.assertEqual(check_refusal(result)['status'], 'FAIL')

    def test_refusal_fails_when_accepted(self):
        result = {'returncode': 0, 'run': {'status': VERIFIED_SMOKE},
                  'generated_exists': True, 'stderr_tail': ''}
        self.assertEqual(check_refusal(result)['status'], 'FAIL')

    def test_replay_above_128_passes(self):
        result = {'returncode': 0, 'report': advice_report(prompt_tokens=180)}
        detail = check_replay(result, min_prompt_tokens=128)
        self.assertEqual(detail['status'], 'PASS')
        self.assertEqual(detail['prompt_token_count'], 180)

    def test_replay_at_or_below_128_fails(self):
        for tokens in (128, 33, None):
            with self.subTest(tokens=tokens):
                result = {'returncode': 0, 'report': advice_report(prompt_tokens=tokens)}
                self.assertEqual(check_replay(result, min_prompt_tokens=128)['status'], 'FAIL')

    def test_replay_unverified_fails(self):
        result = {'returncode': 1, 'report': advice_report(status='ADVICE_UNAVAILABLE')}
        self.assertEqual(check_replay(result, min_prompt_tokens=128)['status'], 'FAIL')

    def test_warm_cold_gate_binds(self):
        evidence = {'ready_wall_seconds': 30.0, 'warmup_generate_ms': 2900.0,
                    'rss_samples_kib': [5_000_000, 5_100_000]}
        detail = check_warm_cold(advice_report('PERSISTENT_WORKER'), advice_report(), evidence)
        self.assertEqual(detail['status'], 'PASS')
        detail = check_warm_cold(advice_report('PERSISTENT_WORKER', note='different note'),
                                 advice_report(), evidence)
        self.assertEqual(detail['status'], 'FAIL')

    def test_overall_status(self):
        self.assertEqual(overall_status({'a': {'status': 'PASS'},
                                         'b': {'status': 'UNASSESSED'}}), acceptance.PASSED)
        self.assertEqual(overall_status({'a': {'status': 'FAIL'}}), acceptance.FAILED)
        self.assertEqual(overall_status({'a': {'status': 'ERROR'}}), acceptance.INCOMPLETE)

    def test_strict_json_rejects_non_finite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text('{"value": NaN}')
            with self.assertRaises(ValueError):
                acceptance._load_json_strict(path)


class FakeWorker:
    def __init__(self):
        self.process = mock.Mock()
        self.process.pid = 4242
        self.stopped = False

    def wait_ready(self, timeout_seconds):
        return {'status': 'WORKER_READY', 'warmup_generate_ms': 2900.0}

    def stop(self):
        self.stopped = True


class FakeExecutors:
    """Scripted device outcomes; records every call the driver makes."""

    def __init__(self, metadata=None, smoke=None, refusal=None, replays=None,
                 warm=None, cold_review_ok=True):
        self.metadata = metadata if metadata is not None else metadata_report()
        self.smoke = smoke if smoke is not None else smoke_result(33)
        self.refusal = refusal if refusal is not None else {
            'returncode': 1, 'run': {'status': 'FAILED_NOT_VERIFIED',
                                     'completion_reason': 'EXITED'},
            'generated_exists': False,
            'stderr_tail': 'ValueError: Prompt plus answer exceeds exported context.'}
        self.replays = replays if replays is not None else {
            'review': advice_report(prompt_tokens=180),
            'reject': advice_report(prompt_tokens=188),
            'camera': advice_report(prompt_tokens=211)}
        self.warm = warm if warm is not None else advice_report('PERSISTENT_WORKER',
                                                                prompt_tokens=180)
        self.cold_review_ok = cold_review_ok
        self.calls = []
        self.worker = FakeWorker()

    def run_child(self, child, extra):
        self.calls.append(('child', child))
        if child == 'metadata':
            return {'returncode': 0, 'report': self.metadata, 'stderr_tail': ''}
        if child == 'fit-prompt':
            output = Path(extra[extra.index('--output') + 1])
            target = int(extra[extra.index('--target') + 1])
            output.write_text('fitted prompt for ' + str(target))
            return {'returncode': 0,
                    'report': {'status': 'PROMPT_FITTED', 'prompt_token_count': target},
                    'stderr_tail': ''}
        raise AssertionError('unexpected child: ' + child)

    def run_smoke(self, prompt_path, max_new_tokens, out_dir):
        self.calls.append(('smoke', max_new_tokens))
        if 'boundary_refused' in str(out_dir):
            return self.refusal
        if 'boundary_accepted' in str(out_dir):
            return smoke_result(prompt_tokens=PROFILE['max_context_length'] - 2,
                                generated=1)
        return self.smoke

    def run_replay(self, record, database, guidance, out_dir, worker_socket=None):
        name = Path(record).name
        self.calls.append(('replay', name, worker_socket is not None))
        if worker_socket is not None:
            report = self.warm
        elif 'review' in name:
            report = self.replays['review'] if self.cold_review_ok else advice_report(
                status='ADVICE_UNAVAILABLE')
        elif 'reject' in name:
            report = self.replays['reject']
        else:
            report = self.replays['camera']
        returncode = 0 if report['status'] == VERIFIED_ADVICE else 1
        return {'returncode': returncode, 'report': report, 'stderr_tail': ''}

    def start_worker(self, socket_path, log_path):
        Path(log_path).touch()
        return self.worker

    @staticmethod
    def sample_rss_kib(pid):
        return 5_000_000


def write_profile(root):
    model = Path(root) / 'model'
    model.mkdir(parents=True)
    (model / 'edge_llama.json').write_text(json.dumps(PROFILE))


def acceptance_config(output, **overrides):
    root = Path(output) / 'llama'
    write_profile(root)
    records = Path(output) / 'records'
    records.mkdir()
    for name in ('review', 'reject', 'camera'):
        (records / (name + '.json')).write_text(json.dumps(
            {'source': {'mode': 'SAVED_IMAGE_REPLAY', 'physical_part': False}}))
    config = {'llama_root': root, 'database': records, 'guidance': records,
              'review_record': records / 'review.json',
              'reject_record': records / 'reject.json',
              'camera_record': None,
              'smoke_prompt': records / 'review.json',
              'output_dir': Path(output) / 'out', 'timeout_seconds': 240,
              'expected_smoke_prompt_tokens': 33}
    config.update(overrides)
    (Path(output) / 'out').mkdir()
    return config


class OrchestrationTests(unittest.TestCase):
    def test_all_checks_pass_without_camera(self):
        with TemporaryDirectory() as directory:
            manifest = run_acceptance(acceptance_config(directory), FakeExecutors())
            self.assertEqual(manifest['status'], acceptance.PASSED)
            self.assertEqual(manifest['checks']['camera_replay']['status'], 'UNASSESSED')
            self.assertIn('camera_replay', manifest['unassessed'])
            self.assertEqual(manifest['checks']['boundary_accepted']['prompt_token_count'], 510)
            self.assertEqual(manifest['checks']['boundary_accepted']['fit_target_tokens'], 510)
            self.assertEqual(manifest['checks']['boundary_refused']['fit_target_tokens'], 511)
            self.assertEqual(manifest['artifact_sha256'], 'a' * 64)
            persisted = json.loads((Path(directory) / 'out/acceptance.json').read_text())
            self.assertEqual(persisted['status'], acceptance.PASSED)

    def test_camera_record_labeled_by_source_mode(self):
        with TemporaryDirectory() as directory:
            config = acceptance_config(directory)
            camera = Path(config['camera_record'] or '')
            camera = Path(directory) / 'records/camera.json'
            camera.write_text(json.dumps({'source': {'mode': 'LIVE_CAMERA_CAPTURE',
                                                     'physical_part': False}}))
            config['camera_record'] = camera
            manifest = run_acceptance(config, FakeExecutors())
            check = manifest['checks']['camera_replay']
            self.assertEqual(check['status'], 'PASS')
            self.assertEqual(check['camera_evidence'], 'RECORDED_AS_DEVICE_CAMERA_CAPTURE')
            self.assertEqual(manifest['status'], acceptance.PASSED)

    def test_synthetic_camera_record_stays_labeled(self):
        with TemporaryDirectory() as directory:
            config = acceptance_config(directory)
            config['camera_record'] = Path(directory) / 'records/camera.json'
            manifest = run_acceptance(config, FakeExecutors())
            check = manifest['checks']['camera_replay']
            self.assertEqual(check['camera_evidence'],
                             'SYNTHETIC_OR_NON_CAMERA_RECORD_LABELED_BY_SOURCE_MODE')

    def test_metadata_mismatch_fails_overall(self):
        with TemporaryDirectory() as directory:
            manifest = run_acceptance(acceptance_config(directory),
                                      FakeExecutors(metadata=metadata_report(seq=128)))
            self.assertEqual(manifest['status'], acceptance.FAILED)
            self.assertEqual(manifest['checks']['metadata_methods']['status'], 'FAIL')

    def test_over_bound_generation_fails_overall(self):
        refusal = {'returncode': 0, 'run': {'status': VERIFIED_SMOKE},
                   'generated_exists': True, 'stderr_tail': ''}
        with TemporaryDirectory() as directory:
            manifest = run_acceptance(acceptance_config(directory),
                                      FakeExecutors(refusal=refusal))
            self.assertEqual(manifest['status'], acceptance.FAILED)
            self.assertEqual(manifest['checks']['boundary_refused']['status'], 'FAIL')

    def test_regression_count_drift_fails_overall(self):
        with TemporaryDirectory() as directory:
            manifest = run_acceptance(acceptance_config(directory),
                                      FakeExecutors(smoke=smoke_result(34)))
            self.assertEqual(manifest['status'], acceptance.FAILED)
            self.assertEqual(manifest['checks']['regression_smoke']['status'], 'FAIL')

    def test_cold_review_failure_skips_warm_gate_honestly(self):
        with TemporaryDirectory() as directory:
            manifest = run_acceptance(acceptance_config(directory),
                                      FakeExecutors(cold_review_ok=False))
            self.assertEqual(manifest['checks']['replay_review']['status'], 'FAIL')
            self.assertEqual(manifest['checks']['warm_cold']['status'], 'UNASSESSED')
            self.assertEqual(manifest['status'], acceptance.FAILED)

    def test_child_crash_records_error_not_exception(self):
        executors = FakeExecutors()
        executors.metadata = None  # child "crashed": no report parsed
        with TemporaryDirectory() as directory:
            manifest = run_acceptance(acceptance_config(directory), executors)
            self.assertEqual(manifest['checks']['metadata_methods']['status'], 'ERROR')
            self.assertEqual(manifest['status'], acceptance.INCOMPLETE)


def real_worker_process(argv_suffix, log_path):
    """A REAL child process speaking (or not speaking) the worker protocol."""
    log = Path(log_path).open('xb')
    process = subprocess.Popen([sys.executable, '-c', argv_suffix],
                               stdout=subprocess.PIPE, stderr=log,
                               start_new_session=True)
    return acceptance.WorkerHandle(process, log)


class RealWorkerHandleTests(unittest.TestCase):
    """Bounded READY wait against actual wedged/exiting/ready child processes.

    The earlier scripted fakes could not hang; these prove the live-silent
    wedge the review found is now bounded and reaped.
    """

    def test_silent_wedged_worker_times_out_and_is_reaped(self):
        with TemporaryDirectory() as directory:
            handle = real_worker_process('import time; time.sleep(120)',
                                         Path(directory) / 'stderr.log')
            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, 'stopped and reaped'):
                handle.wait_ready(0.5)
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 10)
            # The whole process group was terminated and the child reaped.
            self.assertIsNotNone(handle.process.poll())
            handle.stop()  # idempotent after the timeout stop

    def test_worker_exiting_before_ready_raises(self):
        with TemporaryDirectory() as directory:
            handle = real_worker_process('raise SystemExit(3)',
                                         Path(directory) / 'stderr.log')
            with self.assertRaisesRegex(RuntimeError, 'exited before READY'):
                handle.wait_ready(5)
            handle.stop()

    def test_ready_worker_reports_and_stops_cleanly(self):
        with TemporaryDirectory() as directory:
            handle = real_worker_process(
                'import time; print(\'{"status": "WORKER_READY", '
                '"warmup_generate_ms": 1.0}\', flush=True); time.sleep(120)',
                Path(directory) / 'stderr.log')
            payload = handle.wait_ready(5)
            self.assertEqual(payload['status'], 'WORKER_READY')
            handle.stop()
            self.assertIsNotNone(handle.process.poll())

    def test_unbounded_ready_line_is_stopped(self):
        with TemporaryDirectory() as directory:
            handle = real_worker_process(
                'import sys, time; sys.stdout.write("x" * 70000); '
                'sys.stdout.flush(); time.sleep(120)',
                Path(directory) / 'stderr.log')
            with self.assertRaisesRegex(RuntimeError, 'size bound'):
                handle.wait_ready(5)
            self.assertIsNotNone(handle.process.poll())
            handle.stop()

    def test_wait_ready_rejects_bad_timeouts(self):
        with TemporaryDirectory() as directory:
            handle = real_worker_process('import time; time.sleep(5)',
                                         Path(directory) / 'stderr.log')
            for value in (0, -1, 601):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    handle.wait_ready(value)
            handle.stop()


class CliGuardTests(unittest.TestCase):
    def test_driver_requires_records_and_refuses_existing_output(self):
        with TemporaryDirectory() as directory:
            existing = Path(directory) / 'taken'
            existing.mkdir()
            argv = ['prog', '--output', str(existing), '--llama-root', directory,
                    '--database', directory, '--guidance', directory,
                    '--review-record', directory, '--reject-record', directory]
            with mock.patch('sys.argv', argv), self.assertRaises(SystemExit):
                acceptance.main()

    def test_driver_requires_mandatory_arguments(self):
        with TemporaryDirectory() as directory:
            argv = ['prog', '--output', str(Path(directory) / 'new')]
            with mock.patch('sys.argv', argv), self.assertRaises(SystemExit):
                acceptance.main()

    def test_metadata_child_requires_root(self):
        with TemporaryDirectory() as directory:
            argv = ['prog', '--child', 'metadata', '--output', str(Path(directory) / 'x')]
            with mock.patch('sys.argv', argv), self.assertRaises(SystemExit):
                acceptance.main()


if __name__ == '__main__':
    unittest.main()
