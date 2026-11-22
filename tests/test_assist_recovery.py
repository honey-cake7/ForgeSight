"""Recovery hardening for the advisory worker/controller/UI triangle.

Covers the three pre-existing failure modes the boundary audit demonstrated
with stand-in runtimes (no Jetson, no model):

1. A worker alive but wedged inside generate() never hit the idle accept
   timeout and never respawned (poll() is None); the read timeout was
   classified as a plain runtime error, so the trusted cold path never ran.
2. The cold fallback always received a fresh full budget, so a late worker
   death could push the whole replay past the caller's outer kill, leaving
   no report and an output dir that blocked retry as ALREADY_REQUESTED.
3. A non-finite number in native output (accepted by Python's json by
   default) crashed the worker loop on the warm side and, on the cold side,
   raised inside the final report dump, leaving a partial advice_report.json
   that permanently blocked a fresh request.
"""

import argparse
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from forgesight import assist_edge_worker as aew
from forgesight import assist_jetson_replay_once as replay

PROFILE = {"max_seq_length": 512, "max_context_length": 640}


class NanStandInRuntime:
    """Interface-compatible stand-in whose native stats carry a NaN."""

    def __init__(self, artifact):
        self.artifact = Path(artifact)
        self.profile = PROFILE
        self.evidence = {"artifact": {"sha256": "a" * 64},
                         "tokenizer": {"sha256": "b" * 64}}

    def reset(self):
        pass

    def artifact_state(self):
        info = self.artifact.stat()
        return (str(self.artifact), info.st_size, info.st_mtime_ns, info.st_ino)

    def token_count(self, text):
        return len(text.split())

    def generate(self, formatted, max_new_tokens):
        return "note", {"prompt_tokens": len(formatted.split()),
                        "generated_tokens": 1,
                        "tokens_per_second": float("nan")}


class InfEvidenceRuntime(NanStandInRuntime):
    """Finite stats (passes the handle() check) but non-finite floats in the
    runtime evidence: only the serve()-level strict-JSON fallback catches it."""

    def __init__(self, artifact):
        super().__init__(artifact)
        self.evidence = {"artifact": {"sha256": "a" * 64},
                         "load_ms": float("inf")}

    def generate(self, formatted, max_new_tokens):
        return "note", {"prompt_tokens": len(formatted.split()),
                        "generated_tokens": 1}


class WorkerNanTests(unittest.TestCase):
    """A non-finite native statistic must fail the request closed without
    killing the worker loop."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.artifact = Path(self.dir.name) / "model.pte"
        self.artifact.write_bytes(b"standin")

    def tearDown(self):
        self.dir.cleanup()

    def serve_in_thread(self, worker):
        sock = Path(self.dir.name) / "w.sock"
        errors = []

        def run():
            try:
                aew.serve(sock, worker, idle_timeout_seconds=2,
                          request_read_timeout_seconds=5)
            except Exception as error:  # noqa: BLE001 - capture a crash
                errors.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        for _ in range(200):
            if sock.exists() or errors:
                break
            time.sleep(0.05)
        return sock, thread, errors

    def test_nan_stats_error_but_worker_survives(self):
        sock, thread, errors = self.serve_in_thread(
            aew.Worker(NanStandInRuntime(self.artifact)))
        for _ in range(2):  # twice: a crashed loop could not answer again
            with self.assertRaises(RuntimeError) as raised:
                aew.request(sock, {"prompt_text": "hold part", "max_new_tokens": 8},
                            timeout_seconds=5)
            self.assertIn("ValueError", str(raised.exception))
        thread.join(timeout=10)  # exits via the idle timeout, not a crash
        self.assertFalse(thread.is_alive(), "serve() must exit cleanly")
        self.assertFalse(errors, f"serve() must not crash: {errors!r}")

    def test_nonfinite_evidence_surfaces_as_error_and_worker_survives(self):
        sock, thread, errors = self.serve_in_thread(
            aew.Worker(InfEvidenceRuntime(self.artifact)))
        for _ in range(2):
            with self.assertRaises(RuntimeError) as raised:
                aew.request(sock, {"prompt_text": "hold part", "max_new_tokens": 8},
                            timeout_seconds=5)
            self.assertIn("not serializable as strict JSON", str(raised.exception))
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive(), "serve() must exit cleanly")
        self.assertFalse(errors, f"serve() must not crash: {errors!r}")


class HungWorkerControllerTests(unittest.TestCase):
    """Controller against a worker that accepts the request and never replies,
    through the real replay.main() wiring with the cold path mocked."""

    REVIEW_NOTE = ("An anomaly is flagged; type and cause are unknown [E1]. "
                   "Severity HIGH, prototype pixel-response severity, not physical-damage severity. "
                   "Anomaly score 0.682809 vs threshold 0.227766. "
                   "Response area 9015 px across 2 regions; largest region width 139 px height 116 px "
                   "at x 512 y 361 in the right area of the frame. "
                   "Hold for human review; check clarity and lighting; compare the highlighted "
                   "region with a known-good part [G1].")

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        record = json.loads(Path("manifests/operator_app_verification_v1.json").read_text()
                            )["browser_proof"]["inspections"][1]["record"]
        record["model"]["jetson_verified"] = True
        record["source"]["physical_part"] = False
        self.record_path = root / "record.json"
        self.record_path.write_text(json.dumps(record, indent=2) + "\n")
        self.database = root / "inspections.db"
        from forgesight.storage import InspectionStore
        InspectionStore(self.database).save_inspection(record)
        self.output = root / "advice_out"
        self.socket_path = root / "assist.sock"

    def tearDown(self):
        self.dir.cleanup()

    def hanging_worker(self):
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen(1)
        connection, _ = server.accept()
        with connection:
            data = b""
            while b"\n" not in data:
                block = connection.recv(65536)
                if not block:
                    break
                data += block
            time.sleep(30)  # wedged inside generation; never replies
        server.close()

    def dying_worker(self, seconds):
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen(1)
        connection, _ = server.accept()
        with connection:
            data = b""
            while b"\n" not in data:
                block = connection.recv(65536)
                if not block:
                    break
                data += block
            time.sleep(seconds)  # dies mid-generation, this many seconds in
        server.close()

    def run_controller(self, budget, fast=True):
        argv = ["prog", "--record", str(self.record_path),
                "--database", str(self.database),
                "--guidance", "guidance/prototype_inspection_v3.json",
                "--llama-root", str(Path(self.dir.name) / "unused_llama_root"),
                "--output-dir", str(self.output),
                "--llama-timeout-seconds", "1",
                "--llama-worker-socket", str(self.socket_path),
                "--request-deadline-monotonic", str(time.monotonic() + budget)]
        # Bound the test: the real warm timeout here is llama_timeout + 30.
        real_request = aew.request
        fast_request = lambda path, req, timeout_seconds=180, **kw: real_request(
            path, req, timeout_seconds=2)
        patched = mock.patch("forgesight.assist_edge_worker.request", fast_request) \
            if fast else mock.patch("forgesight.assist_edge_worker.request", real_request)
        with mock.patch.object(sys, "argv", argv), patched, \
             mock.patch.object(replay, "_generate_via_subprocess") as cold:
            with self.assertRaises(SystemExit):  # cold is mocked: no advice completes
                replay.main()
        return cold, json.loads((self.output / "advice_report.json").read_text())

    def test_hung_worker_falls_back_cold_and_flags_unhealthy(self):
        thread = threading.Thread(target=self.hanging_worker, daemon=True)
        thread.start()
        cold, report = self.run_controller(budget=30)
        cold.assert_called_once()
        # The fallback receives the remaining shared budget, not a fresh one.
        self.assertLessEqual(cold.call_args.kwargs["timeout_seconds"], 30 - 5)
        self.assertEqual(report["llama_runtime_mode"],
                         "PERSISTENT_WORKER_UNHEALTHY_COLD_FALLBACK")
        self.assertTrue(report["persistent_worker_unhealthy"])
        self.assertIn("TimeoutError", report["persistent_worker_transport_error"])
        self.assertIsNone(report["validation_error"])

    def test_late_worker_death_with_budget_falls_back_within_deadline(self):
        # Death 3 s into the request, generous shared budget: the fallback
        # runs but receives only the remaining budget, so the whole replay
        # still fits the caller's outer kill.
        thread = threading.Thread(target=self.dying_worker, args=(3,), daemon=True)
        thread.start()
        started = time.monotonic()
        cold, report = self.run_controller(budget=46, fast=False)
        elapsed = time.monotonic() - started
        cold.assert_called_once()
        granted = cold.call_args.kwargs["timeout_seconds"]
        self.assertGreater(granted, 0)
        self.assertLessEqual(granted, 46 - 3 - replay.REPORT_WRITE_RESERVE_SECONDS)
        self.assertLess(elapsed, 46)
        self.assertEqual(report["llama_runtime_mode"],
                         "PERSISTENT_WORKER_UNREACHABLE_COLD_FALLBACK")
        self.assertNotIn("cold_fallback_skipped", report)

    def test_late_worker_death_without_remaining_budget_skips_and_reports(self):
        # Same late death, tight shared budget: no honest cold run fits. The
        # skip is recorded and a complete report is still published well
        # inside the budget, instead of the replay being killed mid-write
        # and the half-created output dir blocking retry.
        thread = threading.Thread(target=self.dying_worker, args=(3,), daemon=True)
        thread.start()
        started = time.monotonic()
        cold, report = self.run_controller(budget=12, fast=False)
        elapsed = time.monotonic() - started
        cold.assert_not_called()
        self.assertLess(elapsed, 12)
        self.assertEqual(report.get("cold_fallback_skipped"), "SHARED_REQUEST_DEADLINE")
        self.assertIn("budget", report["runtime_error"])
        self.assertEqual(report["status"], "ADVICE_UNAVAILABLE")
        self.assertTrue(report["inspection_and_sqlite_unchanged"])

    def test_hung_worker_without_budget_skips_fallback_honestly(self):
        thread = threading.Thread(target=self.hanging_worker, daemon=True)
        thread.start()
        # Warm attempt is bounded to ~2s by the patched client; after the
        # reserve and the minimum viable generation, no cold run fits.
        cold, report = self.run_controller(budget=11)
        cold.assert_not_called()
        self.assertEqual(report.get("cold_fallback_skipped"), "SHARED_REQUEST_DEADLINE")
        self.assertIn("budget", report["runtime_error"])
        self.assertTrue(report["persistent_worker_unhealthy"])


class SharedDeadlineTests(unittest.TestCase):
    """Direct _generate_via_worker calls: the cold fallback must receive the
    remaining shared budget, never a fresh full one."""

    def args(self):
        return argparse.Namespace(llama_worker_socket=Path("/tmp/definitely-no-such.sock"),
                                  llama_timeout_seconds=75, llama_root=None)

    def call(self, deadline):
        report = {"status": "ADVICE_UNAVAILABLE", "runtime_requested": False}
        ctx = {"prompt_text": "hold part", "system_text": None}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(replay, "_generate_via_subprocess") as cold:
                replay._generate_via_worker(self.args(), ctx, Path(tmp), report,
                                            "SAVED_IMAGE_REPLAY", deadline=deadline)
        return cold, report

    def test_fallback_gets_remaining_budget(self):
        cold, report = self.call(time.monotonic() + 20)
        cold.assert_called_once()
        timeout = cold.call_args.kwargs["timeout_seconds"]
        self.assertGreater(timeout, 10)
        self.assertLessEqual(timeout, 20 - replay.REPORT_WRITE_RESERVE_SECONDS)
        self.assertEqual(report["llama_runtime_mode"],
                         "PERSISTENT_WORKER_UNREACHABLE_COLD_FALLBACK")

    def test_fallback_skipped_when_deadline_too_close(self):
        def slow_death(*a, **kw):
            time.sleep(2)
            raise ConnectionError("worker died mid-request")

        report = {"status": "ADVICE_UNAVAILABLE", "runtime_requested": False}
        ctx = {"prompt_text": "hold part", "system_text": None}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("forgesight.assist_edge_worker.request", slow_death), \
                 mock.patch.object(replay, "_generate_via_subprocess") as cold:
                replay._generate_via_worker(self.args(), ctx, Path(tmp), report,
                                            "SAVED_IMAGE_REPLAY",
                                            deadline=time.monotonic() + 11)
        cold.assert_not_called()
        self.assertEqual(report.get("cold_fallback_skipped"), "SHARED_REQUEST_DEADLINE")
        self.assertIn("budget", report["runtime_error"])

    def test_no_budget_preserves_legacy_full_cold_timeout(self):
        cold, report = self.call(None)
        cold.assert_called_once()
        # Legacy call shape: no budget override, the cold path applies its
        # own full timeout internally.
        self.assertNotIn("timeout_seconds", cold.call_args.kwargs)
        self.assertEqual(report["llama_runtime_mode"],
                         "PERSISTENT_WORKER_UNREACHABLE_COLD_FALLBACK")


class DeadlineValidationTests(unittest.TestCase):
    """A non-finite absolute deadline would silently disable the shared cap:
    NaN comparisons are false and min(cap, inf) is the legacy cap. Rejected
    at parse time instead."""

    def test_nonpositive_and_nonfinite_deadlines_rejected(self):
        for bad in ("0", "-5", "nan", "inf", "-inf"):
            argv = ["prog", "--record", "x", "--database", "y",
                    "--guidance", "z", "--llama-root", "l",
                    "--output-dir", "o",
                    "--request-deadline-monotonic", bad]
            with mock.patch.object(sys, "argv", argv), \
                 self.assertRaises(SystemExit) as raised:
                replay.main()
            self.assertEqual(raised.exception.code, 2, bad)


class ColdNativeOutputTests(unittest.TestCase):
    """Cold path: malformed or non-finite native output is a recorded error,
    never a crash and never a partial report."""

    def run_cold(self, run_json):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / "out"
        (root / "native").mkdir(parents=True)
        (root / "native" / "run.json").write_text(run_json)
        (root / "native" / "generated.txt").write_text("x")
        args = argparse.Namespace(llama_root=Path(tmp.name), llama_timeout_seconds=75)
        report = {"status": "ADVICE_UNAVAILABLE"}
        ctx = {"prompt_text": "hold part", "system_text": None}
        done = __import__("subprocess").CompletedProcess(args=[], returncode=0)
        with mock.patch("subprocess.run", return_value=done), \
             mock.patch.object(replay, "parse_advisory",
                               return_value={"response": {"generated_note": "n"}}):
            replay._generate_via_subprocess(args, ctx, root, report, "SAVED_IMAGE_REPLAY")
        tmp.cleanup()
        return report

    def test_nan_in_native_run_json_is_rejected_and_report_stays_strict(self):
        report = self.run_cold(json.dumps({
            "status": "JETSON_EXECUTORCH_LLAMA_GENERATION_VERIFIED",
            "process_wall_seconds": float("nan"),
            "sampled_peak_child_rss_kib": 1,
            "worker_result": {"artifact": {"sha256": "a" * 64},
                              "tokenizer": {"sha256": "b" * 64},
                              "prompt_token_count": 5, "stats": [],
                              "generated_text_display": "note"}}))
        self.assertEqual(report["status"], "ADVICE_UNAVAILABLE")
        self.assertIn("non-finite", report["runtime_error"])
        self.assertIsNone(report.get("native"))
        # The final strict report write would have raised on the NaN before.
        json.dumps(report, allow_nan=False)

    def test_truncated_native_run_json_is_a_recorded_error(self):
        report = self.run_cold('{"status": "JETSON_EXECUTORCH')
        self.assertEqual(report["status"], "ADVICE_UNAVAILABLE")
        self.assertIn("unreadable", report["runtime_error"])


class UiRecoveryTests(unittest.TestCase):
    """UI side: incomplete output dirs are quarantined for retry, finished
    requests no longer block, and a wedged worker is replaced exactly once."""

    def service(self, worker=None):
        from forgesight.jetson_live_ui import LiveService
        from forgesight.assist_queue import AdvisoryQueue
        from threading import Lock
        with mock.patch.object(LiveService, "__init__", lambda self, *a, **k: None):
            service = LiveService()
        service.args = argparse.Namespace(
            database=Path("db.sqlite3"), guidance=Path("guidance.json"),
            llama_root=Path("/llama"), llama_timeout_seconds=240,
            llama_worker_socket=Path("/tmp/assist-test.sock"))
        service.llama_ready = True
        service.advice_worker = worker
        service.advice_worker_boot_error = None
        service.advice_worker_respawning = False
        service.advice_worker_respawn_lock = Lock()
        service.advice_queue = AdvisoryQueue()
        service.advice_futures = {}
        service.lock = Lock()
        service.busy = False
        service.advice_root = Path(tempfile.mkdtemp())
        root = Path(tempfile.mkdtemp())
        record = {"inspection_id": "insp-1", "status": "ANALYZED",
                  "candidate": "ANOMALY_CANDIDATE",
                  "source": {"mode": "SAVED_IMAGE_REPLAY"}}
        (root / "record.json").write_text(json.dumps(record))
        service.latest = {**record, "source_mode": "SAVED_IMAGE_REPLAY"}
        service.latest_root = root
        service.store = mock.Mock()
        service.store.get_inspection.return_value = record
        return service

    def submit(self, service):
        with mock.patch("forgesight.jetson_live_ui.subprocess.run") as run:
            status = service.request_advice()
            if status == "PENDING":
                service.advice_futures["insp-1"].result(timeout=10)
        service.advice_queue.close()
        return status, run

    def test_incomplete_output_dir_is_quarantined_and_retried(self):
        service = self.service()
        partial = service.advice_root / "insp-1"
        partial.mkdir()
        (partial / "prompt.txt").write_text("half-written")
        status, run = self.submit(service)
        self.assertEqual(status, "PENDING")
        command = run.call_args.args[0]
        self.assertIn("--request-deadline-monotonic", command)
        deadline = float(command[command.index("--request-deadline-monotonic") + 1])
        self.assertLess(280, deadline - time.monotonic())
        self.assertLessEqual(deadline - time.monotonic(), 240 + 45)
        quarantined = list(service.advice_root.glob("insp-1.incomplete-*"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual((quarantined[0] / "prompt.txt").read_text(), "half-written")
        self.assertFalse(partial.exists())

    def full_record(self, service):
        record = json.loads(Path("manifests/graded_inspection_replay_v1.json").read_text())["records"][0]
        record["inspection_id"] = "insp-1"
        (service.latest_root / "record.json").write_text(json.dumps(record))
        service.store.get_inspection.return_value = record
        return record

    def valid_report(self, record, **overrides):
        from forgesight.assist_contract import canonical_hash
        report = {
            "inspection_id": "insp-1",
            "inspection_sha256": canonical_hash(record),
            "decision_sha256": canonical_hash(record["decision"]),
            "authoritative_grade": record["decision"]["grade"],
            "source_mode": "SAVED_IMAGE_REPLAY",
            "physical_part_inspected": False,
            "inspection_and_sqlite_unchanged": True,
            "status": "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW",
            "llama_runtime_mode": "PERSISTENT_WORKER",
            "native": {"status": "GENERATED_BY_PERSISTENT_WORKER"},
            "advisory": {
                "inspection_id": "insp-1",
                "inspection_sha256": canonical_hash(record),
                "decision_sha256": canonical_hash(record["decision"]),
                "authoritative_grade": record["decision"]["grade"],
                "quality_grade_authority": "DETERMINISTIC_RULES_ONLY",
                "jetson_verified": True,
                "source_mode": "SAVED_IMAGE_REPLAY",
                "guidance_approval": "PROTOTYPE_NOT_FACTORY_APPROVED",
                "response": {"generated_note": "[E1] Note [G1]",
                             "evidence_ids": ["E1", "G1"]}}}
        report.update(overrides)
        return report

    def test_complete_report_still_blocks_duplicate_request(self):
        service = self.service()
        record = self.full_record(service)
        complete = service.advice_root / "insp-1"
        complete.mkdir()
        (complete / "advice_report.json").write_text(json.dumps(self.valid_report(record)))
        status, run = self.submit(service)
        self.assertEqual(status, "ALREADY_REQUESTED")
        run.assert_not_called()
        self.assertFalse(list(service.advice_root.glob("insp-1.incomplete-*")))
        self.assertFalse(list(service.advice_root.glob("insp-1.failed-*")))

    def test_claimed_success_without_provenance_does_not_block_retry(self):
        # A bare {"status": success} report is withheld by advice_for; if it
        # also blocked requests, the UI would offer a retry the server always
        # refuses. It is quarantined aside and a fresh attempt starts.
        for name, content in (("claimed-success", json.dumps(
                                  {"status": "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW"})),
                              ("json-array", "[]"),
                              ("json-null", "null")):
            with self.subTest(case=name):
                service = self.service()
                self.full_record(service)
                claimed = service.advice_root / "insp-1"
                claimed.mkdir()
                (claimed / "advice_report.json").write_text(content)
                status, run = self.submit(service)
                self.assertEqual(status, "PENDING")
                run.assert_called_once()
                quarantined = list(service.advice_root.glob("insp-1.failed-*"))
                self.assertEqual(len(quarantined), 1)
                self.assertEqual((quarantined[0] / "advice_report.json").read_text(), content)
                self.assertFalse(claimed.exists())

    def test_failed_report_is_preserved_aside_and_retried(self):
        # Reproduced from the independent sweep: a saved ADVICE_UNAVAILABLE
        # report used to block every later request with ALREADY_REQUESTED and
        # the UI disabled Generate, leaving no normal retry route.
        for name, content in (("declared-failure", json.dumps({"status": "ADVICE_UNAVAILABLE"})),
                              ("unreadable", "{not json")):
            with self.subTest(case=name):
                service = self.service()
                failed = service.advice_root / "insp-1"
                failed.mkdir()
                (failed / "advice_report.json").write_text(content)
                status, run = self.submit(service)
                self.assertEqual(status, "PENDING")
                run.assert_called_once()
                quarantined = list(service.advice_root.glob("insp-1.failed-*"))
                self.assertEqual(len(quarantined), 1)
                self.assertEqual((quarantined[0] / "advice_report.json").read_text(), content)
                self.assertFalse(failed.exists())

    def test_saved_advisory_reports_the_runtime_that_produced_it(self):
        service = self.service()
        service.advice_worker = None  # no worker alive NOW
        record = json.loads(Path("manifests/graded_inspection_replay_v1.json").read_text())["records"][0]
        record["inspection_id"] = "insp-1"
        (service.latest_root / "record.json").write_text(json.dumps(record))
        service.store.get_inspection.return_value = record
        folder = service.advice_root / "insp-1"
        folder.mkdir()
        from forgesight.assist_contract import canonical_hash
        (folder / "advice_report.json").write_text(json.dumps({
            "inspection_id": "insp-1",
            "inspection_sha256": canonical_hash(record),
            "decision_sha256": canonical_hash(record["decision"]),
            "authoritative_grade": record["decision"]["grade"],
            "source_mode": "SAVED_IMAGE_REPLAY",
            "physical_part_inspected": False,
            "inspection_and_sqlite_unchanged": True,
            "status": "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW",
            "llama_runtime_mode": "PERSISTENT_WORKER",
            "native": {"status": "GENERATED_BY_PERSISTENT_WORKER"},
            "advisory": {
                "inspection_id": "insp-1",
                "inspection_sha256": canonical_hash(record),
                "decision_sha256": canonical_hash(record["decision"]),
                "authoritative_grade": record["decision"]["grade"],
                "quality_grade_authority": "DETERMINISTIC_RULES_ONLY",
                "jetson_verified": True,
                "source_mode": "SAVED_IMAGE_REPLAY",
                "guidance_approval": "PROTOTYPE_NOT_FACTORY_APPROVED",
                "response": {"generated_note": "[E1] Note [G1]",
                             "evidence_ids": ["E1", "G1"]}}}))
        advice = service.advice_state()
        self.assertEqual(advice["status"], "GENERATED_REQUIRES_HUMAN_REVIEW")
        self.assertEqual(advice["llama_runtime"], "PERSISTENT_WORKER")
        self.assertFalse(advice["advice_worker_available_now"])
        service.advice_queue.close()

    def test_legacy_report_without_recorded_runtime_is_not_recorded(self):
        # A valid success report written before runtime recording must not
        # inherit the CURRENT worker health as its producing runtime.
        for name, runtime in (("missing", ...), ("non-string", 42)):
            with self.subTest(case=name):
                service = self.service()
                record = self.full_record(service)
                report = self.valid_report(record)
                if runtime is ...:
                    del report["llama_runtime_mode"]
                else:
                    report["llama_runtime_mode"] = runtime
                folder = service.advice_root / "insp-1"
                folder.mkdir()
                (folder / "advice_report.json").write_text(json.dumps(report))
                advice = service.advice_state()
                self.assertEqual(advice["status"], "GENERATED_REQUIRES_HUMAN_REVIEW")
                self.assertEqual(advice["llama_runtime"], "NOT_RECORDED")
                service.advice_queue.close()

    def test_finished_request_without_report_does_not_block_retry(self):
        service = self.service()
        done = mock.Mock()
        done.done.return_value = True
        service.advice_futures["insp-1"] = done
        status, _ = self.submit(service)
        self.assertEqual(status, "PENDING")

    def test_pending_request_still_blocks(self):
        service = self.service()
        pending = mock.Mock()
        pending.done.return_value = False
        service.advice_futures["insp-1"] = pending
        status, run = self.submit(service)
        self.assertEqual(status, "ALREADY_REQUESTED")
        run.assert_not_called()

    def test_unhealthy_marker_respawns_worker_exactly_once(self):
        worker = mock.Mock()
        worker.poll.return_value = None  # alive but wedged
        service = self.service(worker)
        future = mock.Mock()
        future.done.return_value = True
        future.result.return_value = __import__("subprocess").CompletedProcess(
            args=[], returncode=1,
            stdout=json.dumps({"status": "ADVICE_UNAVAILABLE",
                               "persistent_worker_unhealthy": True}),
            stderr="")
        service.advice_futures["insp-1"] = future
        service._respawn_advice_worker = mock.Mock()
        service.advice_state()
        service.advice_state()
        service._respawn_advice_worker.assert_called_once()
        service.advice_queue.close()


class AdvisoryModeSelectionTests(unittest.TestCase):
    """The advisory contract is an explicit operator choice per request: the
    defect-type profile only gates availability of the reviewed A/B check
    contract, and the record's contents never select the mode."""

    def service(self, profile, record=None):
        from forgesight.jetson_live_ui import LiveService
        from forgesight.assist_queue import AdvisoryQueue
        from threading import Lock
        with mock.patch.object(LiveService, "__init__", lambda self, *a, **k: None):
            service = LiveService()
        service.args = argparse.Namespace(
            database=Path("db.sqlite3"), guidance=Path("guidance.json"),
            llama_root=Path("/llama"), llama_timeout_seconds=240,
            llama_worker_socket=None, defect_type_profile=profile)
        service.llama_ready = True
        service.advice_worker = None
        service.advice_worker_boot_error = None
        service.advice_queue = AdvisoryQueue()
        service.advice_futures = {}
        service.lock = Lock()
        service.busy = False
        service.advice_root = Path(tempfile.mkdtemp())
        root = Path(tempfile.mkdtemp())
        if record is None:
            record = {"inspection_id": "insp-1", "status": "ANALYZED",
                      "candidate": "ANOMALY_CANDIDATE",
                      "source": {"mode": "USER_UPLOADED_IMAGE_DIAGNOSTIC"}}
        (root / "record.json").write_text(json.dumps(record))
        service.latest = {**record, "source_mode": record["source"]["mode"]}
        service.latest_root = root
        service.store = mock.Mock()
        service.store.get_inspection.return_value = record
        return service

    def assessed_record(self):
        from forgesight.defect_severity import assess_severity
        row = json.loads(Path("manifests/operator_app_verification_v1.json")
                         .read_text())["browser_proof"]["inspections"][1]["record"]
        row["source"].update(mode="SAVED_IMAGE_REPLAY", physical_part=False)
        row["assessment"] = {
            "schema_version": 1, "scope": "PROVISIONAL_BENCHMARK_IMAGE",
            "method": "four_type_forest", "defect_type": "scratch",
            "severity": assess_severity("scratch", row["geometry"], (500, 250_000))}
        return row

    def command_for(self, profile, mode="composed", record=None):
        service = self.service(profile, record)
        try:
            with mock.patch("forgesight.jetson_live_ui.subprocess.run") as run:
                status = service.request_advice(mode)
                self.assertEqual(status, "PENDING")
                inspection_id = service.latest["inspection_id"]
                service.advice_futures[inspection_id].result(timeout=10)
            return run.call_args.args[0]
        finally:
            service.advice_queue.close()

    def test_default_request_is_explicitly_composed(self):
        command = self.command_for(profile="configs/defect_type_mvtec_four_v1.json")
        self.assertEqual(command[command.index("--advisory-mode") + 1], "composed")

    def test_operator_can_choose_vision_choice_when_profile_loaded(self):
        command = self.command_for(profile="configs/defect_type_mvtec_four_v1.json",
                                   mode="vision_choice", record=self.assessed_record())
        self.assertEqual(command[command.index("--advisory-mode") + 1], "vision_choice")

    def test_vision_choice_refused_for_record_without_assessment(self):
        service = self.service(profile="configs/defect_type_mvtec_four_v1.json")
        try:
            with mock.patch("forgesight.jetson_live_ui.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    service.request_advice("vision_choice")
            run.assert_not_called()
        finally:
            service.advice_queue.close()

    def test_composed_refused_for_assessed_record(self):
        service = self.service(profile="configs/defect_type_mvtec_four_v1.json",
                               record=self.assessed_record())
        try:
            with mock.patch("forgesight.jetson_live_ui.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    service.request_advice("composed")
            run.assert_not_called()
        finally:
            service.advice_queue.close()

    def test_vision_choice_refused_without_profile(self):
        service = self.service(profile=None)
        try:
            with mock.patch("forgesight.jetson_live_ui.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    service.request_advice("vision_choice")
            run.assert_not_called()
        finally:
            service.advice_queue.close()

    def test_unknown_mode_refused(self):
        service = self.service(profile="configs/defect_type_mvtec_four_v1.json")
        try:
            with mock.patch("forgesight.jetson_live_ui.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    service.request_advice("autopilot")
            run.assert_not_called()
        finally:
            service.advice_queue.close()


class AdviceFuturesBoundTest(unittest.TestCase):
    """LiveService.advice_futures grew by one entry per inspection forever."""

    def service(self):
        from forgesight.jetson_live_ui import LiveService
        service = LiveService.__new__(LiveService)
        service.advice_futures = {}
        return service

    def future(self, done):
        f = mock.Mock()
        f.done.return_value = done
        return f

    def test_completed_futures_pruned_to_newest_bound(self):
        from forgesight.jetson_live_ui import ADVICE_FUTURES_COMPLETED_KEPT
        service = self.service()
        for i in range(ADVICE_FUTURES_COMPLETED_KEPT + 10):
            service.advice_futures["old-%d" % i] = self.future(done=True)
        service._prune_advice_futures()
        self.assertEqual(len(service.advice_futures), ADVICE_FUTURES_COMPLETED_KEPT)
        self.assertNotIn("old-0", service.advice_futures)
        self.assertIn("old-%d" % (ADVICE_FUTURES_COMPLETED_KEPT + 9),
                       service.advice_futures)

    def test_pending_futures_never_evicted(self):
        from forgesight.jetson_live_ui import ADVICE_FUTURES_COMPLETED_KEPT
        service = self.service()
        for i in range(ADVICE_FUTURES_COMPLETED_KEPT + 5):
            service.advice_futures["done-%d" % i] = self.future(done=True)
        for i in range(3):
            service.advice_futures["pending-%d" % i] = self.future(done=False)
        service._prune_advice_futures()
        for i in range(3):
            self.assertIn("pending-%d" % i, service.advice_futures)

    def test_below_bound_is_untouched(self):
        service = self.service()
        service.advice_futures["a"] = self.future(done=True)
        service._prune_advice_futures()
        self.assertIn("a", service.advice_futures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
