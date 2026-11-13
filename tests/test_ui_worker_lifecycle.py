"""UI worker lifecycle: stderr flood cannot deadlock boot; respawn after death."""

import argparse
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from forgesight.jetson_live_ui import LiveService, launch_vision_worker

STUB_READY = textwrap.dedent("""
    import json, sys, time
    sys.stderr.write("x" * 524288)  # 512 KiB: far beyond the 64 KiB pipe buffer
    sys.stderr.flush()
    print(json.dumps({"status": "WORKER_READY"}), flush=True)
    time.sleep(30)  # stay alive until the test terminates the worker
""")

STUB_FAIL = textwrap.dedent("""
    import sys
    sys.stderr.write("hash mismatch on frozen model")
    sys.exit(1)
""")


def make_args(socket_path):
    return argparse.Namespace(
        vision_python=sys.executable, model=Path("model.ts"),
        vision_worker_socket=socket_path, vision_worker_threads=1,
        vision_worker_idle_timeout_seconds=60)


class LaunchVisionWorkerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_stub(self, name, body):
        script = self.dir / name
        script.write_text(body)
        return script

    def test_stderr_flood_does_not_deadlock_boot(self):
        worker = launch_vision_worker(make_args(self.dir / "a.sock"),
                                      worker_script=self.write_stub("flood.py", STUB_READY))
        try:
            self.assertIsNone(worker.poll())
        finally:
            worker.terminate()
            worker.wait(timeout=10)

    def test_boot_failure_reports_stderr_tail(self):
        with self.assertRaises(ValueError) as raised:
            launch_vision_worker(make_args(self.dir / "b.sock"),
                                 worker_script=self.write_stub("fail.py", STUB_FAIL))
        self.assertIn("hash mismatch on frozen model", str(raised.exception))


def bare_service():
    with mock.patch.object(LiveService, "__init__", lambda self, *a, **k: None):
        service = LiveService()
    service.args = argparse.Namespace(
        vision_python=Path("python"), model=Path("model.ts"),
        grade_policy=Path("policy.json"), database=Path("db.sqlite3"),
        camera_index=0, vision_worker_socket=Path("/tmp/worker.sock"),
        overlay_format="png")
    service.worker_respawning = False
    from threading import Lock
    service.worker_respawn_lock = Lock()
    return service


class RespawnTests(unittest.TestCase):
    def inspect_with_record(self, record):
        service = bare_service()
        service._publish_result = mock.Mock()
        service._respawn_vision_worker = mock.Mock()
        with mock.patch("forgesight.vision_edge_live_once.inspect", return_value=record):
            service._inspect_inline(Path("image.png"), "part-t1", Path("/tmp/root-t1"),
                                    dict(mvtec_benchmark_upload=True,
                                         uploaded_image_diagnostic=False,
                                         operator_declared_part=False,
                                         screen_photo_diagnostic=False,
                                         diagnostic_no_part=False))
        return service

    def record(self, mode):
        return {"status": "ANALYZED", "timings": {} if mode is None else {"vision_mode": mode}}

    def test_cold_fallback_marker_triggers_respawn(self):
        service = self.inspect_with_record(
            self.record("COLD_PROCESS_AFTER_WORKER_UNAVAILABLE"))
        service._respawn_vision_worker.assert_called_once()

    def test_warm_and_error_records_do_not_trigger_respawn(self):
        for record in (self.record(None),
                       {"status": "ERROR", "timings": {}, "error": {"message": "x"}}):
            service = self.inspect_with_record(record)
            service._respawn_vision_worker.assert_not_called()

    def test_respawn_replaces_dead_worker(self):
        service = bare_service()
        old = mock.Mock()
        old.poll.return_value = 0  # already dead
        service.vision_worker = old
        new = mock.Mock()
        with mock.patch("forgesight.jetson_live_ui.launch_vision_worker", return_value=new) as launch:
            service._respawn_vision_worker()
            import time
            for _ in range(100):
                if launch.called:
                    break
                time.sleep(0.01)
        launch.assert_called_once_with(service.args)
        self.assertIs(service.vision_worker, new)
        self.assertFalse(service.worker_respawning)


if __name__ == "__main__":
    unittest.main()


class AdviceWorkerWiringTests(unittest.TestCase):
    """The warm advisory path reaches the actual UI flow: request_advice
    passes the worker socket only while the worker lives, respawns when it
    dies, and close() stops it."""

    def service(self, worker):
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

    def captured_call(self, service):
        with mock.patch("forgesight.assist_jetson_replay_once.run_advisory") as run:
            assert service.request_advice() == "PENDING"
            future = service.advice_futures["insp-1"]
            future.result(timeout=10)
        service.advice_queue.close()
        run.assert_called_once()
        return run.call_args

    def test_live_worker_socket_reaches_advice_call(self):
        worker = mock.Mock()
        worker.poll.return_value = None  # alive
        call = self.captured_call(self.service(worker))
        self.assertEqual(call.kwargs["llama_worker_socket"], Path("/tmp/assist-test.sock"))

    def test_dead_worker_cold_path_and_respawn(self):
        worker = mock.Mock()
        worker.poll.return_value = 1  # dead
        service = self.service(worker)
        service._respawn_advice_worker = mock.Mock()
        call = self.captured_call(service)
        self.assertIsNone(call.kwargs["llama_worker_socket"])
        service._respawn_advice_worker.assert_called_once()

    def test_advice_runs_in_process_without_spawn(self):
        """The in-process path writes the same report files and never spawns
        the controller interpreter: real run_advisory against a fake worker."""
        import socket as socket_module
        import threading
        worker = mock.Mock()
        worker.poll.return_value = None
        service = self.service(worker)
        # Real fixtures: immutable record in a real store + real guidance doc.
        record = json.loads(Path("manifests/operator_app_verification_v1.json").read_text()
                            )["browser_proof"]["inspections"][1]["record"]
        record["model"]["jetson_verified"] = True
        record["source"]["physical_part"] = False
        (service.latest_root / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        service.latest = {**record, "source_mode": "SAVED_IMAGE_REPLAY"}
        from forgesight.storage import InspectionStore
        database = service.advice_root / "inspections.db"
        InspectionStore(database).save_inspection(record)
        service.args = argparse.Namespace(
            database=database, guidance=Path("guidance/prototype_inspection_v1.json"),
            llama_root=Path("/unused"), llama_timeout_seconds=240,
            llama_worker_socket=Path("/tmp/assist-test.sock"))
        service.store = InspectionStore(database, create=False)

        note = ("[E1] Anomaly flagged; grade REVIEW; defect type and cause unknown. "
                "[G1] Prototype checklist: hold for human review; check image clarity and lighting.")

        def fake_worker():
            server = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
            server.bind("/tmp/assist-test.sock")
            server.listen(1)
            connection, _ = server.accept()
            with connection:
                raw = b""
                while b"\n" not in raw:
                    raw += connection.recv(65536)
                payload = {"status": "OK", "report": {
                    "status": "GENERATED_BY_PERSISTENT_WORKER",
                    "worker_mode": "PERSISTENT_PROCESS",
                    "worker_started_utc": "2026-11-14T00:00:00Z",
                    "runtime_evidence": {"artifact": {"sha256": "m" * 64},
                                         "tokenizer": {"sha256": "t" * 64}},
                    "prompt_token_count": 40,
                    "stats": {"prompt_tokens": 40, "generated_tokens": 33},
                    "generated_text": note}}
                connection.sendall(json.dumps(payload).encode() + b"\n")
            server.close()

        thread = threading.Thread(target=fake_worker, daemon=True)
        thread.start()
        try:
            with mock.patch("forgesight.jetson_live_ui.subprocess.run",
                            side_effect=AssertionError("spawned a subprocess")):
                assert service.request_advice() == "PENDING"
                service.advice_futures[record["inspection_id"]].result(timeout=10)
        finally:
            service.advice_queue.close()
            Path("/tmp/assist-test.sock").unlink(missing_ok=True)
        thread.join(timeout=10)
        report = json.loads((service.advice_root / record["inspection_id"]
                             / "advice_report.json").read_text())
        self.assertEqual(report["status"], "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW")
        self.assertEqual(report["llama_runtime_mode"], "PERSISTENT_WORKER")
        self.assertTrue(report["inspection_and_sqlite_unchanged"])

    def test_advice_wall_deadline_bounds_a_hung_controller(self):
        """A controller that never returns does not block the advice thread
        past llama_timeout + slack; no report file appears (killed-subprocess
        parity)."""
        import time as time_module
        worker = mock.Mock()
        worker.poll.return_value = None
        service = self.service(worker)
        service.args.llama_timeout_seconds = 0.2
        with mock.patch("forgesight.assist_jetson_replay_once.run_advisory",
                        side_effect=lambda *a, **k: time_module.sleep(30)), \
             mock.patch("forgesight.jetson_live_ui.ADVICE_WALL_SLACK_SECONDS", 0.2):
            started = time_module.monotonic()
            assert service.request_advice() == "PENDING"
            service.advice_futures["insp-1"].result(timeout=10)
            elapsed = time_module.monotonic() - started
        service.advice_queue.close()
        self.assertLess(elapsed, 5)
        self.assertFalse((service.advice_root / "insp-1" / "advice_report.json").exists())

    def test_close_terminates_advice_worker(self):
        worker = mock.Mock()
        worker.wait.return_value = 0
        service = self.service(worker)
        service.vision_worker = None
        service.close()
        worker.terminate.assert_called_once()
