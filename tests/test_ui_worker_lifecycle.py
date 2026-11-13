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
    import json, sys
    sys.stderr.write("x" * 524288)  # 512 KiB: far beyond the 64 KiB pipe buffer
    sys.stderr.flush()
    print(json.dumps({"status": "WORKER_READY"}), flush=True)
    sys.stdout.readline()
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

    def captured_command(self, service):
        with mock.patch("forgesight.jetson_live_ui.subprocess.run") as run:
            assert service.request_advice() == "PENDING"
            future = service.advice_futures["insp-1"]
            future.result(timeout=10)
        service.advice_queue.close()
        run.assert_called_once()
        return run.call_args.args[0]

    def test_live_worker_socket_reaches_advice_command(self):
        worker = mock.Mock()
        worker.poll.return_value = None  # alive
        command = self.captured_command(self.service(worker))
        self.assertIn("--llama-worker-socket", command)
        self.assertIn("/tmp/assist-test.sock", command)

    def test_dead_worker_cold_path_and_respawn(self):
        worker = mock.Mock()
        worker.poll.return_value = 1  # dead
        service = self.service(worker)
        service._respawn_advice_worker = mock.Mock()
        command = self.captured_command(service)
        self.assertNotIn("--llama-worker-socket", command)
        service._respawn_advice_worker.assert_called_once()

    def test_close_terminates_advice_worker(self):
        worker = mock.Mock()
        worker.wait.return_value = 0
        service = self.service(worker)
        service.vision_worker = None
        service.close()
        worker.terminate.assert_called_once()
