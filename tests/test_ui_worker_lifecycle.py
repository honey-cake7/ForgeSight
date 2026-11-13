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
