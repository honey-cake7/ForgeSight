"""UI worker lifecycle: stderr flood cannot deadlock boot; respawn after death."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import threading
import time
from pathlib import Path
from unittest import mock

from forgesight.jetson_live_ui import LiveService, launch_advice_worker, launch_vision_worker

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

STUB_CRASH = textwrap.dedent("""
    import sys
    sys.stderr.write("Traceback (most recent call last): KeyError: 'prompt'")
    sys.exit(1)
""")

STUB_NOISY_READY = textwrap.dedent("""
    import json, time
    print("A runtime", flush=True)
    print('PyTorchObserver {"generated_tokens": 1}', flush=True)
    print(json.dumps({"status": "WORKER_READY"}), flush=True)
    time.sleep(30)
""")


def make_args(socket_path):
    return argparse.Namespace(
        vision_python=sys.executable, model=Path("model.ts"),
        vision_worker_socket=socket_path, vision_worker_threads=1,
        vision_worker_idle_timeout_seconds=60)


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux worker lifetime regression')
class RespawnThreadLifetimeTests(unittest.TestCase):
    def test_workers_survive_the_thread_that_launched_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / 'ready.py'
            stub.write_text('from forgesight.worker_lifecycle import watch_station_parent\nwatch_station_parent()\n'
                            'import json, time\nprint(json.dumps({"status":"WORKER_READY"}), flush=True)\ntime.sleep(30)\n',
                            encoding='utf-8')
            for kind in ('vision', 'advice'):
                with self.subTest(worker=kind):
                    workers, errors = [], []
                    def launch():
                        try:
                            if kind == 'vision':
                                workers.append(launch_vision_worker(make_args(root / 'vision.sock'), worker_script=stub))
                            else:
                                args = argparse.Namespace(llama_root=root, llama_worker_socket=root / 'advice.sock',
                                                          llama_worker_idle_timeout_seconds=60)
                                workers.append(launch_advice_worker(args, python_command=[sys.executable, str(stub)]))
                        except Exception as error:
                            errors.append(error)
                    thread = threading.Thread(target=launch)
                    thread.start()
                    thread.join(timeout=10)
                    try:
                        self.assertFalse(thread.is_alive())
                        self.assertEqual(errors, [])
                        self.assertEqual(len(workers), 1)
                        time.sleep(.2)  # let a parent-thread-death signal be delivered
                        self.assertIsNone(workers[0].poll(), 'Worker died when its respawn thread returned')
                    finally:
                        for worker in workers:
                            if worker.poll() is None:
                                worker.terminate()
                            worker.wait(timeout=5)
                            worker.stdout.close()
                            worker.stderr.close()

    def test_owned_workers_exit_when_the_station_process_exits(self):
        def running(pid):
            try:
                status = Path(f'/proc/{pid}/stat').read_text()
                return status.rsplit(')', 1)[1].split()[0] != 'Z'
            except FileNotFoundError:
                return False

        parent_script = textwrap.dedent('''
            import argparse, sys, threading, time
            from pathlib import Path
            from forgesight.jetson_live_ui import launch_advice_worker, launch_vision_worker
            root, kind = Path(sys.argv[1]), sys.argv[2]
            workers = []
            def launch():
                if kind == 'vision':
                    args = argparse.Namespace(vision_python=sys.executable, model=root/'model.ts',
                        vision_worker_socket=root/'v.sock', vision_worker_threads=1,
                        vision_worker_idle_timeout_seconds=60)
                    workers.append(launch_vision_worker(args, worker_script=root/'stub.py'))
                else:
                    args = argparse.Namespace(llama_root=root, llama_worker_socket=root/'a.sock',
                        llama_worker_idle_timeout_seconds=60)
                    workers.append(launch_advice_worker(args, python_command=[sys.executable, str(root/'stub.py')]))
            thread = threading.Thread(target=launch)
            thread.start()
            thread.join()
            (root/'worker.pid').write_text(str(workers[0].pid))
            time.sleep(30)
        ''')
        for kind in ('vision', 'advice'):
            with self.subTest(worker=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'stub.py').write_text(
                    'from forgesight.worker_lifecycle import watch_station_parent\nwatch_station_parent()\n'
                    'import json,time\nprint(json.dumps({"status":"WORKER_READY"}),flush=True)\ntime.sleep(30)\n',
                    encoding='utf-8')
                parent = subprocess.Popen([sys.executable, '-B', '-c', parent_script, str(root), kind],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                worker_pid = None
                try:
                    deadline = time.monotonic() + 10
                    while not (root / 'worker.pid').exists() and parent.poll() is None and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue((root / 'worker.pid').exists(), 'Test station did not start its worker')
                    worker_pid = int((root / 'worker.pid').read_text())
                    self.assertTrue(running(worker_pid))
                    parent.terminate()
                    parent.wait(timeout=5)
                    deadline = time.monotonic() + 5
                    while running(worker_pid) and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertFalse(running(worker_pid), 'Worker outlived its station process')
                finally:
                    if parent.poll() is None:
                        parent.terminate()
                    parent.wait(timeout=5)
                    parent.stderr.close()
                    if worker_pid is not None and running(worker_pid):
                        os.kill(worker_pid, 15)


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


class LaunchAdviceWorkerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make_args(self):
        return argparse.Namespace(
            llama_root=self.dir, llama_worker_socket=self.dir / "worker.sock",
            llama_worker_idle_timeout_seconds=1)

    def test_boot_crash_reports_exit_and_stderr_not_timeout(self):
        stub = self.dir / "crash.py"
        stub.write_text(STUB_CRASH)
        with self.assertRaises(ValueError) as raised:
            launch_advice_worker(self.make_args(), python_command=[sys.executable, str(stub)])
        message = str(raised.exception)
        self.assertIn("exited during startup", message)
        self.assertIn("KeyError", message)
        self.assertNotIn("180 seconds", message)

    def test_stdout_diagnostics_before_ready_do_not_trigger_cold_fallback(self):
        stub = self.dir / "noisy_ready.py"
        stub.write_text(STUB_NOISY_READY)
        worker = launch_advice_worker(self.make_args(),
                                      python_command=[sys.executable, str(stub)])
        try:
            self.assertIsNone(worker.poll())
        finally:
            worker.terminate()
            worker.wait(timeout=10)


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
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

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

    def replay_with_record(self, mode):
        service = bare_service()
        from threading import Lock
        service.lock = Lock()
        service.busy = True
        service.args.dataset_manifest = Path("manifest")
        service.args.vision_worker_socket = Path("/tmp/worker.sock")
        root = Path(self._tmp.name) / ("replay-" + str(mode is not None))
        result = root / "result"
        result.mkdir(parents=True)
        record = self.record(mode)
        record["inspection_id"] = "00000000-0000-0000-0000-000000000000"
        (result / "record.json").write_text(json.dumps(record))
        service._publish_result = mock.Mock()
        service._respawn_vision_worker = mock.Mock()
        with mock.patch("forgesight.jetson_live_ui.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            service._run_replay("replay-t1", root / "image.png", {"sha256": "a" * 64}, root)
        return service

    def test_replay_cold_fallback_marker_triggers_respawn(self):
        service = self.replay_with_record("COLD_PROCESS_AFTER_WORKER_UNAVAILABLE")
        service._respawn_vision_worker.assert_called_once()

    def test_replay_warm_and_missing_records_do_not_trigger_respawn(self):
        service = self.replay_with_record(None)
        service._respawn_vision_worker.assert_not_called()
        service = bare_service()
        from threading import Lock
        service.lock = Lock()
        service.busy = True
        service._publish_result = mock.Mock()
        service._respawn_vision_worker = mock.Mock()
        root = Path(self._tmp.name) / "replay-no-record"
        service.args.dataset_manifest = Path("manifest")
        with mock.patch("forgesight.jetson_live_ui.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            service._run_replay("replay-t2", root / "image.png", {"sha256": "a" * 64}, root)
        run.assert_called_once()
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
