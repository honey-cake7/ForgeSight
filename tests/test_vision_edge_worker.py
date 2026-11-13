"""Protocol and validation tests for the persistent vision worker; no Torch needed."""

import json
import socket
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from forgesight.vision_edge_worker import request, serve, validate_request


class ValidateRequestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self.tensor = root / "input.npy"
        self.tensor.write_bytes(b"not-a-real-tensor")
        self.out = root / "inference.json"
        self.base = {"tensor": str(self.tensor), "tensor_sha256": "a" * 64,
                     "threshold": 0.2277658, "output": str(self.out)}

    def tearDown(self):
        self._tmp.cleanup()

    def test_minimal_request_accepted(self):
        checked = validate_request(dict(self.base))
        self.assertEqual(checked["threshold"], 0.2277658)
        self.assertIsNone(checked["map_output"])
        self.assertIsNone(checked["source_map_output"])

    def test_unknown_field_rejected(self):
        body = dict(self.base, model="/etc/passwd")
        with self.assertRaises(ValueError):
            validate_request(body)

    def test_missing_tensor_rejected(self):
        body = dict(self.base, tensor=str(self.tensor) + ".absent")
        with self.assertRaises(ValueError):
            validate_request(body)

    def test_bad_hash_shape_rejected(self):
        with self.assertRaises(ValueError):
            validate_request(dict(self.base, tensor_sha256="short"))

    def test_non_finite_threshold_rejected(self):
        for bad in (float("nan"), float("inf"), "0.5"):
            with self.assertRaises(ValueError):
                validate_request(dict(self.base, threshold=bad))

    def test_existing_output_rejected(self):
        self.out.write_text("{}")
        with self.assertRaises(ValueError):
            validate_request(dict(self.base))

    def test_source_map_requires_size(self):
        with self.assertRaises(ValueError):
            validate_request(dict(self.base, source_map_output="map.npy"))
        with self.assertRaises(ValueError):
            validate_request(dict(self.base, source_size=[700, 700]))

    def test_oversized_tensor_file_rejected_before_read(self):
        big = Path(self._tmp.name) / "big.npy"
        big.write_bytes(b"\x00" * (1024 * 1024 + 1))  # one byte past the bound
        with self.assertRaises(ValueError):
            validate_request(dict(self.base, tensor=str(big),
                                  tensor_sha256="b" * 64))
        # The exact np.save contract size and header-variant slack both pass.
        for size in (786560, 900000):
            ok = Path(self._tmp.name) / f"ok_{size}.npy"
            ok.write_bytes(b"\x00" * size)
            checked = validate_request(dict(self.base, tensor=str(ok),
                                            tensor_sha256="b" * 64))
            self.assertEqual(checked["tensor"], ok)

    def test_source_size_bounds(self):
        root = Path(self._tmp.name)
        for bad in ([0, 700], [700, 5000], [700.5, 700], ["700", 700], [700]):
            with self.assertRaises(ValueError):
                validate_request(dict(self.base, source_map_output=str(root / "m.npy"),
                                      source_size=bad))


class StubWorker:
    def __init__(self):
        self.requests = []

    def handle(self, raw):
        self.requests.append(raw)
        if raw.get("threshold") == 13.0:
            raise ValueError("stub rejection")
        return {"status": "ONE_IMAGE_INFERENCE_VERIFIED", "score": 0.5}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.socket_path = Path(self._tmp.name) / "worker.sock"
        self.worker = StubWorker()
        self.thread = threading.Thread(
            target=serve, args=(self.socket_path, self.worker, 30.0), daemon=True)
        self.thread.start()
        import time
        deadline = time.monotonic() + 5
        while not self.socket_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.socket_path.exists(), "worker socket was not created")

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        report = request(self.socket_path, {"tensor": "x", "threshold": 0.1})
        self.assertEqual(report["score"], 0.5)
        self.assertEqual(self.worker.requests, [{"tensor": "x", "threshold": 0.1}])

    def test_worker_error_propagates(self):
        with self.assertRaises(RuntimeError) as caught:
            request(self.socket_path, {"tensor": "x", "threshold": 13.0})
        self.assertIn("stub rejection", str(caught.exception))

    def test_malformed_request_gets_error_payload(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(self.socket_path))
            connection.sendall(b"not json\n")
            reply = connection.recv(65536)
        payload = json.loads(reply.split(b"\n", 1)[0])
        self.assertEqual(payload["status"], "ERROR")

    def test_sequential_requests(self):
        for index in range(3):
            report = request(self.socket_path, {"tensor": f"t{index}", "threshold": 0.1})
            self.assertEqual(report["status"], "ONE_IMAGE_INFERENCE_VERIFIED")
        self.assertEqual(len(self.worker.requests), 3)


class StalledClientTests(unittest.TestCase):
    """A client that stops mid-request must not block every later request."""

    def test_trickling_client_cannot_extend_total_read_deadline(self):
        with TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "worker.sock"
            worker = StubWorker()
            server = threading.Thread(
                target=serve, args=(socket_path, worker, 1.0, 0.5), daemon=True)
            server.start()
            request(socket_path, {"tensor": "ready", "threshold": 0.1},
                    timeout_seconds=5, connect_retries=200)
            worker.requests.clear()
            stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stop = threading.Event()

            def send_slowly():
                while not stop.wait(0.05):
                    try:
                        stalled.sendall(b" ")
                    except OSError:
                        return

            sender = threading.Thread(target=send_slowly, daemon=True)
            try:
                stalled.connect(str(socket_path))
                stalled.sendall(b'{"tensor":')
                sender.start()
                started = time.monotonic()
                report = request(socket_path, {"tensor": "x", "threshold": 0.1},
                                 timeout_seconds=1.5, connect_retries=1)
                waited = time.monotonic() - started
                self.assertEqual(report["score"], 0.5)
                self.assertLess(waited, 1.25, "request bytes extended the total deadline")
                again = request(socket_path, {"tensor": "y", "threshold": 0.1},
                                timeout_seconds=1.5, connect_retries=1)
                self.assertEqual(again["score"], 0.5)
                self.assertEqual(worker.requests, [
                    {"tensor": "x", "threshold": 0.1},
                    {"tensor": "y", "threshold": 0.1}])
            finally:
                stop.set()
                stalled.close()
                if sender.ident is not None:
                    sender.join(timeout=2)
                server.join(timeout=2)
            self.assertFalse(server.is_alive(), "test worker did not shut down")

    def test_stalled_client_dropped_and_next_request_served(self):
        import time
        with TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "worker.sock"
            worker = StubWorker()
            thread = threading.Thread(
                target=serve, args=(socket_path, worker, None, 0.5), daemon=True)
            thread.start()
            deadline = time.monotonic() + 5
            ready = False
            while time.monotonic() < deadline:
                if socket_path.exists():
                    try:
                        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        probe.connect(str(socket_path))
                    except OSError:
                        time.sleep(0.01)
                        continue
                    probe.close()
                    ready = True
                    break
                time.sleep(0.01)
            self.assertTrue(ready, "worker socket was not accepting")
            stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                stalled.connect(str(socket_path))
                stalled.sendall(b'{"tensor":')  # partial request line, then silence
                started = time.monotonic()
                report = request(socket_path, {"tensor": "x", "threshold": 0.1},
                                 timeout_seconds=10, connect_retries=1)
                waited = time.monotonic() - started
            finally:
                stalled.close()
            self.assertEqual(report["score"], 0.5)
            self.assertLess(waited, 8, "legit request waited far beyond the read timeout")
            # The server is still healthy after dropping the stalled client.
            again = request(socket_path, {"tensor": "y", "threshold": 0.1},
                            timeout_seconds=10, connect_retries=1)
            self.assertEqual(again["score"], 0.5)
            self.assertEqual(worker.requests, [{"tensor": "x", "threshold": 0.1},
                                               {"tensor": "y", "threshold": 0.1}])

    def test_slow_drip_client_dropped_by_total_deadline(self):
        """Each chunk arriving under the per-chunk timeout must not extend the
        total request-line deadline: a drip ties up the worker just as a stall."""
        import time
        with TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "worker.sock"
            worker = StubWorker()
            thread = threading.Thread(
                target=serve, args=(socket_path, worker, None, 0.6), daemon=True)
            thread.start()
            deadline = time.monotonic() + 5
            ready = False
            while time.monotonic() < deadline:
                if socket_path.exists():
                    try:
                        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        probe.connect(str(socket_path))
                    except OSError:
                        time.sleep(0.01)
                        continue
                    probe.close()
                    ready = True
                    break
                time.sleep(0.01)
            self.assertTrue(ready, "worker socket was not accepting")
            drip = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                drip.connect(str(socket_path))
                started = time.monotonic()
                # One byte every 0.15s: every chunk beats a per-chunk timeout,
                # but the total 0.6s deadline must still drop this client.
                while time.monotonic() - started < 1.5:
                    try:
                        drip.sendall(b"{")
                    except OSError:
                        break
                    time.sleep(0.15)
                held = time.monotonic() - started
                report = request(socket_path, {"tensor": "x", "threshold": 0.1},
                                 timeout_seconds=10, connect_retries=1)
            finally:
                drip.close()
            self.assertLess(held, 1.2, "drip client held the worker past the total deadline")
            self.assertEqual(report["score"], 0.5)

    def test_unbounded_reply_stream_rejected(self):
        import time
        with TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "worker.sock"
            class StreamingWorker:
                def handle(self, raw):
                    return {"blob": "x" * (17 * 1024 * 1024)}  # beyond the reply bound
            thread = threading.Thread(
                target=serve, args=(socket_path, StreamingWorker(), None, 30), daemon=True)
            thread.start()
            deadline = time.monotonic() + 5
            ready = False
            while time.monotonic() < deadline:
                if socket_path.exists():
                    try:
                        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        probe.connect(str(socket_path))
                    except OSError:
                        time.sleep(0.01)
                        continue
                    probe.close()
                    ready = True
                    break
                time.sleep(0.01)
            self.assertTrue(ready, "worker socket was not accepting")
            with self.assertRaises(ConnectionError):
                request(socket_path, {"tensor": "x", "threshold": 0.1},
                        timeout_seconds=30, connect_retries=1)


if __name__ == "__main__":
    unittest.main()
