"""Protocol and validation tests for the persistent vision worker; no Torch needed."""

import json
import socket
import threading
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


if __name__ == "__main__":
    unittest.main()
