"""Adversarial and boundary coverage for the worker path; no Torch needed.

Pins the risky behaviors: mid-request worker death, partial artifacts,
rejection-vs-transport fallback rules, grading boundary equivalence between
cold and worker modes, malformed/large inputs, duplicate IDs, concurrency,
restart and idle-timeout lifecycle.
"""

import argparse
import json
import socket
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from PIL import Image

from forgesight import vision_edge_live_once as live
from forgesight.vision_edge_worker import request, serve
from tests.test_live_once_worker_mode import fake_inference_outputs

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "configs" / "forgegrade_benchmark_v1.json"
THRESHOLD = json.loads(POLICY.read_text())["image_threshold"]


def wait_socket(path, seconds=5):
    """Wait until the server is actually accepting (past bind AND listen)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if Path(path).exists():
            try:
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                probe.connect(str(path))
            except OSError:
                time.sleep(0.01)
                continue
            probe.close()
            return True
        time.sleep(0.01)
    return False


class LiveOnceAdversarialTests(unittest.TestCase):
    def setUp(self):
        LiveOnceAdversarialTests._seq = getattr(LiveOnceAdversarialTests, "_seq", 0) + 1
        self.seq = LiveOnceAdversarialTests._seq
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.image = self.root / "image.png"
        Image.new("RGB", (700, 700), "gray").save(self.image)
        self.model = self.root / "model.ts"
        self.model.write_bytes(b"stand-in model bytes")
        import hashlib
        self._sha_patch = mock.patch.object(
            live, "MODEL_SHA256", hashlib.sha256(self.model.read_bytes()).hexdigest())
        self._sha_patch.start()

    def tearDown(self):
        self._sha_patch.stop()
        self._tmp.cleanup()

    def make_args(self, worker_socket, part_id="part-t1"):
        return argparse.Namespace(
            image=self.image, part_id=part_id, vision_python=Path("python"),
            model=self.model, model_sha256=live.MODEL_SHA256, grade_policy=POLICY,
            database=self.root / f"db-{self.seq}-{part_id}.sqlite3",
            output_dir=self.root / f"out-{self.seq}-{part_id}",
            camera_index=0, mvtec_benchmark_upload=True, uploaded_image_diagnostic=False,
            operator_declared_part=False, screen_photo_diagnostic=False,
            diagnostic_no_part=False, vision_worker_socket=worker_socket,
            overlay_format="png")

    def fake_cold_child(self, child, score=0.9):
        def run_child(argv):
            if "--tensor" in argv and "--output" not in argv:
                tensor_path = Path(argv[argv.index("--tensor") + 1])
                with tensor_path.open("xb") as stream:
                    np.save(stream, np.zeros((1, 3, 256, 256), dtype=np.float32), allow_pickle=False)
            elif "--output" in argv:
                root = Path(argv[argv.index("--output") + 1]).parent
                fake_inference_outputs(root, root / "input.npy", 700, 700)
                report = json.loads((root / "inference.json").read_text())
                report["score"] = score
                report["candidate"] = "ANOMALY_CANDIDATE" if score > THRESHOLD else "NORMAL_CANDIDATE"
                (root / "inference.json").write_text(json.dumps(report))
            return 1.0
        child.side_effect = run_child

    def fake_worker(self, call, score=0.9, mutate=None):
        def serve_request(sock, req):
            root = Path(req["output"]).parent
            fake_inference_outputs(root, Path(req["tensor"]), 700, 700)
            report = json.loads((root / "inference.json").read_text())
            report["score"] = score
            report["candidate"] = "ANOMALY_CANDIDATE" if score > THRESHOLD else "NORMAL_CANDIDATE"
            if mutate:
                mutate(report)
            (root / "inference.json").write_text(json.dumps(report))
            return {}
        call.side_effect = serve_request

    def test_transport_failure_midrequest_falls_back_and_clears_partials(self):
        args = self.make_args(Path("/tmp/worker.sock"))
        def dying_worker(sock, req):
            root = Path(req["output"]).parent
            root.mkdir(parents=True, exist_ok=True)
            # partial artifacts from a worker that died mid-write
            (root / "inference.json").write_text('{"partial":')
            (root / "map_model.npy").write_bytes(b"\x93NUM")
            raise ConnectionResetError("worker died mid-request")
        with mock.patch.object(live, "worker_request", side_effect=dying_worker), \
             mock.patch.object(live, "child") as child:
            self.fake_cold_child(child)
            record = live.inspect(args)
        self.assertEqual(record["status"], "ANALYZED")
        self.assertEqual(record["timings"]["vision_mode"], "COLD_PROCESS_AFTER_WORKER_UNAVAILABLE")
        self.assertEqual(record["model"]["export_sha256_verified_at"], "THIS_INSPECTION")
        saved = json.loads((args.output_dir / "inference.json").read_text())
        self.assertEqual(saved["status"], "ONE_IMAGE_INFERENCE_VERIFIED")

    def test_request_rejection_never_falls_back(self):
        args = self.make_args(Path("/tmp/worker.sock"))
        with mock.patch.object(live, "worker_request",
                               side_effect=RuntimeError("Vision worker rejected the request: ValueError: bad tensor")), \
             mock.patch.object(live, "child") as child:
            record = live.inspect(args)
        self.assertEqual(record["status"], "ERROR")
        self.assertIn("rejected", record["error"]["message"])
        child.assert_not_called()
        self.assertNotIn("vision_mode", record["timings"])

    def test_wrong_tensor_hash_in_response_rejected(self):
        args = self.make_args(Path("/tmp/worker.sock"))
        with mock.patch.object(live, "worker_request") as call:
            self.fake_worker(call, mutate=lambda r: r.update(tensor_sha256="0" * 64))
            record = live.inspect(args)
        self.assertEqual(record["status"], "ERROR")
        self.assertIn("Inconsistent inference evidence", record["error"]["message"])

    def test_grading_boundary_equivalence_cold_vs_worker(self):
        scores = [THRESHOLD, THRESHOLD + 1e-9, THRESHOLD * 1.05 + 1e-9, THRESHOLD - 1e-9]
        outcomes = {}
        for mode in ("cold", "worker"):
            outcomes[mode] = []
            for index, score in enumerate(scores):
                args = self.make_args(Path("/tmp/worker.sock") if mode == "worker" else None,
                                      part_id=f"part-{mode}-{index}")
                if mode == "worker":
                    with mock.patch.object(live, "worker_request") as call:
                        self.fake_worker(call, score=score)
                        record = live.inspect(args)
                else:
                    with mock.patch.object(live, "child") as child:
                        self.fake_cold_child(child, score=score)
                        record = live.inspect(args)
                self.assertEqual(record["status"], "ANALYZED")
                outcomes[mode].append((record["score"], record["candidate"],
                                       record["decision"]["grade"]))
        self.assertEqual(outcomes["cold"], outcomes["worker"])
        # boundary semantics: strictly greater than threshold flags a candidate
        self.assertEqual(outcomes["worker"][0][1], "NORMAL_CANDIDATE")      # score == threshold
        self.assertEqual(outcomes["worker"][1][1], "ANOMALY_CANDIDATE")     # score just above
        self.assertEqual(outcomes["worker"][3][1], "NORMAL_CANDIDATE")      # score just below

    def test_duplicate_part_id_refused_before_inference(self):
        args = self.make_args(Path("/tmp/worker.sock"))
        with mock.patch.object(live, "worker_request") as call:
            self.fake_worker(call)
            first = live.inspect(args)
            self.assertEqual(first["status"], "ANALYZED")
            args2 = self.make_args(Path("/tmp/worker.sock"))
            args2.database = args.database  # same history
            call.reset_mock()
            with self.assertRaises(ValueError):
                live.inspect(args2)
            call.assert_not_called()

    def test_oversized_image_rejected(self):
        big = self.root / "big.png"
        big.write_bytes(b"\x89PNG" + b"0" * (33 * 1024 * 1024))
        args = self.make_args(Path("/tmp/worker.sock"))
        args.image = big
        with self.assertRaises(ValueError):
            live.inspect(args)

    def test_grayscale_and_truncated_images_rejected(self):
        gray = self.root / "gray.png"
        Image.new("L", (700, 700)).save(gray)
        args = self.make_args(Path("/tmp/worker.sock"))
        args.image = gray
        with self.assertRaises(ValueError):
            live.inspect(args)
        truncated = self.root / "truncated.png"
        truncated.write_bytes(self.image.read_bytes()[:64])
        args.image = truncated
        with self.assertRaises(ValueError):
            live.inspect(args)

    def test_wrong_size_benchmark_upload_rejected(self):
        small = self.root / "small.png"
        Image.new("RGB", (640, 480), "gray").save(small)
        args = self.make_args(Path("/tmp/worker.sock"))
        args.image = small
        with self.assertRaises(ValueError):
            live.inspect(args)


class StubWorker:
    def handle(self, raw):
        if raw.get("explode"):
            raise ValueError("stub rejection")
        return {"status": "ONE_IMAGE_INFERENCE_VERIFIED", "echo": raw.get("marker")}


class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def start_server(self, sock, idle=None):
        thread = threading.Thread(target=serve, args=(sock, StubWorker(), idle), daemon=True)
        thread.start()
        self.assertTrue(wait_socket(sock))
        return thread

    def test_mid_request_death_is_transport_error(self):
        sock = self.dir / "dying.sock"
        def killer():
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(sock))
            server.listen(1)
            connection, _ = server.accept()
            connection.close()  # die without answering
            server.close()
        thread = threading.Thread(target=killer, daemon=True)
        thread.start()
        self.assertTrue(wait_socket(sock))
        with self.assertRaises(ConnectionError):
            request(sock, {"tensor": "x", "threshold": 0.1})
        thread.join(timeout=5)

    def test_concurrent_clients_are_served_sequentially(self):
        sock = self.dir / "conc.sock"
        self.start_server(sock)
        results, errors = [], []
        def hit(marker):
            try:
                results.append(request(sock, {"tensor": "x", "threshold": 0.1, "marker": marker})["echo"])
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=hit, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertEqual(sorted(results), list(range(6)))

    def test_oversized_request_gets_error_and_server_survives(self):
        sock = self.dir / "big.sock"
        self.start_server(sock)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(sock))
            try:
                connection.sendall(b'{"pad": "' + b"x" * (2 * 1024 * 1024) + b'"}')
                connection.shutdown(socket.SHUT_WR)
                reply = connection.recv(65536)
            except (BrokenPipeError, ConnectionResetError):
                reply = b""
        if reply:  # server may answer ERROR before closing, or close outright
            self.assertEqual(json.loads(reply.split(b"\n", 1)[0])["status"], "ERROR")
        report = request(sock, {"tensor": "x", "threshold": 0.1, "marker": 7})
        self.assertEqual(report["echo"], 7)

    def test_restart_on_same_socket_path(self):
        sock = self.dir / "restart.sock"
        thread = self.start_server(sock, idle=0.3)
        thread.join(timeout=5)
        self.assertFalse(Path(sock).exists())
        self.start_server(sock)
        report = request(sock, {"tensor": "x", "threshold": 0.1, "marker": 9})
        self.assertEqual(report["echo"], 9)

    def test_stale_socket_file_reclaimed(self):
        sock = self.dir / "stale.sock"
        sock.write_bytes(b"")  # leftover from a killed process
        self.start_server(sock)
        report = request(sock, {"tensor": "x", "threshold": 0.1, "marker": 3})
        self.assertEqual(report["echo"], 3)

    def test_idle_timeout_then_transport_error(self):
        sock = self.dir / "idle.sock"
        thread = self.start_server(sock, idle=0.2)
        thread.join(timeout=5)
        with self.assertRaises(OSError):
            request(sock, {"tensor": "x", "threshold": 0.1}, connect_retries=3)


if __name__ == "__main__":
    unittest.main()
