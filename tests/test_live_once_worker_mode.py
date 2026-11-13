"""Worker-mode and fallback behavior of vision_edge_live_once, without Torch.

The model process is replaced by mocks; these tests pin the evidence rules:
the frozen model bytes are re-hashed per inspection on the cold path only,
a dead worker falls back to the cold process with honest markers, and a
worker response with mismatched evidence is rejected.
"""

import argparse
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
from PIL import Image

from forgesight import vision_edge_live_once as live

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "configs" / "forgegrade_benchmark_v1.json"
POLICY_DATA = json.loads(POLICY.read_text())
THRESHOLD = POLICY_DATA["image_threshold"]


def model_file_sha(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def fake_inference_outputs(root, tensor, height, width, model_sha=None,
                           tensor_sha=None, threshold=THRESHOLD):
    if model_sha is None:
        model_sha = live.MODEL_SHA256
    """Write the artifacts a real vision child/worker would, with valid shapes."""
    if tensor_sha is None:
        tensor_sha = live.sha256(tensor)
    report = {"status": "ONE_IMAGE_INFERENCE_VERIFIED", "model_sha256": model_sha,
              "tensor_sha256": tensor_sha, "threshold": threshold,
              "score": 0.9, "candidate": "ANOMALY_CANDIDATE",
              "model_forward_ms": 1.0, "process_peak_rss_kib": 1000,
              "map_min": 0.0, "map_max": 1.0, "map_shape": [1, 1, 256, 256]}
    (root / "inference.json").write_text(json.dumps(report))
    np.save(root / "map_model.npy", np.zeros((256, 256), dtype=np.float32))
    source = np.zeros((height, width), dtype=np.float32)
    source[100:200, 100:200] = 1.0  # one strong region for geometry
    np.save(root / "map_source.npy", source)
    return report


class LiveOnceWorkerModeTests(unittest.TestCase):
    _seq = 0

    def setUp(self):
        LiveOnceWorkerModeTests._seq += 1
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.image = self.root / "image.png"
        Image.new("RGB", (700, 700), "gray").save(self.image)
        self.model = self.root / "model.ts"
        self.model.write_bytes(b"stand-in model bytes")
        self._sha_patch = mock.patch.object(live, "MODEL_SHA256", model_file_sha(self.model))
        self._sha_patch.start()

    def tearDown(self):
        self._sha_patch.stop()
        self._tmp.cleanup()

    def make_args(self, worker_socket=None, overlay_format="png"):
        return argparse.Namespace(
            image=self.image, part_id="part-t1", vision_python=Path("python"),
            model=self.model, model_sha256=live.MODEL_SHA256, grade_policy=POLICY,
            database=self.root / f"db-{LiveOnceWorkerModeTests._seq}.sqlite3", output_dir=self.root / f"out-{LiveOnceWorkerModeTests._seq}",
            camera_index=0, mvtec_benchmark_upload=True, uploaded_image_diagnostic=False,
            operator_declared_part=False, screen_photo_diagnostic=False,
            diagnostic_no_part=False, vision_worker_socket=worker_socket,
            overlay_format=overlay_format)

    def test_cold_path_rehashes_model_bytes(self):
        args = self.make_args(worker_socket=None)
        with mock.patch.object(live, "child") as child:
            def run_child(argv):
                if "--tensor" in argv and "--output" not in argv:  # preprocess child
                    tensor_path = Path(argv[argv.index("--tensor") + 1])
                    with tensor_path.open("xb") as stream:
                        np.save(stream, np.zeros((1, 3, 256, 256), dtype=np.float32), allow_pickle=False)
                elif "--output" in argv:  # vision child
                    root = Path(argv[argv.index("--output") + 1]).parent
                    fake_inference_outputs(root, root / "input.npy", 700, 700)
                return 1.0
            child.side_effect = run_child
            record = live.inspect(args)
        self.assertEqual(record["status"], "ANALYZED")
        self.assertEqual(record["model"]["export_sha256_verified_at"], "THIS_INSPECTION")

    def test_cold_path_rejects_wrong_model_bytes(self):
        self.model.write_bytes(b"tampered model bytes")
        args = self.make_args(worker_socket=None)
        with self.assertRaises(ValueError):
            live.inspect(args)  # model bytes no longer match the frozen hash

    def test_worker_mode_skips_model_file_rehash(self):
        args = self.make_args(worker_socket=Path("/tmp/worker.sock"))
        self.model.unlink()  # worker mode must never read the model file
        with mock.patch.object(live, "worker_request") as call:
            def serve_request(sock, req):
                root = Path(req["output"]).parent
                fake_inference_outputs(root, Path(req["tensor"]), 700, 700)
                return {}
            call.side_effect = serve_request
            record = live.inspect(args)
        self.assertEqual(record["status"], "ANALYZED")
        self.assertEqual(record["model"]["export_sha256_verified_at"], "WORKER_START_FROZEN_MODEL")
        self.assertNotIn("vision_mode", record["timings"])

    def test_dead_worker_falls_back_to_cold_process(self):
        args = self.make_args(worker_socket=Path("/tmp/worker.sock"))
        with mock.patch.object(live, "worker_request",
                               side_effect=ConnectionRefusedError("gone")), \
             mock.patch.object(live, "child") as child:
            def run_child(argv):
                if "--tensor" in argv and "--output" not in argv:
                    tensor_path = Path(argv[argv.index("--tensor") + 1])
                    with tensor_path.open("xb") as stream:
                        np.save(stream, np.zeros((1, 3, 256, 256), dtype=np.float32), allow_pickle=False)
                elif "--output" in argv:
                    root = Path(argv[argv.index("--output") + 1]).parent
                    fake_inference_outputs(root, root / "input.npy", 700, 700)
                return 1.0
            child.side_effect = run_child
            record = live.inspect(args)
        self.assertEqual(record["status"], "ANALYZED")
        self.assertEqual(record["timings"]["vision_mode"], "COLD_PROCESS_AFTER_WORKER_UNAVAILABLE")
        # The cold child re-verified the model bytes, so the record must not claim boot-time verification.
        self.assertEqual(record["model"]["export_sha256_verified_at"], "THIS_INSPECTION")

    def test_tampered_worker_evidence_rejected(self):
        args = self.make_args(worker_socket=Path("/tmp/worker.sock"))
        with mock.patch.object(live, "worker_request") as call:
            def serve_request(sock, req):
                root = Path(req["output"]).parent
                fake_inference_outputs(root, Path(req["tensor"]), 700, 700,
                                       model_sha="0" * 64)  # wrong model identity
                return {}
            call.side_effect = serve_request
            record = live.inspect(args)
        self.assertEqual(record["status"], "ERROR")
        self.assertIn("Inconsistent inference evidence", record["error"]["message"])

    def test_jpeg_overlay_artifact(self):
        args = self.make_args(worker_socket=Path("/tmp/worker.sock"), overlay_format="jpeg")
        with mock.patch.object(live, "worker_request") as call:
            def serve_request(sock, req):
                root = Path(req["output"]).parent
                fake_inference_outputs(root, Path(req["tensor"]), 700, 700)
                return {}
            call.side_effect = serve_request
            record = live.inspect(args)
        self.assertEqual(record["status"], "ANALYZED")
        overlay = Path(record["artifacts"]["response_overlay"])
        self.assertEqual(overlay.suffix, ".jpg")
        self.assertTrue(overlay.is_file())
        self.assertEqual(record["artifacts"]["response_overlay_sha256"], live.sha256(overlay))

    def test_default_overlay_stays_png(self):
        args = self.make_args(worker_socket=Path("/tmp/worker.sock"))
        with mock.patch.object(live, "worker_request") as call:
            def serve_request(sock, req):
                root = Path(req["output"]).parent
                fake_inference_outputs(root, Path(req["tensor"]), 700, 700)
                return {}
            call.side_effect = serve_request
            record = live.inspect(args)
        self.assertTrue(record["artifacts"]["response_overlay"].endswith(".png"))


class InlineInspectionTests(unittest.TestCase):
    """jetson_live_ui._inspect_inline publishes exactly what inspect() recorded."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_inline_publishes_record(self):
        from threading import Lock
        from forgesight import jetson_live_ui as ui
        
        service = ui.LiveService.__new__(ui.LiveService)
        service.args = argparse.Namespace(
            vision_python=Path("python"), model=self.root / "model.ts",
            grade_policy=POLICY, database=self.root / "db.sqlite3",
            camera_index=0, vision_worker_socket=Path("/tmp/worker.sock"),
            overlay_format="png")
        class StubStore:
            def get_inspection(self, identifier):
                return record
        service.store = StubStore()
        service.lock = Lock()
        service.latest = service.latest_root = service.latest_source = None
        image = self.root / "image.png"
        Image.new("RGB", (700, 700), "gray").save(image)

        record = {"schema_version": 1, "inspection_id": "123e4567-e89b-42d3-a456-426614174000", "part_id": "part-t1",
                  "timestamp_utc": "2026-11-14T00:00:00+00:00", "status": "ANALYZED",
                  "score": 0.9, "candidate": "ANOMALY_CANDIDATE",
                  "source": {"mode": "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD",
                             "size_px": [700, 700]},
                  "model": {"image_threshold": 0.2277658},
                  "geometry": None, "decision": {"grade": "REVIEW", "reason": "r",
                                                 "operator_action": "check"},
                  "timings": {}, "error": None, "artifacts": {}}

        def fake_inspect(args):
            root = args.output_dir
            root.mkdir(parents=True)
            with (root / "record.json").open("x") as stream:
                json.dump(record, stream)
            return record

        with mock.patch("forgesight.vision_edge_live_once.inspect", side_effect=fake_inspect):
            service._inspect_inline(image, "part-t1", self.root / "run",
                                    {"mvtec_benchmark_upload": True,
                                     "uploaded_image_diagnostic": False,
                                     "operator_declared_part": False,
                                     "screen_photo_diagnostic": False,
                                     "diagnostic_no_part": False})
        self.assertEqual(service.latest["status"], "ANALYZED")
        self.assertEqual(service.latest["inspection_id"], "123e4567-e89b-42d3-a456-426614174000")
        self.assertEqual(service.latest["grade"], "REVIEW")


if __name__ == "__main__":
    unittest.main()
