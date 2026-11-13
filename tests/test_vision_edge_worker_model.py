"""End-to-end Worker tests with a real TorchScript stand-in model.

Every other worker test stubs the model away; this file loads an actual
scripted module through Worker (torch.jit.load + inference_mode forward) and
pins repeatability, bad-input rejection and invalid-output rejection. The
stand-in only matches the frozen export's interface (anomaly map + score);
it says nothing about real EfficientAD accuracy or speed.
"""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

from forgesight.vision_edge_worker import Worker


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipIf(torch is None, "Torch is not installed in this environment")
class WorkerModelTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

        class StandIn(nn.Module):
            def forward(self, x):
                anomaly_map = x.mean(dim=1, keepdim=True)
                return anomaly_map, anomaly_map.amax(dim=(2, 3)).squeeze()

        self.model_path = self.root / "standin.ts"
        torch.jit.script(StandIn()).save(str(self.model_path))
        self.worker = Worker(self.model_path, sha256(self.model_path), threads=2)
        rng = np.random.default_rng(7)
        self.tensor_path = self.root / "input.npy"
        with self.tensor_path.open("wb") as stream:
            np.save(stream, rng.random((1, 3, 256, 256), dtype=np.float32), allow_pickle=False)
        self.tensor_sha = sha256(self.tensor_path)

    def tearDown(self):
        self._tmp.cleanup()

    def request(self, name, **overrides):
        req = {"tensor": str(self.tensor_path), "tensor_sha256": self.tensor_sha,
               "threshold": 0.5, "output": str(self.root / f"{name}.json")}
        req.update(overrides)
        return req

    def test_repeatable_score_and_identical_map_files(self):
        first = self.worker.handle(self.request(
            "a", map_output=str(self.root / "a_map.npy"),
            source_map_output=str(self.root / "a_src.npy"), source_size=[480, 640]))
        second = self.worker.handle(self.request(
            "b", map_output=str(self.root / "b_map.npy"),
            source_map_output=str(self.root / "b_src.npy"), source_size=[480, 640]))
        self.assertEqual(first["score"], second["score"])
        self.assertEqual(first["map_sha256"], second["map_sha256"])
        self.assertEqual(first["source_map_sha256"], second["source_map_sha256"])
        self.assertEqual(first["candidate"], "ANOMALY_CANDIDATE")
        self.assertEqual(first["worker_mode"], "PERSISTENT_PROCESS")

    def test_threshold_boundary_candidate(self):
        score = self.worker.handle(self.request("c"))["score"]
        above = self.worker.handle(self.request("d", threshold=score - 1e-7))
        below = self.worker.handle(self.request("e", threshold=score))
        self.assertEqual(above["candidate"], "ANOMALY_CANDIDATE")
        self.assertEqual(below["candidate"], "NORMAL_CANDIDATE")

    def test_bad_tensors_rejected(self):
        cases = {
            "nan": np.full((1, 3, 256, 256), np.nan, dtype=np.float32),
            "wrong_shape": np.zeros((1, 3, 128, 128), dtype=np.float32),
            "wrong_dtype": np.zeros((1, 3, 256, 256), dtype=np.float64),
            "batched": np.zeros((2, 3, 256, 256), dtype=np.float32),
        }
        for name, array in cases.items():
            path = self.root / f"bad_{name}.npy"
            with path.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            with self.assertRaises(ValueError, msg=name):
                self.worker.handle(self.request(
                    f"bad_{name}", tensor=str(path), tensor_sha256=sha256(path)))

    def test_tampered_tensor_hash_rejected(self):
        with self.assertRaises(ValueError):
            self.worker.handle(self.request("tampered", tensor_sha256="0" * 64))

    def test_invalid_model_outputs_rejected(self):
        class NanModel(nn.Module):
            def forward(self, x):
                return (torch.full((1, 1, 256, 256), float("nan")),
                        torch.tensor(float("nan")))

        class WrongShape(nn.Module):
            def forward(self, x):
                return x.mean(dim=1, keepdim=True)[:, :, ::2, ::2], torch.tensor(0.5)

        for name, module in (("nan", NanModel), ("wrong_shape", WrongShape)):
            path = self.root / f"{name}.ts"
            torch.jit.script(module()).save(str(path))
            worker = Worker(path, sha256(path))
            with self.assertRaises(ValueError, msg=name):
                worker.handle(self.request(f"model_{name}"))

    def test_model_hash_mismatch_refused_at_boot(self):
        with self.assertRaises(ValueError):
            Worker(self.model_path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
