"""The in-process preprocessing used with the worker must be byte-identical
to the reviewed vision_edge_preprocess.py subprocess it replaced."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from forgesight.vision_edge_live_once import preprocess_inprocess, sha256

REPO = Path(__file__).resolve().parent.parent


class PreprocessEquivalenceTests(unittest.TestCase):
    def test_inprocess_matches_subprocess_exactly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(11)
            for name, size in (("camera", (640, 480)), ("benchmark", (700, 700))):
                image_path = root / f"{name}.png"
                Image.fromarray(
                    rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(image_path)
                inprocess_tensor = root / f"{name}_inprocess.npy"
                subprocess_tensor = root / f"{name}_subprocess.npy"
                preprocess_inprocess(image_path, inprocess_tensor)
                result = subprocess.run(
                    [sys.executable, "-B", str(REPO / "forgesight" / "vision_edge_preprocess.py"),
                     "--image", str(image_path), "--tensor", str(subprocess_tensor)],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                reported = json.loads(result.stdout)
                self.assertEqual(sha256(inprocess_tensor), sha256(subprocess_tensor))
                self.assertEqual(reported["tensor_sha256"], sha256(inprocess_tensor))
                loaded = np.load(inprocess_tensor, allow_pickle=False)
                self.assertEqual(loaded.shape, (1, 3, 256, 256))
                self.assertEqual(loaded.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
