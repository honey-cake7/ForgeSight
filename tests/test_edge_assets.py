"""Synthetic file-integrity tests, without model loading or generated text."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from forgesight.edge_assets import inspect_tokenizer, verify_assets


class AssetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="forgesight-assets-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        payloads = {"original/consolidated.00.pth": b"integrity-test-payload",
                    "original/params.json": b'{"fixture": true}',
                    "original/tokenizer.model": b"YQ== 0\nYg== 1\n"}
        self.reference = {"model_id": "synthetic-integrity-fixture", "model_revision": "test",
                          "params": {"fixture": True}, "tokenizer_base_tokens": 2, "files": {}}
        for relative, data in payloads.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.reference["files"][relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def test_valid_fingerprints_still_do_not_claim_inference(self):
        report = verify_assets(self.root, self.reference)
        self.assertEqual(report["status"], "ASSETS_VERIFIED_NOT_EXPORTED")
        self.assertEqual(report["inference_status"], "NOT_RUN")

    def test_same_size_corruption_is_rejected(self):
        path = self.root / "original/consolidated.00.pth"
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            verify_assets(self.root, self.reference)

    def test_missing_file_is_rejected(self):
        (self.root / "original/params.json").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_assets(self.root, self.reference)

    def test_asset_path_traversal_is_rejected(self):
        self.reference["files"] = {"../outside.pth": {"bytes": 0, "sha256": "unused"}}
        with self.assertRaisesRegex(ValueError, "Invalid asset-relative"):
            verify_assets(self.root, self.reference)

    def test_duplicate_tokenizer_ranks_are_rejected(self):
        path = self.root / "original/tokenizer.model"
        path.write_bytes(b"YQ== 0\nYg== 0\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inspect_tokenizer(path, 2)

    def test_missing_tokenizer_ranks_are_rejected(self):
        path = self.root / "original/tokenizer.model"
        path.write_bytes(b"YQ== 0\nYg== 2\n")
        with self.assertRaisesRegex(ValueError, "count/ranks"):
            inspect_tokenizer(path, 2)


if __name__ == "__main__":
    unittest.main()
