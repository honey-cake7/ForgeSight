"""Reject invalid artifact claims before importing a runtime; no simulated inference."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from forgesight.edge_artifact import check_fingerprint, inspect_artifact


class ArtifactTests(unittest.TestCase):
    def test_same_size_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pte"
            original = b"generic bytes for a corruption guard"
            reference = {"artifact": {"bytes": len(original),
                                      "sha256": hashlib.sha256(original).hexdigest()}}
            path.write_bytes(b"x" + original[1:])
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                check_fingerprint(path, reference)

    def test_matching_hash_of_non_pte_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pte"
            data = b"generic non-model content"
            path.write_bytes(data)
            reference = {"artifact": {"bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest()}}
            with self.assertRaisesRegex(ValueError, "schema identifier"):
                check_fingerprint(path, reference)

    def test_wrong_extension_and_absent_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, ".pte"):
                check_fingerprint(Path(directory) / "not-a-model.txt", {})
            with self.assertRaises(FileNotFoundError):
                check_fingerprint(Path(directory) / "missing.pte", {})

    def test_reference_revision_mismatch_is_rejected_before_runtime_import(self):
        reference = {"schema_version": 1, "model_id": "same", "model_revision": "old",
                     "executorch_commit": "runtime"}
        profile = {"model_id": "same", "model_revision": "new", "executorch_commit": "runtime"}
        with self.assertRaisesRegex(ValueError, "differs"):
            inspect_artifact(Path("absent.pte"), reference, profile)


if __name__ == "__main__":
    unittest.main()
