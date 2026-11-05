"""Training data/archive boundary tests; no simulated training successes."""

from copy import deepcopy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from forgesight.vision_assets import extract_verified_archive, safe_destination
from forgesight.vision_train import normal_rows, validate_recipe
from forgesight.vision_checkpoint import load_checkpoint, validate_checkpoint_metadata


class TrainingBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.document = {"counts": {"fit": 1, "validation_normal": 1}, "samples": [
            {"role": "fit", "label": "good", "source_split": "train", "mask": None,
             "path": "train/good/000.png"},
            {"role": "validation_normal", "label": "good", "source_split": "train", "mask": None,
             "path": "train/good/001.png"}]}

    def test_only_frozen_normal_roles_are_permitted(self):
        self.assertEqual(len(normal_rows(self.document, "fit")), 1)
        for role in ("final_holdout", "development_preview", "test"):
            with self.assertRaises(PermissionError):
                normal_rows(self.document, role)

    def test_abnormal_test_or_inconsistent_fit_rows_are_rejected(self):
        for key, value in (("label", "scratch"), ("source_split", "test"),
                           ("mask", {}), ("path", "test/good/000.png")):
            document = deepcopy(self.document)
            document["samples"][0][key] = value
            with self.assertRaisesRegex(ValueError, "adapter"):
                normal_rows(document, "fit")
        document = deepcopy(self.document)
        document["counts"]["fit"] = 2
        with self.assertRaisesRegex(ValueError, "count"):
            normal_rows(document, "fit")

    def test_recipe_cannot_enable_holdout_or_validation_updates(self):
        recipe = json.loads(Path("configs/efficientad_s_training.json").read_text())
        validate_recipe(recipe)
        for key, value in (("holdout_access", True), ("validation_pixel_reads_this_step", 1),
                           ("fit_role", "final_holdout")):
            changed = dict(recipe, **{key: value})
            with self.assertRaisesRegex(ValueError, "boundaries"):
                validate_recipe(changed)

    def test_archive_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../escape", "/absolute", "C:/outside", "root\\outside"):
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    safe_destination(Path(directory), name)

    def test_checkpoint_integrity_is_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pth"
            path.write_bytes(b"generic corruption guard bytes")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_checkpoint(path, "0" * 64)

    def test_checkpoint_metadata_cannot_change_architecture_or_data_boundary(self):
        profile = json.loads(Path("configs/vision_baseline.json").read_text())
        saved = {"format": "forgesight-efficientad-v1", "step": 1, "model_config": profile,
                 "training_config": {"holdout_access": False}}
        validate_checkpoint_metadata(saved)
        changed = deepcopy(saved)
        changed["model_config"]["model_size"] = "medium"
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_checkpoint_metadata(changed)
        changed = deepcopy(saved)
        changed["training_config"]["holdout_access"] = True
        with self.assertRaisesRegex(ValueError, "holdout"):
            validate_checkpoint_metadata(changed)

    def test_tar_link_is_rejected_before_file_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tgz"
            with tarfile.open(archive, "w:gz") as stream:
                member = tarfile.TarInfo("asset/link")
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside"
                stream.addfile(member)
            with self.assertRaisesRegex(ValueError, "links"):
                extract_verified_archive(archive, root / "out")

    def test_safe_archive_extracts_exact_bytes_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("asset/file.txt", b"archive test bytes")
            result = extract_verified_archive(archive, root / "out")
            self.assertEqual(result["files"], 1)
            self.assertEqual((root / "out/asset/file.txt").read_bytes(), b"archive test bytes")
            with self.assertRaises(FileExistsError):
                extract_verified_archive(archive, root / "out")


if __name__ == "__main__":
    unittest.main()
