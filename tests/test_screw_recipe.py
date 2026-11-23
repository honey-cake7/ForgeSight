"""Screw split and integrity boundaries, independent of model training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from forgesight.screw_recipe import build_manifest, validate_manifest
from forgesight.vision_recipe import read_manifest


class ScrewRecipeTests(unittest.TestCase):
    def fixture(self, root):
        for folder, count in (("train/good", 10), ("test/good", 8),
                              ("test/thread_side", 10)):
            for i in range(count):
                path = root / folder / f"{i:03d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                value = i + (30 if folder.startswith("test/good") else
                             60 if "thread_side" in folder else 0)
                Image.new("RGB", (8, 8), (value, value + 1, value + 2)).save(path)
                if "thread_side" in folder:
                    mask = root / "ground_truth/thread_side" / f"{i:03d}_mask.png"
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L", (8, 8), 255).save(mask)

    def test_deterministic_disjoint_roles_and_hash_pinned_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            manifest = build_manifest(root)
            self.assertEqual(manifest, build_manifest(root))
            self.assertEqual(manifest["counts"], {"fit": 8, "validation_normal": 2,
                "threshold_normal": 4, "classifier_fit": 6,
                "classifier_validation": 2, "final_holdout": 6})
            for row in manifest["samples"]:
                if row["role"] in {"fit", "validation_normal"}:
                    self.assertTrue(row["path"].startswith("train/good/"))
                if row["role"] == "threshold_normal":
                    self.assertTrue(row["path"].startswith("test/good/"))
            path = root / "manifest.json"
            data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            path.write_bytes(data)
            path.with_suffix(".sha256").write_text(hashlib.sha256(data).hexdigest())
            self.assertEqual(read_manifest(path), manifest)
            path.write_bytes(data + b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_manifest(path)

    def test_missing_or_orphan_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "ground_truth/thread_side/000_mask.png").rename(
                root / "ground_truth/thread_side/orphan_mask.png")
            with self.assertRaisesRegex(ValueError, "mask"):
                build_manifest(root)

    def test_duplicate_pixels_and_changed_role_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            manifest = build_manifest(root)
            changed = deepcopy(manifest)
            changed["samples"][0]["role"] = "fit"
            with self.assertRaises(ValueError):
                validate_manifest(changed)
            (root / "test/good/000.png").write_bytes((root / "train/good/000.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                build_manifest(root)

    def test_nonbinary_or_misaligned_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            Image.new("L", (8, 8), 127).save(root / "ground_truth/thread_side/000_mask.png")
            with self.assertRaisesRegex(ValueError, "mask"):
                build_manifest(root)

    def test_native_grayscale_is_preserved_in_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root / "train/good/000.png"
            with Image.open(path) as image:
                image.convert("L").save(path)
            manifest = build_manifest(root)
            row = next(r for r in manifest["samples"] if r["path"] == "train/good/000.png")
            self.assertEqual(row["mode"], "L")

    def test_mask_geometry_and_unsupported_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            manifest = build_manifest(root)
            changed = deepcopy(manifest)
            row = next(r for r in changed["samples"] if r["mask"] is not None)
            row["mask"]["size"] = [9, 8]
            with self.assertRaisesRegex(ValueError, "mask"):
                validate_manifest(changed)
            changed = deepcopy(manifest)
            changed["samples"][0]["label"] = "bent_tip"
            with self.assertRaises(ValueError):
                validate_manifest(changed)

    def test_metal_nut_reader_retains_original_validation(self):
        document = read_manifest(Path("manifests/metal_nut_v1.json"))
        self.assertEqual(document["category"], "metal_nut")
        self.assertEqual(document["counts"]["fit"], 176)


if __name__ == "__main__":
    unittest.main()
