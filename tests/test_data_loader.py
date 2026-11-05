"""Exercise loader failures and normal/anomaly semantics with synthetic pixels."""

import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from forgesight.data_loader import MetalNutDataset
from forgesight.data_manifest import encoded_manifest
from tests.fixtures import synthetic_manifest


class LoaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="forgesight-loader-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "data"
        self.document = synthetic_manifest()
        self.normal = next(row for row in self.document["samples"] if row["role"] == "fit")
        self.anomaly = next(row for row in self.document["samples"] if row["role"] == "development_preview")
        self.write_image(self.normal, "RGB", (10, 20, 30))
        self.write_image(self.anomaly, "RGB", (40, 50, 60))
        self.write_image(self.anomaly["mask"], "L", 255)
        self.manifest = Path(temporary.name) / "manifest.json"
        data = encoded_manifest(self.document)
        self.manifest.write_bytes(data)
        self.manifest.with_suffix(".sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")

    def write_image(self, record, mode, color):
        with Image.new(mode, (700, 700), color) as image, BytesIO() as encoded:
            image.save(encoded, format="PNG")
            data = encoded.getvalue()
            record["sha256"] = hashlib.sha256(data).hexdigest()
            record["pixel_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
        path = self.root / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def dataset(self, role="fit", **kwargs):
        return MetalNutDataset(self.root, self.manifest, role, **kwargs)

    def test_normal_gets_explicit_zero_mask_at_source_resolution(self):
        sample = self.dataset()[0]
        with sample.image, sample.mask:
            self.assertEqual(sample.image.size, (700, 700))
            self.assertEqual(sample.image.getpixel((0, 0)), (10, 20, 30))
            self.assertIsNone(sample.mask.getbbox())
            self.assertEqual(sample.mask_origin, "implicit_normal")
            self.assertEqual(sample.ground_truth_type, "good")

    def test_anomaly_loads_its_dataset_mask(self):
        sample = self.dataset("development_preview")[0]
        with sample.image, sample.mask:
            self.assertEqual(sample.mask.getpixel((0, 0)), 255)
            self.assertEqual(sample.mask_origin, "dataset_annotation")
            self.assertEqual(sample.role, "development_preview")

    def test_final_holdout_requires_explicit_opt_in(self):
        with self.assertRaises(PermissionError):
            self.dataset("final_holdout")
        self.assertEqual(len(self.dataset("final_holdout", allow_final_holdout=True)), 111)

    def test_unknown_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.dataset("everything")

    def test_changed_image_is_rejected(self):
        (self.root / self.normal["path"]).write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "File hash mismatch"):
            self.dataset()[0]

    def test_missing_anomaly_mask_is_not_treated_as_normal(self):
        (self.root / self.anomaly["mask"]["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset("development_preview")[0]

    def test_changed_mask_is_rejected(self):
        (self.root / self.anomaly["mask"]["path"]).write_bytes(b"wrong mask")
        with self.assertRaisesRegex(ValueError, "File hash mismatch"):
            self.dataset("development_preview")[0]

    def test_manifest_tampering_is_rejected(self):
        self.manifest.write_bytes(self.manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "Manifest checksum"):
            self.dataset()

    def test_out_of_range_index_fails_explicitly(self):
        with self.assertRaises(IndexError):
            self.dataset()[176]


if __name__ == "__main__":
    unittest.main()
