"""Check overlay pixels and enforce development-only rendering with synthetic data."""

import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from forgesight.data_manifest import encoded_manifest
from forgesight.data_preview import annotation_overlay, create_preview
from tests.fixtures import synthetic_manifest


class OverlayTests(unittest.TestCase):
    def test_zero_mask_preserves_every_pixel(self):
        with Image.new("RGB", (3, 2), (10, 20, 30)) as image, Image.new("L", (3, 2), 0) as mask:
            with annotation_overlay(image, mask) as result:
                self.assertEqual(result.tobytes(), image.tobytes())

    def test_only_annotated_pixels_change(self):
        with Image.new("RGB", (3, 2), (10, 20, 30)) as image, Image.new("L", (3, 2), 0) as mask:
            mask.putpixel((1, 0), 255)
            with annotation_overlay(image, mask) as result:
                for y in range(2):
                    for x in range(3):
                        if (x, y) == (1, 0):
                            self.assertNotEqual(result.getpixel((x, y)), image.getpixel((x, y)))
                        else:
                            self.assertEqual(result.getpixel((x, y)), image.getpixel((x, y)))

    def test_misaligned_and_nonbinary_masks_fail(self):
        with Image.new("RGB", (3, 2)) as image:
            for size, color in [((2, 2), 0), ((3, 2), 128)]:
                with Image.new("L", size, color) as mask, self.assertRaises(ValueError):
                    annotation_overlay(image, mask)


class PreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="forgesight-preview-test-")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.root = base / "data"
        self.output = base / "preview"
        document = synthetic_manifest()
        selected = [next(row for row in document["samples"] if row["role"] == "fit")]
        selected += [row for row in document["samples"] if row["role"] == "development_preview"]
        for index, row in enumerate(selected):
            self.write_png(row, "RGB", (20 * index, 30, 50))
            if row["mask"] is not None:
                self.write_png(row["mask"], "L", 255)
        # All unselected image files, including the entire holdout, are absent.
        # Rendering can succeed only if it loads the five permitted samples.
        self.manifest = base / "manifest.json"
        data = encoded_manifest(document)
        self.manifest.write_bytes(data)
        self.manifest.with_suffix(".sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")

    def write_png(self, record, mode, color):
        with Image.new(mode, (700, 700), color) as image, BytesIO() as stream:
            image.save(stream, format="PNG")
            data = stream.getvalue()
            record["sha256"] = hashlib.sha256(data).hexdigest()
            record["pixel_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
        path = self.root / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def test_only_selected_samples_render_and_keep_ground_truth_provenance(self):
        report = create_preview(self.root, self.manifest, self.output)
        self.assertEqual(len(report["samples"]), 5)
        self.assertEqual(report["holdout_images_displayed"], 0)
        self.assertIsNone(report["prediction_model"])
        self.assertEqual({row["role"] for row in report["samples"]}, {"fit", "development_preview"})
        self.assertEqual(report["samples"][0]["annotation_area_pixels"], 0)
        self.assertTrue(all(row["annotation_area_pixels"] == 490000 for row in report["samples"][1:]))
        with Image.open(self.output / "contact_sheet.png") as sheet:
            sheet.verify()
        self.assertTrue((self.output / "preview.json").is_file())

    def test_existing_preview_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("existing work")
        with self.assertRaises(FileExistsError):
            create_preview(self.root, self.manifest, self.output)
        self.assertEqual(sentinel.read_text(), "existing work")


if __name__ == "__main__":
    unittest.main()
