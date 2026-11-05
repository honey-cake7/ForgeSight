"""Synthetic archive integrity tests; these are not dataset or model evidence."""

from io import BytesIO
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from PIL import Image

from scripts.audit_metal_nut_archive import audit_archive, canonical_path, extract_audited_archive


def png_bytes(mode, color):
    with Image.new(mode, (700, 700), color) as image, BytesIO() as stream:
        image.save(stream, format="PNG")
        return stream.getvalue()


class ArchiveAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Repeated pixels deliberately keep this structural fixture small.
        # Duplicate-content analysis belongs to the later split-manifest step.
        rgb = png_bytes("RGB", (10, 20, 30))
        mask = png_bytes("L", 255)
        cls.entries = {f"metal_nut/train/good/{i:03}.png": rgb for i in range(220)}
        cls.entries.update({f"metal_nut/test/good/{i:03}.png": rgb for i in range(22)})
        for label, count in {"bent": 25, "color": 22, "flip": 23, "scratch": 23}.items():
            for index in range(count):
                cls.entries[f"metal_nut/test/{label}/{index:03}.png"] = rgb
                cls.entries[f"metal_nut/ground_truth/{label}/{index:03}_mask.png"] = mask

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="forgesight-audit-test-")
        self.addCleanup(temporary.cleanup)
        self.archive = Path(temporary.name) / "synthetic.tar"

    def write_archive(self, entries):
        with tarfile.open(self.archive, "w") as archive:
            for name, data in entries:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, BytesIO(data))

    def test_complete_synthetic_category_is_audited_without_extraction(self):
        self.write_archive(self.entries.items())
        report = audit_archive(self.archive)
        self.assertEqual(report["counts"]["train/good"], 220)
        self.assertEqual(report["counts"]["test/good"], 22)
        self.assertEqual(len(report["files"]), 428)
        self.assertEqual(report["mask_values"], [255])
        self.assertEqual(len(report["archive_sha256"]), 64)
        self.assertEqual(list(self.archive.parent.iterdir()), [self.archive])

    def test_unsafe_paths_are_rejected(self):
        for name in ["../escape.png", "/tmp/escape.png", "C:/escape.png",
                     "metal_nut/../escape.png", "metal_nut\\test\\scratch\\000.png"]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe"):
                canonical_path(name)

    def test_root_and_category_prefixed_paths_are_supported(self):
        self.assertEqual(canonical_path("./metal_nut/test/good/000.png"), "test/good/000.png")
        self.assertEqual(canonical_path("train/good/000.png"), "train/good/000.png")

    def test_archive_links_are_rejected(self):
        for link_type in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
            with self.subTest(link_type=link_type):
                with tarfile.open(self.archive, "w") as archive:
                    info = tarfile.TarInfo("metal_nut/train/good/000.png")
                    info.type = link_type
                    info.linkname = "/tmp/outside.png"
                    archive.addfile(info)
                with self.assertRaisesRegex(ValueError, "links or special"):
                    audit_archive(self.archive)

    def test_complete_synthetic_zip_is_supported(self):
        self.archive = self.archive.with_suffix(".zip")
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, data in self.entries.items():
                archive.writestr(name, data)
        report = audit_archive(self.archive)
        self.assertEqual(report["archive_format"], "zip")
        self.assertEqual(len(report["files"]), 428)

    def test_zip_symlink_is_rejected(self):
        self.archive = self.archive.with_suffix(".zip")
        with zipfile.ZipFile(self.archive, "w") as archive:
            info = zipfile.ZipInfo("metal_nut/train/good/000.png")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            archive.writestr(info, "/tmp/outside.png")
        with self.assertRaisesRegex(ValueError, "links or special"):
            audit_archive(self.archive)

    def test_zip_path_traversal_is_rejected(self):
        self.archive = self.archive.with_suffix(".zip")
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("../outside.png", b"unused")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            audit_archive(self.archive)

    def test_extraction_verifies_files_and_refuses_existing_destination(self):
        self.archive = self.archive.with_suffix(".zip")
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, data in self.entries.items():
                archive.writestr(name, data)
        report = audit_archive(self.archive)
        destination = self.archive.parent / "dataset"
        result = extract_audited_archive(self.archive, report, destination)
        self.assertEqual(result["files_verified"], 428)
        self.assertEqual((destination / "train/good/000.png").read_bytes(),
                         self.entries["metal_nut/train/good/000.png"])
        with self.assertRaises(FileExistsError):
            extract_audited_archive(self.archive, report, destination)

    def test_changed_archive_is_rejected_before_extraction(self):
        self.write_archive(self.entries.items())
        report = audit_archive(self.archive)
        with self.archive.open("ab") as stream:
            stream.write(b"changed after audit")
        destination = self.archive.parent / "dataset"
        with self.assertRaisesRegex(ValueError, "hash changed"):
            extract_audited_archive(self.archive, report, destination)
        self.assertFalse(destination.exists())

    def test_duplicate_destinations_are_rejected(self):
        name, data = next(iter(self.entries.items()))
        self.write_archive([(name, data), (name, data)])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_archive(self.archive)

    def test_missing_mask_is_rejected(self):
        entries = dict(self.entries)
        del entries["metal_nut/ground_truth/scratch/000_mask.png"]
        self.write_archive(entries.items())
        with self.assertRaisesRegex(ValueError, "Missing anomaly mask"):
            audit_archive(self.archive)

    def test_corrupt_png_is_rejected(self):
        self.write_archive([("metal_nut/train/good/000.png", b"not a PNG")])
        with self.assertRaises(OSError):
            audit_archive(self.archive)

    def test_empty_anomaly_mask_is_rejected(self):
        self.write_archive([("metal_nut/ground_truth/scratch/000_mask.png", png_bytes("L", 0))])
        with self.assertRaisesRegex(ValueError, "empty mask"):
            audit_archive(self.archive)

    def test_nonbinary_mask_is_rejected(self):
        with Image.new("L", (700, 700), 0) as image, BytesIO() as stream:
            image.putpixel((0, 0), 128)
            image.save(stream, format="PNG")
            data = stream.getvalue()
        self.write_archive([("metal_nut/ground_truth/scratch/000_mask.png", data)])
        with self.assertRaisesRegex(ValueError, "not binary"):
            audit_archive(self.archive)


if __name__ == "__main__":
    unittest.main()
