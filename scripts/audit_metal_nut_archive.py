"""Audit a category ZIP/tar; optionally extract verified files without previews."""

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import zipfile

from PIL import Image


DEFECTS = frozenset({"bent", "color", "flip", "scratch"})
MAX_FILE_BYTES = 16 * 1024 * 1024


def archive_entries(archive):
    """Expose names, sizes, and entry types consistently for ZIP and tar."""
    if isinstance(archive, zipfile.ZipFile):
        for member in archive.infolist():
            mode = stat.S_IFMT(member.external_attr >> 16)
            regular = mode in {0, stat.S_IFREG} and not member.flag_bits & 1
            yield member, member.filename, member.file_size, member.is_dir(), regular
    else:
        for member in archive:
            yield member, member.name, member.size, member.isdir(), member.isfile()


def canonical_path(name: str) -> str:
    """Accept category-root or metal_nut-prefixed paths; reject unsafe names."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    parts = path.parts
    if parts and parts[0] == "metal_nut":
        parts = parts[1:]
    return PurePosixPath(*parts).as_posix() if parts else "."


def file_kind(path: str) -> str:
    parts = PurePosixPath(path).parts
    if len(parts) == 1 and Path(path).suffix.lower() in {".txt", ".md"}:
        return "notice"
    if len(parts) != 3 or not parts[2].endswith(".png"):
        raise ValueError(f"Unexpected category file: {path}")
    split, label, filename = parts
    if split == "train" and label == "good":
        return "image"
    if split == "test" and label in DEFECTS | {"good"}:
        return "image"
    if split == "ground_truth" and label in DEFECTS and filename.endswith("_mask.png"):
        return "mask"
    raise ValueError(f"Unexpected category file: {path}")


def audit_archive(archive_path: Path) -> dict:
    """Fully decode expected PNGs and verify counts/pairing, without model use."""
    before = archive_path.stat()
    with archive_path.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    inventory = {}
    counts = Counter()
    mask_values = set()
    is_zip = zipfile.is_zipfile(archive_path)
    container = zipfile.ZipFile(archive_path) if is_zip else tarfile.open(archive_path, "r:*")
    with container as archive:
        members = []
        seen = set()
        for member, name, size, directory, regular in archive_entries(archive):
            if len(seen) >= 1000:
                raise ValueError("Expected a single category archive, not a full dataset.")
            path = canonical_path(name)
            if path in seen:
                raise ValueError(f"Duplicate archive destination: {path}")
            seen.add(path)
            if directory:
                continue
            if not regular:
                raise ValueError(f"Archive links or special/encrypted files are not accepted: {name}")
            if not 0 <= size <= MAX_FILE_BYTES:
                raise ValueError(f"Archive file exceeds the per-file limit: {path}")
            members.append((member, name, size, path, file_kind(path)))

        for member, name, size, path, kind in members:
            reader = archive.open(member) if is_zip else archive.extractfile(member)
            with reader as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
            if len(data) != size:
                raise ValueError(f"Truncated archive member: {path}")
            record = {
                "archive_path": name,
                "kind": kind,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            if kind != "notice":
                with Image.open(BytesIO(data)) as image:
                    if image.format != "PNG" or image.size != (700, 700):
                        raise ValueError(f"Expected a 700x700 PNG: {path}")
                    allowed_modes = {"RGB"} if kind == "image" else {"1", "L"}
                    if image.mode not in allowed_modes:
                        raise ValueError(f"Unexpected image mode {image.mode}: {path}")
                    image.load()
                    record.update(mode=image.mode, size=list(image.size))
                    record["pixel_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
                    if kind == "mask":
                        values = {value for _, value in image.getcolors(256)}
                        if not values <= {0, 1} and not values <= {0, 255}:
                            raise ValueError(f"Mask is not binary: {path}")
                        if not any(values):
                            raise ValueError(f"Anomalous image has an empty mask: {path}")
                        record["values"] = sorted(values)
                        mask_values.update(values)
                counts["/".join(PurePosixPath(path).parts[:2])] += 1
            inventory[path] = record

    expected = {"train/good": 220, "test/good": 22}
    for path, count in expected.items():
        if counts[path] != count:
            raise ValueError(f"Expected {count} files in {path}; found {counts[path]}")
    if sum(counts[f"test/{label}"] for label in DEFECTS) != 93:
        raise ValueError("Expected 93 anomalous test images.")
    expected_masks = set()
    for path, record in inventory.items():
        parts = PurePosixPath(path).parts
        if record["kind"] == "image" and parts[0] == "test" and parts[1] != "good":
            mask_path = f"ground_truth/{parts[1]}/{PurePosixPath(path).stem}_mask.png"
            if mask_path not in inventory:
                raise ValueError(f"Missing anomaly mask: {mask_path}")
            expected_masks.add(mask_path)
    masks = {path for path, record in inventory.items() if record["kind"] == "mask"}
    if masks != expected_masks:
        raise ValueError("Masks include unpaired or missing files.")
    for label in DEFECTS:
        if counts[f"test/{label}"] < 2:
            raise ValueError(f"Not enough {label} samples for preview and holdout roles.")

    after = archive_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Archive changed during inspection; wait for the download to finish.")

    return {
        "schema_version": 1,
        "dataset": "MVTec AD",
        "category": "metal_nut",
        "archive_format": "zip" if is_zip else "tar",
        "archive_sha256": archive_hash,
        "archive_bytes": after.st_size,
        "counts": dict(sorted(counts.items())),
        "mask_values": sorted(mask_values),
        "files": dict(sorted(inventory.items())),
        "scope": "Archive audit only: no extraction, split, preview, or inference.",
    }


def extract_audited_archive(archive_path: Path, report: dict, destination: Path) -> dict:
    """Copy only audited regular files, refusing an existing destination."""
    if destination.exists():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    with archive_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != report["archive_sha256"]:
            raise ValueError("Archive hash changed after audit; extraction refused.")
    is_zip = report["archive_format"] == "zip"
    container = zipfile.ZipFile(archive_path) if is_zip else tarfile.open(archive_path, "r:*")
    destination.mkdir(parents=True, exist_ok=False)
    # On failure, leave partial files for inspection; never overwrite or clean up
    # an existing dataset automatically. No success report is written on failure.
    with container as archive:
        for path, record in report["files"].items():
            if canonical_path(record["archive_path"]) != path:
                raise ValueError(f"Audit path mismatch: {path}")
            target = destination / path
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Extraction path escapes destination: {path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            reader = archive.open(record["archive_path"]) if is_zip else archive.extractfile(record["archive_path"])
            with reader as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                    raise ValueError(f"Extracted file hash mismatch: {path}")
    return {"path": str(destination), "files_verified": len(report["files"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--report", type=Path, help="Write an audit JSON; refuses to overwrite.")
    parser.add_argument("--source-url", help="Actual archive URL, if known; never guessed.")
    parser.add_argument("--extract-to", type=Path, help="Extract audited files into a new directory.")
    args = parser.parse_args()
    try:
        if args.report and args.report.exists():
            raise FileExistsError(f"Audit report already exists: {args.report}")
        report = audit_archive(args.archive)
        report["source_url"] = args.source_url
        if args.extract_to:
            report["extraction"] = extract_audited_archive(args.archive, report, args.extract_to)
            report["scope"] = "Archive and extracted-file audit only: no split, preview, or inference."
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
        summary = {key: value for key, value in report.items() if key != "files"}
        print(json.dumps(summary, indent=2, sort_keys=True))
        action = "audited and extracted" if args.extract_to else "audited without extraction"
        print(f"PASS: archive {action}. No images displayed and no model run.")
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile, EOFError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
