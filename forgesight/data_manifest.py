"""Freeze and verify the documented metal-nut split before sample viewing."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from PIL import Image


POLICY = "forgesight-metal-nut-v1"
SEED = 42
DEFECTS = ("bent", "color", "flip", "scratch")
ROLES = ("fit", "validation_normal", "development_preview", "final_holdout")
EXPECTED_COUNTS = dict(zip(ROLES, (176, 44, 4, 111)))
DEFAULT_ROOT = Path("data/mvtec_ad/metal_nut")
DEFAULT_MANIFEST = Path("manifests/metal_nut_v1.json")


def safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or ".." in path.parts or "\\" in relative
            or ":" in relative or path.as_posix() != relative):
        raise ValueError(f"Invalid dataset-relative path: {relative!r}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return target


def rank(path: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{POLICY}|{SEED}|{path}".encode("utf-8")).hexdigest()
    return digest, path


def assign_roles(samples: list[dict]) -> list[dict]:
    rows = deepcopy(samples)
    normal = sorted((row["path"] for row in rows if row["source_split"] == "train"), key=rank)
    validation = set(normal[:len(normal) // 5])
    preview = set()
    for label in DEFECTS:
        candidates = [row["path"] for row in rows
                      if row["source_split"] == "test" and row["label"] == label]
        if len(candidates) < 2:
            raise ValueError(f"Not enough {label} samples for preview and holdout.")
        preview.add(min(candidates, key=rank))
    for row in rows:
        if row["source_split"] == "train":
            row["role"] = "validation_normal" if row["path"] in validation else "fit"
        else:
            row["role"] = "development_preview" if row["path"] in preview else "final_holdout"
    return sorted(rows, key=lambda row: row["path"])


def reject_duplicates(samples: list[dict]) -> None:
    for field in ("sha256", "pixel_sha256"):
        groups = defaultdict(list)
        for row in samples:
            key = row[field] if field == "sha256" else (row["mode"], tuple(row["size"]), row[field])
            groups[key].append(row["path"])
        duplicates = [paths for paths in groups.values() if len(paths) > 1]
        if duplicates:
            raise ValueError(f"Duplicate image content ({field}); review before splitting: {duplicates[0]}")


def validate_manifest(manifest: dict) -> None:
    if (manifest.get("schema_version") != 1 or manifest.get("policy_version") != POLICY
            or manifest.get("seed") != SEED or manifest.get("category") != "metal_nut"
            or manifest.get("dataset") != "MVTec AD"):
        raise ValueError("Unsupported manifest schema or split policy.")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.get("source_archive_sha256", "")):
        raise ValueError("Invalid source archive fingerprint.")
    rows = manifest["samples"]
    if len(rows) != 335 or len({row["path"] for row in rows}) != 335:
        raise ValueError("Manifest must contain 335 unique image paths.")
    for row in rows:
        path = PurePosixPath(row["path"])
        safe_path(Path("."), row["path"])
        if len(path.parts) != 3 or path.suffix != ".png":
            raise ValueError(f"Invalid image path: {path}")
        split, label, _ = path.parts
        if (split not in {"train", "test"} or label not in {*DEFECTS, "good"}
                or (split == "train" and label != "good")
                or split != row["source_split"] or label != row["label"]):
            raise ValueError(f"Source split/label mismatch: {path}")
        records = [row]
        if label == "good":
            if row["mask"] is not None:
                raise ValueError(f"Normal sample must have no source mask: {path}")
        else:
            mask = row["mask"]
            expected = f"ground_truth/{label}/{path.stem}_mask.png"
            if not isinstance(mask, dict) or mask.get("path") != expected:
                raise ValueError(f"Missing or mismatched mask: {path}")
            records.append(mask)
        for record in records:
            expected_mode = "RGB" if record is row else "L"
            if record["mode"] != expected_mode or record["size"] != [700, 700]:
                raise ValueError(f"Unexpected image or mask geometry: {path}")
            for field in ("sha256", "pixel_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", record[field]):
                    raise ValueError(f"Invalid {field}: {path}")
    source_counts = Counter(f"{row['source_split']}/{row['label']}" for row in rows)
    if source_counts != {"train/good": 220, "test/good": 22, "test/bent": 25,
                         "test/color": 22, "test/flip": 23, "test/scratch": 23}:
        raise ValueError("Source inventory differs from the acquired category.")
    if Counter(row["role"] for row in rows) != EXPECTED_COUNTS or manifest["counts"] != EXPECTED_COUNTS:
        raise ValueError("Unexpected role counts.")
    expected_roles = {row["path"]: row["role"] for row in assign_roles(rows)}
    if any(row["role"] != expected_roles[row["path"]] for row in rows):
        raise ValueError("Roles differ from the recorded deterministic policy.")
    reject_duplicates(rows)


def build_manifest(root: Path, audit: dict) -> dict:
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(audit["files"]):
        raise ValueError("Local files differ from the acquisition inventory.")
    verified = {}
    for relative, reference in audit["files"].items():
        path = safe_path(root, relative)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != reference["sha256"]:
            raise ValueError(f"File changed since acquisition: {relative}")
        if reference["kind"] == "notice":
            continue
        with Image.open(path) as image:
            image.load()
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if (pixel_hash != reference["pixel_sha256"] or image.mode != reference["mode"]
                    or list(image.size) != reference["size"]):
                raise ValueError(f"Decoded image changed since acquisition: {relative}")
        verified[relative] = {"path": relative, "sha256": digest, "pixel_sha256": pixel_hash,
                              "mode": reference["mode"], "size": reference["size"]}
    samples = []
    for relative, record in verified.items():
        split, label, _ = PurePosixPath(relative).parts
        if split == "ground_truth":
            continue
        mask = None if label == "good" else verified[f"ground_truth/{label}/{PurePosixPath(relative).stem}_mask.png"]
        samples.append({**record, "source_split": split, "label": label, "mask": mask})
    reject_duplicates(samples)
    manifest = {"schema_version": 1, "dataset": "MVTec AD", "category": "metal_nut",
                "policy_version": POLICY, "seed": SEED,
                "source_archive_sha256": audit["archive_sha256"],
                "counts": EXPECTED_COUNTS.copy(),
                "duplicate_check": {"byte_groups": 0, "decoded_pixel_groups": 0,
                                    "physical_part_identity": "NOT VERIFIED"},
                "samples": assign_roles(samples)}
    validate_manifest(manifest)
    return manifest


def encoded_manifest(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_manifest(path: Path) -> dict:
    data = path.read_bytes()
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Manifest checksum mismatch.")
    manifest = json.loads(data)
    validate_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--audit", type=Path, default=Path("artifacts/data/metal_nut_preparation.json"))
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true", help="Rebuild in memory and verify frozen bytes.")
    args = parser.parse_args()
    try:
        manifest = build_manifest(args.root, json.loads(args.audit.read_text(encoding="utf-8")))
        data = encoded_manifest(manifest)
        digest = hashlib.sha256(data).hexdigest()
        checksum = args.output.with_suffix(".sha256")
        if args.check:
            read_manifest(args.output)
            if args.output.read_bytes() != data:
                raise ValueError("Frozen manifest differs from a fresh deterministic build.")
        else:
            if args.output.exists() or checksum.exists():
                raise FileExistsError("Manifest or checksum exists; use --check, never overwrite.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as stream:
                stream.write(data)
            with checksum.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(digest + "\n")
        print(json.dumps({"counts": manifest["counts"], "manifest_sha256": digest,
                          "byte_duplicate_groups": 0, "pixel_duplicate_groups": 0}, indent=2))
        print("PASS: deterministic, disjoint split; no previews or model inference.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
