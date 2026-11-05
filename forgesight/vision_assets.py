"""Acquire hash-pinned EfficientAD training assets without executing archive content."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import urllib.request
import zipfile


def safe_destination(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Archive destination escapes extraction root.")
    return target


def extract_verified_archive(archive: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    is_zip = zipfile.is_zipfile(archive)
    container = zipfile.ZipFile(archive) if is_zip else tarfile.open(archive, "r:gz")
    count, total, seen, files = 0, 0, set(), []
    with container as reader:
        members = reader.infolist() if is_zip else reader.getmembers()
        if len(members) > 50000:
            raise ValueError("Too many archive entries.")
        for member in members:
            name = member.filename if is_zip else member.name
            target = safe_destination(output, name)
            if target in seen:
                raise ValueError("Duplicate archive destination.")
            seen.add(target)
            directory = member.is_dir() if is_zip else member.isdir()
            mode = stat.S_IFMT(member.external_attr >> 16) if is_zip else None
            if (is_zip and (mode not in {0, stat.S_IFREG, stat.S_IFDIR} or member.flag_bits & 1)
                    or not is_zip and not (member.isfile() or directory)):
                raise ValueError("Archive links, special or encrypted files are refused.")
            if directory:
                continue
            size = member.file_size if is_zip else member.size
            total += size
            if not 0 <= size <= 128 * 1024**2 or total > 8 * 1024**3:
                raise ValueError("Archive extraction limits exceeded.")
            files.append((member, target, size))
        for member, target, size in files:
            target.parent.mkdir(parents=True, exist_ok=True)
            source = reader.open(member) if is_zip else reader.extractfile(member)
            with source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
            if target.stat().st_size != size:
                raise ValueError("Truncated archive member.")
            count += 1
    return {"files": count, "uncompressed_bytes": total}


def acquire(spec: dict, root: Path) -> dict:
    name = spec["url"].rsplit("/", 1)[1]
    destination = root / name
    if destination.exists():
        raise FileExistsError(destination)
    digest, size = hashlib.sha256(), 0
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "ForgeSight-research-asset-verifier"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as stream:
        while data := response.read(1024 * 1024):
            size += len(data)
            if size > 2 * 1024**3:
                raise ValueError("Download size limit exceeded.")
            digest.update(data)
            stream.write(data)
        final_url = response.geturl().split("?", 1)[0]
    actual = digest.hexdigest()
    if actual != spec["expected_sha256_from_upstream"]:
        raise ValueError(f"Archive SHA-256 mismatch: {name}")
    print(f"VERIFIED {name}: {size} bytes, {actual}", flush=True)
    extracted = root / (name + ".extracted")
    inventory = extract_verified_archive(destination, extracted)
    return {"source_url": spec["url"], "download_host_path": final_url,
            "archive_path": str(destination), "archive_bytes": size, "archive_sha256": actual,
            "extracted_root": str(extracted), **inventory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/vision_baseline.json"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.root.exists() or args.output.exists():
        parser.error("Use new asset and evidence paths; existing work is preserved.")
    profile = json.loads(args.profile.read_text())
    args.root.mkdir(parents=True)
    report = {"status": "ACQUISITION_IN_PROGRESS", "assets": {}}
    for key in ("pretrained_teacher_archive", "regularization_archive"):
        report["assets"][key] = acquire(profile[key], args.root.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "TRAINING_ASSETS_HASH_VERIFIED"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: archive integrity and bounded extraction; no training or checkpoint loading.")


if __name__ == "__main__":
    main()
