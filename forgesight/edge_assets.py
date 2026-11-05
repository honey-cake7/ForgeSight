"""Verify downloaded source assets against a publisher-checked reference."""

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath


def fingerprint(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during verification: {path.name}")
    return after.st_size, digest.hexdigest()


def inspect_tokenizer(path: Path, expected_count: int) -> int:
    ranks, tokens = set(), set()
    with path.open("rb") as stream:
        for line in stream:
            encoded, rank = line.split()
            token = base64.b64decode(encoded, validate=True)
            value = int(rank)
            if not token or token in tokens or value in ranks:
                raise ValueError("Tokenizer contains empty/duplicate tokens or ranks.")
            tokens.add(token)
            ranks.add(value)
    if len(ranks) != expected_count or ranks != set(range(expected_count)):
        raise ValueError("Tokenizer count/ranks differ from the verified reference.")
    return len(ranks)


def verify_assets(root: Path, reference: dict) -> dict:
    verified = {}
    for relative, expected in reference["files"].items():
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or "\\" in relative
                or ":" in relative or path.as_posix() != relative):
            raise ValueError(f"Invalid asset-relative path: {relative}")
        local = root / relative
        if not local.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Asset path escapes root: {relative}")
        size, digest = fingerprint(local)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"Asset fingerprint mismatch: {relative}")
        verified[relative] = {"bytes": size, "sha256": digest}
    params = json.loads((root / "original/params.json").read_text(encoding="utf-8"))
    if params != reference["params"]:
        raise ValueError("Model parameters differ from the verified source revision.")
    count = inspect_tokenizer(root / "original/tokenizer.model", reference["tokenizer_base_tokens"])
    return {"schema_version": 1, "model_id": reference["model_id"],
            "model_revision": reference["model_revision"], "files": verified,
            "params": params, "tokenizer_base_tokens": count,
            "status": "ASSETS_VERIFIED_NOT_EXPORTED", "inference_status": "NOT_RUN"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("models/llama3_2_1b"))
    parser.add_argument("--reference", type=Path, default=Path("manifests/llama3_2_1b_assets.json"))
    parser.add_argument("--profile", type=Path, default=Path("configs/edge_llama.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        if (reference["schema_version"] != 1 or reference["model_id"] != profile["model_id"]
                or reference["model_revision"] != profile["model_revision"]
                or set(reference["files"]) != set(profile["source_files"])):
            raise ValueError("Asset reference does not match the selected model/revision/file list.")
        report = verify_assets(args.root, reference)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
        print(json.dumps({"status": report["status"], "files_verified": len(report["files"]),
                          "tokenizer_base_tokens": report["tokenizer_base_tokens"],
                          "inference_status": report["inference_status"]}, indent=2))
        print("PASS: source assets verified without deserializing the checkpoint or running a model.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
