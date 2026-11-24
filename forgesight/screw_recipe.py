"""Audit local screw images and freeze leakage-separated recipe partitions."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from forgesight.data_manifest import reject_duplicates, safe_path

POLICY = "forgesight-screw-v1"
DEFECTS = {"manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"}


def rank(row):
    return hashlib.sha256(f"{POLICY}|42|{row['path']}".encode()).hexdigest()


def assign_roles(samples):
    rows = deepcopy(samples)
    groups = {}
    for row in rows:
        groups.setdefault((row["source_split"], row["label"]), []).append(row)
    for (split, label), group in groups.items():
        ordered = sorted(group, key=rank)
        if split == "train":
            n = max(1, len(group) // 5)
            roles = ["validation_normal"] * n + ["fit"] * (len(group) - n)
        elif label == "good":
            n = len(group) // 2
            roles = ["threshold_normal"] * n + ["final_holdout"] * (len(group) - n)
        else:
            n, v = len(group) * 3 // 5, len(group) // 5
            roles = ["classifier_fit"] * n + ["classifier_validation"] * v
            roles += ["final_holdout"] * (len(group) - len(roles))
        for row, role in zip(ordered, roles):
            row["role"] = role
    return sorted(rows, key=lambda row: row["path"])


def image_record(root, relative, mask=False):
    path = safe_path(root, relative)
    data = path.read_bytes()
    with Image.open(path) as image:
        image.load()
        if image.mode not in ({"L"} if mask else {"RGB", "L"}):
            raise ValueError(f"Unexpected image/mask mode: {relative}")
        if mask and (image.getcolors(3) is None
                     or not {value for _, value in image.getcolors(3)} <= {0, 255}
                     or image.getbbox() is None):
            raise ValueError(f"Invalid binary mask: {relative}")
        return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(),
                "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "mode": image.mode, "size": list(image.size)}


def validate_manifest(document):
    if (document.get("schema_version") != 1 or document.get("category") != "screw"
            or document.get("dataset") != "MVTec AD" or document.get("seed") != 42
            or document.get("policy_version") != POLICY):
        raise ValueError("Unsupported screw manifest policy.")
    rows = document["samples"]
    if not rows or len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Empty or repeated sample paths.")
    expected = assign_roles(rows)
    if rows != expected or document["counts"] != dict(Counter(r["role"] for r in rows)):
        raise ValueError("Screw roles/counts differ from the deterministic policy.")
    for row in rows:
        path = safe_path(Path("."), row["path"])
        parts = Path(row["path"]).parts
        if (len(parts) != 3 or path.suffix != ".png" or parts[0] not in {"train", "test"}
                or parts[1] not in DEFECTS | {"good"}
                or parts[0] != row["source_split"] or parts[1] != row["label"]
                or (parts[0] == "train" and parts[1] != "good")):
            raise ValueError("Invalid screw source split/label.")
        records = [(row, {"RGB", "L"})]
        if row["label"] == "good":
            if row["mask"] is not None:
                raise ValueError("Normal image has a mask.")
        else:
            mask = row["mask"]
            target = f"ground_truth/{row['label']}/{path.stem}_mask.png"
            if not isinstance(mask, dict) or mask.get("path") != target or mask["size"] != row["size"]:
                raise ValueError("Missing/misaligned screw mask.")
            records.append((mask, {"L"}))
        for record, modes in records:
            safe_path(Path("."), record["path"])
            if (record["mode"] not in modes or len(record["size"]) != 2
                    or any(type(n) is not int or n <= 0 for n in record["size"])
                    or any(not re.fullmatch(r"[0-9a-f]{64}", record[k])
                           for k in ("sha256", "pixel_sha256"))):
                raise ValueError("Invalid image metadata.")
    reject_duplicates(rows)
    for role in ("fit", "validation_normal", "threshold_normal", "final_holdout"):
        if not document["counts"].get(role):
            raise ValueError(f"Empty required partition: {role}")


def build_manifest(root):
    samples, used_masks = [], set()
    pngs = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.png"))
    for relative in pngs:
        if relative.startswith("ground_truth/"):
            continue
        parts = relative.split("/")
        if len(parts) != 3:
            raise ValueError(f"Unexpected image path: {relative}")
        split, label, _ = parts
        row = image_record(root, relative)
        mask = None
        if label != "good":
            name = f"ground_truth/{label}/{Path(relative).stem}_mask.png"
            if not safe_path(root, name).is_file():
                raise ValueError(f"Missing mask: {name}")
            mask = image_record(root, name, mask=True)
            used_masks.add(name)
        samples.append({**row, "source_split": split, "label": label, "mask": mask})
    if used_masks != {p for p in pngs if p.startswith("ground_truth/")}:
        raise ValueError("Orphan ground-truth mask.")
    rows = assign_roles(samples)
    document = {"schema_version": 1, "dataset": "MVTec AD", "category": "screw",
        "policy_version": POLICY, "seed": 42,
        "counts": dict(Counter(r["role"] for r in rows)),
        "source_counts": dict(Counter(f"{r['source_split']}/{r['label']}" for r in rows)),
        "dimensions": sorted({tuple(r["size"]) for r in rows}),
        "duplicate_check": {"byte_groups": 0, "decoded_pixel_groups": 0,
                            "physical_part_identity": "NOT_VERIFIED"},
        "provenance": "Operator-provided extracted directory; archive provenance not verified",
        "samples": rows}
    # JSON-roundtrip also normalizes dimension tuples before deterministic comparisons.
    document = json.loads(json.dumps(document))
    validate_manifest(document)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/mvtec_ad/screw"))
    parser.add_argument("--output", type=Path, default=Path("manifests/screw_v1.json"))
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".sha256").exists():
        parser.error("Manifest exists; never overwrite a frozen split.")
    document = build_manifest(args.root)
    data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    args.output.with_suffix(".sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")
    print(json.dumps({k: document[k] for k in ("counts", "source_counts", "dimensions")}))


if __name__ == "__main__":
    main()
