"""Deterministic metadata fixtures independent of the real dataset images."""

import hashlib

from forgesight.data_manifest import EXPECTED_COUNTS, assign_roles


def synthetic_manifest():
    rows = []
    for group, count in {"train/good": 220, "test/good": 22, "test/bent": 25,
                         "test/color": 22, "test/flip": 23, "test/scratch": 23}.items():
        split, label = group.split("/")
        for index in range(count):
            path = f"{group}/{index:03}.png"
            digest = hashlib.sha256(path.encode()).hexdigest()
            mask = None if label == "good" else {
                "path": f"ground_truth/{label}/{index:03}_mask.png", "sha256": digest,
                "pixel_sha256": digest, "mode": "L", "size": [700, 700]}
            rows.append({"path": path, "source_split": split, "label": label,
                         "sha256": digest, "pixel_sha256": digest, "mode": "RGB",
                         "size": [700, 700], "mask": mask})
    return {"schema_version": 1, "dataset": "MVTec AD", "category": "metal_nut",
            "policy_version": "forgesight-metal-nut-v1", "seed": 42,
            "source_archive_sha256": "a" * 64, "counts": EXPECTED_COUNTS.copy(),
            "samples": assign_roles(rows)}
