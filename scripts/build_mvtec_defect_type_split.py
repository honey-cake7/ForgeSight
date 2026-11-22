"""Freeze a stratified subtype split using only the original manifest metadata."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from forgesight.data_manifest import DEFAULT_MANIFEST, read_manifest


PROTOCOL = "mvtec-metal-nut-subtype-on-detector-evaluated-images-v1"


def build_split(manifest_path):
    source = read_manifest(manifest_path)
    by_label = defaultdict(list)
    for row in source["samples"]:
        if row["role"] == "final_holdout":
            by_label[row["label"]].append(row["path"])

    def rank(path):
        return hashlib.sha256(f"{PROTOCOL}|42|{path}".encode()).hexdigest(), path

    assignments = {}
    for paths in by_label.values():
        ranked = sorted(paths, key=rank)
        train_end = round(len(ranked) * 0.6)
        validation_end = train_end + round(len(ranked) * 0.2)
        for split, selected in (("train", ranked[:train_end]),
                                ("validation", ranked[train_end:validation_end]),
                                ("test", ranked[validation_end:])):
            assignments.update(dict.fromkeys(selected, split))
    return {"schema_version": 1, "protocol": PROTOCOL, "seed": 42,
            "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source_role": "final_holdout", "target_train_validation_test": [0.6, 0.2, 0.2],
            "assignments": dict(sorted(assignments.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(build_split(DEFAULT_MANIFEST), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
