"""Fit and evaluate a depth-3 defect tree from an independent labeled split."""

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
from sklearn.tree import DecisionTreeClassifier
import torch

from forgesight.data_manifest import DEFAULT_MANIFEST, read_manifest, safe_path
from forgesight.defect_type import FEATURE_NAMES, LABELS, load_profile, predict_type, response_features
from forgesight.regions import response_regions
from scripts.probe_defect_type_features import MODEL_SHA256, infer


SPLITS = ("train", "validation", "test")
SOURCE_LABELS = {"scratch", "discoloration", "good", "bent", "flip", "unknown"}


def load_split(path, protected_manifest):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("Expected a version 1 classifier split.")
    protected = [row for row in read_manifest(protected_manifest)["samples"]
                 if row["role"] == "final_holdout"]
    protected_files = {row["sha256"] for row in protected}
    protected_pixels = {row["pixel_sha256"] for row in protected}
    groups, seen_pixels, samples = {}, set(), defaultdict(list)
    for row in document["samples"]:
        split, group, label = row["split"], row["group"], row["label"]
        if split not in SPLITS or label not in SOURCE_LABELS or not isinstance(group, str) or not group.strip():
            raise ValueError("Expected a train/validation/test role, source group, and supported label.")
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Source group spans multiple splits: {group}")
        payload = safe_path(path.parent, row["path"]).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != row["sha256"] or digest in protected_files:
            raise ValueError(f"Image hash mismatch or protected holdout image: {row['path']}")
        with Image.open(BytesIO(payload)) as source:
            if source.mode != "RGB" or source.size != (700, 700) or source.format != "PNG":
                raise ValueError(f"Expected an RGB 700x700 PNG: {row['path']}")
            pixel_hash = hashlib.sha256(source.tobytes()).hexdigest()
        if pixel_hash in protected_pixels or pixel_hash in seen_pixels:
            raise ValueError(f"Duplicate pixels or protected holdout image: {row['path']}")
        seen_pixels.add(pixel_hash)
        samples[split].append((row, payload))
    for split in SPLITS:
        labels = {target(row["label"]) for row, _ in samples[split]}
        if labels != set(LABELS):
            raise ValueError(f"{split} requires scratch, discoloration, and other/normal examples.")
    return samples


def target(label):
    return label if label in LABELS else "unknown"


def vector(features):
    return [features[name] for name in FEATURE_NAMES]


def export_tree(tree, validation):
    """Abstain at leaves without consistent support from separate validation images."""
    support = defaultdict(list)
    for row in validation:
        if row["features"] is not None:
            leaf = int(tree.apply([vector(row["features"])])[0])
            support[leaf].append(target(row["label"]))

    def export(node_id):
        structure = tree.tree_
        left, right = structure.children_left[node_id], structure.children_right[node_id]
        if left == right:
            label = str(tree.classes_[structure.value[node_id][0].argmax()])
            return {"label": label if support[node_id] and set(support[node_id]) == {label} else "unknown"}
        return {"feature": FEATURE_NAMES[structure.feature[node_id]],
                "threshold": float(structure.threshold[node_id]),
                "left": export(left), "right": export(right)}

    return export(0)


def evaluate(rows, profile):
    decisions = [{**row, "prediction": predict_type(row["features"], profile)
                  if row["flagged"] else "not_flagged"} for row in rows]
    confusion = defaultdict(Counter)
    for row in decisions:
        confusion[row["label"]][row["prediction"]] += 1
    return {"counts_by_true_label": dict(confusion), "images": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256:
        parser.error("Expected the frozen EfficientAD TorchScript export.")
    samples = load_split(args.split, DEFAULT_MANIFEST)
    threshold = json.loads(Path("configs/forgegrade_v1.json").read_text(encoding="utf-8"))["image_threshold"]
    torch.set_num_threads(2)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    rows = defaultdict(list)
    for split, entries in samples.items():
        for row, payload in entries:
            with Image.open(BytesIO(payload)) as image:
                score, values = infer(model, image)
                flagged = score > threshold
                geometry = response_regions(values, threshold) if flagged else None
                features = response_features(image, values, geometry) if flagged else None
            rows[split].append({"image": row["path"], "label": row["label"], "group": row["group"],
                                "flagged": flagged, "score": score, "features": features})
    train = [row for row in rows["train"] if row["features"] is not None]
    if not {"scratch", "discoloration"} <= {target(row["label"]) for row in train}:
        parser.error("EfficientAD must flag training examples of both target defects with usable geometry.")
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit([vector(row["features"]) for row in train], [target(row["label"]) for row in train])
    split_hash = hashlib.sha256(args.split.read_bytes()).hexdigest()
    profile = {"schema_version": 1, "kind": "decision_tree", "feature_names": list(FEATURE_NAMES),
               "profile_id": f"metal-nut-tree-{split_hash[:12]}",
               "split_sha256": split_hash, "model_sha256": MODEL_SHA256, "image_threshold": threshold,
               "tree": export_tree(tree, rows["validation"])}
    args.output.mkdir(parents=True)
    profile_path = args.output / "candidate.json"
    profile_path.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    baseline = load_profile(Path("configs/defect_type_preview_v1.json"))
    report = {"split_counts": {split: dict(Counter(row["label"] for row in rows[split])) for split in SPLITS},
              "features_available_in_training": len(train),
              "candidate_test": evaluate(rows["test"], load_profile(profile_path)),
              "preview_rules_test": evaluate(rows["test"], baseline)}
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
