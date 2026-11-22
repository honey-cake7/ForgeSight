"""Audit detector coverage or train an optional metal-nut subtype classifier."""

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import torch

from forgesight.data_manifest import DEFAULT_MANIFEST, DEFAULT_ROOT, read_manifest, safe_path
from forgesight.defect_type import (FEATURE_NAMES, FOREST_FEATURES, FOUR_TYPES, LABELS, four_type_features,
                                    load_profile, predict_type, response_features)
from forgesight.regions import response_regions
from scripts.build_mvtec_defect_type_split import PROTOCOL, build_split
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
    if not seen_pixels:
        raise ValueError("Expected at least one labeled image.")
    return samples


def target(label):
    return label if label in LABELS else "unknown"


def load_mvtec_experiment(path, dataset_root):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document != build_split(DEFAULT_MANIFEST):
        raise ValueError("MVTec subtype experiment requires the frozen v1 assignments.")
    source = read_manifest(DEFAULT_MANIFEST)
    samples = defaultdict(list)
    assignments = document["assignments"]
    for original in source["samples"]:
        relative = original["path"]
        if relative not in assignments:
            continue
        payload = safe_path(dataset_root, relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != original["sha256"]:
            raise ValueError(f"Original MVTec image hash mismatch: {relative}")
        label = "discoloration" if original["label"] == "color" else original["label"]
        row = {"path": relative, "label": label, "group": original["sha256"]}
        samples[assignments[relative]].append((row, payload))
    return samples


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


def train_four_type(rows, profile):
    from sklearn.ensemble import RandomForestClassifier

    train = [row for row in rows["train"] if row["features"] is not None]
    labels = [row["label"] if row["label"] in FOUR_TYPES else "unknown" for row in train]
    if set(labels) != set(FOUR_TYPES) | {"unknown"}:
        raise ValueError("Four-type training requires usable examples of each type and flagged negatives.")
    classifier = RandomForestClassifier(n_estimators=31, max_depth=5, class_weight="balanced",
                                        random_state=42)
    classifier.fit([[row["features"][name] for name in FOREST_FEATURES] for row in train], labels)

    def export(tree, node=0):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:
            return {"label": str(classifier.classes_[tree.value[node][0].argmax()])}
        return {"feature": FOREST_FEATURES[tree.feature[node]], "threshold": float(tree.threshold[node]),
                "left": export(tree, left), "right": export(tree, right)}

    profile.update(kind="four_type_forest", feature_names=list(FOREST_FEATURES),
                   profile_id="metal-nut-four-type-v1-" + profile["split_sha256"][:12],
                   trees=[export(tree.tree_) for tree in classifier.estimators_])
    # Select a strict majority on validation only. Reward correct type calls and penalize wrong calls; abstentions score zero.
    def utility(min_votes):
        candidate = dict(profile, min_votes=min_votes)
        decisions = evaluate(rows["validation"], candidate)["images"]
        correct = sum(row["prediction"] in FOUR_TYPES and row["prediction"] == row["label"] for row in decisions)
        wrong = sum(row["prediction"] in FOUR_TYPES and row["prediction"] != row["label"] for row in decisions)
        return correct - 2 * wrong, correct, min_votes

    profile["min_votes"] = max(range(16, 32), key=utility)


def evaluate(rows, profile):
    decisions = [{**row, "prediction": predict_type(row["features"], profile)
                  if row["flagged"] else "not_flagged"} for row in rows]
    confusion = defaultdict(Counter)
    for row in decisions:
        confusion[row["label"]][row["prediction"]] += 1
    return {"counts_by_true_label": dict(confusion), "images": decisions}


def compare_subtypes(candidate, baseline, known=("scratch", "discoloration")):

    def summarize(evaluation):
        images = evaluation["images"]
        return {"correct_by_target": {label: sum(row["label"] == row["prediction"] == label
                                                 for row in images) for label in known},
                "incorrect_type_assignments": sum(row["prediction"] in known and row["prediction"] != row["label"]
                                                  for row in images),
                "unknown": sum(row["prediction"] == "unknown" for row in images),
                "not_flagged": sum(row["prediction"] == "not_flagged" for row in images)}

    tree, rules = summarize(candidate), summarize(baseline)
    tree_correct, rules_correct = tree["correct_by_target"], rules["correct_by_target"]
    improves = (sum(tree_correct.values()) > sum(rules_correct.values())
                and all(tree_correct[label] >= rules_correct[label] for label in known)
                and tree["incorrect_type_assignments"] <= rules["incorrect_type_assignments"])
    return {"candidate": tree, "rules": rules, "candidate_passed_promotion_gate": improves}


def coverage_report(rows, four_types=False):
    required_types = set(FOUR_TYPES if four_types else ("scratch", "discoloration"))

    def coverage_label(row):
        return row["label"] if row["label"] in required_types else "unknown"

    coverage = {}
    issues = []
    for split in SPLITS:
        entries = rows.get(split, [])
        missing_labels = (required_types | {"unknown"}) - {coverage_label(row) for row in entries}
        if missing_labels:
            issues.append(f"{split} lacks labels: {', '.join(sorted(missing_labels))}")
        by_label = defaultdict(list)
        for row in entries:
            by_label[row["label"]].append(row)
        coverage[split] = {
            label: {"images": len(items), "source_groups": len({row["group"] for row in items}),
                    "flagged": sum(row["flagged"] for row in items),
                    "usable_features": sum(row["features"] is not None for row in items)}
            for label, items in sorted(by_label.items())
        }
    usable_training_labels = {coverage_label(row) for row in rows.get("train", []) if row["features"] is not None}
    missing_training = (required_types | ({"unknown"} if four_types else set())) - usable_training_labels
    if missing_training:
        issues.append(f"train lacks flagged images with usable features: {', '.join(sorted(missing_training))}")
    return {"splits": coverage, "minimum_training_inputs_met": not issues, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true",
                        help="Save coverage for each split and label without fitting, even if a split is incomplete.")
    parser.add_argument("--mvtec-subtype-experiment", action="store_true",
                        help="Allow only the frozen within-MVTec split of detector-evaluated images.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT,
                        help="Original metal_nut root for the within-MVTec experiment.")
    parser.add_argument("--four-types", action="store_true", help="Fit a four-type voting forest on the frozen MVTec split.")
    args = parser.parse_args()
    if args.four_types and not args.mvtec_subtype_experiment:
        parser.error("--four-types requires --mvtec-subtype-experiment")
    extract = four_type_features if args.four_types else response_features
    if args.output.exists():
        parser.error("Choose a new output directory.")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256:
        parser.error("Expected the frozen EfficientAD TorchScript export.")
    samples = (load_mvtec_experiment(args.split, args.dataset_root) if args.mvtec_subtype_experiment
               else load_split(args.split, DEFAULT_MANIFEST))
    threshold = json.loads(Path("configs/forgegrade_v1.json").read_text(encoding="utf-8"))["image_threshold"]
    torch.set_num_threads(2)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    rows = defaultdict(list)
    preview_test = []
    for split, entries in samples.items():
        for row, payload in entries:
            with Image.open(BytesIO(payload)) as image:
                score, values = infer(model, image)
                flagged = score > threshold
                geometry = response_regions(values, threshold) if flagged else None
                features = extract(image, values, geometry) if flagged else None
                if split == "test" and args.four_types:
                    preview_features = features
                    if flagged and features is None:
                        preview_features = response_features(image, values, geometry)
                    preview_test.append({"image": row["path"], "label": row["label"], "flagged": flagged,
                                         "group": row["group"], "score": score, "features": preview_features})
            rows[split].append({"image": row["path"], "label": row["label"], "group": row["group"],
                                "flagged": flagged, "score": score, "features": features})
    split_hash = hashlib.sha256(args.split.read_bytes()).hexdigest()
    coverage = {**coverage_report(rows, args.four_types), "split_sha256": split_hash, "model_sha256": MODEL_SHA256,
                "image_threshold": threshold}
    if args.mvtec_subtype_experiment:
        coverage["protocol"] = PROTOCOL
    args.output.mkdir(parents=True)
    (args.output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    if args.audit_only:
        print(json.dumps(coverage, indent=2))
        return
    if not coverage["minimum_training_inputs_met"]:
        parser.error("; ".join(coverage["issues"]) + f". See {args.output / 'coverage.json'}.")

    from sklearn.tree import DecisionTreeClassifier

    train = [row for row in rows["train"] if row["features"] is not None]
    profile = {"schema_version": 1, "split_sha256": split_hash, "model_sha256": MODEL_SHA256,
               "image_threshold": threshold}
    if args.four_types:
        train_four_type(rows, profile)
    else:
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit([vector(row["features"]) for row in train], [target(row["label"]) for row in train])
        profile.update(kind="decision_tree", feature_names=list(FEATURE_NAMES),
                       profile_id=f"metal-nut-tree-{split_hash[:12]}", tree=export_tree(tree, rows["validation"]))
    if args.mvtec_subtype_experiment:
        profile["training_protocol"] = PROTOCOL
    profile_path = args.output / "candidate.json"
    profile_path.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    baseline = load_profile(Path("configs/defect_type_preview_v1.json"))
    report = {"split_counts": {split: dict(Counter(row["label"] for row in rows[split])) for split in SPLITS},
              "features_available_in_training": len(train),
              "candidate_test": evaluate(rows["test"], load_profile(profile_path)),
              "preview_rules_test": evaluate(preview_test or rows["test"], baseline)}
    if args.mvtec_subtype_experiment:
        report["protocol"] = PROTOCOL
        report["comparison"] = compare_subtypes(report["candidate_test"], report["preview_rules_test"],
                                                FOUR_TYPES if args.four_types else ("scratch", "discoloration"))
        report["candidate_validation"] = evaluate(rows["validation"], profile)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
