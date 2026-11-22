"""Run the frozen detector and severity policy on the subtype test and replay previews."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch

from forgesight.data_manifest import DEFAULT_MANIFEST, DEFAULT_ROOT, read_manifest, safe_path
from forgesight.defect_type import assess_response, load_profile
from forgesight.regions import response_regions
from scripts.probe_defect_type_features import MODEL_SHA256, infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")

    split = Path("manifests/metal_nut_defect_type_split_v1.json")
    profile = load_profile(Path("configs/defect_type_mvtec_four_v1.json"))
    if (hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256
            or profile["model_sha256"] != MODEL_SHA256
            or hashlib.sha256(split.read_bytes()).hexdigest() != profile["split_sha256"]):
        parser.error("Expected the frozen model, subtype profile and split.")

    torch.set_num_threads(2)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    threshold = profile["image_threshold"]
    assignments = json.loads(split.read_text(encoding="utf-8"))["assignments"]
    report = {"profile_id": profile["profile_id"], "images": {}, "replays": {}}
    for sample in read_manifest(DEFAULT_MANIFEST)["samples"]:
        path = sample["path"]
        if sample["role"] == "development_preview":
            group = "replays"
        elif assignments.get(path) == "test":
            group = "images"
        else:
            continue
        original = safe_path(args.dataset_root, path)
        if hashlib.sha256(original.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError(f"Benchmark image hash mismatch: {path}")
        with Image.open(original) as image:
            score, score_map = infer(model, image)
            flagged = score > threshold
            mode = "SAVED_IMAGE_REPLAY" if group == "replays" else "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD"
            record = {"source": {"mode": mode},
                      "candidate": "ANOMALY_CANDIDATE" if flagged else "NORMAL_CANDIDATE",
                      "model": {"export_sha256": MODEL_SHA256, "image_threshold": threshold},
                      "geometry": response_regions(score_map, threshold, mvtec_nut_layout=True),
                      "timings": {}}
            assess_response(record, image, score_map, profile)
        assessment = record.get("assessment")
        report[group][path] = {"dataset_label": sample["label"],
                               "type": assessment["defect_type"] if assessment else None,
                               "severity": assessment["severity"] if assessment else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
