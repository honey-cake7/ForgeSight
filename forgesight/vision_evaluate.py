"""Evaluate a frozen model once; reproduce metrics later from saved predictions."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from forgesight.data_loader import MetalNutDataset
from forgesight.data_manifest import DEFAULT_ROOT
from forgesight.edge_assets import fingerprint
from forgesight.vision_inference import CalibratedPredictor


def detection_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    if (not labels or len(labels) != len(scores) or set(labels) - {0, 1}
            or not math.isfinite(threshold) or any(not math.isfinite(v) for v in scores)):
        raise ValueError("Invalid binary labels, finite scores or threshold.")
    predicted = [score > threshold for score in scores]
    tp = sum(y == 1 and p for y, p in zip(labels, predicted))
    tn = sum(y == 0 and not p for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p for y, p in zip(labels, predicted))
    fn = sum(y == 1 and not p for y, p in zip(labels, predicted))
    auc = None
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if positives and negatives:
        ordered = sorted(zip(scores, labels))
        rank_sum, offset = 0.0, 0
        while offset < len(ordered):
            end = offset + 1
            while end < len(ordered) and ordered[end][0] == ordered[offset][0]:
                end += 1
            rank_sum += sum(label for _, label in ordered[offset:end]) * (offset + 1 + end) / 2
            offset = end
        auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return {"images": len(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "recall": tp / positives if positives else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "false_alarm_rate": fp / negatives if negatives else None,
            "accuracy": (tp + tn) / len(labels), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "image_auroc": auc}


def summarize(output: Path) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    context = json.loads((output / "protocol.json").read_text())
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    if len(rows) != context["protocol"]["expected_images"] or len({r["source_path"] for r in rows}) != len(rows):
        raise ValueError("Saved evaluation is incomplete or duplicated.")
    metrics = detection_metrics([r["truth_anomalous"] for r in rows], [r["score"] for r in rows], context["threshold"])
    masks, maps = [], []
    for row in rows:
        path = output / row["map_file"]
        if fingerprint(path)[1] != row["map_file_sha256"]:
            raise ValueError("Saved map changed.")
        with np.load(path, allow_pickle=False) as saved:
            masks.append(saved["truth_mask"].reshape(-1))
            maps.append(saved["map_source"].reshape(-1))
    metrics["pixel_auroc_source_grid"] = float(roc_auc_score(np.concatenate(masks), np.concatenate(maps)))
    per_truth_class = {}
    for label in sorted({r["truth_type"] for r in rows}):
        group = [r for r in rows if r["truth_type"] == label]
        per_truth_class[label] = {"images": len(group), "flagged": sum(r["candidate"] == "ANOMALY_CANDIDATE" for r in group),
                                 "note": "Grouped by dataset ground truth, not predicted defect type."}
    return {"schema_version": 1, "status": "FROZEN_HOLDOUT_EVALUATED_NO_TUNING", **context,
            "metrics": metrics, "per_ground_truth_class": per_truth_class,
            "false_negatives": [r["source_path"] for r in rows if r["truth_anomalous"] and r["candidate"] == "NORMAL_CANDIDATE"],
            "false_positives": [r["source_path"] for r in rows if not r["truth_anomalous"] and r["candidate"] == "ANOMALY_CANDIDATE"],
            "predictions_sha256": fingerprint(output / "predictions.jsonl")[1],
            "predicted_defect_type": "unknown", "subtype_classification": "NOT_IMPLEMENTED",
            "pixel_mask_accuracy": "NOT_EVALUATED_NO_SEPARATE_PIXEL_THRESHOLD", "jetson_proof": "NOT_VERIFIED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/holdout_evaluation_v1.json"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-final-holdout", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        report = summarize(args.output)
        target = args.output / "metrics_reproduced.json"
        with target.open("x") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report["metrics"], indent=2))
        return
    if not args.allow_final_holdout or args.output.exists():
        parser.error("Explicit holdout access and a new output directory are required.")
    protocol = json.loads(args.protocol.read_text())
    if protocol["role"] != "final_holdout" or protocol["tune_from_holdout"] is not False:
        parser.error("Expected the frozen evaluation-only protocol.")
    import numpy as np
    import torch

    reference = Path(protocol["calibration_reference"])
    dataset = MetalNutDataset(args.root, Path(protocol["dataset_manifest"]), "final_holdout", allow_final_holdout=True)
    if len(dataset) != protocol["expected_images"]:
        parser.error("Holdout count differs from frozen protocol.")
    ledger = Path("artifacts/vision/holdout_exposure_v1.json")
    if ledger.exists():
        parser.error("This holdout evaluation is already recorded; reproduce saved metrics instead of reinference.")
    predictor = CalibratedPredictor(reference)
    # Check the shared preprocessing on a previously exposed calibration anchor before opening holdout pixels.
    from forgesight.vision_calibrate import tensor_for
    from forgesight.vision_train import normal_rows
    from forgesight.data_manifest import read_manifest
    from forgesight.data_loader import load_verified_image
    normal_document = read_manifest(Path(protocol["dataset_manifest"]))
    known = predictor.reference["bundle_reload_verification"]["checks"][0]
    normal_row = next(r for r in normal_rows(normal_document, "validation_normal") if r["path"] == known["path"])
    with load_verified_image(args.root, normal_row) as image:
        if predictor.predict(image)["score"] != known["score"]:
            raise ValueError("Shared inference path differs from frozen calibration.")
    context = {"protocol": protocol, "protocol_sha256": fingerprint(args.protocol)[1],
               "calibration_reference_sha256": fingerprint(reference)[1],
               "checkpoint_sha256": predictor.reference["calibrated_checkpoint"]["sha256"],
               "policy_sha256": predictor.reference["policy_file_sha256"], "threshold": predictor.policy["threshold"],
               "started_utc": datetime.now(timezone.utc).isoformat(), "output": str(args.output),
               "evaluation_source_sha256": fingerprint(Path(__file__))[1]}
    args.output.mkdir(parents=True)
    (args.output / "maps").mkdir()
    (args.output / "protocol.json").write_text(json.dumps(context, indent=2) + "\n")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("x") as stream:
        json.dump({"status": "HOLDOUT_EXPOSED_EVALUATION_STARTED", **context}, stream, indent=2)
    with (args.output / "predictions.jsonl").open("x") as log:
        for index, sample in enumerate(dataset):
            with sample.image, sample.mask:
                prediction = predictor.predict(sample.image)
                map_file = f"maps/{index:03d}.npz"
                np.savez_compressed(args.output / map_file, map_model=prediction["map_model"],
                                    map_source=prediction["map_source"], truth_mask=np.asarray(sample.mask) > 0)
                row = {"source_path": sample.source_path, "truth_type": sample.ground_truth_type,
                       "truth_anomalous": int(sample.ground_truth_type != "good"),
                       "score": prediction["score"], "candidate": prediction["candidate"],
                       "predicted_defect_type": "unknown", "source_size": list(sample.image.size),
                       "map_file": map_file, "map_file_sha256": fingerprint(args.output / map_file)[1],
                       "model_forward_ms": prediction["model_forward_ms"], "predict_image_ms": prediction["predict_image_ms"]}
                log.write(json.dumps(row) + "\n")
                log.flush()
    report = summarize(args.output)
    (args.output / "evaluation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    ledger.write_text(json.dumps({"status": "HOLDOUT_EXPOSED_EVALUATION_COMPLETED", **context,
                                 "report_sha256": fingerprint(args.output / "evaluation_report.json")[1]}, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
