"""Verify the screw export on calibration normals, then evaluate a locked holdout once."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from forgesight.data_loader import load_verified_image
from forgesight.edge_assets import fingerprint
from forgesight.screw_recipe import validate_manifest
from forgesight.vision_calibrate import classify_score, tensor_for
from forgesight.vision_calibration_bundle import load_calibrated_bundle
from forgesight.vision_evaluate import detection_metrics
from forgesight.vision_recipe import read_manifest

ROOT = Path("data/mvtec_ad/screw")
REFERENCE = Path("manifests/screw_efficientad_s_calibration_v1.json")
EXPORT = Path("plugins/screw/efficientad_s.ts")
PARITY = Path("manifests/screw_torchscript_export_v1.json")
EXPOSURE = Path("artifacts/vision/screw_holdout_exposure_v1.json")
OUTPUT = Path("artifacts/vision/screw_evaluation_v1")


def acceptance_gate(metrics):
    targets = {"image_auroc": .95, "recall": .95, "false_alarm_rate": .10}
    checks = {}
    for key, target in targets.items():
        value = metrics.get(key)
        checks[key] = (isinstance(value, (int, float)) and math.isfinite(value)
                       and (value <= target if key == "false_alarm_rate" else value >= target))
    return {"targets": targets, "checks": checks, "passed": all(checks.values())}


def holdout_rows(document):
    validate_manifest(document)
    rows = [r for r in document["samples"] if r["role"] == "final_holdout"]
    if len(rows) != document["counts"]["final_holdout"]:
        raise ValueError("Holdout count mismatch.")
    if not any(r["label"] == "good" for r in rows) or not any(r["label"] != "good" for r in rows):
        raise ValueError("Both normal and defective holdout images are required.")
    return rows


def claim_holdout(path, context):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(context, indent=2) + "\n")


def write_new(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")


def checked_outputs(values):
    import torch
    if (not isinstance(values, (tuple, list)) or len(values) != 2
            or tuple(values[0].shape) != (1, 1, 256, 256) or values[1].numel() != 1
            or any(not torch.isfinite(v).all().item() for v in values)):
        raise ValueError("Invalid TorchScript score/map output.")
    return values[0], float(values[1].item())


def verify_export(reference_path=REFERENCE, export_path=EXPORT, parity_path=PARITY, root=ROOT):
    if parity_path.exists():
        raise FileExistsError("Export parity evidence exists; never overwrite it.")
    import torch
    reference = json.loads(reference_path.read_text())
    source, policy, saved = load_calibrated_bundle(reference, "cpu")
    if saved["model_config"].get("category") != "screw":
        raise ValueError("Expected the screw calibration bundle.")
    manifest = Path(saved["model_config"]["data_manifest"])
    if fingerprint(manifest)[1] != saved["model_config"]["data_manifest_sha256"]:
        raise ValueError("Training manifest fingerprint mismatch.")
    document = read_manifest(manifest)
    model = torch.jit.load(str(export_path), map_location="cpu").eval()
    scores = reference["normal_scores"]
    # All threshold normals plus four map-fit normals; never open holdout images here.
    selected = [r for r in scores if r["calibration_subset"] == "threshold"]
    selected += [r for r in scores if r["calibration_subset"] == "map_fit"][:4]
    by_path = {r["path"]: r for r in document["samples"]}
    checks = []
    with torch.inference_mode():
        for known in selected:
            row = by_path[known["path"]]
            if row["role"] not in {"validation_normal", "threshold_normal"}:
                raise ValueError("Export parity cannot read holdout/classifier images.")
            tensor = tensor_for(row, root, "cpu")
            expected = source(tensor, normalize=True)
            actual_map, score = checked_outputs(model(tensor))
            error = float((actual_map - expected.anomaly_map).abs().max())
            score_error = abs(score - float(expected.pred_score.item()))
            calibration_error = abs(score - known["score"])
            candidate = classify_score(score, policy["threshold"])
            if (max(error, score_error, calibration_error) > 1e-6
                    or candidate != classify_score(known["score"], policy["threshold"])):
                raise ValueError("Real-image TorchScript calibration parity failed.")
            checks.append({"path": row["path"], "score": score,
                           "map_max_abs_error": error, "score_abs_error": score_error,
                           "calibration_score_abs_error": calibration_error})
    size, digest = fingerprint(export_path)
    report = {"schema_version": 1, "status": "REAL_CALIBRATION_IMAGE_EXPORT_PARITY_VERIFIED_PC_CPU",
        "artifact": str(export_path), "bytes": size, "sha256": digest,
        "calibration_reference_sha256": fingerprint(reference_path)[1],
        "manifest": str(manifest), "manifest_sha256": fingerprint(manifest)[1],
        "calibrated_checkpoint_sha256": reference["calibrated_checkpoint"]["sha256"],
        "policy_sha256": reference["policy_file_sha256"], "threshold": policy["threshold"],
        "checks": checks, "final_holdout_images_read": 0, "jetson_proof": "NOT_VERIFIED"}
    write_new(parity_path, report)
    return {"status": report["status"], "images": len(checks), "sha256": digest}


def summarize(output=OUTPUT):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    context = json.loads((output / "protocol.json").read_text())
    document = read_manifest(Path(context["manifest"]))
    if fingerprint(Path(context["manifest"]))[1] != context["manifest_sha256"]:
        raise ValueError("Evaluation manifest changed.")
    expected = {r["path"]: r for r in holdout_rows(document)}
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    if len(rows) != len(expected) or {r["source_path"] for r in rows} != set(expected):
        raise ValueError("Incomplete/duplicated saved holdout predictions.")
    maps, masks = [], []
    for row in rows:
        source = expected[row["source_path"]]
        if (row["truth_type"] != source["label"] or row["source_sha256"] != source["sha256"]
                or row["truth_anomalous"] != int(source["label"] != "good")
                or row["candidate"] != classify_score(row["score"], context["threshold"])):
            raise ValueError("Saved labels/scores differ from the frozen protocol.")
        path = output / row["map_file"]
        if fingerprint(path)[1] != row["map_sha256"]:
            raise ValueError("Saved evaluation map changed.")
        with np.load(path, allow_pickle=False) as arrays:
            maps.append(arrays["map_model"].reshape(-1))
            masks.append(arrays["truth_mask_model"].reshape(-1))
    metrics = detection_metrics([r["truth_anomalous"] for r in rows],
                                [r["score"] for r in rows], context["threshold"])
    metrics["pixel_auroc_256_grid"] = float(roc_auc_score(np.concatenate(masks), np.concatenate(maps)))
    by_class = {}
    for label in sorted({r["truth_type"] for r in rows}):
        group = [r for r in rows if r["truth_type"] == label]
        by_class[label] = {"images": len(group), "flagged": sum(r["candidate"] == "ANOMALY_CANDIDATE" for r in group)}
    return {"schema_version": 1, "status": "FROZEN_SCREW_HOLDOUT_EVALUATED_NO_TUNING",
        "protocol": context, "metrics": metrics, "gate_1": acceptance_gate(metrics),
        "per_ground_truth_class": by_class,
        "false_negatives": [r["source_path"] for r in rows if r["truth_anomalous"] and r["candidate"] == "NORMAL_CANDIDATE"],
        "false_positives": [r["source_path"] for r in rows if not r["truth_anomalous"] and r["candidate"] == "ANOMALY_CANDIDATE"],
        "predictions_sha256": fingerprint(output / "predictions.jsonl")[1],
        "pixel_metric_scope": "256x256 model map; source masks resized nearest-neighbor; no pixel threshold tuned",
        "defect_classifier": "NOT_IMPLEMENTED", "jetson_proof": "NOT_VERIFIED",
        "physical_camera_accuracy": "NOT_VERIFIED"}


def evaluate():
    import numpy as np
    import torch
    from PIL import Image
    if OUTPUT.exists() or EXPOSURE.exists():
        raise FileExistsError("Holdout already claimed/output exists; summarize saved predictions instead.")
    parity = json.loads(PARITY.read_text())
    if parity["status"] != "REAL_CALIBRATION_IMAGE_EXPORT_PARITY_VERIFIED_PC_CPU":
        raise ValueError("Export parity is not verified.")
    for path, expected in ((EXPORT, parity["sha256"]), (REFERENCE, parity["calibration_reference_sha256"]),
                           (Path(parity["manifest"]), parity["manifest_sha256"])):
        if fingerprint(path)[1] != expected:
            raise ValueError("Frozen export/calibration/manifest fingerprint mismatch.")
    reference = json.loads(REFERENCE.read_text())
    _, policy, _ = load_calibrated_bundle(reference, "cpu")
    if policy["threshold"] != parity["threshold"]:
        raise ValueError("Export threshold differs from frozen calibration.")
    document = read_manifest(Path(parity["manifest"]))
    rows = holdout_rows(document)
    if len(rows) != 50 or sum(r["label"] == "good" for r in rows) != 21:
        raise ValueError("Expected 21 normal and 29 defective holdout images.")
    model = torch.jit.load(str(EXPORT), map_location="cpu").eval()
    # Recheck a calibration anchor after reload before claiming any holdout exposure.
    anchor = next(r for r in document["samples"] if r["path"] == parity["checks"][0]["path"])
    with torch.inference_mode():
        _, score = checked_outputs(model(tensor_for(anchor, ROOT, "cpu")))
    if abs(score - parity["checks"][0]["score"]) > 1e-6:
        raise ValueError("Reloaded export differs from saved calibration anchor.")
    context = {"manifest": parity["manifest"], "manifest_sha256": parity["manifest_sha256"],
        "model": str(EXPORT), "model_sha256": parity["sha256"],
        "calibration_reference_sha256": fingerprint(REFERENCE)[1],
        "export_verification_sha256": fingerprint(PARITY)[1],
        "policy_sha256": parity["policy_sha256"], "threshold": policy["threshold"],
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": "TorchScript CPU FP32",
        "preprocessing": "source grayscale replicated to RGB, full-frame PIL bilinear resize to 256x256, float [0,1]",
        "source_sha256": fingerprint(Path(__file__))[1], "tune_from_holdout": False,
        "targets": acceptance_gate({})["targets"]}
    claim_holdout(EXPOSURE, {"status": "HOLDOUT_EXPOSED_EVALUATION_STARTED", **context})
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "maps").mkdir()
    write_new(OUTPUT / "protocol.json", context)
    with (OUTPUT / "predictions.jsonl").open("x") as log, torch.inference_mode():
        for index, row in enumerate(rows):
            tensor = tensor_for(row, ROOT, "cpu")
            anomaly_map, score = checked_outputs(model(tensor))
            native = anomaly_map.squeeze().numpy().astype(np.float32)
            if row["mask"] is None:
                mask = np.zeros((256, 256), dtype=bool)
            else:
                with load_verified_image(ROOT, row["mask"]) as truth:
                    mask = np.asarray(truth.resize((256, 256), Image.Resampling.NEAREST)) > 0
            relative = f"maps/{index:03d}.npz"
            np.savez_compressed(OUTPUT / relative, map_model=native, truth_mask_model=mask)
            prediction = {"source_path": row["path"], "source_sha256": row["sha256"],
                "truth_type": row["label"], "truth_anomalous": int(row["label"] != "good"),
                "score": score, "candidate": classify_score(score, policy["threshold"]),
                "map_file": relative, "map_sha256": fingerprint(OUTPUT / relative)[1]}
            log.write(json.dumps(prediction) + "\n")
            log.flush()
    report = summarize()
    write_new(OUTPUT / "evaluation_report.json", report)
    write_new(Path("manifests/screw_efficientad_s_evaluation_v1.json"), report)
    EXPOSURE.write_text(json.dumps({"status": "HOLDOUT_EXPOSED_EVALUATION_COMPLETED", **context,
        "report_sha256": fingerprint(OUTPUT / "evaluation_report.json")[1]}, indent=2) + "\n")
    return {"metrics": report["metrics"], "gate_1": report["gate_1"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-export", action="store_true")
    group.add_argument("--allow-final-holdout", action="store_true")
    group.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.verify_export:
        result = verify_export()
    elif args.allow_final_holdout:
        result = evaluate()
    else:
        report = summarize()
        original = json.loads((OUTPUT / "evaluation_report.json").read_text())
        if report != original:
            raise ValueError("Reproduced report differs from saved evaluation.")
        result = {"status": "CACHED_METRICS_REPRODUCED_WITHOUT_INFERENCE", "metrics": report["metrics"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
