"""Persist one Jetson EfficientAD saved-image replay with ForgeGrade and SQLite."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from forgesight.grade import grade_record, validate_policy
from forgesight.regions import response_regions
from forgesight.storage import InspectionStore


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_child(argv):
    started = time.perf_counter()
    child = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    if child.returncode:
        raise RuntimeError("Child command failed: " + child.stderr[-2000:])
    return (time.perf_counter() - started) * 1000


def select_replay_sample(samples, image_sha256, image_path, allow_fit_reference=False):
    allowed_roles = {"development_preview"}
    if allow_fit_reference:
        allowed_roles.add("fit")
    matches = [row for row in samples
               if row["sha256"] == image_sha256 and row["role"] in allowed_roles
               and (row["role"] != "fit" or row["path"] == "train/good/000.png")
               and Path(image_path).as_posix().endswith(row["path"])]
    if len(matches) != 1:
        raise ValueError("Image is not a permitted, uniquely identified replay sample.")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--part-id", required=True, help="Unique replay ID, not a physical part identity")
    parser.add_argument("--vision-python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--grade-policy", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-fit-reference", action="store_true",
                        help="Permit only train/good/000.png as a normal runtime reference; never accuracy evidence")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new replay output directory.")
    if not args.image.is_file() or args.image.stat().st_size > 32 * 1024 * 1024:
        parser.error("Expected a bounded existing saved image.")
    if sha256(args.image) != args.image_sha256:
        parser.error("Saved image SHA-256 differs from the reviewed dataset manifest.")
    dataset = json.loads(args.dataset_manifest.read_text())
    try:
        sample = select_replay_sample(dataset["samples"], args.image_sha256,
                                      args.image, args.allow_fit_reference)
    except ValueError as error:
        parser.error(str(error))
    if args.model_sha256 != "8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd":
        parser.error("Expected the verified frozen EfficientAD edge export.")
    policy = json.loads(args.grade_policy.read_text())
    validate_policy(policy)
    if (policy["checkpoint_sha256"] != "aa2407dba9f74db8472ee6adc5968eefc7c243d9f200d70f2d3a792fd0b7f4fd"
            or policy["score_policy_sha256"] != "33a28c23328c63abf93fa0e01d270eda52f4a0195f7b8df89c14dda44e77a687"):
        parser.error("Grade policy does not match the frozen calibrated model.")
    with Image.open(args.image) as source:
        if source.mode != "RGB" or source.size != (700, 700) or source.format != "PNG":
            parser.error("Expected a 700x700 RGB benchmark PNG.")
        source.load()
    store = InspectionStore(args.database)
    if store.has_part(args.part_id):
        parser.error("Replay ID already recorded; no duplicate inference performed.")

    root = args.output_dir.resolve()
    root.mkdir(parents=True)
    started = time.perf_counter()
    try:
        device_model = Path("/proc/device-tree/model").read_bytes().decode(errors="replace").strip("\x00\n")
    except OSError:
        device_model = ""
    jetson = platform.machine().lower() == "aarch64" and "Jetson Orin Nano" in device_model
    script_dir = Path(__file__).resolve().parent
    tensor, model_map, source_map = root / "input.npy", root / "map_model.npy", root / "map_source.npy"
    record = {
        "schema_version": 1, "inspection_id": str(uuid4()), "part_id": args.part_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.image.resolve()), "mode": "SAVED_IMAGE_REPLAY",
                   "dataset_role": sample["role"], "dataset_path": sample["path"],
                   "physical_part": False,
                   "sha256": args.image_sha256, "size_px": [700, 700]},
        "model": {"name": "EfficientAD-S", "input_size_px": [256, 256], "precision": "fp32",
                  "checkpoint_sha256": policy["checkpoint_sha256"],
                  "export_sha256": args.model_sha256,
                  "score_policy_sha256": policy["score_policy_sha256"],
                  "image_threshold": policy["image_threshold"],
                  "runtime_device": "jetson_cpu" if jetson else "non_jetson_cpu",
                  "jetson_verified": jetson},
        "status": "ERROR", "score": None, "candidate": None,
        "predicted_defect_type": "unknown", "type_status": "NO_VALIDATED_SUBTYPE_CLASSIFIER",
        "geometry": None, "decision": None, "advice_status": "NOT_REQUESTED_STAGE_9_PENDING",
        "error": None, "artifacts": {}, "timings": {}
    }
    try:
        record["timings"]["preprocess_process_ms"] = run_child([
            sys.executable, "-B", str(script_dir / "vision_edge_preprocess.py"),
            "--image", str(args.image), "--tensor", str(tensor)])
        record["artifacts"]["input_tensor_sha256"] = sha256(tensor)
        record["timings"]["vision_process_ms"] = run_child([
            str(args.vision_python), "-B", str(script_dir / "vision_edge_once.py"),
            "--model", str(args.model), "--model-sha256", args.model_sha256,
            "--tensor", str(tensor), "--tensor-sha256", sha256(tensor),
            "--threshold", str(policy["image_threshold"]),
            "--output", str(root / "inference.json"),
            "--map-output", str(model_map),
            "--source-map-output", str(source_map), "--source-size", "700", "700"])
        inference = json.loads((root / "inference.json").read_text())
        if (inference["model_sha256"] != args.model_sha256
                or inference["tensor_sha256"] != sha256(tensor)
                or inference["threshold"] != policy["image_threshold"]):
            raise ValueError("Inconsistent model inference evidence.")
        model_values = np.load(model_map, allow_pickle=False)
        source_values = np.load(source_map, allow_pickle=False)
        if (model_values.shape != (256, 256) or source_values.shape != (700, 700)
                or not np.isfinite(model_values).all() or not np.isfinite(source_values).all()):
            raise ValueError("Invalid saved anomaly maps.")
        record["score"] = inference["score"]
        record["candidate"] = inference["candidate"]
        record["geometry"] = response_regions(source_values, policy["image_threshold"])
        record["artifacts"].update(score_map_model=str(model_map),
                                   score_map_model_sha256=sha256(model_map),
                                   score_map_source=str(source_map),
                                   score_map_source_sha256=sha256(source_map))
        with Image.open(args.image) as source:
            support = Image.fromarray((source_values > policy["image_threshold"]).astype("uint8") * 110)
            overlay = Image.composite(Image.new("RGB", source.size, "red"), source, support)
            draw = ImageDraw.Draw(overlay)
            for region in record["geometry"]["regions"]:
                x, y, width, height = region["bbox_xywh_px"]
                draw.rectangle([x, y, x + width - 1, y + height - 1], outline="yellow", width=2)
            overlay.save(root / "response_overlay.png")
        record["artifacts"]["response_overlay"] = str(root / "response_overlay.png")
        record["artifacts"]["response_overlay_sha256"] = sha256(root / "response_overlay.png")
        record["timings"]["model_forward_ms"] = inference["model_forward_ms"]
        record["status"] = "ANALYZED"
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["decision"] = grade_record(record, policy)
    record["timings"]["replay_before_record_write_ms"] = (time.perf_counter() - started) * 1000
    record["timings"]["scope"] = "preprocessing, child startup/model inference, maps, geometry, grading, overlay; excludes JSON/SQLite writes and Llama"
    with (root / "record.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    store.save_inspection(record)
    report = {"status": "PERSISTED_SAVED_IMAGE_REPLAY" if record["status"] == "ANALYZED" else "PERSISTED_REPLAY_ERROR",
              "inspection_id": record["inspection_id"], "part_id": args.part_id,
              "image_sha256": args.image_sha256, "score": record["score"],
              "candidate": record["candidate"], "grade": record["decision"]["grade"],
              "reason": record["decision"]["reason"], "model_forward_ms": record["timings"].get("model_forward_ms"),
              "replay_to_sqlite_commit_ms": (time.perf_counter() - started) * 1000,
              "record_sha256": sha256(root / "record.json"),
              "database": str(args.database.resolve()), "source_mode": "SAVED_IMAGE_REPLAY",
              "physical_part_inspected": False, "device_model": device_model,
              "jetson_identity_checked": jetson}
    with (root / "run_report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    if record["status"] == "ERROR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
