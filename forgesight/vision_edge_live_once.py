"""Inspect one operator-declared C270 frame on Jetson; keep camera claims provisional."""

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


MODEL_SHA256 = "8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def child(argv):
    started = time.perf_counter()
    result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return (time.perf_counter() - started) * 1000


def inspect(args):
    if args.output_dir.exists():
        raise ValueError("Choose a new output directory for each inspection.")
    if not args.image.is_file() or args.image.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Expected a bounded camera PNG.")
    with Image.open(args.image) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("Expected an RGB PNG.")
        width, height = image.size
        if args.uploaded_image_diagnostic:
            if not 64 <= width <= 1024 or not 64 <= height <= 1024:
                raise ValueError("Expected an uploaded RGB PNG between 64 and 1024 pixels per side.")
        elif image.size != (640, 480):
            raise ValueError("Expected a 640x480 RGB camera PNG.")
        image.load()
    if args.model_sha256 != MODEL_SHA256 or sha256(args.model) != MODEL_SHA256:
        raise ValueError("Frozen EfficientAD export hash mismatch.")
    policy = json.loads(args.grade_policy.read_text())
    validate_policy(policy)
    store = InspectionStore(args.database)
    if store.has_part(args.part_id):
        raise ValueError("Part ID was already inspected; move to a distinct part and use a new ID.")

    root = args.output_dir.resolve()
    root.mkdir(parents=True)
    started = time.perf_counter()
    try:
        device_model = Path("/proc/device-tree/model").read_bytes().decode(errors="replace").strip("\x00\n")
    except OSError:
        device_model = ""
    jetson = platform.machine().lower() == "aarch64" and "Jetson Orin Nano" in device_model
    source_hash = sha256(args.image)
    record = {
        "schema_version": 1, "inspection_id": str(uuid4()), "part_id": args.part_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.image.resolve()),
                   "mode": ("USER_UPLOADED_IMAGE_DIAGNOSTIC" if args.uploaded_image_diagnostic else
                            "LIVE_CAMERA" if args.operator_declared_part else
                            "LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC" if args.screen_photo_diagnostic else
                            "LIVE_CAMERA_DIAGNOSTIC_NO_PART"),
                   "physical_part": "OPERATOR_DECLARED_UNVERIFIED" if args.operator_declared_part else False,
                   "sha256": source_hash,
                   "size_px": [width, height],
                   "camera_index": None if args.uploaded_image_diagnostic else args.camera_index},
        "model": {"name": "EfficientAD-S", "input_size_px": [256, 256], "precision": "fp32",
                  "checkpoint_sha256": policy["checkpoint_sha256"],
                  "export_sha256": args.model_sha256,
                  "score_policy_sha256": policy["score_policy_sha256"],
                  "image_threshold": policy["image_threshold"],
                  "runtime_device": "jetson_cpu" if jetson else "non_jetson_cpu",
                  "jetson_verified": jetson},
        "status": "ERROR", "score": None, "candidate": None,
        "predicted_defect_type": "unknown", "type_status": "NO_VALIDATED_SUBTYPE_CLASSIFIER",
        "geometry": None, "decision": None, "advice_status": "NOT_REQUESTED",
        "error": None, "artifacts": {}, "timings": {}
    }
    try:
        script_dir = Path(__file__).resolve().parent
        tensor, model_map, source_map = root / "input.npy", root / "map_model.npy", root / "map_source.npy"
        record["timings"]["preprocess_process_ms"] = child([
            sys.executable, "-B", str(script_dir / "vision_edge_preprocess.py"),
            "--image", str(args.image), "--tensor", str(tensor)])
        tensor_hash = sha256(tensor)
        record["artifacts"]["input_tensor_sha256"] = tensor_hash
        record["timings"]["vision_process_ms"] = child([
            str(args.vision_python), "-B", str(script_dir / "vision_edge_once.py"),
            "--model", str(args.model), "--model-sha256", args.model_sha256,
            "--tensor", str(tensor), "--tensor-sha256", tensor_hash,
            "--threshold", str(policy["image_threshold"]),
            "--output", str(root / "inference.json"), "--map-output", str(model_map),
            "--source-map-output", str(source_map), "--source-size", str(height), str(width)])
        inference = json.loads((root / "inference.json").read_text())
        if (inference["model_sha256"] != args.model_sha256
                or inference["tensor_sha256"] != tensor_hash
                or inference["threshold"] != policy["image_threshold"]):
            raise ValueError("Inconsistent inference evidence.")
        values = np.load(source_map, allow_pickle=False)
        if values.shape != (height, width) or not np.isfinite(values).all():
            raise ValueError("Invalid source anomaly map.")
        record["score"] = inference["score"]
        record["candidate"] = inference["candidate"]
        record["geometry"] = response_regions(values, policy["image_threshold"])
        record["artifacts"].update({"score_map_model": str(model_map),
                                    "score_map_model_sha256": sha256(model_map),
                                    "score_map_source": str(source_map),
                                    "score_map_source_sha256": sha256(source_map)})
        with Image.open(args.image) as image:
            source = image.convert("RGB")
            support = Image.fromarray((values > policy["image_threshold"]).astype("uint8") * 110)
            overlay = Image.composite(Image.new("RGB", source.size, "red"), source, support)
            draw = ImageDraw.Draw(overlay)
            for region in record["geometry"]["regions"]:
                x, y, width, height = region["bbox_xywh_px"]
                draw.rectangle([x, y, x + width - 1, y + height - 1], outline="yellow", width=2)
            overlay.save(root / "response_overlay.png")
        record["artifacts"]["response_overlay"] = str(root / "response_overlay.png")
        record["artifacts"]["response_overlay_sha256"] = sha256(root / "response_overlay.png")
        record["timings"]["model_forward_ms"] = inference["model_forward_ms"]
        record["timings"]["model_peak_rss_kib"] = inference["process_peak_rss_kib"]
        record["status"] = "ANALYZED"
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["decision"] = grade_record(record, policy)
    record["timings"]["capture_file_to_record_ms"] = (time.perf_counter() - started) * 1000
    record["timings"]["scope"] = "saved frame preprocessing, model process, maps, grade and overlay; excludes camera preview and SQLite write"
    with (root / "record.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    store.save_inspection(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--part-id", required=True)
    parser.add_argument("--vision-python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--grade-policy", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    declaration = parser.add_mutually_exclusive_group(required=True)
    declaration.add_argument("--operator-declared-part", action="store_true")
    declaration.add_argument("--screen-photo-diagnostic", action="store_true")
    declaration.add_argument("--diagnostic-no-part", action="store_true")
    declaration.add_argument("--uploaded-image-diagnostic", action="store_true")
    args = parser.parse_args()
    record = inspect(args)
    print(json.dumps({"inspection_id": record["inspection_id"], "status": record["status"],
                      "grade": record["decision"]["grade"], "score": record["score"],
                      "record": str(args.output_dir / "record.json")}, allow_nan=False))
    if record["status"] != "ANALYZED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
