"""Capture one live C270 frame and save a non-inspection EfficientAD diagnostic."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_frame(path):
    import cv2

    opened_at = time.perf_counter()
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    try:
        if not camera.isOpened():
            raise RuntimeError("C270 /dev/video0 did not open.")
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        open_ms = (time.perf_counter() - opened_at) * 1000
        started = time.perf_counter()
        first_read_ms = None
        frame = None
        brightness = None
        rejected_dark_frames = 0
        for attempt in range(1, 41):
            read_started = time.perf_counter()
            good, candidate = camera.read()
            if first_read_ms is None:
                first_read_ms = (time.perf_counter() - read_started) * 1000
            if not good or candidate is None or candidate.shape != (480, 640, 3):
                raise RuntimeError("C270 did not deliver the requested 640x480 color frame.")
            brightness = float(cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).mean())
            if brightness >= 20.0:
                frame = candidate
                break
            rejected_dark_frames += 1
            time.sleep(0.05)
        if frame is None:
            raise RuntimeError("All 40 camera frames were too dark for this diagnostic; last mean luma %.2f." % brightness)
        warmup_ms = (time.perf_counter() - started) * 1000
        fourcc = int(camera.get(cv2.CAP_PROP_FOURCC))
        format_name = "".join(chr((fourcc >> (8 * i)) & 0xff) for i in range(4))
        encoded, bytes_array = cv2.imencode(".png", frame)
        if not encoded:
            raise RuntimeError("Captured frame could not be encoded as PNG.")
        with path.open("xb") as stream:
            stream.write(bytes_array.tobytes())
        return {"shape_hwc": list(frame.shape), "negotiated_fourcc": format_name,
                "camera_open_ms": open_ms, "first_read_ms": first_read_ms,
                "frames_read": attempt, "rejected_dark_frames": rejected_dark_frames,
                "selected_mean_luma": brightness, "gross_darkness_gate_mean_luma": 20.0,
                "capture_until_usable_ms": warmup_ms, "image_sha256": sha256(path)}
    finally:
        camera.release()


def run_child(argv):
    started = time.perf_counter()
    child = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    if child.returncode:
        raise RuntimeError("Child command failed: " + child.stderr[-2000:])
    return (time.perf_counter() - started) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not math.isfinite(args.threshold):
        parser.error("Output directory must be new and threshold finite.")
    root = args.output_dir.resolve()
    root.mkdir(parents=True)
    started = time.perf_counter()
    image, tensor, map_file = root / "frame.png", root / "input.npy", root / "anomaly_map.npy"
    report = {"schema_version": 1, "status": "ERROR", "mode": "LIVE_CAMERA_DIAGNOSTIC_NO_PART",
              "recorded_utc": datetime.now(timezone.utc).isoformat(), "is_inspection": False,
              "part_id": None, "grade": None, "image": str(image), "map": str(map_file),
              "limitations": "Camera scene has no declared physical nut; score is not a defect or quality finding."}
    try:
        report["capture"] = capture_frame(image)
        script_dir = Path(__file__).resolve().parent
        report["preprocess_process_ms"] = run_child([
            sys.executable, "-B", str(script_dir / "vision_edge_preprocess.py"),
            "--image", str(image), "--tensor", str(tensor)])
        tensor_digest = sha256(tensor)
        report["tensor_sha256"] = tensor_digest
        report["inference_process_ms"] = run_child([
            str(args.vision_python), "-B", str(script_dir / "vision_edge_once.py"),
            "--model", str(args.model), "--model-sha256", args.model_sha256,
            "--tensor", str(tensor), "--tensor-sha256", tensor_digest,
            "--threshold", str(args.threshold), "--output", str(root / "inference.json"),
            "--map-output", str(map_file)])
        inference = json.loads((root / "inference.json").read_text())
        report["inference"] = inference
        import numpy as np
        from PIL import Image

        values = np.load(map_file, allow_pickle=False)
        if values.shape != (256, 256) or not np.isfinite(values).all():
            raise ValueError("Saved anomaly map is invalid.")
        with Image.open(image) as source:
            source = source.convert("RGB")
            resized = Image.fromarray(values, mode="F").resize(source.size, Image.BILINEAR)
            support = (np.asarray(resized) > args.threshold).astype(np.uint8) * 110
            overlay = Image.composite(Image.new("RGB", source.size, "red"),
                                      source, Image.fromarray(support, mode="L"))
            overlay.save(root / "response_overlay.png")
        report["overlay_sha256"] = sha256(root / "response_overlay.png")
        report["map_sha256"] = sha256(map_file)
        report["status"] = "CAMERA_MODEL_DIAGNOSTIC_VERIFIED_NO_PART"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["camera_to_report_ms"] = (time.perf_counter() - started) * 1000
    report["timing_scope"] = "capture open/read, PNG save, preprocessing child, vision child, map and overlay; excludes this program startup and report write"
    with (root / "diagnostic.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    if report["status"] == "ERROR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
