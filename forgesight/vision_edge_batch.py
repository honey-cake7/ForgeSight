"""Score the frozen MVTec metal_nut holdout in one model process.

Run the controller with system Python (Pillow/NumPy) and the worker with the
isolated Torch vision Python. This is a score-only batch benchmark: it does not
measure GUI latency, save anomaly maps, or make physical-part decisions.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time

from forgesight.grade import validate_policy


MODEL_SHA256 = "8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def jetson_identity():
    if platform.system() != "Linux" or platform.machine().lower() != "aarch64":
        return False
    try:
        model = Path("/proc/device-tree/model").read_bytes().decode(errors="replace")
    except OSError:
        return False
    return "Jetson Orin Nano" in model


def worker(request_path):
    import resource
    import numpy as np
    import torch

    request = json.loads(request_path.read_text())
    if sha256(request["model"]) != MODEL_SHA256 or request["model_sha256"] != MODEL_SHA256:
        raise ValueError("Frozen model hash mismatch")
    if not 1 <= request["threads"] <= 4:
        raise ValueError("Expected one to four Torch threads")
    torch.set_num_threads(request["threads"])
    started = time.perf_counter()
    model = torch.jit.load(request["model"], map_location="cpu").eval()
    model_load_ms = (time.perf_counter() - started) * 1000
    rows = []
    with Path(request["predictions"]).open("x", encoding="utf-8") as stream:
        for sample in request["samples"]:
            started = time.perf_counter()
            if sha256(sample["tensor"]) != sample["tensor_sha256"]:
                raise ValueError("Prepared tensor hash mismatch")
            prepared = np.load(sample["tensor"], allow_pickle=False)
            if (prepared.shape != (1, 3, 256, 256) or prepared.dtype != np.float32
                    or not np.isfinite(prepared).all()):
                raise ValueError("Invalid prepared tensor")
            tensor = torch.from_numpy(prepared)
            forward_started = time.perf_counter()
            with torch.inference_mode():
                anomaly_map, score_tensor = model(tensor)
            forward_ms = (time.perf_counter() - forward_started) * 1000
            score = float(score_tensor.item())
            if (anomaly_map.shape != (1, 1, 256, 256) or not torch.isfinite(anomaly_map).all()
                    or not math.isfinite(score)):
                raise ValueError("Invalid model output")
            row = {"source_path": sample["source_path"], "truth_type": sample["truth_type"],
                   "truth_anomalous": sample["truth_anomalous"], "source_sha256": sample["source_sha256"],
                   "score": score,
                   "candidate": "ANOMALY_CANDIDATE" if score > request["threshold"] else "NORMAL_CANDIDATE",
                   "model_forward_ms": forward_ms,
                   "worker_image_ms": (time.perf_counter() - started) * 1000}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
    summary = {"model_load_ms": model_load_ms, "torch_version": torch.__version__,
               "torch_threads": torch.get_num_threads(),
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "completed": len(rows)}
    Path(request["worker_summary"]).write_text(json.dumps(summary, indent=2) + "\n")


def prepare(args):
    import numpy as np
    from PIL import Image

    if args.output.exists():
        raise ValueError("Choose a new batch output directory")
    if not args.vision_python.is_file() or sha256(args.model) != MODEL_SHA256:
        raise ValueError("Vision runtime or frozen model is unavailable")
    policy = json.loads(args.policy.read_text())
    validate_policy(policy)
    if (policy.get("benchmark_upload_scope") != "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD_ONLY"
            or policy["image_threshold"] != 0.22776579856872559):
        raise ValueError("Expected the frozen MVTec benchmark score policy")
    manifest = json.loads(args.manifest.read_text())
    samples = [row for row in manifest["samples"] if row["role"] == "final_holdout"]
    if len(samples) != 111 or len({row["path"] for row in samples}) != 111:
        raise ValueError("Expected 111 unique final-holdout images")
    args.output.mkdir(parents=True)
    (args.output / "tensors").mkdir()
    started = time.perf_counter()
    prepared = []
    for index, sample in enumerate(sorted(samples, key=lambda row: row["path"])):
        relative = PurePosixPath(sample["path"])
        if (relative.is_absolute() or relative.parts[0] != "test" or ".." in relative.parts
                or sample["label"] not in ("good", "bent", "color", "flip", "scratch")):
            raise ValueError("Invalid holdout path or label")
        source = args.dataset_root.joinpath(*relative.parts)
        if not source.is_file() or sha256(source) != sample["sha256"]:
            raise ValueError("Missing or changed holdout image: " + sample["path"])
        with Image.open(source) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (700, 700):
                raise ValueError("Unexpected holdout image format: " + sample["path"])
            resized = image.resize((256, 256), Image.BILINEAR)
            array = np.asarray(resized, dtype=np.uint8)
        tensor = np.transpose(array, (2, 0, 1)).astype(np.float32)[None] / 255.0
        tensor_path = args.output / "tensors" / f"{index:03d}.npy"
        with tensor_path.open("xb") as stream:
            np.save(stream, tensor, allow_pickle=False)
        prepared.append({"source_path": sample["path"], "source_sha256": sample["sha256"],
                         "truth_type": sample["label"], "truth_anomalous": int(sample["label"] != "good"),
                         "tensor": str(tensor_path.resolve()), "tensor_sha256": sha256(tensor_path)})
    preprocess_ms = (time.perf_counter() - started) * 1000
    request = {"model": str(args.model.resolve()), "model_sha256": MODEL_SHA256,
               "manifest_sha256": sha256(args.manifest), "policy_sha256": sha256(args.policy),
               "threshold": policy["image_threshold"], "threads": args.threads,
               "predictions": str((args.output / "predictions.jsonl").resolve()),
               "worker_summary": str((args.output / "worker_summary.json").resolve()),
               "samples": prepared}
    request_path = args.output / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    return request_path, preprocess_ms, policy


def summarize(args, preprocess_ms, process_ms, policy):
    rows = [json.loads(line) for line in (args.output / "predictions.jsonl").read_text().splitlines()]
    if len(rows) != 111 or len({row["source_path"] for row in rows}) != 111:
        raise ValueError("Incomplete or duplicated batch predictions")
    counts = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    grade_counts = {key: 0 for key in ("PASS", "REVIEW", "REJECT")}
    errors = []
    threshold = policy["image_threshold"]
    review_start = threshold * (1 - policy["normal_review_margin_fraction"])
    for row in rows:
        flagged = row["score"] > threshold
        counts[("tp" if row["truth_anomalous"] else "fp") if flagged else
               ("fn" if row["truth_anomalous"] else "tn")] += 1
        grade = "REJECT" if flagged else "REVIEW" if row["score"] >= review_start else "PASS"
        grade_counts[grade] += 1
        if flagged != bool(row["truth_anomalous"]):
            errors.append({"source_path": row["source_path"], "truth_type": row["truth_type"],
                           "score": row["score"], "candidate": row["candidate"], "benchmark_grade": grade})
    worker_summary = json.loads((args.output / "worker_summary.json").read_text())
    report = {"status": "FROZEN_MVTEC_BATCH_SCORE_EVALUATED", "created_utc": datetime.now(timezone.utc).isoformat(),
              "device": platform.platform(), "physical_part_inspected": False,
              "jetson_verified": jetson_identity(),
              "scope": "111 original MVTec holdout PNGs; score-only batch; excludes maps, overlays, SQLite, GUI and Llama",
              "model_sha256": MODEL_SHA256, "manifest_sha256": sha256(args.manifest),
              "policy_sha256": sha256(args.policy), "threshold": threshold, "review_start": review_start,
              "images": len(rows), "confusion": counts, "benchmark_grade_counts": grade_counts,
              "errors": errors, "preprocess_wall_ms": preprocess_ms, "vision_process_wall_ms": process_ms,
              "worker": worker_summary,
              "model_forward_ms": {"p50": percentile([row["model_forward_ms"] for row in rows], .5),
                                   "p95": percentile([row["model_forward_ms"] for row in rows], .95)},
              "worker_image_ms": {"p50": percentile([row["worker_image_ms"] for row in rows], .5),
                                  "p95": percentile([row["worker_image_ms"] for row in rows], .95)},
              "predictions_sha256": sha256(args.output / "predictions.jsonl")}
    if args.pc_predictions is not None:
        old = {row["source_path"]: row for row in
               (json.loads(line) for line in args.pc_predictions.read_text().splitlines())}
        if set(old) != {row["source_path"] for row in rows}:
            raise ValueError("PC reference images do not match the batch")
        report["pc_parity"] = {"reference_sha256": sha256(args.pc_predictions),
                               "max_absolute_score_difference": max(abs(row["score"] - old[row["source_path"]]["score"])
                                                                    for row in rows),
                               "candidate_disagreements": [row["source_path"] for row in rows
                                                           if row["candidate"] != old[row["source_path"]]["candidate"]]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/metal_nut_v1.json"))
    parser.add_argument("--policy", type=Path, default=Path("configs/forgegrade_benchmark_v1.json"))
    parser.add_argument("--vision-python", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--pc-predictions", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.request is None:
            parser.error("Worker request is required")
        worker(args.request)
        return
    if (args.dataset_root is None or args.vision_python is None or args.model is None or args.output is None
            or not 1 <= args.threads <= 4):
        parser.error("Dataset root, vision Python, model, output and 1-4 threads are required")
    started = time.perf_counter()
    request, preprocess_ms, policy = prepare(args)
    command = [str(args.vision_python), "-B", "-m", "forgesight.vision_edge_batch",
               "--worker", "--request", str(request)]
    process_started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    process_ms = (time.perf_counter() - process_started) * 1000
    if result.returncode:
        raise RuntimeError("Vision batch failed: " + result.stderr[-2000:])
    report = summarize(args, preprocess_ms, process_ms, policy)
    report["controller_total_wall_ms"] = (time.perf_counter() - started) * 1000
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("images", "confusion", "benchmark_grade_counts",
                                                     "preprocess_wall_ms", "vision_process_wall_ms",
                                                     "model_forward_ms", "worker_image_ms", "pc_parity") if key in report}, indent=2))


if __name__ == "__main__":
    main()
