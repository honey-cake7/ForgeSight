"""Run one hash-checked frozen EfficientAD TorchScript inference on a prepared image."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--tensor-sha256", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-output", type=Path)
    parser.add_argument("--source-map-output", type=Path)
    parser.add_argument("--source-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file.")
    if args.map_output is not None and args.map_output.exists():
        parser.error("Choose a new anomaly-map file.")
    if (args.source_map_output is None) != (args.source_size is None):
        parser.error("Source-map output and source size must be supplied together.")
    if args.source_map_output is not None and (args.source_map_output.exists()
            or min(args.source_size) <= 0 or max(args.source_size) > 4096):
        parser.error("Choose a new source-map file and a bounded positive size.")
    if sha256(args.model) != args.model_sha256 or sha256(args.tensor) != args.tensor_sha256:
        parser.error("Model or tensor SHA-256 mismatch.")
    if not math.isfinite(args.threshold):
        parser.error("Threshold must be finite.")

    import numpy as np
    import torch

    prepared = np.load(args.tensor, allow_pickle=False)
    if prepared.shape != (1, 3, 256, 256) or prepared.dtype != np.float32 or not np.isfinite(prepared).all():
        parser.error("Expected one finite FP32 RGB 256x256 tensor.")
    torch.set_num_threads(2)
    started = time.perf_counter()
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    load_ms = (time.perf_counter() - started) * 1000
    tensor = torch.from_numpy(prepared)
    started = time.perf_counter()
    with torch.inference_mode():
        anomaly_map, score_tensor = model(tensor)
    forward_ms = (time.perf_counter() - started) * 1000
    score = float(score_tensor.item())
    if anomaly_map.shape != (1, 1, 256, 256) or not math.isfinite(score) or not torch.isfinite(anomaly_map).all():
        raise ValueError("Invalid model output.")
    report = {"status": "ONE_IMAGE_INFERENCE_VERIFIED", "runtime": "TorchScript CPU FP32",
              "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
              "model_sha256": args.model_sha256, "tensor_sha256": args.tensor_sha256,
              "model_load_ms": load_ms, "model_forward_ms": forward_ms,
              "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "score": score, "threshold": args.threshold,
              "candidate": "ANOMALY_CANDIDATE" if score > args.threshold else "NORMAL_CANDIDATE",
              "map_min": float(anomaly_map.min()), "map_max": float(anomaly_map.max()),
              "map_shape": list(anomaly_map.shape),
              "timing_scope": "model load and first forward only; excludes image decode, preprocessing, process startup, UI, logging and Llama"}
    if args.map_output is not None:
        args.map_output.parent.mkdir(parents=True, exist_ok=True)
        with args.map_output.open("xb") as stream:
            np.save(stream, anomaly_map.squeeze().numpy().astype(np.float32), allow_pickle=False)
        report["map_file"] = str(args.map_output)
        report["map_sha256"] = sha256(args.map_output)
    if args.source_map_output is not None:
        from torch.nn import functional as F

        resized = F.interpolate(anomaly_map, size=tuple(args.source_size),
                                mode="bilinear", align_corners=False)
        args.source_map_output.parent.mkdir(parents=True, exist_ok=True)
        with args.source_map_output.open("xb") as stream:
            np.save(stream, resized.squeeze().numpy().astype(np.float32), allow_pickle=False)
        report["source_map_file"] = str(args.source_map_output)
        report["source_map_sha256"] = sha256(args.source_map_output)
        report["source_map_shape"] = list(resized.shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
