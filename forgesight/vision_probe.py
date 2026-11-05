"""Time untrained model architectures on synthetic inputs; never evaluate defect accuracy."""

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import statistics
import time
import traceback


def timing_summary(values: list[float]) -> dict:
    if len(values) < 20 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Timing needs at least 20 positive finite samples.")
    return {"samples": len(values), "p50_ms": statistics.median(values),
            "p95_ms": statistics.quantiles(values, n=100, method="inclusive")[94],
            "min_ms": min(values), "max_ms": max(values)}


def estimated_bank_size(fit_count: int, patches_per_image: int, ratio: float) -> int:
    if fit_count <= 0 or patches_per_image <= 0 or not 0 < ratio <= 1:
        raise ValueError("Invalid synthetic memory-bank sizing inputs.")
    return max(1, int(fit_count * patches_per_image * ratio))


def measure_case(spec: dict, device: str, fit_count: int, warmup: int, samples: int) -> dict:
    import torch
    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.empty_cache()
    result = {"case": spec, "device": device, "accuracy": "NOT_EVALUATED",
              "status": "RUNTIME_PROBE_PASSED_NOT_TRAINED", "batch_size": 1}
    if spec["model"] == "efficientad":
        size = EfficientAdModelSize.S if spec["size"] == "small" else EfficientAdModelSize.M
        model = EfficientAdModel(model_size=size, padding=False, pad_maps=True)
    else:
        model = PatchcoreModel(layers=["layer2", "layer3"], backbone=spec["backbone"],
                               pre_trained=False, num_neighbors=9)
    model = model.eval().to(device)
    side = spec["resolution"]
    tensor = torch.rand(1, 3, side, side, device=device)
    if spec["model"] == "patchcore":
        with torch.inference_mode():
            features = {name: model.feature_pooler(value)
                        for name, value in model.feature_extractor(tensor).items()}
            embedding = model.reshape_embedding(model.generate_embedding(features))
            rows, dimensions = embedding.shape
            bank_rows = spec.get("bank_rows") or estimated_bank_size(fit_count, rows, 0.1)
            model.memory_bank = torch.randn(bank_rows, dimensions, device=device)
            result["synthetic_bank"] = {"rows": bank_rows, "dimensions": dimensions,
                                        "patches_per_image": rows,
                                        "bytes": model.memory_bank.numel() * model.memory_bank.element_size(),
                                        "note": "Random values; no fitting or coreset selection performed."}
            del features, embedding
    result["parameters"] = sum(p.numel() for p in model.parameters())
    result["parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
    source = Path(inspect.getfile(type(model)))
    result["implementation_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()

    def forward():
        with torch.autocast(device_type=device, dtype=torch.float16,
                            enabled=spec["precision"] == "amp_fp16"):
            return model(tensor)

    with torch.inference_mode():
        output = forward()
        if (tuple(output.anomaly_map.shape[-2:]) != (side, side)
                or output.pred_score.numel() != 1
                or not torch.isfinite(output.anomaly_map).all().item()
                or not torch.isfinite(output.pred_score).all().item()):
            raise ValueError("Invalid/nonfinite architecture-probe outputs.")
        result["output_shape"] = list(output.anomaly_map.shape)
        del output
        for _ in range(warmup):
            forward()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated_before = torch.cuda.memory_allocated()
        wall, gpu = [], []
        for _ in range(samples):
            if device == "cuda":
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            if device == "cuda":
                begin.record()
            output = forward()
            if device == "cuda":
                end.record()
                end.synchronize()
            wall.append((time.perf_counter() - start) * 1000)
            if device == "cuda":
                gpu.append(begin.elapsed_time(end))
            del output
        result["wall_model_only"] = timing_summary(wall)
        result["raw_wall_ms"] = wall
        if device == "cuda":
            result["cuda_event_model_only"] = timing_summary(gpu)
            result["raw_cuda_ms"] = gpu
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_incremental_peak_bytes"] = torch.cuda.max_memory_allocated() - allocated_before
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--amp-only", action="store_true", help="Compare CUDA autocast compatibility without repeating FP32/CPU probes.")
    args = parser.parse_args()
    if args.output.exists() or not 20 <= args.samples <= 200 or not 1 <= args.warmup <= 50:
        parser.error("Use a new output directory, 20-200 samples and 1-50 warmups.")
    if not 1 <= args.cpu_threads <= 16:
        parser.error("Use 1-16 CPU threads.")
    args.output.mkdir(parents=True)
    import torch

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = json.loads(Path("manifests/metal_nut_v1.json").read_text())
    report = {"status": "UNTRAINED_ARCHITECTURE_TIMINGS_ONLY", "accuracy": "NOT_EVALUATED",
              "dataset_images_read": 0, "holdout_images_read": 0, "optimizer_steps": 0,
              "weights": "RANDOM_INITIALIZATION_NO_PRETRAINED_DOWNLOADS",
              "input": "Synthetic RGB uniform random, device-resident batch 1",
              "scope": "Forward including native anomaly-map/score calculations; excludes I/O, resize, H2D/D2H, camera, grading, logs, UI and Llama contention",
              "jetson_latency": "NOT_VERIFIED", "latency_target_ms": None,
              "cpu_threads": args.cpu_threads, "warmup": args.warmup, "samples": args.samples,
              "tf32": False, "cudnn_benchmark": False, "seed": 42,
              "fit_count_for_bank_estimate": manifest["counts"]["fit"],
              "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "host": platform.platform(), "python": platform.python_version(),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "torchvision", "anomalib", "timm", "numpy", "lightning")},
              "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "results": []}
    specs = [
        {"model": "efficientad", "size": "small", "resolution": 256, "precision": "fp32"},
        {"model": "efficientad", "size": "small", "resolution": 384, "precision": "fp32"},
        {"model": "efficientad", "size": "medium", "resolution": 256, "precision": "fp32"},
        {"model": "patchcore", "backbone": "resnet18", "bank_rows": 1000,
         "resolution": 256, "precision": "fp32"},
        {"model": "patchcore", "backbone": "resnet18", "bank_rows": None,
         "resolution": 256, "precision": "fp32"},
        {"model": "patchcore", "backbone": "wide_resnet50_2", "bank_rows": None,
         "resolution": 256, "precision": "fp32"},
    ]
    amp_specs = [dict(spec, precision="amp_fp16") for spec in specs
                 if spec["resolution"] == 256 and (spec["model"] == "patchcore" or spec.get("size") == "small")]
    if args.amp_only:
        if not torch.cuda.is_available():
            parser.error("AMP comparison requires the actual CUDA GPU.")
        specs = amp_specs
    elif torch.cuda.is_available():
        specs.extend(amp_specs)
    for spec in specs:
        devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        if spec["precision"] == "amp_fp16":
            devices = ["cuda"]
        for device in devices:
            try:
                result = measure_case(spec, device, manifest["counts"]["fit"], args.warmup, args.samples)
                print(json.dumps({"case": spec, "device": device, "latency": result["wall_model_only"]}), flush=True)
            except Exception as error:
                result = {"case": spec, "device": device, "status": "FAILED_NOT_VERIFIED",
                          "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
                print(json.dumps(result), flush=True)
            report["results"].append(result)
            (args.output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if any(result["status"] == "FAILED_NOT_VERIFIED" for result in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
