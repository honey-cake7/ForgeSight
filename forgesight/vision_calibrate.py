"""Calibrate fixed map/threshold normal subsets, without reading the final holdout."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from forgesight.data_loader import load_verified_image
from forgesight.data_manifest import DEFAULT_ROOT, read_manifest
from forgesight.edge_assets import fingerprint
from forgesight.vision_checkpoint import load_checkpoint
from forgesight.vision_probe import timing_summary
from forgesight.vision_train import normal_rows, state_digest


def calibration_split(document: dict, policy: dict) -> tuple[list, list]:
    if policy["validation_role"] != "validation_normal" or policy["final_holdout_access"] is not False:
        raise ValueError("Calibration must preserve the frozen normal/holdout boundaries.")
    rows = normal_rows(document, "validation_normal")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate calibration paths are not permitted.")
    count = policy["map_fit_count"]
    if type(count) is not int or not 0 < count < len(rows):
        raise ValueError("Map and threshold subsets must both be nonempty.")
    ordered = sorted(rows, key=lambda row: (
        hashlib.sha256(f'{policy["split_seed"]}:{row["path"]}'.encode()).hexdigest(), row["path"]))
    return ordered[:count], ordered[count:]


def rank_threshold(scores: list[float], alpha: float) -> dict:
    if not scores or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Finite scores and alpha between zero and one are required.")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Nonfinite calibration scores cannot define a threshold.")
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise ValueError("Too few independent normal samples for a finite threshold at this nominal level.")
    return {"threshold": sorted(scores)[rank - 1], "n": n, "order_statistic_rank": rank,
            "nominal_alpha": alpha, "rank_tail_fraction_under_assumptions": (n + 1 - rank) / (n + 1)}


def classify_score(score: float, threshold: float) -> str:
    if not math.isfinite(score) or not math.isfinite(threshold):
        raise ValueError("Nonfinite inference score or threshold is an error, not a normal decision.")
    return "ANOMALY_CANDIDATE" if score > threshold else "NORMAL_CANDIDATE"


def validate_quantiles(values: dict) -> None:
    for branch in ("st", "ae"):
        qa, qb = values[f"qa_{branch}"], values[f"qb_{branch}"]
        if not math.isfinite(qa) or not math.isfinite(qb) or not qb > qa:
            raise ValueError("Each branch needs finite, nondegenerate calibration quantiles.")


def configure_math(recipe: dict) -> None:
    import torch
    torch.set_num_threads(recipe["cpu_threads"])
    torch.backends.cudnn.benchmark = recipe["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = recipe["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = recipe["tf32"]
    torch.backends.cuda.matmul.allow_tf32 = recipe["tf32"]


def tensor_for(row: dict, root: Path, device: str):
    from torchvision.transforms import Compose, Resize, ToTensor
    with load_verified_image(root, row) as image:
        return Compose([Resize((256, 256), antialias=True), ToTensor()])(image).unsqueeze(0).to(device)


def calibrate(args) -> dict:
    import torch

    policy = json.loads(args.policy.read_text())
    if (policy["map_quantiles"] != [0.9, 0.995] or policy["branch_scale"] != 0.1
            or policy["branch_weights"] != [0.5, 0.5] or policy["comparison"] != "strict_greater_than"):
        raise ValueError("Unreviewed upstream map scaling or decision comparison.")
    training = json.loads(args.training_record.read_text())
    if training["status"] != "TRAINED_UNCALIBRATED_NOT_ACCURACY_VERIFIED":
        raise ValueError("Expected the verified original uncalibrated training record.")
    configure_math(training["training_config"])
    source_checkpoint = training["final_checkpoint"]
    model, saved = load_checkpoint(Path(source_checkpoint["path"]), source_checkpoint["sha256"], args.device)
    profile = saved["model_config"]
    manifest_path = Path(profile["data_manifest"])
    if fingerprint(manifest_path)[1] != profile["data_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the frozen training reference.")
    document = read_manifest(manifest_path)
    map_rows, threshold_rows = calibration_split(document, policy)
    if len(map_rows) + len(threshold_rows) != profile["validation_normal_images"]:
        raise ValueError("Calibration partition count differs from the selected profile.")
    before = {name: state_digest(getattr(model, name)) for name in ("teacher", "student", "ae", "mean_std")}
    if any(before[name] != training["state_sha256_final"][name] for name in ("teacher", "student", "ae")):
        raise ValueError("Source checkpoint network states differ from verified training.")
    if model.is_set(model.quantiles):
        raise ValueError("Source checkpoint already has map calibration; preserve the original experiment.")
    cache, quantiles = [], {}
    with torch.inference_mode():
        for row in map_rows + threshold_rows:
            tensor = tensor_for(row, args.root, args.device)
            st, ae = model.get_maps(tensor, normalize=False)
            if not torch.isfinite(st).all().item() or not torch.isfinite(ae).all().item():
                raise ValueError("Nonfinite raw validation map.")
            cache.append((row, tensor, st, ae))
        for index, branch in ((2, "st"), (3, "ae")):
            pixels = torch.cat([entry[index].flatten() for entry in cache[:len(map_rows)]])
            qa, qb = torch.quantile(pixels, torch.tensor(policy["map_quantiles"], device=args.device)).tolist()
            quantiles.update({f"qa_{branch}": qa, f"qb_{branch}": qb})
        validate_quantiles(quantiles)
        for name, value in quantiles.items():
            model.quantiles[name].copy_(torch.tensor(value, device=args.device))
        scores, max_error = [], 0.0
        for index, (row, tensor, st, ae) in enumerate(cache):
            reference = 0.5 * (0.1 * (st - quantiles["qa_st"]) / (quantiles["qb_st"] - quantiles["qa_st"]))
            reference += 0.5 * (0.1 * (ae - quantiles["qa_ae"]) / (quantiles["qb_ae"] - quantiles["qa_ae"]))
            actual = model(tensor, normalize=True)
            error = float((reference - actual.anomaly_map).abs().max().item())
            max_error = max(max_error, error)
            if not torch.isfinite(actual.anomaly_map).all().item() or error > 1e-6:
                raise ValueError("Upstream calibrated maps do not match the recorded normalization.")
            scores.append({"path": row["path"], "source_sha256": row["sha256"],
                           "calibration_subset": "map_fit" if index < len(map_rows) else "threshold",
                           "score": float(actual.pred_score.item())})
        threshold_info = rank_threshold([r["score"] for r in scores if r["calibration_subset"] == "threshold"],
                                        policy["nominal_normal_flag_level"])
        for row in scores:
            row["candidate"] = classify_score(row["score"], threshold_info["threshold"])
        # Model-only timing on an already-used, cached normal image; no threshold tuning.
        anchor = cache[len(map_rows)][1]
        for _ in range(10):
            model(anchor, normalize=True)
        timings = []
        for _ in range(50):
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model(anchor, normalize=True)
            if args.device == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    after = {name: state_digest(getattr(model, name)) for name in before}
    if before != after:
        raise ValueError("Calibration changed trained networks or fitting-only teacher statistics.")
    checkpoint = args.output / "efficientad_s_calibrated.pth"
    torch.save({"format": saved["format"], "step": saved["step"], "model_config": profile,
                "training_config": saved["training_config"], "state_dict": model.state_dict(),
                "calibration": {"quantiles": quantiles, "policy": policy, **threshold_info},
                "source_checkpoint_sha256": source_checkpoint["sha256"]}, checkpoint)
    concrete_policy = {**policy, **threshold_info, "map_quantile_values": quantiles,
                       "calibrated_checkpoint_sha256": fingerprint(checkpoint)[1],
                       "source_checkpoint_sha256": source_checkpoint["sha256"]}
    (args.output / "score_policy.json").write_text(json.dumps(concrete_policy, indent=2) + "\n")
    return {"schema_version": 1, "status": "CALIBRATED_POLICY_FROZEN_HOLDOUT_NOT_EVALUATED",
            "policy": concrete_policy, "model_reference": training["model_config"],
            "dataset_manifest_sha256": profile["data_manifest_sha256"],
            "source_checkpoint": source_checkpoint, "calibrated_checkpoint": {
                "path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": fingerprint(checkpoint)[1]},
            "policy_file_sha256": fingerprint(args.output / "score_policy.json")[1],
            "state_sha256_before": before, "state_sha256_after": after,
            "quantile_pixel_count_per_branch": len(map_rows) * 256 * 256,
            "normal_scores": scores, "upstream_map_max_abs_error": max_error,
            "validation_images_read": len(cache), "map_fit_images": len(map_rows),
            "threshold_images": len(threshold_rows), "final_holdout_images_read": 0,
            "development_images_read": 0, "optimizer_steps": 0,
            "normal_flag_counts_resubstitution_only": {group: sum(r["candidate"] == "ANOMALY_CANDIDATE"
                for r in scores if r["calibration_subset"] == group) for group in ("map_fit", "threshold")},
            "timing": {"device": args.device, "warmup": 10, **timing_summary(timings),
                "raw_ms": timings, "scope": "Cached input, calibrated FP32 model/map/score forward plus synchronization; excludes I/O, resize, transfers, candidate comparison, logs/UI/Llama; one desktop process"},
            "source_sha256": {str(path): fingerprint(path)[1] for path in
                (args.policy, args.training_record, Path(__file__), Path("requirements/pylock.vision.toml"))},
            "false_alarm_generalization": "NOT_VERIFIED", "defect_recall": "NOT_EVALUATED",
            "candidate_flags_are_quality_grades": False, "jetson_proof": "NOT_VERIFIED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("configs/vision_score_policy.json"))
    parser.add_argument("--training-record", type=Path, default=Path("manifests/efficientad_s_training_v1.json"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new calibration output directory.")
    args.output.mkdir(parents=True)
    report = {"status": "FAILED_NOT_VERIFIED", "final_holdout_images_read": 0}
    try:
        report = calibrate(args)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    (args.output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "final_holdout_images_read")}, indent=2))
    if report["status"] == "FAILED_NOT_VERIFIED":
        print(report["error"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
