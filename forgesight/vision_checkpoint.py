"""Load a fingerprinted tensor-only EfficientAD checkpoint for later calibration."""

import importlib.metadata
import argparse
import json
import math
from pathlib import Path

from forgesight.edge_assets import fingerprint


def validate_checkpoint_metadata(saved: dict) -> None:
    if saved.get("format") != "forgesight-efficientad-v1" or not 0 < saved.get("step", 0) <= 70000:
        raise ValueError("Unknown checkpoint format or invalid completed step count.")
    profile = saved["model_config"]
    if (profile["algorithm"] != "EfficientAD" or profile["model_size"] != "small"
            or profile["teacher_out_channels"] != 384 or profile["input"]["resolution"] != [256, 256]):
        raise ValueError("Checkpoint architecture differs from the selected baseline.")
    if saved["training_config"]["holdout_access"] is not False:
        raise ValueError("Checkpoint does not preserve the frozen holdout boundary.")


def load_checkpoint(path: Path, expected_sha256: str, device: str = "cpu"):
    if fingerprint(path)[1] != expected_sha256:
        raise ValueError("Checkpoint fingerprint mismatch.")
    import torch
    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize

    saved = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_metadata(saved)
    profile = saved["model_config"]
    if importlib.metadata.version("anomalib") != profile["implementation_version"]:
        raise ValueError("Checkpoint requires the pinned Anomalib environment.")
    model = EfficientAdModel(teacher_out_channels=384, model_size=EfficientAdModelSize.S,
                            padding=profile["padding"], pad_maps=profile["pad_maps"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device), saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new reload-evidence file.")
    report = {"status": "FAILED_NOT_VERIFIED", "accuracy": "NOT_EVALUATED", "holdout_images_read": 0}
    try:
        import torch
        from torchvision.transforms import Compose, Resize, ToTensor
        from forgesight.data_manifest import DEFAULT_ROOT, read_manifest
        from forgesight.data_loader import load_verified_image
        from forgesight.vision_train import normal_rows, state_digest

        training = json.loads(args.training_report.read_text())
        if (training["status"] != "TRAINED_UNCALIBRATED_NOT_ACCURACY_VERIFIED"
                or training["completed_steps"] != training["training_config"]["steps"]):
            raise ValueError("Training did not finish the planned budget successfully.")
        recipe = training["training_config"]
        torch.set_num_threads(recipe["cpu_threads"])
        torch.backends.cudnn.benchmark = recipe["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = recipe["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = recipe["tf32"]
        torch.backends.cuda.matmul.allow_tf32 = recipe["tf32"]
        checkpoint = training["checkpoints"][-1]
        model, saved = load_checkpoint(Path(checkpoint["path"]), checkpoint["sha256"], args.device)
        if saved["step"] != training["completed_steps"]:
            raise ValueError("Checkpoint step differs from completed training budget.")
        hashes = {name: state_digest(getattr(model, name)) for name in ("teacher", "student", "ae")}
        if hashes != training["state_sha256_final"]:
            raise ValueError("Reloaded network states differ from the actual trained model.")
        row = normal_rows(read_manifest(Path(saved["model_config"]["data_manifest"])), "fit")[0]
        with load_verified_image(args.root or DEFAULT_ROOT, row) as image:
            tensor = Compose([Resize((256, 256), antialias=True), ToTensor()])(image).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            result = model(tensor, normalize=False)
        score = float(result.pred_score.item())
        if (not torch.isfinite(result.anomaly_map).all().item()
                or not math.isclose(score, training["fit_anchor_raw_score"], rel_tol=1e-4, abs_tol=1e-6)):
            raise ValueError("Reload output is nonfinite or differs from the saved fit-anchor result.")
        report.update(status="CHECKPOINT_RELOAD_VERIFIED_NOT_ACCURACY_EVALUATED", step=saved["step"],
                      checkpoint=checkpoint, device=args.device, state_sha256=hashes,
                      fit_anchor=row["path"], raw_score=score,
                      score_difference=abs(score - training["fit_anchor_raw_score"]),
                      map_shape=list(result.anomaly_map.shape), map_calibration="NOT_RUN",
                      decision_threshold="NOT_SET", jetson_proof="NOT_VERIFIED")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    if report["status"] == "FAILED_NOT_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
