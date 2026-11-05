"""Train the pinned EfficientAD core on frozen normals, without evaluating holdout."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import random
from pathlib import Path
import time
import traceback

from forgesight.data_loader import load_verified_image
from forgesight.data_manifest import DEFAULT_ROOT, read_manifest
from forgesight.edge_assets import fingerprint


def normal_rows(document: dict, role: str) -> list[dict]:
    if role not in {"fit", "validation_normal"}:
        raise PermissionError("Training adapter permits frozen fit/normal-validation roles only.")
    rows = [row for row in document["samples"] if row["role"] == role]
    if not rows or len(rows) != document["counts"][role]:
        raise ValueError("Frozen role count is missing or inconsistent.")
    for row in rows:
        if (row["label"] != "good" or row["source_split"] != "train" or row["mask"] is not None
                or not row["path"].startswith("train/good/")):
            raise ValueError("Non-training/abnormal data cannot enter this adapter.")
    return rows


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def state_digest(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_recipe(recipe: dict) -> None:
    if (recipe["fit_role"] != "fit" or recipe["normal_validation_role"] != "validation_normal"
            or recipe["holdout_access"] is not False or recipe["validation_pixel_reads_this_step"] != 0):
        raise ValueError("Training recipe cannot change frozen data access boundaries.")
    if (recipe["batch_size"] != 1 or recipe["precision"] != "fp32"
            or not 100 <= recipe["steps"] <= 70000 or not 0 < recipe["learning_rate"] < 1
            or not 0 < recipe["lr_decay_fraction"] < 1):
        raise ValueError("Unreviewed training budget, batch size, precision or optimizer.")


def train(args, report: dict) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import CenterCrop, Compose, RandomGrayscale, Resize, ToTensor
    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize

    profile = json.loads(args.profile.read_text())
    recipe = json.loads(args.recipe.read_text())
    validate_recipe(recipe)
    if importlib.metadata.version("anomalib") != profile["implementation_version"]:
        raise ValueError("Use the pinned vision environment.")
    if (profile["algorithm"] != "EfficientAD" or profile["model_size"] != "small"
            or profile["input"]["resolution"] != [256, 256] or profile["input"]["external_normalization"]):
        raise ValueError("This trainer implements the reviewed small/256 baseline only.")
    if not torch.cuda.is_available():
        raise RuntimeError("The planned FP32 training requires the actual desktop CUDA GPU.")
    document = read_manifest(Path(profile["data_manifest"]))
    if hashlib.sha256(Path(profile["data_manifest"]).read_bytes()).hexdigest() != profile["data_manifest_sha256"]:
        raise ValueError("Selected training manifest differs from frozen reference.")
    fit = normal_rows(document, "fit")
    validation = normal_rows(document, "validation_normal")  # Metadata only; no validation pixels.
    if len(fit) != profile["fit_normal_images"] or len(validation) != profile["validation_normal_images"]:
        raise ValueError("Selected role counts differ from baseline configuration.")
    assets = json.loads(args.assets.read_text())["assets"]
    for key in ("pretrained_teacher_archive", "regularization_archive"):
        record = assets[key]
        size, digest = fingerprint(Path(record["archive_path"]))
        if (size != record["archive_bytes"] or digest != profile[key]["expected_sha256_from_upstream"]
                or digest != record["archive_sha256"]):
            raise ValueError("Training archive differs from verified publisher fingerprint.")
    teachers = list(Path(assets["pretrained_teacher_archive"]["extracted_root"]).rglob("pretrained_teacher_small.pth"))
    aux_roots = list(Path(assets["regularization_archive"]["extracted_root"]).glob("imagenette2/train"))
    if len(teachers) != 1 or len(aux_roots) != 1:
        raise ValueError("Expected exactly one small teacher and Imagenette training root.")
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    torch.set_num_threads(recipe["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = recipe["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = recipe["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = recipe["tf32"]
    torch.backends.cuda.matmul.allow_tf32 = recipe["tf32"]
    model = EfficientAdModel(teacher_out_channels=384, model_size=EfficientAdModelSize.S,
                             padding=profile["padding"], pad_maps=profile["pad_maps"])
    implementation = Path(inspect.getfile(EfficientAdModel))
    expected_source = "8f80c2db5e546e5209f105c757d8c88c70c48e86bcf1d036b797fff88be3bd08"
    if hashlib.sha256(implementation.read_bytes()).hexdigest() != expected_source:
        raise ValueError("EfficientAD core differs from the reviewed source.")
    model.teacher.load_state_dict(torch.load(teachers[0], map_location="cpu", weights_only=True), strict=True)
    model.teacher.requires_grad_(False)
    model.mean_std.requires_grad_(False)
    model.quantiles.requires_grad_(False)
    model = model.cuda()
    initial = {name: state_digest(getattr(model, name)) for name in ("teacher", "student", "ae")}
    transform = Compose([Resize((256, 256), antialias=True), ToTensor()])
    images = []
    for row in fit:
        with load_verified_image(args.root, row) as image:
            images.append(transform(image))
    cache = torch.stack(images).pin_memory()
    del images
    model.teacher.eval()
    total, sum_y, sum_sq = 0, None, None
    with torch.no_grad():
        for image in cache:
            features = model.teacher(image.unsqueeze(0).cuda()).double()
            sums = features.sum(dim=(0, 2, 3))
            squares = features.square().sum(dim=(0, 2, 3))
            sum_y = sums if sum_y is None else sum_y + sums
            sum_sq = squares if sum_sq is None else sum_sq + squares
            total += features.shape[0] * features.shape[2] * features.shape[3]
        mean = sum_y / total
        std = (sum_sq / total - mean.square()).clamp_min(0).sqrt()
        clamped = int((std < recipe["teacher_std_epsilon"]).sum().item())
        std = std.clamp_min(recipe["teacher_std_epsilon"])
        model.mean_std["mean"].copy_(mean.float()[None, :, None, None])
        model.mean_std["std"].copy_(std.float()[None, :, None, None])
    auxiliary = ImageFolder(aux_roots[0], transform=Compose([
        Resize((512, 512), antialias=True), RandomGrayscale(p=0.3), CenterCrop((256, 256)), ToTensor()]))
    auxiliary_generator = torch.Generator().manual_seed(recipe["seed"] + 1)
    loader = DataLoader(auxiliary, batch_size=1, shuffle=True, pin_memory=True,
                        num_workers=recipe["auxiliary_workers"], worker_init_fn=seed_worker,
                        generator=auxiliary_generator, persistent_workers=recipe["auxiliary_workers"] > 0)
    aux_iterator = iter(loader)
    sampler = torch.Generator().manual_seed(recipe["seed"] + 2)
    optimizer = torch.optim.Adam(list(model.student.parameters()) + list(model.ae.parameters()),
                                 lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=int(recipe["steps"] * recipe["lr_decay_fraction"]), gamma=recipe["lr_decay_gamma"])
    report.update(status="TRAINING_IN_PROGRESS", model_config=profile, training_config=recipe,
                  gpu=torch.cuda.get_device_name(0), packages={name: importlib.metadata.version(name) for name in
                      ("torch", "torchvision", "anomalib", "numpy", "pillow")},
                  input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                      (args.profile, args.recipe, args.assets, Path(profile["data_manifest"]),
                       Path("requirements/pylock.vision.toml"), Path(__file__))},
                  teacher_file={"bytes": teachers[0].stat().st_size, "sha256": fingerprint(teachers[0])[1]},
                  teacher_statistics={"fit_images": len(fit), "spatial_values_per_channel": total,
                                      "std_clamped_channels": clamped, "accumulation_dtype": "float64"},
                  fit_paths=[row["path"] for row in fit], validation_metadata_count=len(validation),
                  dataset_pixel_reads={"fit": len(fit), "validation_normal": 0,
                                       "development_preview": 0, "final_holdout": 0},
                  auxiliary_images=len(auxiliary), auxiliary_classes=len(auxiliary.classes),
                  auxiliary_split="Imagenette train only; labels discarded; regularization only",
                  state_sha256_initial=initial, checkpoints=[], completed_steps=0,
                  map_calibration="NOT_RUN", threshold="NOT_SET", accuracy="NOT_EVALUATED",
                  jetson_proof="NOT_VERIFIED", bitwise_reproducibility="NOT_CLAIMED")
    (args.output / "run_config.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "training_start", "fit_images": len(fit), "auxiliary_images": len(auxiliary),
                      "steps": recipe["steps"], "std_clamped_channels": clamped}), flush=True)
    started, window = time.monotonic(), []
    torch.cuda.reset_peak_memory_stats()
    model.train()
    model.teacher.eval()
    order, offset = [], 0
    with (args.output / "losses.jsonl").open("x") as log:
        for step in range(1, recipe["steps"] + 1):
            if offset >= len(order):
                order = torch.randperm(len(fit), generator=sampler).tolist()
                offset = 0
            batch = cache[order[offset]].unsqueeze(0).cuda(non_blocking=True)
            offset += 1
            try:
                ordinary, _ = next(aux_iterator)
            except StopIteration:
                aux_iterator = iter(loader)
                ordinary, _ = next(aux_iterator)
            optimizer.zero_grad(set_to_none=True)
            losses = model(batch, ordinary.cuda(non_blocking=True))
            loss = sum(losses)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                       for group in optimizer.param_groups for parameter in group["params"]):
                raise RuntimeError(f"Nonfinite gradient at step {step}")
            optimizer.step()
            scheduler.step()
            values = [float(value.detach().item()) for value in losses]
            window.append(values)
            report["completed_steps"] = step
            if step == 1 or step % recipe["log_every"] == 0 or step == recipe["steps"]:
                row = {"step": step, "total_loss": sum(values), "loss_st": values[0],
                       "loss_ae": values[1], "loss_stae": values[2],
                       "window_mean_total": sum(map(sum, window)) / len(window),
                       "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.monotonic() - started}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                window.clear()
            if step in recipe["checkpoint_steps"] or step == recipe["steps"]:
                path = args.output / f"checkpoint_step_{step:06d}.pth"
                if path.exists():
                    raise FileExistsError(path)
                torch.save({"format": "forgesight-efficientad-v1", "step": step,
                            "model_config": profile, "training_config": recipe,
                            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "calibration": "NOT_RUN"}, path)
                report["checkpoints"].append({"path": str(path), "bytes": path.stat().st_size,
                                              "sha256": fingerprint(path)[1], "step": step})
            if step % recipe["log_every"] == 0:
                (args.output / "progress.json").write_text(json.dumps({"step": step, "target": recipe["steps"],
                    "elapsed_seconds": time.monotonic() - started}) + "\n")
    torch.cuda.synchronize()
    report["training_wall_seconds"] = time.monotonic() - started
    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    final = {name: state_digest(getattr(model, name)) for name in ("teacher", "student", "ae")}
    if final["teacher"] != initial["teacher"] or any(final[name] == initial[name] for name in ("student", "ae")):
        raise RuntimeError("Teacher changed or trainable networks did not update.")
    report["state_sha256_final"] = final
    model.eval()
    with torch.inference_mode():
        anchor = model(cache[:1].cuda(), normalize=False)
        report["fit_anchor_raw_score"] = float(anchor.pred_score.item())
        if not torch.isfinite(anchor.anomaly_map).all().item():
            raise RuntimeError("Trained model produced nonfinite raw map.")
    report["status"] = "TRAINED_UNCALIBRATED_NOT_ACCURACY_VERIFIED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/vision_baseline.json"))
    parser.add_argument("--recipe", type=Path, default=Path("configs/efficientad_s_training.json"))
    parser.add_argument("--assets", type=Path, default=Path("artifacts/vision/stage3_2/assets.json"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new training output directory.")
    args.output.mkdir(parents=True)
    report = {"status": "FAILED_NOT_VERIFIED", "completed_steps": 0}
    try:
        train(args, report)
    except Exception as error:
        report.update(status="FAILED_NOT_VERIFIED", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "completed_steps")}), flush=True)
    if report["status"] == "FAILED_NOT_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
