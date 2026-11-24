"""Read-only training readiness and isolated post-training command preparation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from forgesight.data_manifest import safe_path
from forgesight.vision_recipe import read_manifest
from forgesight.vision_train import validate_recipe

DEFAULT_PROTOCOL = Path("configs/screw_30k_review_v2.json")


def load_protocol(path):
    protocol = json.loads(path.read_text())
    if (protocol["schema_version"] != 1 or protocol["expected_steps"] != 30000
            or protocol["candidate_steps"] != [10000, 20000, 30000]
            or protocol["automatic_promotion"] is not False
            or protocol["reuse_exposed_holdout_for_acceptance"] is not False):
        raise ValueError("Unreviewed experiment budget or promotion policy.")
    if protocol["development_manifest"] is not None or protocol["acceptance_manifest"] is not None:
        raise ValueError("New independent datasets require a separate audit before activation.")
    for name in ("reload_report", "calibration_directory", "export_model", "export_parity"):
        value = protocol["paths"][name]
        safe_path(Path("."), value)
        if not value.startswith("artifacts/vision/screw_30k_v2/"):
            raise ValueError("Review outputs must remain isolated from training and promoted plugins.")
    recipe = json.loads(Path(protocol["training_recipe"]).read_text())
    validate_recipe(recipe)
    if recipe["steps"] != protocol["expected_steps"] or recipe["checkpoint_steps"] != protocol["candidate_steps"]:
        raise ValueError("Recipe and review protocol budgets differ.")
    profile = json.loads(Path(protocol["model_profile"]).read_text())
    manifest = Path(protocol["dataset_manifest"])
    document = read_manifest(manifest)
    if (document["category"] != "screw" or profile["data_manifest"] != str(manifest).replace("\\", "/")
            or hashlib.sha256(manifest.read_bytes()).hexdigest() != profile["data_manifest_sha256"]):
        raise ValueError("Frozen training manifest differs from the selected profile.")
    policy = json.loads(Path(protocol["calibration_policy"]).read_text())
    from forgesight.vision_calibrate import calibration_split
    calibration_split(document, policy)  # Metadata only; never opens dataset pixels.
    return protocol


def read_json_if_available(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None  # A running worker may be between progress-file writes.


def review_status(protocol):
    paths = protocol["paths"]
    result = {"experiment": protocol["experiment"], "postprocessing_ready": False,
        "development_comparison_status": "BLOCKED_MISSING_INDEPENDENT_DEVELOPMENT_DATA",
        "acceptance_status": "BLOCKED_MISSING_FRESH_ACCEPTANCE_DATA",
        "model_inference_started": False, "promotion_allowed": False}
    run_config = read_json_if_available(Path(paths["run_config"]))
    hashes = run_config.get("input_sha256") if isinstance(run_config, dict) else None
    valid_hashes = (isinstance(hashes, dict) and bool(hashes)
                    and all(isinstance(k, str) and isinstance(v, str)
                            and re.fullmatch(r"[0-9a-f]{64}", v) for k, v in hashes.items()))
    required = {protocol[k] for k in ("training_recipe", "model_profile", "dataset_manifest")}
    evidence_complete = (valid_hashes and required.issubset(hashes)
                         and any(k.replace("\\", "/").endswith("forgesight/vision_train.py") for k in hashes))
    result["training_input_status"] = "INPUT_HASHES_MATCH" if evidence_complete else "MISSING_OR_INVALID_INPUT_EVIDENCE"
    if valid_hashes:
        for name, expected in hashes.items():
            # WSL records absolute Linux source paths; support checking on Windows too.
            local = "C:/" + name[7:] if sys.platform == "win32" and name.startswith("/mnt/c/") else name
            source = Path(local)
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                return {**result, "training_status": "BLOCKED_TRAINING_INPUT_CHANGED", "changed_input": name}
    training = read_json_if_available(Path(paths["training_report"]))
    if training is not None:
        if not evidence_complete:
            return {**result, "training_status": "BLOCKED_MISSING_TRAINING_INPUT_EVIDENCE"}
        if training.get("input_sha256") != hashes:
            return {**result, "training_status": "BLOCKED_TRAINING_INPUT_EVIDENCE_MISMATCH"}
        recipe = json.loads(Path(protocol["training_recipe"]).read_text())
        ready = (training.get("status") == "TRAINED_UNCALIBRATED_NOT_ACCURACY_VERIFIED"
                 and training.get("completed_steps") == protocol["expected_steps"]
                 and training.get("training_config") == recipe
                 and [c["step"] for c in training.get("checkpoints", [])] == protocol["candidate_steps"])
        return {**result, "training_status": "TRAINING_REPORTED_COMPLETE_RELOAD_REQUIRED" if ready else "TRAINING_FAILED_OR_INCOMPLETE",
                "completed_steps": training.get("completed_steps"), "postprocessing_ready": ready}
    progress = read_json_if_available(Path(paths["progress"]))
    return {**result, "training_status": "TRAINING_IN_PROGRESS" if progress else "AWAITING_VALID_PROGRESS_OR_REPORT",
            "completed_steps": progress.get("step") if progress else None,
            "target_steps": protocol["expected_steps"]}


def postprocess_commands(protocol, protocol_path=DEFAULT_PROTOCOL):
    paths = protocol["paths"]
    reference = str(Path(paths["calibration_directory"]) / "calibration_report.json")
    python = [sys.executable, "-B", "-m"]
    return [
        python + ["forgesight.vision_checkpoint", "--training-report", paths["training_report"],
                  "--output", paths["reload_report"], "--root", protocol["dataset_root"], "--device", "cpu"],
        python + ["forgesight.vision_calibrate", "--policy", protocol["calibration_policy"],
                  "--training-record", paths["training_report"], "--root", protocol["dataset_root"],
                  "--device", "cpu", "--output", paths["calibration_directory"]],
        python + ["forgesight.vision_edge_export", "--reference", reference, "--output", paths["export_model"]],
        python + ["forgesight.screw_run_review", "--protocol", str(protocol_path), "--verify-export"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true")
    group.add_argument("--commands", action="store_true")
    group.add_argument("--verify-export", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    status = review_status(protocol)
    if args.commands or args.verify_export:
        if not status["postprocessing_ready"]:
            parser.error("Training must report all 30,000 steps complete before postprocessing.")
        if args.commands:
            status["commands_run_sequentially_after_each_passes"] = postprocess_commands(protocol, args.protocol)
        else:
            from forgesight.screw_evaluate import verify_export
            paths = protocol["paths"]
            status["export_verification"] = verify_export(
                reference_path=Path(paths["calibration_directory"]) / "calibration_report.json",
                export_path=Path(paths["export_model"]), parity_path=Path(paths["export_parity"]),
                root=Path(protocol["dataset_root"]))
            status["model_inference_started"] = True
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
