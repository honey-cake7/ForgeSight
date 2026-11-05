"""Load only a fingerprint-bound calibrated model and its matching score policy."""

import json
import math
from pathlib import Path

from forgesight.edge_assets import fingerprint
from forgesight.vision_checkpoint import load_checkpoint


def validate_bundle_policy(policy: dict) -> None:
    if (policy["comparison"] != "strict_greater_than" or policy["above_threshold"] != "ANOMALY_CANDIDATE"
            or policy["at_or_below_threshold"] != "NORMAL_CANDIDATE"
            or policy["is_probability"] is not False or policy["is_final_quality_grade"] is not False
            or policy["final_holdout_access"] is not False):
        raise ValueError("Unreviewed candidate-score semantics or data access policy.")
    if not math.isfinite(policy["threshold"]):
        raise ValueError("Policy threshold must be finite.")


def load_calibrated_bundle(reference: dict, device: str = "cpu"):
    if reference["status"] != "CALIBRATED_POLICY_FROZEN_HOLDOUT_NOT_EVALUATED":
        raise ValueError("Expected the frozen calibration reference.")
    checkpoint = reference["calibrated_checkpoint"]
    policy_path = Path(checkpoint["path"]).parent / "score_policy.json"
    if fingerprint(policy_path)[1] != reference["policy_file_sha256"]:
        raise ValueError("Score policy file fingerprint mismatch.")
    policy = json.loads(policy_path.read_text())
    validate_bundle_policy(policy)
    if policy != reference["policy"] or policy["calibrated_checkpoint_sha256"] != checkpoint["sha256"]:
        raise ValueError("Policy is not bound to this calibrated checkpoint/reference.")
    model, saved = load_checkpoint(Path(checkpoint["path"]), checkpoint["sha256"], device)
    import torch
    from forgesight.vision_calibrate import configure_math

    if saved["training_config"]["precision"] != "fp32" or next(model.parameters()).dtype != torch.float32:
        raise ValueError("This frozen policy requires its FP32 reference score function.")
    configure_math(saved["training_config"])
    calibration = saved["calibration"]
    if (saved["source_checkpoint_sha256"] != policy["source_checkpoint_sha256"]
            or calibration["threshold"] != policy["threshold"]
            or calibration["quantiles"] != policy["map_quantile_values"]
            or {name: value.item() for name, value in model.quantiles.items()} != calibration["quantiles"]):
        raise ValueError("Checkpoint calibration does not match its recorded policy.")
    return model, policy, saved
