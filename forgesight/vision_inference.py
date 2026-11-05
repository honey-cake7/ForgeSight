"""Shared real FP32 inference path for evaluation and later inspection records."""

import json
import math
from pathlib import Path
import time

from forgesight.vision_calibration_bundle import load_calibrated_bundle
from forgesight.vision_calibrate import classify_score


class CalibratedPredictor:
    def __init__(self, reference: Path = Path("manifests/efficientad_s_calibration_v1.json"), device="cuda"):
        self.reference = json.loads(reference.read_text())
        self.model, self.policy, self.saved = load_calibrated_bundle(self.reference, device)
        self.device = device

    def predict(self, image) -> dict:
        import numpy as np
        import torch
        from torch.nn import functional as F
        from torchvision.transforms import Compose, Resize, ToTensor

        if image.mode != "RGB" or not 0 < image.width * image.height <= 16777216:
            raise ValueError("Expected a nonempty RGB inspection image within the size bound.")
        started = time.perf_counter()
        tensor = Compose([Resize((256, 256), antialias=True), ToTensor()])(image).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with torch.inference_mode():
            result = self.model(tensor, normalize=True)
            if self.device == "cuda":
                torch.cuda.synchronize()
            forward_ms = (time.perf_counter() - forward_start) * 1000
            score = float(result.pred_score.item())
            native = result.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
            source = F.interpolate(result.anomaly_map, size=(image.height, image.width),
                                   mode="bilinear", align_corners=False).squeeze().cpu().numpy().astype(np.float32)
        if not math.isfinite(score) or not np.isfinite(source).all() or native.shape != (256, 256):
            raise ValueError("Model produced invalid score/map output.")
        return {"score": score, "candidate": classify_score(score, self.policy["threshold"]),
                "map_model": native, "map_source": source, "predicted_defect_type": "unknown",
                "model_forward_ms": forward_ms, "predict_image_ms": (time.perf_counter() - started) * 1000,
                "timing_scope": "predict_image includes resize/transfers/maps, excludes file decode/hash, regions, grading, output I/O and Llama"}
