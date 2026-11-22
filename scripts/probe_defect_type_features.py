"""Measure defect-type cues on the four role-restricted MVTec previews."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional

from forgesight.data_loader import MetalNutDataset
from forgesight.data_manifest import DEFAULT_MANIFEST, DEFAULT_ROOT
from forgesight.defect_type import response_features
from forgesight.regions import response_regions


MODEL_SHA256 = "8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd"


def infer(model, image):
    pixels = np.asarray(image.resize((256, 256), Image.BILINEAR), dtype=np.uint8)
    tensor = np.transpose(pixels, (2, 0, 1)).astype(np.float32)[None] / 255.0
    with torch.inference_mode():
        anomaly_map, score = model(torch.from_numpy(tensor))
        resized = functional.interpolate(anomaly_map, size=(image.height, image.width),
                                         mode="bilinear", align_corners=False)
    return float(score.item()), resized[0, 0].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256:
        parser.error("Expected the frozen EfficientAD TorchScript export.")
    policy = json.loads(Path("configs/forgegrade_v1.json").read_text(encoding="utf-8"))
    threshold = policy["image_threshold"]
    dataset = MetalNutDataset(args.data_root, args.manifest, role="development_preview")
    torch.set_num_threads(2)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    args.output.mkdir(parents=True)
    rows = []
    for sample in dataset:
        with sample.image, sample.mask:
            score, score_map = infer(model, sample.image)
            support = score_map > threshold
            geometry = response_regions(score_map, threshold, mvtec_nut_layout=True)
            started = perf_counter()
            features = response_features(sample.image, score_map, geometry) if score > threshold else None
            feature_ms = (perf_counter() - started) * 1000
            # Overlap is diagnostic only. The annotation never supplies classifier features.
            annotation = np.asarray(sample.mask) != 0
            primary = geometry["regions"][0] if geometry["regions"] else None
            rows.append({"image": sample.source_path, "label": sample.ground_truth_type,
                         "score": score, "flagged": score > threshold,
                         "map_annotation_iou": float((support & annotation).sum() / (support | annotation).sum()),
                         "primary_region": primary, "features": features, "feature_ms": feature_ms})
            overlay = Image.composite(Image.new("RGB", sample.image.size, "red"), sample.image,
                                      Image.fromarray(support.astype(np.uint8) * 110))
            if primary:
                x, y, width, height = primary["bbox_xywh_px"]
                ImageDraw.Draw(overlay).rectangle((x, y, x + width - 1, y + height - 1),
                                                   outline="yellow", width=2)
            overlay.save(args.output / (sample.ground_truth_type + "_response.png"))
    report = {"dataset_role": dataset.role, "model_sha256": MODEL_SHA256,
              "image_threshold": threshold, "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
