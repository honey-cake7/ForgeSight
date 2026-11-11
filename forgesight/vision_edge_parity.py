"""Compare a TorchScript edge export against the frozen Anomalib source on fit images."""

import argparse
import json
from pathlib import Path

from forgesight.vision_calibration_bundle import load_calibrated_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from torchvision.transforms import Compose, Resize, ToTensor

    reference = json.loads(Path("manifests/efficientad_s_calibration_v1.json").read_text())
    source, _, _ = load_calibrated_bundle(reference, "cpu")
    edge = torch.jit.load(str(args.model), map_location="cpu").eval()
    transform = Compose([Resize((256, 256), antialias=True), ToTensor()])
    results = []
    with torch.inference_mode():
        for path in args.image:
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0)
            expected = source(tensor, normalize=True)
            actual_map, actual_score = edge(tensor)
            map_difference = float((expected.anomaly_map - actual_map).abs().max())
            score_difference = float((expected.pred_score - actual_score).abs().max())
            if map_difference > 1e-6 or score_difference > 1e-6:
                raise ValueError(f"Frozen model/export mismatch for {path}")
            results.append({"image": str(path), "score": float(actual_score.item()),
                            "map_max_absolute_difference": map_difference,
                            "score_absolute_difference": score_difference})
    print(json.dumps({"status": "SOURCE_EXPORT_PARITY_VERIFIED_PC_ONLY", "results": results}, indent=2))


if __name__ == "__main__":
    main()
