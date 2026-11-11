"""Export the frozen calibrated EfficientAD forward pass for a Torch-only edge runtime."""

import argparse
import json
from pathlib import Path

from forgesight.edge_assets import fingerprint
from forgesight.vision_calibration_bundle import load_calibrated_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("manifests/efficientad_s_calibration_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path.")

    import torch

    reference = json.loads(args.reference.read_text())
    model, policy, _ = load_calibrated_bundle(reference, "cpu")

    class ForwardOnly(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.source = source

        def forward(self, image):
            result = self.source(image, normalize=True)
            return result.anomaly_map, result.pred_score

    wrapper = ForwardOnly(model).eval()
    example = torch.zeros(1, 3, 256, 256)
    with torch.inference_mode():
        expected = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=True)
        actual = traced(example)
    differences = [float((a - b).abs().max()) for a, b in zip(expected, actual)]
    if any(value > 1e-6 for value in differences):
        raise ValueError(f"Export parity failed: {differences}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    size, digest = fingerprint(args.output)
    print(json.dumps({"artifact": str(args.output), "bytes": size, "sha256": digest,
                      "source_checkpoint_sha256": reference["calibrated_checkpoint"]["sha256"],
                      "policy_sha256": reference["policy_file_sha256"],
                      "threshold": policy["threshold"], "zero_input_max_absolute_difference": differences,
                      "status": "EXPORTED_PC_ONLY"}, indent=2))


if __name__ == "__main__":
    main()
