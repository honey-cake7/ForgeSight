"""Export the hash-checked frozen EfficientAD TorchScript forward to fixed-shape ONNX."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_MODEL_SHA256 = "8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new ONNX output file")
    if sha256(args.model) != EXPECTED_MODEL_SHA256 or args.opset not in (16, 17):
        parser.error("Frozen model hash or requested ONNX opset is invalid")

    import onnx
    import torch

    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    example = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    with torch.inference_mode():
        expected_map, expected_score = model(example)
    if tuple(expected_map.shape) != (1, 1, 256, 256) or expected_score.numel() != 1:
        raise ValueError("Unexpected frozen model output shapes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (example,), str(args.output), opset_version=args.opset,
                      input_names=["image"], output_names=["anomaly_map", "score"],
                      export_params=True, do_constant_folding=True, dynamo=False)
    exported = onnx.load(str(args.output))
    onnx.checker.check_model(exported)
    report = {"status": "ONNX_EXPORTED_PC_ONLY", "source_model_sha256": EXPECTED_MODEL_SHA256,
              "onnx_sha256": sha256(args.output), "onnx_bytes": args.output.stat().st_size,
              "onnx_ir_version": exported.ir_version,
              "opsets": {item.domain or "ai.onnx": item.version for item in exported.opset_import},
              "input_names": [value.name for value in exported.graph.input],
              "output_names": [value.name for value in exported.graph.output],
              "node_count": len(exported.graph.node),
              "torch_version": torch.__version__, "onnx_version": onnx.__version__}
    report_path = args.output.with_suffix(".json")
    if report_path.exists():
        raise FileExistsError("Choose an export path with no existing report")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
