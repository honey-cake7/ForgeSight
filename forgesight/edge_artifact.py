"""Check a recorded PTE fingerprint and load its program metadata, without generation."""

import argparse
import importlib.metadata
import json
from pathlib import Path

from forgesight.edge_assets import fingerprint


def check_fingerprint(model: Path, reference: dict) -> dict:
    if model.suffix != ".pte":
        raise ValueError("Expected an ExecuTorch .pte artifact.")
    size, digest = fingerprint(model)
    expected = reference["artifact"]
    if size == 0 or (size, digest) != (expected["bytes"], expected["sha256"]):
        raise ValueError("Exported artifact fingerprint mismatch.")
    with model.open("rb") as stream:
        header = stream.read(8)
    if header[4:8] != b"ET12":
        raise ValueError("Artifact does not have the pinned program schema identifier.")
    return {"bytes": size, "sha256": digest, "file_identifier": "ET12"}


def inspect_artifact(model: Path, reference: dict, profile: dict) -> dict:
    if (reference["schema_version"] != 1 or reference["model_id"] != profile["model_id"]
            or reference["model_revision"] != profile["model_revision"]
            or reference["executorch_commit"] != profile["executorch_commit"]):
        raise ValueError("Export reference differs from the selected model/runtime.")
    artifact = check_fingerprint(model, reference)
    version = importlib.metadata.version("executorch")
    if version.split("+", 1)[0] != profile["executorch_version"]:
        raise ValueError("Use the pinned ExecuTorch environment.")
    # These private readers are deliberately tied to the pinned release.
    from executorch.exir._serialize._program import _get_extended_header, _flatbuffer_to_program
    from executorch.runtime import Runtime

    with model.open("rb") as stream:
        header = _get_extended_header(stream.read(64))
        if (header is None or not header.is_valid()
                or not 64 <= header.program_size <= min(64 * 1024 * 1024, artifact["bytes"])):
            raise ValueError("Expected bounded program metadata in the pinned extended header.")
        stream.seek(0)
        metadata = _flatbuffer_to_program(stream.read(header.program_size))
    delegates = {plan.name: sorted({delegate.id for delegate in plan.delegates})
                 for plan in metadata.execution_plan}
    if profile["backend"] != "xnnpack" or "XnnpackBackend" not in delegates.get("forward", []):
        raise ValueError("Expected a serialized XNNPACK delegate in forward.")
    # InternalConsistency verification is the upstream default. No method is executed.
    program = Runtime.get().load_program(model.resolve())
    methods = sorted(program.method_names)
    if not {"forward", "get_bos_id", "get_eos_ids"} <= set(methods):
        raise ValueError("Expected forward and configured token metadata methods.")
    return {"status": "ARTIFACT_LOADED_NOT_INFERENCE_VERIFIED", "artifact": artifact,
            "executorch_version": version, "methods": methods,
            "program_metadata_bytes": header.program_size, "delegates": delegates,
            "verification": "Runtime.load_program default InternalConsistency",
            "method_execution": "NOT_RUN", "generated_text": None,
            "quality": "NOT_VERIFIED", "jetson_proof": "NOT_VERIFIED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("manifests/llama3_2_1b_export.json"))
    parser.add_argument("--profile", type=Path, default=Path("configs/edge_llama.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence output exists; choose a new file.")
    report = {"status": "FAILED_NOT_VERIFIED", "method_execution": "NOT_RUN",
              "generated_text": None, "jetson_proof": "NOT_VERIFIED"}
    try:
        report = inspect_artifact(args.model, json.loads(args.reference.read_text()),
                                  json.loads(args.profile.read_text()))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    if report["status"] == "FAILED_NOT_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
