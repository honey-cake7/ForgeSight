"""Render a source-pinned preparation recipe; never download, export, or build."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess


def validate_export_config(config: dict, schema: dict) -> None:
    fields = schema["fields"]
    for section, values in config.items():
        if section not in {"base", "model", "export", "quantization", "backend", "debug"}:
            raise ValueError(f"Unknown upstream config section: {section}")
        if not isinstance(values, dict) or not set(values) <= set(fields[section]):
            raise ValueError(f"Unsupported pinned upstream keys in {section}")
    backend = config.get("backend", {})
    if set(backend) != {"xnnpack"} or not set(backend["xnnpack"]) <= set(fields["xnnpack"]):
        raise ValueError("Recipe must use the reviewed XNNPACK config.")
    export = config["export"]
    if (type(export["max_seq_length"]) is not int or type(export["max_context_length"]) is not int
            or not 0 < export["max_seq_length"] <= export["max_context_length"]):
        raise ValueError("Invalid sequence/context lengths.")


def export_config(profile: dict, weights: Path, output: Path) -> dict:
    if profile["backend"] != "xnnpack" or profile["quantization"] != "8da4w":
        raise ValueError("Unreviewed backend/quantization combination.")
    if profile["model_class"] != "llama3_2":
        raise ValueError("Unreviewed model class.")
    return {
        "base": {"model_class": "llama3_2",
                 "checkpoint": str((weights / "original/consolidated.00.pth").resolve()),
                 "params": str((weights / "original/params.json").resolve()),
                 "metadata": json.dumps({"get_bos_id": 128000, "get_eos_ids": [128009, 128001]})},
        "model": {"use_kv_cache": True, "use_sdpa_with_kv_cache": True, "dtype_override": "fp32"},
        "quantization": {"qmode": profile["quantization"], "group_size": profile["group_size"],
                         "embedding_quantize": profile["embedding_quantize"]},
        "backend": {"xnnpack": {"enabled": True, "extended_ops": True}},
        "export": {"max_seq_length": profile["max_seq_length"],
                   "max_context_length": profile["max_context_length"],
                   "output_dir": str((output / "exported").resolve()),
                   # Upstream treats names ending in .pte as verbatim paths,
                   # bypassing output_dir. A stem keeps output under that directory.
                   "output_name": "llama3_2_1b_xnnpack"},
        "debug": {"verbose": True},
    }


def create_recipe(profile: dict, schema: dict, checkout: Path, weights: Path, output: Path) -> dict:
    if profile["executorch_commit"] != schema["executorch_commit"]:
        raise ValueError("Profile and upstream schema refer to different source commits.")
    if output.exists():
        raise FileExistsError(f"Recipe output exists: {output}")
    config = export_config(profile, weights, output)
    validate_export_config(config, schema)
    missing = [name for name in profile["source_files"] if not (weights / name).is_file()]
    blockers = ["Export dependency closure, model conversion, and target runtime are NOT VERIFIED."]
    if profile["model_access"] != "ACCESS_GRANTED":
        blockers.append("Meta Llama access is not granted in the recorded profile.")
    if missing:
        blockers.append("Required licensed model files are missing.")
    checkout_commit = None
    if (checkout / ".git").exists():
        result = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0:
            checkout_commit = result.stdout.strip()
    if checkout_commit != profile["executorch_commit"]:
        blockers.append("A matching pinned ExecuTorch source checkout is not verified.")
    try:
        installed = importlib.metadata.version("executorch")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    # Official CPU wheels carry a PEP 440 local build suffix (e.g. +cpu).
    if installed is None or installed.split("+", 1)[0] != profile["executorch_version"]:
        blockers.append("The selected ExecuTorch export environment is not installed here.")
    config_path = (output / "export_config.yaml").resolve()
    recipe = {
        "schema_version": 1, "status": "PREPARATION_ONLY_NOT_RUN", "ready_to_claim_export": False,
        "model_id": profile["model_id"], "model_revision": profile["model_revision"],
        "executorch_commit": profile["executorch_commit"], "checkout_observed_commit": checkout_commit,
        "installed_executorch": installed, "missing_model_files": missing, "blockers": blockers,
        "download_commands_not_run": [
            {"argv": ["git", "clone", "--branch", "v" + profile["executorch_version"],
                      "--recurse-submodules", "https://github.com/pytorch/executorch.git", str(checkout.resolve())]},
            {"argv": ["hf", "download", profile["model_id"], "--revision", profile["model_revision"],
                      "--local-dir", str(weights.resolve()), *profile["source_files"]]},
        ],
        "export_command_not_run": {"cwd": str(checkout.resolve()),
                                   "argv": ["python3", "-m", "extension.llm.export.export_llm",
                                            "--config", str(config_path)]},
        "native_target_build_commands_not_run": [
            {"cwd": str(checkout.resolve()), "argv": ["cmake", "--workflow", "--preset", "llm-release"]},
            {"cwd": str((checkout / "examples/models/llama").resolve()),
             "argv": ["cmake", "--workflow", "--preset", "llama-release"]},
        ],
        "target_build_condition": "Run natively only after Jetson preflight, ABI, dependencies, and CMake presets are checked.",
        "planned_pte": str((output / "exported/llama3_2_1b_xnnpack.pte").resolve()),
        "tokenizer_format": "original/tokenizer.model, paired with this model revision",
        "generated_text": None,
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, document in (("export_config.yaml", config), ("recipe.json", recipe)):
        with (output / name).open("x", encoding="utf-8") as stream:
            # JSON syntax is a YAML subset; no extra parser/package is needed to render it.
            json.dump(document, stream, indent=2)
            stream.write("\n")
    return recipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/edge_llama.json"))
    parser.add_argument("--schema", type=Path, default=Path("configs/edge_upstream_schema.json"))
    parser.add_argument("--checkout", type=Path, default=Path("data/tools/executorch"))
    parser.add_argument("--weights", type=Path, default=Path("models/llama3_2_1b"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/edge/export_recipe_v1"))
    args = parser.parse_args()
    try:
        recipe = create_recipe(json.loads(args.profile.read_text()), json.loads(args.schema.read_text()),
                               args.checkout, args.weights, args.output)
        print(json.dumps({key: recipe[key] for key in ("status", "ready_to_claim_export",
                                                      "missing_model_files", "blockers")}, indent=2))
        print("PASS: recipe syntax/keys checked. No packages installed, weights downloaded, or model exported.")
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
