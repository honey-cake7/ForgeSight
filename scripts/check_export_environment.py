"""Check real exporter imports/configuration without loading or converting weights."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new evidence directory.")
    args.output.mkdir(parents=True)
    report = {"status": "FAILED_NOT_VERIFIED", "model_export": "NOT_RUN",
              "model_inference": "NOT_RUN", "jetson_proof": "NOT_VERIFIED"}
    try:
        profile = json.loads(Path("configs/edge_llama.json").read_text())
        reference = json.loads(Path("configs/edge_upstream_schema.json").read_text())
        assert sys.prefix != sys.base_prefix, "Use the separate export venv."
        assert platform.system() == "Linux" and platform.machine() == "x86_64"
        assert sys.version_info[:2] == (3, 12), "This lock targets CPython 3.12."
        commit = subprocess.check_output(
            ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
        assert commit == profile["executorch_commit"], "Source revision differs."
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "torchvision", "executorch", "torchao", "torchtune",
                     "transformers", "hydra-core", "cmake", "ninja")}
        assert versions["executorch"] == profile["executorch_version"] + "+cpu"
        assert versions["torch"] == "2.14.0+cpu"
        imports = io.StringIO()
        try:
            with redirect_stdout(imports), redirect_stderr(imports):
                import torch
                from omegaconf import OmegaConf
                from executorch.extension.llm.export.config import llm_config
        finally:
            (args.output / "imports.log").write_text(imports.getvalue())
        source = args.checkout / reference["source_path"]
        observed_hashes = [hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (source, Path(llm_config.__file__))]
        assert observed_hashes == [reference["source_sha256"]] * 2, "Schema differs."
        assert torch.version.cuda is None, "Expected the isolated CPU build."
        cpu_result = (torch.tensor([[1., 2.]]) @ torch.tensor([[3.], [4.]])).item()
        assert cpu_result == 11.0, "CPU tensor operation failed."
        structured = OmegaConf.merge(OmegaConf.structured(llm_config.LlmConfig),
                                     OmegaConf.load(args.config.resolve()))
        parsed = OmegaConf.to_object(structured)
        assert isinstance(parsed, llm_config.LlmConfig)
        command = [sys.executable, "-B", "-m", "extension.llm.export.export_llm", "--help"]
        with (args.output / "cli_stdout.log").open("w") as out, \
                (args.output / "cli_stderr.log").open("w") as err:
            result = subprocess.run(command, cwd=args.checkout, stdout=out, stderr=err,
                                    timeout=120, check=False)
        assert result.returncode == 0, "Actual exporter CLI failed; inspect logs."
        assert "--config" in (args.output / "cli_stdout.log").read_text()
        report.update(status="EXPORT_ENVIRONMENT_VERIFIED_NOT_EXPORTED",
                      python=sys.version.split()[0], platform=platform.platform(),
                      interpreter=sys.executable, source_commit=commit, packages=versions,
                      schema_sha256=observed_hashes[0], config=str(args.config.resolve()),
                      config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                      structured_config=type(parsed).__name__, cpu_matmul=cpu_result,
                      torch_cuda_build=torch.version.cuda, cli_command=command,
                      cli_exit_code=result.returncode,
                      warning="See imports.log and cli_stderr.log; optional TorchAO GPU libraries failed to load.")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] == "FAILED_NOT_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
