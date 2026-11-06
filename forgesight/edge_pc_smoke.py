"""Run a bounded desktop-only generation test through ExecuTorch's native LLM runner."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

from forgesight.edge_artifact import check_fingerprint
from forgesight.edge_assets import fingerprint
from forgesight.edge_capture import capture_process


def chat_prompt(text: str, *, system_text: str | None = None) -> str:
    text = text.strip()
    if not text or len(text.encode("utf-8")) > 4096:
        raise ValueError("Smoke prompt must contain 1-4096 UTF-8 bytes.")
    if "<|" in text or "|>" in text:
        raise ValueError("Smoke prompt must not contain chat control markers.")
    # BOS is explicit, so GenerationConfig must use num_bos=0.
    system = ''
    if system_text is not None:
        if (not isinstance(system_text, str) or not system_text.strip()
                or len(system_text.encode('utf-8')) > 4096 or '<|' in system_text or '|>' in system_text):
            raise ValueError('System text must be bounded plain text without chat controls.')
        system = '<|start_header_id|>system<|end_header_id|>\n\n' + system_text.strip() + '<|eot_id|>'
    return ("<|begin_of_text|>" + system + "<|start_header_id|>user<|end_header_id|>\n\n"
            + text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def validate_token_budget(prompt_tokens: int, max_new_tokens: int, profile: dict) -> None:
    if not 1 <= max_new_tokens <= 128:
        raise ValueError("Generation limit must be 1-128 new tokens.")
    if not 0 < prompt_tokens < profile["max_seq_length"]:
        raise ValueError("Prompt exceeds the exported dynamic prefill token bound.")
    if prompt_tokens + max_new_tokens > profile["max_context_length"]:
        raise ValueError("Prompt and generation budget exceed exported context capacity.")


def worker(request: dict) -> dict:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("This test is desktop Linux x86_64 only; it cannot certify a Jetson.")
    profile = json.loads(Path(request["profile"]).read_text())
    reference = json.loads(Path(request["reference"]).read_text())
    assets = json.loads(Path(request["assets"]).read_text())
    for key in ("model_id", "model_revision"):
        if reference[key] != profile[key] or assets[key] != profile[key]:
            raise ValueError("Model/tokenizer reference differs from the selected profile.")
    if reference["executorch_commit"] != profile["executorch_commit"]:
        raise ValueError("Export runtime source revision differs from the profile.")
    version = importlib.metadata.version("executorch")
    if version != profile["executorch_version"] + "+cpu":
        raise ValueError("Use the pinned isolated CPU ExecuTorch environment.")
    model = Path(request["model"])
    tokenizer_path = Path(request["tokenizer"])
    artifact = check_fingerprint(model, reference)
    tokenizer_size, tokenizer_hash = fingerprint(tokenizer_path)
    expected = assets["files"]["original/tokenizer.model"]
    if (tokenizer_size, tokenizer_hash) != (expected["bytes"], expected["sha256"]):
        raise ValueError("Tokenizer fingerprint differs from the pinned source assets.")
    source_text = Path(request["prompt"]).read_text(encoding="utf-8")
    formatted = chat_prompt(source_text, system_text=request.get('system_text'))

    import torch
    import executorch
    from pytorch_tokenizers import CppTiktoken
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
    import executorch.extension.llm.runner._llm_runner as native

    if torch.version.cuda is not None:
        raise ValueError("Expected CPU-only PyTorch tools.")
    torch.set_num_threads(8)
    # The wheel ships these kernels but the runner does not auto-register them.
    kernel_libraries = []
    for relative in ("lib/libexecutorch_kernels_quantized.so",
                     "extension/llm/custom_ops/libcustom_ops_aot_lib.so"):
        library = Path(executorch.__path__[0]) / relative
        torch.ops.load_library(str(library))
        size, digest = fingerprint(library)
        kernel_libraries.append({"relative_path": relative, "bytes": size, "sha256": digest})
    tokenizer = CppTiktoken()
    tokenizer.load(str(tokenizer_path))
    ids = tokenizer.encode(formatted, 0, 0)
    header_count = 3 if request.get('system_text') is not None else 2
    if (ids[0] != 128000 or ids.count(128000) != 1
            or ids.count(128006) != header_count or ids.count(128007) != header_count
            or ids.count(128009) != header_count - 1):
        raise ValueError("Chat markers did not tokenize into the expected Llama IDs.")
    validate_token_budget(len(ids), request["max_new_tokens"], profile)
    output = Path(request["output"])
    generation = {"echo": False, "temperature": 0.0, "warming": False,
                  "max_new_tokens": request["max_new_tokens"], "seq_len": -1,
                  "num_bos": 0, "num_eos": 0}
    native_size, native_hash = fingerprint(Path(native.__file__))
    report = {"scope": "DESKTOP_SMOKE_ONLY", "model_id": profile["model_id"],
              "model_revision": profile["model_revision"], "artifact": artifact,
              "tokenizer": {"bytes": tokenizer_size, "sha256": tokenizer_hash},
              "host": {"system": platform.system(), "architecture": platform.machine(),
                       "platform": platform.platform()},
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "executorch", "pytorch-tokenizers")},
              "native_runner": {"path": native.__file__, "bytes": native_size,
                                "sha256": native_hash},
              "kernel_libraries": kernel_libraries,
              "prompt_text": source_text.strip(), "formatted_prompt": formatted,
              "system_text": request.get('system_text'),
              "prompt_token_ids": ids, "prompt_token_count": len(ids),
              "generation_config": generation, "torch_num_threads": torch.get_num_threads(),
              "offline_status": "NOT_VERIFIED", "jetson_proof": "NOT_VERIFIED",
              "quality": "NOT_EVALUATED", "stats": []}
    with (output / "inputs.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    started = time.monotonic()
    runner = TextLLMRunner(str(model), str(tokenizer_path))
    report["runner_constructor_seconds"] = time.monotonic() - started
    chunks = []
    with (output / "generated.txt").open("x", encoding="utf-8") as stream:
        def token_callback(piece):
            chunks.append(piece)
            stream.write(piece)
            stream.flush()

        def stats_callback(stats):
            # Copy values immediately; the native Stats object belongs to the runner.
            report["stats"].append(json.loads(stats.to_json_string()))

        started = time.monotonic()
        runner.generate(formatted, GenerationConfig(**generation), token_callback, stats_callback)
        report["generate_call_seconds"] = time.monotonic() - started
    text = "".join(chunks)
    report["generated_text"] = text
    report["callback_chunks"] = len(chunks)
    if not text.strip() or not report["stats"]:
        raise ValueError("No nonempty generated output and native statistics captured.")
    stats = report["stats"][-1]
    if (stats["prompt_tokens"] != len(ids)
            or not 0 < stats["generated_tokens"] <= request["max_new_tokens"]):
        raise ValueError("Native token statistics do not match the bounded request.")
    report["status"] = "PC_GENERATION_CAPTURED_REQUIRES_REVIEW"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("artifacts/edge/export_attempt_01/exported/llama3_2_1b_xnnpack.pte"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/llama3_2_1b/original/tokenizer.model"))
    parser.add_argument("--prompt", type=Path, default=Path("configs/edge_smoke_prompt.txt"))
    parser.add_argument("--profile", type=Path, default=Path("configs/edge_llama.json"))
    parser.add_argument("--reference", type=Path, default=Path("manifests/llama3_2_1b_export.json"))
    parser.add_argument("--assets", type=Path, default=Path("manifests/llama3_2_1b_assets.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        request = json.loads(args.worker.read_text())
        result = worker(request)
        with (Path(request["output"]) / "worker_result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(result["status"])
        return
    if not args.output or args.output.exists():
        parser.error("Choose a new --output directory.")
    if not 0 < args.timeout <= 600 or not 1 <= args.max_new_tokens <= 128:
        parser.error("Use timeout 0-600 seconds and 1-128 new tokens.")
    args.output.mkdir(parents=True, exist_ok=False)
    request = {name: str(getattr(args, name).resolve()) for name in
               ("model", "tokenizer", "prompt", "profile", "reference", "assets", "output")}
    request["max_new_tokens"] = args.max_new_tokens
    request_path = args.output / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    command = ["env", "OMP_NUM_THREADS=8", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
               sys.executable, "-B", "-m", "forgesight.edge_pc_smoke", "--worker", str(request_path.resolve())]
    report = {"status": "FAILED_NOT_VERIFIED", "scope": "DESKTOP_SMOKE_ONLY",
              "jetson_proof": "NOT_VERIFIED", "offline_status": "NOT_VERIFIED",
              "generated_text": None, "worker_result": None}
    try:
        result = capture_process(command, args.output, args.timeout)
        report["process"] = result
        worker_result = args.output / "worker_result.json"
        if result["returncode"] == 0 and result["completion_reason"] == "EXITED" and worker_result.is_file():
            report["worker_result"] = json.loads(worker_result.read_text())
            report["generated_text"] = report["worker_result"]["generated_text"]
            report["status"] = report["worker_result"]["status"]
    except OSError as error:
        report["error"] = f"{type(error).__name__}: {error}"
    with (args.output / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "scope", "generated_text", "jetson_proof")}, indent=2))
    if report["status"] == "FAILED_NOT_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
