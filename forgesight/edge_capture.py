"""Capture a native Jetson ExecuTorch runner attempt; never certify compliance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from forgesight.edge_preflight import collect_preflight


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_aarch64_elf(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(20)
    if (len(header) < 20 or header[:4] != b"\x7fELF" or header[4:6] != bytes((2, 1))
            or int.from_bytes(header[18:20], "little") != 183):
        raise ValueError("Runner must be a native little-endian AArch64 ELF executable.")
    if not os.access(path, os.X_OK):
        raise ValueError("Runner is not executable.")


def terminate_process(process) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()


def capture_process(argv: list[str], output: Path, timeout: float) -> dict:
    """Bound a real process and capture logs; this helper does not identify a model."""
    if not 0 < timeout <= 3600:
        raise ValueError("Timeout must be positive and at most one hour.")
    started = time.monotonic()
    peak_rss = None
    reason = "EXITED"
    stdout_path, stderr_path = output / "stdout.log", output / "stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, shell=False,
                                   start_new_session=os.name == "posix")
        try:
            while process.poll() is None:
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text()
                    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
                    if match:
                        peak_rss = max(peak_rss or 0, int(match[1]))
                except OSError:
                    pass
                if time.monotonic() - started >= timeout:
                    reason = "TIMEOUT"
                    terminate_process(process)
                    break
                if stdout_path.stat().st_size + stderr_path.stat().st_size > 16 * 1024 * 1024:
                    reason = "OUTPUT_LIMIT"
                    terminate_process(process)
                    break
                time.sleep(0.05)
        finally:
            if process.poll() is None:
                terminate_process(process)
    return {"argv": argv, "returncode": process.returncode, "completion_reason": reason,
            "process_wall_seconds": time.monotonic() - started,
            "sampled_peak_rss_kib": peak_rss, "rss_sampling_interval_seconds": 0.05,
            "memory_note": "Sampled process RSS; may miss peaks and excludes other processes.",
            "stdout_bytes": stdout_path.stat().st_size, "stderr_bytes": stderr_path.stat().st_size}


def source_revision(checkout: Path) -> str | None:
    if not (checkout / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=10, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def capture_run(profile: dict, runner: Path, model: Path, tokenizer: Path, prompt: Path,
                checkout: Path, output: Path, timeout: float = 180,
                max_new_tokens: int = 64, network_state: str = "unknown") -> dict:
    if output.exists():
        raise FileExistsError(f"Capture output exists: {output}")
    if not 0 < timeout <= 3600 or not 1 <= max_new_tokens <= 128:
        raise ValueError("Invalid timeout or generation token limit.")
    preflight = collect_preflight()
    observed_commit = source_revision(checkout)
    blockers = []
    if preflight["host_kind"] != "JETSON_ORIN_NANO_CANDIDATE":
        blockers.append("Device execution refused: host is not an identified Jetson Orin Nano ARM64 target.")
    if observed_commit != profile["executorch_commit"]:
        blockers.append("Pinned runtime source checkout is not verified.")
    files = {"runner": runner, "model": model, "tokenizer": tokenizer, "prompt": prompt}
    for label, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            blockers.append(f"Missing or empty {label} file.")
    if model.suffix != ".pte":
        blockers.append("Expected an ExecuTorch .pte artifact.")
    if runner.is_file():
        try:
            require_aarch64_elf(runner)
        except (OSError, ValueError) as error:
            blockers.append(str(error))
    if prompt.is_file() and prompt.stat().st_size > 4096:
        blockers.append("Smoke prompt exceeds the 4096-byte collection limit.")
    report = {"schema_version": 1, "status": "BLOCKED_NOT_VERIFIED",
              "requested_model_id": profile["model_id"], "requested_model_revision": profile["model_revision"],
              "requested_executorch_commit": profile["executorch_commit"],
              "observed_source_commit": observed_commit, "preflight_before": preflight,
              "network_state_operator_reported": network_state, "offline_status": "NOT_VERIFIED",
              "artifacts": {}, "blockers": blockers, "process": None,
              "raw_stdout_text": None, "generated_text_status": "NOT_RUN",
              "stage_2_gate": "NOT_VERIFIED",
              "review_required": "Verify model/export provenance, linked runtime/backend, actual generated text, and device/network evidence."}
    output.mkdir(parents=True, exist_ok=False)
    if not blockers:
        report["artifacts"] = {label: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                                       "sha256": file_sha256(path)} for label, path in files.items()}
        argv = [str(runner.resolve()), "--model_path", str(model.resolve()),
                "--tokenizer_path", str(tokenizer.resolve()), "--prompt_file", str(prompt.resolve()),
                "--max_new_tokens", str(max_new_tokens), "--temperature", "0", "--num_bos", "1"]
        try:
            result = capture_process(argv, output, timeout)
            report["process"] = result
            report["raw_stdout_text"] = (output / "stdout.log").read_text(errors="replace")
            report["status"] = "CAPTURED_NOT_VERIFIED" if (
                result["returncode"] == 0 and result["completion_reason"] == "EXITED"
                and report["raw_stdout_text"].strip()) else "FAILED_NOT_VERIFIED"
            report["generated_text_status"] = "RAW_STDOUT_REQUIRES_REVIEW"
        except OSError as error:
            report["status"] = "FAILED_NOT_VERIFIED"
            report["capture_error"] = f"{type(error).__name__}: {error}"
        report["preflight_after"] = collect_preflight()
    with (output / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/edge_llama.json"))
    parser.add_argument("--checkout", type=Path, default=Path("data/tools/executorch"))
    parser.add_argument("--runner", type=Path, default=Path("data/tools/executorch/cmake-out/examples/models/llama/llama_main"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/edge/export_recipe_v1/exported/llama3_2_1b_xnnpack.pte"))
    parser.add_argument("--tokenizer", type=Path, default=Path("models/llama3_2_1b/original/tokenizer.model"))
    parser.add_argument("--prompt", type=Path, default=Path("configs/edge_smoke_prompt.txt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/edge/jetson_capture_v1"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--network-state", choices=("unknown", "connected", "physically_disconnected"), default="unknown")
    args = parser.parse_args()
    try:
        report = capture_run(json.loads(args.profile.read_text()), args.runner, args.model, args.tokenizer,
                             args.prompt, args.checkout, args.output, args.timeout, args.max_new_tokens,
                             args.network_state)
        print(json.dumps({key: report[key] for key in ("status", "blockers", "stage_2_gate",
                                                      "generated_text_status", "process")}, indent=2))
        if report["status"] != "CAPTURED_NOT_VERIFIED":
            parser.exit(2, "Actual device inference is not verified.\n")
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
