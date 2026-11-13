"""Persistent hash-verified Llama/ExecuTorch advisory worker for the Jetson.

The measured advisory path pays its full fixed cost on every request: three
nested Python processes, torch/executorch imports and a fresh SHA-256 of the
802 MB model each time (native process-wall p50 20.690 s for 33 generated
tokens in manifests/jetson_offline_advice_latency_v1.json, while the
generation itself measured about 2 s in manifests/jetson_llama_generation_v1.json;
different profiles, so the series is the baseline and the probe only
decomposes it). This worker verifies the target, assets and runner once at
startup, then serves repeated advisory generations over a Unix socket. The
host contract is unchanged: the caller still runs prepare_advisory before and
parse_advisory after, grades stay deterministic, and the per-request
inspection/SQLite immutability re-check stays in the controller.

State isolation: nothing about one request is carried into the next. The
worker keeps no reply cache; every request re-tokenizes and re-generates.
Grounded in the installed runtime source (executorch 1.5.1,
extension/llm/runner/text_llm_runner.cpp): generate() does NOT reset state -
pos_ accumulates across calls, so a naive second call would continue inside
the previous prompt's KV cache. TextLLMRunner::reset() (pos_ = 0, stats
reset) is Python-bound as "Reset the runner state and KV cache"
(extension/llm/runner/pybindings.cpp). The worker therefore calls the
runtime's reset() before EVERY request; KV rows past a sequence's length are
never attended after prefill restarts at position 0. As belt and braces, the
device acceptance check still compares one worker-mode advisory against the
cold-process output for the same prompt (temperature 0) before adoption.

Artifact safety: the model/tokenizer are stat-checked on every request
(size, mtime, inode). A replaced artifact fails the request and the worker
must be restarted; a restart re-runs full hash verification and refuses to
serve on mismatch, in which case the controller's cold subprocess path
remains the fallback.

Protocol: one newline-terminated UTF-8 JSON object per request, one back,
one at a time. Request fields: prompt_text, max_new_tokens, and optional
system_text. The same total-read-deadline and peer-uid rules as the vision
worker apply. This module is self-contained so it does not depend on
unmerged vision-worker changes; the duplication is deliberate.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import stat
import struct
import time

from forgesight.edge_jetson_smoke import chat_prompt

MAX_REQUEST_BYTES = 64 * 1024      # prompts are bounded to 4096 + 2048 bytes
MAX_REPLY_BYTES = 1024 * 1024      # a note is <= 600 chars; stats are small
MAX_PROMPT_BYTES = 4096            # chat_prompt's own bound
MAX_SYSTEM_BYTES = 2048            # chat_prompt's own bound
MAX_NEW_TOKENS = 64                # edge_jetson_smoke validate_budget bound


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError("Expected a JSON request object.")
    unknown = set(request) - {"prompt_text", "system_text", "max_new_tokens"}
    if unknown:
        raise ValueError(f"Unexpected request fields: {sorted(unknown)}")
    prompt = request.get("prompt_text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Expected a nonempty prompt string.")
    if len(prompt.strip().encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError("Prompt exceeds the bounded size.")
    system = request.get("system_text")
    if system is not None:
        if not isinstance(system, str) or not system.strip():
            raise ValueError("Expected a nonempty system string.")
        if len(system.strip().encode("utf-8")) > MAX_SYSTEM_BYTES:
            raise ValueError("System prompt exceeds the bounded size.")
    tokens = request.get("max_new_tokens")
    if type(tokens) is not int or not 1 <= tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"Generation limit must be 1-{MAX_NEW_TOKENS} new tokens.")
    return {"prompt_text": prompt, "system_text": system, "max_new_tokens": tokens}


class Worker:
    """Holds one verified runtime; each request is independent."""

    def __init__(self, runtime):
        # The runtime object owns target/asset verification, imports and the
        # loaded runner. Interface: reset() (drops generation state between
        # requests), token_count(text) -> int, generate(formatted,
        # max_new_tokens) -> (text, stats dict), evidence -> dict,
        # artifact_state() -> hashable, profile -> dict.
        self.runtime = runtime
        self.startup_state = runtime.artifact_state()
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def handle(self, raw_request):
        request = validate_request(raw_request)
        if self.runtime.artifact_state() != self.startup_state:
            raise RuntimeError("Model/tokenizer/runtime artifact changed; restart the worker.")
        # Drop any generation state before this request; see the module
        # docstring for why generate() alone does not isolate requests.
        self.runtime.reset()
        formatted = chat_prompt(request["prompt_text"], request["system_text"])
        prompt_tokens = self.runtime.token_count(formatted)
        profile = self.runtime.profile
        if not 0 < prompt_tokens < profile["max_seq_length"]:
            raise ValueError("Prompt exceeds the exported prefill bound.")
        if prompt_tokens + request["max_new_tokens"] > profile["max_context_length"]:
            raise ValueError("Prompt plus answer exceeds the exported context.")
        started = time.perf_counter()
        text, stats = self.runtime.generate(formatted, request["max_new_tokens"])
        generate_ms = (time.perf_counter() - started) * 1000
        if (stats.get("prompt_tokens") != prompt_tokens
                or not 0 < stats.get("generated_tokens", 0) <= request["max_new_tokens"]):
            raise ValueError("Native token statistics do not match the bounded request.")
        return {"status": "GENERATED_BY_PERSISTENT_WORKER",
                "worker_mode": "PERSISTENT_PROCESS",
                "worker_started_utc": self.started_utc,
                "runtime_evidence": self.runtime.evidence,
                "prompt_token_count": prompt_tokens,
                "max_new_tokens": request["max_new_tokens"],
                "generated_text": text,
                "stats": stats, "generate_ms": generate_ms,
                "state_isolation": "no reply cache; every request re-tokenized and re-generated",
                "timing_scope": "persistent worker: tokenize+generate per request; verification/imports/model load once at worker start"}


def serve(socket_path, worker, idle_timeout_seconds=None, request_read_timeout_seconds=30):
    if not 0 < request_read_timeout_seconds <= 300:
        raise ValueError("Request read timeout must be between 0 and 300 seconds.")
    socket_path = Path(socket_path)
    if socket_path.exists():
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(str(socket_path))
        except OSError:
            socket_path.unlink()
        else:
            probe.close()
            raise ValueError("Choose a new worker socket path.")
        finally:
            probe.close()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        os.chmod(socket_path, stat.S_IRUSR | stat.S_IWUSR)
        server.listen(4)
        if idle_timeout_seconds is not None:
            server.settimeout(idle_timeout_seconds)
        while True:
            try:
                connection, _ = server.accept()
            except socket.timeout:
                return
            with connection:
                try:
                    _pid, peer_uid, _gid = struct.unpack(
                        "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                except OSError:
                    continue
                if peer_uid != os.getuid():
                    continue  # same-user only; the socket mode already restricts this
                # One total deadline for the whole request line: a stalled or
                # dripping client must not hold the only advisory worker.
                deadline = time.monotonic() + request_read_timeout_seconds
                chunks, remaining = [], MAX_REQUEST_BYTES
                try:
                    while remaining > 0:
                        left = deadline - time.monotonic()
                        if left <= 0:
                            raise socket.timeout()
                        connection.settimeout(left)
                        block = connection.recv(min(65536, remaining))
                        if not block:
                            break
                        chunks.append(block)
                        remaining -= len(block)
                        if b"\n" in block:
                            break
                except socket.timeout:
                    continue  # stalled or dripping client; serve the next connection
                raw = b"".join(chunks).split(b"\n", 1)[0]
                try:
                    request = json.loads(raw.decode("utf-8"))
                    report = worker.handle(request)
                    payload = {"status": "OK", "report": report}
                except Exception as error:
                    payload = {"status": "ERROR",
                               "error": {"type": type(error).__name__, "message": str(error)}}
                try:
                    connection.sendall(json.dumps(payload, allow_nan=False).encode("utf-8") + b"\n")
                except OSError:
                    pass  # client went away mid-reply; keep serving the next request
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


def request(socket_path, request_dict, timeout_seconds=180, connect_retries=50):
    """Client: send one advisory request, return the report or raise."""
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(timeout_seconds)
        for attempt in range(connect_retries):
            try:
                connection.connect(str(socket_path))
                break
            except (ConnectionRefusedError, FileNotFoundError):
                if attempt == connect_retries - 1:
                    raise
                time.sleep(0.02)
        connection.sendall(json.dumps(request_dict, allow_nan=False).encode("utf-8") + b"\n")
        chunks, remaining = [], MAX_REPLY_BYTES
        while remaining > 0:
            block = connection.recv(min(65536, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
            if b"\n" in block:
                break
        else:
            raise ConnectionError("Advisory worker reply exceeded the size bound")
    finally:
        connection.close()
    if not chunks:
        raise ConnectionError("Advisory worker closed without a response")
    payload = json.loads(b"".join(chunks).split(b"\n", 1)[0].decode("utf-8"))
    if payload.get("status") != "OK":
        error = payload.get("error") or {}
        raise RuntimeError(f"Advisory worker error: {error.get('type')}: {error.get('message')}")
    return payload["report"]


def build_jetson_runtime(root):
    """Jetson-only: verify the device and assets once, load the runner once.

    Cannot run on PC: verify_target requires the physical Orin Nano and the
    private 3.12.14 venv. Raises on any mismatch; the caller falls back to
    the cold subprocess path.
    """
    import importlib.metadata
    from forgesight import edge_jetson_smoke as smoke

    root = Path(root).resolve()
    spec = {"root": str(root), "model": str(root / "model/llama3_2_1b_xnnpack.pte"),
            "tokenizer": str(root / "model/tokenizer.model"),
            "profile": str(root / "model/edge_llama.json"),
            "export": str(root / "model/llama3_2_1b_export.json"),
            "assets": str(root / "model/llama3_2_1b_assets.json")}
    host, profile, files, model_hash, tokenizer_hash = smoke.verify_inputs(spec)
    if (importlib.metadata.version("torch") != "2.14.0+cpu"
            or importlib.metadata.version("executorch") != "1.5.1+cpu"
            or importlib.metadata.version("pytorch-tokenizers") != "1.6.0"):
        raise ValueError("Unexpected runtime package versions.")
    import torch
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
    import executorch
    from pytorch_tokenizers import CppTiktoken
    if torch.version.cuda is not None:
        raise ValueError("Expected the separate CPU Torch build.")
    torch.set_num_threads(2)
    for relative in ("lib/libexecutorch_kernels_quantized.so",
                     "extension/llm/custom_ops/libcustom_ops_aot_lib.so"):
        torch.ops.load_library(str(Path(next(iter(executorch.__path__))) / relative))
    import executorch.extension.llm.runner._llm_runner as native_runner
    tokenizer = CppTiktoken()
    tokenizer.load(str(files["tokenizer"]))
    runner = TextLLMRunner(str(files["model"]), str(files["tokenizer"]))
    watched = [files["model"], files["tokenizer"], Path(native_runner.__file__)]
    watched += [Path(next(iter(executorch.__path__))) / relative
                for relative in ("lib/libexecutorch_kernels_quantized.so",
                                 "extension/llm/custom_ops/libcustom_ops_aot_lib.so")]

    class JetsonRuntime:
        def __init__(self):
            self.profile = profile
            self.evidence = {"host": host, "artifact": model_hash,
                             "tokenizer": tokenizer_hash,
                             "model_id": profile["model_id"],
                             "model_revision": profile["model_revision"],
                             "backend_requested": profile["backend"]}

        def reset(self):
            runner.reset()

        def artifact_state(self):
            state = []
            for path in watched:
                info = Path(path).stat()
                state.append((str(path), info.st_size, info.st_mtime_ns, info.st_ino))
            return tuple(state)

        def token_count(self, text):
            return len(tokenizer.encode(text, 0, 0))

        def generate(self, formatted, max_new_tokens):
            pieces, collected = [], []
            config = GenerationConfig(echo=False, temperature=0.0, warming=False,
                                      max_new_tokens=max_new_tokens, seq_len=-1,
                                      num_bos=0, num_eos=0)
            runner.generate(formatted, config, pieces.append,
                            lambda stats: collected.append(json.loads(stats.to_json_string())))
            stats = collected[-1] if collected else {}
            return "".join(pieces), stats

    return JetsonRuntime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--idle-timeout-seconds", type=float, default=None)
    parser.add_argument("--request-read-timeout-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.idle_timeout_seconds is not None and not 0 < args.idle_timeout_seconds <= 3600:
        parser.error("Idle timeout must be between 0 and 3600 seconds.")
    runtime = build_jetson_runtime(args.llama_root)
    print(json.dumps({"status": "WORKER_READY",
                      "model_sha256": runtime.evidence["artifact"]["sha256"]}), flush=True)
    serve(args.socket, Worker(runtime), args.idle_timeout_seconds,
          args.request_read_timeout_seconds)


if __name__ == "__main__":
    main()
