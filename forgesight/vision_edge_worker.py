"""Persistent hash-checked frozen EfficientAD TorchScript inference worker for Jetson.

Serves the same one-image inference as vision_edge_once.py over a Unix domain
socket, so the vision Python, Torch import and model load happen once per UI
session instead of once per inspection. The per-request validations, output
files and report fields match vision_edge_once.py exactly; the cold one-shot
script remains the fallback and is unchanged.

Protocol: one newline-terminated UTF-8 JSON object per request on a connected
socket, one newline-terminated JSON object back. Requests are handled one at
a time. Request fields mirror the vision_edge_once.py CLI:
  tensor, tensor_sha256, threshold, output, map_output (optional),
  source_map_output (optional), source_size (optional [HEIGHT, WIDTH]).
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import stat
import struct
import time


MAX_REQUEST_BYTES = 1024 * 1024
MAX_SOURCE_SIDE = 4096
# A prepared tensor is one FP32 256x256 RGB image: 786,432 data bytes, 786,560
# as written by np.save here. The 1 MiB bound leaves room for .npy header
# format/version differences while refusing anything that is not a one-image
# tensor before hashing or loading the whole file.
MAX_TENSOR_BYTES = 1024 * 1024
MAX_REPLY_BYTES = 16 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_new_path(value, allow_existing=False):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected an output path string.")
    path = Path(value)
    if not allow_existing and path.exists():
        raise ValueError("Choose a new output file.")
    return path


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError("Expected a JSON request object.")
    allowed = {"tensor", "tensor_sha256", "threshold", "output",
               "map_output", "source_map_output", "source_size"}
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"Unexpected request fields: {sorted(unknown)}")
    tensor = Path(request.get("tensor", ""))
    if not tensor.is_file():
        raise ValueError("Prepared tensor is missing.")
    if tensor.stat().st_size > MAX_TENSOR_BYTES:
        raise ValueError("Prepared tensor exceeds the one-image size bound.")
    tensor_sha = request.get("tensor_sha256")
    if not isinstance(tensor_sha, str) or len(tensor_sha) != 64:
        raise ValueError("Expected a tensor SHA-256.")
    threshold = request.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    output = _bounded_new_path(request.get("output"))
    map_output = request.get("map_output")
    if map_output is not None:
        map_output = _bounded_new_path(map_output)
    source_map = request.get("source_map_output")
    source_size = request.get("source_size")
    if (source_map is None) != (source_size is None):
        raise ValueError("Source-map output and source size must be supplied together.")
    if source_map is not None:
        source_map = _bounded_new_path(source_map)
        if (not isinstance(source_size, list) or len(source_size) != 2
                or any(type(v) is not int for v in source_size)
                or min(source_size) <= 0 or max(source_size) > MAX_SOURCE_SIDE):
            raise ValueError("Choose a bounded positive source size.")
    return {"tensor": tensor, "tensor_sha256": tensor_sha, "threshold": float(threshold),
            "output": output, "map_output": map_output,
            "source_map_output": source_map, "source_size": source_size}


class Worker:
    """Loads the frozen model once; each request re-verifies its inputs."""

    def __init__(self, model_path, model_sha256, threads=2):
        if sha256(model_path) != model_sha256:
            raise ValueError("Frozen model hash mismatch")
        if not 1 <= threads <= 4:
            raise ValueError("Expected one to four Torch threads")
        self.model_path = str(model_path)
        self.model_sha256 = model_sha256
        self.threads = threads
        import torch
        torch.set_num_threads(threads)
        started = time.perf_counter()
        self.model = torch.jit.load(self.model_path, map_location="cpu").eval()
        self.model_load_ms = (time.perf_counter() - started) * 1000
        self.torch_version = torch.__version__
        self._torch = torch

    def handle(self, raw_request):
        import numpy as np
        torch = self._torch
        request = validate_request(raw_request)
        if sha256(request["tensor"]) != request["tensor_sha256"]:
            raise ValueError("Prepared tensor hash mismatch")
        prepared = np.load(request["tensor"], allow_pickle=False)
        if (prepared.shape != (1, 3, 256, 256) or prepared.dtype != np.float32
                or not np.isfinite(prepared).all()):
            raise ValueError("Expected one finite FP32 RGB 256x256 tensor.")
        tensor = torch.from_numpy(prepared)
        started = time.perf_counter()
        with torch.inference_mode():
            anomaly_map, score_tensor = self.model(tensor)
        forward_ms = (time.perf_counter() - started) * 1000
        score = float(score_tensor.item())
        if (anomaly_map.shape != (1, 1, 256, 256) or not math.isfinite(score)
                or not torch.isfinite(anomaly_map).all()):
            raise ValueError("Invalid model output.")
        report = {"status": "ONE_IMAGE_INFERENCE_VERIFIED", "runtime": "TorchScript CPU FP32",
                  "torch_version": self.torch_version, "torch_threads": self.threads,
                  "worker_mode": "PERSISTENT_PROCESS",
                  "model_sha256": self.model_sha256, "tensor_sha256": request["tensor_sha256"],
                  "model_load_ms": self.model_load_ms, "model_forward_ms": forward_ms,
                  "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "score": score, "threshold": request["threshold"],
                  "candidate": "ANOMALY_CANDIDATE" if score > request["threshold"] else "NORMAL_CANDIDATE",
                  "map_min": float(anomaly_map.min()), "map_max": float(anomaly_map.max()),
                  "map_shape": list(anomaly_map.shape),
                  "timing_scope": "persistent worker: forward only per request; model load once at worker start; excludes image decode, preprocessing, UI, logging and Llama"}
        if request["map_output"] is not None:
            request["map_output"].parent.mkdir(parents=True, exist_ok=True)
            with request["map_output"].open("xb") as stream:
                np.save(stream, anomaly_map.squeeze().numpy().astype(np.float32), allow_pickle=False)
            report["map_file"] = str(request["map_output"])
            report["map_sha256"] = sha256(request["map_output"])
        if request["source_map_output"] is not None:
            from torch.nn import functional as F
            resized = F.interpolate(anomaly_map, size=tuple(request["source_size"]),
                                    mode="bilinear", align_corners=False)
            request["source_map_output"].parent.mkdir(parents=True, exist_ok=True)
            with request["source_map_output"].open("xb") as stream:
                np.save(stream, resized.squeeze().numpy().astype(np.float32), allow_pickle=False)
            report["source_map_file"] = str(request["source_map_output"])
            report["source_map_sha256"] = sha256(request["source_map_output"])
            report["source_map_shape"] = list(resized.shape)
        request["output"].parent.mkdir(parents=True, exist_ok=True)
        with request["output"].open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        return report


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
                # Bound the whole request transfer, including clients that keep
                # sending bytes just before an individual recv would time out.
                read_deadline = time.monotonic() + request_read_timeout_seconds
                chunks, remaining = [], MAX_REQUEST_BYTES
                try:
                    while remaining > 0:
                        seconds_left = read_deadline - time.monotonic()
                        if seconds_left <= 0:
                            raise socket.timeout("Request read deadline exceeded")
                        connection.settimeout(seconds_left)
                        block = connection.recv(min(65536, remaining))
                        if not block:
                            break
                        chunks.append(block)
                        remaining -= len(block)
                        if b"\n" in block:
                            break
                except socket.timeout:
                    continue  # stalled client; serve the next connection
                # The transfer deadline does not bound model inference or reply
                # writes. Preserve the separate reply timeout after reading.
                connection.settimeout(request_read_timeout_seconds)
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
    """Client: send one inference request, return the report or raise."""
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
            raise ConnectionError("Vision worker reply exceeded the size bound")
    finally:
        connection.close()
    if not chunks:
        raise ConnectionError("Vision worker closed without a response")
    payload = json.loads(b"".join(chunks).split(b"\n", 1)[0].decode("utf-8"))
    if payload.get("status") != "OK":
        error = payload.get("error") or {}
        raise RuntimeError(f"Vision worker rejected the request: {error.get('type')}: {error.get('message')}")
    return payload["report"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--idle-timeout-seconds", type=float, default=None,
                        help="Exit after this many seconds without a connection; default runs forever.")
    parser.add_argument("--request-read-timeout-seconds", type=float, default=30,
                        help="Drop a client whose request line takes longer than this to arrive; default 30.")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("Expected one to four Torch threads")
    if not 0 < args.request_read_timeout_seconds <= 300:
        parser.error("Request read timeout must be between 0 and 300 seconds")
    worker = Worker(args.model, args.model_sha256, args.threads)
    print(json.dumps({"status": "WORKER_READY", "model_sha256": args.model_sha256,
                      "model_load_ms": worker.model_load_ms, "socket": str(args.socket)}), flush=True)
    serve(args.socket, worker, args.idle_timeout_seconds, args.request_read_timeout_seconds)


if __name__ == "__main__":
    main()
