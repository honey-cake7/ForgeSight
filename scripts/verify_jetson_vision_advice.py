"""Check real vision-to-ExecuTorch advice in the offline Jetson operator app."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
from time import monotonic
from urllib.request import Request, urlopen

from forgesight.assist_vision import VISION_OUTPUT_POLICY
from forgesight.storage import InspectionStore
from scripts.verify_jetson_defect_type import state, wait_for


CASES = {
    "bent": ("test/bent/004.png", "LOW"),
    "flip": ("test/flip/005.png", "UNASSESSED"),
    "scratch": ("test/scratch/013.png", "MEDIUM"),
    "discoloration": ("test/color/012.png", "HIGH"),
    "unknown": ("test/bent/000.png", "UNASSESSED"),
}


def submit(base, endpoint, payload, headers):
    request = Request(base + endpoint, data=payload, headers={"Origin": base, **headers})
    with urlopen(request, timeout=5) as response:
        if response.status != 202:
            raise RuntimeError(f"Unexpected {endpoint} status: {response.status}")


def verify_case(base, output, store, label, path, payload, level):
    submit(base, "/api/upload-diagnostic", payload,
           {"Content-Type": "image/png", "X-Upload-Id": label, "X-Mvtec-Benchmark": "yes"})

    def analyzed():
        snapshot = state(base)
        latest = snapshot["latest"] or {}
        if latest.get("part_id") == label and not snapshot["busy"]:
            return latest
        return None

    latest = wait_for(analyzed)
    inspection_id = latest["inspection_id"]
    record = store.get_inspection(inspection_id)
    if (latest["status"] != "ANALYZED" or record["assessment"]["defect_type"] != label
            or record["assessment"]["severity"]["level"] != level):
        raise RuntimeError(f"Unexpected vision result for {label}")

    started = monotonic()
    submit(base, "/api/advice", b"", {})
    submission_ms = (monotonic() - started) * 1000
    pending = False

    def complete():
        nonlocal pending
        snapshot = state(base)
        if snapshot["latest"].get("inspection_id") != inspection_id:
            raise RuntimeError("The advisory lost its inspection reference.")
        advice = snapshot["advice"]
        pending |= advice["status"] == "PENDING"
        if advice["status"] in ("GENERATED_REQUIRES_HUMAN_REVIEW", "ADVICE_UNAVAILABLE"):
            return advice
        return None

    displayed = wait_for(complete, seconds=90)
    elapsed_ms = (monotonic() - started) * 1000
    directory = output / "advice" / inspection_id
    result_file = directory / "advice_report.json"
    result = json.loads(result_file.read_text()) if result_file.is_file() else {}
    if displayed["status"] != "GENERATED_REQUIRES_HUMAN_REVIEW":
        reason = result.get("validation_error") or result.get("runtime_error") or result.get("input_refusal")
        raise RuntimeError(f"{label} advice failed: {reason}")
    evidence = json.loads((directory / "evidence.json").read_text())
    advisory = result["advisory"]
    response = advisory["response"]
    vision = evidence["evidence"]["E1"]["vision_assessment"]
    native = result["native"]
    if (evidence["output_policy_version"] != VISION_OUTPUT_POLICY
            or vision["type"] != label or vision["severity"] != level
            or result["llama_runtime_mode"] != "PERSISTENT_WORKER"
            or native["status"] != "GENERATED_BY_PERSISTENT_WORKER"
            or advisory["jetson_verified"] is not True or not pending
            or not result["inspection_and_sqlite_unchanged"]
            or record != store.get_inspection(inspection_id)
            or displayed["advisory"]["note"] != response["generated_note"]
            or displayed["advisory"]["note_origin"] != response["note_origin"]):
        raise RuntimeError(f"Advisory provenance or display differed for {label}")
    return {"image": path, "grade": latest["grade"], "vision": vision, "choice": response["model_action_choice"],
            "note": response["generated_note"], "runtime": result["llama_runtime_mode"],
            "advice_report_sha256": hashlib.sha256(result_file.read_bytes()).hexdigest(),
            "llama_model_sha256": native["model_sha256"], "prompt_tokens": native["prompt_tokens"],
            "generated_tokens": native["generation_stats"][0]["generated_tokens"],
            "pending_observed": pending, "submission_ms": submission_ms, "advice_ready_ms": elapsed_ms,
            "generation_request_ms": result["controller_runtime_wall_ms"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "model", "vision-python", "dataset-root", "llama-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    device = Path("/proc/device-tree/model").read_bytes().decode().rstrip("\x00")
    interfaces = [name for _, name in socket.if_nameindex()]
    if platform.machine() != "aarch64" or "Jetson Orin Nano" not in device or interfaces != ["lo"]:
        parser.error("Run on the Jetson inside a network namespace containing only loopback.")
    hashes = {row["path"]: row["sha256"] for row in json.loads(
        Path("manifests/metal_nut_v1.json").read_text())["samples"]}
    args.output.mkdir()
    database = args.output / "inspections.sqlite3"
    port = 8787
    command = [sys.executable, "-B", "-m", "forgesight.jetson_live_ui",
               "--port", str(port), "--camera-index", "99", "--model", str(args.model),
               "--vision-python", str(args.vision_python), "--llama-root", str(args.llama_root),
               "--dataset-root", str(args.dataset_root), "--database", str(database),
               "--runs-root", str(args.output / "records"), "--advice-root", str(args.output / "advice"),
               "--llama-worker-socket", str((args.output / "llama.sock").resolve()),
               "--vision-worker-socket", str((args.output / "vision.sock").resolve()),
               "--defect-type-profile", "configs/defect_type_mvtec_four_v1.json"]
    base = f"http://127.0.0.1:{port}"
    report = {"device": device, "interfaces": interfaces, "cases": {}}
    with (args.output / "app.log").open("w") as log, subprocess.Popen(command, stdout=log, stderr=log) as app:
        try:
            def ready():
                if app.poll() is not None:
                    raise RuntimeError("Operator app exited. See app.log.")
                try:
                    return state(base)
                except OSError:
                    return None

            wait_for(ready, seconds=80)
            store = InspectionStore(database, create=False)
            for label, (path, level) in CASES.items():
                payload = (args.dataset_root / path).read_bytes()
                if hashlib.sha256(payload).hexdigest() != hashes[path]:
                    raise ValueError(f"Benchmark image hash mismatch: {path}")
                report["cases"][label] = verify_case(base, args.output, store, label, path, payload, level)
        finally:
            if app.poll() is None:
                app.send_signal(signal.SIGINT)
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
