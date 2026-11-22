"""Compare the Jetson operator app with and without the preview type profile."""

import argparse
import json
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
from time import monotonic, sleep
from urllib.request import Request, urlopen

from forgesight.storage import InspectionStore


CASES = (
    ("color_replay", "test/color/009.png", False, "discoloration"),
    ("color_upload", "test/color/009.png", True, "discoloration"),
    ("color_repeat", "test/color/009.png", True, "discoloration"),
    ("bent", "test/bent/017.png", False, "unknown"),
    ("flip", "test/flip/012.png", False, "unknown"),
    ("scratch", "test/scratch/020.png", False, None),
)


def wait_for(check, seconds=35):
    deadline = monotonic() + seconds
    while monotonic() < deadline:
        result = check()
        if result:
            return result
        sleep(0.2)
    raise RuntimeError("Jetson operator app did not reach the expected state.")


def state(base):
    with urlopen(base + "/api/state", timeout=5) as response:
        return json.load(response)


def exercise(args, name, enabled):
    run = args.output / name
    run.mkdir()
    database = run / "inspections.sqlite3"
    command = [sys.executable, "-B", "-m", "forgesight.jetson_live_ui",
               "--port", str(args.port), "--model", str(args.model),
               "--vision-python", str(args.vision_python), "--camera-index", "99",
               "--database", str(database), "--runs-root", str(run / "records"),
               "--advice-root", str(run / "advice"), "--dataset-root", str(args.dataset_root),
               "--vision-worker-socket", str(run / "vision.sock")]
    if enabled:
        command += ["--defect-type-profile", "configs/defect_type_preview_v1.json"]
    base = f"http://127.0.0.1:{args.port}"
    with (run / "app.log").open("w") as log, subprocess.Popen(command, stdout=log, stderr=log) as app:
        try:
            def ready():
                if app.poll() is not None:
                    raise RuntimeError(f"Operator app exited; see {run / 'app.log'}")
                try:
                    return state(base)
                except OSError:
                    return None

            wait_for(ready)
            with urlopen(base + "/", timeout=5) as response:
                if b'id="defect-assessment"' not in response.read():
                    raise RuntimeError("The staged app did not serve the updated result UI.")
            store = InspectionStore(database, create=False)
            facts = []
            for part, dataset_path, upload, expected in CASES:
                if upload:
                    endpoint = "/api/upload-diagnostic"
                    payload = (args.dataset_root / dataset_path).read_bytes()
                    headers = {"Origin": base, "Content-Type": "image/png", "X-Upload-Id": part,
                               "X-Mvtec-Benchmark": "yes"}
                else:
                    endpoint = "/api/replay-local"
                    payload = json.dumps({"replay_id": part, "dataset_path": dataset_path}).encode()
                    headers = {"Origin": base, "Content-Type": "application/json"}
                with urlopen(Request(base + endpoint, data=payload, headers=headers), timeout=5) as response:
                    if response.status != 202:
                        raise RuntimeError(f"Inspection submission failed: {part}")

                def finished():
                    current = state(base)
                    latest = current["latest"] or {}
                    return current if latest.get("part_id") == part and not current["busy"] else None

                current = wait_for(finished)
                latest = current["latest"]
                if latest["status"] != "ANALYZED":
                    raise RuntimeError(f"Inspection failed: {latest}")
                record = store.get_inspection(latest["inspection_id"])
                history = next(row for row in current["history"] if row["inspection_id"] == record["inspection_id"])
                assessment = record.get("assessment")
                expected_type = expected if enabled else None
                observed_type = assessment["defect_type"] if assessment else None
                if (observed_type != expected_type or latest["assessment"] != assessment
                        or history["assessment"] != assessment or record["predicted_defect_type"] != "unknown"):
                    raise RuntimeError(f"Assessment differed across expected result, UI, history and SQLite: {part}")
                facts.append({"part": part, "type": observed_type, "score": record["score"],
                              "candidate": record["candidate"], "decision": record["decision"],
                              "model_map_sha256": record["artifacts"]["score_map_source_sha256"],
                              "overlay_sha256": record["artifacts"]["response_overlay_sha256"],
                              "feature_ms": record["timings"].get("defect_type_ms"),
                              "model_forward_ms": record["timings"]["model_forward_ms"]})
            return facts
        finally:
            # SIGINT lets the service shut down workers and release sockets.
            if app.poll() is None:
                app.send_signal(signal.SIGINT)
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vision-python", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8783)
    args = parser.parse_args()
    device = Path("/proc/device-tree/model").read_bytes().decode().rstrip("\x00")
    interfaces = [name for _, name in socket.if_nameindex()]
    if platform.machine() != "aarch64" or "Jetson Orin Nano" not in device or interfaces != ["lo"]:
        parser.error("Run on the Jetson inside a network namespace containing only loopback.")
    args.output.mkdir(parents=True)
    baseline = exercise(args, "baseline", False)
    enabled = exercise(args, "enabled", True)
    for original, updated in zip(baseline, enabled):
        for key in ("score", "candidate", "decision", "model_map_sha256", "overlay_sha256"):
            if original[key] != updated[key]:
                raise RuntimeError(f"Enabled profile changed {key} for {original['part']}")

    def concise(facts):
        return [{**row, "decision": row["decision"]["grade"]} for row in facts]

    report = {"device": device, "application_namespace_interfaces": interfaces,
              "sqlite_and_history_match": True, "exact_scores_maps_overlays_and_decisions": True,
              "baseline": concise(baseline), "enabled": concise(enabled)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
