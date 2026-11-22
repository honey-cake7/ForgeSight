"""Compare Jetson operator inspections across two defect-type configurations."""

import argparse
import hashlib
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
PREVIEW_PROFILE = Path("configs/defect_type_preview_v1.json")


def evaluation_cases(path, profile_path):
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    split_path = Path("manifests/metal_nut_defect_type_split_v1.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if (evaluation["protocol"] != split["protocol"] or profile["training_protocol"] != split["protocol"]
            or profile["split_sha256"] != hashlib.sha256(split_path.read_bytes()).hexdigest()):
        raise ValueError("Evaluation, profile and frozen subtype split must agree.")
    paths = sorted(path for path, role in split["assignments"].items() if role == "test")

    def cases(key):
        rows = evaluation[key]["images"]
        predictions = {row["image"]: row["prediction"] for row in rows}
        if len(rows) != len(paths) or sorted(predictions) != paths:
            raise ValueError("Evaluation must cover exactly the frozen subtype test images.")
        return [(f"subtype_{index:02d}", path, True,
                 None if predictions[path] == "not_flagged" else predictions[path])
                for index, path in enumerate(paths)]

    return cases("preview_rules_test"), cases("candidate_test")


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


def exercise(args, name, profile, cases):
    run = args.output / name
    run.mkdir()
    database = run / "inspections.sqlite3"
    command = [sys.executable, "-B", "-m", "forgesight.jetson_live_ui",
               "--port", str(args.port), "--model", str(args.model),
               "--vision-python", str(args.vision_python), "--camera-index", "99",
               "--database", str(database), "--runs-root", str(run / "records"),
               "--advice-root", str(run / "advice"), "--dataset-root", str(args.dataset_root),
               "--vision-worker-socket", str(run / "vision.sock")]
    if profile:
        command += ["--defect-type-profile", str(profile)]
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
            for part, dataset_path, upload, expected in cases:
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
                expected_type = expected if profile else None
                observed_type = assessment["defect_type"] if assessment else None
                if (observed_type != expected_type or latest["assessment"] != assessment
                        or history["assessment"] != assessment or record["predicted_defect_type"] != "unknown"):
                    raise RuntimeError(f"Assessment differed across expected result, UI, history and SQLite: {part}")
                facts.append({"part": part, "dataset_path": dataset_path,
                              "image_sha256": record["source"]["sha256"],
                              "type": observed_type, "score": record["score"],
                              "candidate": record["candidate"], "decision": record["decision"],
                              "model_map_sha256": record["artifacts"]["score_map_model_sha256"],
                              "source_map_sha256": record["artifacts"]["score_map_source_sha256"],
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
    parser.add_argument("--evaluation", type=Path, help="Compare both profiles on the frozen subtype test.")
    parser.add_argument("--candidate-profile", type=Path, help="Tree JSON produced by the evaluated PC run.")
    args = parser.parse_args()
    if bool(args.evaluation) != bool(args.candidate_profile):
        parser.error("Supply both --evaluation and --candidate-profile for a tree comparison.")
    device = Path("/proc/device-tree/model").read_bytes().decode().rstrip("\x00")
    interfaces = [name for _, name in socket.if_nameindex()]
    if platform.machine() != "aarch64" or "Jetson Orin Nano" not in device or interfaces != ["lo"]:
        parser.error("Run on the Jetson inside a network namespace containing only loopback.")
    baseline_cases, enabled_cases = (evaluation_cases(args.evaluation, args.candidate_profile)
                                     if args.evaluation else (CASES, CASES))
    original_hashes = {row["path"]: row["sha256"] for row in json.loads(
        Path("manifests/metal_nut_v1.json").read_text(encoding="utf-8"))["samples"]}
    for _, path, _, _ in enabled_cases:
        if hashlib.sha256((args.dataset_root / path).read_bytes()).hexdigest() != original_hashes[path]:
            raise ValueError(f"Benchmark image hash mismatch: {path}")
    args.output.mkdir(parents=True)
    baseline = exercise(args, "baseline", PREVIEW_PROFILE if args.evaluation else None, baseline_cases)
    enabled = exercise(args, "enabled", args.candidate_profile or PREVIEW_PROFILE, enabled_cases)
    for original, updated in zip(baseline, enabled):
        for key in ("image_sha256", "score", "candidate", "decision", "model_map_sha256",
                    "source_map_sha256", "overlay_sha256"):
            if original[key] != updated[key]:
                raise RuntimeError(f"Enabled profile changed {key} for {original['part']}")

    def concise(facts):
        return [{**row, "decision": row["decision"]["grade"]} for row in facts]

    report = {"device": device, "application_namespace_interfaces": interfaces,
              "sqlite_and_history_match": True, "exact_scores_maps_overlays_and_decisions": True,
              "baseline_profile": str(PREVIEW_PROFILE) if args.evaluation else None,
              "enabled_profile": str(args.candidate_profile or PREVIEW_PROFILE),
              "baseline": concise(baseline), "enabled": concise(enabled)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
