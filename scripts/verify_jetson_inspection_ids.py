"""Verify the inspection counter and durable failure log on an isolated Jetson."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from forgesight.storage import InspectionStore
from scripts.verify_jetson_defect_type import state, wait_for


PORT = 8788
BASE = f"http://127.0.0.1:{PORT}"
UPLOAD = "test/scratch/013.png"
REPLAY = "test/color/009.png"


@contextmanager
def operator_app(args, model, database, log_name):
    command = [sys.executable, "-B", "-m", "forgesight.jetson_live_ui",
               "--port", str(PORT), "--camera-index", "99",
               "--vision-python", str(args.vision_python), "--model", str(model),
               "--dataset-root", str(args.dataset_root), "--database", str(database),
               "--runs-root", str(args.output / "records"), "--advice-root", str(args.output / "advice"),
               "--defect-type-profile", "configs/defect_type_mvtec_four_v1.json"]
    with (args.output / log_name).open("w") as log, subprocess.Popen(command, stdout=log, stderr=log) as app:
        try:
            def ready():
                if app.poll() is not None:
                    raise RuntimeError(f"Operator app exited. See {log_name}.")
                try:
                    return state(BASE)
                except OSError:
                    return None

            wait_for(ready)
            yield
        finally:
            if app.poll() is None:
                app.send_signal(signal.SIGINT)
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait()


def request(endpoint, data, content_type, **headers):
    return urlopen(Request(BASE + endpoint, data=data, headers={
        "Origin": BASE, "Content-Type": content_type, **headers}), timeout=5)


def inspect(store, number, status, endpoint, data, content_type, **headers):
    part_id = f"INS-{number:06d}"
    with request(endpoint, data, content_type, **headers) as response:
        if response.status != 202 or json.load(response)["part_id"] != part_id:
            raise RuntimeError(f"Expected accepted inspection {part_id}")

    def finished():
        snapshot = state(BASE)
        latest = snapshot["latest"] or {}
        if latest.get("part_id") == part_id and not snapshot["busy"]:
            return snapshot
        return None

    snapshot = wait_for(finished, seconds=100)
    latest = snapshot["latest"]
    record = store.get_inspection(latest["inspection_id"])
    history = next(row for row in snapshot["history"] if row["part_id"] == part_id)
    if latest["status"] != status or record["status"] != status or history["status"] != status:
        raise RuntimeError(f"Latest result, SQLite and history do not agree for {part_id}")
    report = {"part_id": part_id, "inspection_id": latest["inspection_id"], "status": status,
              "source_mode": record["source"]["mode"], "grade": record["decision"]["grade"]}
    if status == "ERROR":
        if record["decision"]["grade"] != "REVIEW" or latest["support_id"] != history["support_id"]:
            raise RuntimeError("Error did not preserve its REVIEW result and support ID.")
        report["support_id"] = latest["support_id"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "model", "vision-python", "dataset-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    device = Path("/proc/device-tree/model").read_bytes().decode().rstrip("\x00")
    interfaces = [name for _, name in socket.if_nameindex()]
    if platform.machine() != "aarch64" or "Jetson Orin Nano" not in device or interfaces != ["lo"]:
        parser.error("Run on the Jetson inside a network namespace containing only loopback.")
    hashes = {row["path"]: row["sha256"] for row in json.loads(
        Path("manifests/metal_nut_v1.json").read_text())["samples"]}
    for path in (UPLOAD, REPLAY):
        if hashlib.sha256((args.dataset_root / path).read_bytes()).hexdigest() != hashes[path]:
            raise ValueError(f"Benchmark image hash mismatch: {path}")
    upload = (args.dataset_root / UPLOAD).read_bytes()
    args.output.mkdir()
    model = args.output / "isolated-model.ts"
    shutil.copyfile(args.model, model)
    database = args.output / "inspections.sqlite3"
    store = InspectionStore(database)
    results = []
    with operator_app(args, model, database, "first.log"):
        results.append(inspect(store, 1, "ANALYZED", "/api/upload-diagnostic",
                               upload, "image/png", **{"X-Mvtec-Benchmark": "yes"}))
        try:
            with request("/api/upload-diagnostic", b"invalid png", "image/png"):
                raise RuntimeError("Invalid PNG was accepted.")
        except HTTPError as error:
            if error.code != 409:
                raise
            refused = json.load(error)["support_id"]
        results.append(inspect(store, 2, "ANALYZED", "/api/replay-local",
                               json.dumps({"dataset_path": REPLAY}).encode(), "application/json"))
        unavailable = model.with_suffix(".unavailable")
        model.rename(unavailable)
        try:
            results.append(inspect(store, 3, "ERROR", "/api/upload-diagnostic",
                                   upload, "image/png", **{"X-Mvtec-Benchmark": "yes"}))
        finally:
            unavailable.rename(model)
    with operator_app(args, model, database, "restarted.log"):
        results.append(inspect(store, 4, "ANALYZED", "/api/upload-diagnostic",
                               upload, "image/png", **{"X-Mvtec-Benchmark": "yes"}))
    operations = (args.output / "records/logs/forgesight-operations.log").read_text()
    for support_id in (refused, results[2]["support_id"]):
        if f"support_id={support_id}" not in operations:
            raise RuntimeError("Support ID did not match the persisted operation log.")
    if store.summary()["inspection_count"] != len(results):
        raise RuntimeError("Unexpected number of rows in SQLite.")
    report = {"device": device, "interfaces": interfaces, "results": results,
              "refused_input_support_id": refused, "support_ids_match_operation_log": True,
              "counter_survived_restart": True, "error_injected_by_hiding_isolated_model": True}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
