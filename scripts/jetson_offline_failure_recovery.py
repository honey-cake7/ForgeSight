"""Bounded disconnected Jetson missing-model/corrupt-input recovery probe."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import sqlite3
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.jetson_offline_replay_pilot import (  # noqa: E402
    MODEL_SHA256, command, disconnected, network_state, run_as_nvidia, sha256,
)


IMAGE = "data/mvtec_ad/metal_nut/train/good/000.png"
IMAGE_SHA256 = "b0309325bc305ae8b19246884f20397b951285786aecca0ebf1d009e9799e350"
WATCHDOG_UNIT = "forgesight-offline-recovery-reconnect-v1"


def preflight(prefix):
    files = [prefix / IMAGE, prefix / "efficientad_s_calibrated.ts",
             prefix / "manifests/metal_nut_v1.json", prefix / "configs/forgegrade_v1.json",
             prefix / "venv/bin/python"]
    missing = [str(path) for path in files if not path.is_file()]
    if missing or sha256(prefix / IMAGE) != IMAGE_SHA256 or sha256(files[1]) != MODEL_SHA256:
        raise RuntimeError("Frozen local image/model assets are missing or changed: " + ", ".join(missing))
    rows = json.loads(files[2].read_text())["samples"]
    if sum(row["path"] == "train/good/000.png" and row["sha256"] == IMAGE_SHA256
           and row["role"] == "fit" for row in rows) != 1:
        raise RuntimeError("Normal reference role differs from frozen manifest.")
    state = network_state()
    if (state["wifi"]["stdout"].lower() != "100 (connected)"
            or "wlan0" not in state["default_route"]["stdout"]
            or state["ethernet_ipv4"]["stdout"]):
        raise RuntimeError("Expected connected wlan0 as sole upstream route.")
    return {"image_sha256": IMAGE_SHA256, "model_sha256": MODEL_SHA256,
            "network_before": state}


def replay_argv(prefix, output, name, image, image_sha256, model):
    return ["/usr/bin/python3", "-B", "-m", "forgesight.vision_edge_replay",
            "--image", str(image), "--image-sha256", image_sha256,
            "--dataset-manifest", str(prefix / "manifests/metal_nut_v1.json"),
            "--part-id", "mvtec-fit-000-offline-recovery-" + name,
            "--vision-python", str(prefix / "venv/bin/python"),
            "--model", str(model), "--model-sha256", MODEL_SHA256,
            "--grade-policy", str(prefix / "configs/forgegrade_v1.json"),
            "--database", str(output / "inspections.sqlite3"),
            "--output-dir", str(output / name), "--allow-fit-reference"]


def sqlite_records(path):
    if not path.exists():
        return []
    with sqlite3.connect(str(path)) as database:
        return [(row[0], json.loads(row[1])) for row in database.execute(
            "SELECT part_id, record_json FROM inspections ORDER BY persisted_at_utc")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prefix, output = args.prefix.resolve(), args.output_dir.resolve()
    checked = preflight(prefix)
    if args.preflight_only:
        print(json.dumps(checked, indent=2))
        return
    if os.geteuid() != 0 or output.exists():
        parser.error("Run detached as root with a fresh output directory.")
    output.mkdir(parents=True)
    account = pwd.getpwnam("nvidia")
    os.chown(output, account.pw_uid, account.pw_gid)
    report = {"schema_version": 1, "stage": "10.4_disconnected_input_failure_recovery",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_mode": "SAVED_IMAGE_REPLAY", "physical_part_inspected": False,
              "status": "INCOMPLETE", **checked}
    def save():
        (output / "recovery_report.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    disconnected_by_run = False
    monitor_stop = threading.Event()
    samples = []
    monitor_thread = None
    def monitor():
        started = time.monotonic()
        while not monitor_stop.is_set():
            stamp = time.monotonic()
            try:
                state = network_state()
                samples.append({"elapsed_seconds": round(stamp - started, 3),
                                "disconnected": disconnected(state), "state": state})
            except Exception as error:
                samples.append({"elapsed_seconds": round(stamp - started, 3),
                                "disconnected": False, "error": str(error)})
            monitor_stop.wait(max(0, 1 - (time.monotonic() - stamp)))
    try:
        watchdog = command(["systemd-run", "--on-active=240s", "--unit=" + WATCHDOG_UNIT,
                            "/usr/bin/nmcli", "device", "connect", "wlan0"], timeout=15)
        report["reconnect_watchdog"] = watchdog
        if watchdog["returncode"]:
            raise RuntimeError("Independent reconnect timer was not scheduled.")
        down = command(["nmcli", "device", "disconnect", "wlan0"], timeout=20)
        report["disconnect_command"] = down
        disconnected_by_run = down["returncode"] == 0
        if not disconnected_by_run:
            raise RuntimeError("Wi-Fi disconnect failed.")
        time.sleep(2)
        report["network_at_start"] = network_state()
        if not disconnected(report["network_at_start"]):
            raise RuntimeError("Network route remained after disconnect.")
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        image = prefix / IMAGE
        model = prefix / "efficientad_s_calibrated.ts"
        database = output / "inspections.sqlite3"

        # Inject an absent path. Never rename or remove the working frozen model.
        absent = output / "intentionally_missing_export.ts"
        if absent.exists():
            raise RuntimeError("Missing-model injection path unexpectedly exists.")
        process = run_as_nvidia(replay_argv(prefix, output, "missing_model", image,
                                             IMAGE_SHA256, absent), prefix, output,
                                "missing_model", 45)
        missing_record = json.loads((output / "missing_model/record.json").read_text())
        missing_run = json.loads((output / "missing_model/run_report.json").read_text())
        records = sqlite_records(database)
        if (process["returncode"] == 0 or process["timed_out"]
                or missing_run["status"] != "PERSISTED_REPLAY_ERROR"
                or missing_record["status"] != "ERROR"
                or missing_record["decision"]["grade"] != "REVIEW"
                or missing_record["decision"]["reason"] != "INSPECTION_ERROR_OR_INCOMPLETE"
                or len(records) != 1 or records[0][1] != missing_record):
            raise RuntimeError("Missing model was not saved as an ERROR/REVIEW record.")
        report["missing_model"] = {"process": process, "status": "PERSISTED_REPLAY_ERROR",
                                   "record_status": "ERROR", "grade": "REVIEW",
                                   "error_type": missing_record["error"]["type"],
                                   "sqlite_rows_after": 1,
                                   "record_sha256": sha256(output / "missing_model/record.json")}
        save()

        # A short, invalid PNG must be rejected at the manifest/hash intake gate.
        corrupt = output / "corrupt.png"
        corrupt.write_bytes(b"not a valid MVTec PNG\n")
        process = run_as_nvidia(replay_argv(prefix, output, "corrupt_input", corrupt,
                                             IMAGE_SHA256, model), prefix, output,
                                "corrupt_input", 45)
        stderr = (output / "corrupt_input.stderr.log").read_text()
        if (process["returncode"] == 0 or process["timed_out"]
                or "Saved image SHA-256 differs" not in stderr
                or (output / "corrupt_input").exists()
                or len(sqlite_records(database)) != 1):
            raise RuntimeError("Corrupt input was not rejected without creating an inspection.")
        report["corrupt_input"] = {"process": process, "intake_rejected": True,
                                   "rejection": "SAVED_IMAGE_SHA256_MISMATCH",
                                   "sqlite_rows_after": 1, "corrupt_file_sha256": sha256(corrupt)}
        save()

        process = run_as_nvidia(replay_argv(prefix, output, "valid_recovery", image,
                                             IMAGE_SHA256, model), prefix, output,
                                "valid_recovery", 45)
        valid_record = json.loads((output / "valid_recovery/record.json").read_text())
        valid_run = json.loads((output / "valid_recovery/run_report.json").read_text())
        records = sqlite_records(database)
        if (process["returncode"] != 0 or process["timed_out"]
                or valid_run["status"] != "PERSISTED_SAVED_IMAGE_REPLAY"
                or valid_record["status"] != "ANALYZED"
                or valid_record["decision"]["grade"] != "PASS"
                or len(records) != 2
                or records[0][1] != missing_record or records[1][1] != valid_record):
            raise RuntimeError("Valid replay did not recover while preserving the ERROR row.")
        report["valid_recovery"] = {"process": process, "status": "ANALYZED",
                                    "grade": "PASS", "score": valid_record["score"],
                                    "model_forward_ms": valid_record["timings"]["model_forward_ms"],
                                    "sqlite_rows_after": 2,
                                    "record_sha256": sha256(output / "valid_recovery/record.json")}
        report["source_model_unchanged"] = sha256(model) == MODEL_SHA256
        report["source_image_unchanged"] = sha256(image) == IMAGE_SHA256
        report["recovery_verified"] = report["source_model_unchanged"] and report["source_image_unchanged"]
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
    finally:
        monitor_stop.set()
        if monitor_thread:
            monitor_thread.join(timeout=12)
        report["network_samples"] = samples
        report["network_sample_count"] = len(samples)
        report["all_network_samples_disconnected"] = bool(samples) and all(
            sample["disconnected"] for sample in samples)
        report["maximum_sample_gap_seconds"] = max(
            (b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(samples, samples[1:])),
            default=None)
        report["network_before_recovery"] = network_state() if disconnected_by_run else None
        if disconnected_by_run:
            report["reconnect_command"] = command(["nmcli", "device", "connect", "wlan0"], timeout=35)
            for _ in range(30):
                state = network_state()
                if "wlan0" in state["default_route"]["stdout"] and state["wifi_ipv4"]["stdout"]:
                    break
                time.sleep(1)
        report["network_after_recovery"] = network_state()
        report["wifi_reconnected"] = "wlan0" in report["network_after_recovery"]["default_route"]["stdout"]
        if report["wifi_reconnected"]:
            report["watchdog_cancel"] = command(["systemctl", "stop", WATCHDOG_UNIT + ".timer"])
        if (report.get("recovery_verified")
                and report["all_network_samples_disconnected"]
                and disconnected(report["network_before_recovery"])
                and report["wifi_reconnected"]):
            report["status"] = "DISCONNECTED_SAVED_REPLAY_FAILURE_RECOVERY_VERIFIED"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({key: report.get(key) for key in
                      ("status", "network_sample_count", "wifi_reconnected", "error")}, indent=2))
    if report["status"] == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
