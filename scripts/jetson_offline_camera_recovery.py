"""Bounded disconnected camera-open failure and live diagnostic recovery."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import pwd
import stat
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.jetson_offline_replay_pilot import (  # noqa: E402
    MODEL_SHA256, command, disconnected, network_state, run_as_nvidia, sha256,
)


WATCHDOG_UNIT = "forgesight-offline-camera-reconnect-v1"
CAMERA = Path("/dev/video0")
UNAVAILABLE = Path("/dev/video99")


def preflight(prefix):
    model = prefix / "efficientad_s_calibrated.ts"
    files = [model, prefix / "venv/bin/python",
             prefix / "forgesight/vision_edge_camera_diagnostic.py"]
    missing = [str(path) for path in files if not path.is_file()]
    device_model = Path("/proc/device-tree/model").read_bytes().decode(errors="replace").strip("\x00\n")
    if (missing or sha256(model) != MODEL_SHA256 or not CAMERA.exists()
            or not stat.S_ISCHR(CAMERA.stat().st_mode)
            or UNAVAILABLE.exists() or platform.machine().lower() != "aarch64"
            or "Jetson Orin Nano" not in device_model):
        raise RuntimeError("Jetson, camera node, or frozen model preflight failed: " + ", ".join(missing))
    state = network_state()
    if (state["wifi"]["stdout"].lower() != "100 (connected)"
            or "wlan0" not in state["default_route"]["stdout"]
            or state["ethernet_ipv4"]["stdout"]):
        raise RuntimeError("Expected connected wlan0 as sole upstream route.")
    return {"device_model": device_model, "camera_node": str(CAMERA),
            "model_sha256": MODEL_SHA256, "network_before": state}


def diagnostic_argv(prefix, output, name, camera_index):
    return ["/usr/bin/python3", "-B", "-m", "forgesight.vision_edge_camera_diagnostic",
            "--vision-python", str(prefix / "venv/bin/python"),
            "--model", str(prefix / "efficientad_s_calibrated.ts"),
            "--model-sha256", MODEL_SHA256,
            "--threshold", "0.22776579856872559",
            "--output-dir", str(output / name), "--camera-index", str(camera_index)]


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
    report = {"schema_version": 1, "stage": "10.5_disconnected_camera_failure_recovery",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "physical_part_inspected": False, "inspection_row_created": False,
              "status": "INCOMPLETE", **checked}
    def save():
        (output / "camera_recovery_report.json").write_text(json.dumps(report, indent=2) + "\n")
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
        watchdog = command(["systemd-run", "--on-active=180s", "--unit=" + WATCHDOG_UNIT,
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

        failed_process = run_as_nvidia(diagnostic_argv(prefix, output, "unavailable_camera", 99),
                                      prefix, output, "unavailable_camera", 30)
        failed = json.loads((output / "unavailable_camera/diagnostic.json").read_text())
        if (failed_process["returncode"] == 0 or failed_process["timed_out"]
                or failed["status"] != "ERROR" or failed["is_inspection"] is not False
                or failed["grade"] is not None or failed["camera_index"] != 99
                or failed["error"]["type"] != "RuntimeError"
                or (output / "unavailable_camera/frame.png").exists()):
            raise RuntimeError("Unavailable camera was not reported without an inspection.")
        report["unavailable_camera"] = {
            "process": failed_process, "status": "ERROR", "is_inspection": False,
            "grade": None, "frame_created": False,
            "diagnostic_sha256": sha256(output / "unavailable_camera/diagnostic.json")}
        save()

        live_process = run_as_nvidia(diagnostic_argv(prefix, output, "live_recovery", 0),
                                    prefix, output, "live_recovery", 75)
        live = json.loads((output / "live_recovery/diagnostic.json").read_text())
        if (live_process["returncode"] != 0 or live_process["timed_out"]
                or live["status"] != "CAMERA_MODEL_DIAGNOSTIC_VERIFIED_NO_PART"
                or live["is_inspection"] is not False or live["grade"] is not None
                or live["camera_index"] != 0
                or live["capture"]["shape_hwc"] != [480, 640, 3]
                or live["capture"]["image_sha256"] != sha256(output / "live_recovery/frame.png")
                or live["map_sha256"] != sha256(output / "live_recovery/anomaly_map.npy")):
            raise RuntimeError("C270 capture/model diagnostic did not recover.")
        report["live_recovery"] = {
            "process": live_process, "status": live["status"], "is_inspection": False,
            "grade": None, "capture": live["capture"],
            "model_forward_ms": live["inference"]["model_forward_ms"],
            "camera_to_report_ms": live["camera_to_report_ms"],
            "frame_sha256": sha256(output / "live_recovery/frame.png"),
            "map_sha256": sha256(output / "live_recovery/anomaly_map.npy"),
            "diagnostic_sha256": sha256(output / "live_recovery/diagnostic.json")}
        report["model_unchanged"] = sha256(prefix / "efficientad_s_calibrated.ts") == MODEL_SHA256
        report["recovery_verified"] = report["model_unchanged"]
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
            report["status"] = "DISCONNECTED_CAMERA_OPEN_FAILURE_AND_DIAGNOSTIC_RECOVERY_VERIFIED"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({key: report.get(key) for key in
                      ("status", "network_sample_count", "wifi_reconnected", "error")}, indent=2))
    if report["status"] == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
