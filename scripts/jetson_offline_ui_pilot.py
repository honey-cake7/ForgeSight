"""Exercise the Jetson replay UI over loopback during a bounded Wi-Fi outage."""

import argparse
from datetime import datetime, timezone
import http.client as http_client
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.jetson_offline_replay_pilot import command, disconnected, network_state, sha256  # noqa: E402


RECORD_SHA256 = "fb31af0dcefb666ea4826aef0e78d9a6aa34e6cd0924c89195d40bb066f58a6a"
DATABASE_SHA256 = "e6a27fc55199c27932d59cd46f041f164263f11c84af160adc9e679f821a7b35"
WATCHDOG_UNIT = "forgesight-offline-ui-reconnect-v1"
SOURCE_RUN = "runs/offline_case_series_01"


def preflight(prefix, llama_root):
    record = prefix / SOURCE_RUN / "bent_preview/record.json"
    database = prefix / SOURCE_RUN / "inspections.sqlite3"
    files = [record, database, prefix / "guidance/prototype_inspection_v1.json",
             prefix / "forgesight/assist_jetson_ui.py", llama_root / "venv/bin/python"]
    missing = [str(path) for path in files if not path.is_file()]
    if missing or sha256(record) != RECORD_SHA256 or sha256(database) != DATABASE_SHA256:
        raise RuntimeError("Saved inspection or local runtime assets changed: " + ", ".join(missing))
    state = network_state()
    if (state["wifi"]["stdout"].lower() != "100 (connected)"
            or "wlan0" not in state["default_route"]["stdout"]
            or state["ethernet_ipv4"]["stdout"]):
        raise RuntimeError("Expected sole active upstream route through connected wlan0.")
    saved = json.loads(record.read_text())
    if (saved["source"]["mode"] != "SAVED_IMAGE_REPLAY"
            or saved["source"]["physical_part"] is not False
            or saved["decision"]["grade"] != "REVIEW"):
        raise RuntimeError("Expected the prior saved REVIEW replay.")
    return {"record_sha256": RECORD_SHA256, "database_sha256": DATABASE_SHA256,
            "inspection_id": saved["inspection_id"], "grade": "REVIEW", "network_before": state}


def http(port, method, path):
    conn = http_client.HTTPConnection("127.0.0.1", port, timeout=3)
    started = time.monotonic()
    headers = {"Host": "127.0.0.1:" + str(port)}
    if method == "POST":
        headers["Origin"] = "http://127.0.0.1:" + str(port)
        headers["Content-Length"] = "0"
    try:
        conn.request(method, path, body=b"" if method == "POST" else None, headers=headers)
        response = conn.getresponse()
        body = response.read()
        return {"code": response.status, "body": body,
                "duration_ms": (time.monotonic() - started) * 1000,
                "content_security_policy": response.getheader("Content-Security-Policy")}
    finally:
        conn.close()


def json_body(response):
    return json.loads(response["body"].decode())


def start_ui(prefix, llama_root, record, database, guidance, advice_root, port, timeout, output, name):
    argv = ["runuser", "-u", "nvidia", "--", "/usr/bin/python3", "-B", "-m",
            "forgesight.assist_jetson_ui", "--record", str(record),
            "--database", str(database), "--guidance", str(guidance),
            "--llama-root", str(llama_root), "--advice-root", str(advice_root),
            "--llama-timeout-seconds", str(timeout), "--port", str(port)]
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    env.pop("PYTHONPATH", None)
    log = (output / (name + ".log")).open("w")
    process = subprocess.Popen(argv, cwd=str(prefix), env=env,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(50):
        if process.poll() is not None:
            log.close()
            raise RuntimeError("UI process exited before serving: " + name)
        try:
            if http(port, "GET", "/api/state")["code"] == 200:
                return process, log
        except (OSError, http_client.HTTPException):
            pass
        time.sleep(.1)
    stop_ui(process, log)
    raise RuntimeError("UI startup deadline exceeded: " + name)


def stop_ui(process, log):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
    log.close()


def poll_state(port, expected, seconds):
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        state = json_body(http(port, "GET", "/api/state"))
        if state["advice"]["status"] == expected:
            return {"elapsed_ms": (time.monotonic() - started) * 1000,
                    "state": state}
        time.sleep(.25)
    raise RuntimeError("Timed out waiting for UI advice status " + expected)


def run_ui_case(prefix, llama_root, output, name, port, timeout, expected, checked):
    record = prefix / SOURCE_RUN / "bent_preview/record.json"
    database = prefix / SOURCE_RUN / "inspections.sqlite3"
    guidance = prefix / "guidance/prototype_inspection_v1.json"
    advice_root = output / (name + "_advice")
    process, log = start_ui(prefix, llama_root, record, database, guidance,
                            advice_root, port, timeout, output, name)
    try:
        assets = {}
        for path, marker in (("/", b"ForgeSight saved-image replay"),
                             ("/app.js", b"/api/state"),
                             ("/style.css", b"body")):
            response = http(port, "GET", path)
            if response["code"] != 200 or marker not in response["body"]:
                raise RuntimeError("Local UI asset failed: " + path)
            assets[path] = {"status": response["code"], "sha256": sha256_bytes(response["body"])}
        before = json_body(http(port, "GET", "/api/state"))
        if (before["inspection"]["grade"] != checked["grade"]
                or before["inspection"]["inspection_id"] != checked["inspection_id"]
                or before["advice"]["status"] != "NOT_REQUESTED"):
            raise RuntimeError("UI did not start with the saved inspection.")
        requested = http(port, "POST", "/api/advice")
        busy = http(port, "POST", "/api/advice")
        if (requested["code"] != 202 or json_body(requested)["status"] != "PENDING"
                or busy["code"] != 409 or json_body(busy)["status"] != "BUSY"):
            raise RuntimeError("UI advice queue did not report PENDING then BUSY.")
        finished = poll_state(port, expected, timeout + 45)
        state = finished["state"]
        if state["inspection"]["grade"] != checked["grade"]:
            raise RuntimeError("Advisory changed the authoritative grade.")
        note = state["advice"]["advisory"]
        if expected == "GENERATED_REQUIRES_HUMAN_REVIEW":
            if not note or "[E1]" not in note["note"] or "[G1]" not in note["note"]:
                raise RuntimeError("Generated note lacks the reviewed local evidence IDs.")
        elif note is not None:
            raise RuntimeError("Unavailable advice exposed a note.")
        result = {"assets": assets, "request_status": "PENDING",
                  "busy_status": "BUSY", "status": expected,
                  "status_visible_ms": finished["elapsed_ms"],
                  "grade_unchanged": True, "note_present": note is not None,
                  "note_sha256": sha256_bytes(note["note"].encode()) if note else None}
    finally:
        stop_ui(process, log)
    # Restart the same server against the same persisted report, while still disconnected.
    process, log = start_ui(prefix, llama_root, record, database, guidance,
                            advice_root, port, timeout, output, name + "_restart")
    try:
        after = json_body(http(port, "GET", "/api/state"))
        again = http(port, "POST", "/api/advice")
        if (after["inspection"]["grade"] != checked["grade"]
                or after["advice"]["status"] != expected
                or again["code"] != 409 or json_body(again)["status"] != "ALREADY_REQUESTED"):
            raise RuntimeError("UI restart lost its saved grade or advice state.")
        result["restart_preserved_state"] = True
        result["restart_refused_duplicate_advice"] = True
    finally:
        stop_ui(process, log)
    return result


def sha256_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prefix, llama_root, output = args.prefix.resolve(), args.llama_root.resolve(), args.output_dir.resolve()
    checked = preflight(prefix, llama_root)
    if args.preflight_only:
        print(json.dumps(checked, indent=2))
        return
    if os.geteuid() != 0 or output.exists():
        parser.error("Run detached as root with a fresh output directory.")
    output.mkdir(parents=True)
    account = pwd.getpwnam("nvidia")
    os.chown(output, account.pw_uid, account.pw_gid)
    report = {"schema_version": 1, "stage": "10.3_disconnected_loopback_ui_pilot",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_mode": "SAVED_IMAGE_REPLAY", "physical_part_inspected": False,
              "browser_rendering_tested": False, "status": "INCOMPLETE", **checked}
    def save():
        (output / "offline_ui_report.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    monitor_stop = threading.Event()
    samples = []
    monitor_thread = None
    disconnected_by_run = False
    def monitor():
        start = time.monotonic()
        while not monitor_stop.is_set():
            stamp = time.monotonic()
            state = network_state()
            samples.append({"elapsed_seconds": round(stamp - start, 3),
                            "disconnected": disconnected(state), "state": state})
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
        report["success"] = run_ui_case(prefix, llama_root, output, "success", 8768, 35,
                                        "GENERATED_REQUIRES_HUMAN_REVIEW", checked)
        save()
        report["timeout"] = run_ui_case(prefix, llama_root, output, "timeout", 8769, 1,
                                        "ADVICE_UNAVAILABLE", checked)
        if (sha256(prefix / SOURCE_RUN / "bent_preview/record.json") != RECORD_SHA256
                or sha256(prefix / SOURCE_RUN / "inspections.sqlite3") != DATABASE_SHA256):
            raise RuntimeError("Original inspection or SQLite database changed.")
        report["saved_inspection_unchanged"] = True
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
        if (report.get("saved_inspection_unchanged")
                and report["all_network_samples_disconnected"]
                and disconnected(report["network_before_recovery"])
                and report["wifi_reconnected"]):
            report["status"] = "DISCONNECTED_LOOPBACK_UI_SUCCESS_TIMEOUT_RESTART_VERIFIED"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({key: report.get(key) for key in
                      ("status", "network_sample_count", "wifi_reconnected", "error")}, indent=2))
    if report["status"] == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
