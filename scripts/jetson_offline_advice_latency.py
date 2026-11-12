"""Measure repeated ExecuTorch advisory latency during a recoverable outage."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.jetson_offline_replay_pilot import (  # noqa: E402
    command, disconnected, network_state, run_as_nvidia, sha256,
)
from scripts.jetson_offline_vision_latency import percentile, thermal_memory_snapshot  # noqa: E402


SOURCE_RUN = "runs/offline_case_series_01"
RECORD_SHA256 = "fb31af0dcefb666ea4826aef0e78d9a6aa34e6cd0924c89195d40bb066f58a6a"
DATABASE_SHA256 = "e6a27fc55199c27932d59cd46f041f164263f11c84af160adc9e679f821a7b35"
WATCHDOG_UNIT = "forgesight-offline-advice-latency-reconnect-v1"
REPETITIONS = 8


def preflight(prefix, llama_root):
    record = prefix / SOURCE_RUN / "bent_preview/record.json"
    database = prefix / SOURCE_RUN / "inspections.sqlite3"
    files = [record, database, prefix / "guidance/prototype_inspection_v1.json",
             llama_root / "venv/bin/python"]
    missing = [str(path) for path in files if not path.is_file()]
    if missing or sha256(record) != RECORD_SHA256 or sha256(database) != DATABASE_SHA256:
        raise RuntimeError("Saved inspection or isolated Llama runtime changed: " + ", ".join(missing))
    saved = json.loads(record.read_text())
    if (saved["source"]["mode"] != "SAVED_IMAGE_REPLAY"
            or saved["source"]["physical_part"] is not False
            or saved["decision"]["grade"] != "REVIEW"):
        raise RuntimeError("Expected the prior saved REVIEW replay.")
    state = network_state()
    if (state["wifi"]["stdout"].lower() != "100 (connected)"
            or "wlan0" not in state["default_route"]["stdout"]
            or state["ethernet_ipv4"]["stdout"]):
        raise RuntimeError("Expected connected wlan0 as sole upstream route.")
    return {"record_sha256": RECORD_SHA256, "database_sha256": DATABASE_SHA256,
            "inspection_id": saved["inspection_id"], "authoritative_grade": "REVIEW",
            "repetitions": REPETITIONS, "network_before": state}


def advice_argv(prefix, llama_root, output, name):
    return ["/usr/bin/python3", "-B", "-m", "forgesight.assist_jetson_replay_once",
            "--record", str(prefix / SOURCE_RUN / "bent_preview/record.json"),
            "--database", str(prefix / SOURCE_RUN / "inspections.sqlite3"),
            "--guidance", str(prefix / "guidance/prototype_inspection_v1.json"),
            "--llama-root", str(llama_root),
            "--output-dir", str(output / name), "--llama-timeout-seconds", "35"]


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
    report = {"schema_version": 1, "stage": "10.7_disconnected_executorch_advice_latency",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_mode": "SAVED_IMAGE_REPLAY", "physical_part_inspected": False,
              "new_inspections": 0, "status": "INCOMPLETE", "runs": [], **checked}
    def save():
        (output / "advice_latency_report.json").write_text(json.dumps(report, indent=2) + "\n")
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
                                "disconnected": disconnected(state), "network": state,
                                "resources": thermal_memory_snapshot()})
            except Exception as error:
                samples.append({"elapsed_seconds": round(stamp - started, 3),
                                "disconnected": False, "error": str(error)})
            monitor_stop.wait(max(0, 1 - (time.monotonic() - stamp)))
    try:
        watchdog = command(["systemd-run", "--on-active=420s", "--unit=" + WATCHDOG_UNIT,
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
        report["resources_at_start"] = thermal_memory_snapshot()
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        for index in range(REPETITIONS):
            name = "run_%02d" % (index + 1)
            process = run_as_nvidia(advice_argv(prefix, llama_root, output, name),
                                    prefix, output, name, 45)
            if process["returncode"] != 0 or process["timed_out"]:
                raise RuntimeError("ExecuTorch advice failed or timed out: " + name)
            advice = json.loads((output / name / "advice_report.json").read_text())
            native = advice.get("native") or {}
            response = (advice.get("advisory") or {}).get("response") or {}
            note = response.get("generated_note")
            if (advice["status"] != "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW"
                    or advice["inspection_id"] != checked["inspection_id"]
                    or advice["authoritative_grade"] != "REVIEW"
                    or advice["inspection_and_sqlite_unchanged"] is not True
                    or native.get("status") != "JETSON_EXECUTORCH_LLAMA_GENERATION_VERIFIED"
                    or not isinstance(note, str) or "[E1]" not in note or "[G1]" not in note):
                raise RuntimeError("Advice provenance, output or grade mismatch: " + name)
            report["runs"].append({"index": index + 1,
                                   "controller_process_wall_ms": process["wall_seconds"] * 1000,
                                   "controller_runtime_wall_ms": advice["controller_runtime_wall_ms"],
                                   "native_process_wall_ms": native["process_wall_seconds"] * 1000,
                                   "native_peak_rss_kib": native["sampled_peak_child_rss_kib"],
                                   "generated_tokens": native["generation_stats"][0]["generated_tokens"],
                                   "advice_report_sha256": sha256(output / name / "advice_report.json")})
            save()
        record = prefix / SOURCE_RUN / "bent_preview/record.json"
        database = prefix / SOURCE_RUN / "inspections.sqlite3"
        report["original_record_unchanged"] = sha256(record) == RECORD_SHA256
        report["original_database_unchanged"] = sha256(database) == DATABASE_SHA256
        if not report["original_record_unchanged"] or not report["original_database_unchanged"]:
            raise RuntimeError("Advisory series changed the authoritative inspection.")
        report["resources_at_end"] = thermal_memory_snapshot()
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
        if (len(report["runs"]) == REPETITIONS
                and report.get("original_record_unchanged")
                and report.get("original_database_unchanged")
                and report["all_network_samples_disconnected"]
                and disconnected(report["network_before_recovery"])
                and report["wifi_reconnected"]):
            report["status"] = "DISCONNECTED_EXECUTORCH_ADVICE_LATENCY_SERIES_VERIFIED"
            report["percentile_method"] = "linear interpolation at (n-1)*p"
            report["metrics"] = {}
            for key in ("controller_process_wall_ms", "controller_runtime_wall_ms",
                        "native_process_wall_ms", "native_peak_rss_kib"):
                values = [run[key] for run in report["runs"]]
                report["metrics"][key] = {"n": len(values), "min": min(values),
                                           "p50": percentile(values, .5),
                                           "p95": percentile(values, .95),
                                           "max": max(values)}
            memory = [sample["resources"]["mem_available_kib"] for sample in samples
                      if "resources" in sample and sample["resources"]["mem_available_kib"] is not None]
            report["minimum_sampled_mem_available_kib"] = min(memory) if memory else None
            sensors = {}
            for sample in samples:
                for label, value in sample.get("resources", {}).get("sensors_celsius", {}).items():
                    sensors.setdefault(label, []).append(value)
            report["thermal_observations_celsius"] = {
                label: {"min": min(values), "max": max(values), "samples": len(values)}
                for label, values in sensors.items()}
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({key: report.get(key) for key in
                      ("status", "network_sample_count", "wifi_reconnected", "error")}, indent=2))
    if report["status"] == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
