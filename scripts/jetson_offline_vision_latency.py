"""Measure repeated cold saved-image vision replays during a recoverable outage."""

import argparse
from datetime import datetime, timezone
import json
import math
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


CASES = (
    ("normal_fit", "train/good/000.png", "b0309325bc305ae8b19246884f20397b951285786aecca0ebf1d009e9799e350", "fit"),
    ("bent_preview", "test/bent/017.png", "b49cc898143c46a8c3a8697411d04a078b36f60f4798ccc68aea33ea8fcac026", "development_preview"),
)
WATCHDOG_UNIT = "forgesight-offline-vision-latency-reconnect-v1"
REPETITIONS = 20


def preflight(prefix):
    files = [prefix / "efficientad_s_calibrated.ts", prefix / "venv/bin/python",
             prefix / "manifests/metal_nut_v1.json", prefix / "configs/forgegrade_v1.json"]
    files += [prefix / "data/mvtec_ad/metal_nut" / case[1] for case in CASES]
    missing = [str(path) for path in files if not path.is_file()]
    if missing or sha256(files[0]) != MODEL_SHA256:
        raise RuntimeError("Frozen model/runtime assets missing or changed: " + ", ".join(missing))
    samples = json.loads(files[2].read_text())["samples"]
    for _, path, digest, role in CASES:
        if (sha256(prefix / "data/mvtec_ad/metal_nut" / path) != digest
                or sum(row["path"] == path and row["sha256"] == digest and row["role"] == role
                       for row in samples) != 1):
            raise RuntimeError("Saved image or dataset role changed: " + path)
    state = network_state()
    if (state["wifi"]["stdout"].lower() != "100 (connected)"
            or "wlan0" not in state["default_route"]["stdout"]
            or state["ethernet_ipv4"]["stdout"]):
        raise RuntimeError("Expected connected wlan0 as sole upstream route.")
    return {"model_sha256": MODEL_SHA256, "repetitions": REPETITIONS,
            "case_names": [case[0] for case in CASES], "network_before": state}


def thermal_memory_snapshot():
    sensors = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            label = (zone / "type").read_text().strip()
            value = int((zone / "temp").read_text().strip()) / 1000
            if -20 <= value <= 150:
                sensors[zone.name + ":" + label] = value
        except (OSError, ValueError):
            pass
    available_kib = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            break
    return {"sensors_celsius": sensors, "mem_available_kib": available_kib}


def percentile(values, fraction):
    """Linearly interpolate at (n-1)*fraction; descriptive for this small sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def replay_argv(prefix, output, index, case):
    name, path, digest, role = case
    argv = ["/usr/bin/python3", "-B", "-m", "forgesight.vision_edge_replay",
            "--image", str(prefix / "data/mvtec_ad/metal_nut" / path),
            "--image-sha256", digest,
            "--dataset-manifest", str(prefix / "manifests/metal_nut_v1.json"),
            "--part-id", "mvtec-repeat-%02d-%s" % (index + 1, name),
            "--vision-python", str(prefix / "venv/bin/python"),
            "--model", str(prefix / "efficientad_s_calibrated.ts"),
            "--model-sha256", MODEL_SHA256,
            "--grade-policy", str(prefix / "configs/forgegrade_v1.json"),
            "--database", str(output / "inspections.sqlite3"),
            "--output-dir", str(output / ("run_%02d" % (index + 1)))]
    if role == "fit":
        argv.append("--allow-fit-reference")
    return argv


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
    report = {"schema_version": 1, "stage": "10.6_disconnected_saved_replay_vision_latency",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_mode": "SAVED_IMAGE_REPLAY", "physical_part_inspected": False,
              "distinct_physical_parts": 0, "new_accuracy_evidence": False,
              "status": "INCOMPLETE", "runs": [], **checked}
    def save():
        (output / "latency_report.json").write_text(json.dumps(report, indent=2) + "\n")
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
                                "disconnected": disconnected(state),
                                "network": state, "resources": thermal_memory_snapshot()})
            except Exception as error:
                samples.append({"elapsed_seconds": round(stamp - started, 3),
                                "disconnected": False, "error": str(error)})
            monitor_stop.wait(max(0, 1 - (time.monotonic() - stamp)))
    try:
        watchdog = command(["systemd-run", "--on-active=480s", "--unit=" + WATCHDOG_UNIT,
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
        database = output / "inspections.sqlite3"
        for index in range(REPETITIONS):
            case = CASES[index % len(CASES)]
            name = "run_%02d" % (index + 1)
            process = run_as_nvidia(replay_argv(prefix, output, index, case),
                                    prefix, output, name, 20)
            if process["returncode"] != 0 or process["timed_out"]:
                raise RuntimeError("Saved replay failed or timed out: " + name)
            row = json.loads((output / name / "run_report.json").read_text())
            record = json.loads((output / name / "record.json").read_text())
            inference = json.loads((output / name / "inference.json").read_text())
            if (row["status"] != "PERSISTED_SAVED_IMAGE_REPLAY"
                    or row["image_sha256"] != case[2]
                    or record["source"]["dataset_role"] != case[3]
                    or not row["jetson_identity_checked"]
                    or inference["model_sha256"] != MODEL_SHA256):
                raise RuntimeError("Inconsistent replay evidence: " + name)
            report["runs"].append({"index": index + 1, "case": case[0],
                                   "dataset_role": case[3], "image_sha256": case[2],
                                   "inspection_id": row["inspection_id"],
                                   "grade": row["grade"], "score": row["score"],
                                   "process_wall_ms": process["wall_seconds"] * 1000,
                                   "replay_to_sqlite_commit_ms": row["replay_to_sqlite_commit_ms"],
                                   "model_forward_ms": row["model_forward_ms"],
                                   "vision_process_peak_rss_kib": inference["process_peak_rss_kib"],
                                   "record_sha256": sha256(output / name / "record.json")})
            save()
        with sqlite3.connect(str(database)) as conn:
            rows = conn.execute("SELECT inspection_id, record_json FROM inspections").fetchall()
        records = {inspection_id: json.loads(body) for inspection_id, body in rows}
        if len(records) != REPETITIONS:
            raise RuntimeError("Not all replay records survived in fresh SQLite.")
        for run in report["runs"]:
            saved = json.loads((output / ("run_%02d" % run["index"]) / "record.json").read_text())
            if records.get(run["inspection_id"]) != saved:
                raise RuntimeError("SQLite differs from replay record.")
        report["sqlite_exact_readback_for_all_runs"] = True
        report["resources_at_end"] = thermal_memory_snapshot()
        report["source_model_unchanged"] = sha256(prefix / "efficientad_s_calibrated.ts") == MODEL_SHA256
        report["source_images_unchanged"] = all(
            sha256(prefix / "data/mvtec_ad/metal_nut" / case[1]) == case[2] for case in CASES)
        if not report["source_model_unchanged"] or not report["source_images_unchanged"]:
            raise RuntimeError("Frozen source asset changed during measurement.")
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
                and report.get("sqlite_exact_readback_for_all_runs")
                and report["all_network_samples_disconnected"]
                and disconnected(report["network_before_recovery"])
                and report["wifi_reconnected"]):
            report["status"] = "DISCONNECTED_SAVED_REPLAY_VISION_LATENCY_SERIES_VERIFIED"
            report["latency_percentile_method"] = "linear interpolation at (n-1)*p"
            report["metrics"] = {}
            for key in ("model_forward_ms", "replay_to_sqlite_commit_ms", "process_wall_ms",
                        "vision_process_peak_rss_kib"):
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
