"""Recoverable, sampled-network Jetson replay of three saved MVTec cases."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import statistics
import subprocess
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
    ("color_preview", "test/color/009.png", "4717a9a1124e824e578093d97d23e0e6e32d05e4e6369f4e2325be1335dc0b10", "development_preview"),
)


def preflight(prefix, llama_root):
    manifest = prefix / "manifests/metal_nut_v1.json"
    files = [manifest, prefix / "configs/forgegrade_v1.json",
             prefix / "guidance/prototype_inspection_v1.json",
             prefix / "efficientad_s_calibrated.ts", prefix / "venv/bin/python",
             llama_root / "venv/bin/python"]
    files += [prefix / "data/mvtec_ad/metal_nut" / case[1] for case in CASES]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError("Missing local runtime assets: " + ", ".join(missing))
    if sha256(prefix / "efficientad_s_calibrated.ts") != MODEL_SHA256:
        raise RuntimeError("Frozen model SHA-256 changed.")
    rows = json.loads(manifest.read_text())["samples"]
    for _, path, digest, role in CASES:
        if sha256(prefix / "data/mvtec_ad/metal_nut" / path) != digest:
            raise RuntimeError("Saved image SHA-256 changed: " + path)
        if sum(row["path"] == path and row["sha256"] == digest and row["role"] == role
               for row in rows) != 1:
            raise RuntimeError("Saved image role differs from frozen manifest: " + path)
    state = network_state()
    if (state["wifi"]["returncode"] != 0
            or "100 (connected)" not in state["wifi"]["stdout"].lower()
            or "wlan0" not in state["default_route"]["stdout"]
            or state["ethernet_ipv4"]["stdout"]):
        raise RuntimeError("Expected connected wlan0 as sole upstream interface.")
    return {"model_sha256": MODEL_SHA256, "cases": [case[0] for case in CASES],
            "network_before": state}


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
    report = {"schema_version": 1, "stage": "10.2_disconnected_saved_case_series",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_mode": "SAVED_IMAGE_REPLAY", "physical_part_inspected": False,
              "new_accuracy_evidence": False, "status": "INCOMPLETE", **checked}
    def save():
        (output / "offline_report.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    watchdog_unit = "forgesight-offline-series-reconnect-v1"
    disconnected_by_run = False
    monitor_stop = threading.Event()
    samples = []
    def monitor():
        while not monitor_stop.is_set():
            stamp = time.monotonic()
            try:
                state = network_state()
                samples.append({"elapsed_seconds": round(stamp - monitor_start, 3),
                                "disconnected": disconnected(state), "state": state})
            except Exception as error:
                samples.append({"elapsed_seconds": round(stamp - monitor_start, 3),
                                "disconnected": False, "error": str(error)})
            monitor_stop.wait(max(0, 1 - (time.monotonic() - stamp)))
    monitor_thread = None
    try:
        watchdog = command(["systemd-run", "--on-active=240s", "--unit=" + watchdog_unit,
                            "/usr/bin/nmcli", "device", "connect", "wlan0"], timeout=15)
        report["reconnect_watchdog"] = watchdog
        if watchdog["returncode"]:
            raise RuntimeError("Independent Wi-Fi recovery watchdog was not scheduled.")
        save()
        down = command(["nmcli", "device", "disconnect", "wlan0"], timeout=20)
        disconnected_by_run = down["returncode"] == 0
        report["disconnect_command"] = down
        if not disconnected_by_run:
            raise RuntimeError("Wi-Fi disconnect failed.")
        time.sleep(2)
        report["network_at_start"] = network_state()
        if not disconnected(report["network_at_start"]):
            raise RuntimeError("Network route remained after disconnect.")
        monitor_start = time.monotonic()
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        database = output / "inspections.sqlite3"
        report["vision_cases"] = []
        for name, dataset_path, digest, role in CASES:
            argv = ["/usr/bin/python3", "-B", "-m", "forgesight.vision_edge_replay",
                    "--image", str(prefix / "data/mvtec_ad/metal_nut" / dataset_path),
                    "--image-sha256", digest,
                    "--dataset-manifest", str(prefix / "manifests/metal_nut_v1.json"),
                    "--part-id", "mvtec-" + name + "-offline-series-01",
                    "--vision-python", str(prefix / "venv/bin/python"),
                    "--model", str(prefix / "efficientad_s_calibrated.ts"),
                    "--model-sha256", MODEL_SHA256,
                    "--grade-policy", str(prefix / "configs/forgegrade_v1.json"),
                    "--database", str(database),
                    "--output-dir", str(output / name)]
            if role == "fit":
                argv.append("--allow-fit-reference")
            result = run_as_nvidia(argv, prefix, output, name, 45)
            if result["returncode"] or result["timed_out"]:
                raise RuntimeError("Vision replay failed: " + name)
            row = json.loads((output / name / "run_report.json").read_text())
            record = json.loads((output / name / "record.json").read_text())
            if (row["status"] != "PERSISTED_SAVED_IMAGE_REPLAY"
                    or row["image_sha256"] != digest
                    or record["source"]["dataset_role"] != role
                    or not row["jetson_identity_checked"]):
                raise RuntimeError("Inconsistent Jetson replay: " + name)
            report["vision_cases"].append({"name": name, "dataset_path": dataset_path,
                                           "dataset_role": role, "process": result,
                                           "inspection_id": row["inspection_id"],
                                           "score": row["score"], "candidate": row["candidate"],
                                           "grade": row["grade"],
                                           "model_forward_ms": row["model_forward_ms"],
                                           "replay_to_sqlite_commit_ms": row["replay_to_sqlite_commit_ms"],
                                           "record_sha256": sha256(output / name / "record.json")})
            save()
        advice_argv = ["/usr/bin/python3", "-B", "-m", "forgesight.assist_jetson_replay_once",
                       "--record", str(output / "bent_preview/record.json"),
                       "--database", str(database),
                       "--guidance", str(prefix / "guidance/prototype_inspection_v1.json"),
                       "--llama-root", str(llama_root),
                       "--output-dir", str(output / "advice"),
                       "--llama-timeout-seconds", "35"]
        report["advice_process"] = run_as_nvidia(advice_argv, prefix, output, "advice", 50)
        if report["advice_process"]["returncode"] or report["advice_process"]["timed_out"]:
            raise RuntimeError("ExecuTorch advisory failed.")
        advice = json.loads((output / "advice/advice_report.json").read_text())
        bent = report["vision_cases"][1]
        if (advice["status"] != "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW"
                or advice["inspection_id"] != bent["inspection_id"]
                or advice["authoritative_grade"] != bent["grade"]
                or advice["inspection_and_sqlite_unchanged"] is not True):
            raise RuntimeError("Advice did not preserve the authoritative record.")
        report["advice"] = {"inspection_id": advice["inspection_id"],
                            "status": advice["status"],
                            "authoritative_grade": advice["authoritative_grade"],
                            "inspection_and_sqlite_unchanged": True}
        report["local_components_completed"] = True
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
    finally:
        monitor_stop.set()
        if monitor_thread:
            monitor_thread.join(timeout=12)
        report["network_samples"] = samples
        report["network_sample_count"] = len(samples)
        report["network_sample_max_gap_seconds"] = (max((b["elapsed_seconds"] - a["elapsed_seconds"]
                                                          for a, b in zip(samples, samples[1:])), default=None))
        report["all_network_samples_disconnected"] = bool(samples) and all(
            sample["disconnected"] for sample in samples)
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
            report["watchdog_cancel"] = command(["systemctl", "stop", watchdog_unit + ".timer"])
        if (report.get("local_components_completed")
                and report["all_network_samples_disconnected"]
                and disconnected(report["network_before_recovery"])
                and report["wifi_reconnected"]):
            report["status"] = "DISCONNECTED_SAVED_CASE_SERIES_VERIFIED"
            values = [case["model_forward_ms"] for case in report["vision_cases"]]
            report["model_forward_median_ms"] = statistics.median(values)
            report["model_forward_max_ms"] = max(values)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({key: report.get(key) for key in
                      ("status", "network_sample_count", "wifi_reconnected", "error")}, indent=2))
    if report["status"] == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
