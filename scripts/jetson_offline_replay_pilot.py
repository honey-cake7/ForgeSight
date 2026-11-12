"""Timed, recoverable Jetson Wi-Fi-disconnected saved-image vision/advice pilot."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import time


IMAGE_SHA256 = "b49cc898143c46a8c3a8697411d04a078b36f60f4798ccc68aea33ea8fcac026"
MODEL_SHA256 = "8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd"
WATCHDOG_UNIT = "forgesight-offline-reconnect-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(argv, timeout=10):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def network_state():
    return {"wifi": command(["nmcli", "-g", "GENERAL.STATE", "device", "show", "wlan0"]),
            "default_route": command(["ip", "-4", "route", "show", "default"]),
            "wifi_ipv4": command(["ip", "-4", "-o", "addr", "show", "dev", "wlan0"]),
            "ethernet_ipv4": command(["ip", "-4", "-o", "addr", "show", "dev", "eth0"])}


def disconnected(state):
    return (all(state[key]["returncode"] == 0 for key in state)
            and "disconnected" in state["wifi"]["stdout"].lower()
            and not state["default_route"]["stdout"]
            and not state["wifi_ipv4"]["stdout"]
            and not state["ethernet_ipv4"]["stdout"])


def run_as_nvidia(argv, cwd, log_root, name, timeout):
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    environment.pop("PYTHONPATH", None)
    process = subprocess.Popen(["runuser", "-u", "nvidia", "--"] + argv,
                               cwd=str(cwd), env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        timed_out = True
    (log_root / (name + ".stdout.log")).write_text(stdout[-20000:])
    (log_root / (name + ".stderr.log")).write_text(stderr[-20000:])
    return {"returncode": process.returncode, "timed_out": timed_out,
            "wall_seconds": time.monotonic() - started}


def preflight(prefix, llama_root):
    image = prefix / "data/mvtec_ad/metal_nut/test/bent/017.png"
    model = prefix / "efficientad_s_calibrated.ts"
    files = {"image": image, "model": model,
             "manifest": prefix / "manifests/metal_nut_v1.json",
             "grade_policy": prefix / "configs/forgegrade_v1.json",
             "guidance": prefix / "guidance/prototype_inspection_v1.json",
             "vision_python": prefix / "venv/bin/python",
             "llama_python": llama_root / "venv/bin/python"}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing or sha256(image) != IMAGE_SHA256 or sha256(model) != MODEL_SHA256:
        raise RuntimeError("Missing or changed local image/model/runtime assets: " + ", ".join(missing))
    state = network_state()
    if (state["wifi"]["returncode"] != 0 or "connected" not in state["wifi"]["stdout"].lower()
            or "wlan0" not in state["default_route"]["stdout"]
            or state["ethernet_ipv4"]["stdout"]):
        raise RuntimeError("Expected sole active upstream route through connected wlan0.")
    return {"file_sha256": {"image": IMAGE_SHA256, "model": MODEL_SHA256},
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
    report = {"schema_version": 1, "stage": "10.1_disconnected_saved_replay_pilot",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_mode": "SAVED_IMAGE_REPLAY", "physical_part_inspected": False,
              "status": "INCOMPLETE", **checked}
    def save():
        (output / "offline_report.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    disconnected_by_pilot = False
    try:
        watchdog = command(["systemd-run", "--on-active=180s", "--unit=" + WATCHDOG_UNIT,
                            "/usr/bin/nmcli", "device", "connect", "wlan0"], timeout=15)
        report["reconnect_watchdog"] = watchdog
        if watchdog["returncode"] != 0:
            raise RuntimeError("Recovery watchdog was not scheduled.")
        save()
        down = command(["nmcli", "device", "disconnect", "wlan0"], timeout=20)
        disconnected_by_pilot = down["returncode"] == 0
        report["disconnect_command"] = down
        if not disconnected_by_pilot:
            raise RuntimeError("Wi-Fi disconnect failed; no offline run attempted.")
        time.sleep(2)
        report["network_during_vision_start"] = network_state()
        if not disconnected(report["network_during_vision_start"]):
            raise RuntimeError("Device still had a network route after Wi-Fi disconnect.")
        save()

        vision = ["/usr/bin/python3", "-B", "-m", "forgesight.vision_edge_replay",
                  "--image", str(prefix / "data/mvtec_ad/metal_nut/test/bent/017.png"),
                  "--image-sha256", IMAGE_SHA256,
                  "--dataset-manifest", str(prefix / "manifests/metal_nut_v1.json"),
                  "--part-id", "mvtec-dev-bent-017-offline-pilot-01",
                  "--vision-python", str(prefix / "venv/bin/python"),
                  "--model", str(prefix / "efficientad_s_calibrated.ts"),
                  "--model-sha256", MODEL_SHA256,
                  "--grade-policy", str(prefix / "configs/forgegrade_v1.json"),
                  "--database", str(output / "inspections.sqlite3"),
                  "--output-dir", str(output / "vision")]
        report["vision_process"] = run_as_nvidia(vision, prefix, output, "vision", 75)
        report["network_after_vision"] = network_state()
        if (report["vision_process"]["returncode"] != 0
                or not disconnected(report["network_after_vision"])):
            raise RuntimeError("Vision replay failed or network returned during vision.")
        save()

        advice = ["/usr/bin/python3", "-B", "-m", "forgesight.assist_jetson_replay_once",
                  "--record", str(output / "vision/record.json"),
                  "--database", str(output / "inspections.sqlite3"),
                  "--guidance", str(prefix / "guidance/prototype_inspection_v1.json"),
                  "--llama-root", str(llama_root),
                  "--output-dir", str(output / "advice"),
                  "--llama-timeout-seconds", "60"]
        report["advice_process"] = run_as_nvidia(advice, prefix, output, "advice", 75)
        report["network_after_advice"] = network_state()
        if (report["advice_process"]["returncode"] != 0
                or not disconnected(report["network_after_advice"])):
            raise RuntimeError("Advice failed or network returned during ExecuTorch.")
        vision_report = json.loads((output / "vision/run_report.json").read_text())
        advice_report = json.loads((output / "advice/advice_report.json").read_text())
        if (vision_report["status"] != "PERSISTED_SAVED_IMAGE_REPLAY"
                or advice_report["status"] != "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW"
                or advice_report["inspection_id"] != vision_report["inspection_id"]
                or advice_report["authoritative_grade"] != vision_report["grade"]
                or advice_report["inspection_and_sqlite_unchanged"] is not True):
            raise RuntimeError("Offline replay/advice evidence did not match.")
        report["inspection_id"] = vision_report["inspection_id"]
        report["authoritative_grade"] = vision_report["grade"]
        report["offline_vision_and_advice_verified"] = True
        save()
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
    finally:
        if disconnected_by_pilot:
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
        report["status"] = ("DISCONNECTED_SAVED_REPLAY_VISION_AND_ADVICE_VERIFIED"
                            if report.get("offline_vision_and_advice_verified") and report["wifi_reconnected"]
                            else "INCOMPLETE")
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({key: report.get(key) for key in
                      ("status", "inspection_id", "authoritative_grade", "error", "wifi_reconnected")}, indent=2))
    if report["status"] == "INCOMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
