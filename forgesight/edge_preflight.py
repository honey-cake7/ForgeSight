"""Read host facts without installing anything or claiming model inference."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def optional_text(path: Path) -> dict:
    try:
        return {"value": path.read_text(errors="replace").replace("\x00", "").strip(), "error": None}
    except OSError as error:
        return {"value": None, "error": type(error).__name__}


def memory_kib(text: str | None) -> dict:
    values = {}
    for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
        match = re.search(rf"^{key}:\s+(\d+)\s+kB$", text or "", re.MULTILINE)
        values[key] = int(match[1]) if match else None
    return values


def identify_host(system: str, machine: str, model: str | None, l4t: str | None) -> str:
    name = (model or "").lower()
    if (system == "Linux" and machine.lower() in {"aarch64", "arm64"}
            and "jetson" in name and "orin nano" in name and l4t
            and re.search(r"# R\d+", l4t)):
        return "JETSON_ORIN_NANO_CANDIDATE"
    return "NOT_VERIFIED_JETSON_ORIN_NANO"


def version_probe(name: str) -> dict:
    executable = shutil.which(name)
    if not executable:
        return {"path": None, "version_line": None, "error": "NOT_INSTALLED"}
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                timeout=5, check=False)
        lines = (result.stdout or result.stderr).splitlines()
        return {"path": executable, "version_line": lines[0] if lines else None,
                "returncode": result.returncode, "error": None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"path": executable, "version_line": None, "error": type(error).__name__}


def collect_preflight() -> dict:
    system, machine = platform.system(), platform.machine()
    model = optional_text(Path("/proc/device-tree/model"))
    l4t = optional_text(Path("/etc/nv_tegra_release"))
    meminfo = optional_text(Path("/proc/meminfo"))
    os_release = optional_text(Path("/etc/os-release"))
    packages = {}
    for name in ("torch", "executorch", "torchao", "pytorch-tokenizers", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    kind = identify_host(system, machine, model["value"], l4t["value"])
    blockers = []
    if kind != "JETSON_ORIN_NANO_CANDIDATE":
        blockers.append("This host is not an identified Jetson Orin Nano ARM64 target.")
    if packages["executorch"] is None:
        blockers.append("ExecuTorch is not installed in this interpreter.")
    blockers.append("Model access, weights, runner, and generated text are not verified by preflight.")
    disk = shutil.disk_usage(Path.cwd())
    return {"schema_version": 1, "captured_utc": datetime.now(timezone.utc).isoformat(),
            "host_kind": kind, "system": system, "machine": machine,
            "python": platform.python_version(), "python_executable": sys.executable,
            "libc": list(platform.libc_ver()), "device_model": model, "l4t_release": l4t,
            "os_release": os_release, "memory_kib": memory_kib(meminfo["value"]),
            "memory_read_error": meminfo["error"], "disk_free_bytes": disk.free,
            "tools": {name: version_probe(name) for name in ("git", "cmake", "g++")},
            "packages": packages, "camera_nodes": sorted(str(path) for path in Path("/dev").glob("video*")),
            "ipv4_routes": optional_text(Path("/proc/net/route")),
            "ipv6_routes": optional_text(Path("/proc/net/ipv6_route")),
            "offline_status": "NOT_VERIFIED", "inference_status": "NOT_RUN",
            "stage_2_status": "BLOCKED_NOT_VERIFIED", "blockers": blockers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-jetson", action="store_true")
    args = parser.parse_args()
    try:
        report = collect_preflight()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
        print(json.dumps({key: report[key] for key in ("host_kind", "machine", "python", "packages",
                                                      "inference_status", "stage_2_status", "blockers")}, indent=2))
        if args.require_jetson and report["host_kind"] != "JETSON_ORIN_NANO_CANDIDATE":
            parser.exit(2, "BLOCKED: device-only operation refused on this host.\n")
    except OSError as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
