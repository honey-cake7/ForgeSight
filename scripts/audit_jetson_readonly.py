"""Read device inventory without installing packages, streaming video or importing ML runtimes.

Compatible with Python 3.8+ so an older Jetson image can be audited before any
environment choice. Print one JSON document to stdout; do not write on-device.
"""

from datetime import datetime, timezone
import glob
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def read_text(path):
    try:
        return {"value": Path(path).read_text(errors="replace").replace("\x00", "\n").strip(), "error": None}
    except OSError as exc:
        return {"value": None, "error": "{}: {}".format(type(exc).__name__, exc)}


def run(argv):
    executable = shutil.which(argv[0])
    if executable is None and not Path(argv[0]).is_file():
        return {"argv": argv, "error": "NOT_INSTALLED_OR_NOT_ON_PATH", "returncode": None}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout[:65536], "stderr": result.stderr[:16384],
                "output_truncated": len(result.stdout) > 65536 or len(result.stderr) > 16384}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "error": "{}: {}".format(type(exc).__name__, exc)}


def inventory():
    started = time.monotonic()
    files = ["/proc/device-tree/model", "/proc/device-tree/compatible",
             "/proc/device-tree/nvidia,dtsfilename", "/etc/nv_tegra_release",
             "/etc/os-release", "/proc/meminfo", "/proc/swaps",
             "/usr/local/cuda/version.json", "/usr/local/cuda/version.txt"]
    packages = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless",
                 "executorch", "torchao", "pytorch-tokenizers", "fastapi", "uvicorn", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    module_locations = {}
    for name in ("torch", "torchvision", "numpy", "cv2", "executorch"):
        try:
            spec = importlib.util.find_spec(name)
            module_locations[name] = spec.origin if spec else None
        except (ImportError, ValueError) as exc:
            module_locations[name] = str(exc)
    disks = {}
    for path in ("/", str(Path.home())):
        usage = shutil.disk_usage(path)
        disks[path] = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    videos = []
    for node in sorted(glob.glob("/sys/class/video4linux/video*")):
        videos.append({"node": "/dev/" + Path(node).name, "name": read_text(Path(node)/"name"),
                       "sysfs_device": str((Path(node)/"device").resolve())})
    commands = {
        "identity": ["id"], "kernel": ["uname", "-a"],
        "installed_nvidia_packages": ["dpkg-query", "-W", "-f=${Package}\t${Version}\n",
                                      "nvidia-l4t-*", "nvidia-jetpack", "nvidia-cuda", "libcudnn*", "python3-opencv"],
        "storage_layout": ["lsblk", "-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT"],
        "usb": ["lsusb"], "camera_devices": ["v4l2-ctl", "--list-devices"],
        "power_mode_query": ["/usr/sbin/nvpmodel", "-q"],
        "git": ["git", "--version"], "cmake": ["cmake", "--version"],
        "compiler": ["g++", "--version"],
    }
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    commands["cuda_compiler"] = [nvcc, "--version"]
    if videos:
        commands["camera_formats"] = ["v4l2-ctl", "--device", videos[0]["node"], "--list-formats-ext"]
    command_results = {name: run(argv) for name, argv in commands.items()}
    project_candidates = {}
    for relative in ("ForgeSight", "Forge_Sight", "Desktop/ForgeSight", "Desktop/Forge_Sight",
                     "projects/ForgeSight", "projects/Forge_Sight"):
        candidate = Path.home()/relative
        if candidate.exists():
            project_candidates[str(candidate)] = {"exists": True, "git_present": (candidate/".git").exists()}
            if (candidate/".git").exists():
                project_candidates[str(candidate)]["git_status"] = run(["git", "--no-optional-locks", "-C", str(candidate), "status", "--short"])
    return {"schema_version": 1, "status": "INVENTORY_CAPTURED_NOT_INFERENCE_PROOF",
            "scope": "AUTHENTICATED_DEVICE_READ_ONLY_INVENTORY",
            "remote_reported_utc": datetime.now(timezone.utc).isoformat(),
            "host": {"system": platform.system(), "machine": platform.machine(),
                     "libc": list(platform.libc_ver()), "python": platform.python_version(),
                     "python_executable": sys.executable, "cwd": str(Path.cwd())},
            "files": {path: read_text(path) for path in files},
            "python_package_versions_without_importing_ml": packages,
            "python_module_locations_without_importing_ml": module_locations,
            "disk": disks, "video_nodes": videos, "commands": command_results,
            "existing_project_candidates": project_candidates,
            "elapsed_seconds": time.monotonic() - started,
            "packages_installed": False, "device_files_written_by_script": False,
            "power_mode_changed": False, "camera_frame_captured": False,
            "vision_inference_run": False, "llama_inference_run": False,
            "offline_disconnection_verified": False}


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2, allow_nan=False))
