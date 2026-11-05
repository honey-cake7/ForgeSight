# Hardware evidence

Observed during Stage 0.1 on 2026-11-05 (Asia/Calcutta). Memory availability and
free disk space are snapshots, not reserved resources.

## Development PC: verified observations

| Item | Observed value | Evidence |
| --- | --- | --- |
| Workstation | HP Z2 Tower G9 Workstation Desktop PC | `Get-CimInstance Win32_ComputerSystem` |
| OS | Microsoft Windows 11 Pro, 64-bit, 10.0.26300 | `Get-CimInstance Win32_OperatingSystem` |
| CPU | Intel Core i9-14900K, 24 cores, 32 logical processors | `Get-CimInstance Win32_Processor` |
| RAM | 63.71 GiB usable physical RAM | Windows `GlobalMemoryStatusEx` |
| GPU | NVIDIA RTX 2000 Ada Generation | `nvidia-smi` |
| GPU memory | 16,380 MiB | `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` |
| GPU driver | 582.08 | Same GPU query |
| C: storage | 475.68 GiB total; about 245.96 GiB free | Python `shutil.disk_usage` |
| D: storage | 1,863.00 GiB total; about 1,824.01 GiB free | Python `shutil.disk_usage` |
| WSL | Ubuntu, Linux 6.18.33.2-microsoft-standard-WSL2, x86_64 | `wsl --list --quiet`; `uname -srm` in Ubuntu |

Initial sandboxed Windows management queries were denied. Repeating the
read-only queries with approved escalation returned the observations above.
The registry's legacy Windows product-name string differed from the OS query;
the table uses the Windows management OS caption.

## Platform eligibility clarification

The organizer reply shared by the user on 2026-11-06 accepts NVIDIA Jetson
Orin as an eligible PS5 platform, subject to mandatory on-device Llama through
ExecuTorch and local defect/quality-control operation without cloud dependency.
Continue with the proposed Orin Nano route. See the
[recorded reply and implications](ORGANIZER_CLARIFICATION.md).
Platform-family eligibility does not verify our actual module or deployment.

## Jetson device inventory: BLOCKED - NOT VERIFIED

The proposed target is an NVIDIA Jetson Orin Nano. The user confirmed that the
device is not currently available. Do not infer module variant, RAM, JetPack,
CPU configuration, power mode, or camera from the proposed product name.

During the audit:

- Present-device enumeration returned no Jetson/APX/NVIDIA USB, relevant USB
  serial/network device, or camera matching the audit filters.
- The PC's physical Ethernet adapter reported disconnected; Wi-Fi was active.
- No Jetson SSH address or authenticated device session was established.

Absence from these filters alone does not prove that no Jetson is connected by
another route. No Jetson-side command has been executed.

When the device becomes available, run a separate read-only audit on it. Start
with the following Linux commands, recording errors as evidence too:

```sh
tr -d '\000' < /proc/device-tree/model
uname -a
uname -m
cat /etc/os-release
free -h
df -h
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack nvidia-l4t-core
ls /dev/video*
```

The `nvidia-jetpack` metapackage or camera nodes may be absent. Absence needs
interpretation, not an installation attempt during the audit. Device model
alone may not identify module SKU or physical RAM; confirm those separately.
Camera presence does not prove capture, and free RAM does not establish LLM
runtime feasibility.
