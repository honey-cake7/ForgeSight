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

## Jetson device inventory: REPORTED AVAILABLE - DEVICE AUDIT PENDING

On 2026-11-11 (Asia/Calcutta), the user reported receiving an **NVIDIA Jetson
Orin Nano 8GB and a Logitech USB webcam**. This supersedes the previous report
that no device was available. The exact module/SKU, physical memory, JetPack/L4T,
storage, power mode, webcam model and capture capability have not been read from
the device. Treat 8GB and Logitech USB as user-reported facts until audited.

At the arrival check, the Windows workspace was on `main` at `461276d`, clean before this
documentation update. The read-only connection check found an SSH client, no
configured SSH host in the user's config, and no matching present Jetson/USB
network/serial/Logitech device through the selected Windows enumeration filter.
This does not rule out a Jetson on the network. No connection address, login or
authenticated device session has been established. Boot/setup state is unknown.
No installation, flashing, power-mode change or camera capture was attempted.

Next: obtain the boot state and, if available, the SSH username/address. Run the
read-only inventory below on the actual Jetson before choosing target packages
or adapting the desktop runner. Then resume the earliest open Stage 2 mandatory
Llama/ExecuTorch proof, followed by the Stage 8 vision/camera deployment gate.
PC Substep 9.3 is secondary while hardware access is being established.

During the original 2026-11-05 audit:

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

## Through Stage 5: hardware evidence boundary

Stages 3.4?5 ran in the existing desktop WSL vision environment on the RTX 2000
Ada. They add saved-image evaluation, replay gating and pixel grading only.
No Jetson connection, camera command, JetPack change or on-device measurement
occurred. Device availability is unchanged: BLOCKED - NOT VERIFIED. Desktop
cached model-only timings are not complete inspection or Jetson timings.

## Stage 6 persistence evidence

The SQLite writer ran on desktop WSL; separate project-environment Python
processes queried the committed records after the writer exited. Storage
requires no GPU or added package. Actual vision replay still used the desktop
RTX 2000 Ada, not Jetson. No device became connected or available in this step.
The stage does not establish camera identity, edge database compatibility,
thermal/latency behavior or disconnected Jetson operation.

## Stage 7 local operator proof

The API/model ran in a new WSL app environment on the desktop RTX 2000 Ada.
Windows headless Chromium exercised the real loopback UI. Both valid image
scores/geometry matched earlier results; server restart preserved two records.
No Jetson or camera connected, and browser-local requests do not establish a
disconnected edge test. Actual device inventory and camera domain remain the
next Stage 8 gate. No desktop UI timing is promoted to Jetson performance.

## Stage 9.1 advisory hardware boundary

The actual Llama advisory test used the existing Linux x86_64 CPU ExecuTorch
runner on desktop WSL, with eight PC CPU threads. One cold child process took
8.457 seconds, native inference 1.348 seconds, sampled peak RSS 1,220,356 KiB.
These exclude parent/UI/vision memory and are not Jetson limits or benchmarks.
The 1B artifact, 128/512 prefill/context limits and 96-token generation bound
were retained. The worker explicitly rejects unverified ARM use; actual native
Jetson execution/compatibility and camera validation remain BLOCKED.

## Stage 9.2 refusal-check boundary

Input/output contract checks also passed in the minimal image environment,
without Torch or Llama imports. This supports a lightweight host contract, not
ARM execution proof. Two normal-length native PC runs used the same quantized
1B artifact and 82-token prompt, with native inference intervals 1.361/1.362
seconds and sampled child RSS 1,219,960/1,219,204 KiB. Truncation, timeout and
missing-runtime checks kept advice unavailable. Whole-process load costs and
CPU contention still need target measurements. No Jetson or camera was used;
native ARM ExecuTorch execution and disconnected operation remain unverified.
