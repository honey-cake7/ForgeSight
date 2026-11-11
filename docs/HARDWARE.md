# Hardware evidence

Latest runtime checkpoint: Substep 2.10 generated 40 Llama tokens through
ExecuTorch on this Orin Nano. The separate ARM64 CPU environment remains in use;
system Python is still 3.8.10. Camera inspection remains open.

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

## Jetson device inventory: AUTHENTICATED READ-ONLY AUDIT VERIFIED

On 2026-11-11 an SSH session authenticated after the user-supplied device-console
ED25519 fingerprint matched the advertised key. Strict host-key checking was
retained; the verified key was saved on the PC. Credentials are not recorded in
the repository or audit artifacts. The session closed normally after inspection.
No package install, model transfer, flashing, power-mode change or camera stream
was performed. This completes the first device inventory, not Stages 2 or 8.

| Item | Actual observation |
| --- | --- |
| Device-tree model | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| Module / carrier identifiers | `p3767-0003` / `p3768-0000`, `super` configuration |
| Module interpretation | Orin Nano 8GB; NVIDIA maps P3767-0003 to this variant |
| Architecture / CPU count | Linux `aarch64`; 6 online CPUs |
| Kernel / L4T | `5.10.216-tegra`; release 35.6.1 |
| OS / libc | Ubuntu 20.04.6 LTS; glibc 2.31 |
| System Python | `/usr/bin/python3`, 3.8.10 |
| Memory | 7,629,624 KiB (7.28 GiB) reported total; 4.28 GiB available at the final snapshot |
| Swap | 3.64 GiB configured across six zram devices; not additional physical RAM |
| Root storage | SD, `mmcblk1`, 119.2 GiB device; about 92.70 GiB free on root at the snapshot |
| CUDA / cuDNN | nvcc 11.4.315; cuDNN package 8.6.0.166 for CUDA 11.4 |
| Camera | Logitech C270, USB `046d:0825`; names at `/dev/video0` and `/dev/video1` refer to the same USB function |
| Existing Python modules | NumPy 1.17.4, Pillow 7.0.0; `cv2` discoverable without importing it |
| ML/advisory stack in system Python | Torch, torchvision and ExecuTorch not found by module/metadata checks; FastAPI/Uvicorn metadata absent |
| Toolchain | GCC/G++ 9.4.0, CMake 3.16.3, Git 2.25.1 |
| Power query | Reports `25W`, mode 1, but also EMC permission errors; full power/clock verification remains open |

The hardware mapping comes from NVIDIA's [module table](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/IN/QuickStart.html).
NVIDIA maps [L4T 35.6.1 to JetPack 5.1.5](https://developer.nvidia.com/embedded/jetson-linux-r3561).
The `nvidia-jetpack` metapackage is absent, so record this as a release-family
mapping rather than a fully installed/verified JetPack bundle. The package query
returned 1 because some requested package names were absent; its successful
component-version output is retained. `v4l2-ctl` is unavailable, leaving format
enumeration unverified. No attempt was made to install it. User-reported camera
preview remains distinct from project-controlled frame capture and inference.

`scripts/audit_jetson_readonly.py` ran on the actual Python 3.8 interpreter and
uses only standard-library inspection. Existing project checks were limited to
six named home/Desktop/projects paths; none existed. This is not an exhaustive
search or permission to overwrite another directory. System ML modules were not
imported, and no CUDA tensor or model forward was executed.

The console initially altered long JSON lines; a first encoded capture also hit
the capture token limit. Both attempts are retained. The final short-line
compressed transfer passed SHA-256 verification before JSON parsing. Authoritative
local evidence is `artifacts/hardware/jetson_audit_2026_11_11_01/device_audit.json`;
the committed summary/provenance is `manifests/jetson_device_audit_v1.json`.
The script was checked for Python 3.8 syntax and the transported source matched
its local SHA-256. No broad PC test rerun or model-performance claim is made.

Substep 2.8 subsequently read definitions from ten actual system libraries over
authenticated SSH, without writing target files or importing ML runtimes. All
33 ELF files in the three selected core CPU wheels match those library/version
requirements. The target remains Python 3.8.10/glibc 2.31. No new interpreter,
package, model or manual-audit folder was created. The user reports removing the
manual-audit copy after running it; that folder was not recreated or relied on.
See `docs/JETSON_RUNTIME.md` and the compatibility manifest.

The candidate uses separate CPython 3.12.14 and CPU wheels. The interpreter asset
was subsequently downloaded/hash-verified on the PC; its ten ARM native files
bring the checked total to 43. A fresh read-only SSH probe confirmed the two
additional libraries, libcrypt and libutil, for a total of 12 system libraries.
System Python is still 3.8.10, the package inventory and previous ten library
hashes are unchanged, and no interpreter/package/files were installed on target.
The deleted manual-audit folder was not recreated. A native build remains a fallback and would need
newer CMake. No OS change or source build on Jetson is selected. Llama generation,
GPU vision, camera capture, latency/memory and offline operation remain unverified.

### Arrival and connection history

On 2026-11-11 (Asia/Calcutta), the user reported receiving an **NVIDIA Jetson
Orin Nano 8GB and a Logitech USB webcam**. This supersedes the previous report
that no device was available. The exact module/SKU, physical memory, JetPack/L4T,
storage, power mode, webcam model and capture capability have not been read from
the device. Treat 8GB and Logitech USB as user-reported facts until audited.

At the arrival check, the Windows workspace was on `main` at `461276d`, clean before this
documentation update. The read-only connection check found an SSH client, no
configured SSH host in the user's config, and no matching present Jetson/USB
network/serial/Logitech device through the selected Windows enumeration filter.
This did not rule out a Jetson on the network. At that initial check, no login
or address had been supplied and boot/setup state was unknown. No authenticated
device session had been established.
No installation, flashing, power-mode change or camera capture was attempted.

The user subsequently reported a running Ubuntu desktop, working webcam preview
in its camera app, login username `nvidia`, two IPv4 addresses and SSH service
status `active`. These are supplied observations, not an authenticated audit.
Private connection addresses stay in the ignored local evidence rather than
the public documentation.

A strict-host-key, batch-mode Windows SSH attempt to the supplied LAN address
timed out on port 22 before authentication. A separate WSL TCP connection check
returned `OSError: [Errno 113] No route to host`. Windows selected an on-link
Wi-Fi route whose /23 subnet includes the supplied Jetson address. This alone
does not establish reachability. Network isolation, endpoint configuration and
address changes are possibilities, not confirmed causes. No host-key bypass,
firewall change, installation or Jetson command was performed.

The user then pasted Jetson terminal output: `wlan0` is UP with the supplied
LAN address and /23 prefix; `docker0` is DOWN with the second address. SSH
listens on IPv4 `0.0.0.0:22` and IPv6 `[::]:22`. A subsequent strict Windows
SSH retry reached a server at the expected address, but stopped because no
ED25519 host key is trusted yet. A read-only key scan captured the advertised
public key and its SHA-256 fingerprint. Its identity is NOT VERIFIED until
compared with the fingerprint read directly on the Jetson. No host-key check
was bypassed, no known_hosts entry was added and no login was attempted.

Raw initial results are in
`artifacts/hardware/connection_attempt_2026_11_11_01/connection_report.json`;
the reachable-server/candidate-key evidence is in
`artifacts/hardware/connection_attempt_2026_11_11_02/`.
Next: obtain `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` directly from
the Jetson and compare fingerprints, then establish authentication. Run the
read-only inventory below before choosing target packages
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

## Substep 2.9 actual device runtime boundary

The actual Orin Nano now runs the separate Python 3.12.14 CPU venv at
`/home/nvidia/.local/share/forgesight/llama_cpu_v1/venv`. Torch 2.14.0+cpu and
ExecuTorch 1.5.1+cpu imported, CPU arithmetic returned 10.0, XNNPACK is registered,
and loading the pinned libraries registered all three required Llama operators.
System Python remains 3.8.10; system packages and five checked system/configuration
files match before/after. No sudo, CUDA, camera, power-mode or OS changes were used.

The one import/tensor probe took 12.284 seconds with peak process RSS 388,604 KiB.
These exclude a loaded model and future vision/UI work; they are not inference
benchmarks or proof of combined memory fit. The one-thread probe setting is not
a model thread-budget decision. No PTE/tokenizer weights were copied and no
Llama-generated response was produced. See `manifests/jetson_runtime_installation_v1.json`.

## Substep 2.10 actual Llama model run

The identified Orin Nano, reached at its new Wi-Fi address after matching the
previously confirmed SSH host fingerprint, ran the 802,247,040-byte quantized
Llama 3.2 1B PTE with the checked tokenizer. Two CPU threads produced 40 tokens.
Native model load was 3.087 s; native inference 2.129 s; whole child process
30.856 s. Sampled peak child RSS was 1,189,184 KiB. These are one-run values
with no concurrent camera/vision/UI and no p50/p95 meaning. Full hashes, output
and limits are in `manifests/jetson_llama_generation_v1.json`. No camera frame
was captured, no physical part inspected and no disconnected test was run.
