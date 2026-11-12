# Jetson Llama runtime — Substeps 2.8 and 2.9

Status: **Llama 3.2 1B generated real text through native ExecuTorch on the
actual Orin Nano** in Substep 2.10. Stage 10.1 later ran one saved-image
vision-plus-advice replay while Wi-Fi was disconnected. See
`manifests/jetson_offline_saved_replay_pilot_v1.json`; full physical-camera
offline acceptance remains open.

## Stage 10.7 disconnected ExecuTorch advisory latency

`scripts/jetson_offline_advice_latency.py` verifies the prior saved REVIEW
record, SQLite hash, isolated Llama runtime and Wi-Fi route. It schedules an
independent seven-minute reconnect timer, disconnects Wi-Fi and runs eight
fresh advisory processes through ExecuTorch. It checks each note's local
evidence IDs and unchanged deterministic grade, samples route/memory/thermal
state, then restores Wi-Fi. No new vision inference or inspection occurs.

```sh
cd /home/nvidia/.local/share/forgesight/vision_cpu_v1
python3 -B scripts/jetson_offline_advice_latency.py --prefix "$PWD" --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 --output-dir "$PWD/runs/offline_advice_latency_01" --preflight-only
sudo systemd-run --unit=forgesight-offline-advice-latency-v1 --collect /usr/bin/python3 -B "$PWD/scripts/jetson_offline_advice_latency.py" --prefix "$PWD" --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 --output-dir "$PWD/runs/offline_advice_latency_01"
```

Use fresh unit/output names for a repeat. Raw advice remains in the private
Jetson runtime; the report is copied to ignored PC artifacts. See
`manifests/jetson_offline_advice_latency_v1.json`.

## Stage 10.6 disconnected saved-replay vision latency

`scripts/jetson_offline_vision_latency.py` checks the two known image hashes,
dataset roles, frozen model and Wi-Fi route. After scheduling an independent
eight-minute reconnect timer, it disconnects Wi-Fi and runs 20 fresh
EfficientAD/ForgeGrade/SQLite replay processes. It measures model forward,
replay-to-commit and process-wall durations separately, then checks all 20
SQLite records. Network, thermal sensors and MemAvailable are sampled about
once per second. Wi-Fi is restored and the timer canceled. The two images
repeat; no physical-part rate is calculated.

```sh
cd /home/nvidia/.local/share/forgesight/vision_cpu_v1
python3 -B scripts/jetson_offline_vision_latency.py --prefix "$PWD" --output-dir "$PWD/runs/offline_vision_latency_01" --preflight-only
sudo systemd-run --unit=forgesight-offline-vision-latency-v1 --collect /usr/bin/python3 -B "$PWD/scripts/jetson_offline_vision_latency.py" --prefix "$PWD" --output-dir "$PWD/runs/offline_vision_latency_01"
```

Use fresh unit/output names for a repeat. Raw maps and records stay in the
private runtime; the report and SQLite copy are ignored PC artifacts. See
`manifests/jetson_offline_vision_latency_v1.json`.

## Stage 10.5 disconnected camera recovery diagnostic

`scripts/jetson_offline_camera_recovery.py` checks the physical Jetson,
`/dev/video0` and frozen model hash, schedules an independent three-minute
Wi-Fi reconnect timer, and disconnects Wi-Fi. The camera diagnostic first
uses absent V4L2 index 99, then the connected C270 index 0. It verifies the
error report, real frame and anomaly map, and records interface/route samples
before restoring Wi-Fi. The camera stays connected; no unplug/replug is
claimed. The frame contains no presented nut and is not an inspection.

```sh
cd /home/nvidia/.local/share/forgesight/vision_cpu_v1
python3 -B scripts/jetson_offline_camera_recovery.py --prefix "$PWD" --output-dir "$PWD/runs/offline_camera_recovery_01" --preflight-only
sudo systemd-run --unit=forgesight-offline-camera-recovery-v1 --collect /usr/bin/python3 -B "$PWD/scripts/jetson_offline_camera_recovery.py" --prefix "$PWD" --output-dir "$PWD/runs/offline_camera_recovery_01"
```

Use fresh output and transient unit names for a repeat. Raw camera imagery
stays private under the Jetson runtime and the ignored PC artifact directory.
See `manifests/jetson_offline_camera_recovery_v1.json`.

## Stage 10.4 disconnected failure recovery

`scripts/jetson_offline_failure_recovery.py` checks the source image/model
hashes and sole Wi-Fi route, schedules an independent four-minute reconnect
timer, then disconnects Wi-Fi. It passes a nonexistent path in place of the
model argument without touching the working export, tests a corrupt image
copy against the manifest integrity gate, and runs the valid frozen image
again. It checks SQLite rows after each step, samples network state, restores
Wi-Fi and cancels the timer. Use a fresh output directory and transient unit
name for a repeat. Device commands:

```sh
cd /home/nvidia/.local/share/forgesight/vision_cpu_v1
python3 -B scripts/jetson_offline_failure_recovery.py --prefix "$PWD" --output-dir "$PWD/runs/offline_failure_recovery_01" --preflight-only
sudo systemd-run --unit=forgesight-offline-failure-recovery-v1 --collect /usr/bin/python3 -B "$PWD/scripts/jetson_offline_failure_recovery.py" --prefix "$PWD" --output-dir "$PWD/runs/offline_failure_recovery_01"
```

Raw reports and SQLite are private under that run directory, with an ignored
PC copy under `artifacts/offline_stage10_jetson_v1/offline_failure_recovery_01`.
The corrupt file fails integrity checking before decoder use; decoder-level
failure remains a separate test. See the Stage 10.4 manifest.

## Stage 10.3 disconnected loopback UI pilot

`scripts/jetson_offline_ui_pilot.py` reads the prior saved REVIEW replay,
checks its record and SQLite SHA-256, schedules an independent three-minute
Wi-Fi recovery timer, and disconnects Wi-Fi. A local HTTP client on the Jetson
fetches the operator page files and exercises the queue and status endpoints.
It checks generated advice, a separate one-second native timeout, and UI
process restart after each result. It then restores Wi-Fi and cancels the
timer. The script uses a fresh output directory, does not install packages,
and does not modify the original inspection. Device commands:

```sh
cd /home/nvidia/.local/share/forgesight/vision_cpu_v1
python3 -B scripts/jetson_offline_ui_pilot.py --prefix "$PWD" --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 --output-dir "$PWD/runs/offline_ui_01" --preflight-only
sudo systemd-run --unit=forgesight-offline-ui-pilot-v1 --collect /usr/bin/python3 -B "$PWD/scripts/jetson_offline_ui_pilot.py" --prefix "$PWD" --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 --output-dir "$PWD/runs/offline_ui_01"
```

The output directory and transient unit names must be changed for a repeat.
Raw logs and advice live in the private runtime directory, with an ignored PC
copy under `artifacts/offline_stage10_jetson_v1/offline_ui_01`. No Jetson
browser executable was present, so this proves local HTTP/API behavior and
assets, not offline browser rendering. See the Stage 10.3 manifest.

## Stage 10.2 disconnected saved-case series

`scripts/jetson_offline_case_series.py` checks frozen local model/image hashes
and dataset roles, schedules an independent four-minute Wi-Fi reconnect timer,
then runs three saved-image replays and one separate ExecuTorch advisory with
Wi-Fi disconnected. It records interface and route state about once per
second, reconnects Wi-Fi, and cancels the timer. One explicitly allowed fitting
normal image is a runtime reference only; held-out evaluation samples remain
blocked. It uses a fresh output directory and SQLite database. No Python
packages or system environment were changed. A repeat requires fresh transient
unit names and output path after checking systemd state. Device commands:

```sh
cd /home/nvidia/.local/share/forgesight/vision_cpu_v1
python3 -B scripts/jetson_offline_case_series.py --prefix "$PWD" --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 --output-dir "$PWD/runs/offline_case_series_01" --preflight-only
sudo systemd-run --unit=forgesight-offline-case-series-v1 --collect /usr/bin/python3 -B "$PWD/scripts/jetson_offline_case_series.py" --prefix "$PWD" --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 --output-dir "$PWD/runs/offline_case_series_01"
```

The raw report, SQLite file, maps and advice are in the private runtime path;
a copy is ignored under `artifacts/offline_stage10_jetson_v1/offline_case_series_01`.
The manifest contains measured results and limits.

## Stage 10.1 disconnected saved-image pilot

`scripts/jetson_offline_replay_pilot.py` checked the local image/model hashes
and sole Wi-Fi route, scheduled a three-minute systemd reconnect watchdog,
disconnected `wlan0`, ran the existing vision replay and advice controller as
the Jetson user, sampled network state before and after each, and restored
Wi-Fi. It used a fresh SQLite database and output directory. No Python package
was installed or changed. The device and copied reports agree by SHA-256.
The old native runner's default `offline_status` is not an observation for this
pilot; the separate `offline_report.json` records the network evidence. This
does not establish camera or full UI operation without a network.

The device-side preflight and detached launch used these commands from the
private vision directory (the launch briefly removes SSH access):

```sh
cd /home/nvidia/.local/share/forgesight/vision_cpu_v1
/usr/bin/python3 -B jetson_offline_replay_pilot.py --prefix "$PWD" --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 --output-dir "$PWD/runs/offline_pilot_01" --preflight-only
sudo systemd-run --unit=forgesight-offline-pilot-v1 --collect /usr/bin/python3 "$PWD/jetson_offline_replay_pilot.py" --prefix "$PWD" --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 --output-dir "$PWD/runs/offline_pilot_01"
```

The script refuses an existing output directory and refuses to disconnect if
the watchdog cannot be scheduled. Its transient unit names are for this one
pilot; use fresh names and output for a repeat after checking systemd state.

## Actual Llama generation: Substep 2.10

The ARM runner `forgesight/edge_jetson_smoke.py` checks the physical device,
private Python environment, package versions, model/export/tokenizer hashes,
chat tokens and generation bounds before calling `TextLLMRunner`. A separate
child process enforces a 240-second timeout and saves native stdout/stderr,
actual generated text, native token statistics and sampled process memory.
The existing desktop worker still rejects ARM use.

The verified PTE/tokenizer live in the existing private prefix's `model/`
directory. After `cd /home/nvidia/.local/share/forgesight/llama_cpu_v1/model`,
the actual command was:

```sh
../venv/bin/python -B edge_jetson_smoke.py --root .. --model llama3_2_1b_xnnpack.pte --tokenizer tokenizer.model --profile edge_llama.json --export llama3_2_1b_export.json --assets llama3_2_1b_assets.json --prompt edge_smoke_prompt.txt --output ../runs/llama_probe_01 --threads 2 --max-new-tokens 64 --timeout 240
```

The output directory must be fresh for another run. The prompt asked for one
sentence explaining stable lighting in image comparison. The Jetson generated:

> Stable lighting helps compare two inspection images by ensuring that the lighting conditions are consistent across both images, which helps to eliminate any potential differences in illumination that may be contributing to discrepancies in the inspection results.

Native statistics: 33 prompt tokens, 40 generated tokens, 3.087 s model load,
2.129 s inference (0.472 s prefill and 1.657 s decode), with 24.14 decode
tokens/s in this single run. Full child process wall time was 30.856 s, which
includes Python imports and startup. Sampled peak child RSS was 1,189,184 KiB;
the sample can miss a higher peak and excludes concurrent vision/UI memory.
The process exited 0. PTE, tokenizer and executed runner hashes matched both
the pinned records and the transferred files. The device stayed on Wi-Fi, so
disconnected cold-start operation is not yet verified. The generated sentence
is a smoke output, not an industrial cause or approved precaution.

The reviewable record is `manifests/jetson_llama_generation_v1.json`; full raw
logs and text are kept on Jetson in `runs/llama_probe_01/` and copied to the
ignored PC path `artifacts/edge/jetson_llama_v1/llama_probe_01/`. The three
principal output hashes matched between device and copy. All 151 PC tests pass.
Next: Stage 8.1, inspect the camera formats and Jetson vision dependencies.

## Earlier setup evidence

## Actual installation: Substep 2.9

The working SDK is `/home/nvidia/.local/share/forgesight/llama_cpu_v1`. Keep it;
this is separate from the previously deleted manual audit folder. It contains
`python/` (standalone interpreter), `venv/` (isolated CPU packages), `stage/`
(verified inputs), `logs/` and private caches. Persistent installation files
stay here; pip may use ordinary OS temporary build storage. No sudo, system
Python replacement, OS upgrade, CUDA change or power-mode change was used.

| Check on actual Jetson | Observed result |
| --- | --- |
| System interpreter | `/usr/bin/python3` remains 3.8.10 |
| Separate interpreter / venv | ARM64 Python 3.12.14; user site disabled |
| Core packages | Torch 2.14.0+cpu, ExecuTorch 1.5.1+cpu, pytorch-tokenizers 1.6.0, TorchAO 0.18.0 |
| Installation | 50 hashed package archives; 49 runtime dependencies plus wheel and bundled pip = 51 distributions |
| Dependency check | `No broken requirements found.` |
| CPU arithmetic | Small tensor expression returned 10.0; Torch has no CUDA build |
| Native bindings | TextLLMRunner, GenerationConfig and CppTiktoken loaded/constructed as applicable; no runner/model constructed |
| Backend registry | XnnpackBackend present; delegation/model execution not tested |
| Required operators | embedding_4bit.dtype_out, update_cache.out and custom_sdpa.out all present after explicit library loading |
| Preservation | System package inventory and five system/configuration hashes identical before/after |

The required operators were absent before loading the two pinned kernel
libraries, then present afterward. The native runner and both kernel-library
hashes match the publisher-verified ARM CPU wheel. This establishes imports and
registration; actual Llama operator execution is still a separate test.

The import/tensor probe took 12.284 seconds and reported process peak RSS
388,604 KiB. These are **not model latency or model memory measurements** and
exclude future Llama/vision/UI work. One CPU thread was used for this probe only;
the eventual model thread budget remains unselected.

Inputs and returned evidence archives were SHA-256 verified. Installation used
`--no-index --find-links ... --require-hashes`; only the two direct CPU wheel
selectors were rewritten to exact local `+cpu` versions, retaining all hashes.
Pinned setuptools/wheel were installed first; the antlr4 source dependency built
on target with `--no-build-isolation`. This local-file install does not establish
disconnected cold-start operation. No PTE/tokenizer weights were transferred.

The committed record is `manifests/jetson_runtime_installation_v1.json`.
Full logs, inventories, inputs and returned evidence remain locally under
`artifacts/edge/jetson_runtime_setup_v1/`. `installation.json` is the earlier
pre-probe checkpoint; `native_probe.json` records the subsequent successful check.
All 147 PC software tests passed, including seven new setup safeguards. Two PC
preparation errors were fixed before any target changes; those attempts remain
recorded. Five PC environments/eight protected files match the prior snapshot.

To inspect the installed versions in the Jetson SSH terminal:

```sh
/usr/bin/python3 --version
/home/nvidia/.local/share/forgesight/llama_cpu_v1/venv/bin/python --version
/home/nvidia/.local/share/forgesight/llama_cpu_v1/venv/bin/python -B -m pip check
```

These should report 3.8.10, 3.12.14 and no broken requirements respectively.
The installed SDK is not disposable: do not rerun initialization/bootstrap over it.

## Selected first route

Use a separate ARM64 CPython 3.12.14 environment with ExecuTorch 1.5.1+cpu,
PyTorch 2.14.0+cpu, pytorch-tokenizers 1.6.0 and TorchAO 0.18.0. Keep the
existing Llama 3.2 1B artifact, 128-token prefill bound and 512-token context.
The first bounded generation will use at most 64 new tokens. Do not substitute
another Llama runtime or copy the desktop x86_64 binary environment.

This route follows the already working PC native bindings, with matching
versions and operator libraries. CPU/XNNPACK advice can live separately from
the later CUDA vision environment. It is a feasibility choice, not a claim of
better Jetson latency. Thread count and combined CPU/RAM pressure need real
measurement; the desktop eight-thread setting is not a target budget.

The system Python remains `/usr/bin/python3` 3.8.10, Ubuntu remains 20.04.6,
L4T remains 35.6.1 and CUDA remains 11.4. Substep 2.8 read library metadata;
2.9 installed only the isolated CPU runtime. The deleted audit folder was not recreated.

| Option | Trade-off | Decision |
| --- | --- | --- |
| Isolated Python + verified ARM CPU wheels | Closest to tested PC bindings; no native build needed if actual imports work; brings Torch and export-related package dependencies | First candidate |
| Native C++ ExecuTorch runner | Could reduce Python/export-package overhead, but needs additional build/toolchain and adapter work | Fallback if runtime or resource tests fail |
| Change JetPack/Ubuntu | Changes a working camera/driver baseline; system migration has wider consequences | Not selected; static findings do not establish a need |

## Historical static evidence: Substep 2.8, before installation

Four official ARM64 wheels were downloaded to the PC and checked against
publisher SHA-256 values: CPU ExecuTorch, CPU Torch, pytorch-tokenizers and the
plain PyPI ExecuTorch comparison wheel. No ARM wheel was installed or executed
on the PC or Jetson. The selected CPU wheel contains the actual native runner,
XNNPACK delegate, quantized kernels and LLM custom-operator library.

The reusable probe inspected **33 ELF binaries** in the three selected core
wheels. Each reports AArch64 machine ID 183. Its exact required GNU library
versions were compared with definitions read from **ten real Jetson system
libraries**, in an authenticated read-only SSH session. All checked requirements
matched. The highest core GLIBC requirement is 2.28; target libc is 2.31. No
CUDA library dependency was observed in these selected core ELF dependencies.

This is necessary static evidence only. It does not prove loader search paths,
CPU-instruction compatibility, Python imports, operator registration, model
execution, combined memory fit or performance. Other resolved dependencies
have compatible wheel tags according to the target resolver, but their complete
ELF/runtime behavior was not checked by this core probe. The interpreter archive
was subsequently acquired and inspected as described below; its executable has
not been run.

`requirements/pylock.jetson_llama.toml` records **49 resolved packages** for
CPython 3.12.14 and `aarch64-manylinux_2_31`. The hash-locked requirements file
provides pip-compatible installation input. These are CPU runtime dependencies;
they must not replace the Jetson's CUDA vision packages or system packages.

One dependency, antlr4-python3-runtime 4.9.3, is source-only in the lock. Its
publisher-hashed source successfully built into a `py3-none-any` wheel with no
native binaries in a new isolated PC environment, using the pinned setuptools
84.2.1 and wheel 0.45.1 build tools. The five existing environments are preserved.
No target build/install was attempted. A future target installation must use the
locked build tools and `--no-build-isolation`, so build-tool selection is controlled.

The interpreter asset is pinned to CPython 3.12.14, standalone build 20260901,
SHA-256 `577b4bec0793ad1ff0cbff9adbd0df078eddde38a4c41bf5d83ad381a85ee39d`.
Official release metadata confirms its name and 29,199,399-byte size. The initial
Astral URL returned HTTP 403 and canonical GitHub attempts timed out. Those
failures and the initial SSH banner timeout remain retained, not counted as
successful checks. After the user restored network connectivity, the canonical
GitHub download completed and matched the exact size and pinned SHA-256.
An initial CPU-wheel URL encoding error was corrected; all four wheel hashes
then passed.

The interpreter follow-up checked all 4,533 archive paths and 1,048 links for
prefix containment. Its ten ELF files are ARM64; `venv` and `ensurepip` source
files are present. The extra required `libcrypt.so.1` and `libutil.so.1` were
read from the actual Jetson after SSH connectivity was restored. Their required
GLIBC_2.17 definitions are present. Combined with the original core probe, all
43 checked native files match 12 system-library inventories. The archive's
literal `$ORIGIN/../lib/libpython3.12.so.1.0` dependency resolves to the included
library within the archive. This is static resolution, not a loader execution test.

The target still reports Python 3.8.10. Its package inventory and the hashes of
the previous ten system libraries match the earlier read-only probe. No new
interpreter/package/model was installed, executed or copied to the Jetson; only
in-memory system-Python queries ran. `manifests/jetson_python_archive_v1.json`
records this follow-up. Old failure evidence and the initial compatibility
manifest remain historical snapshots. The broad PC tests were not rerun because
runtime code is unchanged; the last full suite result is 140 passing tests.

## Files and reproducible PC checks

- `configs/jetson_llama_runtime_v1.json`: exact candidate, limits, hashes and gates.
- `requirements/jetson_llama.in`: four direct runtime requirements.
- `requirements/pylock.jetson_llama.toml`: ARM-targeted dependency resolution.
- `requirements/jetson_llama.lock.txt`: hash-locked runtime installation input.
- `requirements/jetson_build_tools.in` and `.lock.txt`: pinned source-build tools.
- `forgesight/edge_compatibility.py`: static inspector and read-only target collector.
- `manifests/jetson_runtime_compatibility_v1.json`: actual results and limitations.

On the configured PC, choose a fresh output directory:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.edge_compatibility --wheels artifacts/edge/jetson_compatibility_v1/downloaded_wheels.json --target artifacts/edge/jetson_compatibility_v1/target_libraries.json --output artifacts/edge/jetson_compatibility_v1/recheck-001
wsl -d Ubuntu --exec /root/.cache/forgesight/app-v1/venv/bin/python -B -m unittest tests.test_edge_compatibility -q
```

The first command parses wheel binaries with `readelf` on the PC; it does not
execute them. The target collector (`--collect-target`) runs from memory with
Python 3.8-compatible source, reads loader-cache/version metadata, and writes no
target files. Missing libraries, symbol versions, required runner files, wrong
architecture, wheel hash changes and unexpected CUDA dependencies fail checks.

Local evidence is under `artifacts/edge/jetson_compatibility_v1/`, including
downloaded wheels, target-library data, ELF reports, Python release metadata,
download failures and the isolated pure-Python build log. Raw artifacts stay
outside Git; no model/data or credential values are committed.

## Guarded installation sequence used in Substep 2.9

The tested installer is `forgesight/edge_runtime_setup.py`, whose bootstrap
functions run under system Python 3.8. Initialization refuses an existing prefix,
root execution, a different device or insufficient storage. Extraction validates
all paths/links before writing and verifies archive/input hashes. The executed
source hashes are recorded; a later module-docstring clarification changed no
executable AST. The sequence used was:

1. Use a **fresh** user-owned prefix beneath `~/.local/share/forgesight/`.
   Refuse an existing prefix; never overwrite it or change `/usr/bin/python3`.
2. Extract the verified interpreter there and invoke its absolute executable
   to create a separate venv. Verify Python 3.12.14, AArch64 and isolation.
3. Stage the committed hash locks. From that isolated venv, the planned pip
   operations were (shown with shortened paths):

   ```sh
   python -m pip --isolated install --no-index --find-links wheelhouse --require-hashes --only-binary=:all: -r build_tools.lock.txt
   python -m pip --isolated install --no-index --find-links wheelhouse --require-hashes --no-build-isolation -r offline_runtime.lock.txt
   python -m pip check
   ```

   Here `python` is the absolute isolated venv executable. The exact executed
   argument lists and log hashes are in the installation manifest. The lock and
   wheelhouse are verified files inside `stage/`, not arbitrary downloads.
4. Test CPU Torch, native bindings, tokenizer and actual operator registration.
   Record failures and memory before model transfer; don't install the GPU
   vision stack into this CPU-only environment.
The native probe is `forgesight/edge_runtime_probe.py`. It requires the actual
device and isolated interpreter, checks versions/module locations, performs CPU
arithmetic and registers operators. It deliberately loads no model. To repeat
it manually, choose a new output filename and keep existing evidence:

```sh
forgesight_runtime="$HOME/.local/share/forgesight/llama_cpu_v1"
"$forgesight_runtime/venv/bin/python" -B "$forgesight_runtime/stage/edge_runtime_probe.py" --prefix "$forgesight_runtime" --output "$forgesight_runtime/native_probe_repeat_001.json"
```

This was the completed 2.9 setup sequence. Substep 2.10 subsequently transferred
the verified assets and ran the bounded generation above. The next action is
Stage 8.1 camera and vision compatibility work.

## Primary sources

- [Pinned ExecuTorch package metadata](https://pypi.org/pypi/executorch/1.5.1/json)
- [Official CPU wheel index](https://download.pytorch.org/whl/cpu/executorch/)
- [Pinned LLM Python bindings](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/extension/llm/runner/__init__.py)
- [Pinned native build](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/CMakeLists.txt)
- [Python standalone release](https://github.com/astral-sh/python-build-standalone/releases/tag/20260901)
- [Release asset metadata](https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/20260901)

The Raspberry Pi tutorial is an ARM reference; its system-library replacement
instructions are not adopted. The on-device generation proof is the new 2.10
record above. Camera inference and disconnected operation remain unverified.
