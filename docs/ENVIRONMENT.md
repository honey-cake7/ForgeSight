# Development environment and audit

The Windows and shared WSL observations below are audit records, not a full
project dependency lock. No packages were installed during Stage 0.1 or Substep
0.2. Substep 0.3 installed the single pinned dependency in an isolated project
environment and verified the setup as described below.

## Windows shell

- Python 3.10.11, AMD64.
- Executable: `C:\Users\RF AND SIMULATION\AppData\Local\Programs\Python\Python310\python.exe`.
- `sys.prefix == sys.base_prefix`: no active virtual environment.
- pip 23.0.1; Git 2.56.0.windows.2.
- Checked packages absent: torch, torchvision, executorch, anomalib, OpenCV,
  NumPy, ONNX Runtime, TensorRT, and the checked PDF-reading libraries.
- `py -0p` reported no registered Pythons even though `python` runs. Resolve
  interpreters explicitly rather than assuming the launcher selects correctly.

Read-only inspection commands from PowerShell:

```powershell
Get-Command python,git,wsl,nvidia-smi
python -B --version
python -B -m pip --version
git --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

## Existing Ubuntu WSL environment

| Component | Observed value |
| --- | --- |
| Interpreter | `/root/pytorch-env/bin/python3` |
| Python | 3.12.14 |
| Environment prefix | `/root/pytorch-env` |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| NumPy | 1.26.4 |
| PyTorch CUDA build | 12.4 |
| `torch.cuda.is_available()` | `True` |
| `torch.cuda.get_device_name(0)` | `NVIDIA RTX 2000 Ada Generation` |

ExecuTorch, anomalib, OpenCV, ONNX Runtime, and TensorRT were absent in this
environment. The observations apply to this interpreter, not every installed
environment on the machine.

To repeat the check, enter the existing distribution from PowerShell:

```powershell
wsl -d Ubuntu
```

Then, in its Linux shell:

```sh
/root/pytorch-env/bin/python3 -B - <<'PY'
import sys
import torch
import torchvision
import numpy

print('Python:', sys.version)
print('Executable:', sys.executable)
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('numpy:', numpy.__version__)
print('PyTorch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
```

The recorded check establishes importability and CUDA device discovery. It does
not prove a model executes correctly or establishes any performance result.
PyTorch's CUDA build version is not proof that a standalone CUDA toolkit is
installed. WSL runs on the PC's x86_64 CPU and desktop GPU, not Jetson ARM64.

## Project environment: Stage 0.3

Target tested: Ubuntu WSL, Linux x86_64, CPython 3.12.14. This is a PC image-input
environment; Jetson dependencies need a separate compatibility decision after
the device audit.

Use `.venv` for this project. The standard-library `venv` defaults to excluding
system packages. The existing `/root/pytorch-env` supplies the Python interpreter
for creation; its installed packages are not inherited or modified. Its complete
87-package `pip freeze --all` inventory was compared before and after setup.

Only Pillow 13.0.0 is needed to start reading and inspecting image files.
Its CPython 3.12 Linux x86_64 wheel is pinned by SHA-256 in `requirements.txt`.
PyPI declares Python >=3.11 and the MIT-CMU license for this release. Do not use
this one-wheel lock as a Windows, ARM64, or other Python-version setup.

Sources: [Pillow release](https://pypi.org/project/pillow/13.0.0/),
[Python venv documentation](https://docs.python.org/3.12/library/venv.html).

For a fresh environment, run the following from PowerShell. The guard protects
an existing environment from being silently reused:

```powershell
Set-Location -LiteralPath 'C:\Users\RF AND SIMULATION\Desktop\honey\ForgeSight'
if (Test-Path -LiteralPath .venv) { throw 'A .venv already exists; inspect it first.' }
wsl -d Ubuntu --exec /root/pytorch-env/bin/python3 -B -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
wsl -d Ubuntu --exec .venv/bin/python -B -m pip --require-virtualenv install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
wsl -d Ubuntu --exec .venv/bin/python -B scripts/check_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Smoke check failed.' }
wsl -d Ubuntu --exec .venv/bin/python -B -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
```

On another Linux development machine, use an explicitly installed CPython 3.12
interpreter to create its own venv; the `/root/pytorch-env` path above is specific
to this workstation. Provisioning a fresh OS/Python installation is not verified
by this step. Always recreate environments locally rather than copying `.venv`.

Observed project output:

```text
Python: 3.12.14
Platform: linux x86_64
Isolation: PASS (virtual environment; user site disabled)
Pillow: 13.0.0
PNG round trip: PASS (3x2 RGB; all pixel bytes preserved)
Environment smoke check: PASS; no defect model or Jetson was tested.
No broken requirements found.
```

The only installed distributions were `pillow==13.0.0` and bundled
`pip==25.0.1`. The bootstrap pip version is recorded, not included in the image
dependency lock. The lock does not recreate Python, OS libraries, or drivers.
Inspection confirmed `include-system-site-packages = false` and no visibility
of torch, torchvision, NumPy, ExecuTorch, or OpenCV from the project environment.
The image checks use known synthetic pixels solely to exercise PNG codecs; they
are not defect predictions or dataset evidence.

A second fresh venv in a temporary Linux directory installed from the same
requirements file and passed both the dependency and codec checks. The
temporary directory was then removed automatically. This proves the package
setup can be repeated on the tested host, not that it works offline or on Jetson.

## Separate exporter environment (Stage 2.5)

The image environment above remains unchanged. Export tools are installed in
`/root/.cache/forgesight/et1.5.1/venv`, with source alongside it on WSL's Linux
filesystem. CPU PyTorch 2.14.0+cpu and ExecuTorch 1.5.1+cpu passed actual startup
and configuration checks. The shared CUDA environment also remains unchanged.
The 86-dependency lock was reproduced in a fresh temporary venv. See
[export setup and limitations](EDGE_RUNTIME.md#25-isolated-export-environment)
for pinned versions, commands, and logs. Stage 2.6 subsequently completed actual
conversion and program metadata loading in that same environment, without
dependency changes. Stage 2.7 then generated real text using the wheel's native
TextLLMRunner, registering its quantized/custom-op libraries explicitly.
No packages or source builds were added; both original environment inventories
still match their baseline. Quality, device execution, and offline operation
remain unverified. See the [generation evidence](EDGE_RUNTIME.md#27-real-desktop-generation).

## Isolated desktop GPU vision environment (Stage 3.1)

The new environment is `/root/.cache/forgesight/vision-v1/venv`. It contains
PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, Anomalib 2.6.2, NumPy 2.2.6,
timm 1.0.31 and Lightning 2.6.6. `requirements/pylock.vision.toml` pins 89
dependencies, excluding bundled pip. CPU/image/shared GPU/ExecuTorch environments
were not upgraded or used as an installation target. This CUDA x86_64 lock
must not be installed as the Jetson ARM64 software stack.

Dependency checking, actual CPU/CUDA architecture probes, and a fresh temporary
lock-only installation with real CUDA forward checks passed. The temporary
environment was removed automatically. Upstream timm/dinomaly deprecation
warnings are retained; they did not prevent the selected models' execution.
See [vision selection and reproducible commands](VISION_BASELINE.md) for
source pins, scope, timing caveats, and pending training assets.

Stage 3.2 acquired the pinned teacher and Imagenette archives on WSL's Linux
filesystem and completed the planned 10,000-step FP32 first-baseline run using
this same environment. No packages were installed. Full before/after package
inventories match for the image, shared GPU, exporter and vision environments;
the vision environment's dependency check also passes. Final checkpoint
integrity and independent CUDA reload passed. See
[training assets, recipe and scope](VISION_TRAINING.md). The raw assets,
checkpoints, plot and logs remain ignored; no trained inference optimization,
accuracy result or Jetson software setup is established by training.

Stage 3.3 reuses the same vision environment to fit map scales and a candidate
threshold without optimizer updates or dependency installations. It creates a
new weights/metadata-only calibrated checkpoint, preserving the original
training snapshots. The actual model-only FP32 timing uses the desktop CUDA GPU
with the training recipe's math flags; it is not a Jetson or end-to-end result.
See [calibration/reload commands and limitations](VISION_CALIBRATION.md).

## Stage 7: isolated operator app and browser QA

The desktop app runs in a new Linux environment at
`/root/.cache/forgesight/app-v1/venv`. All 89 package versions from the original
vision lock are preserved; 12 additions make 101 locked packages. Inputs are
`requirements/app.in` and `requirements/vision.in`; the installed, hashed lock
is `requirements/pylock.app.toml`. FastAPI 0.162.0, Uvicorn 0.54.0 and HTTPX
0.28.1 are pinned. This is a CPython 3.12/Linux x86_64/CUDA desktop lock, not
an unverified Jetson installation recipe.

From PowerShell, choose a fresh environment root rather than overwriting an
existing one. The following illustrates the actual created environment:

```powershell
wsl -d Ubuntu --exec /root/.local/bin/uv venv --python /root/.cache/forgesight/vision-v1/venv/bin/python /root/.cache/forgesight/app-v1/venv
wsl -d Ubuntu --exec /root/.local/bin/uv pip sync --python /root/.cache/forgesight/app-v1/venv/bin/python requirements/pylock.app.toml
wsl -d Ubuntu --exec /root/.local/bin/uv pip check --python /root/.cache/forgesight/app-v1/venv/bin/python
```

The lock was compiled with uv 0.12.15 using a constraint inventory of all
existing vision distributions except pip. Its versions were checked against
the committed 89-package vision lock before installation. An initial blanket
binary-only resolution refused the pre-existing source-only FreIA dependency;
that failure is retained. Resolution then used the already reviewed identical
source-only distribution entries/hashes, without relaxing package versions.
New desktop dependencies resolved from official PyPI metadata. The normal
installation command uses the reviewed lock, not a new unconstrained resolve.

A second fresh temporary environment reproduced the complete app lock,
dependency check and FastAPI/operator/test-client imports. Actual browser
inference in the primary app environment matched the earlier same-image scores
and geometry exactly. Full before/after inventories of the image, shared GPU,
ExecuTorch export and vision environments are identical. Existing environments
were not upgraded. Evidence is under `artifacts/app_stage7/`.

Windows browser QA uses a separate environment at
`%LOCALAPPDATA%\forgesight\ui-qa-v1`, with existing Windows Python 3.10.11.
`requirements/pylock.ui_qa.toml` pins Playwright 1.64.1 and its dependencies;
Chromium is downloaded into that QA root. Browser tools are not runtime app
dependencies and no Linux browser/system packages were installed.

Illustrative PowerShell reproduction, using a new QA root if one already exists:

```powershell
$qaRoot = Join-Path $env:LOCALAPPDATA 'forgesight\ui-qa-v1'
uv venv --python 3.10 $qaRoot
uv pip sync --python "$qaRoot\Scripts\python.exe" requirements/pylock.ui_qa.toml
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $qaRoot 'browsers'
& "$qaRoot\Scripts\python.exe" -m playwright install chromium
# Against a running real loopback server with a fresh empty proof database:
& "$qaRoot\Scripts\python.exe" -B scripts/verify_operator_browser.py --output artifacts/app_stage7/browser_first
```

The full current 119-test suite uses the app environment. The browser script
requires its specific proof IDs and a new output directory; restart/layout
checks can use its explicit restart/resume modes. See
[the operator app, launch command and proof limits](OPERATOR_APP.md).

## Stage 6: SQLite without another dependency

Storage uses Python's built-in sqlite3. The existing project `.venv` reports
Python 3.12.14 and SQLite 3.53.1; it can run history, retry and restart-verification
commands without Torch or NumPy. The real inference replay and complete tests
use the unchanged locked vision environment. No packages were installed or
upgraded. The database is local to the project artifacts directory on this
WSL host; actual Jetson filesystem/runtime checks remain unverified. See
[storage setup, schema and commands](STORAGE.md).

## Stage 9.1: reuse existing isolated environments

The controller runs in the app environment; the actual Llama child runs in the
existing ExecuTorch 1.5.1+cpu environment with PyTorch 2.14.0+cpu. No packages
were installed or upgraded. Full before/after inventories match for all five
Linux environments (image, shared GPU, exporter, vision and app). The vision
model, score policy and original inspection database hashes remain unchanged.
The native smoke worker now supports an optional bounded system message;
default historical chat formatting is unchanged and regression-tested. The
current child remains x86_64-only: this is not a Jetson installation recipe.
See [the advisory component and recorded limitations](FORGEASSIST.md).

## Stage 9.2: contract checks without additional dependencies

The 132-test suite passed in the existing app environment. A separate 27-case
contract-only evaluation passed in the minimal image `.venv`, without importing
Torch/Llama or requesting a native worker. Actual Llama probes reused the same
isolated desktop ExecuTorch environment. All five before/after package
inventories match and no installation occurred. The initial inventory attempt
found that the app environment has no pip module; read-only inventory used
standard-library `importlib.metadata` instead, with the issue retained locally.
This is PC portability evidence only; do not copy x86_64 environments to Jetson.
See `artifacts/assist_stage9_2/` and the committed refusal-verification manifest.

## First authenticated Jetson inventory (2026-11-11)

The target is ARM64/glibc 2.31, Ubuntu 20.04.6, L4T 35.6.1, CUDA 11.4.315,
cuDNN 8.6.0.166 and system Python 3.8.10. Metadata shows NumPy 1.17.4 and Pillow
7.0.0; OpenCV's module is discoverable, while Torch/ExecuTorch are not. No ML
runtime was imported and no tensor/CUDA inference was attempted. No package
or model was installed. This is inventory, not a new tested app environment.

The standalone standard-library audit script ran on the actual Python 3.8
interpreter. Its source also passes Python 3.8 syntax parsing on the PC. The
last full app-suite result remains the Stage 9.2 132-test PC result; that suite
was not rerun for this standalone audit and has not run on Jetson.

The pinned ExecuTorch 1.5.1 package requires Python >=3.10/<3.15; its native
source requires CMake >=3.24, while the Jetson has 3.16.3. PyPI metadata lists
an ARM64 CPython 3.12 manylinux_2_28 wheel, but a filename does not establish
all dependency/kernel or target compatibility. No wheel was installed or run.
Next audit ARM package contents/dependencies and native build alternatives,
then choose a separate reproducible runtime path. Do not replace system Python,
mix the x86/CUDA 12.4 PC stack into CUDA 11.4, or infer that an OS reflash is
necessary from these initial version differences.

## Substep 2.8: target resolution and static native-library checks

The selected recipe uses isolated ARM64 CPython 3.12.14, Torch 2.14.0+cpu,
ExecuTorch 1.5.1+cpu, pytorch-tokenizers 1.6.0 and TorchAO 0.18.0. Both pylock
and pip-compatible hash locks resolve 49 packages for `aarch64-manylinux_2_31`.
Runtime versions/architecture are pinned; the existing x86 environments and
system Jetson Python 3.8 are not installation targets. Prebuilt CPU wheels avoid
requiring the target's old CMake for this first candidate.

Four publisher-hashed ARM wheels were downloaded to ignored PC artifacts;
`readelf` inspected the selected core wheels without executing them. Their
33 ELF files match exact GNU version definitions from ten actual Jetson system
libraries. Other dependency/interpreter runtime behavior remains unverified.
The official standalone interpreter archive has pinned metadata/hash, but its
download failed, so it has not been inspected, extracted or executed.

The source-only antlr4 dependency successfully built into a pure-Python wheel in
a fresh PC-only cache venv, using the committed build-tool hash lock. No existing
environment was modified. The 140-test suite passes in the unchanged app env,
including eight static-guard tests. No model/holdout inference was requested.
Full target import/dependency checks and real Llama execution remain future work.
