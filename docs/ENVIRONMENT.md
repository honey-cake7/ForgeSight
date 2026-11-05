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
