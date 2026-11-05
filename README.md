# ForgeSight AI

An incremental project for offline industrial surface-defect inspection on an
NVIDIA Jetson Orin Nano, with deterministic grading, local inspection history,
and a Llama-family advisory model running on the device through ExecuTorch.

**Current state:** Stage 1 data/input preparation is verified on the PC, including
a frozen split, role-based image/mask loader, and inspected development previews. There is
no complete inspection application or verified Jetson runtime yet.
The user reports that the Jetson is currently unavailable.
Stage 2.7 generated real text from the exported Llama 3.2 1B model through
ExecuTorch's native C++ runner, called from Python on desktop WSL. The original
PTE/tokenizer fingerprints were verified. Actual Llama/ExecuTorch Jetson
inference remains blocked by device unavailability. See
[the runtime evidence and candidate](docs/EDGE_RUNTIME.md).
The source-pinned export recipe is preparation only; it reports its missing
inputs and does not download weights, install packages, or create a model.
The device capture harness still refuses this PC as a Jetson target. All 86
project tests pass. Desktop text is real; advice quality, defect detection,
Jetson performance, and disconnected operation remain unverified.

Stage 3.1 selected EfficientAD-S at 256x256 for first training, balancing fixed
runtime cost, memory and potential lower precision. Actual untrained architecture
probes compared it with medium/higher-resolution variants and compact/heavier
PatchCore candidates. These are not accuracy or Jetson benchmarks. See
[the trade-offs, measurements and training prerequisites](docs/VISION_BASELINE.md).

Stage 3.2 completed real EfficientAD-S training: 10,000 FP32 updates on the 176
normal fitting images, with the verified teacher and generic-image regularization.
The final checkpoint passes independent reload/state/output checks. Its original
training snapshot remains uncalibrated. No
validation or holdout pixels were used in training. See the
[training evidence, learning curve, and commands](docs/VISION_TRAINING.md).

Stage 3.3 fitted a separate calibrated model/policy bundle using 22 map-fitting
and 22 threshold-setting normals. The provisional image-score threshold is
0.2277658; this yields candidate flags, not quality grades or probabilities.
Three map-fitting normals were flagged; zero threshold-setting normals were
flagged by construction. Independent accuracy and recall remain unevaluated.
The trained calibrated PC model measured 14.54/14.80 ms p50/p95 for cached-input
FP32 forward only. See [calibration evidence and limits](docs/VISION_CALIBRATION.md).

Stage 3.4 now supplies real held-out evidence: 87/89 defects detected, two
missed scratched nuts, and two false alarms among 22 good images. Threshold and
model remain frozen. Types remain unknown. See [evaluation](docs/VISION_EVALUATION.md).
Earlier stage paragraphs above describe historical snapshots, superseded by
this evaluation for benchmark detection only. Jetson and camera proof remain open.

Stage 4 adds [real inspection records and a one-part gate](docs/INSPECTION.md).
The saved-image replay turns 9 frames into 4 records, with 5 repeated frames
suppressed. All 94 tests pass in the existing locked vision environment.
Use that environment for the full suite, which now includes NumPy/OpenCV tests.

## Start here

1. [Step-by-step roadmap](FORGESIGHT_STEP_BY_STEP_ROADMAP.md): authoritative
   stage order and acceptance gates.
2. [Engineering prompt](FORGESIGHT_CODEX_MASTER_PROMPT_V3.md): teaching and
   implementation rules.
3. [Current status](docs/STATUS.md): completed evidence and the next small task.

The original `Edge AI Quality Control Model Pipeline.pdf` remains in the local
workspace as a technical reference and is excluded from Git. Its performance
figures and architectural assumptions are not verified project results.

## Evidence and decisions

- [Hardware audit](docs/HARDWARE.md)
- [Observed development environment](docs/ENVIRONMENT.md)
- [Dataset source and split policy](docs/DATASET.md)
- [Architecture decisions](docs/DECISIONS.md)
- [Requirement evidence](docs/COMPLIANCE.md)
- [Learning notes](docs/LEARNING_LOG.md)

The benchmark domain is MVTec AD `metal_nut`. Its real archive and extracted
335 images/93 masks have passed structural validation; the license, source
fingerprint, counts, and split policy are in `docs/DATASET.md`. A frozen manifest
assigns 176 fit, 44 normal-validation, four development, and 111 holdout samples.
A verified loader reads fitting, validation, and development roles. A local
contact sheet shows supplied ground-truth annotations; no model predictions
exist yet. Final evaluation uses a disclosed custom holdout that excludes
development previews. Controlled physical samples and actual Jetson camera
validation are separate later evidence.

Repeat the category archive audit without extracting or displaying images:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B scripts/audit_metal_nut_archive.py data/downloads/metal_nut.zip
```

Verify the frozen split against the current dataset:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.data_manifest --check
```

Load and verify the four development image/mask pairs:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.data_loader --role development_preview
```

Generate the development-only contact sheet (output directory must be new):

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.data_preview
```

Open `artifacts/data/development_preview_v1/contact_sheet.png`. Its five rows
contain one normal fitting image plus bent, color, flip, and scratch development
samples. The red overlays are dataset annotations, not detector predictions.
The source, license, manifest fingerprint, and pixel-area metadata accompany
the image in `preview.json`. No holdout images are displayed.

Run the 86 current data-integrity, edge, training/calibration-boundary and measurement tests:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m unittest discover -s tests -v
```

Read the Stage 2 [candidate, blockers, recipe, and capture instructions](docs/EDGE_RUNTIME.md)
before installing export dependencies or attempting a device run.

Verify the downloaded source model against the publisher-checked reference:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.edge_assets
```

The verified source weights are not an exported `.pte` or a successful runtime
test. A separate CPU export environment now passes real exporter startup and
configuration parsing. Its [locked setup and checks](docs/EDGE_RUNTIME.md#25-isolated-export-environment)
are verified on this WSL host. Conversion and desktop generation now pass;
quality and Jetson execution remain open. The
[export record and verification command](docs/EDGE_RUNTIME.md#26-actual-model-export)
keep these distinct.

Run the actual desktop generation smoke test with a new evidence directory:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/et1.5.1/venv/bin/python -B -m forgesight.edge_pc_smoke --output artifacts/edge/pc_generation_recheck
```

This uses local model/tokenizer files, the native ExecuTorch runner, and explicit
registration of wheel-provided quantized/custom kernels. It does not certify
offline or Jetson operation. See [the actual transcript and limits](docs/EDGE_RUNTIME.md#27-real-desktop-generation).

## Development

Use Ubuntu WSL with CPython 3.12. The project `.venv` contains Pillow 13.0.0,
pinned by version and wheel hash in [requirements.txt](requirements.txt).
It is isolated from the existing GPU-accessible PyTorch environment.

From PowerShell in this project directory, after following the
[environment setup](docs/ENVIRONMENT.md#project-environment-stage-03):

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B scripts/check_environment.py
wsl -d Ubuntu --exec .venv/bin/python -B -m pip check
```

The smoke check verifies environment isolation and exact PNG pixel preservation.
It does not load a defect dataset, run a model, or test Jetson. Installation and
the check were also reproduced in a second fresh temporary environment.

Datasets, model binaries, secrets, runtime databases, and generated outputs stay
outside Git. A local commit records a snapshot on this PC; pushing to GitHub
requires a separate authorized operation.
