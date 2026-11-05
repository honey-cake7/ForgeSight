# ForgeSight AI

An incremental project for offline industrial surface-defect inspection on an
NVIDIA Jetson Orin Nano, with deterministic grading, local inspection history,
and a Llama-family advisory model running on the device through ExecuTorch.

**Current state: Stage 5 desktop saved-image prototype verified.** The system
runs a real trained EfficientAD-S model, highlights response regions, saves one
record per declared part and adds deterministic pixel-only decisions. It is
not yet a complete application or a verified Jetson/camera system.

- [Vision training and choice](docs/VISION_BASELINE.md): EfficientAD-S 256x256,
  10,000 FP32 training steps on 176 normal images; [training evidence](docs/VISION_TRAINING.md).
- [Frozen calibration](docs/VISION_CALIBRATION.md): threshold 0.2277658; cached
  desktop forward-only p50/p95 14.54/14.80 ms, excluding image I/O and app work.
- [Held-out evaluation](docs/VISION_EVALUATION.md): flagged 87/89 defects,
  missed two scratches, false alarms on 2/22 good images. No test-driven retuning.
- [Part gate and records](docs/INSPECTION.md): nine replay frames -> four declared
  part records, five repeats suppressed, including an explicit input error.
- [ForgeGrade prototype](docs/FORGEGRADE.md): real replay gives PASS, REVIEW,
  PASS, REVIEW. Types remain unknown. REJECT size/location rules pass synthetic
  policy tests; automatic type-based rejection is not yet supported by the model.
  The two missed scratches still receive provisional PASS in a cache-only rule
  diagnostic; production use is not established. No millimeters or certification.
- [Llama / ExecuTorch evidence](docs/EDGE_RUNTIME.md): genuine Llama 3.2 1B text
  generation through the native ExecuTorch runner on desktop WSL. Actual Jetson
  inference and integration as inspection advice remain open.

All **101 tests pass** in the existing locked vision environment. No packages
were installed/upgraded in Stages 3.4?5. The Jetson remains unavailable; exact
hardware, camera performance and disconnected edge operation are unverified.
**Next stage: Stage 6 SQLite history and restart/duplicate checks**, after review.

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
contact sheet shows supplied ground-truth annotations; separate model overlays
now exist in local evaluation/inspection outputs. Evaluation uses a custom holdout that excludes
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

Run all 101 data-integrity, edge, vision-boundary, inspection and grading tests:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m unittest discover -s tests -v
```

Read the Stage 2 [candidate, blockers, recipe, and capture instructions](docs/EDGE_RUNTIME.md)
before installing export dependencies or attempting a device run.

Verify the downloaded source model against the publisher-checked reference:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.edge_assets
```

Source-asset verification alone is not export or inference. A separate CPU
export environment passed exporter startup and configuration parsing; later
steps completed real export and desktop execution. Its [locked setup and checks](docs/EDGE_RUNTIME.md#25-isolated-export-environment)
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
outside Git. Verified steps are committed and pushed under the user's standing authorization; recorded manifests contain permitted metadata only.
