# Screw plugin

Approved order: EfficientAD training/export, classifier/policies/guidance, then UI hot-swap.
Current scope: Stage 1 training only; no UI edits or Llama retraining.

- Model: existing pinned EfficientAD-S, 256px RGB, FP32, 10,000 steps, seed 42.
- Data: `data/mvtec_ad/screw`; frozen hashes/splits in `manifests/screw_v1.json`.
- Train normals: 80% fit, 20% reserved for map calibration. Test normals: half
  threshold calibration, half final evaluation. Defects: 60/20/remainder for
  classifier fit/validation/final evaluation. Actual folder labels are preserved.
- Reuse existing isolated WSL vision runtime, teacher and Imagenette regularization.
- Native grayscale images retain source hashes/mode; training replicates them to RGB.
- Training evidence: `artifacts/vision/screw_train_v1/`; raw data/weights stay ignored.
- Training completion does not establish accuracy; calibration, TorchScript export,
  untouched evaluation and separate Jetson proof remain subsequent gates.

Audit: 320 train normals, 41 test normals, 119 defects/masks; all 1024x1024.
Frozen roles: 256 fit, 64 map normals, 20 threshold normals, 69 classifier fit,
21 classifier validation, 50 final evaluation (21 normal + 29 defective).
Source archive provenance and physical-camera accuracy remain unverified.
Dataset license notice: CC BY-NC-SA 4.0; retained under the ignored data directory.

Training command (existing WSL vision Python):
```text
python -B -m forgesight.vision_train --profile configs/screw_efficientad_s_v1.json --recipe configs/screw_training_v1.json --root data/mvtec_ad/screw --output artifacts/vision/screw_train_v1
```

Status: **training completed and checkpoint reload verified** (10,000 FP32 steps,
1,163.54 seconds on RTX 2000 Ada). Peak PyTorch training allocation: 542,685,184
bytes; this is not Jetson memory usage. Reloaded score difference: 0.0.
Evidence: `manifests/screw_efficientad_s_training_v1.json` (checkpoint and input hashes).
Checks: 32 focused data/training tests passed; independent code review found no
significant issues. Metal-nut assets, UI and Llama runtime remain unchanged.

Next: calibrate maps using reserved train normals and threshold using the 20
reserved test normals, export `plugins/screw/efficientad_s.ts`, then evaluate once
on the untouched holdout. Gate 1 targets: AUROC >= 0.95, recall >= 95%, false-alarm
rate <= 10%. No accuracy, classifier or Jetson deployment claim yet.
