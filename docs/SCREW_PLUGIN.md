# Screw plugin

Approved order: EfficientAD training/export, classifier/policies/guidance, then UI hot-swap.
Current scope: Stage 1 calibration/export/evaluation; no UI edits or Llama retraining.

- Model: existing pinned EfficientAD-S, 256px RGB, FP32, 10,000 steps, seed 42.
- Data: `data/mvtec_ad/screw`; frozen hashes/splits in `manifests/screw_v1.json`.
- Train normals: 80% fit, 20% reserved for map calibration. Test normals: half
  threshold calibration, half final evaluation. Defects: 60/20/remainder for
  classifier fit/validation/final evaluation. Actual folder labels are preserved.
- Reuse existing isolated WSL vision runtime, teacher and Imagenette regularization.
- Native grayscale images retain source hashes/mode; training replicates them to RGB.
- Training evidence: `artifacts/vision/screw_train_v1/`; raw data/weights stay ignored.
- TorchScript candidate and score policy: `plugins/screw/`; weights are local and
  Git-ignored. `vision.json` pins artifact hashes and disables station promotion.

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
Checks: 49 focused data/training/calibration/evaluation tests passed; code review found no
significant issues. Metal-nut assets, UI and Llama runtime remain unchanged.

Calibration/export/evaluation completed on PC. **Gate 1 FAILED; Stage 1 is not
accepted. Stage 2 and UI integration remain blocked.** Threshold: 0.19198325276374817,
strict `score > threshold`; frozen before testing. Export matched the calibrated
model on 24 normal images with zero map/score error; no holdout images were used
for calibration or parity checks. The inherited calibration source-policy status
describes its input state; packaged policy status explicitly records calibration.

| Holdout metric | Measured | Required | Result |
| --- | --- | --- | --- |
| Accuracy | 84% (42/50) | Report only | — |
| Image AUROC | 0.940887 | >= 0.95 | FAIL |
| Defect recall | 75.86% (22/29) | >= 95% | FAIL |
| Normal false alarms | 4.76% (1/21) | <= 10% | PASS |

Misses: 2 manipulated_front, 3 scratch_head, 2 thread_side. scratch_neck 5/5 and
thread_top 6/6 were flagged; these are ground-truth groups, not type predictions.
Pixel AUROC: 0.974787 on the 256px grid with nearest-neighbor resized masks;
no segmentation threshold or physical-camera accuracy is verified.

Evidence: `manifests/screw_efficientad_s_{calibration,evaluation}_v1.json` and
`manifests/screw_torchscript_export_v1.json`. The 50 predictions/maps are preserved
under `artifacts/vision/screw_evaluation_v1/`. Metrics were reproduced from cached
outputs without inference. The exposure ledger prevents another final test run.

Reproduce metrics using the existing isolated WSL vision Python:
```text
python -B -m forgesight.screw_evaluate --summarize-only
```

Next: review a separate model-improvement experiment and obtain a new untouched
acceptance set. Do not retune the frozen threshold using this exposed holdout.
Jetson validation, classifier, grading policy and guidance remain unverified.
