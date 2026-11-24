# Screw plugin

Approved order: EfficientAD training/export, classifier/policies/guidance, then UI hot-swap.
Current scope: the selected 30k detector and an optional dual-recipe station
demonstration. The selector is enabled only with `--plugins-root`. This
demonstration does not pass the detector or subtype reliability gates.
The existing single-part station remains the default.

## Optional station demonstration assets

Model binaries are excluded from Git. Before launching with `--plugins-root
plugins`, place the frozen metal-nut TorchScript export at
`plugins/metal_nut/efficientad_s.ts` and the selected 30k screw TorchScript
export at `plugins/screw/efficientad_s.ts`. Obtain both from existing local
training/export artifacts or a trusted project transfer. Check their SHA-256
values against each `recipe.json` model entry before launch. The recipe
registry checks every component at startup and names a missing asset. Keep
model files and the licensed MVTec datasets outside Git. The configured
baseline `--model` must have the same pinned metal-nut bytes.

For optional screw replay, stage the original MVTec screw dataset under
`data/mvtec_ad/screw`. Camera capture and arbitrary uploads remain
diagnostic because physical-part acceptance has not passed. The selector
changes the active recipe for new inspections. Saved records retain their
original evidence and plugin ID.

## Selected 30k development candidate

The operator requested development continuation with the measured recall limitation.
This does not pass or lower the original 95% recall target. Fresh acceptance and
physical-camera accuracy remain unverified.

- Training: 30,000 FP32 steps; final checkpoint reload verified on CPU.
- CPU TorchScript: `plugins/screw/efficientad_s.ts`, SHA-256
  `494726eeececa7b6f194ea5ec6b471873aa41085913eab9b65ef64120519618e`.
- CPU threshold: `0.2372681200504303`, strict `score > threshold`, from reserved
  normals only. The GPU diagnostic threshold is retained separately. A CPU/GPU
  rounding difference at a threshold-normal tie required CPU-native calibration;
  CPU source/export parity then passed on 24 reserved normals with zero map/score error.
- Reused-image CPU diagnostic: 47/50 correct, 26/29 defects flagged, 0/21 false
  alarms, AUROC 0.98358. This is descriptive evidence, not a fresh acceptance test.
- Screw subtype tree: five actual dataset labels; 69 classifier-fit and 21
  classifier-validation images. No annotation masks supply features. Selection
  compares 27 small trees on validation only, prioritizing support for all types.
  The selected depth-3 tree makes 16 correct type calls out of 19 validation calls;
  three wrong calls and two detector misses remain. Validation also selects leaf
  support, so these numbers are not unbiased final-test accuracy.
- Final CPU bundle diagnostic: the classifier correctly labels 19/25 type calls
  across the 29 exposed defective images (six wrong calls, one unknown, three
  detector misses). Every native class has at least one correct call, but
  `thread_side` remains weak. Stage 2 is a tested development prototype; its
  reliability gate is not passed. The opt-in UI demonstration carries this
  limitation and must not be presented as production inspection.
- `grade_policy.json` keeps physical images and anomalies at REVIEW. Benchmark
  normals outside a 10% numerical review margin can receive provisional PASS;
  no automatic physical-part release or manufacturing tolerance is authorized.
- `guidance.json` contains pinned A/B manual inspection checks. All six original
  label/unknown prompts produced valid choices on one physical Jetson ExecuTorch
  worker. That report is preserved as historical evidence; it does not verify
  the revised hybrid guidance. A fresh device prompt check of the revised text
  matched 5/6 screw choices; `scratch_neck` required the pinned checklist fallback.
  Across both parts, 12/15 replies matched and all 15 passed PC output guards.
  The existing Llama worker stayed loaded; the installed station was not upgraded.
  See `manifests/jetson_hybrid_guidance_842c793_v1.json`. Full PR deployment and
  site QC approval remain pending. Llama weights have not been retrained.
- Reviews added detector/threshold binding before benchmark PASS, geometry-cutoff
  consistency before subtype assessment, and float32 tree traversal matching
  sklearn. All 90 cached classifier rows and 50 diagnostic predictions retained
  their results after these guards. JSON hashes use LF bytes for Windows/Jetson portability.

Run the classifier with the existing isolated WSL vision Python and a fresh output:
```text
python -B -m forgesight.screw_classifier --output artifacts/vision/screw_classifier_new_run
```
Model binaries stay ignored. `vision.json` pins the selected weights and policies;
production readiness remains false while classification reliability and device
vision are unverified. All 84 focused bundle/pipeline/metal-nut/dashboard tests passed.
Existing v1 manifests below remain historical evidence.

## Original 10k experiment

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

## 30k experiment preparation

Recipe: `configs/screw_training_30k_v2.json`; review protocol:
`configs/screw_30k_review_v2.json`. The running trainer and its inputs are unchanged.
Review outputs use `artifacts/vision/screw_30k_v2/`, separate from training and v1.

Using the existing WSL vision Python:
```text
python -B -m forgesight.screw_run_review --check
python -B -m forgesight.screw_run_review --commands
```
The first command reads progress and checks training input hashes without inference.
The second refuses incomplete training; after all 30k steps, it prints four commands
to execute sequentially: CPU checkpoint reload, normal-only calibration, TorchScript
export, then 24-image calibration parity. Each must pass before the next is run.
Existing output files are preserved; nothing promotes the candidate automatically.
Preparation verification: 56 focused tests passed. Readiness requires matching,
nonempty start/final training-input hashes; missing or malformed evidence blocks it.

Checkpoint comparison protocol: compare 10k/20k/30k using image AUROC on independent
development normals and defects; break ties using the earlier checkpoint. Training
loss is not detection accuracy. Independent development normals are currently missing;
classifier-development defects alone cannot measure false alarms or binary AUROC.
No new dataset roles were assigned and no comparison was run. Fresh independent
acceptance data must be audited separately; exposed v1 test images cannot provide it.
