# Stage 3.2: first EfficientAD-S training run

This step trains the selected small/256 model on the frozen normal fitting
partition. It does not select a pass/reject threshold, measure defect accuracy,
read final-holdout images, export an optimized engine, or verify Jetson execution.
The planned run completed successfully and its final checkpoint passed an
independent reload. The result remains explicitly uncalibrated and unevaluated.
The later [Stage 3.3 calibration record](VISION_CALIBRATION.md) describes a
separate calibrated checkpoint/policy; the original training snapshot stays
unchanged and no accuracy result is implied by calibration.

## Sources and verified local assets

The archive fingerprints are taken from Anomalib's
[pinned training implementation](https://github.com/open-edge-platform/anomalib/blob/cc5f400a4a4b1b14b5a3ee5078063c250a8f522e/src/anomalib/models/image/efficient_ad/lightning_model.py).
The [publisher release](https://github.com/open-edge-platform/anomalib/releases/tag/efficientad_pretrained_weights)
provides the teacher weights. [Fastai's Imagenette repository](https://github.com/fastai/imagenette)
links the original full-size subset of ImageNet. Repository code licenses are
Apache-2.0; this record does not claim to relicense underlying photographs.
These assets support the local noncommercial research prototype and remain
outside Git. Raw data and binary checkpoints are not redistributed here.

| Asset | Downloaded bytes | Verified SHA-256 |
| --- | --- | --- |
| Teacher ZIP | 39,960,466 | `c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a` |
| Full Imagenette TAR.GZ | 1,557,161,267 | `6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd` |
| Extracted small teacher | 10,779,695 | `a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d` |

The ZIP contains two teacher files; the Imagenette archive contains 13,395
files. The training loop selects only the **9,469 Imagenette train images**.
Their generic class labels are discarded; they are regularization examples,
not metal-nut defect labels or replacements for the domain fitting images.

`forgesight.vision_assets` verifies archive hashes before bounded extraction,
rejecting traversal, duplicate destinations, links, special/encrypted files,
and excessive file/count/total sizes. New directories are required. Partial or
failed downloads remain evidence, not verified training inputs. Local assets
live under `/root/.cache/forgesight/vision-v1/training_assets/` on Linux storage.
`manifests/vision_training_assets.json` records source, integrity and review
metadata. It does not contain model weights or photographs.

The downloaded teacher loads with `weights_only=True` and strict state keys.
An actual forward on `train/good/000.png` produced finite features of shape
1x384x56x56. That probe took zero optimizer steps and read zero holdout images.

## Frozen data access and preprocessing

`normal_rows` permits fit and normal-validation metadata only. It rejects test
origin, abnormal labels/masks, inconsistent counts, and other roles. The existing
manifest sidecar and profile SHA-256 must match. Source images still pass their
byte, pixel, size and mode verification through `load_verified_image`.

The 176 normal fit images are resized as full RGB frames to 256x256, converted
to float [0,1], then cached in host memory. No center crop removes part edges.
The upstream networks perform their own ImageNet normalization. Validation
metadata confirms 44 disjoint normal samples, but no validation pixels enter
this training phase. Development and final-holdout pixels remain unread.

Generic Imagenette augmentation follows the pinned recipe: resize to 512x512,
random grayscale with probability 0.3, center crop to 256x256 and tensor
conversion. This crop applies only to generic regularization images. A seeded
four-worker loader shuffles the auxiliary train split. Fit images use seeded
shuffled passes through all 176 samples, with batch size one.

## Training recipe and deliberate differences

`configs/efficientad_s_training.json` fixes this **10,000-step first baseline**,
not a full paper reproduction or a convergence claim. It uses FP32, Adam with
learning rate 1e-4 and weight decay 1e-5; StepLR multiplies learning rate by 0.1
at step 9,500. The budget was chosen before inspecting held-out outcomes.
Training steps do not change the deployed graph's size or precision. Quality
and latency remain separate questions.

The loop calls the unchanged pinned EfficientAdModel loss implementation:
student-teacher hard mismatch plus generic-image penalty, autoencoder loss,
and student-autoencoder consistency. Teacher weights and teacher-statistic/map
parameters are excluded from optimization. Teacher channel mean/std use only
the 176 fit images, with float64 accumulation and a declared 1e-6 numerical
floor; this run required zero clamped channels.

The custom loop is used to enforce frozen roles and avoid a framework default
dataset/validation routine consuming the original test split. It uses only the
Imagenette training subset. Random/Python/NumPy/Torch/loader seeds are recorded.
TF32 and cuDNN autotuning are off, cuDNN deterministic mode is on. Bilinear
gradient operations and platform differences mean bitwise retraining identity
is not claimed. Recipe/data/source hashes make the actual run auditable.

Each optimizer step checks finite loss and gradients. Logs record the three
losses, total loss, recent-window mean, learning rate and elapsed time every
100 steps. New checkpoints are saved at 1,000, 5,000 and 10,000 steps. Output
directories are never overwritten. Intermediate states stay available even if
a later step fails. Training loss is not accuracy or a severity measure.

## Checkpoint proof and scope

The actual 1,000-step checkpoint was reloaded independently using a verified
hash, `weights_only=True`, metadata guards and strict state keys. It produced a
finite raw map on a previously used fit image. Teacher weights were unchanged;
both student and autoencoder weights had changed. The intermediate test is
stored in `artifacts/vision/stage3_2/intermediate_reload.json`.

Final proof requires completion of the planned budget, finite final output,
unchanged teacher / updated trainable networks, saved checkpoint fingerprint,
and independent final reload. A seen normal-image smoke check cannot establish
unseen defect detection. Calibration parameters and decision threshold remain
unset; no PASS/REJECT predictions or defect-type labels are justified yet.

| Actual completed-run evidence | Result |
| --- | --- |
| Optimizer steps | 10,000, full configured first-baseline budget |
| Optimization loop plus checkpoints | 1,158.35 seconds (about 19 minutes 18 seconds; excludes asset/setup time) |
| GPU | Desktop NVIDIA RTX 2000 Ada Generation |
| Peak PyTorch training allocation | 542,685,184 bytes (about 517.5 MiB; not total VRAM or inference allocation) |
| Recent mean total loss near step 100 / final 100 steps | 16.3052 / 2.2713 |
| Final checkpoint bytes | 75,199,220, including optimizer state; not an exported inference engine |
| Final checkpoint SHA-256 | `85106f4a9ef6ec3e9c8acb4056dc2db2124e54284209e10d686dd36dea66ca0d` |
| Teacher / student / autoencoder state | Teacher unchanged; both trainable networks updated |
| Independent final reload | Strict state/hash checks passed; fit-anchor raw-score difference 0.0 |
| Reload map shape | 1x1x256x256, finite; previously used normal fit image only |
| Validation / development / final-holdout pixel reads in training | 0 / 0 / 0 |
| Tests / dependencies | All 78 tests passed; no broken package requirements |
| Map calibration / decision threshold / accuracy | NOT RUN / NOT SET / NOT EVALUATED |

The recent means use the logged windows; the first optimizer-step total was
21.7769, so a single step and a window average must not be confused. Loss
reduction and changed network weights establish optimization, not convergence,
generalization, defect recall or false-alarm rates. The budget is a first
baseline, not a full paper-protocol or bitwise retraining reproduction.

The actual learning curve is in
`artifacts/vision/efficientad_s_train_v1/training_loss.png`. It was generated
from the 101 real logged observations and visually reviewed; it is not an
accuracy curve. The completed record is `manifests/efficientad_s_training_v1.json`.
The image/teacher smoke checks and reloads each additionally read the already
used fit anchor; none opened validation, development or holdout images.

## Commands and local evidence

From PowerShell in the project directory, asset acquisition used:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.vision_assets --root /root/.cache/forgesight/vision-v1/training_assets --output artifacts/vision/stage3_2/assets.json
```

Those paths now exist. Re-acquisition requires fresh asset/evidence paths;
preserve the existing verified archives. Training used:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.vision_train --output artifacts/vision/efficientad_s_train_v1
```

Choose a new output directory for any deliberate rerun. Training does not
download assets automatically or install packages. `run_config.json` captures
the exact input profile and recipe before optimization. Checkpoints include
tensor state, optimizer/scheduler state and plain metadata. There is no automatic
resume promise; restart/resume design needs review before using saved optimizer
state as a new experiment. GPU allocation statistics do not represent whole
device RAM, driver overhead or eventual vision-plus-Llama resource use.

The final checkpoint was independently checked with:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.vision_checkpoint --training-report artifacts/vision/efficientad_s_train_v1/training_report.json --output artifacts/vision/stage3_2/final_reload.json --device cuda
```

Use a new evidence filename on repeats. The loader verifies the recorded
fingerprint before tensor-only loading, requires matching model metadata and
Anomalib version, uses strict state keys and disables gradients for inference.
The verifier compares all three network-state digests to the trained model and
checks a finite raw fit-anchor output against the saved result. This is a
checkpoint integrity/reload test, not an unseen-image evaluation.

Logs are under `artifacts/vision/stage3_2/`; the run directory holds
`losses.jsonl`, `progress.json`, checkpoints and the final training report.
The binaries, logs and plots remain ignored. The eventual committed training
manifest records measured results and provenance only. Full package inventories
for all four existing environments matched their before snapshots; no packages
were installed during this step. Intermediate 1,000/5,000-step checkpoints are
preserved alongside the final one.

**Next, Stage 3.3:** calibrate the two map branches with the 44 normal validation
images, then choose/document a normal-only threshold policy. Keep final holdout
closed until the model and policy are frozen for the separate final evaluation.
Actual camera, subtype labels, deterministic severity, optimized latency and
mandatory on-device Llama through ExecuTorch remain their own open gates.
