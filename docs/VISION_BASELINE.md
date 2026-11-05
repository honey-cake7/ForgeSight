# Stage 3.1: choose a balanced vision baseline

The user wants good detection accuracy with lower inference time and has no
fixed per-image latency or production-rate target. Select **EfficientAD-S,
256x256 RGB, batch 1** for the first training run. This is a starting baseline,
not a trained detector, proven accuracy winner, or final Jetson configuration.
Keep **PatchCore with ResNet-18** as the fallback/comparison candidate.

This page records the Stage 3.1 selection and untrained architecture probes.
The subsequent [Stage 3.2 asset verification and training record](VISION_TRAINING.md)
tracks progress toward an actual trained checkpoint; it does not turn these
earlier synthetic timings into trained accuracy or Jetson evidence.

Stage 2.7 has real desktop Llama generation through ExecuTorch. Actual Jetson
execution remains blocked; the roadmap permits independent PC vision work.
On-device Llama through ExecuTorch remains mandatory and will run separately
from vision. Neither this selection nor the probes close the device gate.

## How the options differ

| Option | Benefits | Costs and uncertainty | Decision |
| --- | --- | --- | --- |
| EfficientAD-S | Fixed teacher/student/autoencoder graphs, no feature-bank search; potential static-shape GPU deployment | Requires pretrained teacher, optimization training, normal-feature statistics and map calibration; training assets needed; actual recall unmeasured | First training baseline |
| EfficientAD-M | More network capacity | Measured larger/slower here; no trained quality evidence justifies the added cost yet | Escalate only if validated need |
| PatchCore ResNet-18 | Fast fitting of normal feature bank; no iterative backbone training; compact backbone | Bank coverage, coreset selection and search cost trade off; bank changes can affect exported runtime; fastest small bank is not an accuracy result | Fallback/comparator |
| PatchCore Wide-ResNet-50 | Larger feature representation used in the original work | More weights, bank dimensions, temporary search memory and compute in our probe | Avoid as the edge default |
| PaDiM | Normal Gaussian statistics; no growing nearest-neighbor bank | Per-location covariance storage/inversion, feature-dimension and alignment choices; not benchmarked here | Suitable reserve option, not selected |

EfficientAD combines teacher/student mismatch and an autoencoder branch. These
two maps are not two independent known/unknown defect models and do not satisfy
ForgeDual. Its actual teacher here is a small **Patch Description Network**,
not EfficientNet, despite an upstream descriptive docstring. Its convolution
definitions are the source of truth. See the [paper](https://arxiv.org/abs/2303.14535)
and [pinned implementation](https://github.com/open-edge-platform/anomalib/blob/cc5f400a4a4b1b14b5a3ee5078063c250a8f522e/src/anomalib/models/image/efficient_ad/torch_model.py).

PatchCore stores representative normal patch features and searches for nearest
matches. Reducing the bank can improve speed but changes coverage; it must be
validated rather than chosen from timing alone. See the
[authors' implementation](https://github.com/amazon-science/patchcore-inspection)
and our [pinned implementation](https://github.com/open-edge-platform/anomalib/blob/cc5f400a4a4b1b14b5a3ee5078063c250a8f522e/src/anomalib/models/image/patchcore/torch_model.py).
PaDiM's statistical alternative is described by its
[authors](https://arxiv.org/abs/2011.08785). Other pretrained-backbone/flow methods
would add another training/deployment comparison without resolving the current
accuracy gap. A supervised fast detector alone would require valid defect
labels and would not establish detection of unknown anomalies.

## Actual compatibility and architecture timings

A new isolated Linux environment is at
`/root/.cache/forgesight/vision-v1/venv`. It uses CPython 3.12.14, PyTorch
2.6.0+cu124, torchvision 0.21.0+cu124, Anomalib 2.6.2, NumPy 2.2.6,
timm 1.0.31 and Lightning 2.6.6. Anomalib source is pinned to
`cc5f400a4a4b1b14b5a3ee5078063c250a8f522e`; both installed model modules match
the source SHA-256. This release supports the tested existing CUDA 12.4-era
stack; choosing a library's newest version is not itself evidence of better
model quality. No existing environment was upgraded.

The probe executes actual upstream model code with **random weights, synthetic
device-resident inputs and random PatchCore banks**. There were no fitting
steps, teacher downloads, dataset pixel reads or accuracy measurements. Its
finite score/map checks establish execution, not meaningful predictions.

Every case uses batch 1, 10 warmups and 50 measurements. Timings below are
synchronized model-call wall times on the desktop RTX 2000 Ada. TF32 and cuDNN
autotuning are disabled. They include model-native map/score calculations but
exclude image decode, resize, transfers, capture, grading, database/UI, and
concurrent Llama. Trained normalization/calibration can add operations. GPU
memory is peak PyTorch allocation, not total driver/process/unified memory.

| Architecture scenario | FP32 GPU p50 / p95 ms | Peak GPU MiB |
| --- | --- | --- |
| EfficientAD-S, 256 | 14.64 / 15.20 | 101.7 |
| EfficientAD-S, 384 | 33.41 / 34.17 | 187.2 |
| EfficientAD-M, 256 | 40.60 / 41.37 | 211.0 |
| PatchCore ResNet-18, 256, 1,000 random bank rows | 4.17 / 4.76 | 29.0 |
| PatchCore ResNet-18, 256, 18,022 random bank rows | 10.18 / 11.04 | 118.7 |
| PatchCore Wide-ResNet-50, 256, 18,022 random bank rows | 25.24 / 25.79 | 327.9 |

The 18,022-row scenario estimates 10% of 176 fitting images times 1,024 patches,
not an actual selected coreset. Its ResNet-18 bank alone is 26.4 MiB;
Wide-ResNet's is 105.6 MiB. EfficientAD-S has about 30.7 MiB of parameter storage,
independent of the number of fitting images. These storage figures exclude
activations, runtime overhead, and future advisory/classification memory.

An additional AMP-FP16 compatibility comparison gave EfficientAD-S 256 p50/p95
7.49/8.19 ms. The earlier separate run was 5.71/6.08 ms; retain both rather than
select the faster observation. AMP PatchCore-ResNet-18 with 18,022 rows measured
13.85/14.30 ms, so AMP did not automatically help that path. This probes
autocast execution, not native FP16-weight deployment or preserved accuracy.
FP32 remains the reference for training/evaluation. FP16 adoption requires
trained-score, map, threshold and false-negative checks.

CPU FP32 p50 was 88.03 ms for EfficientAD-S 256 versus 43.63 ms for the
18,022-row ResNet-18 PatchCore scenario on eight x86 threads. We therefore do
not claim EfficientAD is universally fastest. Jetson ARM timings may differ
substantially. The balanced choice favors fixed runtime complexity, no
normal-bank search, measured memory, and an executable lower-precision path;
PatchCore remains a credible alternative if training/assets/quality disappoint.

The complete labeled summaries are in `manifests/vision_architecture_probe.json`.
Raw samples, first probe source snapshot, package logs, and source comparisons
are under ignored `artifacts/vision/stage3_1/`. The AMP-only extension changed
probe selection/provenance fields, not the FP32 timing implementation. Exact
latencies will vary with clocks, load and software; no industrial FPS, parts/min,
Jetson latency, confidence interval, or measured detection accuracy is claimed.

## Accuracy and deployment conditions

- Use the frozen 176 normal fit images and 44 normal validation images. The four
  development anomalies can support sanity checks, not robust recall tuning.
  Freeze model/threshold decisions before the final 111-image holdout evaluation.
- Start with full-frame resize to 256, without center-cropping away part edges.
  A 700-to-256 resize can hide thin scratches. If development evidence requires
  more detail, train/recalibrate a 384 variant and compare its cost before
  opening the holdout. Do not change resolution based on final-holdout outcomes.
- The pinned training module requires batch 1, pretrained PDN teacher assets,
  normal teacher statistics and Imagenette regularization. Internal networks
  apply ImageNet normalization; do not normalize inputs a second time. Review
  asset provenance/terms and verify archive hashes before use. Assets have not
  been acquired in this step. Do not silently remove the regularization loss.
- Fixed-size convolutions make an ONNX/TensorRT path worth testing later, but
  neither export nor TensorRT/FP16 correctness on Jetson has been verified.
  Keep FP32 reference results and evaluate each optimization. INT8 needs a
  separate representative calibration/accuracy test; it is not a default.
- Share the eventual Jetson's memory budget with camera buffers and the
  mandatory Llama worker. Exact RAM/JetPack, GPU/CPU contention, power/thermal
  behavior, and end-to-end p50/p95 must be measured on actual hardware. Llama
  advice stays asynchronous and cannot override deterministic quality decisions.
- Anomaly score/map does not establish defect type, severity or millimeters.
  Type classification needs valid same-domain labeled evidence and an unknown
  option; severity/calibration/logging remain their own stages.

Training requirements and asset fingerprints come from the
[pinned training implementation](https://github.com/open-edge-platform/anomalib/blob/cc5f400a4a4b1b14b5a3ee5078063c250a8f522e/src/anomalib/models/image/efficient_ad/lightning_model.py).
Paper/architecture speed claims are background references, not our device proof.

## Repeat setup and probes

`requirements/vision.in` is the reviewed input; normal installation uses
`requirements/pylock.vision.toml` with 89 dependencies, exact distribution
hashes and the tested uv 0.12.15. This Linux x86_64 / CPython 3.12 / CUDA 12.4
lock is not a Jetson installation recipe. Python, drivers, OS/compiler and
source-build tooling remain outside the lock. uv's pylock support is experimental.

For a fresh environment, from Ubuntu WSL in the project directory, first choose
a new root instead of replacing the existing workstation environment:

```bash
export FORGESIGHT_VISION_ROOT=/root/.cache/forgesight/vision-v1
test ! -e "$FORGESIGHT_VISION_ROOT" || { echo 'Path exists; inspect or choose a fresh root.'; exit 1; }
/root/pytorch-env/bin/python3 -m venv "$FORGESIGHT_VISION_ROOT/venv"
/root/.local/bin/uv pip sync --python "$FORGESIGHT_VISION_ROOT/venv/bin/python" requirements/pylock.vision.toml
/root/.local/bin/uv pip check --python "$FORGESIGHT_VISION_ROOT/venv/bin/python"
```

On another host, use its explicitly installed CPython 3.12 rather than assume
these workstation-specific paths.
From PowerShell in this project, select a new evidence directory for each run:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.vision_probe --output artifacts/vision/probe_recheck
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.vision_probe --amp-only --output artifacts/vision/amp_recheck
```

The default probe now includes all AMP candidates; AMP-only can extend an older
probe without repeating CPU/FP32 cases. Existing evidence directories are refused.
No output maps are displayed or labeled as trained defect detections.

A fresh temporary environment installed only from the committed lock, passed
dependency checking and actual CUDA EfficientAD-S/PatchCore-ResNet-18 forward
checks, then was removed automatically. Evidence is in
`artifacts/vision/stage3_1/lock_recheck.log`. All 70 project tests pass, including
percentile validity and bank-size scaling checks. Full package inventories for
the original image, shared GPU and ExecuTorch environments match their before
snapshots; `preserved_environments.json` records the checks.

Stage 3.1 stops at model selection and execution compatibility. **Next, Stage
3.2:** acquire/verify the required teacher and regularization assets under their
terms, connect the frozen fit/validation loaders without consuming holdout,
and fit/train the selected baseline with a saved reproducible checkpoint.
Threshold selection and final held-out evaluation follow their own proof checks.
