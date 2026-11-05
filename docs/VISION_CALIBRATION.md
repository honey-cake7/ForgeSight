# Stage 3.3: calibrate maps and freeze a normal-only score policy

The 10,000-step FP32 EfficientAD-S checkpoint is now paired with fitted map
scales and a versioned image-score threshold. This supplies candidate anomaly
flags, not final quality grades, defect types, severity, probabilities, or
independent accuracy evidence. No optimizer updates were made in this step.
The final 111-image holdout remains unopened for evaluation.

## Why the 44 normals are split again

The outer frozen manifest and its checksum remain unchanged. Within its
validation_normal role, 22 images fit map scaling and 22 different images set
the score threshold. Selection is deterministic: sort by SHA-256 of seed:path
with seed 42, then path; the first 22 are map-fit. The complete assignments,
source fingerprints and scores are in `manifests/efficientad_s_calibration_v1.json`.

This makes the score function fixed independently of the threshold-setting
examples. Fitting both scales and threshold on the same 44 would not support
the independent-calibration rank interpretation used here. The images remain
outside gradient training; teacher/student/autoencoder weights and the original
fit-only teacher mean/std are unchanged. Fit/development/holdout pixels were not
read. Independent reload later reread two already-used validation anchors.

## Map normalization

The pinned upstream
[map calibration](https://github.com/open-edge-platform/anomalib/blob/cc5f400a4a4b1b14b5a3ee5078063c250a8f522e/src/anomalib/models/image/efficient_ad/lightning_model.py)
uses 90% and 99.5% quantiles of normal branch-map pixels. We use exactly
1,441,792 pixels per branch from the 22 map-fit images, without subsampling.

| Branch | q90 | q99.5 |
| --- | --- | --- |
| Student-teacher | 0.2203965485 | 0.2654230595 |
| Student-autoencoder | 0.0470330380 | 0.1132587194 |

Each raw branch map becomes `0.1 * (map - q90) / (q99.5 - q90)`. The combined
map is their equal-weighted average, and its maximum is the image score. This
matches the [pinned core](https://github.com/open-edge-platform/anomalib/blob/cc5f400a4a4b1b14b5a3ee5078063c250a8f522e/src/anomalib/models/image/efficient_ad/torch_model.py).
Direct model outputs agreed with the recorded normalization within
2.98e-8 maximum absolute map error. Degenerate or nonfinite quantiles fail
instead of silently changing the formula. Values may be negative or exceed
one; the score is dimensionless, not a defect probability.

## Threshold choice and limits

`configs/vision_score_policy.json` predeclares the recipe before reading scores.
The provisional nominal normal-flagging level is 5%, chosen as a first
normal-only reference because the user has not set a production error target.
With n=22 independent threshold scores, choose the k-th smallest score with
`k = ceil((n+1)*(1-alpha)) = 22`. Thus the observed threshold is the maximum
threshold-group normal score:

**threshold = 0.22776579856872559**

| Comparison | Candidate flag |
| --- | --- |
| Score strictly greater than threshold | ANOMALY_CANDIDATE |
| Score equal to or below threshold | NORMAL_CANDIDATE |
| Nonfinite score | Error, never a normal decision |

The finite-sample rank construction follows
[Tibshirani's one-sided/split conformal notes](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf).
Under exchangeability with future normals and an independently fixed score
function, this rank has a nominal upper tail of 1/23, approximately 4.35%.
Those assumptions, physical-part independence, camera-domain stability and
production false-alarm performance are not verified here. This is not a
guaranteed production error rate or a conditional per-run confidence statement.
A finite 1% rank threshold needs at least 99 independent threshold normals,
in addition to separate map-fitting data; this recipe rejects unsupported
levels instead of claiming such evidence from 22 samples.

No defect examples were used, so this policy cannot optimize or establish
recall, precision, best F1, false-negative cost or the best accuracy/speed
trade-off. It is frozen for the next independent evaluation. ForgeGrade and
any safety/quality decision remain separate stages. This step sets an image
score threshold, not a separately validated pixel-mask or severity policy.

## Actual normal-data results, including flags

The 22 threshold-setting normals produce 0 flags by construction. The 22
map-fitting normals produce **3 flags**, preserved without retuning:

| Known-normal image | Score |
| --- | --- |
| train/good/020.png | 0.2371618152 |
| train/good/064.png | 0.2715546787 |
| train/good/156.png | 0.2875634432 |

Both groups participated in calibration. Neither their individual counts nor
their combined count estimates independent accuracy or future false alarms.
The observed normal variation is relevant evidence to carry into evaluation,
not grounds to relabel normal images or move the threshold to hide flags.
The actual [local score plot](../artifacts/vision/calibration_v1/normal_scores_reviewed.png)
shows all 44 values by group; no holdout or defect images appear in that plot.

## Saved bundle and independent reload

The source checkpoint remains unchanged at SHA-256
`85106f4a9ef6ec3e9c8acb4056dc2db2124e54284209e10d686dd36dea66ca0d`.
The new calibrated checkpoint contains weights/plain metadata without optimizer
state: 32,258,710 bytes, SHA-256
`aa2407dba9f74db8472ee6adc5968eefc7c243d9f200d70f2d3a792fd0b7f4fd`.
Only the four map-quantile parameters changed; trained networks and fitting-only
teacher statistics retained their exact digests.

`forgesight.vision_calibration_bundle` checks the policy file fingerprint,
model fingerprint, policy-reference equality, source-checkpoint identity, and
quantiles/threshold consistency inside the saved checkpoint. It also requires
FP32 parameters and restores the reference math flags. The existing
tensor-only loader checks format, model metadata, version and strict state keys.
An independent CUDA process reproduced two saved validation scores exactly,
including the highest threshold-setting score. Equality still produced
NORMAL_CANDIDATE. All 86 tests pass, including split leakage, finite-sample rank,
ties, insufficient calibration data, nonfinite values and corrupted policy
guards. Unit tests do not simulate a successful detector.

The versioned reference is `manifests/efficientad_s_calibration_v1.json`.
The checkpoint, concrete score_policy.json, logs and plots remain ignored under
`artifacts/vision/calibration_v1/` and `artifacts/vision/stage3_3/`. A changed
model, scaling, threshold or precision requires a new version and corresponding
validation; do not overwrite this frozen bundle after examining holdout results.

## Trained model timing

On an already-used cached normal image, calibrated FP32 CUDA forward measured
**14.54 ms p50 / 14.80 ms p95**, with 10 warmups and 50 measured calls. It
includes the network and native map/image-score computation with synchronization,
but excludes image decode, resizing, transfers, candidate comparison, database,
UI, camera, and Llama contention. These are desktop RTX 2000 Ada measurements,
not end-to-end performance or a Jetson prediction. No AMP/TensorRT precision
change was adopted. Raw timings remain in the record; no parts/minute claim is
derived from them.

## Repeat the step

From PowerShell in the project directory, choose a fresh output directory:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.vision_calibrate --output artifacts/vision/calibration_recheck
```

This verifies the original checkpoint and frozen dataset manifest, reads only
the 44 selected normals, fits four map scalars and the declared threshold, checks
upstream equivalence, saves a new bundle and measures cached-input model timing.
It refuses existing output directories and preserves the original training
checkpoint. All four environments' full package inventories match the before
snapshots; dependency checking passes and no packages were installed.

**Next, Stage 3.4:** evaluate this frozen model/policy once on the 111 reserved
holdout images, record actual missed defects/false alarms and continuous anomaly
maps, then review the results. Do not select thresholds, precision or model
variants from final-holdout outcomes. Stage 3, actual Jetson/offline proof,
defect-type classification, severity and the full inspection application remain
incomplete.
