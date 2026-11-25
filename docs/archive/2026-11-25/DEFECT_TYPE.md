> Historical snapshot archived on 25 November 2026. This page preserves development history; use the [current documentation](../../../README.md) for supported instructions.

# Provisional metal-nut defect type

## Four-type benchmark classifier

Use `--defect-type-profile configs/defect_type_mvtec_four_v1.json` with the Jetson
operator app for original saved-image replays or declared MVTec PNG uploads.
The result and inspection history display the provisional label. The classifier
runs after EfficientAD flags an image. Live camera input is outside its
validated scope.

The four-type profile also enables [provisional severity](DEFECT_SEVERITY.md).

The PC fitted 31 depth-limited trees on 53 flagged training images. Bent (14),
flip (13), scratch (11), and discoloration (13) each had a separate training
label. Two flagged good images supplied the unknown class. Features combine
the existing local color and elongation cues with response size and position,
nut outline, and face and rim appearance measured at 96 by 96 pixels. The
profile stores JSON thresholds and leaf labels. Each tree casts one vote.
Validation selected 17 of 31 votes to emit a type. Lower agreement yields
`unknown`. Jetson inference uses the existing OpenCV and NumPy runtime.

| Actual type | Correct / tested | Other outcomes |
| --- | ---: | --- |
| Bent | 4/5 | 1 unknown |
| Flip | 5/5 | |
| Scratch | 5/5 | |
| Discoloration | 3/4 | 1 unknown |
| Good | 5/5 unflagged | |

This is the same 24-image MVTec subtype test used for the earlier profile.
It was excluded from fitting and vote-threshold selection, and has been
evaluated previously. Validation included one scratch predicted as bent.
Discoloration coverage fell from 4/4 with the earlier tree to 3/4. The
four-type profile stays opt-in under the existing promotion criterion.

Threat limit: an assessment stored on a saved inspection is validated for
shape and consistency when used, but it is not independently re-derived from
the image at display time. Someone with write access to a stored record
could replace an assessment with a different valid-looking one (for example
a type that yields an unassessed severity) without detection. The
authoritative grade, detector score and severity evidence are unaffected -
they are recomputed from the record and contradicting evidence fails closed -
but the provisional type label alone carries no tamper-evident provenance
guarantee.
The isolated Orin Nano app reproduced all 48 expected outcomes (24 per profile).
Scores, maps, overlays and grade agreed across paired runs. Current result,
history and SQLite agreed. Four-type assessment on 19 flagged images took
17.6 ms median and 62.2 ms p95.
The JSON profile is approximately 49 KiB. Incremental RAM, physical-camera
accuracy and generalization to new parts are unmeasured.

Train from the repository root using a fresh output directory:

```sh
python -B -m scripts.train_defect_type --four-types --mvtec-subtype-experiment --split manifests/metal_nut_defect_type_split_v1.json --model artifacts/vision/edge_export_v1/efficientad_s_calibrated.ts --output artifacts/vision/four_type_reproduction
```

To repeat the device check, use the isolated Jetson comparison command below
with the generated `evaluation.json` and `configs/defect_type_mvtec_four_v1.json`.
The [evaluation record](../../../manifests/mvtec_four_type_v1.json) lists the split,
confusion counts and device measurements.

## Earlier two-type and preview profiles

ForgeSight can attach a provisional `assessment` to an EfficientAD anomaly on
an original dataset replay or a 700 × 700 upload declared as MVTec metal nut.
The current preview profile reports `discoloration` when it sees a local color
shift and returns `unknown` otherwise. The optional within-MVTec tree also
supports `scratch`. The operator page displays possible types and includes them in
recent history. SQLite stores the assessment with the inspection record.

EfficientAD supplies the existing score, anomaly map, and largest response
region. The assessment measures the ratio of the region's principal-axis
spreads and the distance between median red–green and blue–green differences
in the region and in a nearby ring. The ring excludes other response pixels.
It can still cross a part boundary or contain a reflection. Those cases need
independent evaluation.
The profile pins the detector export and threshold. ForgeGrade and existing
advisory contracts continue to use the authoritative inspection fields.

## Four-image PC experiment

Run from the checkout with PyTorch, NumPy, Pillow, and OpenCV available.
The observed run used Python 3.11, PyTorch 2.8.0+cu128 on CPU, NumPy 2.2.6,
Pillow 12.3.0, and OpenCV 5.0.0. The script verifies the model and loads only
the four `development_preview` images through the dataset manifest.

```powershell
python -B -m scripts.probe_defect_type_features --model artifacts/vision/edge_export_v1/efficientad_s_calibrated.ts --output artifacts/vision/defect_type_preview_check
```

| Preview | EfficientAD score | Region elongation | Chroma difference | Preview assessment |
| --- | ---: | ---: | ---: | --- |
| `bent/017.png` | 0.682809 | 1.587 | 1.414 | Unknown |
| `color/009.png` | 0.820713 | 2.773 | 12.207 | Possible discoloration |
| `flip/012.png` | 1.105062 | 1.554 | 2.236 | Unknown |
| `scratch/020.png` | 0.155062 | — | — | No anomaly flagged |

The chroma threshold of 7.0 is an exploratory choice from these previews.
The scratch score fell below the frozen cutoff of 0.227766, so the subtype
stage had no region to measure. Ground-truth masks are used only for the
probe's map-overlap diagnostic. The output directory holds the report and
overlays locally. The four observations establish feasibility for the color
cue. The scratch cue, classifier accuracy, and camera-domain validity remain
open.

## Enable the feature on Jetson

Add the profile argument to the [existing operator launch](JETSON_LIVE_UI.md).
From the Jetson vision directory with the branch files staged locally:

```sh
/usr/bin/python3 -B -m forgesight.jetson_live_ui --port 8770 \
  --defect-type-profile configs/defect_type_preview_v1.json
```

Use a saved dataset replay or select the MVTec declaration when uploading an
original benchmark PNG. A normal candidate, arbitrary upload, or camera
capture has no assessment. A flagged benchmark image with insufficient
evidence shows `Defect type unknown`. Existing history entries without an
assessment still render normally. The same profile switch is available on
both `vision_edge_live_once` and `vision_edge_replay`.

## Train and evaluate the decision tree on the PC

### Frozen within-MVTec experiment

For the experiment using the existing MVTec data, the subtype split assigns
66 images to training, 21 to validation, and 24 to test. The builder ranks
original paths by a seeded hash within each truth label. It reads the original
manifest metadata and excludes all four development previews. The source
images had already been scored in the original detector benchmark. The new
test split is separate from subtype fitting and measures within-MVTec behavior.
Use a separate camera-domain dataset to evaluate transfer to physical parts.

Holdout scope: the 111 images are the detector's frozen final-holdout
evaluation set (`metal-nut-frozen-holdout-v1`), repartitioned 66/21/24 for the
classifier. The same 24-image test subset has reported numbers for successive
classifier candidates (the two-type tree and the four-type forest), so those
numbers come from small, repeatedly evaluated holdout-derived samples. The
detector's frozen-threshold metrics are unaffected: the classifier only
consumes detector outputs and changes no detector artifact. Classifier
accuracy remains provisional until confirmed on fresh images none of the
candidates has seen.

```powershell
python -B -m scripts.build_mvtec_defect_type_split --output artifacts/vision/mvtec_subtype_split_check.json
python -B -m scripts.train_defect_type --mvtec-subtype-experiment --split manifests/metal_nut_defect_type_split_v1.json --model artifacts/vision/edge_export_v1/efficientad_s_calibrated.ts --output artifacts/vision/defect_type_mvtec_v1
```

The explicit experiment switch verifies the frozen assignments and all original
image hashes. EfficientAD uses the existing export and threshold. The tree is
fit on training features, and validation decides which leaves can emit a
known label. The comparison uses the untouched subtype test split. Advance
the tree to a deployment benchmark if it makes more correct type assignments
in total, preserves or improves correct assignments for both target types,
and makes no additional wrong type assignments. Detector misses are counted
separately as `not_flagged` for both methods.

The PC run used 53 flagged training images with usable features, including 11
scratches and 13 discoloration images. EfficientAD missed two training
scratches. In the 24-image subtype test it flagged all 19 defect images and
none of the five good images. Image-level candidate flags agreed with the
archived Jetson scores for each test image.

| Subtype test outcome | Color rule | Depth-3 tree |
| --- | ---: | ---: |
| Correct scratch labels | 0/5 | 1/5 |
| Correct discoloration labels | 4/4 | 4/4 |
| Incorrect type assignments | 2 | 0 |
| Unknown | 13 | 14 |
| Below anomaly threshold | 5 | 5 |

The tree passed the comparison gate. It abstained on four flagged scratches
and on all ten bent and flip images. Both good-image false alarms from the
historical benchmark fell in the subtype training split, so the subtype test
contains no flagged good images. This small test supports a provisional demo
profile and leaves reliable scratch classification open. To try the
tree on benchmark PNGs, run the operator app with the option below. See the
[PC results and artifact hashes](../../../manifests/mvtec_defect_type_evaluation_v1.json).

```sh
/usr/bin/python3 -B -m forgesight.jetson_live_ui --port 8770 \
  --defect-type-profile configs/defect_type_mvtec_tree_v1.json
```

### Independent labeled split

Gather independent labeled 700 × 700 RGB PNGs. Provide a local version 1 JSON
split manifest with `schema_version: 1` and a `samples` array. Each sample has
`path` relative to the manifest, file `sha256`, `group` for the original part or
source, `split` (`train`, `validation`, or `test`), and `label`. Each split
needs examples of `scratch`, `discoloration`, and other or normal images.
Other supported labels are `good`, `bent`, `flip`, and `unknown`. Count separate
parts when deciding how much evaluation evidence you have. Keep images of
the same source or part in one split. The trainer rejects duplicate pixels,
cross-split groups, and file or pixel matches to the existing detector holdout.

Run the detector coverage check while assembling the split. This accepts
incomplete splits and saves per-label image counts, distinct source groups,
anomaly flags, and usable feature rows in `coverage.json`. The report also
identifies missing minimum training inputs. Every training run saves the same
report before fitting. The minimum input check establishes that the procedure
can run. Review sample counts, source independence, and representativeness
before interpreting accuracy.

```powershell
python -B -m scripts.train_defect_type --audit-only --split data/defect_type_v1/split.json --model artifacts/vision/edge_export_v1/efficientad_s_calibrated.ts --output artifacts/vision/defect_type_coverage_v1
```

```powershell
python -B -m scripts.train_defect_type --split data/defect_type_v1/split.json --model artifacts/vision/edge_export_v1/efficientad_s_calibrated.ts --output artifacts/vision/defect_type_candidate_v1
```

This command uses scikit-learn on the PC (1.9.0 in the smoke test). The exported JSON tree has maximum
depth 3 and runs with the existing Jetson packages. Leaves with missing or
inconsistent validation support return `unknown`. Compare both profiles in
`evaluation.json` on the independent test set, including upstream `not_flagged`
results and false type assignments. Use `candidate.json` with the same Jetson
profile switch only after reviewing those results and timing it on-device.
The independent-data trainer was smoke-tested on nine generated appearances
from distinct fitting images. It produced a tree and comparison report. These
generated appearances provide CLI verification only.

### Independent data review

The local inventory contained only the original MVTec nut dataset. Running the
coverage command on its four eligible development previews yielded one usable
feature row each for discoloration, bent, and flip, and zero for scratch.
There were no independent validation or test groups available for this run.

A separate read of the cached Jetson detector predictions verified the
prediction-file hash and matched all 111 rows to the frozen manifest. The
detector flagged **20/22 scratches**, **21/21 discoloration images**, and
**2/22 good images**. This measures upstream detector coverage only. The file
contains image scores and carries no classifier training features. The
classifier run used the four development previews only. See the
[Jetson evaluation evidence](../../../manifests/jetson_holdout_batch_v1.json).

The closest published sources reviewed were:

- **ReliVision metal-nut playbook.** The download requires access. The three
  example classes, counts, and part appearance resemble a subset of MVTec.
  Confirm sample provenance and permission before treating the collection as
  independent. [Publisher playbook](https://www.relimetrics.com/knowledge-hub/playbooks/metal-nuts-use-case)
- **ALPS Alpine metal-nut set.** Its published description identifies the
  scratch and color categories as inherited from MVTec and adds one simulated
  category. [Dataset notes](https://github.com/ALPSALPINE-ProSysDevDept/industrial-ad-dataset#metal_nut_dataset)
- **CAD-SD.** Its sample annotations place scratching and paint on screw rods
  in assembled fasteners. [Original dataset](https://github.com/IshidaKengo/Co-occurrence-Anomaly-Detection-Screw-Dataset)
- **Roboflow Nuts defect.** The catalog labels include scratches and rust.
  Download requests for the reviewed variants were blocked, and independent
  source groups and equivalent discoloration annotations remain unverified.
  [Dataset catalog](https://universe.roboflow.com/nuts-defect/nuts-ytdpr)

## Reproduce the isolated Jetson comparison

Use a separate checkout with the four original previews locally available.
The command starts its own operator service with a persistent vision worker,
isolated output folders, a loopback-only network namespace, and a fresh
database. Supply the actual installed vision-runtime prefix in `FS_VISION`.

```sh
export FS_VISION=/home/nvidia/.local/share/forgesight/vision_cpu_v1
unshare --user --map-root-user --net sh -c '
  ip link set lo up && /usr/bin/python3 -B -m scripts.verify_jetson_defect_type \
    --output runs/defect_type_check --model "$FS_VISION/efficientad_s_calibrated.ts" \
    --vision-python "$FS_VISION/venv/bin/python" --dataset-root data/mvtec_ad/metal_nut
'
```

The comparison checks byte-equal source-map and overlay hashes, exact scores
and decisions, and matching assessment fields across the current result API,
history API, and SQLite. The network namespace isolates the test application
while the Jetson host retains its normal network and station service.

On the Orin Nano, all six profile-enabled inspections matched their baseline
inspection scores, candidates, source maps, overlays, and decisions. Three
color inspections reported possible discoloration, bent and flip abstained,
and scratch remained below the anomaly threshold. The two inline benchmark
uploads measured **7.2 and 8.8 ms** in assessment. The eligible replay
subprocesses measured **32.5–179.0 ms**. These are individual timings for the
selected cases. The loopback test exercised the HTML and inspection API with
real inference, and SQLite readbacks matched history. Browser rendering,
camera accuracy, and full-station offline acceptance remain separate checks.
See the [measured evidence](../../../manifests/jetson_defect_type_preview_v1.json).

To compare the color rule and tree instead, stage the 24 original PNGs in the
frozen subtype test and the unedited PC `evaluation.json`. Add the following
arguments inside the same isolated verification command:

```sh
--evaluation artifacts/vision/defect_type_mvtec_v1/evaluation.json \
--candidate-profile configs/defect_type_mvtec_tree_v1.json
```

The Orin Nano comparison processed 24 images per profile. All 48 assessment
outcomes matched the PC results and were consistent across the result API,
history, and SQLite. Both profiles yielded exactly the same detector score,
candidate, maps, overlay, and ForgeGrade decision per image. The 19 flagged
images per profile took a median **6.4 ms** and p95 **34.9 ms** for tree
assessment, compared with **6.3 ms** and **31.0 ms** for the rule. This timing
covers feature extraction and the final decision, excluding detector inference.
The active station retains its own configuration. The staged upload of
`test/scratch/013.png` exercised the tree's positive scratch label.
See the [device measurements](../../../manifests/jetson_mvtec_defect_type_tree_v1.json).
