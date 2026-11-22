# Provisional metal-nut defect type

ForgeSight can attach a provisional `assessment` to an EfficientAD anomaly on
an original dataset replay or a 700 × 700 upload declared as MVTec metal nut.
The current preview profile reports `discoloration` when it sees a local color
shift and returns `unknown` otherwise. A future trained profile also supports
`scratch`. The operator page displays possible types and includes them in
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
There is no trained defect-type artifact in this branch. Deployment of a
trained profile awaits an independent same-domain split with usable flagged
training examples of both target defects. The trainer was smoke-tested on
nine generated appearances from distinct fitting images. It produced a tree
and comparison report as expected. These generated appearances provide CLI
verification only.

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
[Jetson evaluation evidence](../manifests/jetson_holdout_batch_v1.json).

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
See the [measured evidence](../manifests/jetson_defect_type_preview_v1.json).
