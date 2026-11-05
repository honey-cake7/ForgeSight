# MVTec AD metal_nut: source and evaluation policy

Stage 1, Substeps 1.1-1.5. Reviewed and locally audited 2026-11-05
(Asia/Calcutta). The category archive and extracted files have been verified.
The split policy below is implemented in a frozen manifest (Substep 1.3).
Development-only ground-truth previews are verified. No model inference has
been run.

## Source and permitted scope

Use the original 2D MVTec AD `metal_nut` category. The
[official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
describes normal-only training, mixed normal/anomalous testing, pixel-level
annotations, and CC BY-NC-SA 4.0 licensing. The legacy `/company/research/`
address redirected to this page with HTTP 200 during this review. The current
page presents a download form. The user supplied `metal_nut.zip` and explicitly
confirmed downloading through the official MVTec page. The assistant submitted
no personal details or forms. No unofficial mirror or guessed archive address
was selected. MVTec AD 2 is a different dataset.

Use this benchmark for the noncommercial learning/research prototype.
Commercial use is excluded by the stated license. When sharing dataset images
or adaptations, retain attribution and license notices, identify changes, and
apply the required ShareAlike terms to shared adaptations. This step does not
establish commercial rights for trained artifacts or decide whether a particular
competition use qualifies as noncommercial. Source code and future controlled
physical samples have separate provenance and licensing.

License references: [CC summary](https://creativecommons.org/licenses/by-nc-sa/4.0/)
and [legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).
Keep raw archives, extracted images/masks, and generated previews outside Git
under the already ignored `data/` and `artifacts/` paths. This repository policy
does not mean the license forbids every form of redistribution.

Dataset attribution: MVTec Software GmbH; Paul Bergmann, Michael Fauser,
David Sattlegger, and Carsten Steger, *MVTec AD -- A Comprehensive Real-World
Dataset for Unsupervised Anomaly Detection*, CVPR 2019, pp. 9592-9600.

## Published facts versus local evidence

The original authors' [paper, Table 1](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf)
reports 220 normal training images, 22 normal test images, 93 anomalous test
images, four defect groups, and a 700-pixel image side length for metal nuts.
These published totals now agree with the local audit below. The four labels
in our contract are `bent`, `color`, `flip`, and `scratch`; their names also appear
in a [published evaluation's supplementary table](https://openaccess.thecvf.com/content/ICCV2023W/LXCV/supplemental/Gula_Gaussian_Image_Anomaly_ICCVW_2023_supplemental.pdf).
Actual directories, per-type counts, image modes, and mask values were verified
in Substep 1.2. The `flip` label concerns an orientation
anomaly; a dataset label does not establish physical damage or severity.

Verified local layout:

```text
data/mvtec_ad/metal_nut/
  train/good/<id>.png
  test/good/<id>.png
  test/<defect>/<id>.png
  ground_truth/<defect>/<id>_mask.png
```

The [officially linked evaluation archive](https://www.mydrive.ch/shares/163177/45dca6b52b7453388ebe2b82e97e064b/download/439517473-1665675012/mvtec_ad_evaluation.tar.xz)
was inspected in memory, without installing or executing its code. Version 1.0's
`evaluate_experiment.py` confirms test/ground-truth pairing by matching defect
directory and filename stem, adding `_mask.png` for defective images, and using
no ground-truth file for `good` images. Positive mask pixels represent anomalies;
do not assume a specific on-disk encoding before checking real masks. A missing
mask for an anomalous image is an error, not an implicit normal image.

Reference archive: 11,056 bytes; SHA-256
`dfcda7d67eee25316ec6ae5042c0b1684a4cabf33b2346be351e2ce36013f220`.
It is reference material only; its historical Python/dependency recipe was
not adopted. No external evaluation files were copied into this repository.

## Local acquisition and actual validation: Stage 1.2

The user supplied the extracted folder under Downloads. Inspection of matching
`metal_nut*` entries also found the original `metal_nut.zip`; the original ZIP
was used for acquisition, leaving both Downloads copies untouched.

| Provenance item | Actual record |
| --- | --- |
| User-provided archive | `C:\Users\RF AND SIMULATION\Downloads\metal_nut.zip` |
| Project archive copy | `data/downloads/metal_nut.zip` |
| Archive format/size | ZIP; 165,672,999 bytes |
| Extracted category | `data/mvtec_ad/metal_nut/` |
| Publisher source | Official MVTec page above, confirmed by the user |
| Exact archive URL | Not captured; Windows download-origin metadata was absent |

Archive SHA-256:
`5468b22edb8e3ebcb2e415bcc6adbed571e1b8a4dec03d1747e52d696384d838`.
This is a locally calculated fingerprint, not a comparison with a publisher-
supplied signed checksum. The Downloads archive, project archive copy, and
audit report agree on this hash.

Actual image/mask counts:

| Source directory | Images | Matching masks |
| --- | --- | --- |
| `train/good` | 220 | Not provided for normal images |
| `test/good` | 22 | Not provided for normal images |
| `test/bent` | 25 | 25 |
| `test/color` | 22 | 22 |
| `test/flip` | 23 | 23 |
| `test/scratch` | 23 | 23 |
| Total | 335 | 93 |

Every image and mask was fully decoded for structural validation. All are
700x700 PNGs: images use RGB, masks use grayscale `L` with binary values 0 and
255. Every anomalous image has a matching nonempty mask, with no orphan masks.
No image was displayed, no split assigned, and no prediction computed.

The ZIP contains 444 entries, including directories. Its 430 regular files
consist of 335 images, 93 masks, `readme.txt`, and `license.txt`. The notices
confirm dataset attribution and CC BY-NC-SA 4.0 licensing. All 430 extracted
files matched their audited per-file hashes and the final inventory matched
the audit. These checks do not establish distinct physical part IDs or remove
the need for duplicate analysis before partitioning.

Ignored local evidence:

- `data/metadata/metal_nut_download_origin.json`: source confirmation and archive
  fingerprint; no invented direct download URL.
- `artifacts/data/metal_nut_archive_audit.json`: archive-only audit.
- `artifacts/data/metal_nut_preparation.json`: archive and extraction proof,
  including per-file hashes, decoded-pixel hashes, sizes, modes, and pairing.

Repeat the read-only audit from PowerShell in the project directory:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B scripts/audit_metal_nut_archive.py data/downloads/metal_nut.zip
```

For a fresh setup only, after obtaining the archive and when both the destination
directory and report path are absent:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B scripts/audit_metal_nut_archive.py data/downloads/metal_nut.zip --extract-to data/mvtec_ad/metal_nut --report artifacts/data/metal_nut_preparation.json
```

The script accepts category ZIP/tar archives, checks paths and entry types,
decodes PNGs, checks counts/mask pairing, and optionally copies only audited
regular files. It refuses existing extraction destinations and report files.
Extraction verifies the archive fingerprint again and compares every extracted
file hash; failure leaves partial files for inspection rather than replacing or
cleaning up local work. No extraction occurs during a default audit.

Verification command:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m unittest discover -s tests -v
```

Fourteen synthetic tests passed, including valid ZIP/tar layouts, unsafe paths,
links, duplicate destinations, corrupt PNGs, missing/empty/nonbinary masks,
changed archives, and refusal to overwrite an extracted dataset. Synthetic
fixtures test validator behavior; the actual ZIP audit above is separate evidence.

## Project split policy v1: choose before viewing samples

Anomaly detection learns normal appearance; it does not learn four defect
classes merely because folder names exist. Keep the original source folders
unchanged and assign roles in a separate manifest before generating previews.

| Source | Project role | Expected count | Permitted use |
| --- | --- | --- | --- |
| `train/good` | fit | 176 | Fit model and training-dependent preprocessing |
| `train/good` | validation_normal | 44 | Select settings/calibrate normal false-alarm behavior |
| One image per anomalous test folder | development_preview | 4 | Learn the data and debug image/mask display |
| All remaining original test images | final_holdout | 111: 22 normal, 89 anomalous | One final evaluation after configuration is frozen |

Counts agree with the acquired inventory. Substep 1.3 verified zero byte-identical
and zero decoded-pixel-identical image groups and froze all role assignments.

Deterministic selection rule for the later manifest implementation:

1. Use seed `42` and version string `forgesight-metal-nut-v1`. Paths are POSIX
   paths relative to the category root, such as `train/good/000.png`.
2. Rank paths by the hexadecimal SHA-256 of UTF-8 text
   `forgesight-metal-nut-v1|42|<relative_path>`; break ties by relative path.
3. For normal training images, take the first floor(20% of the count) as
   validation; use the remainder for fitting.
4. Within each of the four anomalous test folders, reserve the first ranked
   image and its mask for development previews. All other original test images,
   including every `test/good` image, enter the final holdout.
5. Record source paths, roles, content hashes, mask links, and the policy version.
   Check source and decoded-pixel duplicates before accepting the partition;
   conflicting roles require review and a documented policy update before
   training. Do not silently accept leakage or change the seed to improve results.

The development positives never enter fitting or threshold calibration. Normal-
only validation can estimate false alarms, with substantial uncertainty from
44 samples; it cannot optimize defect recall or F1. Do not use final holdout
images/masks for model selection, resizing decisions, threshold tuning, or
preview contact sheets. Image hashing and structural inventory are permitted
to establish provenance before evaluation.

Trade-off: reserving four examples permits honest hands-on learning with real
anomalies, but final results use a custom 111-image project holdout, not the full
115-image official test set. The 80/20 normal split also fits fewer images than
an all-training-images baseline. Report both differences when comparing models.
Keep the same frozen manifest for fair project baseline comparisons. If final
test results guide another revision, those images are no longer an untouched
test set; record that and obtain independent evaluation evidence.

Physical part IDs are not yet verified. Byte/pixel duplicate checks alone cannot
prove that different photographs show different physical parts. Later camera
validation requires distinct controlled samples and part-level splitting.

## Frozen manifest: Stage 1.3

`forgesight/data_manifest.py` rebuilt fingerprints for all current source files
against the acquisition report, checked image byte/pixel duplicates, and assigned
the recorded roles. Actual counts are 176 fit, 44 normal validation, four
development previews, and 111 final holdout. No path appears twice; source test
images cannot enter fitting or normal validation. Changing roles even while
preserving counts fails validation against the recorded deterministic policy.

Committed metadata lives in `manifests/metal_nut_v1.json` and its `.sha256`
sidecar. It contains relative paths, ground-truth labels, image/mask hashes,
geometry, source archive fingerprint, and roles; it contains no image pixels.
Manifest SHA-256:
`14b11ebd03d5107c7ef3478e1c4d20887237e2fbef58601ac043014b8f0e340f`.
A fresh build in memory must reproduce the frozen bytes exactly. The checksum
detects accidental changes; Git history records deliberate policy revisions.

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.data_manifest --check
```

For a fresh checkout with the acquired dataset and preparation report in place,
the check above verifies the committed manifest. The creation command without
`--check` is only for a new manifest path; it refuses existing files. Eight new
split-integrity tests and the existing archive tests passed (22 tests total).

## Verified image/mask loader: Stage 1.4

`forgesight/data_loader.py` exposes one manifest role through `MetalNutDataset`.
`InspectionInput` keeps `ground_truth_type` separate from any future prediction.
Every loaded file is checked against its byte hash, decoded-pixel hash, size,
and mode. It preserves source resolution and orientation; it does no training
transforms or resizing. Missing/changed anomalous masks fail explicitly.

Normal samples receive an all-zero `L` mask with `mask_origin=implicit_normal`.
Anomalous samples retain the dataset annotation with
`mask_origin=dataset_annotation`. Ground truth is not an inference result.

Real verification loaded all 176 fit and 44 normal-validation samples with zero
nonempty masks, and all four development samples with one nonempty mask per
defect type. The final-holdout role was refused without explicit evaluation
opt-in, before loading pixels. This gate prevents accidental development
access; it is not an access-control system against deliberate code changes.
Nine new loader tests passed, bringing the suite to 31 tests.

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.data_loader --role fit
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.data_loader --role validation_normal
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.data_loader --role development_preview
```

## Verified development previews: Stage 1.5

`forgesight/data_preview.py` uses the loader to render the first fitting normal
image and the four frozen development samples. It has no holdout-selection
argument and opens only fitting/development image files. Five new tests check
overlay alignment/binary masks, unchanged background pixels, rendering with all
unselected files absent, and refusal to replace an existing output directory.
All 36 current tests passed.

Real outputs were generated and visually inspected at
`artifacts/data/development_preview_v1/contact_sheet.png` and `preview.json`.
The sheet labels every panel as source, annotation mask, or red annotation
overlay; the title states that these are ground truth, not model predictions.
Source/license attribution and display modifications are included. Original
700x700 inputs remain unchanged; 300x300 tiles are only a display choice.

| Displayed source | Role | Annotation area in source pixels |
| --- | --- | --- |
| `train/good/000.png` | fit | 0 |
| `test/bent/017.png` | development_preview | 9,041 |
| `test/color/009.png` | development_preview | 8,684 |
| `test/flip/012.png` | development_preview | 234,723 |
| `test/scratch/020.png` | development_preview | 16,654 |

These areas describe supplied masks, not measured physical defects, severity,
or model detections. The flip example illustrates why large annotation area
alone does not establish surface-damage severity. Zero final-holdout images
were displayed. Previews and their metadata remain ignored local artifacts.

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.data_preview
```

This command creates a new output directory and refuses to overwrite an existing
one. To regenerate into another new local directory, use
`--output artifacts/data/development_preview_review`.

## Stage 1 gate and next task

Stage 1 technical proof is observed on the PC: real inputs/masks load, actual
labels/counts are reported, source/license and partitions are documented,
development overlays are saved and inspected, and raw dataset pixels are not
tracked. Await user review before advancing to Stage 2.

No anomaly detector, subtype classifier, severity policy, predictions, or model
metrics exist. Physical camera generalization and every Jetson requirement
remain unverified. Stage 2.1 next reviews the Llama/ExecuTorch ARM64 runtime
prerequisites and device access; actual on-device proof remains blocked until
the Jetson is available. A PC run or exported file cannot satisfy that gate.
