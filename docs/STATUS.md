# Current status

Updated 25 November 2026. This page describes the current implementation;
historical checkpoints are in the [archive](archive/README.md).

## Implemented

Phase 1 source includes camera/upload inspection, EfficientAD detection,
provisional type/severity, deterministic grading, SQLite history/review/report
export, local ExecuTorch Llama advice, two part recipes, persistent workers,
recovery/logging, storage health and desktop launch tools. See the
[feature list](../README.md#features) and [user guide](USER_GUIDE.md).

## Recorded evidence

- Offline saved-image vision, grading, SQLite and real Llama advice ran on
  the physical Jetson; loopback/recovery checks are recorded.
- Metal-nut holdout: TP 87, TN 20, FP 2, FN 2; Jetson matched PC decisions.
- Supplied warm timings: roughly 2.17-2.20 s inspection work and a separate
  1.62 s direct Llama call; recipe switching averaged 0.50-0.54 ms.
- Module input power: 8.52 W sampled peak / 6.60 W active-phase mean across
  12 saved-image attempts with live preview; [power evidence](../manifests/jetson_power_measurement_v1.json).
  Real Llama generation was confirmed; all 12 advisory reports used checklist fallback.
- Reviewer application package: fresh private install, bounded native Llama
  generation, generated desktop entry/visible window and both saved-image recipes
  passed on the recorded Jetson. [Evidence](../manifests/reviewer_application_v1.json).
  Application assets: [v1.0.0 release](https://github.com/honey-cake7/ForgeSight/releases/tag/v1.0.0).
- Latest recorded hybrid prompt check: 12/15 model choices matched; three
  checklist fallbacks. This was not a whole-station upgrade.

Measurement boundaries and supporting manifests are in [TECHNICAL_REPORT](TECHNICAL_REPORT.md).

## Still to document or verify

Fresh subtype/screw reliability, physical-camera accuracy, newest-build full
disconnected cold start, full-cycle throughput/resource peaks, wall power and worst-case power
remain scoped acceptance items. The roughly 3.8 s timing sum is an estimate.
Link the demo and confirm asset handoff/code licensing for submission; artifacts
may exist outside this repo. The consolidated technical report now exists here.

Next action: add the release/demo links to the hackathon entry and close
submission items before Phase 2. The user authorized source push and release publication on 25 November 2026.

## Continue on another PC

- Clone `https://github.com/honey-cake7/ForgeSight.git` and use `main`.
  Start with [SETUP](SETUP.md); both runtimes and device evidence are in
  [TECHNICAL_REPORT](TECHNICAL_REPORT.md).
- User confirmed the Downloads package installed and opened successfully with
  camera index 0 on 25 November. Desktop portal/keyring warnings appeared but
  the application worked. This is user-reported operation, not new accuracy or
  disconnected acceptance evidence.
- Active user installation on Jetson:
  `/home/nvidia/.local/share/forgesight/reviewer_camera_v1`;
  configuration `station.json`, port `18770`. It was installed with
  `--no-shortcuts`; the older desktop icon still points to the old station.
- Application archive and its checksum/manifest are in Jetson
  `/home/nvidia/Downloads/ForgeSight/` and this PC's `artifacts/submission/`.
  Archive SHA-256:
  `f284c9d555ac6cde8109c5f7e6cca96925d583875e14087753ba774248db2691`.
  These binaries are ignored by Git; download them separately from the
  [release](https://github.com/honey-cake7/ForgeSight/releases/tag/v1.0.0).
  The checksum sidecar uses Linux line endings.
- This PC also has an untracked user file `scripts/record_timings.py`; it was
  preserved and must be copied separately if needed.
- Remaining submission work: add release/demo URLs to the hackathon entry,
  decide the source-code license, and complete
  any remaining scoped device acceptance. Do not claim checklist fallback as
  accepted Llama advice or a successful camera scene as measured accuracy.
