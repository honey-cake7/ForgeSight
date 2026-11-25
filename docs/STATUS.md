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
- Latest recorded hybrid prompt check: 12/15 model choices matched; three
  checklist fallbacks. This was not a whole-station upgrade.

Measurement boundaries and supporting manifests are in [TECHNICAL_REPORT](TECHNICAL_REPORT.md).

## Still to document or verify

Fresh subtype/screw reliability, physical-camera accuracy, newest-build full
disconnected cold start, full-cycle throughput/resource peaks and actual power
draw remain scoped acceptance items. The roughly 3.8 s timing sum is an estimate.
Link the demo and confirm asset handoff/code licensing for submission; artifacts
may exist outside this repo. The consolidated technical report now exists here.

Next action: review the concise documentation and close the required evidence/
submission items before the roadmap's Phase 2 extensions. Push requires explicit
user confirmation; current work remains local.
