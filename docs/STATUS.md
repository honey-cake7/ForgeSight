# Current status

Updated 25 November 2026. Phase 1 includes inspection, two part recipes,
deterministic grading, local ExecuTorch Llama advice, history/review/report tools
and the Jetson desktop launcher. See the [features](../README.md#features) and
[user guide](USER_GUIDE.md).

## Submission materials

| Material | Link |
| --- | --- |
| Application and trained-model package | [v1.0.0 release](https://github.com/honey-cake7/ForgeSight/releases/tag/v1.0.0) |
| Two-minute demonstration | [YouTube](https://youtu.be/NZccgnyDJrY) |
| Technical report | [PDF](https://github.com/honey-cake7/ForgeSight/releases/download/v1.0.0/ForgeSight_Submission_Report_Revised.pdf) / [repository report](TECHNICAL_REPORT.md) |
| LaTeX report source | [Portable ZIP](https://github.com/honey-cake7/ForgeSight/releases/download/v1.0.0/ForgeSight_LaTeX_Revised.zip) |

## Evidence

- Disconnected saved-image inspection, SQLite, real Llama advice and loopback/
  recovery workflows have physical Jetson evidence.
- Frozen metal-nut holdout: TP 87, TN 20, FP 2, FN 2; Jetson matched PC decisions.
- Recorded warm inspection work: 2.17-2.20 s; separate direct Llama call: 1.62 s.
  Their roughly 3.8 s sum is an estimate, not full-cycle latency.
- Sampled module input power: 8.52 W peak / 6.60 W active-phase mean in the
  recorded saved-image workload. Wall power and worst-case peaks were not measured.
- The release passed fresh private-environment installation, native generation,
  visible desktop launch and both saved-image recipes on the recorded Jetson OS.
  The owner subsequently confirmed operation with camera 0. These checks do not
  establish physical-camera accuracy or a wiped-OS installation.

Measurement scopes, fallback outcomes and supporting manifests are in
[TECHNICAL_REPORT](TECHNICAL_REPORT.md). Install instructions are in the
[README](../README.md#download-and-install-on-jetson) and [SETUP](SETUP.md).

## Remaining work

- Confirm the release/demo links are included in the hackathon entry and choose
  a source-code license; no root code LICENSE is currently supplied.
- Validate fresh subtype/screw reliability, physical-camera accuracy and the
  latest complete disconnected cold start.
- Measure full-cycle throughput/latency, simultaneous resource peaks, wall power
  and long-duration/worst-case behavior before making those claims.

Future extensions are listed in the [roadmap](../README.md#future-development).
Historical checkpoints remain in the [archive](archive/README.md).
