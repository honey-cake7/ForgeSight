# Challenge requirements and evidence

| Requirement | Current record |
| --- | --- |
| Detect defects/anomalies | Real EfficientAD metal-nut benchmark and Jetson parity; camera-domain accuracy remains separate. |
| Classify and estimate severity | Provisional part-specific profiles and deterministic pixel rules; fresh independent reliability/calibration still required. |
| Alert and advise | Local operator decisions and real ExecuTorch Llama guidance, with unavailable/fallback handling. |
| Log inspections | SQLite/JSON records, accepted errors, restart-safe IDs and append-only operator claims. |
| Run locally without cloud inference | Recorded offline Jetson saved-image workflows; latest combined cold-start/physical-camera acceptance remains scoped. |
| Mandatory Llama / ExecuTorch | Llama 3.2 1B Instruct actually generated on-device via ExecuTorch/XNNPACK. Vision uses its separate CPU Torch runtime. |
| Source and device application | Source/setup plus prepared reviewer application package with trained models and Jetson desktop launcher; [fresh-install/device proof](../manifests/reviewer_application_v1.json). Package attachment/publication remains. |
| Demo video | No video link currently recorded in the repository. |
| Technical report | [Consolidated report](TECHNICAL_REPORT.md), including both runtimes, results and limits. |

The guide permits static/camera/video demonstrations and publishes no fixed FPS
target. The project roadmap adds a physical-part proof gate. Module input power
now has [sampled workload evidence](../manifests/jetson_power_measurement_v1.json):
8.52 W peak / 6.60 W active-phase mean. Wall power, full-cycle throughput/latency
and simultaneous resource peaks remain unmeasured; 25 W is the mode setting.

[Current status](STATUS.md), [measurement/evidence links](TECHNICAL_REPORT.md),
[original requirement history](archive/2026-11-25/COMPLIANCE.md),
[organizer reply](archive/2026-11-25/ORGANIZER_CLARIFICATION.md).
Dataset/model terms do not license project code; a root code LICENSE is not
currently supplied. No repository setup alone establishes completed acceptance.
