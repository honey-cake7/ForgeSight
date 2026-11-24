# Submission records and documentation audit

Reviewed 24 November 2026 against local source `081ba00` on
`feat/hybrid-llama-advisory`, using the supplied challenge guide and
[README plan](superpowers/plans/2026-11-24-hackathon-readme.md).
This is a repository audit, not a fresh hardware or accuracy test.

**Finding: core methods and device evidence are documented; the complete
submission package is not ready.** Before this change there were 29 tracked
Markdown documents under `docs/` and 73 tracked JSON manifests. All 73 parsed.
That confirms readable metadata, not independent reproduction or continued
availability of private raw logs and models.

**Offline proof is already present and verified for saved-image workflows on
the Jetson.** The disconnected integrated replay, case series, loopback service
and repeated vision/advice records are linked below and in the README. Remaining
newest-build and physical-camera checks do not invalidate that offline proof.

## Required deliverables

| Item | Found | Remaining gap |
| --- | --- | --- |
| Runnable documented source | Modules, tests, policies, separate locks, asset hashes; [setup index](SETUP.md) | Clean-checkout reproduction and reviewer-accessible asset handoff are not established. Root requirements alone are insufficient. |
| Target-device application | Camera/UI, saved-image inspection, SQLite and real Llama/ExecuTorch evidence | Latest whole-build deployment/acceptance is pending; retain the build/scope of historical results. |
| Demo showing detection, alerts and logging | Device/browser records can inform a demo | No demo video or video URL identified in the reviewed tracked README/docs. Link an existing artifact or record/publish the workflow with source mode labeled. |
| Technical report: design, runtimes, results, limits | [Runtime report](MERGE_LATENCY_REPORT.md), methods, hardware, compliance and offline ledger | No consolidated submission report identified in the repo. Link it if prepared elsewhere, or assemble from these sources, including failures and unmeasured metrics. |

## Evidence inventory

| Subject | Documents | Committed records and scope |
| --- | --- | --- |
| Hardware/eligibility | [Hardware](HARDWARE.md), [organizer clarification](ORGANIZER_CLARIFICATION.md) | [Device audit](../manifests/jetson_device_audit_v1.json); eligible ARM route, hardware/runtime facts separately recorded |
| Data/fitting | [Dataset](DATASET.md), [training](VISION_TRAINING.md), [calibration](VISION_CALIBRATION.md) | [Data](../manifests/metal_nut_v1.json), [training](../manifests/efficientad_s_training_v1.json), [calibration](../manifests/efficientad_s_calibration_v1.json) |
| Metal-nut detection | [Evaluation](VISION_EVALUATION.md) | [PC](../manifests/efficientad_s_evaluation_v1.json), [Jetson batch](../manifests/jetson_holdout_batch_v1.json): 87 TP, 20 TN, 2 FP, 2 FN |
| Type/severity | [Type](DEFECT_TYPE.md), [severity](DEFECT_SEVERITY.md), [grade](FORGEGRADE.md) | [Four-type profile](../manifests/mvtec_four_type_v1.json), [severity](../manifests/mvtec_defect_severity_v1.json); reused subtype test, no physical-domain validation |
| Logs/operator UI | [Storage](STORAGE.md), [Jetson UI](JETSON_LIVE_UI.md), [PC UI](OPERATOR_APP.md) | [IDs/errors/restart](../manifests/jetson_inspection_ids_v1.json), [PC app](../manifests/operator_app_verification_v1.json) |
| ExecuTorch advisory | [Runtime](JETSON_RUNTIME.md), [ForgeAssist](FORGEASSIST.md) | [First generation](../manifests/jetson_llama_generation_v1.json), [persistent advice](../manifests/jetson_persistent_advice_success_v1.json) |
| Latest guidance | [Grounded advice](GROUNDED_ADVICE.md), [screw notes](SCREW_PLUGIN.md) | [Latest record](../manifests/jetson_hybrid_guidance_842c793_v1.json): 12/15 matching model choices, three fallbacks, 15 guarded outputs; unchanged worker, no station upgrade |
| Offline/recovery/performance | [Stage 10 ledger](STAGE10_VERIFICATION.md), [runtime report](MERGE_LATENCY_REPORT.md) | [Cold vision](../manifests/jetson_offline_vision_latency_v1.json), [cold advice](../manifests/jetson_offline_advice_latency_v1.json), [failure recovery](../manifests/jetson_offline_failure_recovery_v1.json); earlier snapshots, separate replay processes |
| Screw extension | [Screw plugin](SCREW_PLUGIN.md) | [CPU diagnostic](../manifests/screw_cpu_bundle_diagnostic_30k_v3.json), [training](../manifests/screw_efficientad_s_training_30k_v2.json); exposed samples, reliability gate failed |

The latest hybrid manifest's four source hashes match current files after LF
normalization: `forgesight/grounded_advice.py`, `forgesight/screw_guidance.py`,
`guidance/grounded_advice_v1.json`, `plugins/screw/guidance.json`. This binds those
files to the prompt check, not all current application modules to deployment.

Historical export/config records retain NOT_RUN/BLOCKED checkpoint fields.
Later device manifests supersede conclusions only within their tested scope.
Different Llama export hashes must not be mixed. Private device run directories
are references, not public downloadable logs.

## Evaluation coverage

| Requested measurement | Status |
| --- | --- |
| Precision, recall, F1, false alarms/misses | Frozen metal-nut benchmark recorded. F1 recomputes to 97.75% from TP=87, FP=2, FN=2; no physical-camera accuracy claim. |
| Per-image latency | Jetson CPU cold series and separate persistent observations recorded. |
| End-to-end latency | Replay-to-SQLite and advisory scopes exist separately. Supplied warm inspection and direct Llama timings are recorded in [JETSON_TIMINGS](JETSON_TIMINGS.md); their roughly 3.8 s sum is an estimate, not a camera-to-displayed-advice distribution. |
| FPS / parts throughput | No accepted application FPS or distinct physical parts/minute. Reciprocal forward latency does not fill this gap. |
| Memory | Per-process RSS, sampled system observations and a supplied resident dual-recipe station snapshot exist; simultaneous peak remains unestablished. |
| Power | Supplied report names a 25 W mode; no measured electrical consumption series is established. |
| Llama TTFT / tokens per second | Supplied direct-worker microbenchmark reports 28.18 decode tokens/s and 237 ms prefill/TTFT. Raw first-token timing and a repeated current-build throughput distribution are not independently verified. |
| Response time | Historical cold distributions, a first persistent request and supplied 1,622.65 ms warm direct-worker call exist. Different scopes/prompts prevent an equivalent speedup claim. |
| Advice quality | Contract/prompt checks include wrong replies and fallbacks. Operator usefulness and site QC approval are unverified. |

## Outstanding submission checklist

- [ ] Link the real demo video, publishing it if it is not yet accessible.
- [ ] Link or assemble the consolidated technical report, including both runtimes.
- [ ] Supply an authorized asset acquisition/handoff route and test a clean checkout.
- [ ] Revalidate the newest combined build from a disconnected cold start; existing saved-image offline proof is already verified.
- [ ] Meet the project's physical-part camera gate; screen photos remain diagnostics.
- [ ] Measure missing metrics or explicitly report them unavailable.
- [ ] Obtain fresh subtype/severity evidence; resolve screw reliability before stronger claims.
- [ ] Confirm team attribution, code license and distribution notices.
- [ ] Check portal rules absent from the guide: deadline, file formats and video limits.

No root code `LICENSE` was found. Dataset/model license references do not
license project code or establish every distribution permission. The existing
`assets/forgesight-station.png` was visually inspected: it is a project icon,
not an application screenshot or device proof.

## Documentation verification

Initial checks covered JSON parsing, selected source hashes, metric recalculation,
relative links/anchors, CLI option definitions, Markdown structure and whitespace.
The initial WSL attempt returned `Wsl/Service/E_ACCESSDENIED`; the later feature
audit successfully used the existing isolated interpreter as described below.
No packages, device files, datasets, inference outputs or existing evidence
manifests were changed. Push is withheld pending explicit user confirmation.

## Phase 1 feature catalogue follow-up

The user clarified that Phase 1 implementation is complete and that older docs
may omit features. The README catalogue was therefore checked against current
source, templates, JavaScript, recipe definitions and existing tests, rather
than relying only on stage summaries.

Newly expanded coverage includes image comparison/heatmap/contours, camera
controls, two preloaded part recipes, verified bundle switching, grounded
automatic advice, paginated history, editable/hideable review claims, HTML
inspection reports, GTK WebKit PDF export, storage health, rotating logs,
worker recovery/cleanup, desktop launch and model activation/rollback tools.

Scope corrections from source inspection:

- Current camera Analyze declares `physical_nut`; the backend supports
  `screen_photo`, but the current page has no separate source selector.
- Jetson-copy replay endpoints remain implemented, while the current station
  page presents Camera and Upload PNG tabs rather than a replay button.
- History delete controls conceal entries/claims; original audit data remains.
- Ordinary browsers download self-contained HTML reports. Native PDF saving
  belongs to the GTK WebKit desktop viewer; it is not a universal browser export.
- The plugin registry explicitly supports metal_nut and screw. More categories
  require registry integration, assets and validation, not just a new folder.
- Current operator claims and local checklists are foundations for ForgeLoop
  and ForgeAssist+; verified-label curation and approved SOP retrieval are future work.

Focused verification on 24 November 2026 used the existing isolated PC/WSL app
interpreter, with no installs or model/device inference:

```sh
FS_APP_PY=/root/.cache/forgesight/app-v1/venv/bin/python
"$FS_APP_PY" -B -m unittest tests.test_plugin_station tests.test_grounded_advice tests.test_grounded_live tests.test_storage_health tests.test_operation_log tests.test_jetson_desktop_launcher tests.test_screw_bundle -q
# 58 tests, OK

"$FS_APP_PY" -B -m unittest tests.test_jetson_live_ui tests.test_storage tests.test_ui_worker_lifecycle tests.test_worker_respawn_lifecycle tests.test_assist_queue -q
# 63 tests, OK
```

Both commands passed: 121 focused checks in total, not the full suite. These
exercise fixtures, mocks and local helper processes/HTTP services. Injected
failure logs and unclosed-file ResourceWarnings appeared; neither run reported
test failures. PDF rendering, a new browser session and Jetson hardware were
not exercised by these runs. Device claims continue to refer to their recorded
manifests. The future-work table follows roadmap Stages 12-17 and does not
present those extensions as already delivered.

Independent documentation review found no important issues. Its small
corrections were applied: include native grayscale screw PNG intake, and
reconcile the linked Jetson guide with the current Camera/Upload PNG interface
instead of its historical source selector.

## Supplied timing report follow-up

The user supplied `artifacts/jetson_timings_report.md`. The original report and
untracked collector remain unchanged. A portable, scope-qualified account is
saved in [JETSON_TIMINGS](JETSON_TIMINGS.md), with a
[structured transcription](../manifests/jetson_timings_summary_v1.json) and
the source SHA-256. No raw device JSON was present locally and no new benchmark
was run. This adds warm inspection, direct Llama throughput, preloaded recipe
switching and resident-memory observations to the README.

The source's cycle-time sum, unprofiled stage diagram, additive RAM pie,
no-thrashing/OOM assertions and factory-SOP wording are not adopted as measured
claims. Timer boundaries were checked against current source and the collector;
the unpinned deployed revision remains a provenance gap. The source's TTFT
label is attributed, not promoted to independently observed UI token latency.

Independent read-only review found no significant transcription or scope errors.
Its native token-count clarification was applied: the reported 39 is the
runner's decode-loop statistic, not an asserted complete output length.
