# Requirement evidence

No challenge requirement is marked satisfied by repository setup alone.

Stage 2.7 adds real desktop text generation through ExecuTorch's native C++
runner, using the recorded PTE and tokenizer. The transcript is in
`manifests/llama3_2_1b_pc_smoke.json`. This is a functional PC smoke test;
on-device advisory integration and quality evaluation remain open. All device,
quality, performance, and offline acceptance requirements below remain open.

Stage 3.1 selects EfficientAD-S using clearly labeled untrained architecture
timings and resource trade-offs, not accuracy results. The model has not been
fitted/trained. No dataset or holdout pixels were read by the probes. No anomaly
score/map from this step is evidence of a working detector, defect type,
severity, or verified edge performance. See `docs/VISION_BASELINE.md`.

Stage 3.2 supersedes the no-training snapshot: the first EfficientAD-S baseline
completed 10,000 real FP32 optimizer steps and its final checkpoint reloads with
matching state/output. The teacher stayed frozen and both trainable networks
changed. Validation/development/holdout pixels remained unused; calibration,
threshold, accuracy, defect type and severity are still unverified. Training
loss reduction is not acceptance evidence for those requirements. See
`manifests/efficientad_s_training_v1.json` and `docs/VISION_TRAINING.md`.

Stage 3.3 adds map normalization and a frozen normal-only candidate threshold.
It uses 22 map-fit plus 22 separate threshold normals, with zero holdout reads
and unchanged trained network weights. Three map-fit normals are flagged and
zero threshold normals are flagged by construction; neither is independent
accuracy evidence. Model-only desktop timing and successful bundle reload do
not verify defect recall, production false alarms, final grades, or Jetson
performance. All 86 tests pass. See `docs/VISION_CALIBRATION.md`.

Stages 3.4?5 supersede the earlier unevaluated snapshots for desktop benchmark
work. The frozen baseline detected 87/89 anomalies with 2/22 false alarms.
Real replay verified capture gating, graded JSON records and explicit errors.
Unknown anomaly types remain REVIEW. REJECT geometry rules are synthetic policy
tests; no actual type classifier or industrial acceptance policy is available.
Two missed scratches still get provisional PASS in a cached rule diagnostic.
Useful Llama advisory integration remains open; deterministic hold/check action
text is not generated advice. See `docs/FORGEGRADE.md`.

Stage 6 adds actual SQLite storage and separate-process restart queries for
full graded/error records. Original content and part IDs cannot be silently
replaced; exact retries are idempotent. One synthetic unverified correction
note demonstrates storage, not an actual operator finding. History summaries
count original decisions. No new model/threshold/holdout work, network service,
camera, Jetson or power-loss acceptance claim follows. See `docs/STORAGE.md`.

Stage 7 adds a real local operator workflow. Browser Inspect clicks ran the GPU
model and saved one PASS/one REVIEW; duplicate submission, refresh, filtering
and full server restart retained the same two records. The screen shows real
images, unknown type and pending advice. All 119 tests pass. Browser requests
were local, but network disconnection and Jetson execution remain unverified.
No new holdout access/model tuning occurred. See `docs/OPERATOR_APP.md`.

The [organizer clarification shared by the user](ORGANIZER_CLARIFICATION.md)
accepts the Jetson Orin platform family for PS5, subject to the mandatory
technologies. It clears the platform-eligibility question; actual device
inventory, runtime, and offline compliance still require measured evidence.

| Requirement | Status | Acceptance evidence needed |
| --- | --- | --- |
| Jetson Orin platform-family eligibility for PS5 | CONFIRMED in organizer reply shared by user | Recorded clarification; exact device inventory and deployment remain separate gates |
| Real surface anomaly detection | VERIFIED on PC benchmark only | Frozen holdout: 87/89 defects, 2/22 false alarms; see VISION_EVALUATION.md. Camera/edge remain open |
| Defect type | NOT VERIFIED | Same-domain labeled evaluation; retain `unknown` when unsupported |
| Severity and operator decisions | PC PIXEL-PROTOTYPE VERIFIED; industrial severity open | Versioned rules, 101 tests, real replay; unknown types REVIEW, no physical tolerances |
| Useful operator recommendations | NOT VERIFIED | Actual evidence-linked advice with uncertainty and next checks |
| Every-inspection history | PC SQLITE REPLAY VERIFIED; edge/failure recovery open | Original graded/error records survive writer exit and fresh-process queries; 4 retries add no rows; see STORAGE.md |
| One event per physical part | REPLAY + PERSISTENT ID GATE VERIFIED; physical proof open | 9 frames -> 4 declared-part records; known IDs blocked after restart; actual physical identity/camera unverified |
| Exact eligible edge hardware | BLOCKED - NOT VERIFIED | Device-side module/RAM/architecture/JetPack inventory; device currently unavailable |
| Llama-family inference via ExecuTorch on Jetson | BLOCKED - NOT VERIFIED | Actual locally generated text plus model/runtime/artifact provenance and measurements |
| Physical camera validation | BLOCKED - NOT VERIFIED | Controlled real samples captured and inspected on Jetson |
| Offline edge operation | BLOCKED - NOT VERIFIED | Disconnected cold start and real end-to-end device run |
| Real-time performance | NOT VERIFIED | Device vision/advice latency, distinct parts/minute, RAM and thermal observations |
| Technical report documents hardware, software, deployment, and inference flow | NOT VERIFIED | Actual device configuration, versions/hashes/commands, pipeline description, and observed inference evidence |
| Reproducible source/application/report/video | NOT VERIFIED | Reproducible setup, actual application, report, demo and repository delivery |

HailoRT is not applicable to the proposed Jetson route without Hailo hardware.
No calibrated millimeters, certified industrial grades, PDF performance,
factory root causes, or actual offline compliance are claimed.

Development-time access to GitHub or documentation is separate from the later
offline runtime test. Dataset/model licensing and redistribution conditions
must be checked before acquisition and publication of those artifacts.

## Dataset review evidence

Stage 1.1 documented the MVTec AD source, noncommercial license scope,
attribution, expected image/mask layout, and evaluation partition policy in
`docs/DATASET.md`. Review, acquisition, actual image/mask validation, and hashed
extraction are verified. The frozen split, role-based loader, and development
previews are verified on the PC. The project holdout excludes four
development images from the original test set and must be reported accordingly.
The exact archive URL was not captured; the user confirmed the publisher's
download flow and the archive fingerprint is recorded. No requirement above
is satisfied merely by acquiring or validating benchmark files.

Stage 1 gate evidence: 176 fit, 44 normal validation, four development, and 111
final-holdout samples; zero exact byte/pixel duplicate image groups; 36 passing
tests; real image/mask/overlay contact sheet inspected. Preview ground truth is
explicitly labeled, and zero final-holdout images were displayed. This completes
data/input preparation, not model detection, type classification, severity,
Jetson runtime, or offline compliance.

## Stage 2 source/preflight evidence

Model/runtime candidate pinned and sources reviewed in `docs/EDGE_RUNTIME.md`.
The initial review recorded no Meta Llama access (cleared in Substep 2.4).
Actual PC preflight identified x86_64 and
refused a device-only operation. No weights, model export, runner build,
generated text, device timing, or disconnected execution is verified.

Source-pinned recipe generation and device-only capture guards are now tested.
All 52 project tests passed, including generic process log/failure/timeout tests.
A real PC capture attempt recorded no runner process and no generated text.
Recipe/harness readiness is separate from actual export/runtime proof. A
successful capture remains pending review; neither source tags, artifact
filenames, stdout, nor an operator network declaration automatically pass the
Llama/ExecuTorch or offline requirements.

Substep 2.4 verified all five acquired original model files against the pinned
publisher revision. User-confirmed access and local source assets are now
verified; fingerprint metadata is committed and all model bytes stay ignored.
All 58 tests passed. This clears the source-asset blocker only: no model export,
generated text, native target build, device execution, or offline proof exists.

Substep 2.6 supersedes the earlier no-export snapshot: the actual PTE is
802,247,040 bytes, with hash/provenance in `manifests/llama3_2_1b_export.json`.
The artifact verifier loads its program metadata through the real pinned
ExecuTorch runtime; no forward or token-metadata method is executed. XNNPACK
delegate presence establishes serialization, not successful operator execution,
ARM portability, generated content, quantized quality, or offline edge proof.
All 63 tests passed. The Stage 2 device compliance gate remains open.

Substep 2.7 supersedes the no-generation snapshot: actual desktop generation
succeeded after registering wheel-provided quantized and LLM custom kernels.
The runner used local files and generated a general lighting explanation.
No cloud runtime, Ollama, alternate model, or fake answer was used. Offline
environment flags are not a network-isolation test. Both the first failure
and successful evidence are preserved; all 67 tests pass. This supplies no
Jetson, defect-inspection, calibrated severity, or advisory-quality proof.
