# Requirement evidence

No challenge requirement is marked satisfied by repository setup alone.

Stage 10.7 measured eight repeated, fresh-process Llama advisories through
ExecuTorch while Jetson Wi-Fi was disconnected. Every note passed the local
evidence contract; the saved REVIEW inspection and SQLite file were unchanged.
Native process p50/p95 and memory/thermal observations are recorded separately
from vision. These repeated prompts are not operator-usefulness evidence or a
physical camera event. See `manifests/jetson_offline_advice_latency_v1.json`.

Stage 10.6 measured 20 cold-process saved-image replays while the Jetson's
Wi-Fi was disconnected. All 20 persisted in fresh SQLite and matched exact
readbacks; model-forward and replay-to-SQLite p50/p95, process RSS, system
memory and thermal sensor observations are in the manifest. The two images
were each repeated ten times, so the results do not measure distinct physical
parts per minute, physical false positives or production camera throughput.
See `manifests/jetson_offline_vision_latency_v1.json`.

Stage 10.5 tested an unavailable camera index followed by actual C270 capture
and frozen-model diagnostic while the Jetson's Wi-Fi was disconnected. The
failure created no frame or grade; recovery saved a live frame and map. The
frame was visually a room scene with no declared nut, so it was explicitly
not an inspection and added no SQLite inspection row. Eleven route/interface
samples supported the disconnected observation. This is not unplug/replug
proof, camera-domain accuracy or real-part acceptance. See
`manifests/jetson_offline_camera_recovery_v1.json`.

Stage 10.4 tested two controlled failure inputs on the disconnected Jetson.
A nonexistent model path caused an ERROR inspection with deterministic REVIEW
and a persisted SQLite row; an invalid image copy failed the manifest hash
check before image decoding and created no inspection row. A valid replay then
produced PASS with both accepted rows intact. The working model and dataset
image were not modified. Eleven interface/route samples showed no upstream
route during the probe. This does not test actual camera failure, decoder-
level image corruption, physical parts or complete offline acceptance. See
`manifests/jetson_offline_failure_recovery_v1.json`.

Stage 10.3 exercised the Jetson's saved-replay operator page over loopback
HTTP with Wi-Fi disconnected. The local client received HTML/CSS/JavaScript,
PENDING/BUSY states, a real ExecuTorch note and a separate native TIMEOUT that
showed ADVICE_UNAVAILABLE. UI process restarts preserved the original REVIEW
grade and stored advice state. Thirty-three route/interface samples supported
the disconnected run. No local browser rendered the page, and no new image
inference, physical camera event or full application cold start occurred. See
`manifests/jetson_offline_loopback_ui_v1.json`.

Stage 10.2 ran one fitting normal reference and two previously exposed
development images through frozen EfficientAD, ForgeGrade and SQLite on the
Jetson while Wi-Fi was disconnected. Forty-six interface/route samples at
about one-second intervals showed no Wi-Fi/ethernet IPv4 or default route;
one ExecuTorch advisory also completed during the interval. The normal
reference received PASS; both anomaly candidates received REVIEW because
subtype classification is unvalidated. Three copied SQLite records matched
their JSON files exactly. The samples do not prove the absence of a brief
between-sample link, and these images do not provide new accuracy, physical
part identity, full UI or end-to-end camera evidence. See
`manifests/jetson_offline_case_series_v1.json`.

Stage 10.1 ran one saved MVTec development image through frozen EfficientAD,
ForgeGrade, SQLite and ExecuTorch Llama on the physical Jetson while its Wi-Fi
was disconnected. Network snapshots before and after each inference step showed
no upstream route or Wi-Fi/ethernet IPv4 address; a timed recovery watchdog was
scheduled, Wi-Fi later reconnected, and the timer was canceled. This is a
single replay pilot, not the required physical camera, normal/defective/uncertain
case set, full UI flow, p50/p95, throughput or complete offline acceptance.
The nested native runner still contains a historical default offline field;
the external network report is the evidence for this run. See
`manifests/jetson_offline_saved_replay_pilot_v1.json`.

Stage 9.4 links the bounded queue to a loopback status page running on the
physical Jetson. Browser proof through an SSH tunnel observed a real
ExecuTorch-generated note and a separate native `TIMEOUT` that left advice
unavailable. Both paths kept the original `REVIEW` inspection unchanged; the
page stayed responsive to status reads while Llama ran. This is saved-image
replay with connected hardware, so physical camera inspection and full offline
operation remain unverified. See `manifests/jetson_advice_ui_replay_v1.json`.

Stage 9.3 displays the copied Jetson saved-replay advisory in a local PC
browser as a separate, read-only panel. The API checks the report against the
immutable inspection and decision hashes. The browser showed the real note
beside the unchanged `REVIEW` grade and fixed action; no external requests or
page errors occurred. This is not a live Jetson UI/camera event, physical-nut
inspection, or disconnected offline proof. See
`manifests/operator_advice_display_v1.json`.

Stage 9.1 generated one real, strictly validated advisory on the Jetson from
an already persisted saved-image inspection, using Llama 3.2 1B through
ExecuTorch/XNNPACK. The separate payload cites `[E1][G1]` and keeps the
deterministic `REVIEW` grade unchanged. Missing runtime produced no advice and
left SQLite intact. This is a connected-device, replay-only proof; live
physical-nut advice, live Jetson UI delivery, useful
factory-approved procedures and disconnected operation remain open. See
`manifests/jetson_forgeassist_replay_v1.json`.

The Jetson now has one real saved-image EfficientAD/ForgeGrade/SQLite replay
with a stored anomaly map, overlay and exact fresh-process record readback.
It used an already exposed development-preview nut image, so it adds no
accuracy result. Source mode is `SAVED_IMAGE_REPLAY` and physical-part status
is false. Live physical-nut inspection, camera alert/UI, Llama advice and
disconnected end-to-end operation remain open. See
`manifests/jetson_saved_nut_replay_v1.json`.

The Stage 8 camera-to-model diagnostic reached the frozen EfficientAD export
and saved its map/overlay from a live C270 frame. The view contained no
declared nut, so it was explicitly marked `is_inspection=false`; no grade,
alert or SQLite inspection row was claimed. A physical-part inspection and
its measured end-to-end result remain the Stage 8 proof gate. See
`manifests/jetson_camera_model_diagnostic_v1.json`.

Stage 8.2 verified one hash-checked frozen EfficientAD-S image inference on
the Jetson CPU, using a benchmark training image. It produced a finite score
and anomaly map; the one-run forward and process RSS are recorded in
`manifests/jetson_efficientad_single_image_v1.json`. No live part was inspected,
so actual camera defect detection, accuracy, throughput, alert/log integration,
combined Llama operation and disconnected end-to-end operation remain open.

Stage 8.1 verified camera format negotiation and one live 640x480 frame read on
the actual Jetson. The frame was not saved and no physical component or vision
model was evaluated. The isolated Llama runtime lacks the frozen EfficientAD
model's vision dependencies and has CPU-only Torch. Thus camera access is
verified, while on-device defect detection, camera alerts, integrated advice
and disconnected operation remain open. See
`manifests/jetson_camera_vision_preflight_v1.json`.

Substep 2.10 adds the required **actual on-device Llama-family execution through
ExecuTorch** for one connected Jetson smoke prompt. The 802 MB PTE/tokenizer
matched pinned hashes after transfer; the ARM64 native runner returned 40 new
tokens with matching ExecuTorch statistics. Device model load, inference, total
process time, sampled memory and two-thread setting are recorded in
`manifests/jetson_llama_generation_v1.json`. This meets the Stage 2 functional
generation gate for one prompt. It does not establish useful industrial advice,
camera/vision integration, a latency distribution or disconnected operation.
The prior checkpoint descriptions below are historical.

Latest device checkpoint, Substep 2.9: isolated ARM64 Python/CPU Torch/ExecuTorch
installation, dependency checks, native imports and operator registration passed
on the actual Jetson. System Python is unchanged. No model was loaded or run;
this closes the SDK setup prerequisite, **not the Llama-generation requirement**.
See `manifests/jetson_runtime_installation_v1.json`. The earlier entries below
describe their own checkpoints rather than the latest installation state.

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

Stages 3.4 through 5 supersede the earlier unevaluated snapshots for desktop benchmark
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

Stage 9.1 supplies a real source-linked desktop Llama/ExecuTorch note for one
saved anomaly/REVIEW record. It repeats supplied evidence/checklist facts only.
Six content/format failures and a seventh validator false positive are retained;
the identical final source-faithful output passes the corrected vocabulary
check. Broad advice usefulness is NOT VERIFIED. Guidance is prototype-only;
causal explanations are withheld. No UI/background integration, actual Jetson
or offline proof follows. See `docs/FORGEASSIST.md`; all 128 tests pass.

The [organizer clarification shared by the user](ORGANIZER_CLARIFICATION.md)
accepts the Jetson Orin platform family for PS5, subject to the mandatory
technologies. It clears the platform-eligibility question; actual device
inventory, runtime, and offline compliance still require measured evidence.

Stage 9.2 adds explicit input-refusal reports without Llama calls, canonical
guidance identity checks and two reviewed output statement forms. A handwritten
false-cause fixture exposed the previous vocabulary-only weakness; it is now
withheld. All 27 input/output contract checks matched and 132 software tests
passed. Two actual PC native runs produced the same source-linked note from one
unique prompt. Timeout, missing runtime and incomplete generation kept advice
unavailable. The original protocol remains 31/32 matching expectations because
one-token failure happened at native validation before the anticipated parser;
a separately declared eight-token probe verified parser refusal. No independent
general advice-quality, causal diagnosis, approved factory SOP, ARM deployment,
background/UI responsiveness or offline edge claim follows. See
`manifests/forgeassist_refusal_verification_v1.json` and `docs/FORGEASSIST.md`.

| Requirement | Status | Acceptance evidence needed |
| --- | --- | --- |
| Jetson Orin platform-family eligibility for PS5 | CONFIRMED in organizer reply shared by user | Recorded clarification; exact device inventory and deployment remain separate gates |
| Real surface anomaly detection | VERIFIED on PC benchmark only | Frozen holdout: 87/89 defects, 2/22 false alarms; see VISION_EVALUATION.md. Camera/edge remain open |
| Defect type | NOT VERIFIED | Same-domain labeled evaluation; retain `unknown` when unsupported |
| Severity and operator decisions | PC PIXEL-PROTOTYPE VERIFIED; industrial severity open | Versioned rules, 101 tests, real replay; unknown types REVIEW, no physical tolerances |
| Useful operator recommendations | ONE PC CHECKLIST NOTE VERIFIED; broad usefulness open | Actual ExecuTorch note cites real record/prototype checklist; no factory diagnosis, approved SOP, UI or device proof |
| Every-inspection history | PC SQLITE REPLAY VERIFIED; edge/failure recovery open | Original graded/error records survive writer exit and fresh-process queries; 4 retries add no rows; see STORAGE.md |
| One event per physical part | REPLAY + PERSISTENT ID GATE VERIFIED; physical proof open | 9 frames -> 4 declared-part records; known IDs blocked after restart; actual physical identity/camera unverified |
| Exact eligible edge hardware | AUTHENTICATED INVENTORY VERIFIED | Orin Nano 8GB module identifiers, ARM64, RAM, Ubuntu/L4T, SD storage and Logitech C270 USB/video nodes recorded; capture/inference are separate open gates |
| Llama-family inference via ExecuTorch on Jetson | VERIFIED for one connected-device prompt | Llama 3.2 1B generated 40 real tokens through native ExecuTorch; hashes, statistics and single-run timing/RSS saved in jetson_llama_generation_v1.json. Advisory integration and offline run are separate gates |
| Physical camera validation | BLOCKED - NOT VERIFIED | Controlled real samples captured and inspected on Jetson |
| Offline edge operation | BLOCKED - NOT VERIFIED | Disconnected cold start and real end-to-end device run |
| Real-time performance | NOT VERIFIED | Device vision/advice latency, distinct parts/minute, RAM and thermal observations |
| Technical report documents hardware, software, deployment, and inference flow | NOT VERIFIED | Actual device configuration, versions/hashes/commands, pipeline description, and observed inference evidence |
| Reproducible source/application/report/video | NOT VERIFIED | Reproducible setup, actual application, report, demo and repository delivery |

HailoRT is not applicable to the proposed Jetson route without Hailo hardware.

The 2026-11-11 authenticated inventory closes the device-identification gate
only. Strict SSH identity checks were retained; no credentials are stored in
the project. The audit script ran on the actual ARM64 Python 3.8.10 interpreter,
with lossless JSON transport checked by SHA-256. No installation, runtime/model
execution, frame capture or disconnected test occurred. The 35.6.1 release maps
to the JetPack 5.1.5 family, but the JetPack metapackage is absent. Power-mode
query errors and missing camera-format tooling remain explicit. The next task
is a compatibility recipe for isolated modern Python/native ExecuTorch on this
userspace, not a claim that current PC environments run on Jetson. See
`manifests/jetson_device_audit_v1.json` and `docs/HARDWARE.md`.

Substep 2.8 adds static ARM library checks and a pinned candidate recipe, not
runtime proof. The core wheels' 33 ELF files match ten authenticated target
library/version inventories. The target-specific lock resolves 49 packages;
the sole source-only package built as a pure-Python wheel on the PC. All 140
software tests pass. No environment/package/model was installed on Jetson.
The pinned interpreter archive initially failed to download. A later authorized
retry passed its SHA-256/size check and statically inspected its ten ARM64 native
files. The combined 43 files match 12 observed target system libraries. This
closes the static-preparation prerequisite only; no new interpreter was executed
or installed on Jetson, and no claim of native imports, Llama execution, memory
fit or offline compliance follows. See `docs/JETSON_RUNTIME.md`,
`manifests/jetson_python_archive_v1.json` and
`manifests/jetson_runtime_compatibility_v1.json`.
Substep 2.9 supersedes that no-installation checkpoint: 51 distributions are
installed in the separate ARM64 CPU venv, `pip check` passed, CPU arithmetic
executed and the native runner/tokenizer bindings loaded. XNNPACK and all three
required operators are registered. Exact system snapshots match before/after;
147 PC tests pass. The setup used only staged hashed files and a local package
index policy, but the network was not disconnected for acceptance testing.
No model/tokenizer weights were transferred; no Llama response, GPU vision,
physical-camera inspection, combined resource fit or offline compliance follows.
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
