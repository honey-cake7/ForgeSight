# ForgeSight AI — Dependency-Based, Learn-As-You-Build Roadmap

**Project:** https://github.com/honey-cake7/Forge_Sight.git  
**Machine:** NVIDIA Jetson Orin Nano (exact module/RAM/JetPack/camera must be verified on-device)  
**Primary objective:** Complete a genuine, offline inspection system for one controlled manufactured component using real vision inference, deterministic severity, operator alerts, local inspection history, and a Llama-family model actually running through ExecuTorch on the edge device.

## How to use this roadmap

This is **not a calendar**. Every stage depends on the prior acceptance gate. At each stage, Codex must:

1. **Explain the idea:** what this solves, the concept, why chosen over alternatives, and what remains unknown.
2. **Show a tiny plan:** one small implementation task, files to change, commands and expected behavior.
3. **Implement one verified unit of work:** don't write the entire stage in one turn.
4. **Prove it works:** run tests, show actual outputs, and label anything untested as `NOT VERIFIED`.
5. **Make a focused Git commit** after a meaningful tested change, using a clear factual message. Push to the configured remote only when authorized. Never pretend to have run commands or pushed successfully.
6. **Teach and summarize:** explain the important functions/files, one or two comprehension checks, and the *next exact step*.
7. **Stop at the verification gate** so I can review before advancing to another stage.

Track status in `docs/STATUS.md`, decisions in `docs/DECISIONS.md`, hardware in `docs/HARDWARE.md`, compliance in `docs/COMPLIANCE.md`, and learning notes in `docs/LEARNING_LOG.md`. Distinguish PC prototype, Jetson prototype, and tested Jetson offline operation.

## Minimum working path — Phase 1

### Stage 0 — Audit and protect the workspace
**Learn:** Git local vs remote, environment reproducibility, CPU/GPU/device capabilities.

**Do:** Inspect local files, git status, branches, remotes, untracked and ignored files before modifying anything. Inspect the actual PC OS, GPU, Python, disk, and installed packages. Inspect Jetson separately when physically available (model, RAM, JetPack, Linux AArch64, camera). Check the GitHub repository URL. Decide how to connect/push without overwriting local work. Add essential documentation, `.gitignore`, and a minimal repeatable development environment only after the audit.

**Proof:** `git status`, repo inventory, working Python environment, documented hardware facts; clean repository initialized/connected safely without data loss.

**NEXT:** Set up a small real input-to-output vision experiment and investigate ExecuTorch feasibility in parallel. Sample commit: `chore: document environment and initialize project safely`.

### Stage 1 — Learn the data and make a reproducible inspection input
**Learn:** anomaly detection vs classification, images vs masks, train/validation/test, leakage, why fixed light and consistent part positioning matter.

**Do:** Choose a single component domain (start with MVTec AD `metal_nut` as a dataset prototype). Inspect license, directory layout, defect categories, example normal and defective images, and anomaly masks. Keep datasets out of Git. Build an image loader and a visualization script that displays images/masks and saves diagnostic previews. Decide the later physical component/fixture separately; do not claim benchmark images equal the Jetson camera domain.

**Proof:** Real samples load; example labels/masks and counts are printed; splits and legal usage documented; no raw dataset committed.

**NEXT:** Establish the actual edge Llama runtime feasibility (Stage 2), and then train/evaluate a vision baseline. Sample commit: `feat(data): add inspection dataset loader and sample visualization`.

### Stage 2 — Prove mandatory Llama + ExecuTorch on ARM64 early
**Learn:** model weights, tokenizers, on-device runtimes, export/quantization, ARM CPU vs Jetson GPU, why a `.pte` artifact alone is not inference.

**Do:** Check Jetson variant, memory, JetPack, installed toolchain, ExecuTorch version/docs, model license and access. Select the smallest feasible Llama-family candidate (e.g. Llama 3.2 1B Instruct, if permitted and supported). Establish a reproducible export/runner setup and run an actual prompt through a local ExecuTorch runner on **the Jetson**. Record model ID, artifact hash, version, exact command, memory, actual generated text, latency, and internet state. Do not silently substitute llama.cpp, Ollama, or hosted APIs. Keep vision running separately from this worker.

**If hardware/weights are inaccessible:** mark `BLOCKED — NOT VERIFIED` and keep subsequent PC-side tasks moving. This gate must be completed before claiming full submission compliance.

**Proof:** The edge device actually generates non-placeholder text from the model through ExecuTorch; reproducible device-side record saved.

**NEXT:** Implement the real vision baseline (Stage 3). Sample commit: `feat(edge): add verified ExecuTorch Llama runner` only if verified; otherwise `docs(edge): record ExecuTorch setup and unresolved device blocker`.

### Stage 3 — Build and evaluate one real vision model
**Learn:** anomaly scores, supervised classes, segmentation masks, threshold selection, validation/test separation, false positives and false negatives.

**Do:** Start with a simple compatible normal/anomaly baseline (PatchCore or EfficientAD; choose after testing installation/runtime), train/fit on the permitted training partition, tune threshold using validation data, and evaluate once on held-out test images. Produce real anomaly maps and saved model metadata. If actual defect-type labels can be used with a leakage-safe split, train/test a same-domain subtype classifier; otherwise output `unknown` or a clearly identified descriptive morphology, and explicitly track the unmet subtype classification requirement. Do not claim a model trained on NEU steel strips predicts categories for metal nuts.

**Proof:** A previously unseen image gets a real model-produced score/overlay; evaluation reports show confusion, missed defects and false alarms; a test script repeats the result.

**NEXT:** Turn raw predictions into inspection records (Stage 4). Sample commit: `feat(vision): add trained anomaly inference and evaluation`.

### Stage 4 — Define what counts as ONE inspection
**Learn:** frames vs parts, event-driven pipelines, schema design and state machines.

**Do:** Create a structured `InspectionRecord` with unique part/inspection ID, timestamp, image reference, model version, scores, regions, known class or `unknown`, and statuses. Build a testable inspect-one-image function. Implement a re-arm / new-part capture gate so repeated camera frames of a stationary part are not counted as new units.

**Proof:** One part produces one recorded event; repeated frames do not inflate counts; invalid inputs return explicit errors.

**NEXT:** Add deterministic severity and decisions (Stage 5). Sample commit: `feat(core): define inspection schema and one-part capture gate`.

### Stage 5 — ForgeGrade: deterministic decisions
**Learn:** model confidence vs severity, contour geometry, inspection zones, configurable policies and calibration.

**Do:** Measure anomaly area/length/bounding box in **pixels** initially. Define documented rules for `PASS`, `REVIEW`, `REJECT`, based on observed defect and optional configurable zones. Mark low confidence/unseen types as `REVIEW`. Store the policy/version alongside each decision. Convert to millimeters only after a camera calibration test shows verified scale and uncertainty. Llama must never override final safety/quality status.

**Proof:** Unit tests show identical inputs give identical grades; location/size rules work; an uncalibrated system never claims mm or certified industrial grades.

**NEXT:** Persist records locally (Stage 6). Sample commit: `feat(grade): add rule-based inspection decisions`.

### Stage 6 — Store every inspection in SQLite
**Learn:** local persistence, primary keys, database migrations and audit trails.

**Do:** Save `PASS`, `REVIEW`, `REJECT`, failures and corrections locally. Include source image path/reference, timestamps, model/rule versions, advice state, and provenance. Add API/query methods for history and summaries. Use DB migrations or versioned schema setup; avoid keeping large raw images directly in SQLite.

**Proof:** Query rows after app restart; distinct inspection counts stay correct; no duplicates; abnormal cases and clean cases are both present.

**NEXT:** Create the smallest useful operator workflow (Stage 7). Sample commit: `feat(storage): persist inspection events in SQLite`.

### Stage 7 — Implement local backend + operator screen
**Learn:** FastAPI endpoints, real-time updates, API contracts, UI state and error handling.

**Do:** Build a local API with health, inspect, latest result, history and configuration endpoints. Build a clean, minimal dashboard that shows the real source image, overlay, finding, decision, alert and latest records. Prefer a small working UI over polishing a fake analytics dashboard. Support sample-image/replay input before live camera integration.

**Proof:** Clicking Inspect runs real inference and writes a real SQLite row; refresh shows the persisted result; no hardcoded predictions are presented as live.

**NEXT:** Connect an actual camera or documented video source (Stage 8). Sample commit: `feat(ui): display real local inspection results`.

### Stage 8 — Connect the camera and stabilize the edge vision path
**Learn:** OpenCV camera capture, image registration, lighting, frame sampling, queue/backpressure and edge GPU deployment.

**Do:** On Jetson, validate USB/CSI camera and compatible vision dependencies. Build a simple fixed inspection rig and capture a part only when ready. Handle camera disconnection and lighting changes. Compare camera images to benchmark dataset; gather a small actual-device test set with distinct physical part IDs. Optimize model format only after measuring baseline speed. A replay mode is acceptable for development, but distinguish it visibly from live camera mode.

**Proof:** One live physical component yields an actual image, real inference, visible alert and saved result; timings recorded on the actual Jetson.

**NEXT:** Wire the real ExecuTorch Llama advisory into the inspection event (Stage 9). Sample commit: `feat(camera): add single-part Jetson capture pipeline`.

### Stage 9 — ForgeAssist: asynchronous, evidence-based advice
**Learn:** structured prompting, local procedure retrieval, hallucinations, timeouts and non-blocking workers.

**Do:** Pass compact structured `InspectionRecord` fields, deterministic severity, recent recurrence summary if available, and permitted local SOP text to the **real ExecuTorch Llama runner**. Produce a separate advisory payload containing possible explanations, evidence IDs, uncertainty and next verification step. Never invent a machine ID, factory root cause or an official standard clause. On timeout/failure, keep the inspection decision and visibly mark advice unavailable. Use a bounded queue so inspection continues while Llama generates.

**Proof:** A real defect event produces actual locally generated advice from ExecuTorch; timeouts do not crash the UI or stall capture; inference provenance is recorded.

**NEXT:** Perform complete end-to-end, offline and performance verification (Stage 10). Sample commit: `feat(assist): integrate asynchronous ExecuTorch advisory`.

### Stage 10 — Verify actual offline edge operation
**Learn:** performance benchmarking, p50/p95 latency, RAM, thermal pressure, reliability and reproducibility.

**Do:** Disconnect the device from the network, cold-start the system and run normal, defective and uncertain cases. Measure model latency and advisory latency separately; record distinct parts per minute, false positives and memory/thermal observations. Test camera failure, missing model, corrupt input, restart and recovery. Mark any unavailable acceptance evidence as incomplete.

**Proof:** Real Jetson offline demo with visible run logs and a table of measured metrics; all mandatory components actually execute locally; instructions reproduce the test.

**NEXT:** Assemble and audit the submission artifacts (Stage 11). Sample commit: `test(edge): add offline verification and measured benchmarks`.

### Stage 11 — Deliver the Phase 1 application and technical report
**Learn:** honest experimental reporting and reproducible public repositories.

**Do:** Audit license compliance; include README, exact setup commands, dependency lock/pins, tests, architecture diagram, documented shortcomings, real demo video, source code, application launch instructions, training/evaluation procedure and technical report. Explicitly distinguish PC experiments, Jetson measurements, benchmark-image results and physical camera results. Commit only permissible files; push to `https://github.com/honey-cake7/Forge_Sight.git` with the user's permission, using genuine incremental history.

**Proof:** A new teammate can follow the README, reproduce supported tests, identify hardware needs and watch a legitimate demo. GitHub actually displays the committed files; submission artifacts are ready.

**NEXT:** Begin the high-value extensions below only after the core system works.

## Differentiation path — Phase 2 (in priority order)

**Stage 12 — ForgeTrace:** Track defect rate and location/orientation patterns across **distinct parts**, batches and zones, with proper denominators. Build rolling windows, false-alarm safeguards and evidence-linked recurring-pattern alerts. **Prove:** seeded event histories show genuine recurrence and non-recurrence, not duplicate frames. **Then:** ForgeDual.

**Stage 13 — ForgeDual:** Combine a *same-domain* supervised known-defect detector/classifier with separately trained anomaly detector for unfamiliar appearances. Hold out some defect categories when evaluating unknown detection. **Prove:** known-only, anomaly-only, and fused precision/recall/false positives/latency, with clear data provenance. **Then:** ForgeGrade+.

**Stage 14 — ForgeGrade+:** Add calibrated physical geometry, defect orientation, critical zone profiles and policy review. **Prove:** calibration error measured; two similar defects in different zones receive appropriate deterministic outcomes. **Then:** ForgeAssist+.

**Stage 15 — ForgeAssist+:** Improve local SOP retrieval, source attribution, structured outputs, uncertainty and user usefulness. **Prove:** real operator scenarios evaluated against local source documents, with hallucinations counted. **Then:** ForgeLoop.

**Stage 16 — ForgeLoop:** Let operators confirm/correct results, record later-verified causes, queue images for labeling, and export curated training batches. **Prove:** full audit trail, no automatic promotion of model guesses to ground truth. **Then:** robustness and ablations.

**Stage 17 — Final engineering proof:** Benchmark glare, focus, unusual poses, drift, unknown defects, power/thermal reliability and multi-hour operation. Demonstrate the measured value of each differentiator with ablations and a polished *real* physical demonstration. **Prove:** comparative metrics, reproducible report and honest limitations.

## Gate rules

- A stage is not `DONE` because code exists; it is `DONE` only when its stated proof is observed.
- If Jetson is inaccessible, continue PC-only work while keeping `ExecuTorch on Jetson` and `offline edge test` explicitly `BLOCKED`, not `DONE`.
- Do not claim type classification, millimeter measurements, real-time performance or root-cause certainty without corresponding evidence.
- The HailoRT instruction applies to Hailo-equipped Raspberry Pi setups, not this Jetson route.
- A camera/benchmark mismatch must be tracked as **domain shift**; never merge unrelated NEU steel and MVTec metal-nut scores into one product metric.
- If a stage becomes unexpectedly complex, split it into smaller verified tasks and maintain the same ordering of dependencies.

## Required Codex answer at every completed step

```
CURRENT STAGE / SUBSTEP:
WHAT YOU LEARNED (simple explanation):
WHY THIS DESIGN (alternatives/trade-offs):
FILES AND COMMANDS:
ACTUAL TESTS AND RESULTS:
WHAT IS STILL UNVERIFIED:
GIT COMMIT (hash, or why none):
ONE OR TWO CONCEPT CHECKS:
NEXT EXACT ACTION:
WAIT FOR MY REVIEW BEFORE NEXT STAGE:
```
