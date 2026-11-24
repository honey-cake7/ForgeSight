# Hackathon README Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan one documentation task at a time. Steps use checkboxes for tracking. This request covers planning only; README implementation and submission packaging are subsequent work.

**Goal:** Make ForgeSight understandable, reproducible and assessable from its GitHub landing page, with every submission claim linked to evidence.

**Architecture:** Use a concise README as the entry point for judges and developers. Keep experimental details in the existing technical documents and link the four required deliverables prominently. Separate current source behavior, historically tested device builds, benchmark evidence and unverified deployment claims.

**Tech stack:** Markdown, GitHub-compatible Mermaid, existing Python/WSL and Jetson workflows; no new application dependencies.

**Spec:** User-supplied *On-Device Quality Inspector Challenge* guide, sections 1-10, reviewed on 2026-11-24. Repository constraints: `AGENTS.md`, local roadmap and master engineering prompt. This plan summarizes the guide so it does not depend on a private attachment path.

## Global constraints

- This is a documentation plan, not authorization to install software, change Jetson runtimes, run hardware experiments or implement later roadmap stages.
- Target: NVIDIA Jetson Orin Nano 8GB, with fully local detection, advice and logging.
- Llama-family advisory inference must execute on-device through ExecuTorch. Vision may use a different runtime.
- Vision supplies findings; deterministic rules control the quality decision; Llama explains and advises without overriding that decision.
- Keep PC measurements, Jetson measurements, benchmark images, screen-photo diagnostics and physical components distinct.
- Report observed results with their sample count, build, measurement boundary and limitations. Do not promise FPS, accuracy or full submission compliance.
- Preserve existing work, including the unrelated untracked `scripts/record_timings.py` observed during this review.
- No README rewrite, application change, commit or push was performed by this planning task.

## Review focus

1. **Contradictory status:** README must not call Jetson Llama execution pending beside verified device evidence. Task 1 checks each claim against a specific build and source.
2. **Fresh clone cannot run:** datasets, models and databases are ignored; root `requirements.txt` is only an early Pillow environment. Task 3 checks asset acquisition, environment selection and all setup prerequisites.
3. **Misleading performance comparison:** cold, persistent, PC and Jetson timings have different scopes. Task 2 requires separate labeled rows; Task 4 checks their labels and evidence.
4. **Incomplete deployment presented as current proof:** newer classifiers/guidance/plugin source has not all been tested together on-device. Tasks 1 and 4 check source/deployment versions explicitly.
5. **Broken submission evidence:** local `artifacts/` and private device paths are inaccessible on GitHub. Tasks 3 and 4 require accessible deliverables and permitted media or explicitly unresolved status.

## What the supplied guide actually asks for

| Category | Guide requirement | What belongs in README |
| --- | --- | --- |
| Required deliverables | Runnable documented source; working target-device application; demo video showing detection, alerts and logging; technical report covering design, runtimes, results and limits | A prominent link or entry for each, plus reproducible setup and launch instructions |
| Core behavior | Detect anomalies; classify type; estimate severity; alert with actionable advice; structured inspection logging; edge operation without cloud inference | Feature/status table with implementation, tested scope, evidence and remaining gaps |
| Mandatory technology | Llama family for advisory; on-device ExecuTorch; vision detects/classifies and hands structured findings to Llama | Two-runtime architecture, exact model/runtime/backend configuration and hand-off description |
| Suggested evaluation | Detection quality, device latency/FPS, grounded advice, operator experience and rigor documenting both runtimes | Short measured-results table and links to detailed evidence |
| Benchmark checklist | Precision, recall, F1, FPS, per-image and end-to-end latency, memory, power, Llama TTFT, tokens/s and total response time | Cover every metric; publish measured values with scope, or state not measured |
| Scope and decisions | One inspection domain; documented severity; confidence is not severity; rule-based decisions allowed | Explain chosen metal-nut baseline, optional screw experiment, severity and PASS/REVIEW/REJECT policy |
| Permitted approaches | Static jig, live camera or recorded video; CPU-only allowed; vision runtime flexible; fine-tuning/Hailo/YOLO/OpenCV/conveyor not compulsory | Explain our actual choices without inventing extra organizer requirements |

No published FPS target, submission deadline, required README template, video length or report length appears in the supplied guide. Obtain event-specific portal rules separately if available. The guide permits non-conveyor demonstrations; our roadmap's physical-part acceptance gate is a separate, stricter project requirement.

## Recommended approach

Use a submission-focused README with reproducibility instructions and links to existing deep documentation. Keep the measured metal-nut baseline as the primary result; show the opt-in screw plugin as an experimental extension with separate data and limitations. This is a recommendation, not an assumption that the user selected a new submission scope.

Alternatives considered:

- A single exhaustive README would duplicate the existing logs and make the four deliverables hard to find.
- A short promotional README would hide missing assets, runtime requirements and incomplete acceptance gates.
- The recommended structure gives judges a quick overview and developers a route to the exact setup and evidence.

Target approximately 1,000-1,500 words, excluding command blocks and tables; keep training/export history and full experiment records in linked documents.

## Proposed README order

| Order | Section | Content and source |
| --- | --- | --- |
| 1 | Project identity | ForgeSight AI, one-sentence problem/solution, challenge name and team attribution when supplied. One authentic screenshot or short clip; label replay/diagnostic/live source. |
| 2 | Submission links | Demo video, technical report, source repository and target-device setup/launch guide. Do not publish invented links or mark missing artifacts complete. |
| 3 | What works today | Compact table for detection, classification, severity, alerts/advice, SQLite logging and offline operation. State tested build and current integration gaps. Link `docs/STATUS.md` and `docs/STAGE10_VERIFICATION.md`. |
| 4 | Problem and workflow | One component/inspection at a time: choose source, inspect, view response/decision, request advice, inspect saved history. Describe response overlays as model responses, not confirmed damage masks. |
| 5 | Architecture | Small Mermaid flow: camera/upload/replay -> EfficientAD -> structured inspection -> ForgeGrade -> SQLite/UI; bounded asynchronous advisory -> on-device Llama/ExecuTorch -> separate advice state. Mark optional classification and experimental screw recipe. |
| 6 | Hardware and both runtimes | Orin Nano 8GB, C270, observed Linux/L4T/JetPack evidence, storage, vision format/backend and independent advisory environment/model/quantization. Use `docs/HARDWARE.md`, `docs/JETSON_VISION.md`, `docs/JETSON_RUNTIME.md`, `docs/EDGE_RUNTIME.md`. Do not infer a JetPack version from an unverified mapping. |
| 7 | Data and models | MVTec AD metal_nut; normal-only EfficientAD fitting; frozen splits/threshold; separately labeled provisional subtype results. Explain pretrained Llama structured prompting rather than implying fine-tuning. Link training/calibration/evaluation and data provenance. |
| 8 | Quick start | Clone and enter the repository, obtain permitted assets, select the correct environment, verify assets, start the supported app, open local URL and perform one inspection. Separate PC/WSL and Jetson instructions. State whether each path was reproduced from a clean checkout. |
| 9 | Operator behavior and records | Explain source modes, pixel-only severity, decision vs advisory, unknown/unassessed outcomes, advice timeout/fallback and one inspection ID per event. Include a small example derived from an actual saved record and identify its provenance. |
| 10 | Measured results | Small detection table plus device performance table. Each row identifies model/profile, dataset/split, build, n, device, timing scope and evidence link. Include false alarms and misses. |
| 11 | Reproduce and verify | Correct environment-specific test command, training/evaluation links, frozen-asset hash checks and device verification links. Summarize cached predictions without silently rerunning exposed holdouts. Avoid hard-coded test counts. |
| 12 | Limitations and remaining work | Physical-camera domain shift, incomplete full offline/cold-start acceptance, provisional subtype/severity, screw reliability gap, deployment gap, no confirmed GPU inference speedup, and unmeasured metrics. Keep roadmap short. |
| 13 | Repository map | Brief purpose of `forgesight/`, `configs/`, `guidance/`, `plugins/`, `requirements/`, `scripts/`, `tests/`, `manifests/`, `docs/`. State why datasets/weights/generated records are absent. |
| 14 | Attribution and licenses | Dataset attribution and documented noncommercial/share-alike terms; Llama license/access obligations; EfficientAD/ExecuTorch and reused code citations; actual project code-license status; team/acknowledgments. Do not assign a license on the owner's behalf. |

## Evidence we can use, with qualifications

These are findings from repository documents, not experiments rerun during this planning review.

| Topic | Recorded evidence | How to present it |
| --- | --- | --- |
| Metal-nut anomaly detector | Frozen 111-image holdout: 87/89 defects flagged; 20/22 good unflagged; two misses and two false alarms. Precision/recall 97.75%, accuracy 96.40%. Jetson reproduced candidate decisions. | Label as saved MVTec benchmark results. If publishing F1, explicitly recompute it from TP=87, FP=2, FN=2; do not substitute classification accuracy. Sources: `docs/VISION_EVALUATION.md`, `manifests/jetson_holdout_batch_v1.json`. |
| Provisional metal-nut classification | Four-type profile: 17/19 defective test images labeled correctly, two unknown; five good images unflagged; PC/Jetson agreement recorded. | Opt-in, repeatedly evaluated 24-image subtype test; not fresh independent validation or physical-camera type accuracy. Source: `docs/DEFECT_TYPE.md`. |
| Severity and grade | Deterministic pixel-response rules and optional type-conditioned severity; unknown/orientation cases can be Unassessed. | Distinguish displayed prototype severity, optional assessment metadata and authoritative grade; no calibrated mm or certified damage size. Sources: `docs/FORGEGRADE.md`, `docs/DEFECT_SEVERITY.md`, current `docs/STATUS.md`. |
| Offline CPU vision series | n=20: forward p50/p95 1,585/1,684 ms; replay-to-SQLite 7,499/7,677 ms; peak child RSS p95 437,163 KiB. | Historical cold saved-image processes repeating two images; not current persistent throughput or physical parts/minute. Source: `docs/STAGE10_VERIFICATION.md`. |
| Offline Llama series | n=8: native process wall p50/p95 20,690/20,942 ms; sampled peak child RSS p95 1,189,735 KiB. | Historical fresh-process advisories from one inspection; separate from vision, not combined system peak or TTFT. Source: `docs/STAGE10_VERIFICATION.md`. |
| Persistent runtime snapshot | At `ca499bd`, one saved-image inspection took 2.337 s excluding SQLite/UI/camera/advice; first advisory worker request took 5.311 s including reported model load. | Different boundaries and prompts from cold series; not warm percentiles, percentage speedup or proof of the latest main build offline. Source: `docs/MERGE_LATENCY_REPORT.md`. |
| Current hybrid guidance | Existing physical Jetson worker: 12/15 revised prompt replies matched allowed checks; three needed pinned checklist fallback. PC guards accepted all 15 guarded outputs. | Report raw Llama outcomes separately from guarded system behavior. Station upgrade/restart and whole-build deployment remain pending. Source: `manifests/jetson_hybrid_guidance_842c793_v1.json`. |
| Screw extension | Reused-image diagnostic: 26/29 defects detected; classifier 19/25 correct type calls, six wrong calls, one unknown and three detector misses. | Experimental opt-in recipe, separate benchmark, failed reliability gate; no blended nut/screw accuracy claim. Source: `docs/SCREW_PLUGIN.md`. |

FPS, distinct-parts throughput, complete camera-to-advice latency, total simultaneous peak memory, power, TTFT and a comparable current-build tokens/s series need explicit evidence review. Mark unsupported entries not measured. Do not turn reciprocal model-forward latency into application FPS or sum separately measured RSS peaks.

## Current README and submission gaps

- README contains stale statements that Jetson generation, advisory integration and worker work remain pending, despite later repository evidence.
- It claims both 151 and 128 tests and uses hard-coded `/root/.cache/...` launch paths from the configured developer machine.
- Root `requirements.txt` does not install the vision/app/advisory stack. The separate lock files and architectures must be explained.
- Existing launcher instructions assume source/model/data already staged. A fresh clone needs a complete permitted asset acquisition/export/transfer route with hashes and model access prerequisites.
- No demo video file/link or consolidated submission technical report was identified in the reviewed tracked file listing. Existing runtime/vision reports supply material but are not a completed submission report.
- No root code `LICENSE` was identified. Record owner-selected licensing separately from dataset/model licenses; do not claim unrestricted reuse.
- `assets/forgesight-station.png` exists; inspect it before deciding whether it is suitable evidence. A concept image must not masquerade as a real app/device screenshot.
- Latest main source is ahead of some deployed-device evidence. The new README needs an explicit tested-build statement.

## Task 1: Reconcile the README facts

**Files:** Modify `README.md` in the subsequent implementation; use existing `docs/STATUS.md`, `docs/COMPLIANCE.md`, `docs/STAGE10_VERIFICATION.md` and referenced manifests as evidence. Correct deeper documents only if needed to resolve a misleading current instruction; preserve historical checkpoint descriptions.

**Consumes:** This plan and existing evidence.
**Produces:** One current feature/status table and a set of traceable, scoped README claims.

- [ ] Record current source commit and the commit/build for each published device measurement.
- [ ] For the six core functions, identify verified scope, provisional behavior and unresolved acceptance gates.
- [ ] Reconcile old README statements and test counts against the evidence without treating historical records as current whole-build verification.
- [ ] Verify that every positive device/offline/classification claim has an accessible manifest or clearly scoped evidence page.

**Proof:** A reader can distinguish supported saved-image operation, physical-camera diagnostics and unverified physical-part acceptance without reading a chronological log.

## Task 2: Write the submission overview

**Files:** Modify `README.md`; reuse existing `docs/` links. Add `docs/assets/` media only when authentic, permitted and reviewed.

**Consumes:** Task 1 claims and the section order above.
**Produces:** Judge-facing narrative, architecture and results sections.

- [ ] Write the opening problem/solution, four deliverable entries and feature/status table.
- [ ] Add the architecture diagram showing separate vision and ExecuTorch advisory runtimes and rule authority.
- [ ] Add concise hardware, models/data, operator workflow, record example and separate benchmark tables.
- [ ] Add limitations, optional-extension status, source tree and attribution/license status.
- [ ] Check every number against its source, sample count and measurement boundary; retain failure cases and fallbacks.

**Proof:** A judge can find the mandatory runtime, detection quality, logging behavior, device evidence and incomplete items from one page. Cold and persistent timings remain explicitly separate.

## Task 3: Make reproduction and deliverables concrete

**Files:** Modify README setup links and commands. Proposed follow-up deliverables: `docs/TECHNICAL_REPORT.md` and `docs/SETUP.md` only if existing guides cannot cover the clean-checkout sequence. Video is linked once actually recorded/published in an accessible location. Do not generate a purported demo or completed report from missing evidence.

**Consumes:** Existing runtime/setup guides, lock files, model manifests and supplied artifacts.
**Produces:** Reproducible supported launch paths and an accurate deliverable index.

- [ ] Document asset names/destinations, source/access requirements, hashes and missing-asset behavior for datasets, EfficientAD export, Llama PTE and tokenizer.
- [ ] Separate PC development, Jetson vision and Jetson ExecuTorch environments; explain which locks apply. Use variables or documented example paths in place of unexplained developer-specific paths.
- [ ] Establish the exact clone/setup/launch/test sequence for each supported platform without changing system Python or global packages.
- [ ] During implementation, run appropriate PC checks in the existing isolated environment. Verify the documented launch path and one persisted inspection if permitted assets are available; record unavailable clean-machine/device checks honestly. Do not rerun final holdout inference to validate documentation.
- [ ] Plan the demo sequence: show target hardware and source mode, inspect normal and defective/uncertain cases, show alert/decision, real generated advice and SQLite history/restart; show actual offline evidence for the demonstrated build. If replay is used, label it visibly throughout. A final physical-part demonstration still follows the roadmap gate.
- [ ] Outline the separate report: hardware/software, both runtimes/backends, models and data splits, JSON interface, severity/decision rules, deployment/offline method, detection and latency/resource results, advice validation/fallbacks, failure analysis and limits.
- [ ] Track final video URL, report link, team details, portal-specific instructions and code-license decision as open until supplied or completed.

**Proof:** A new reader can identify everything absent from a clone, follow the supported environment path and tell which submission deliverables exist. Unavailable assets or checks are named, not hidden behind a generic pip install command.

## Task 4: Audit the rendered README

**Files:** Review `README.md` and any documentation/media added in Tasks 1-3.

**Consumes:** Completed draft and deliverable links.
**Produces:** Reviewed documentation and a truthful submission-readiness checklist.

- [ ] Resolve every relative link and anchor against tracked GitHub files; check filename case and exclude private/local evidence destinations.
- [ ] Preview tables, Mermaid and media in a GitHub-compatible renderer; manually inspect command blocks and source-mode captions.
- [ ] Map every guide row above to a README section or an explicitly missing deliverable/measurement.
- [ ] Confirm that newer main-source features are not represented as tested in an older device snapshot; verify all current-build claims.
- [ ] Review focused documentation diffs with `git diff --check`, `git diff -- README.md docs/` and `git status --short`; leave `scripts/record_timings.py` untouched.
- [ ] Commit only the reviewed authorized documentation. Push only under confirmed session authorization; this planning request alone does not authorize publishing a submission.

**Proof:** No broken submission links, fabricated artifacts, unsupported performance claims, misleading readiness badges or accidental local files. Documentation readiness does not close missing device proof gates.

## Next exact action

Review this proposed README structure and scope. Then implement Task 1, reconciling the feature/status facts before writing the README narrative. No application or device work is needed to start that documentation substep.

## Execution record: 24 November 2026

User authorized auditing records/docs and updating README from this plan against
the latest commit, with **no push until confirmation**. Implementation base is
`081ba00` on existing branch `feat/hybrid-llama-advisory`.

- Task 1: reconciled source, latest guidance manifest and historical device scopes.
  Inventory: 29 existing tracked docs and 73 JSON manifests; all JSON parsed;
  four latest guidance source hashes matched after LF normalization.
- Task 2: replaced chronological README with the proposed submission overview,
  architecture, both runtimes, benchmark tables, actual saved-image example,
  experimental screw scope, limitations and attribution.
- Task 3: added `docs/SETUP.md` and `docs/SUBMISSION_AUDIT.md`. Video, consolidated
  report, clean-checkout asset package, missing measurements and device acceptance
  remain explicitly open. No final deliverable or hardware result was fabricated.
- Task 4: checked relative links, JSON evidence, baseline F1, CLI option definitions,
  Markdown fences and whitespace; requested independent documentation review.
  Full GitHub/Mermaid rendering is not verified by these structural checks.
  Independent review found no blocking issues. Its two minor findings were
  addressed: distinguish unavailable from optional fallback in the diagram,
  and include an evidence-linked saved-image inspection/advice example.

Ruling: work in the user's current feature checkout for this documentation-only
update, preserving unrelated work; do not install dependencies or change devices.
Application tests were not rerun because WSL returned `E_ACCESSDENIED`.
The existing untracked `scripts/record_timings.py` remains outside the change.
This execution record supersedes the original planning-only scope for the
authorized documentation edits; it does not authorize submission publication.

### Feature catalogue expansion

The user requested an exhaustive Phase 1 feature inventory, including plugins
and features absent from older docs, plus future development for report writing.
The README was expanded from source/templates/tests into six capability groups
and a roadmap-ordered future-work table. Source inspection also clarified
history concealment, HTML versus GTK PDF export, current camera-source UI,
backend-only replay routes and the fixed two-recipe registry. An absent repo
link is now distinguished from an artifact potentially prepared elsewhere.

Focused existing PC/WSL tests passed in two runs: 58 plugin/advice/health/logging/
desktop/bundle checks and 63 UI/storage/worker/queue checks. Earlier WSL access
denial was overcome for these isolated software tests. No new hardware proof,
packages, application features or future-stage implementation were introduced.
Push remains prohibited until the user confirms.

Independent catalogue review found no important issues. Its two minor
corrections were addressed: grayscale screw uploads are listed, and the
linked Jetson UI guide now distinguishes current controls from historical
source-selector proof. The final catalogue contains 38 feature entries across
six groups. No source-code changes were needed.
