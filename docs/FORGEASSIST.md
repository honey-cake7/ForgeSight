# ForgeAssist advisory component — Stages 9.1–9.2, PC only

Stage 8 camera/hardware work is BLOCKED: the user still has no Jetson. Under
the roadmap's hardware-blocker rule, this independent PC substep takes one
existing real inspection through the actual ExecuTorch Llama runner. It does
not complete Stage 9, connect advice to the UI, or certify edge deployment.

## What is implemented

`assist_contract.py` checks a saved analyzed anomaly with the current unknown
type and deterministic REVIEW decision. It recomputes the recorded grade from
its policy snapshot and refuses inconsistent evidence. The full record and
decision have canonical SHA-256 identifiers. The compact prompt contains only
the anomaly flag, fixed REVIEW, unknown type/cause and permitted checklist text.
Free-text part IDs, image paths and operator notes do not enter the prompt.
Score and geometry stay in the linked evidence record, not in the Llama prompt.
Llama receives no image and cannot perform another visual inspection.

Two aliases identify the supplied sources:

- **E1:** the immutable real inspection, linked by its UUID and metadata hash.
- **G1:** the versioned local checklist, linked by document ID/hash and exact text.

`guidance/prototype_inspection_v1.json` is explicitly developer-authored
**PROTOTYPE_NOT_FACTORY_APPROVED** guidance. Its permitted action is to hold
for human review and check image clarity/lighting. It is not an official SOP,
manufacturing tolerance or permission to alter machinery. No approved factory
procedure, machine history, confirmed cause or recurrence analysis is supplied.

`assist_once.py` writes the input snapshot, evidence, prompt and runtime request,
then launches the existing isolated ExecuTorch environment as a bounded child
process. The native TextLLMRunner loads the pinned PTE/tokenizer and generates
real text. All stdout/stderr, native statistics and generated tokens are saved.
The original inspection JSON and SQLite database are not modified.

The host code owns the structured advisory payload. The model supplies a short
cited note; it cannot supply or overwrite a grade field. Known citations,
length, uncertainty, a review/check step and restricted source vocabulary are
checked. Stage 9.2 additionally requires one of two reviewed complete statement
forms, closing a demonstrated word-rearrangement weakness described below.
An unsupported or malformed answer stays **ADVICE_UNAVAILABLE**, with
raw output retained for inspection. Accepted text is still marked
**GENERATED_REQUIRES_HUMAN_REVIEW**. The current profile is intentionally limited
to copying/rephrasing the supplied checklist. Broader usefulness, diagnosis and
operator acceptance remain unverified.

The host explicitly withholds `possible_explanation` as null because no causal
evidence was supplied. That null is a host policy decision, not a claim that
Llama diagnosed or ruled out a cause. The response clearly labels its origin.

## Actual result and rejected attempts

The input is the actual saved anomalous browser inspection `UI-PROOF-B`, with
score 0.6828089 and 9,015 response pixels. No vision or holdout inference was
rerun. Its authoritative decision remains REVIEW and type remains unknown.

The final native model output was:

> [E1] Anomaly flagged; grade REVIEW; defect type and cause unknown.
>
> [G1] Prototype checklist: Hold for human review; check image clarity and lighting.

This is a faithful repetition of the supplied facts/checklist. It is a small
functional grounding demonstration, **not proof of novel reasoning, diagnosis,
general advisory reliability or factory usefulness**.

All eight development attempts are retained. The first six had formatting,
missing-content or unsupported-content problems: truncated/speculative JSON,
null checks, extra reliability claims, unwanted prose or missing useful text.
They were never accepted as advice. The seventh output was factually within the
sources but was refused by an overly narrow vocabulary guard that omitted the
supplied words "prototype checklist". The guard was corrected to include source
vocabulary, with a regression test. The eighth generated the identical text and
passed. This is not an independent 1/8 success-rate experiment: prompts and
validation evolved on one example. It does show that the initial interface was
unreliable, and the final contract needs broader testing before UI integration.

The separate system-message option in the native wrapper establishes behavior
while the user message carries the evidence. Both are recorded. The historical
single-user smoke prompt formatting remains byte-identical; default behavior
and new system marker/input guards are tested. The same model, tokenizer,
quantization, native libraries and exported context were retained.

| Final actual desktop observation | Value |
| --- | ---: |
| Prompt tokens | 82 |
| Generated tokens reported by native runner | 33 |
| Output limit | 96 tokens |
| Exported dynamic-prefill/context bounds | 128 / 512 |
| Native inference interval | 1.348 seconds |
| Whole child process, including Python/import/hash/model load | 8.457 seconds |
| Sampled child peak RSS | 1,220,356 KiB |

These are one desktop cold-process observations, not p50/p95 or Jetson results.
RSS sampling can miss peaks and excludes the parent/UI/vision processes. The
worker uses eight desktop CPU threads; that is not a Jetson thread-budget
recommendation. This substep intentionally pays per-process loading overhead;
resident model lifetime, thread contention and queue behavior remain future work.

## Verification and files

At the Stage 9.1 checkpoint, all **128 tests passed**. New tests covered deterministic context, excluding prompt
injection through free-text metadata, invalid/incomplete evidence, labelled
guidance, cited output, rejection of unsupported actions/causes/measurements,
real child-process timeout and unavailable runtime. Output strings in unit
tests are explicitly fixtures; the native run above is the execution evidence.

The timeout/failure paths keep advice unavailable and preserve the original
record. They do not establish UI responsiveness because this component is not
yet called by the UI. All five existing Linux environment inventories are
unchanged. No packages, models or datasets were installed/downloaded. Trained
vision checkpoint, score policy, original inspection JSON and proof database
hashes remain unchanged. No new holdout/model inference was performed for vision.

Local evidence is under `artifacts/assist_stage9_1/attempt_01` through
`attempt_08`, with raw `native/generated.txt`, request, native results, logs,
input snapshots, evidence and advice reports. The final accepted candidate is
in `attempt_08/advice_report.json`. Committed summary and failed-attempt evidence
are in `manifests/forgeassist_pc_note_v1.json`.

From PowerShell in the project folder, use a new output directory:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/app-v1/venv/bin/python -B -m forgesight.assist_once --record artifacts/app_stage7/proof_runtime/records/d9130609-fecb-4a37-a78b-9f28401169b2.json --output artifacts/assist/manual-001
wsl -d Ubuntu --exec /root/.cache/forgesight/app-v1/venv/bin/python -B -m unittest discover -s tests -q
```

The controller accepts `--timeout`, `--guidance` and `--runtime-python`. The
current profile accepts this first unknown-type anomaly/REVIEW case only;
normal/error/other-policy scenarios are refused, not given fabricated advice.

## Jetson boundary and next gate

Keeping the 1B Llama, existing quantized PTE, compact 82-token prompt and bounded
96-token reply avoids increasing the export's memory/context requirements.
The evidence schema and host-controlled decision separation are device-neutral.
However, this CLI deliberately invokes the **desktop x86_64 guarded worker**;
it is not an ARM deployment adapter. On Jetson we still need exact hardware and
JetPack inventory, a verified native AArch64 ExecuTorch build/adapter, appropriate
CPU-thread/memory budgets and actual generated text. Do not copy this x86_64
environment to ARM or treat its timings as target performance. Offline flags
are not a network-disconnection test.

The operator UI correctly still says Llama advice is not connected. No advisory
table/migration, background queue, automatic delivery or UI generation state
was added. Stage 9 is **not DONE**.

## Stage 9.2: declared refusal scenarios and a narrower output contract

The PC-only follow-up has now tested the contract before UI integration.
`configs/forgeassist_scenarios_v1.json` declares 12 input cases, 15 handwritten
output fixtures and five runtime checks. Its original bytes/hash and source
fingerprints were saved before execution. Cases reuse existing real normal,
anomaly and failed-input records. Tampered records and generated-output examples
are explicitly test fixtures, not new model predictions or factory evidence.

The old vocabulary-only guard accepted this handwritten unsupported statement:
"The cause is lighting; type unknown. Hold for human review and check image
clarity [E1][G1]." It contained only allowed words. The baseline acceptance is
retained under `artifacts/assist_stage9_2/baseline_weakness.json`; this was not
an observed Llama answer. A word list cannot preserve meaning.

Output policy `forgeassist-reviewed-note-forms-v2` now permits only two reviewed
complete notes: the source-copy form shown above and a fixed paraphrase stating
unknown type/cause and human review/clarity/lighting checks. Only case,
whitespace and evidence-tag placement may vary. Other text is withheld even if
its individual words were supplied. This intentionally sacrifices wording
flexibility. A general semantic judge or second LLM would add complexity and
compute without established quality or Jetson budgets. This narrow contract
does not establish independent reasoning, novel recommendations or root causes;
a deterministic template would suffice for its present contents. The actual
Llama/ExecuTorch execution remains mandatory and is preserved, while useful
domain-specific guidance needs approved source material and separate evaluation.

Guidance is checked against its reviewed canonical JSON SHA-256, not just a
document ID or an approval label. Modified text under the same identity is
refused. Updating legitimate guidance requires deliberate version/hash and
statement-policy review. The checklist remains **PROTOTYPE_NOT_FACTORY_APPROVED**.

`AdvisoryRefusal` provides a stable reason code. `advise_once` now saves an
input snapshot and **ADVICE_UNAVAILABLE** report for refused inputs, with
`runtime_requested=false`, `process=null`, and no generated note. Normal
results keep their original provisional action; failed inspections keep their
original error/REVIEW. Neither needs a Llama call in this profile. Invalid grades,
unsupported defect names, incomplete geometry and altered guidance are refused.
This saves unnecessary compute and keeps the original records unchanged.

### Actual checks and their limits

- All **27 contract checks** matched: 12 eligibility cases (including ten
  refusals before runtime) and 15 output fixtures (two accepted, 13 withheld).
- Two normal-length native Llama runs generated the same source-linked note,
  using 82 prompt tokens and reporting 33 generated tokens each. Adding malicious
  part-ID/path/note metadata changed neither prompt nor generated note because
  those fields never enter model input. This is exclusion, not a test of Llama
  resisting an injected prompt. Both runs have the **same single prompt** and
  are not independent advice-quality scenarios.
- Timeout and missing-runtime probes saved unavailable reports without advice.
- The one-token probe was also safely unavailable, but the native statistics
  guard rejected it before the expected text-parser layer: native statistics
  reported zero generated tokens. Therefore the original full protocol report
  remains **31/32 matching expectations, `passed=false`**. It is not rewritten
  or presented as a fully passing predeclared experiment.
- A separate prospective eight-token follow-up was declared before execution.
  It generated `[E1] Anomaly flagged; grade`, which the text parser withheld for
  missing evidence. This verifies the additional intended failure path; it does
  not retroactively change the original experiment.
- All **132 software tests passed** in 8.325 seconds. The five environment
  inventories and 13 protected file hashes stayed unchanged. No dependency
  installation, new vision/holdout inference, retraining or score/policy tuning
  occurred. No UI, database migration or queue was added.

The two successful desktop native inference intervals were 1.361/1.362 seconds;
whole child-process times were 9.115/8.602 seconds, including imports, hashing
and cold model load. Sampled child peak RSS was 1,219,960/1,219,204 KiB. These
exclude parent/vision/UI memory, may miss peaks, and are not percentiles or
Jetson performance claims. The existing 1B quantized artifact, tokenizer,
128-token prefill bound and 512-token context were unchanged.

Saved outputs, initial mismatch and follow-up are under
`artifacts/assist_stage9_2/`. The committed summary is
`manifests/forgeassist_refusal_verification_v1.json`. Reproduce contract-only
checks without loading Llama, using a fresh output directory:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/app-v1/venv/bin/python -B -m forgesight.assist_evaluate --output artifacts/assist/manual-contract-001
```

Adding `--run-native` runs the original five PC probes too, including the
documented one-token failure-layer mismatch; expect that original protocol to
exit 1. The eight-token follow-up is reproduced by calling
`advise_once(record_path, fresh_output, max_new_tokens=8)` on the saved anomaly.
Contract/fixture checks are standard-library Python and device-neutral; the
actual runner remains explicitly desktop x86_64 only. A native ARM adapter,
target-compatible dependencies, hardware inventory and memory/thread measurements
must be verified on Jetson. CPU thread counts are not target recommendations.

**Next exact PC task, Substep 9.3:** implement a bounded, single-worker advisory
queue with immediate busy refusal and timeout/failure isolation, proving that
inspection results remain available while advice is pending. Persisted advisory
state and UI delivery follow as separate substeps. Approved factory procedures
are needed before expanding beyond prototype image checks. Stage 8 and all
on-device/offline gates remain BLOCKED - NOT VERIFIED. Stop for review here.

## Stage 9.1 Jetson replay advisory update

At the user's request, work continued using the already persisted Jetson
`SAVED_IMAGE_REPLAY` record while no physical nut is available. The new
`forgesight.assist_jetson_replay_once` controller checks that the JSON matches
SQLite, validates the deterministic decision and reviewed prototype guidance,
then prepares compact `[E1]` and `[G1]` text. It runs the private Jetson Llama
3.2 1B PTE through the existing ExecuTorch/XNNPACK ARM adapter and saves a
separate advice payload. It never updates the inspection record or grade.

The ARM adapter now accepts an optional system-role prompt. Its earlier
user-role-only probe generated an incomplete note without `[G1]`, which the
host contract withheld. The system-role run generated the reviewed cited
note, and `parse_advisory` accepted it. A separate missing-runtime test saved
`ADVICE_UNAVAILABLE` and no advisory; a fresh SQLite read still matched the
original `REVIEW` record. The actual command on Jetson, from the private
vision directory, was:

```sh
/usr/bin/python3 -B -m forgesight.assist_jetson_replay_once --record runs/dev_bent_017_replay_02/record.json --database inspections_replay_v2.sqlite3 --guidance guidance/prototype_inspection_v1.json --llama-root "$HOME/.local/share/forgesight/llama_cpu_v1" --output-dir "$PWD/runs/forgeassist_replay_advice_01"
```

Choose a fresh output path to repeat it. Evidence and exact hashes are in
`manifests/jetson_forgeassist_replay_v1.json`; raw logs and text remain in
private Jetson storage with an ignored PC copy. One run used 82 prompt and
33 generated tokens, 1.955 s native inference and 22.157 s native child wall
time. The checklist is developer-authored and **not factory-approved**. The
note is a narrow reviewed form, not independent diagnosis or a confirmed cause.
The controller is synchronous; a bounded queue, timeout proof, UI delivery,
physical-camera advice and network-disconnected run remain open.
