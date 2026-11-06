# ForgeAssist first advisory component — Stage 9.1, PC only

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
checked. An unsupported or malformed answer stays **ADVICE_UNAVAILABLE**, with
raw output retained for inspection. Accepted text is still marked
**GENERATED_REQUIRES_HUMAN_REVIEW**. Vocabulary checks do not prove semantic
correctness: different word order can change meaning, so human review and
broader evaluation are still required.

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

All **128 tests pass**. New tests cover deterministic context, excluding prompt
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

Next PC substep: predeclare advisory scenarios and evaluate/extend the contract
for normal, anomalous and failed inspections, including refusal behavior. Then
add the bounded background worker and separately persisted/UI advice states
without changing original grades. Approved factory procedures are needed before
expanding beyond prototype image checks. Stage 8 and on-device/offline gates
remain BLOCKED - NOT VERIFIED. Stop for review after this substep.
