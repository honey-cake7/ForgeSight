# ForgeGrade pixel prototype — Stage 5

The saved-image pipeline now adds an explicit, deterministic action to each
inspection, including errors. The same evidence and policy always produce the
same decision. No model training or threshold changes were made in this stage.

This is **prototype rule behavior**, not approval for production use. Two
previously documented missed scratches still receive provisional PASS in the
cached benchmark diagnostic. Defect-type classification is still an open
requirement. Those limitations are not fixed by adding rules.

## Policy, order and trade-offs

`configs/forgegrade_v1.json` pins the model and score-policy SHA-256, source
size, cutoff, example geometry limits and version. Each decision stores the
full snapshot and a canonical SHA-256; the verification report also hashes the
literal policy file. Rules run in this order:

1. Failed/incomplete inference, mismatched model/policy, invalid/nonfinite or
   inconsistent evidence -> **REVIEW**, unassessed severity.
2. A normal candidate with zero response pixels and a score strictly below
   90% of the frozen image cutoff -> **provisional PASS**. Scores at or above
   that numerical guard, including equality at the image threshold, -> REVIEW.
   This 10% proximity band is a transparent prototype guard, not calibrated
   model confidence or a guarantee against missed defects.
3. An anomaly with unknown/unseen type or without validated, reliable same-domain
   type evidence -> **REVIEW**, even when its response area is large. Every real
   anomaly from the current EfficientAD model follows this path.
4. For a future reliably typed anomaly only: response area at least 1% of the
   full 700x700 image, a region's box diagonal at least 140 pixels, or overlap
   with a configured critical-zone rectangle -> example **REJECT** rule.
   Smaller known responses -> REVIEW. Zone overlap is conservative bounding-box
   intersection, not a pixel-accurate zone/damage intersection. Default zones
   are empty because no approved part drawing defines critical zones yet.

The 1%/140-pixel cutoffs illustrate configurable geometry decisions; they are
not inferred manufacturing tolerances, tuned benchmark optima or validated
physical-defect limits. Reliable type evidence is not supplied by this project
yet. The REJECT branch is verified with explicitly synthetic rule fixtures,
not presented as an actual classifier result. Changing to automatic rejection
or production PASS needs new type/geometry/acceptance evidence.

Area counts source-grid model-response pixels above the predeclared frozen
image threshold, with 8-connectivity and no morphology. Bounding boxes and their
diagonals are pixels; diagonal is an extent proxy, not scratch length. The pixel
mask cutoff has no separate localization calibration. No millimeters, damage
probabilities or industrial certification are claimed. Geometry cutoffs depend
on this image profile, so other sizes are reviewed/refused rather than silently
scaled to physical units.

Llama has no decision-making input to the grading function. An unrelated advice
payload cannot change its result. Advice integration remains Stage 9 with the
mandatory real Llama-family model through ExecuTorch on-device. The current
action text is deterministic guidance to hold/check/segregate a part; it is not
Llama advice, a root-cause diagnosis or an official SOP citation.

## Actual saved outputs

`manifests/graded_inspection_replay_v1.json` records a genuine rerun of the
development-image pipeline after grading integration. Nine frames produce four
declared-part records, five repeats suppressed, with decisions:

| Declared part | Actual result | Action |
| --- | --- | --- |
| replay-A | Normal fit image, score 0.0284541, zero response pixels | PASS (provisional) |
| replay-B | Development bent image, score 0.6828089, 9,015 response pixels; predicted type unknown | REVIEW |
| replay-C | Identical bytes to A, explicitly distinct declared part | PASS (provisional) |
| replay-invalid | Missing file, explicit ERROR | REVIEW |

Raw JSON records, maps and overlays stay under
`artifacts/core/graded_replay_v1/records/`. The visual result is
`artifacts/core/forgegrade_proof_v1/graded_results.png`. Original Stage 4 outputs
are preserved. Record-write errors still propagate; database durability,
history queries and restart-safe identity are Stage 6 work.

A separate diagnostic applies the new rules to the already saved holdout maps,
without any further holdout model inference or physical-part events:

| Dataset truth | Provisional PASS | REVIEW | REJECT |
| --- | ---: | ---: | ---: |
| Good | 17 | 5 | 0 |
| Defective | 2 | 87 | 0 |

The two provisional passes are `test/scratch/000.png` and
`test/scratch/022.png`. The rules were written after the original baseline
evaluation was exposed, so this is a **post-evaluation diagnostic**, not a fresh
independent grade-acceptance test. No cutoffs were subsequently changed from
these results. Every anomalous candidate remains unknown-type REVIEW.

## Reproduce and verify

Use the existing locked vision environment; no packages were installed or
upgraded in Stages 3.4–5. Output directories must be new; original evidence is
preserved.

```powershell
# Full current suite, including pixel geometry.
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m unittest discover -s tests -q
# Real integrated pipeline, three forwards plus one invalid-input record.
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.inspection_replay --output artifacts/core/graded_replay_v1
# Cache-only verification: records plus original holdout maps. No model loads.
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.grade_proof --output artifacts/core/forgegrade_proof_v1
# Add a decision to an existing record while preserving the source file.
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.grade RECORD.json --output NEW_GRADED_RECORD.json
```

All 101 tests passed. They cover deterministic repeatability, unchanged input,
PASS/REVIEW/REJECT rules, exact area boundaries, length proxy, configurable zone
location, unknown/unseen/unreliable types, threshold equality, corrupt evidence,
policy fingerprints, refusal of millimeter/certification claims and inability
of advice to override rules. Synthetic fixtures only test rule behavior; the
separate replay proves the real detector-to-record-to-grade path.

The proof CLI verifies saved map/prediction hashes and reproduces decisions.
Committed proof: `manifests/forgegrade_verification_v1.json`. Recorded pipeline
timings include their scope; cached forward-only 14.54/14.80 ms p50/p95 from
Stage 3.3 excludes decoding, map/overlay writes and rules. First inference can
be much slower. No end-to-end Jetson latency, accuracy target, camera throughput
or offline-device result is asserted.

Next is **Stage 6: SQLite persistence**, including errors, distinct inspection
IDs and duplicate/restart checks. Jetson execution, physical camera tests,
reliable defect types, approved geometry tolerances and offline proof remain
open. The current authorization stops at Stage 5.

## Stage 7 current command update

The full current suite includes FastAPI contract tests and uses
`/root/.cache/forgesight/app-v1/venv/bin/python`, with 119 tests passing. Earlier
stage-specific test counts/vision-environment commands above describe their
historical snapshots. See [the current operator workflow](OPERATOR_APP.md).
