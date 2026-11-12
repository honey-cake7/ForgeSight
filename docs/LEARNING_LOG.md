# Learning log

## Stage 10.5: Test camera-open failure without altering hardware

The V4L2 OpenCV build used by this project accepted camera indices; a named
device-path approach showed an incompatibility warning in the focused Linux
test. Passing absent index 99 let the same capture code exercise an open
failure without unplugging the user's C270. A later index-0 run captured a
usable frame after 16 dark startup frames. The frame showed the room, so the
model response is diagnostic data only and cannot be graded as a part.

## Stage 10.4: Intake rejection differs from an inspection error

The replay accepts a hash-checked image before creating an inspection record.
When a model path is missing after that point, the failed attempt is saved as
ERROR with a deterministic REVIEW and is still visible in SQLite. A corrupt
image that fails the frozen dataset hash check is rejected before an
inspection starts, so it creates no inspection row. Keeping those outcomes
distinct makes recovery evidence easier to audit. A later valid replay used
the unchanged model and preserved the ERROR row.

## Stage 10.3: Local UI service under disconnection

Serving page files and exercising the queue through Jetson loopback HTTP
shows that the UI service needs no upstream route for those operations.
It does not show that a browser rendered the page: this Jetson had no local
browser executable. An advisory timeout must leave a visible unavailable
state with no note, while the deterministic inspection remains readable.
Restarting the UI process after both success and timeout checked that those
states came from disk and could not trigger duplicate advice.

## Stage 10.2: Repeated saved cases under sampled disconnection

A fitting image can verify that the runtime follows the normal branch, but it
cannot be counted as fresh normal accuracy evidence. The replay gate therefore
requires an explicit flag and permits only `train/good/000.png`; final holdout
remains blocked. Two already exposed development images exercised the anomaly
branch. Both yielded REVIEW because defect subtype is unvalidated, not because
the system identified their filename labels. Sampling network interfaces and
routes every second gives stronger evidence than only entry/exit snapshots,
but does not rule out brief changes between samples. Three first-forward
measurements do not justify p95 latency or parts-per-minute claims.

## Stage 0.1: Learn what exists before changing it

The PC is a Windows x86_64 workstation with an RTX 2000 Ada GPU. Its Ubuntu WSL
environment can import PyTorch and discover that GPU. WSL is still running on
the PC; it does not turn desktop results into Jetson ARM64 evidence.

A working `python` command does not guarantee that a Python launcher or another
shell uses the same interpreter. Record the executable path and inspect that
environment's packages explicitly.

No local repository existed. The original four files are valuable local work
even though the remote repository has no branches. The user later confirmed
that the Jetson is unavailable.

## Stage 0.2: Protect local work with honest version control

- `git init -b main` creates repository metadata and names the initial branch.
- `git remote add origin URL` stores a remote address; it does not connect local
  history to remote history or transfer files.
- `git add` selects the exact contents for the next snapshot.
- `git diff --cached` reviews that selected content.
- `git commit` creates a local snapshot; `git push` is a separate upload.
- `.gitignore` keeps matching untracked files out of ordinary staging. It does
  not remove already tracked secrets and is not a backup system.

The PDF is retained locally and ignored; the instruction Markdown files and
new project documentation belong in the initial source snapshot. Raw data,
weights, runtime history, generated artifacts, and credentials stay outside
ordinary staging.

`.gitattributes` keeps text line endings consistent between Windows and WSL.
The original roadmap's intentional Markdown hard breaks have a file-specific
whitespace exception, so validation does not require changing that reference.

Checks for this step: inspect the staged inventory, validate ignore examples
without creating data/secret files, check staged whitespace, compare original
file hashes, and inspect final Git status. Report actual outcomes in the step
response rather than inventing a future commit hash inside these notes.

Concept checks:

1. Does adding `origin` or making a local commit upload anything to GitHub?
2. If a secret was already committed, does adding it to `.gitignore` erase it
   from history?

## Stage 0.3: Build a small repeatable image environment

An isolated virtual environment gives this project its own packages. Creating
it with an existing environment's Python does not copy that environment's
installed packages by default. Calling `.venv/bin/python` explicitly avoids
depending on shell activation or an ambiguous `python` command.

`requirements.txt` pins Pillow's version and the exact Linux x86_64 CPython 3.12
wheel hash. `pip install --require-hashes --only-binary=:all:` checks artifact
integrity and avoids an unplanned source build. Python, OS libraries, and GPU
drivers are separate prerequisites, not captured by this package pin.

`scripts/check_environment.py` contains one `main()` function. It checks that a
venv is active and the user package site is disabled, then encodes and decodes
a tiny in-memory RGB image and compares every pixel byte. This tests the image
codec and Python environment, not anomaly detection. `pip check` separately
checks installed dependency consistency.

Both checks passed in the project environment and a second fresh temporary
environment. The project environment cannot see the shared environment's
PyTorch, torchvision, NumPy, ExecuTorch, or OpenCV packages.

Concept checks:

1. Why pin both a package version and a wheel hash?
2. What does a successful PNG round trip prove, and what does it leave untested?

## Stage 1.1: Understand data roles before opening images

Anomaly detection asks whether appearance differs from normal. Classification
asks which learned class applies. A folder label is ground truth for checking
a prediction, not evidence that a model can predict that type.

An image shows appearance; a mask marks anomalous pixels for localization
evaluation. The publisher's evaluation code pairs anomalous images with masks
using the defect folder and filename stem. Normal samples do not have mask
files; a loader can represent their ground truth as an all-zero mask. It must
not treat a missing anomalous mask as a normal sample.

Fitting learns from the training partition. Validation supports configuration
and threshold choices. Final evaluation checks the frozen system. Repeatedly
adjusting settings after examining final test results leaks information from
evaluation into development and makes performance claims optimistic.

Stage 1 requires real defective previews for learning. Reserve four development
examples deterministically before inspecting them; exclude them from the final
holdout. This gives an expected custom 111-image holdout, not an untouched
115-image official test set. Our 44-image normal-only validation partition can
help calibrate false alarms but cannot establish defect recall. No files have
actually been assigned yet; these are a recorded protocol and expected counts.

The dataset has noncommercial licensing conditions and attribution obligations.
Keeping raw data out of Git is our repository policy, not a claim that all
redistribution is prohibited. The detailed source review is in `DATASET.md`.

Concept checks:

1. Why must an image used to debug an overlay be excluded from the untouched
   final holdout?
2. Why can't normal-only validation tell us how many defects the system misses?

## Stage 1.2: Verify the actual archive before using it

The real archive contains 335 inspection images and 93 masks. Images are RGB;
masks are grayscale, with 0 for normal pixels and 255 for anomalous pixels.
Mask brightness is a binary annotation, not a probability or severity score.

SHA-256 fingerprints the archive bytes. Matching hashes prove our copy matches
the supplied file; they do not independently authenticate publisher origin.
Here the user confirmed downloading through MVTec, while the exact archive URL
and publisher checksum were unavailable. Keep those evidence types separate.

`scripts/audit_metal_nut_archive.py` checks archive entries and fully decodes
the PNGs without displaying them. `canonical_path()` prevents path traversal;
`audit_archive()` checks structure/counts/pairing and records file/pixel hashes;
`extract_audited_archive()` writes into a new directory and verifies each file.
Normal images have no mask files; anomalous images must have nonempty masks.

Fourteen temporary synthetic tests in `tests/test_archive_audit.py` exercised
valid and malformed archives. The actual dataset was then independently audited
and extracted: 430 regular files, including the original license and readme,
matched their hashes. The test fixtures are not claimed as inspection samples.

No split or preview exists yet. First create a frozen manifest and check
duplicate content, then use only the designated development examples to learn
image/mask visualization. Structural validation does not demonstrate detection
accuracy or camera generalization.

Concept checks:

1. What does matching the original archive's SHA-256 prove about our copy?
2. Why is a missing anomalous mask an error, while normal samples have no mask?

## Stage 1.3: Freeze sample roles before learning from previews

`forgesight/data_manifest.py` verifies the current files against acquisition
hashes, rejects exact image-byte or decoded-pixel duplicates, and ranks portable
relative paths with the recorded seed/version. The manifest stores metadata,
not raw dataset pixels. It assigns 176 fitting, 44 normal validation, four
development, and 111 final-evaluation samples.

The real dataset had zero byte/pixel duplicate image groups. This does not rule
out near-duplicate photographs or prove physical-part independence. A second
build in memory reproduced the manifest bytes. Eight split tests protect
order-independent assignments, role integrity, mask pairing, duplicate
rejection, and path boundaries; all 22 current tests passed.

Concept checks: Why must two runs produce identical role assignments? Why do
zero exact duplicates still leave physical-part independence unverified?

## Stage 1.4: Load verified inputs without inventing predictions

`MetalNutDataset` exposes one role from the frozen manifest. On every load,
the file and decoded-pixel hashes must match. An `InspectionInput` carries the
source image, mask, role, and explicitly named ground-truth type. Nothing in
this loader predicts anomalies or severity.

For a normal sample, the zero mask is an explicit representation of the normal
annotation. A missing anomalous mask is instead an error. Real loading passed
for all 176 fitting, 44 validation, and four development samples. The loader
preserves 700x700 source geometry, while final-holdout pixels require opt-in.
Nine new synthetic tests cover changed data/masks, missing annotations,
checksum tampering, role gates, indexing, and normal/anomaly mask behavior.
All 31 current tests passed. Commit `7ff8846` for the prior manifest step was
successfully pushed before beginning this loader work.

Concept checks: Why name the label `ground_truth_type`? Why should a missing
anomalous mask never become an all-zero mask silently?

## Stage 1.5: Inspect real annotations without consuming the holdout

The contact sheet shows a normal fitting image and four reserved development
examples, each beside its mask and red overlay. It was visually inspected:
normal pixels are unchanged by the overlay, and the annotated regions align
with the source images. The title explicitly identifies ground truth rather
than model output. Five new preview tests passed; all 36 tests now pass.

The normal mask is empty. The flip annotation covers most of the nut, while
the other displayed annotations are smaller localized regions. That difference
does not establish which defect is more severe. The recorded areas are source
pixel counts from annotations; no calibration or quality decision is claimed.

Only display tiles are resized. Binary masks use nearest-neighbor resampling,
while originals stay at 700x700. The local output includes source/license
attribution, exact sample paths, manifest fingerprint, mask provenance, and a
record that zero holdout images were displayed.

The manifest step (`7ff8846`) and loader step (`0eada8b`) were each committed
and pushed before proceeding. This third step completes Stage 1's technical
proof and stops at the review gate before mandatory ExecuTorch device work.

## Stage 2.1: Separate a supported route from proof on our hardware

The user accepted Stage 1 and authorized three Stage 2 steps with pushes. Public
sources identify a manually gated Llama 3.2 1B model and a pinned ExecuTorch
1.5.1 source candidate. A source-pinned CPU/XNNPACK route is not evidence of
Jetson GPU use, sufficient RAM, binary compatibility, or generated text.

The user has no Meta Llama access, and the Jetson remains unavailable. Preflight
recorded real WSL host facts and refused the device gate; four new parsing and
guard tests passed. No packages were installed and no model text was simulated.
The three-state distinction matters: source review can be verified while export
is NOT RUN and actual device inference stays BLOCKED - NOT VERIFIED.

## Stage 2.2: A checked recipe is still not an export

The recipe generator writes exact argument lists/configuration without running
commands. Pin both model and ExecuTorch revisions so that future exports do not
silently switch APIs or checkpoints. AST-derived field metadata checks the
configuration against the chosen source without installing the exporter.

Six tests passed. The actual recipe correctly reports missing model access,
weights, checkout, and export environment. No generated model file or text
exists. Dependency installation/locking and actual export remain unverified.
The prior preflight step was successfully pushed as `51cdb0f` before this step.

## Stage 2.3: Capture evidence without promoting it to proof automatically

The capture harness refuses non-Jetson execution, checks native runner identity
at the ELF architecture level, fingerprints local files, and records bounded
process logs, exit/timeout status, elapsed wall time, and sampled RSS. It does
not know from a filename whether a model is Llama or whether printed text is
new generation. Those facts require export/build provenance and text review.

All 52 tests passed. The six capture tests use generic child processes and
synthetic guard inputs, never fake Llama replies. The actual PC capture attempt
was blocked before any runner started; generated text is absent. No source tag,
recipe, exported file, or capture report alone closes the mandatory device gate.

The first two preparation commits (`51cdb0f`, `c06e15e`) were each pushed before
the next step. Stop after the third push for review. The user can request model
access while independent Stage 3 PC work proceeds under an explicit Stage 2
blocker. No Jetson performance or offline success is claimed.

## Stage 2.4: Confirm the downloaded model is the intended source asset

The user obtained access and downloaded five files. A filename alone is not
enough: the checkpoint and tokenizer matched publisher SHA-256 fingerprints,
and the small files matched the pinned revision's Git objects. A portable
reference now lets us repeat local verification without authentication or
network access. No Hub token or credential cache was inspected.

The checkpoint is about 2.47 GB. The original parameter file describes the
selected 1B configuration; the tokenizer has 128,000 valid base entries with
unique tokens/ranks. These checks read bytes and JSON rather than loading a
Python checkpoint. Six integrity tests and the full 58-test suite passed.

Source files are ready for export preparation, but no export or inference was
performed. The separate export environment, native runner build, and actual
Jetson execution remain next-stage evidence, not consequences of a download.

## Stage 2.5: Prepare the conversion workshop

A virtual environment gives this exporter its own tools, so its newer PyTorch
does not replace the GPU tools or the small image environment. The source Git
commit pins the program; the package lock pins its dependencies. CPU wheels
are sufficient for the startup/config checks performed here and avoid a second
CUDA stack. This does not establish how fast the eventual Jetson model runs.

Actual checks passed: CPU matrix multiplication returned 11.0, source/installed
schemas matched the reference hash, our recipe became a real upstream LlmConfig,
and the pinned checkout's exporter printed its actual help with exit code zero.
A fresh temporary installation from the lock repeated those checks. Full
inventories show both older environments unchanged; all 59 project tests pass.
The CPU build suffix is now recognized without accepting another release.

Optional TorchAO GPU libraries reported load warnings. These did not prevent
startup, but quantization compatibility remains unproven until real conversion.
No checkpoint was loaded, no model exported, and no text generated. Next:
attempt the real export and inspect its result before building a device runner.

## Stage 2.6: Convert source weights into an ExecuTorch program

Source weights contain learned numbers. Export combines those numbers with a
computation graph in the runtime's PTE format. The selected 8da4w path uses
dynamic 8-bit activations and 4-bit linear weights; embeddings use 4-bit groups
of 32. Other operations can retain higher precision. Smaller storage alone
does not establish acceptable advice quality or faster execution on Jetson.

We kept the reviewed XNNPACK recipe instead of switching to a larger-precision
fallback or another runtime. The real conversion succeeded in 248.46 seconds
on desktop WSL and produced 802,247,040 bytes. The file's recorded hash detects
later byte changes. ExecuTorch loads its program metadata, including forward,
token metadata, and an XNNPACK delegate. No method was executed or text generated.

The real run exposed a filename rule missed by configuration-key checks:
an output name ending in .pte bypasses output_dir. Preserve that first output,
verify an identical local copy, and use a stem in future recipes. The actual
upstream resolver confirms the corrected path without another full conversion.

All 63 tests pass, including corruption, wrong-format, missing-file, and
reference-mismatch guards. These tests reject generic invalid files; they do
not simulate a successful model. Both original environments remain unchanged.
Next is a real desktop generation smoke test; hardware, quality evaluation,
offline operation, and the mandatory Jetson proof still need separate evidence.

## Stage 2.7: Generate actual text through ExecuTorch

A tokenizer turns our prompt into integer IDs. ExecuTorch runs the exported
graph to predict the next token, then repeats using its cache until a stop token
or limit. A chat wrapper tells this instruction-tuned Llama where the user turn
ends and the assistant answer begins. BOS is included once; echo is disabled
so a repeated input cannot be mistaken for generated output.

We used the wheel's real native TextLLMRunner through Python bindings instead
of building another runner or substituting a different runtime. The first
attempt failed before generation: the wheel's quantized embedding and Llama
cache/attention kernels were present but not registered. Loading those supplied
libraries explicitly resolved it. No model/package/source change was needed.

Actual output explains that consistent lighting makes image comparison more
accurate and reliable, without claiming an inspected part. The tokenizer counted
33 prompt tokens; native Stats reported 30 generated tokens and the raw callback
included the end-of-turn marker. The final generate call took 4.08 seconds,
including lazy model loading; native inference timing alone was 1.154 seconds.
These single desktop observations cannot predict Jetson latency or p50/p95.

Raw logs, failed-attempt evidence and token callbacks remain local. The committed
smoke record includes the actual prompt, answer, native statistics and binary
fingerprints. All 67 tests pass; new unit tests cover input/control/token-budget
guards without simulated generation. Desktop generation works, but quantized
advisory quality, actual Jetson execution and offline end-to-end inspection
remain open. Next, select a compatible vision baseline in Stage 3.1; fitting
or training follows that selection.

## Stage 3.1: Choose for a useful accuracy/speed balance

Model inference time is different from training time and from camera-to-result
latency. A model can be quick to fit yet costly to run: PatchCore searches a
normal-feature bank at inference. EfficientAD pays optimization training cost
once, then runs fixed teacher/student/autoencoder networks. Neither approach's
accuracy follows from parameter count or a successful forward pass.

We measured real untrained model code with synthetic inputs, not a fake defect
detector: zero dataset pixels and optimizer steps. Fifty samples after ten
warmups provide model-only desktop p50/p95, not Jetson or whole-app performance.
Compact PatchCore was faster in FP32, while EfficientAD-S used fixed parameters
and had a useful lower-precision execution path. Larger models and higher input
resolution cost more. Smaller banks or lower precision can change accuracy,
so the fastest timing is not automatically the right model.

EfficientAD-S 256 is selected for first training, with PatchCore-ResNet-18 kept
as a fallback. Thin scratches may disappear when resizing 700 to 256; higher
resolution needs trained-quality checks and recalibration. Neither map branch
supplies defect-type labels or deterministic severity. Llama stays a separate
asynchronous ExecuTorch worker and its real Jetson proof remains mandatory.

The isolated GPU environment and fresh lock-only CUDA checks pass; both model
modules match pinned upstream source. All 70 tests pass, including integrity of
percentile reporting and memory-bank sizing. Teacher and Imagenette training
assets are still pending. Next: verify those assets and train on the 176 frozen
normal fit images, preserving 44 normal validation images and the final holdout.

## Stage 3.2: The first model has actually learned from normal parts

The pretrained teacher produces visual features; its weights stay frozen. The
student and autoencoder learn from our normal fitting images. The unchanged
upstream losses also use generic Imagenette images to discourage imitation
outside the normal domain. Those generic labels do not teach metal-nut types.

Both downloaded archives matched publisher hashes and were safely extracted.
Only 176 fitting normals entered domain training and teacher statistics. The
44 normal validation images were reserved without reading their pixels; no
development or final-holdout images were read. We ran the predeclared 10,000
FP32 optimizer steps, a first baseline rather than a full paper reproduction.

Training finished in about 19 minutes. Recent mean total loss fell from 16.31
near step 100 to 2.27 over the final 100 steps. Teacher weights remained identical;
student and autoencoder weights changed. The actual learning curve is saved,
and the final checkpoint reloads independently with exactly the same saved
fit-image output. All 78 tests pass and existing environment inventories match.

This establishes training and checkpoint integrity, not accuracy: a model can
fit its normals and still miss unseen defects. The checkpoint is uncalibrated,
with no pass/reject threshold. Next, use the 44 reserved normals to calibrate
the map branches and document a threshold policy. Final held-out accuracy,
defect type/severity, trained low-latency deployment and all Jetson/offline
requirements remain separate evidence.

## Stage 3.3: Turn two map signals into a reproducible candidate rule

Calibration changes scales and a cutoff, not the learned teacher/student/AE
weights. We used 22 reserved normal images to make the two map branches
comparable, then 22 different normals to set an image-score threshold. Keeping
those jobs separate avoids claiming independent threshold calibration after
fitting the score function on the same examples.

The declared rank policy sets a provisional threshold of 0.2277658. A score
strictly above it is an anomaly candidate; equality remains a normal candidate
and nonfinite values are errors. Scores are not probabilities, severity or
final quality grades. The 5% nominal level relies on assumptions that are not
verified production behavior, and 22 threshold normals cannot support a finite
1% rank cutoff.

The threshold-setting normals have zero flags by construction. Three map-fit
normals were flagged; their paths/scores remain in the report. These are
calibration diagnostics, not independent accuracy. No defects or holdout pixels
were read. The saved bundle reloads with exact anchor scores and unchanged
trained weights/statistics. All 86 tests pass. Actual calibrated desktop FP32
model-only p50/p95 is 14.54/14.80 ms; image I/O, transfers, app work, Llama and
Jetson behavior remain outside that timing. Next is the frozen-policy final
holdout evaluation to measure real misses and false alarms.

## Stage 3.4: Test on pictures the model did not train on

The model flagged 87 of 89 defective nuts and two of 22 good nuts. It missed two
scratches. Saving every result lets us see failures instead of just attractive
heatmaps. We froze the cutoff before testing: changing it now would turn the
test set into another tuning set. AUROC describes ranking across possible
cutoffs; it does not mean 97% of each defect's physical area was measured.
Next we turn each real model result into one record for one declared part.

## Stage 4: A camera frame is not a new nut

One stationary nut can appear in many frames. We now accept one attempt and
wait for a new-part signal. In the real model replay, nine frames gave four
records, including one unreadable-input error. Two declared nuts used the same
picture and still counted separately. Regions report response pixels and box
size, not millimeters. Next a separate, predictable rule decides what action
to suggest; the language model will never own that decision.

## Stage 5: Make the action predictable, and keep uncertainty visible

The program now adds a repeatable action to each saved inspection. The same
picture evidence and rule file always give the same action. A clear normal
candidate gets a provisional PASS; an unusual picture with unknown defect type
gets REVIEW, as does a failed inspection. The real replay produced two passes
and two reviews, with repeated frames still suppressed.

We tested example rejection rules with controlled size/location fixtures; those
are tests of rules, not a newly trained classifier. Highlighted pixel area is
not true damage area or millimeters. Applying the rules to the existing test
caches still provisionally passes the two missed scratches. A rule engine cannot
repair evidence that the detector missed. Real manufacturing tolerances, defect
names, camera/Jetson tests and offline Llama advice remain future proof. Next,
Stage 6 will store records so history can be queried after restarting.

## Stage 6: Keep inspection history after closing the program

SQLite is a local notebook the program can search reliably. A unique inspection
ID prevents saving the same result twice. A unique declared part ID also stops
a restart from treating an old part as a new one. Photo hashes do not identify
physical parts: two distinct good nuts can look identical.

The real replay saved four results, including an input error. New Python
processes read all four back unchanged; four resave attempts added no rows.
A changed result or reused part ID was refused. Correction notes stay separate
and unverified, so a proposal cannot silently rewrite the original evidence.
The demonstration correction is explicitly synthetic, not a real judgment.

We still save the JSON first. If database writing fails, the original result
can be imported later without inspecting the image again. Abrupt power-loss
recovery is not tested. All 112 software tests pass; the next stage will let
an operator inspect images and view this stored history on a small local screen.

## Stage 7: Connect a button to the actual inspection pipeline

The screen is now a working front door to the model, rules and database.
Clicking Inspect sends the declared part/photo IDs to the local backend. It
runs the real detector, saves one result and returns the actual image response
and action. Refreshing asks the database for history; it never reruns the model.

A browser inspected a normal example and an anomalous example, saving one
provisional PASS and one REVIEW with unknown defect type. Repeating the same
part was refused. The records and images still appeared after restarting the
whole server. A narrow-screen layout bug was found and fixed using the existing
records, without repeating inference. All 119 software tests pass.

Everything shown is saved-image replay on the PC. Camera capture, real physical
part identity and Llama advice are clearly pending. Next we need the actual
Jetson and controlled camera samples, because a familiar benchmark picture is
not evidence that a different camera/lighting setup will work equally well.

## Stage 9.1: Give one inspection to the real local language model

We took a saved anomaly whose fixed decision was REVIEW and asked the actual
Llama model, running through ExecuTorch, for a source-linked note. No image was
sent: the input was a short description plus a labelled prototype checklist.
The final output repeats the unknown type/cause and asks for human review and
image clarity/lighting checks. It does not diagnose a machine fault.

Testing exposed real limitations. Early answers were too long, empty, badly
formatted or unsupported, so they stayed unavailable. A later faithful answer
was blocked by a vocabulary-check bug; correcting it allowed the identical text.
All failures are saved. A single final example does not prove advice works
reliably across the application. Code preserves the original grade and input;
128 tests pass, including timeout/failure and citation/claim checks.

The small exported model and memory/context limits stayed unchanged, with no
package installation. The screen still says advice is not connected. Next we
will evaluate the advisory contract on declared scenarios before adding its
background worker and UI delivery. Jetson and camera proof remain blocked.

## Stage 9.2: Checking words is not enough to check meaning

"Cause unknown" and "the cause is lighting" can use the same small word list
while saying very different things. A handwritten example exposed that gap in
our old check. The first prototype now accepts only two reviewed ways to state
the recorded facts and checklist. That is less flexible, but it prevents this
particular unsupported claim. It does not prove general reasoning or diagnosis.

Normal and failed inspections do not need this anomaly-note generator. They
now get a saved refusal reason without loading Llama. Changed procedures, fake
defect names and inconsistent evidence are refused too. The normal inspection
and failure still keep their original decisions and history.

All 27 contract checks matched; 132 software tests passed. Two actual Llama runs
gave the supported note using the same prompt. An intentionally tiny output
budget was safely refused, but by an earlier runtime check than we predicted;
that initial experiment remains 31/32, not a claimed perfect score. A separately
declared slightly longer incomplete answer reached the text checker and was
refused there too. Keeping unexpected results helps us understand the system.

The model, compact prompt, package environments and vision evidence stayed
unchanged. We still need verified industrial procedures for specific precautions
and confirmed process evidence for causes. The next task is a small background
queue so slow advice cannot hold up inspection; actual Jetson behavior requires
device-side tests when hardware arrives.

## Hardware arrival: inspect the actual device before deployment

On 2026-11-11 the user reported receiving an Orin Nano 8GB and a Logitech USB
webcam. We can now return to the open hardware steps. First we need to know
whether it boots and how to connect; then we read its actual system information
without installing anything. Owning a webcam does not yet prove it captures
usable inspection images, and PC text generation does not prove Jetson execution.
The reported hardware is documented separately from measured device facts.

## First actual Jetson audit: hardware is present, deployment is still separate

We authenticated to the Jetson and read its real configuration without installing
anything. It is the 8GB Orin Nano variant, with six CPU cores, SD storage and a
Logitech C270 webcam. Linux reports 7.28 GiB of memory, with about 4.28 GiB available
at that moment. Available memory is a snapshot, not a guarantee that both models
fit together under load. Two video nodes do not mean two physical webcams.

The Jetson has Ubuntu 20.04/Python 3.8/CUDA 11.4. The PC uses different software
and a different processor architecture, so copying its environment is not a
deployment recipe. Our pinned ExecuTorch Python package needs newer Python; we
will investigate a separate environment or native runner before installing.
This does not automatically mean the Jetson needs a new operating system.

The audit confirmed camera detection and existing software, not defect detection
or Llama execution. A power query had permission errors; camera-format tooling
was missing. We saved those gaps. The initial terminal mangled long output, so
the final inventory was transferred in short encoded lines and checked against
its device-reported SHA-256. Next resolve the ARM runtime recipe, then prove
real Llama generation on this device. Stop at this hardware-inventory gate.

## Substep 2.8: Check the parts before installing the runtime

An ARM label alone does not tell us whether a package works with our Jetson's
system libraries. We inspected the official packages and compared 33 binary
files with ten real libraries on the Jetson. Those checked requirements match.
We also froze 49 dependency versions, and tested building one Python-only
dependency on the PC. The Jetson itself received no installation or file changes.

Our first candidate is a separate Python 3.12 environment for CPU ExecuTorch,
keeping system Python 3.8 and the camera/driver setup intact. It follows the
working PC reference while leaving GPU vision packages separate. We still need
actual imports, model output and memory/speed tests before calling it compatible
in practice. Static evidence cannot prove the whole program works.

The official interpreter download has failed so far. Its expected hash is saved,
but we cannot call the file verified without its bytes. Next obtain/check that
file, then create the separate environment. All 140 software tests pass; no
Jetson model execution or offline proof is claimed.

## Interpreter download follow-up: verified file, still no installation

The official interpreter download now works and matches its expected size and
SHA-256. We inspected its contents on the PC: the archive stays inside its own
prefix, contains ARM64 binaries, and includes the files needed for virtual
environments. We also checked the two extra system libraries on the actual
Jetson. All 43 checked interpreter/runtime native files match 12 system-library
inventories. This still does not prove that imports or Llama generation work.

Downloading and inspecting an interpreter does not replace system Python.
The Jetson still reports Python 3.8.10, with unchanged package/library snapshots.
Next create a separate user-owned interpreter/environment and test it directly.
The prior 140-test PC result remains valid history; this follow-up verified the
new artifact and target-library evidence without a broad software-test rerun.

## Substep 2.9: build and test Llama's separate workspace on Jetson

The Jetson now has a separate place for the CPU Llama runtime. Think of this
as preparing a workbench before putting the model on it. The original system
Python is still 3.8.10; only the new workbench uses Python 3.12.14. Keeping it
separate protects the working Ubuntu/camera setup and leaves future GPU vision
dependencies independent.

We checked every downloaded package against its saved fingerprint, installed
from local files, and checked that the packages agree about dependencies.
The actual Jetson calculated a small tensor result (10.0), loaded ExecuTorch's
native bindings, and found the required operators after their libraries loaded.
This means the tools can start; it does not mean Llama has answered a question.

All 147 software tests pass, including checks that refuse existing folders,
unsafe archives and changed hashes. Two preparation mistakes were caught before
changing the device and fixed. System snapshots match before/after; the existing
PC environments and protected project/model files also match their earlier record.
Logs and outputs are saved, and the working SDK should be kept on the device.

Next is 2.10: connect the ARM device runner to the verified model/tokenizer,
then ask one short question and save the real response, time and memory use.
Camera inspection and a disconnected full-system demonstration come later.

Knowledge check: why does successfully loading the runtime still fall short of
proving that Llama generates text on this Jetson?

## Substep 2.10: Llama answered on the actual Jetson

We copied the already prepared Llama 3.2 1B model and its tokenizer into the
Jetson's private ForgeSight folder. Their fingerprints matched the PC copies.
The Jetson then answered a short question about steady lighting. ExecuTorch
reported 33 input tokens and 40 new answer tokens, so we now have actual
on-device generation rather than only a ready runtime.

One run used two CPU threads. Loading the model took 3.087 seconds and native
inference 2.129 seconds; the full program took 30.856 seconds including startup.
Its sampled peak process memory was about 1.13 GiB. Logs, text and hashes are
saved on both machines; 151 software tests pass. The answer was a simple smoke
test while Wi-Fi was connected. It does not tell us how well the full inspection
system works or how it behaves without internet.

Next we inspect the Logitech camera's real image formats and determine how to
run the already trained vision model on the Jetson. Later we connect each real
inspection to the Llama advisory and test the whole system disconnected.

## Stage 8.1: What the Jetson camera can deliver

The Jetson recognized the Logitech C270. Its camera driver offered MJPEG and
YUY2 video; MJPEG includes 640x480 at 30 frames per second as an advertised
setting. We asked the Jetson's existing OpenCV to read one live frame and got a
640x480 color image. That proves the software can access the camera. It does
not prove a nut was inspected or that 30 complete inspections happen each second.

The vision model needs extra packages unavailable in the existing private Llama
environment. We will set up and measure the vision path in the next substep,
keeping the working Llama setup intact. Knowledge check: why is a camera's
advertised frame rate different from the speed of a full inspection?

## Stage 8.2: A model result from one saved image

The calibrated EfficientAD model was exported with the same fixed 256x256
input and checked against its source on two already used training images. We
then moved the export and one training image to a new private vision runtime
on the Jetson. File hashes and the prepared image tensor hash matched across
machines. The Jetson returned an anomaly score and map in one CPU run.

The 1.699-second forward is a first-run measurement, not a frame rate or a
typical latency. The process used at most 431,704 KiB according to Linux's
peak RSS counter; that does not include a simultaneous Llama process. The
next check uses an actual captured nut and measures the complete path through
inference and logging. Knowledge check: why can this familiar training image
prove the model runs while telling us little about real camera accuracy?

## Stage 8 camera diagnostic: capture needs exposure time

The first live frame was almost black. Watching several frames showed that
the C270 needed about 12 frames to adjust exposure. We added a bounded guard
that rejects grossly dark startup frames, then saved a visible live frame,
its model response map and an overlay. This proves the camera and model can
run in sequence on the Jetson. The camera currently sees a room, and no nut
is available, so its high anomaly score is not a defect finding. The complete
diagnostic took about ten seconds in this one run, including two child-process
startups. Next we need a fixed lit surface and a real distinct part.

## Stage 8 saved-nut replay: a full stored result on Jetson

We used one already exposed development image of a metal nut to exercise the
Jetson model, deterministic grade and SQLite together. The record says
`SAVED_IMAGE_REPLAY`, and a new process read the exact stored record back.
The model response crossed the score threshold, but it does not classify
the defect type. The provisional rule therefore said `REVIEW`, which is a
request for human inspection, not a certified rejection. The source-grid map
and overlay are saved privately. The next proof still requires a real nut
in a fixed camera view; this replay cannot establish camera-domain accuracy.

## Stage 9.1: on-device advice stays separate from the grade

The Jetson's Llama model read only compact facts from the stored replay and
the local prototype image-check checklist. The first prompt omitted a needed
citation, so the host refused it. With an explicit system role, ExecuTorch
generated the cited note; the host accepted only the reviewed wording. A
missing-runtime trial saved an unavailable status instead. In both cases the
original `REVIEW` decision and SQLite row stayed unchanged. This is a first
device generation from inspection evidence, not proof of useful factory
diagnosis, asynchronous delivery or offline operation.

## Stage 9.4: keep a slow advisor outside the inspection decision

The Jetson page returned the saved ForgeGrade decision before Llama finished.
One worker ran the real ExecuTorch controller; another request got `BUSY`.
The browser saw pending then a cited note, and status reads still worked.
With a one-second native deadline, the worker recorded `TIMEOUT` and the page
showed advice unavailable without touching the SQLite inspection. The visible
state must come from the worker/report, while the grade remains independent.
These runs use a saved image and do not measure physical capture throughput.

## Stage 10.1: check the network boundary outside model reports

The old Llama runner prints a connected-device default offline field even when
called during a later disconnected run. Its field is historical metadata, not
this run's network measurement. The detached pilot sampled NetworkManager,
Wi-Fi/ethernet addresses and default route around the actual vision and advice
processes. Both produced local results with no upstream route in those
snapshots, then Wi-Fi returned through the recovery path. One replay cannot
support accuracy, throughput or camera claims.
