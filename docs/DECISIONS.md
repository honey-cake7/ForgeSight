# Decisions and unresolved choices

## Stage 10.2: Restrict the normal replay reference

Use only the frozen manifest's `train/good/000.png` for a normal runtime
branch check, behind an explicit replay flag. Keep final holdout images
unavailable to this workflow. Reuse two exposed development images to check
the anomaly/REVIEW path while Wi-Fi is disconnected. Record the dataset role
with every inspection, and do not count these three runs toward new accuracy
or distinct physical-part throughput. Keep the independent Wi-Fi reconnect
timer and sample interface/route state throughout the run.

## 001: Preserve the existing workspace in place

The audited directory contains three project instruction documents and the
original architecture PDF, with no Git repository. GitHub's branch list was
empty when checked again before initialization.

Initialize local Git on `main` in this directory and configure
`https://github.com/honey-cake7/Forge_Sight.git` as `origin`. This preserves the
existing paths and contents. A local commit and remote configuration do not
upload anything. Push only after user authorization and an actual credential
check. Keep the original PDF locally outside source history.

## 002: Follow the stage roadmap and preserve evidence boundaries

The roadmap overrides the master prompt and reference PDF when they conflict.
Distinguish PC results, Jetson results, and disconnected Jetson end-to-end
evidence. Jetson unavailability blocks device proof, while independent PC work
may proceed after its own gates are reviewed.

## 003: Use one component domain first

Start the dataset prototype with MVTec AD `metal_nut`, subject to Stage 1's
license and split review. Physical camera images form a separate domain and
require controlled real samples. Do not merge unrelated steel/PCB benchmarks
into the metal-nut evaluation.

## 004: Keep vision decisions separate from language advice

Proposed later flow: capture one distinct part, run vision, create a structured
inspection record, apply versioned deterministic grading, and persist locally.
A separate bounded queue feeds structured evidence to an asynchronous Llama
worker running through ExecuTorch on Jetson.

Llama cannot set or override PASS/REVIEW/REJECT. Advice must retain uncertainty
and evidence provenance. A worker timeout must leave inspection usable and
mark advice unavailable. SQLite and a minimal local API/UI remain roadmap
choices for later stages, not implemented features.

## 005: Treat the PDF as a technical reference

Preserve its separate vision/advisory paths, anomaly localization, compact
telemetry, local logging, and evidence-linked procedure retrieval.

Replace or verify its Raspberry Pi/Hailo runtime and CPU-affinity assumptions
for the actual Jetson/JetPack combination. Begin geometry in pixels; homography
alone does not establish a calibrated physical scale. Its two anomaly maps do
not prove named defect types, certified grades, or manufacturing causes.
Neither its 8-15 ms/60 FPS figures nor its FP16 and sample-count assumptions
are measured project results. Defer lock-free buffers, SPC, and advanced
retrieval until their complexity is justified at the corresponding stage.

## 006: Defer model selection until compatible runtime tests

EfficientAD is the provisional deployment candidate: teacher/student and
autoencoder discrepancies provide local and global anomaly evidence, with a
training and export-validation cost. PatchCore is the reference alternative:
normal-feature memory banks simplify fitting but incur storage and search
costs. PaDiM is a statistical alternative using patchwise Gaussian models;
covariance storage and spatial consistency need evaluation.

These are design considerations, not measurements. Stage 3 decides using
compatible dependencies and leakage-safe `metal_nut` evaluation. No method is
selected, trained, exported, or benchmarked yet. Defect-type classification
remains a separate requirement.

Primary technical references reviewed during the audit:

- [EfficientAD paper](https://arxiv.org/abs/2303.14535)
- [PatchCore paper](https://arxiv.org/abs/2106.08265)
- [PaDiM paper](https://arxiv.org/abs/2011.08785)

## 007: Preserve actual Llama + ExecuTorch device inference

Llama 3.2 1B Instruct is a candidate subject to licensing/access and actual
device memory/runtime checks. Linux ARM64 CPU execution through XNNPACK is a
feasibility route, not a verified Jetson implementation. Quantization and
KleidiAI behavior must be verified for the chosen versions and device.

Stage 2 requires actual Jetson-generated text, artifact hashes, runtime/model
versions, exact commands, memory, latency, and network state. Exporting a
`.pte` file, running on this PC, or substituting another LLM runtime does not
meet that gate.

Reference: [ExecuTorch XNNPACK documentation](https://docs.pytorch.org/executorch/stable/backends/xnnpack/xnnpack-overview.html).

## 008: Isolate the minimal PC image environment

The existing WSL environment imports CUDA-enabled PyTorch and discovers the
desktop GPU. Preserve it and use its CPython 3.12.14 interpreter to create a
separate project `.venv` without system-package access. Windows-native Python
would create a different environment; using WSL keeps the first image tools on
Linux while remaining explicitly a PC development result.

Substep 0.3 installed only Pillow 13.0.0, pinned by version and the official
CPython 3.12 Linux x86_64 wheel hash. Reading and inspecting images does not yet
require NumPy, OpenCV, matplotlib, PyTorch, or ExecuTorch. Add dependencies only
when their roadmap tasks need them. This lock targets the tested WSL platform,
not Jetson. A second fresh environment reproduced the setup and PNG smoke check.

## 009: Separate development previews from final evaluation

Stage 1.1 reviewed the publisher's dataset/license and official evaluation
reference. `docs/DATASET.md` records the source evidence, expected layout, and
versioned deterministic partition policy before any local images are viewed.

Reserve 20% of normal training images for validation and one anomalous test
image per defect folder for development previews. Keep the rest of the source
test set for final evaluation. Expected roles are 176 fit, 44 normal validation,
4 development previews, and 111 final holdout. Verify counts and duplicate
content before accepting an actual manifest. The trade-off is a smaller custom
holdout, requiring explicit disclosure rather than claiming the full official
benchmark protocol. Normal-only calibration cannot optimize defect recall.

Use benchmark data for the noncommercial research prototype under its stated
CC BY-NC-SA 4.0 terms. No commercial or competition-specific licensing conclusion
is claimed. Raw data stays outside Git. Camera samples and model artifacts need
their own provenance review. The four test labels do not supply a trained
subtype classifier; this requirement remains open.

## 010: Audit the original category ZIP and preserve local copies

The user acquired `metal_nut.zip` through the official MVTec flow and supplied
its local location. Copy the original archive into ignored `data/downloads/`,
verify source/copy hashes, audit all entries/PNGs/masks, then extract only audited
regular files into a new ignored category directory. Preserve the original ZIP
and the user's already extracted Downloads folder. Record the exact archive
fingerprint and observed counts in `docs/DATASET.md`.

Support ZIP and tar because the actual supplied category archive is ZIP. Use
only Python's standard archive libraries and the already installed Pillow.
Reject unsafe paths, links, corrupt files, and mask-pairing errors; refuse
existing extraction destinations and report files. Check extracted-file hashes
instead of assuming an extraction command succeeded. Fourteen synthetic tests
exercise this behavior; the real archive independently passed all checks.

No publisher-provided checksum or exact download URL was supplied. User-confirmed
publisher origin and our measured fingerprint are recorded as different evidence.
No image visualization, partition assignment, or inference belongs in this step.

## 011: Freeze metadata, verify role-based inputs, and label preview evidence

Substeps 1.3-1.5 implement the previously recorded partition policy without
changing seed, counts, or preview selection. Commit portable metadata and its
checksum; keep source images and generated contact sheets outside Git. Byte and
decoded-pixel duplicate checks passed; near-duplicates and physical identity
are still not established.

Use Pillow and the standard library for loading and display. Preserve source
resolution, verify file/pixel fingerprints on load, and gate final-holdout
access with explicit evaluation opt-in. A normal sample gets an explicit zero
mask; a missing anomalous mask fails. Avoid adding model dependencies until
their roadmap stage requires them.

Render only one fitting normal sample plus the four frozen development
examples. Mark labels and overlays as ground truth and retain attribution.
Nearest-neighbor resizing preserves binary mask display; source inputs are
unchanged. Annotation area in pixels is not a physical severity measurement.

The user explicitly authorized these three steps and a push after each verified
commit. The first two pushes succeeded before starting the following step;
the third step's final report records its actual push result. Preserve the
concurrent local master-prompt edit outside the implementation commits.

## 012: Prepare Stage 2 without fabricating blocked execution

The user accepted Stage 1 and authorized three further steps with pushes.
The Jetson is unavailable and the user reports no Meta Llama access. Pin the
1B model revision and ExecuTorch 1.5.1 source, use the upstream XNNPACK candidate,
and keep actual export/build/inference explicitly blocked. Public source APIs
and stable documentation differ in current release labels; record that and
validate recipe fields against a specific commit rather than mix latest APIs.

Preflight reads host facts, recipe generation emits commands without execution,
and capture requires the actual ARM64 Jetson candidate and local native runner.
Do not alter either existing Python environment or download gated weights.
Build dependency installation/locking remains unverified. Capture uses only
local artifacts and never substitutes another model runtime.

Capture records raw evidence and process-level measurements, not automatic
compliance. Model and binary provenance, actual generated content, memory
sampling limitations, and network state need review. The PC refusal and generic
child-process tests validate tooling; they are not an LLM simulation or Jetson
proof. Preserve the mandatory gate while independent PC vision work continues
after review, as the roadmap permits.

## 013: Verify source assets before creating an export environment

The user obtained Meta Llama access and downloaded the exact original files
requested by the recipe. Verify publisher LFS SHA-256 and Git blob fingerprints
at the pinned revision, then commit only a portable hash/parameter reference.
Keep all binary assets and Hub cache files ignored. Do not deserialize the
checkpoint or install PyTorch merely to confirm download integrity.

The offline asset verifier rechecks streaming file hashes, model parameters,
and tokenizer structure against that reference. All five real files matched;
six new integrity tests and all 58 project tests passed. Clear the model-access
and missing-file blockers while preserving export/build/device NOT VERIFIED.
Render an updated recipe into a new ignored directory so the earlier evidence
is preserved. The next separate task prepares an isolated export environment.

## 014: Isolate and lock the CPU exporter before converting weights

Use a new venv and pinned ExecuTorch checkout under WSL's Linux filesystem,
outside both the image environment and shared CUDA environment. CPU wheels
avoid adding CUDA packages to a conversion-only environment; this does not
choose or verify Jetson GPU execution. Use the pinned release's actual
dependency requirements, not the desktop's older shared PyTorch installation.

Lock 86 resolved dependencies using uv 0.12.15 and PEP 751 pylock format,
including distribution hashes and torchtune's exact Git commit. This uv format
is experimental; retain the input specification and tested tool version.
The lock targets this CPython 3.12/Linux x86_64 setup, not Jetson or Windows.
VCS source is pinned by commit, not a package-distribution hash; compiler,
Python, OS, and build-isolation tools are not made reproducible by this lock.

Verify real CLI startup and structured configuration parsing without loading
weights. TorchAO's optional GPU library load failures are recorded rather than
suppressed; neither successful imports nor a dependency check proves the later
8da4w quantization/export path. Accept official +cpu build suffixes when checking
the selected ExecuTorch release, while rejecting other releases/prereleases.
The actual model conversion is the next separate step.

## 015: Preserve actual export provenance and verify the serialized artifact

Export the existing 8da4w XNNPACK candidate rather than change precision or
runtime before observing a real attempt. Actual conversion succeeded with the
isolated CPU tools; retain all warnings, command, config hash, input references,
exit status, elapsed time, and sampled memory. Export timing/memory describe
this desktop process only. Commit metadata, not the licensed checkpoint/PTE.

Upstream treats output_name ending in .pte as a verbatim path and ignores
output_dir. The successful first artifact therefore landed in the source
checkout. Preserve it, copy it into ignored project artifacts, and confirm both
fingerprints. Correct future recipes to use an extension-free stem and verify
the resulting path with upstream's actual resolver. Do not claim a second
conversion occurred to validate that path correction.

Use a streaming fingerprint and pinned ET12 identifier, parse bounded metadata
with the pinned release's private serializer helpers, require a serialized
XNNPACK forward delegate, then perform the real runtime's default
InternalConsistency program load. Private APIs require review on any runtime
upgrade. No method is executed; listing methods is not generation or operator
coverage proof. A newly re-exported artifact may have a different hash and needs
its own reviewed provenance rather than silently replacing this reference.

## 016: Record organizer acceptance of the Jetson Orin route

The user supplied an organizer reply accepting NVIDIA Jetson Orin as an eligible
PS5 platform, subject to Llama-family advisory/reasoning, on-device ExecuTorch,
and on-device defect/quality-control operation without cloud dependency.
Preserve the supplied reply in `docs/ORGANIZER_CLARIFICATION.md` and continue
the planned Orin Nano route. Keep platform eligibility separate from the
unverified physical hardware inventory and actual device compliance.

The reply also explicitly requires the technical report to document hardware,
software, deployment, and inference flow. Track that report requirement without
inventing a Jetson configuration or advancing the implementation stage. The
current desktop export is preparation; next remains the desktop generation
smoke test, with actual disconnected Jetson proof still mandatory.

## 017: Use the native wheel runner and register its supplied LLM kernels

Use ExecuTorch 1.5.1+cpu's TextLLMRunner C++ bindings in the existing isolated
environment. This avoids an unnecessary source rebuild or a hand-written
generation loop. It does not replace the device-only capture guard or change
the selected model, quantization, runtime, or mandatory Jetson gate.

The first real attempt failed because embedding_4bit.dtype_out,
llama::update_cache.out, and llama::custom_sdpa.out were not registered. The
wheel already supplies them in its quantized and LLM custom-op shared libraries.
Explicit torch.ops.load_library calls register them with the shared runtime;
real generation then succeeds without dependency changes or compilation.
Record all three native binaries' fingerprints and preserve the failed attempt.

Use a controlled Llama chat prompt, one explicit BOS, no added BOS/EOS, greedy
temperature zero and echo disabled. Verify the matching tokenizer, encoded
control IDs and strict prefill/context budgets. Capture token callbacks and
native statistics in a bounded child; generated text is reviewed separately
from runtime exit status. Preserve the raw end-of-turn marker in the evidence.
The result is a PC functional test, not advice-quality or offline evidence.

All 67 tests pass; unit tests exercise prompt/control/budget guards without
fake model replies. Both older environment inventories remain unchanged. Next
is Stage 3.1 vision-baseline compatibility/selection under the roadmap's device
blocker rule; stop for review before that different stage.

## 018: Balance accuracy risk, latency and fixed deployment cost

The user prioritizes good accuracy with lower inference time and sets no hard
latency or production-rate limit. Select EfficientAD-S at 256x256 for first
training; keep PatchCore-ResNet-18 as the fallback/comparator. This is not a
trained accuracy comparison or final device model selection.

Actual synthetic probes showed compact PatchCore faster in FP32 on the desktop,
but its search cost/memory grew with the bank. EfficientAD-S has fixed parameter
storage, no growing feature-bank search, measured lower GPU allocation than the
larger-bank comparison, and an executable AMP path. Medium/higher-resolution
variants cost more without measured quality gains yet. Keep all observations,
including variation between AMP runs, rather than select the fastest number.

Train/evaluate with FP32 reference and frozen data roles. Only adopt lower
precision after trained-score/map/threshold checks; 256 can hide thin scratches
and 384 requires separately validated training/calibration. Keep subtype and
severity gaps open. Teacher/Imagenette assets and their terms/hashes require
verification before fitting; do not silently change the loss or teacher.

The separate Anomalib 2.6.2/CUDA 12.4 environment matches the tested stack.
Installed model sources match the pinned commit; the locked setup reproduces.
All 70 tests pass. Jetson memory/latency, TensorRT, Llama contention and end-to-end
performance remain unverified. No pretrained model or dataset pixels were used
in the architecture timing probes. See `docs/VISION_BASELINE.md` for sources,
alternatives and measurements. The next task is actual first training.

## 019: Train an explicit first baseline without consuming evaluation data

Use the pinned EfficientAD core's three unchanged loss components in a small
PyTorch loop, rather than a default framework data module that could expose
the original test split. Domain optimization and teacher statistics use only
the 176 frozen fitting normals. Metadata reserves 44 normal validation samples;
their pixels and all development/final-holdout pixels stay unused here.
Imagenette train images provide generic regularization with labels discarded,
not defect supervision. Verify archive hashes before bounded extraction and
strict, weights-only teacher loading; keep all raw assets and binaries private.

Fix a 10,000-step first-baseline budget before evaluation, FP32 Adam with the
pinned rates and 95%-budget scheduler. This is not full paper reproduction or
proof of convergence. Compute teacher statistics with float64 accumulation
and a declared numerical floor (no channels needed clamping). Record seeds,
source/data hashes and any reproducibility limits rather than promise bitwise
CUDA retraining identity. Calibration and decision threshold are deferred.

The run completed in 1,158.35 seconds. It preserves 1,000/5,000/final checkpoints,
checks finite loss/gradients, and proves frozen teacher plus changed student/AE
weights. The final checkpoint independently reloads with exact state digests
and the same fit-anchor output. All 78 tests pass and all four environments'
package inventories are unchanged. No defect accuracy, type, severity, optimized
latency or device compliance is inferred. Next is normal-validation calibration
and threshold policy before any final-holdout evaluation.

## 020: Separate map fitting from normal-score threshold calibration

Use a deterministic 22/22 subdivision of the existing 44 normal validation
samples. Keep the outer frozen manifest and the trained networks/teacher
statistics unchanged. Fit upstream q90/q99.5 branch scales on the first subset;
use only the other subset for a predeclared finite-sample rank threshold.

The provisional nominal normal-flagging level is 5%, not a user-certified
production requirement or a recall guarantee. With n=22 the selected rank is
22, giving threshold 0.22776579856872559 and strict greater-than comparison.
The rank interpretation assumes exchangeability and an independently fixed
score function; device/domain/part independence are unverified. Reject
unsupported finite thresholds instead of promising 1% from 22 examples.

Keep candidate flags separate from final quality grades, probabilities and
severity. Preserve all three known-normal flags from the map-fit subset and
the zero threshold-set flags by construction; neither is an independent error
estimate. No defect images were inspected or thresholds retuned to hide flags.

Bind policy and calibrated checkpoint fingerprints and verify an independent
reload, including equality at threshold. Raw normalization conformance error
is below 3e-8; reloaded anchor scores match exactly. All 86 tests pass. Actual
trained PC-only model timing is recorded with scope limits. Next evaluate the
frozen model/policy once on the reserved holdout, without tuning on its results.

## 021: Expose the frozen holdout once and retain every failure

Use the unchanged calibrated FP32 bundle, strict image threshold and original
source-grid masks for evaluation. Save all predictions/maps; guard repeated
inference with an exposure ledger. Recompute reports from caches. Preserve both
missed scratches and both false alarms. Pixel AUROC is ranking evidence only;
using the image threshold for response extent does not validate a damage mask.
Do not infer defect types from folder labels. Stage 4 is now authorized by the
user; no claim of production acceptance or Jetson proof follows these metrics.

## 022: Count explicit part events rather than deduplicating image hashes

Latch before the first inference attempt, including errors, and require a new
unused declared ID to re-arm. Byte-identical images can represent distinct
parts; image hashing provides provenance, not identity. Persist JSON records,
map references and explicit errors now; restart-safe database handling stays
in Stage 6. Region geometry uses the previously declared response threshold,
8-connectivity and no morphology. All 94 tests and the real replay pass.

## 023: Separate anomaly response, type uncertainty and prototype quality rules

Use a pure deterministic function with a pinned model/score policy, snapshot
and canonical policy hash on every decision. Add a numerical 10% normal-score
review band; do not describe it as a confidence probability. Unknown/unseen or
unreliable anomaly types always review before any size/location rejection rule.
Example area/box-diagonal and optional rectangular-zone limits demonstrate
configuration only; no approved drawing or physical tolerances exist. Default
zones remain empty. Pixel response extent is not a calibrated damage mask.

Integrate rules into actual inspect-one-image, including failures; preserve
historical outputs and prove new replay separately. REJECT fixtures explicitly
assert synthetic reliable-type evidence; the actual model does not supply it.
A cache-only post-evaluation diagnostic still passes two known scratches; record
that limitation and do not tune cutoffs to erase it. No new holdout inference,
training, dependencies, SQLite, UI or advisory wiring is added. Llama remains
mandatory via ExecuTorch on-device and cannot override decisions. Stop at Stage 5.

## 024: Use versioned SQLite transactions and immutable inspection originals

Use Python's existing sqlite3 module, with no new installation or database
server. Initialize only an empty database, set the ForgeSight application ID
and schema version 1 atomically, and refuse unrelated/newer files. Store full
metadata JSON plus query indexes, referencing image/map files outside SQLite.
Use short explicit transactions, bound parameters, foreign-key enforcement,
FULL synchronous commits and the default rollback journal for this local path.

Make inspection UUID and declared part ID unique within the database. Exact
canonical retries add zero rows; changed-content/same-part conflicts are errors.
Image hashes remain provenance, never deduplication keys. Check history before
re-arm/model loading, and retain the unique constraint for races. Original
records stay immutable; append unverified correction claims without rewriting
original grades/counts or promoting claims to confirmed training data.

Write JSON before database commit, propagate storage failure and permit explicit
JSON import retry without model execution. No cross-file atomicity, automatic
pending recovery, power-loss guarantee or camera identity proof is claimed.
All 112 tests and the actual 4-record separate-process proof pass. Stop after
Stage 6; Stage 7 API/UI is the next review gate.

## 025: Keep the operator app local, real and limited to exposed samples

Use FastAPI with buildless local HTML/CSS/JS. A new isolated app environment
adds 12 dependencies while preserving all 89 pinned vision versions. Keep the
four existing environments unchanged and reproduce the app lock in a fresh
temporary environment. Browser QA is isolated in a Windows-only environment.
No hosted deployment, frontend package service, CDN or cloud runtime is added.

Load the calibrated model once, use explicit CUDA/CPU choice, and permit one
inspection at a time with immediate busy refusal. Reuse the real part gate,
grading function and SQLite commit path. Config is read-only and the sample
catalog contains fit/development images only. Serve recorded source/overlay
references, verify source and new overlay hashes, and never expose arbitrary
paths. Render unknown type, pixel-only rules, the two missed scratches, replay
mode and pending Llama advice explicitly. Operator text is deterministic.

Actual browser clicks produced two real records. Keep those records after the
mobile overflow failure, fix the containing block and resume layout verification
without new inference. Refresh and a restarted server retain the results.
119 tests pass; no new holdout inference or tuning. Stop for review before the
physical Jetson/camera Stage 8. ExecuTorch on-device remains mandatory and open.

## 026: Keep original planning instructions local

At the user's request, stop tracking the three original planning/instruction
Markdown files and ignore their exact root paths. Preserve the local contents
and keep public status, engineering decisions and operator instructions tracked.
Update README navigation to public documents. This removes the planning files
from the current repository tree; existing commits remain unchanged and still
contain the earlier copies. No history rewrite or force push is performed.

## 027: Prove one constrained Llama note before background/UI integration

Stage 8 is blocked by reconfirmed Jetson unavailability. Follow the hardware
blocker rule for independent PC work. Use the existing 1B 8da4w/4-bit-embedding
ExecuTorch artifact and native runner in its unchanged CPU environment, with
82 prompt tokens, a 96-token output cap and unchanged export/context bounds.
Do not increase model/context memory or substitute another runtime.

Keep real-record/decision hashes and E1/G1 source mapping outside the tight
prompt. Exclude free-text IDs/notes, photos and numerical score/geometry from
model input. Guidance is a developer prototype checklist, not an approved SOP.
The host owns JSON, causal withholding and grade authority; Llama supplies a
short cited note subject to source-vocabulary checks and human review.

Retain all eight attempts: six formatting/content failures, a vocabulary guard
false positive, then identical source-faithful output accepted after correction.
Separate system instructions from user evidence; retain/test historical default
chat formatting. The final note repeats supplied facts/checks and is not broad
reasoning/usefulness evidence. All 128 tests pass. Source inspection/database,
model/policy and all five environments remain unchanged. No UI/queue/storage
migration yet; next evaluate scenario/refusal behavior before that integration.
Desktop worker guard remains: a verified native ARM adapter and target budget
measurement are required on Jetson. Stop after this PC substep for review.

## 028: Show operator essentials before technical detail

The user requested a simple display. Keep photo/highlight, result, type and one
short next action visible; preserve provisional PASS and unknown/not-assessed
states. Move area/timing/full action to Details, history to a closed disclosure,
and prototype limitations to About. Leave unavailable cause/advisory fields out
of the main view. Full records and evidence are preserved without backend or
grading changes. Future cause/precaution summaries require supported sources.

Validate the presentation on desktop/mobile using two previously verified
records copied to a separate test database. Leave the user's live default-port
app running; stop only the owned isolated preview. No new model inference or
project-stage advancement is claimed by this UI maintenance change.

## 029: Validate reviewed statement forms and refuse before model execution

The vocabulary-only guard accepted a handwritten causal claim assembled from
permitted words. Retain that baseline failure. For this narrow prototype, accept
two reviewed complete note forms with whitespace/case/citation variation only.
Do not add a second model, larger context or open paraphrase judge without quality
and target-resource evidence. Present contents are checklist repetition, not
demonstrated reasoning; deterministic templates could express the same facts.
Llama-family generation through ExecuTorch remains mandatory and actually runs.

Pin the canonical prototype guidance content, not merely its ID/approval field.
Save refusal reasons/snapshots for normal, failed or inconsistent inputs without
requesting a language-model process. Keep all original grades and records intact.
No factory procedure, defect subtype or cause is invented. Guidance expansion
requires new source/content-policy review and real operator evaluation.

Freeze input/output/runtime scenarios before execution. All 27 contract checks
match. Preserve the original 31/32 runtime-inclusive report: one-token generation
was safely rejected by the native token-statistics guard, not the expected text
parser. Declare a separate eight-token follow-up and observe parser refusal,
without rewriting the first report. Two normal-length native outputs share one
prompt, so do not claim independent scenario quality or prompt-injection resistance.
132 tests pass; no installations, new vision/holdout work or model/policy changes.
The standard-library contract/evaluation code is portable; the current native
worker still requires a separate verified ARM implementation and device budgets.
Stop after Substep 9.2; bounded background execution is the next PC task.

## 030: Resume device verification when hardware arrives

The user reported receiving an Orin Nano 8GB and Logitech USB webcam on
2026-11-11. Prioritize the read-only device/connection audit and the earliest
open Stage 2 native Llama/ExecuTorch gate, then Stage 8 camera/vision proof.
The PC background worker remains pending. Treat the reported hardware as a
lead for verification, not observed module/RAM/camera or deployment evidence.
Inspect boot state, software and existing work before installing or flashing.

## 031: Base Jetson deployment on observed userspace, not the PC lock

The first authenticated read-only audit identifies an Orin Nano 8GB in Super
configuration, Ubuntu 20.04.6/L4T 35.6.1, ARM64/glibc 2.31, Python 3.8.10 and
CUDA 11.4. The Logitech camera is a C270. Preserve this working installation
while choosing the next recipe; do not install, reflash, replace system Python
or change power mode during the audit. JetPack metapackage absence is not
absence of NVIDIA drivers. The 25W query also produced permission errors and
does not certify effective clocks, thermals or power limits.

Python 3.8 cannot host the pinned ExecuTorch Python distribution; the installed
CMake is also below the pinned native build floor. Investigate isolated modern
Python/ARM wheel dependencies and native C++ runner alternatives before choosing
a deployment path. Wheel architecture tags and available RAM are not proof of
complete runtime compatibility or model fit. Keep the Llama/ExecuTorch mandate;
no substitute runtime or larger model is selected. Current PC guards stay intact.

Record actual inventory as completed, while Llama generation, vision/camera and
offline device gates remain open. Preserve the terminal-transport failures;
use short encoded lines and a verified device-reported SHA-256 for the final
authoritative JSON. Test the standalone script on the real target; a full PC
test rerun is unnecessary for this inventory-only change. Stop for review here.

## 032: Try isolated matching ARM CPU wheels before changing the Jetson OS

Publisher-hashed CPU ExecuTorch 1.5.1 and Torch 2.14.0 ARM wheels contain the
native runner and required operator libraries. Their selected 33 ELF files
match exact definitions in ten target system libraries. Resolve/lock 49 packages
for Python 3.12.14/ARM and retain the existing small Llama artifact/context.
Choose an isolated CPU reference runtime as the first candidate; keep CUDA vision
dependencies separate. No target package or interpreter is installed here.

This minimizes initial adapter/build differences from the PC reference, but
brings Torch/export-related dependency overhead. Native C++ remains a fallback
if actual imports, memory or contention tests fail. Static matching does not
prove CPU instruction support, loader paths, operator registration or generation.
Do not promise latency/memory or change system Python, libc, CMake or JetPack.

Pin separate Python archive metadata and require a verified download before
setup. Official downloads failed; preserve that open prerequisite. The sole
source-only dependency built with pinned tools in a fresh PC environment; use
locked tools and controlled build isolation in the future target setup. Existing
environments/assets remain unchanged; 140 tests pass. Stop after the recipe and
static checks; interpreter acquisition and isolated setup are Substep 2.9.

## 033: Verify the acquired interpreter before isolated setup

After the user reported restored internet, retry the exact official standalone
archive and retain the prior failures. The acquired size and SHA-256 match the
pinned release metadata. Inspect paths/links and native files on the PC without
executing the ARM interpreter. Normalize the archive's internal $ORIGIN library
reference against the actual included member rather than treating it as a system
library. Ten interpreter ELF files plus 33 core files match exact definitions in
12 real target libraries, including freshly observed libcrypt and libutil.

Preserve system Python 3.8.10, package inventory, library hashes and the deleted
manual-audit folder. No target prefix, interpreter or package is installed in
this follow-up. Code/lock files are unchanged, so validate the actual archive and
library data without rerunning the broad 140-test PC suite. Update the active
recipe, leave earlier failure manifests as historical snapshots, and stop for
review before the isolated-environment setup in Substep 2.9.

## 034: Install and prove the isolated ARM SDK before transferring Llama

Use the publisher-hashed standalone interpreter and matching CPU wheels in one
fresh, marked user prefix. Refuse existing directories, wrong devices, root
execution and unsafe archive paths/links. Validate all archive members before
writing and recheck during extraction. Keep system Python 3.8.10, JetPack/CUDA
and the deleted audit folder untouched. Two preparation defects (drive-letter
link validation and regex replacement escaping) were fixed and tested before
target initialization; preserve those attempts.

Stage all 50 hashed package archives. Derive an offline lock by changing only
the two direct CPU wheel URL selectors to their exact +cpu versions, retaining
all hashes. Install pinned build tools first, then runtime dependencies without
unpinned build isolation or remote index lookup. This avoids target download
dependence during installation without claiming disconnected application proof.

The actual SDK passed dependency checks, CPU arithmetic, native bindings and
explicit registration of the required operators/XNNPACK. System before/after
snapshots match; five PC environments/eight protected files match the earlier
snapshot; 147 tests pass. Keep the working SDK at
`/home/nvidia/.local/share/forgesight/llama_cpu_v1`. Its 51 packages include the
49 runtime dependencies, wheel and bundled pip. Package overhead remains a
trade-off; native C++ stays a fallback if full-model resource tests fail.

Do not transfer weights or weaken the desktop x86 guard during setup. An import
probe is not generation. Substep 2.10 needs an explicit ARM adapter and real
bounded Llama output with hashes/timing/RAM/thread/network evidence. The setup
probe's 12.284 seconds/388,604 KiB describes imports/tensor work only. No model
latency, combined memory fit or offline compliance is established.

## 035: Use the verified 1B PTE in an explicit ARM smoke runner

Transfer only the existing publisher-linked Llama 3.2 1B tokenizer and checked
802 MB ExecuTorch PTE to the working private Jetson runtime. Match both hashes
after transfer. Keep the frozen export profile untouched so its hash remains
consistent with the model record; update the active Jetson status separately.

Use a dedicated ARM runner rather than disabling the desktop x86 guard. Require
the actual Orin Nano/isolated Python, exact package and asset versions, bounded
Llama chat tokens, a fresh output directory, two CPU threads and a child-process
timeout. This first run returned 40 tokens with native statistics and exit 0.
The model load/inference/whole-process times and sampled RSS are one observation
under connected Wi-Fi. Do not infer p50/p95 or combined camera throughput.

This closes the Stage 2 functional device-generation proof. Stage 8.1 next checks
Logitech capture formats and a compatible Jetson vision path before deploying
the frozen EfficientAD model. Async advice, physical samples and disconnected
full-system operation remain later gates.

## 036: Keep camera capture and vision model setup as distinct device checks

The Jetson's system Python 3.8/OpenCV 4.5.4 can read the C270 using V4L2 and
MJPEG at 640x480. The already installed Python 3.12 ExecuTorch environment has
CPU-only Torch and no OpenCV, Pillow, torchvision or Anomalib; it cannot load
the frozen EfficientAD checkpoint yet. Do not assume the system OpenCV extension
works in Python 3.12 or that this Torch build uses the Jetson GPU. Preserve the
working Llama environment. Next prepare a separate compatible vision runtime,
measure a single-image baseline on-device and only then optimize or connect
live camera records. MJPEG is the initial capture candidate because 1280x720
YUY2 advertises at most 10 fps, but actual capture quality, latency, model
accuracy and simultaneous resource use remain unmeasured.

## 037: Use a frozen TorchScript CPU baseline for first Jetson vision proof

Export only the calibrated EfficientAD-S forward pass from the pinned desktop
Anomalib environment. The separate Jetson vision venv installs CPU Torch and
NumPy from already staged ARM wheels; system Python's existing Pillow prepares
the input. This avoids changing the working Llama venv or system packages.
The frozen source and export agreed exactly on two fit images, and the Jetson
produced a matching tensor hash and score within 2.3e-7. CPU execution is a
functional baseline, not a speed choice: one first forward was 1.699 s. Torch
2.14 warns that TorchScript loading is deprecated, so any format migration
needs new output-parity proof. Live capture and concurrent runtime costs are
the next Stage 8 checks.

## 038: Reject black startup frames and keep no-part runs diagnostic

The first saved C270 frame was nearly black. A short device probe found 12
dark frames before exposure settled. For the current diagnostic, read at most
40 frames and accept the first with mean grayscale value at least 20; fail
closed if none qualifies. This is only a gross startup-darkness check, not a
validated lighting standard for metal nuts. Save the source, raw anomaly map
and overlay in private ignored artifacts. With no physical nut or declared
part ID, never create a ForgeGrade decision or SQLite inspection row.

## 039: Replay a development nut image before physical-part capture

With no physical nut available, use `test/bent/017.png`, which is in the
predeclared development-preview set and was already used in earlier PC replay.
The Jetson replay checks the committed dataset manifest, source hash, exact
model export and frozen ForgeGrade policy. It saves both native and 700x700
maps, an overlay, a structured record and one SQLite row. The dataset path is
source provenance, not a model-predicted subtype. Keep `unknown` type and
the policy's `REVIEW` decision. This proves edge replay/persistence without
reopening the 111-image final holdout or claiming live inspection.

## 040: Begin Stage 9 on the saved replay while physical capture remains open

The user confirmed no physical nut will be available and asked to proceed.
Use the already persisted Jetson replay as the Stage 9 input while retaining
Stage 8's physical-component proof as open. Keep Llama advice in a separate
payload; it cannot rewrite the ForgeGrade decision. Add optional system-role
chat formatting to the verified ARM runner because a user-role-only probe
omitted the required G1 citation and was withheld. The system-role run
generated the reviewed cited note. This is one synchronous advice proof;
the bounded queue, UI delivery and full offline gate remain separate.

## 041: Use a standard-library loopback page for the Jetson replay gate

The Jetson's system Python 3.8 has no FastAPI, while the working vision and
ExecuTorch environments are separate. A small `http.server` page reads the
saved SQLite inspection, starts one bounded advisory worker, and exposes
not-requested, pending, generated and unavailable states. It binds only to
Jetson loopback; an SSH tunnel supplied PC browser access for verification.
This avoids changing system Python or the Llama runtime. The page is a Stage 9
replay proof, not the final camera operator application. A bounded native
timeout was added to the existing controller so a real device failure can be
observed without changing the authoritative grade.

## 042: Use a detached offline pilot with independent Wi-Fi recovery

The Stage 10 first gate uses a fresh saved-image replay because no physical nut
is available. The Jetson is reached only through Wi-Fi, so an in-session
disconnect would end SSH before recovery. A detached transient system service
schedules a separate three-minute reconnect timer before dropping `wlan0`.
The service runs vision and advice as the unprivileged Jetson user, restores
Wi-Fi in `finally`, and cancels the timer after verifying a returned route.
The watchdog provides a recovery path if the pilot crashes. A fresh database
keeps earlier proof unchanged. The result is replay-only evidence and does not
close the physical camera or full Stage 10 gates.
