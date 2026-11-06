# Decisions and unresolved choices

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
