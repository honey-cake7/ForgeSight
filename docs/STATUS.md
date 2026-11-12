# Project status

## Current stage

The consolidated [Stage 10 verification ledger](STAGE10_VERIFICATION.md)
lists measured Jetson results and every remaining acceptance gate. Stage 10
is **INCOMPLETE**. The user clarified that only an MVTec nut photo on a phone
screen is available, not a physical nut. The C270 photographed that screen
and the Jetson ran the model, ForgeGrade and SQLite. The broad response covered
76.4% of the frame and was held at REVIEW. This is a screen-photo diagnostic,
not physical-part defect evidence. See
`manifests/jetson_phone_screen_diagnostic_v1.json`. Physical-part and offline
acceptance remain unverified.

A later displayed-photo capture was sharper, but the broad model response
persisted at 76.3% of the frame. Its UI source was accidentally selected as
`LIVE_CAMERA`; the visual scene and user clarification show this is still no
physical-nut evidence. The GUI confirmation now spells out the source choice.
See `manifests/jetson_screen_photo_quality_check_v1.json`.

The [Jetson live camera UI prototype](JETSON_LIVE_UI.md) was deployed to the
physical Jetson. Its C270 preview and loopback page worked; a PC Edge browser
rendered the room view through SSH. That preflight scene contained no nut;
physical nut capture and its model decision remain **NOT VERIFIED**. PC/WSL
checks separately exercised a previously captured no-part
Jetson frame through those components. An operator click then persisted an
ERROR/REVIEW record when the launcher bypassed the vision venv's NumPy. The
path was fixed and a fresh Jetson no-part diagnostic ran EfficientAD and saved
a map and overlay. The GUI now separates screen-photo diagnostics from real
parts. See `manifests/jetson_live_ui_preflight_v1.json` and
`manifests/jetson_live_ui_venv_fix_v1.json`.

**Stage 10.7 disconnected ExecuTorch advisory latency series VERIFIED on the
physical Jetson; full Stage 10 acceptance PENDING.** Eight fresh-process
Llama advisories from the same saved REVIEW inspection all completed with
evidence-linked notes; the authoritative grade, record and SQLite remained
unchanged. Native process-wall p50/p95 were 20.690/20.942 s and sampled
native peak RSS p95 was 1,189,735 KiB. During the run, 169 roughly one-
second network samples showed no Wi-Fi/ethernet IPv4 or default route; Wi-Fi
recovered. This is repeated advice timing, not new image or operator
usefulness evidence. See `manifests/jetson_offline_advice_latency_v1.json`.

### Earlier checkpoint

**Stage 10.6 disconnected saved-replay latency series VERIFIED on the
physical Jetson; full Stage 10 acceptance PENDING.** Twenty cold-process
replays of two already known images all completed, persisted in fresh SQLite,
and read back exactly. Model-forward p50/p95 were 1,585/1,684 ms; replay-to-
SQLite p50/p95 were 7,499/7,677 ms. The process peak RSS p95 was 437,163 KiB.
During the run, 162 roughly one-second network samples showed no Wi-Fi/
ethernet IPv4 or default route; Wi-Fi recovered. The same two images were
repeated, so this is not distinct-parts throughput or new accuracy evidence.
See `manifests/jetson_offline_vision_latency_v1.json`.

### Earlier checkpoint

**Stage 10.5 disconnected camera failure/recovery diagnostic VERIFIED on the
physical Jetson; full Stage 10 acceptance PENDING.** An unavailable V4L2
camera index failed safely with no frame or grade. The still-connected C270
then captured a live 640x480 MJPG frame and ran the frozen EfficientAD model,
saving an anomaly map. The frame shows a room scene, so this is no physical
nut inspection and no SQLite inspection row was created. Eleven network
samples showed no Wi-Fi/ethernet IPv4 or default route; Wi-Fi recovered.
See `manifests/jetson_offline_camera_recovery_v1.json`.

### Earlier checkpoint

**Stage 10.4 disconnected saved-replay failure recovery VERIFIED on the
physical Jetson; full Stage 10 acceptance PENDING.** With Wi-Fi disconnected,
an intentionally missing model path produced a persisted ERROR inspection and
conservative REVIEW decision. A deliberately corrupt image copy failed the
frozen-image SHA-256 intake check before inference and created no inspection
row. A subsequent valid normal replay produced PASS; both accepted records
remained in fresh SQLite and matched their JSON files. The real frozen model
and image hashes were unchanged. Eleven roughly one-second network samples
showed no Wi-Fi/ethernet IPv4 or default route, and Wi-Fi recovered. See
`manifests/jetson_offline_failure_recovery_v1.json`.

### Earlier checkpoint

**Stage 10.3 disconnected loopback operator-page service VERIFIED on the
physical Jetson; full Stage 10 acceptance PENDING.** With Wi-Fi disconnected,
a local HTTP client fetched the page's HTML, JavaScript and CSS and exercised
the real queue/API against the saved REVIEW inspection. Advice progressed from
PENDING to a real ExecuTorch note; overlapping requests returned BUSY. A
one-second native deadline yielded ADVICE_UNAVAILABLE with no note. Restarting
the UI process preserved both completed states and the original grade, and
refused duplicate advice. Thirty-three roughly one-second network samples
showed no Wi-Fi/ethernet IPv4 or default route; Wi-Fi recovered afterward.
No browser was installed on the Jetson, so actual page rendering while
disconnected was not tested. See
`manifests/jetson_offline_loopback_ui_v1.json`.

### Earlier checkpoint

**Stage 10.2 disconnected saved-case series VERIFIED on the physical Jetson;
full Stage 10 acceptance PENDING.** The frozen EfficientAD model and
ForgeGrade processed a fitting normal reference and two previously exposed
development images with Wi-Fi disconnected. Their decisions were PASS, REVIEW,
and REVIEW; all three records survived in fresh SQLite. ExecuTorch Llama
generated a separate note for one REVIEW record without changing its grade.
Forty-six roughly one-second network samples showed disconnected Wi-Fi, no
Wi-Fi/ethernet IPv4 address and no default route. Wi-Fi recovered and the
independent watchdog was canceled. These repeated saved images add no accuracy
evidence or physical-part throughput. See
`manifests/jetson_offline_case_series_v1.json`.

### Earlier checkpoint

**Stage 10.1 Wi-Fi-disconnected saved-image replay pilot VERIFIED on the
physical Jetson; full Stage 10 acceptance PENDING.** A detached service
scheduled an independent three-minute reconnect watchdog, disconnected
`wlan0`, then ran the frozen EfficientAD replay, ForgeGrade, SQLite save, and
ExecuTorch Llama advisory in fresh processes. Network snapshots before and
after both inference steps showed Wi-Fi disconnected, no Wi-Fi/ethernet IPv4
address and no default route. The saved grade was `REVIEW`; a fresh SQLite read
matched the exact record. Wi-Fi reconnected and the timer was canceled. See
`manifests/jetson_offline_saved_replay_pilot_v1.json`. This is one previously
used development image, without physical camera capture, normal/uncertain
cases, throughput distributions or a full UI run. The inner runner's old
default offline field remains unverified; the separate network snapshots are
the evidence for this pilot.

### Earlier checkpoint

**Stage 9.4 saved-replay queue-to-UI delivery VERIFIED on the physical Jetson.** A loopback-only page on Jetson system Python
used the existing bounded queue and separate ExecuTorch controller. Chromium
through an SSH tunnel observed `PENDING` then a real cited note; a second
request was refused as `BUSY`, and a status read during generation returned
in 124 ms. A separate one-second native deadline recorded `TIMEOUT`; the page
showed `ADVICE_UNAVAILABLE` with no note in 2.07 s. Both runs kept the original
`REVIEW` grade and SQLite record unchanged. See
`manifests/jetson_advice_ui_replay_v1.json`. This is a saved MVTec image,
not camera capture, physical-nut inspection or offline-disconnected proof.
The physical-part gate stays open until a nut is available.

### Earlier checkpoint

**Stage 9.3 read-only saved-replay advisory display VERIFIED in a PC browser.** The operator API reads a separate local
advice report by inspection ID and checks its record/decision hashes before
display. Chromium showed the actual copied Jetson `[E1][G1]` note beside the
unchanged `REVIEW` decision and deterministic action, with no page errors or
external requests. The proof server had no vision model or accessible Jetson
image files, so this is display proof only. See
`manifests/operator_advice_display_v1.json`. Next: connect queue states to a
local Jetson UI and test failure behavior. Stage 8's physical-nut gate remains
open.

### Earlier checkpoint

**Stage 9.2 bounded saved-replay advice queue VERIFIED on the physical Jetson.** A single worker accepted a replay advisory in 1.214 ms,
refused an overlapping request in 0.0052 ms, and kept the original SQLite
inspection readable while real ExecuTorch generation ran. The generated note
remained separate from the unchanged `REVIEW` decision. The Jetson run exited
successfully; its controller took 20.761 s and sampled 1,156,568 KiB peak
child RSS. See `manifests/jetson_forgeassist_queue_v1.json`. Timeout/failure
recovery has a local unit test, but a real Jetson timeout and Jetson UI behavior are
still unverified. Stage 8's physical-nut gate remains open.

### Earlier checkpoint

**Stage 9.1 saved-replay advisory VERIFIED on the physical Jetson.** At the user's request, the already persisted
`test/bent/017.png` replay was used while no physical nut is available. The
reviewed `[E1]` inspection facts and `[G1]` prototype checklist went through
Llama 3.2 1B on-device via ExecuTorch/XNNPACK. The model generated a cited
note that passed the strict host output contract; the original SQLite record
and authoritative `REVIEW` grade stayed unchanged. One run used 82 prompt and
33 generated tokens, with 1.955 s native inference, 22.157 s native child wall
time and 1,156,388 KiB sampled peak child RSS. A missing-runtime probe saved
`ADVICE_UNAVAILABLE` with no note and preserved the original record. See
`manifests/jetson_forgeassist_replay_v1.json`. Stage 8's live physical-nut gate
also remains open.

### Earlier checkpoints

**Stage 8 Jetson saved-image replay and SQLite persistence VERIFIED; live
physical-nut inspection PENDING.** A previously exposed MVTec development
image (`test/bent/017.png`) passed dataset-role and SHA-256 checks on the
actual Jetson. The frozen model produced score 0.682808876 and the same
9,015-pixel response area as the earlier PC replay. ForgeGrade assigned
`REVIEW` because defect type remains unvalidated. JSON, raw maps, overlay and
one SQLite row were saved; a fresh process read back the exact record. One
CPU forward took 1,618 ms and replay-to-SQLite commit took 7,443 ms. See
`manifests/jetson_saved_nut_replay_v1.json`. This is a **saved-image replay**,
not a camera inspection or new accuracy result. No nut is available yet.
**Next: put a distinct physical nut in a fixed lit C270 view and prove one
conservatively reviewed, persisted camera inspection.**

### Earlier checkpoints

**Stage 8 live camera-to-model diagnostic VERIFIED; physical-part inspection
PENDING.** The C270 delivered a usable 640x480 frame after 12 black startup
frames were discarded. The Jetson ran the frozen EfficientAD export, saved a
raw anomaly map and response overlay, and recorded 1,637 ms model forward and
10,047 ms camera-to-report time for this one diagnostic. The camera viewed a
room, not a nut or inspection rig. The score has no defect meaning; no part ID,
grade, inspection row or Llama advice was produced. The user has no nut yet.
See `manifests/jetson_camera_model_diagnostic_v1.json`. **Next: position a
distinct nut in a fixed lit view and verify capture, inference, deterministic
review/alert, persistence and end-to-end timing.**

### Earlier checkpoints

**Stage 8.2 single-image EfficientAD inference VERIFIED on the physical
Jetson.** A separate Python 3.12 CPU vision venv loaded a hash-checked
TorchScript export of the frozen calibrated EfficientAD-S checkpoint. The
Jetson preprocessed a previously used MVTec training image and its tensor
hash matched the PC. One first forward took 1,699 ms; process peak RSS was
431,704 KiB. The score (0.0284537) matched the PC export within 2.3e-7 and
produced a 256x256 anomaly map. See
`manifests/jetson_efficientad_single_image_v1.json`. This is one CPU baseline
on a familiar benchmark image, not a camera inspection, accuracy/FPS estimate
or offline system proof. **Next: connect one physical nut capture to model
inference and logging, with end-to-end timing.**

### Earlier checkpoints

**Stage 8.1 camera and dependency preflight VERIFIED on the physical Jetson.**
The Logitech C270 advertised MJPEG 1280x720/30 and 640x480/30 modes; a direct
OpenCV V4L2 read negotiated MJPG and returned one 640x480 color frame. The
system OpenCV is 4.5.4 under Python 3.8.10. The separate Python 3.12 Llama
environment has CPU-only Torch 2.14.0 and lacks OpenCV, torchvision, Pillow
and Anomalib, so the frozen EfficientAD model cannot run there yet. The 0.339 s
first-read time and reported 30 fps are not measured inspection throughput.
No frame was saved, no nut was inspected and no vision inference occurred.
Evidence: `manifests/jetson_camera_vision_preflight_v1.json`. **Next: Stage 8.2**,
prepare a separate compatible Jetson vision path, transfer the frozen model with
hash checks and measure one image inference before using live physical parts.

### Earlier checkpoints

**Substep 2.10 VERIFIED for one connected-device Llama prompt.** The exact
802,247,040-byte PTE and 2,183,982-byte tokenizer matched their pinned hashes
after transfer to the actual Orin Nano. The explicit ARM runner used the isolated
ExecuTorch 1.5.1+cpu environment, two CPU threads and a 64-token cap. It produced
40 tokens of non-placeholder text, with matching native token statistics and
exit code zero. The native model load took 3.087 s, native inference 2.129 s,
whole child process 30.856 s, and sampled peak child RSS 1,189,184 KiB. These
are one-run measurements, not p50/p95 or combined camera/vision/UI performance.
The device was connected to Wi-Fi; disconnected operation remains unverified.

The text, logs, hashes and exact command are in
`manifests/jetson_llama_generation_v1.json`, with raw target output retained on
Jetson and an ignored local copy under `artifacts/edge/jetson_llama_v1/`. All
**151 software tests pass**, including four new ARM runner safeguards. The
desktop x86_64 worker guard and system Python 3.8.10 remain intact. The Stage 2
on-device Llama generation proof is met for this one prompt; next is **Stage 8.1**:
check the Logitech camera's actual formats and Jetson vision dependencies before
moving the frozen EfficientAD model. Physical nut inspection, generated advice
from real records, and a disconnected full-system run remain open.

### Earlier checkpoints

**Substep 2.9 VERIFIED: the isolated CPU runtime is installed on the actual
Jetson, and its native imports/registration probe passed.** System Python is
still 3.8.10. The separate interpreter/venv uses ARM64 Python 3.12.14, Torch
2.14.0+cpu and ExecuTorch 1.5.1+cpu at
`/home/nvidia/.local/share/forgesight/llama_cpu_v1`. Keep this installed SDK;
it is separate from the deleted manual-audit folder, which was not recreated.

All 50 staged package archives matched their pinned hashes. Local-only package
installation and `pip check` passed; 51 distributions are present (49 runtime
dependencies plus wheel and bundled pip). CPU arithmetic returned 10.0. The
native runner/tokenizer bindings loaded, XNNPACK is registered, and explicit
library loading registered all three required operators. This is native runtime
setup proof, **not Llama generation**. No model/tokenizer weights were transferred.

Before/after system-Python package inventories and five system/configuration
file hashes match exactly. The five existing PC environments and eight protected
files match the prior verified snapshot. All **147 software tests pass**, including
seven new installation safeguards. Actual logs/results are saved locally under
`artifacts/edge/jetson_runtime_setup_v1/`; the committed record is
`manifests/jetson_runtime_installation_v1.json`. Two preparation bugs were caught
and fixed before device changes; their evidence is retained. Next: **2.10**, an
explicit ARM adapter and a bounded real Llama generation test. Stage 2 remains
incomplete; camera/vision and disconnected operation remain unverified.

### Earlier checkpoints

Substep 2.8 completed the read-only core ARM compatibility investigation and
prepared a candidate isolated-runtime recipe. All 33 selected ARM ELF binaries
match the library/version definitions read from ten actual Jetson system
libraries. A 49-package ARM/Python 3.12.14 dependency lock resolved, and its sole
source-only dependency built as a pure-Python wheel in a fresh PC build env.
All 140 software tests pass. No target interpreter/package installation or model
execution occurred; system Python remains 3.8.10. The deleted manual audit folder
was not recreated. See `docs/JETSON_RUNTIME.md` and
`manifests/jetson_runtime_compatibility_v1.json`.

The user reported restored internet and authorized another download attempt.
The exact 29,199,399-byte standalone interpreter archive now matches its pinned
SHA-256. PC inspection checked 4,533 paths, 1,048 contained links and ten ARM64
ELF files. A further authenticated read-only target probe confirmed libcrypt and
libutil: all 43 interpreter/core native files match the checked definitions in
12 target libraries. The existing target Python-package inventory and previous
ten library hashes are unchanged. See `manifests/jetson_python_archive_v1.json`.

Static preparation is now VERIFIED; no isolated interpreter was installed or
executed, and package imports/kernel/model behavior remain unverified. The
current code and 140-test result are unchanged; this follow-up used archive/hash
and native-library checks rather than repeating the broad PC suite. The first
download failures remain retained. CPU wheels are the first candidate; native
C++ remains a fallback, with no OS change selected.

Stage 9.2 PC-only advisory/refusal checks are observed. All 27 declared contract
checks matched and all 132 software tests passed. Normal/error/inconsistent
inputs now save unavailable reports without requesting a Llama process. Guidance
content must match the reviewed prototype hash. A handwritten false-cause
fixture exposed the old vocabulary guard; two reviewed complete statement forms
now prevent that recombination. Notes still repeat supplied facts/checks, with
type/cause unknown and REVIEW unchanged; broad usefulness remains unverified.

Two actual ExecuTorch generations produced the same source-linked note, with
82 prompt and 33 generated tokens each, using the unchanged small Llama/context.
Metadata injection was excluded before generation. These share one prompt and
are not independent advice scenarios. Timeout, missing runtime and truncated
generation all kept advice unavailable. The original runtime-inclusive protocol
remains 31/32 matching expectations, passed=false: a one-token probe was rejected
at native statistics validation before the anticipated text parser. A separately
declared eight-token follow-up generated an incomplete note and proved parser
refusal. The initial mismatch is preserved, not retrospectively passed.

All five environments and 13 protected source/model/policy/inspection/database
hashes remain unchanged. No new vision/holdout inference or package installs.
Earlier Stage 9.1 attempts remain preserved. See `docs/FORGEASSIST.md`,
`manifests/forgeassist_pc_note_v1.json` and
`manifests/forgeassist_refusal_verification_v1.json`.

Stage 7 UI remains verified; About this prototype explains advice is not connected.
Per the user's request, the main view is simplified to photo/highlight, result,
type and one short next action. History, full record details and prototype
limitations are expandable. Browser checks passed on desktop/mobile using
stored-record replay only; no backend, grade or model changes occurred. See
`docs/OPERATOR_APP.md` and `manifests/operator_ui_simplification_v1.json`.
This substep remains a CLI component only: no background queue, advisory DB migration
or UI advice states were added. Stage 9 is NOT DONE. Guidance is explicitly
PROTOTYPE_NOT_FACTORY_APPROVED; possible causes are withheld. Human review and
broader domain/usefulness testing remain required for richer advice.

On 2026-11-11 the first authenticated read-only Jetson audit completed after
matching the user-provided console host fingerprint. The actual device reports
Orin Nano 8GB identifiers, ARM64, 6 CPUs, Ubuntu 20.04.6, L4T 35.6.1, Python
3.8.10, CUDA 11.4, 7.28 GiB reported RAM and about 92.70 GiB free SD storage.
Logitech C270 USB/video nodes are confirmed; no project frame was captured.
The system interpreter has no discoverable Torch/ExecuTorch, and `v4l2-ctl` is
missing. Power query reports 25W with EMC permission errors. Full provenance and
limitations are in `docs/HARDWARE.md` and `manifests/jetson_device_audit_v1.json`.
No packages, models, device files or configuration were installed/changed by the
audit; the SSH session closed normally. Device inventory is VERIFIED, while
actual Llama/ExecuTorch generation and Stage 8 camera/vision remain NOT VERIFIED.
The current
native wrapper is desktop x86_64 guarded and requires a verified ARM adapter
before Jetson deployment. Desktop CPU threads/timing/memory are not target claims.

PC advisory work stopped for review after Substep 9.2. Substep 2.9 has now
prepared and tested the isolated runtime. Resolve the earliest open Stage 2
Llama/ExecuTorch generation gate in 2.10 before claiming edge readiness.
PC Substep 9.3 (bounded background worker)
and separate stored/UI advice states remain pending.
Reliable defect names, approved industrial limits, the two missed scratches and
mandatory Jetson Llama/ExecuTorch/offline proof remain open.

Stage 2.7 desktop Llama generation through ExecuTorch is verified. Stage 1 and
the transition to Stage 3 were accepted by the user. Actual Stage 2 inference
on Jetson is NOT VERIFIED: actual model generation is still missing, although
native runtime setup and authenticated device inventory are now verified. Desktop text
generation is verified; model execution on Jetson, quality and offline proof are not.

The organizer reply shared by the user confirms Jetson Orin platform-family
eligibility for PS5, conditional on mandatory Llama, on-device ExecuTorch, and
local defect/quality-control operation without cloud dependency. See
`docs/ORGANIZER_CLARIFICATION.md`. Actual hardware and execution gates remain
open; desktop vision work proceeds under the roadmap's hardware-blocker rule.

## Evidence already observed

- Substep 0.1 audited all four original workspace files without changing them.
- No local `.git` directory existed at audit time.
- GitHub's repository metadata reported size zero, and the branch-list endpoint
  returned `[]`; this was checked again before repository initialization.
- Windows Python 3.10.11 runs. Ubuntu WSL has an existing Python 3.12.14
  environment with PyTorch 2.6.0+cu124; PyTorch reports CUDA available and the
  desktop RTX 2000 Ada GPU name.
- At the original audit the user reported that Jetson was unavailable; on
  2026-11-11 the user reported receiving the Orin Nano 8GB and Logitech USB webcam.
- The original architecture PDF has been read; its proposed measurements and
  performance are not accepted as verified results.

## Completion evidence for Substep 0.2

Local commit `fd58c96` initialized `main` and recorded 12 intended files. The
remote `origin` is `https://github.com/honey-cake7/Forge_Sight.git`. No push was
performed. Staged whitespace and inventory checks passed. All four original
workspace files retained their audited SHA-256 hashes; the three original
Markdown files were also byte-identical in the staged snapshot. Seven
representative local/generated paths were checked against `.gitignore`.
Git reported a clean `main` branch after the commit.

## Completion evidence for Substep 0.3

- Created the project `.venv` using WSL CPython 3.12.14 with system packages
  excluded. Only pinned Pillow 13.0.0 and bundled pip 25.0.1 were installed.
- Installation used the published wheel hash, binary-only artifacts, and the
  official PyPI index. The exact recipe is in `docs/ENVIRONMENT.md`.
- `scripts/check_environment.py` passed isolation and lossless PNG pixel checks.
- `pip check` reported `No broken requirements found.`
- A second fresh temporary environment reproduced installation and both checks.
- No model or dataset was downloaded, and no inference was performed.

## Completion evidence for Substep 1.1

- Reviewed the official MVTec AD page and its CC BY-NC-SA 4.0 license, including
  the linked Creative Commons summary and legal code. No download form submitted.
- Checked published metal-nut counts in the original authors' paper: 220 normal
  training images, 22 normal and 93 anomalous test images. These are not yet
  locally verified counts.
- Inspected the publisher-linked 11,056-byte evaluation archive in memory;
  its code confirms image/mask pairing and absence of mask files for `good`.
  No code from the archive was executed or installed.
- Recorded a deterministic split policy before sample viewing: expected
  176 fit, 44 normal validation, 4 development previews, 111 final holdout.
  This is a project protocol, not the full official test protocol.
- See `docs/DATASET.md` for sources, limitations, and the exact proposed rule.
  No dataset images downloaded, no split manifest generated, and no loader
  implemented during this substep.

## Completion evidence for Substep 1.2

- The user supplied a Downloads path and explicitly confirmed official MVTec
  acquisition. The original `metal_nut.zip` was copied without changing it.
  Exact archive URL was unavailable; this source limitation is recorded.
- Archive: 165,672,999 bytes, SHA-256
  `5468b22edb8e3ebcb2e415bcc6adbed571e1b8a4dec03d1747e52d696384d838`.
- Actual counts: 220 normal training images; 22 normal and 93 anomalous test
  images; 93 matching masks. Defect counts: bent 25, color 22, flip 23, scratch 23.
- Every PNG fully decoded: 700x700 RGB images, grayscale `L` masks, values 0/255.
  Every anomalous image has a nonempty paired mask; no orphan masks.
- Safely extracted to ignored `data/mvtec_ad/metal_nut/`; all 430 regular files
  matched audited hashes, including the two retained attribution/license notices.
- Fourteen synthetic ZIP/tar validation/extraction tests passed. This is separate
  from the real data audit; neither is model-performance evidence.
- Source, archive, extracted data, and generated audit reports stay outside Git.
  No split manifest generated, no sample previews displayed, and no model run.

## Completion evidence for Substeps 1.3-1.5

Substep 1.3 verified all current files against acquisition fingerprints and
found zero byte/pixel duplicate image groups. The frozen manifest assigns
176 fit, 44 normal validation, 4 development preview, and 111 final-holdout
samples. A fresh rebuild reproduces its bytes; checksum:
`14b11ebd03d5107c7ef3478e1c4d20887237e2fbef58601ac043014b8f0e340f`.
All 22 current tests passed. No previews or inference were performed.

The user authorized the next three Stage 1 substeps and `git push origin main`
after each verified commit. Actual push results are reported at each checkpoint.

Substep 1.3 was committed and pushed as `7ff8846`. Substep 1.4 was committed and
pushed as `0eada8b`; it verified loading
all 176 fit, 44 normal-validation, and four development samples with byte/pixel
hashes, modes, sizes, and mask semantics checked. Default final-holdout access
was refused. All 31 current tests passed. No visual previews or predictions yet.

Substep 1.5 generated and visually verified the contact sheet and provenance
JSON in `artifacts/data/development_preview_v1/`. Only `train/good/000.png` and
four frozen development examples were displayed. Annotation areas are recorded
in `docs/DATASET.md`; no predictions, severity, or physical dimensions are
claimed. Five new preview tests passed, bringing the total to 36 tests.
The commit containing this evidence records the third authorized substep;
its actual commit hash and push result are supplied in the final step report.

## Open gates

Substep 3.3 fitted q90/q99.5 scales on 22 normals, then used the 22nd order
statistic of 22 separate normal scores. The threshold group has 0 flags by
construction; the map-fit group has 3 known-normal flags, all preserved in the
record. Neither is an independent false-alarm estimate. Rank assumptions and
production performance are unverified. No fit/development/holdout pixels were
read and no optimizer updates occurred. An independent CUDA reload reproduced
two scores exactly, including equality at threshold. All 86 tests pass.
Actual calibrated FP32 model-only desktop p50/p95 were 14.54/14.80 ms, not camera
or end-to-end Jetson results. See `docs/VISION_CALIBRATION.md` and
`manifests/efficientad_s_calibration_v1.json` for the frozen bundle/provenance.

Substep 3.2 uses 176 verified normal fitting images and 9,469 generic Imagenette
train images for the unchanged upstream regularization loss. No normal-validation,
development or final-holdout pixels enter the optimizer. Both archive hashes
match the pinned source; strict teacher loading and an actual forward passed.
Training completed in the existing vision environment without installing packages:
1,158.35 seconds, recent mean total loss 16.3052 near step 100 to 2.2713 at the
end. The final 75,199,220-byte checkpoint SHA-256 is
`85106f4a9ef6ec3e9c8acb4056dc2db2124e54284209e10d686dd36dea66ca0d`.
Independent reload matched all network digests and the saved fit-anchor output
exactly. All 78 tests passed; all four environment inventories are unchanged.
This is a trained first baseline with no calibration, threshold or accuracy
claim. See `docs/VISION_TRAINING.md` and the committed training manifest.

Substep 3.1 prepared a separate GPU environment with Anomalib 2.6.2 and
PyTorch/torchvision 2.6.0/0.21.0+cu124. Both model modules match the pinned
Anomalib source hashes. Seventeen untrained synthetic case measurements across
two runs (including a repeated EfficientAD-S AMP case)
covered EfficientAD-S/M, 256/384 inputs, PatchCore ResNet-18/Wide-ResNet-50,
bank-size changes, CPU/GPU, and AMP compatibility. Zero dataset pixels or
holdout images were read and zero optimizer steps were taken. FP32 compact
PatchCore was faster on this PC; the primary choice favors fixed-cost,
bank-free EfficientAD-S with measured memory and optional AMP headroom.
Accuracy, trained FP16 behavior, TensorRT, and Jetson latency remain unverified.
The 89-dependency lock reproduced in a fresh temporary environment; all 70
project tests pass. See `docs/VISION_BASELINE.md`, the candidate config, and
the architecture-probe manifest. Teacher/regularization assets remain pending.

Substep 2.7 uses the installed native TextLLMRunner via its Python bindings,
with no new packages or compilation. The first attempt failed on unregistered
4-bit embedding/cache/attention kernels. Explicitly loading the wheel's two
quantized/custom-op libraries resolved registration; failed logs remain saved.
The final actual run exited 0, with 33 prompt tokens and 30 generated tokens
reported by native Stats (31 callback pieces including the end marker).
The generated sentence explains consistent lighting without claiming inspection.
The generate call took 4.08 seconds including lazy model load; native inference
timing was 1.154 seconds. These are single-run desktop observations, not Jetson
benchmarks. All 67 tests pass. Both original environments remain unchanged.
Transcript, binaries' fingerprints and measurements are committed in
`manifests/llama3_2_1b_pc_smoke.json`; raw logs stay ignored.

Substep 2.6 exported the actual pinned checkpoint using 8da4w and 4-bit
embeddings. Conversion exited 0 after 248.46 seconds on desktop WSL; sampled
export-process peak RSS was 8,237,948 KiB, not device inference memory. The
802,247,040-byte PTE has SHA-256
`b82d0de5e0785269112d818be5a6c675225aab02c4e2a3b6c98d9e599da86739`.
ExecuTorch's InternalConsistency program load succeeded; metadata contains ten
methods and an XnnpackBackend delegate for forward. No method was executed.
The output-name suffix initially bypassed output_dir; the original is preserved
and a hash-identical copy is under ignored artifacts. The corrected recipe
was checked against upstream's real filename resolver. All 63 tests pass;
both original environments remain unchanged. See the committed export manifest.

Substep 2.5 installed a separate WSL/Linux-filesystem environment with
ExecuTorch 1.5.1+cpu, PyTorch 2.14.0+cpu, and TorchAO 0.18.0. The source checkout
matches the pinned commit; source and installed config schemas match the
committed SHA-256. CPU tensor calculation, real exporter CLI startup, structured
LlmConfig parsing, and dependency compatibility passed. Optional TorchAO GPU
library warnings are retained. Full conversion compatibility is still untested.
The hashed package lock was reproduced in a fresh temporary environment.
Both previous environments' full package inventories remained unchanged.
All 59 project tests passed. See `docs/EDGE_RUNTIME.md` for commands and limits.

Substep 2.4: the user obtained access and downloaded the five original model
files. All five matched pinned Hugging Face metadata (LFS SHA-256 for checkpoint
and tokenizer; Git blob IDs for small files). The checkpoint is 2,471,677,246
bytes; tokenizer has 128,000 unique base tokens with contiguous ranks. Parameters
match the source 1B configuration. Source assets are verified, not exported or
executed. The updated recipe has no missing-model-file/access blockers; source
checkout, export environment, conversion, target build, and device remain unverified.
Six new integrity tests passed; all 58 tests passed together. See
`manifests/llama3_2_1b_assets.json` and `docs/EDGE_RUNTIME.md`.

Substep 2.1 was committed and pushed as `51cdb0f`. Substep 2.2 was committed and
pushed as `c06e15e`; it rendered the
ignored recipe/config and validated fields parsed from the pinned upstream
schema; six new tests passed. The actual recipe reports missing model files,
source checkout, and export environment, with PREPARATION_ONLY_NOT_RUN. No model
download, package installation, export, or native compilation was performed.

Substep 2.3 added a native-ARM64/device guard and bounded process/evidence
capture. Six new tests and the full 52-test suite passed. The actual PC attempt
saved BLOCKED_NOT_VERIFIED with `process=null` and no generated text; no runner
was started. Runtime and offline gates remain unverified. The commit containing
this evidence is the third authorized step; the final report supplies its
actual hash and push result.

Stage 1 preview commit `323f499` was pushed successfully. The user's own
`dcba7fb` master-prompt commit is preserved. The new request authorizes the next
three steps and a push after each. Stage 2.1 pinned a model/runtime candidate
and tested preflight parsing/guards (four passing tests). The actual WSL run
recorded x86_64, no ExecuTorch in the image environment, and NOT_RUN inference;
device-only operation correctly exited 2. See `docs/EDGE_RUNTIME.md`.

| Gate | Status | Missing evidence |
| --- | --- | --- |
| Repeatable minimal PC image environment | DONE on tested host | Pinned install and smoke check reproduced in two isolated environments |
| Exact Jetson hardware and camera inventory | VERIFIED by authenticated audit | Orin Nano 8GB identifiers, ARM64, OS/RAM/storage and Logitech C270 USB/video nodes recorded; controlled capture remains separate |
| Llama through ExecuTorch on Jetson | VERIFIED for one connected-device smoke | 40 real tokens; model/tokenizer/runner hashes, native statistics, timing and memory in jetson_llama_generation_v1.json; offline/advice integration separate |
| Meta Llama access and local source assets | VERIFIED | User-confirmed access; five files match pinned publisher metadata |
| Desktop Llama export and artifact metadata | VERIFIED, not inference | Recorded PTE hash and program metadata; generation has separate evidence and quality remains open |
| Desktop Llama generation via ExecuTorch | VERIFIED, PC-only smoke | Real native runner output and provenance; no device or advice-quality claim |
| Vision benchmark anomaly detection | VERIFIED on PC only | Frozen 111-image evaluation: 87/89 defects flagged, 2/22 false alarms; physical camera/edge proof open |
| Defect-type classification | NOT VERIFIED | Current predicted types remain unknown; need valid same-domain classification evidence |
| Pixel-only decision rules | PC PROTOTYPE VERIFIED | Deterministic policy tests/replay; physical tolerances, severity and industrial certification open |
| Local history and duplicates after restart | PC SQLITE REPLAY VERIFIED | Four original records survive writer exit, exact retries add zero rows; no physical identity or power-loss claim |
| Offline end-to-end inspection | PARTIAL: saved-case replay, loopback UI service, failure paths, camera diagnostic recovery and separate vision/advice latency under disconnection | Physical camera part, rendered offline browser, decoder failure and physical throughput open |

Stage 1 input preparation is verified and its review gate was accepted. The
Stage 2 device-generation gate is verified for one prompt. Desktop data work
and GPU availability still do not establish a Jetson camera-inspection system.

## Next exact task

Continue Stage 10 with decoder-level corrupt-input handling and any available
offline browser proof. Physical-nut camera acceptance and distinct-parts
throughput remain open until a nut is available.
