# Jetson live camera inspection prototype

The live page shows the C270 camera feed and captures one frame when the operator
clicks **Capture & analyze**. That frame is passed to the frozen EfficientAD-S
TorchScript model. The source image, two anomaly maps, overlay, ForgeGrade
decision and immutable SQLite record are saved locally. A new capture ID is
required for each capture; video frames do not create extra inspections.

The operator chooses **Dataset photo on a screen** or **Physical nut** before
capture. The former is saved as `LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC` with
`physical_part: false`. A photograph displayed on a monitor is a useful
camera-path diagnostic, but glare, screen pixels and camera perspective change
the image. It cannot establish physical-part accuracy or real-nut acceptance.

The MVTec score threshold has not been calibrated for the physical camera.
ForgeGrade therefore assigns `REVIEW` with reason `CAMERA_DOMAIN_UNVALIDATED`
for every analyzed camera frame, regardless of the raw model candidate. The
overlay marks model response, not confirmed damage. Defect type remains
`unknown`; physical quality accuracy, throughput and offline end-to-end
operation remain unverified.

## Upload, model response and manual review

Use **Upload your own test PNG** to choose a file on the computer running the
browser. Enter a fresh upload ID. For an original-size 700x700 MVTec metal_nut
PNG, check the MVTec declaration to show the provisional benchmark grade.
This declaration is not proof of file origin. Other PNGs remain diagnostic
REVIEW; camera results also remain REVIEW pending camera-domain validation.

The result shows the largest connected model response, its image-pixel area,
other response regions and an overlay. For 700x700 MVTec-layout results, an
image-centered radial rule may describe the central opening, inner rim,
annular face or outer perimeter. It may report two adjacent areas or leave the
area undetermined. Camera views receive no anatomical label. These approximate
locations and the response-pixel percentage do not measure physical damage
or establish a defect type; the pixel threshold has not been validated against
ground-truth masks. The predicted defect type remains `unknown`.

In **Manual review**, open the saved original image and compare it with the
overlay. Enter an operator ID, choose **No visible issue**, **Visible issue**,
or **Uncertain / capture again**, and describe what you observed. SQLite saves
an append-only, unverified claim linked to that inspection. The choice proposes
PASS, REJECT or REVIEW respectively; the original ForgeGrade grade never
changes. Reviewing the image helps identify bad captures and disagreements for
later analysis, but it is not certified ground truth or a trained defect-type
label. Llama advice remains a separate optional note for eligible saved-image
REVIEW events and does not change the grade.

## Launch after copying this source to the Jetson vision directory

For camera capture, keep the C270 attached and aim it at the selected source
under even lighting. For saved-image replay, the camera need not be ready. In
a Jetson terminal:

```sh
cd /home/nvidia/.local/share/forgesight/vision_cpu_v1
/usr/bin/python3 -B -m forgesight.jetson_live_ui --port 8770
```

The server binds to Jetson loopback only. With the current Jetson IP in place
of `<JETSON_IP>`, run this on the PC in a separate terminal:

```sh
ssh -N -L 8770:127.0.0.1:8770 nvidia@<JETSON_IP>
```

Open `http://127.0.0.1:8770` in the PC browser, or open the same address in
Firefox on the Jetson desktop. For camera use, choose the correct source mode,
enter a new capture ID, confirm it is visible, then click **Capture & analyze**. The camera
preview stays visible while the saved frame is analyzed. Stop the server with
Ctrl+C. Local output is under `runs/live_inspections_v1/` in the Jetson vision
directory; the SQLite file is `runs/live_inspections_v1/inspections.sqlite3`.

The source files needed in that directory are `forgesight/grade.py`,
`forgesight/vision_edge_live_once.py`, `forgesight/vision_edge_replay.py`,
`forgesight/jetson_live_ui.py`, and
`forgesight/web/jetson_live.{html,css,js}`. Existing `forgesight` modules,
`venv/bin/python`, `efficientad_s_calibrated.ts`, and
`configs/forgegrade_v1.json` are reused. The Llama runtime and system Python
packages are unchanged. The frozen `manifests/metal_nut_v1.json` and original
development-preview PNGs are required for replay. A physical nut inspection
remains unverified.

## Current proof

- On the PC/WSL, an earlier captured Jetson room frame (no nut) completed
  frozen EfficientAD inference, produced a 640×480 source map and overlay,
  and saved one matching SQLite record. The result was `REVIEW`; the one
  measured PC model forward was 1,103.361 ms. This is a plumbing check, not
  a physical nut inspection or accuracy result.
- Reusing the same part ID was rejected before inference. The database still
  contained exactly one row.
- The local loopback page, script and state endpoint returned HTTP 200. With
  no WSL camera attached, state correctly reported camera unavailable.
- The six new source files were copied to the actual Jetson with matching
  SHA-256 hashes. The UI started under system Python 3.8, its loopback page
  and frame returned HTTP 200, and a PC Edge browser displayed the live C270
  room view through a temporary SSH tunnel. The scene contained no nut, so
  physical capture, model inference and SQLite persistence remain pending.
  See `manifests/jetson_live_ui_preflight_v1.json`.
- The first operator click produced a correctly persisted `ERROR`/`REVIEW`
  record because the launcher followed the vision venv's Python symlink to
  the base interpreter, bypassing NumPy. The launcher now preserves the venv
  path. A fresh Jetson no-part diagnostic completed model inference, saved
  source map and overlay, and produced `REVIEW`; this still is not a nut
  inspection. See `manifests/jetson_live_ui_venv_fix_v1.json`.
- The user then photographed a displayed MVTec nut image on a phone with the
  C270. A screen-photo diagnostic was saved as `ANALYZED`/`REVIEW` with a
  234,723-pixel response (76.4% of the 640×480 frame). This is a broad model
  response, not confirmed damage or a physical nut inspection. The GUI now
  explains broad responses and keeps the phone-screen source label visible.
  See `manifests/jetson_phone_screen_diagnostic_v1.json`.
- A later displayed-photo capture was visibly sharper, but 76.3% of its frame
  still exceeded the benchmark pixel threshold. The operator had selected
  `LIVE_CAMERA` for that capture, despite no real nut being available; that
  immutable record is not physical-part evidence. The GUI now requires an
  explicit fresh confirmation when switching to physical-nut mode. See
  `manifests/jetson_screen_photo_quality_check_v1.json`.
