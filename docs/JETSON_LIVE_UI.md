# Jetson live camera inspection prototype

The live page shows the C270 camera feed and captures one frame when the operator
clicks **Capture & inspect**. That frame is passed to the frozen EfficientAD-S
TorchScript model. The source image, two anomaly maps, overlay, ForgeGrade
decision and immutable SQLite record are saved locally. A new part ID is
required for each distinct part; video frames do not create extra inspections.

The MVTec score threshold has not been calibrated for the physical camera.
ForgeGrade therefore assigns `REVIEW` with reason `CAMERA_DOMAIN_UNVALIDATED`
for every analyzed camera frame, regardless of the raw model candidate. The
overlay marks model response, not confirmed damage. Defect type remains
`unknown`; physical quality accuracy, throughput and offline end-to-end
operation remain unverified.

## Launch after copying this source to the Jetson vision directory

Keep the C270 attached. Place one nut on a plain stable surface under even
lighting. Frame the entire nut without hands in view. In a Jetson terminal:

```sh
cd /home/nvidia/.local/share/forgesight/vision_cpu_v1
/usr/bin/python3 -B -m forgesight.jetson_live_ui --port 8770
```

The server binds to Jetson loopback only. With the current Jetson IP in place
of `<JETSON_IP>`, run this on the PC in a separate terminal:

```sh
ssh -N -L 8770:127.0.0.1:8770 nvidia@<JETSON_IP>
```

Open `http://127.0.0.1:8770` in the PC browser. Enter a distinct part ID,
check the operator declaration, then click **Capture & inspect**. The camera
preview stays visible while the saved frame is analyzed. Stop the server with
Ctrl+C. Local output is under `runs/live_inspections_v1/` in the Jetson vision
directory; the SQLite file is `runs/live_inspections_v1/inspections.sqlite3`.

The source files needed in that directory are `forgesight/grade.py`,
`forgesight/vision_edge_live_once.py`, `forgesight/jetson_live_ui.py`, and
`forgesight/web/jetson_live.{html,css,js}`. Existing `forgesight` modules,
`venv/bin/python`, `efficientad_s_calibrated.ts`, and
`configs/forgegrade_v1.json` are reused. The Llama runtime and system Python
packages are unchanged. An SSH transfer and physical Jetson browser test must
be completed before this launch can be called device verified.

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
- The real Jetson was unreachable at both previously known IP addresses when
  this feature was prepared. Device deployment, live physical capture and
  browser rendering are pending.
