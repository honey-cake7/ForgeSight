# Stage 8.2: one saved-image Jetson vision baseline

The frozen calibrated EfficientAD-S checkpoint is exported on the PC with
`forgesight.vision_edge_export`. The export contains the model's normalized
anomaly-map and image-score forward pass for a fixed batch of one RGB 256x256
FP32 tensor. Its 32,320,235 bytes and SHA-256 are recorded in
`manifests/jetson_efficientad_single_image_v1.json`. Keep the binary and MVTec
image outside Git. The score policy remains the existing frozen threshold,
0.22776579856872559. No weights were changed or recalibrated.

From the repository root in the pinned PC vision environment:

```sh
python -B -m forgesight.vision_edge_export --output artifacts/vision/edge_export_v1/efficientad_s_calibrated.ts
python -B -m forgesight.vision_edge_parity --model artifacts/vision/edge_export_v1/efficientad_s_calibrated.ts --image data/mvtec_ad/metal_nut/train/good/000.png --image data/mvtec_ad/metal_nut/train/good/001.png
```

The second command checks exported scores/maps against the original calibrated
Anomalib model on two fit images. Both maximum absolute differences were zero.
Check the export's SHA-256 before transfer.

On the Jetson, create a **new** private venv with the already installed
standalone interpreter and staged ARM wheelhouse. The Llama venv and system
Python are not installation targets:

```sh
mkdir -p "$HOME/.local/share/forgesight/vision_cpu_v1"
"$HOME/.local/share/forgesight/llama_cpu_v1/python/bin/python3.12" -m venv "$HOME/.local/share/forgesight/vision_cpu_v1/venv"
"$HOME/.local/share/forgesight/vision_cpu_v1/venv/bin/python" -m pip install --no-index --find-links "$HOME/.local/share/forgesight/llama_cpu_v1/stage/wheelhouse" 'torch==2.14.0+cpu' 'numpy==2.5.4'
"$HOME/.local/share/forgesight/vision_cpu_v1/venv/bin/python" -m pip check
```

Transfer the export, one already used `train/good/000.png` benchmark image,
`vision_edge_preprocess.py` and `vision_edge_once.py` into that private vision
directory using authenticated SCP. Verify the export and image SHA-256 against
the manifest. The target's existing system Python/Pillow/NumPy prepares the
tensor; its SHA-256 must match the PC-prepared tensor before inference:

```sh
cd "$HOME/.local/share/forgesight/vision_cpu_v1"
/usr/bin/python3 -B vision_edge_preprocess.py --image 000.png --tensor fit_000.npy
venv/bin/python -B vision_edge_once.py --model efficientad_s_calibrated.ts --model-sha256 8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd --tensor fit_000.npy --tensor-sha256 40f6e9ea512322e76a238204c7706f7eff91a7ca0f7edf5e1fab57818e290480 --threshold 0.22776579856872559 --output inference_01.json
```

The measured target result is in the manifest and the ignored local
`artifacts/vision/edge_export_v1/jetson_inference_01.json`. First CPU forward:
1,699 ms; model load: 173 ms; process peak RSS: 431,704 KiB. It produced
score 0.0284537226 and map shape `[1,1,256,256]`. These are one-run values
for a familiar training image. Image decode/preprocessing, process startup,
camera, grading, logging, UI and Llama are outside the timing; there is no
measured throughput or physical-part accuracy yet. The device was connected
to a network. Torch 2.14 emits a TorchScript deprecation warning; a new model
format would need a separate output-parity check.

## Live camera diagnostic while no nut is available

`vision_edge_camera_diagnostic.py` runs under the existing system Python 3.8
because that environment already has OpenCV and Pillow. It reads up to 40 C270
frames, discards grossly dark startup frames (mean grayscale below 20), saves
the selected PNG, invokes the separate vision venv for inference, and saves
`anomaly_map.npy` plus `response_overlay.png`. It writes `diagnostic.json` with
`is_inspection=false`, no part ID and no grade. Use a fresh output directory:

```sh
cd "$HOME/.local/share/forgesight/vision_cpu_v1"
/usr/bin/python3 -B vision_edge_camera_diagnostic.py --vision-python "$PWD/venv/bin/python" --model "$PWD/efficientad_s_calibrated.ts" --model-sha256 8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd --threshold 0.22776579856872559 --output-dir "$PWD/runs/camera_diagnostic_02"
```

The actual run selected frame 13 after rejecting 12 black frames. It saved a
640x480 room view, a 256x256 raw map, an overlay and JSON under that private
Jetson directory. An ignored PC copy is at
`artifacts/edge/camera_diagnostic_02/`. One model forward took 1,637 ms;
camera-to-report took 10,047 ms. The model score is out of domain and has no
defect meaning. No physical nut inspection, quality grade, SQLite record,
throughput or offline operation is proved. The frame may contain people or
room details; keep image artifacts private and outside Git.

## Saved metal-nut replay with ForgeGrade and SQLite

The replay uses the already exposed `test/bent/017.png` development-preview
image. Transfer it, the committed dataset manifest, frozen grade policy and
the small `forgesight` replay/storage/grade/regions modules to the private
vision directory. Verify their SHA-256 values before running. From that
directory, with a fresh replay ID and output path:

```sh
/usr/bin/python3 -B -m forgesight.vision_edge_replay --image data/mvtec_ad/metal_nut/test/bent/017.png --image-sha256 b49cc898143c46a8c3a8697411d04a078b36f60f4798ccc68aea33ea8fcac026 --dataset-manifest manifests/metal_nut_v1.json --part-id mvtec-dev-bent-017-jetson-replay-02 --vision-python "$PWD/venv/bin/python" --model "$PWD/efficientad_s_calibrated.ts" --model-sha256 8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd --grade-policy configs/forgegrade_v1.json --database "$PWD/inspections_replay_v2.sqlite3" --output-dir "$PWD/runs/dev_bent_017_replay_02"
```

The script checks the source's `development_preview` role and exact model
export hash before inference. It runs the image through system-Python
preprocessing and the separate Torch CPU venv, saves 256x256 and 700x700
maps plus an overlay, applies the existing ForgeGrade policy, then writes JSON
and SQLite. The actual result was score 0.682808876, `REVIEW` for unknown
defect type, 1,618 ms first forward and 7,443 ms replay-to-SQLite commit.
A fresh Jetson process read one identical record back. The raw files remain
private under `runs/dev_bent_017_replay_02/`, with an ignored PC copy at
`artifacts/edge/dev_bent_017_replay_02/`. See
`manifests/jetson_saved_nut_replay_v1.json` for hashes and limits. This is
neither a new accuracy test nor a physical-camera inspection.
