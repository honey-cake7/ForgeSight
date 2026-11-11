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
