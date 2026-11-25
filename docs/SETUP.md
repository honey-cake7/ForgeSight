# Setup

Run commands from a matching ForgeSight checkout. Models, datasets and runtime
databases are excluded from Git. The procedures below use the recorded isolated
environments; a clean-machine asset package has not yet been published.

## 1. Source and assets

```sh
git clone https://github.com/honey-cake7/ForgeSight.git
cd ForgeSight
```

| Required asset | Where it belongs / how it is checked |
| --- | --- |
| Original MVTec AD metal_nut images | `data/mvtec_ad/metal_nut/`; hashes/split in [metal_nut_v1.json](../manifests/metal_nut_v1.json). Obtain through [MVTec](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) under its terms. |
| PC calibrated model | `artifacts/vision/calibration_v1/efficientad_s_calibrated.pth` and sibling `score_policy.json`, bound by [calibration metadata](../manifests/efficientad_s_calibration_v1.json). |
| Jetson vision export | `~/.local/share/forgesight/vision_cpu_v1/efficientad_s_calibrated.ts`; SHA-256 `8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd`. |
| Llama runtime bundle | `~/.local/share/forgesight/llama_cpu_v1/`, including matching PTE, tokenizer, profile/export/asset manifests and native libraries. Obtain publisher model access; preserve the bundle's hash chain. |
| Optional screw recipe | Original screw data at `data/mvtec_ad/screw/`; stage both `plugins/metal_nut/efficientad_s.ts` and `plugins/screw/efficientad_s.ts` with hashes matching their recipes. |

Restore authorized project artifacts for the exact tested bytes. Re-training
creates a new experiment and need not reproduce identical model hashes.
Historical Llama exports differ: do not combine files from different bundles.

## 2. PC / Ubuntu WSL

The app lock targets Linux x86_64, Python 3.12.14 and a CUDA-capable desktop.
Root `requirements.txt` installs only the early Pillow environment. With
compatible Python and uv already available, choose a fresh isolated location:

```sh
export FS_APP_ENV="$HOME/.cache/forgesight/app-readme-v1/venv"
uv venv --python 3.12.14 "$FS_APP_ENV"
uv pip sync --python "$FS_APP_ENV/bin/python" requirements/pylock.app.toml
uv pip check --python "$FS_APP_ENV/bin/python"
export FS_APP_PY="$FS_APP_ENV/bin/python"
"$FS_APP_PY" -B -m forgesight.data_manifest --check
"$FS_APP_PY" -B -m forgesight.operator_app
```

Open `http://127.0.0.1:8765`. The desktop app defaults to CUDA; add
`--device cpu` to select CPU. Existing lab interpreter:
`/root/.cache/forgesight/app-v1/venv/bin/python`. Do not overwrite an existing
environment. This path uses the PC checkpoint; it does not attach Jetson workers.

## 3. Prepared Jetson

Keep the isolated vision and Llama prefixes separate. `/usr/bin/python3` runs
the controller with the device's existing OpenCV/Pillow/NumPy; do not install
into system Python. Initial ARM interpreter, wheelhouse and native-library
setup are detailed in the archived [Llama runtime](archive/2026-11-25/JETSON_RUNTIME.md)
and [vision setup](archive/2026-11-25/JETSON_VISION.md).

Stage the matching `forgesight/`, `configs/`, `guidance/`, `manifests/` and
required data in the source root. Then launch:

```sh
export FS_VISION="$HOME/.local/share/forgesight/vision_cpu_v1"
sha256sum "$FS_VISION/efficientad_s_calibrated.ts"
/usr/bin/python3 -B -m forgesight.jetson_live_ui \
  --vision-python "$FS_VISION/venv/bin/python" \
  --model "$FS_VISION/efficientad_s_calibrated.ts" --port 8770
```

Open `http://127.0.0.1:8770` on the Jetson. Ctrl+C stops the service. Outputs
default to `runs/live_inspections_v1/` relative to the launch directory. The
default advisory path is manual; operator steps are in [USER_GUIDE](USER_GUIDE.md).

## 4. Optional plugin / automatic-advice station

After staging both recipe bundles, use a dedicated socket directory:

```sh
export FS_SOCKET_DIR="$HOME/.cache/forgesight-station"
mkdir -p "$FS_SOCKET_DIR"
/usr/bin/python3 -B -m forgesight.jetson_live_ui \
  --vision-python "$FS_VISION/venv/bin/python" \
  --model "$FS_VISION/efficientad_s_calibrated.ts" \
  --vision-worker-socket "$FS_SOCKET_DIR/vision.sock" \
  --llama-worker-socket "$FS_SOCKET_DIR/llama.sock" \
  --plugins-root plugins --port 8770
```

Plugin mode enables automatic grounded advice and the Part Recipe selector.
Both vision models stay resident; the baseline model must match the metal-nut
recipe. Stop an existing station before starting another on the same port.
For single-part automatic advice, omit `--plugins-root` and add
`--grounded-auto-advice` with the dedicated Llama socket. Optional metal-nut
assessment uses `--defect-type-profile configs/defect_type_mvtec_four_v1.json`.
The page's manual advice button auto-routes supported assessed records to
`vision_choice`; standalone replay CLI callers choose `--advisory-mode vision_choice`.
Unsupported record/profile combinations are refused. See [advice modes](USER_GUIDE.md#advice-modes).

For connected PC viewing, run `ssh -N -L 8770:127.0.0.1:8770 nvidia@<JETSON_IP>`
on the PC. A tunnel is not disconnected proof. Desktop/menu shortcuts and
Chromium/GTK viewer setup are in the archived [launcher guide](archive/2026-11-25/JETSON_DESKTOP_LAUNCHER.md).

## 5. Verify and reproduce

```sh
"$FS_APP_PY" -B -m unittest discover -s tests -q
```

Tests need the appropriate locked dependencies/assets. PC checks are separate
from device proof. Training/export/cached evaluation commands are indexed in
[TECHNICAL_REPORT](TECHNICAL_REPORT.md#reproducibility). Preserve existing
artifacts and final-holdout exposure records; never clear them to tune a model.
Exact clean-checkout reproduction still needs an authorized asset handoff.
