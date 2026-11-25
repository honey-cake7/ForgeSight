# Setup

Run commands from a matching ForgeSight checkout. Models, datasets and runtime
databases are excluded from Git. The reviewer application package supplies the
exact trained models and offline ARM runtime inputs separately. Publication of
that package requires the owner to attach it to the submission or a release.

## Reviewer application: install once, then click ForgeSight

Use **`forgesight-reviewer-v1.tar`** and its **`.tar.sha256`** file from the
submission handoff. The package includes application source, both trained
TorchScript vision exports, the calibrated PC checkpoint/policy, matching Llama
PTE/tokenizer/manifests and model terms, pinned ARM Python and the offline
wheelhouse. It excludes datasets, credentials and previous inspection history.
It is a runnable application distribution, separate from the source-only clone.
No public download URL is recorded until the owner publishes the prepared files.
[Verified application evidence](../manifests/reviewer_application_v1.json): fresh
private installation, native Llama generation, visible desktop window and both
saved-image recipes passed on the recorded Jetson. This was not a wiped-OS test;
camera index 99 preserved the running station, and advice used checklist fallback.

The supported baseline is **Jetson Orin Nano 8GB, Ubuntu 20.04.6 / L4T 35.6.1,
ARM64, system Python 3.8.10**, with a graphical session and USB camera. Other
JetPack/Python baselines require separate validation. Reserve at least **8 GiB
free**. Setup preserves system Python, CUDA, power settings and existing stations.
The controller needs system OpenCV, NumPy and Pillow; the viewer needs Chromium
or GTK3/WebKit2 4.0/PyGObject. Preflight reports missing dependencies; the installer
does not use sudo or install system packages. On the recorded Ubuntu baseline,
an administrator can provision missing controller/viewer packages with
`sudo apt-get install python3-opencv python3-numpy python3-pil python3-gi gir1.2-gtk-3.0 gir1.2-webkit2-4.0`.

Obtain the original **metal_nut** and **screw** categories from
[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), accepting its
terms. Extract into a directory containing `metal_nut/train`, `metal_nut/test`,
`screw/train`, `screw/test` and their original `ground_truth` directories.
The installer verifies the recorded image/mask hashes without changing the data.
Do not substitute AD 2 or resized images. Review the bundled Llama license and
use policy before using its model. No cloud inference or publisher token is
needed to launch an already-authorized package.

From the folder containing the two submission files, in a fresh working directory:

```sh
sha256sum -c forgesight-reviewer-v1.tar.sha256
# Compare the checksum with the independently supplied submission checksum.
tar -xf forgesight-reviewer-v1.tar app
export FS_REVIEWER="$HOME/.local/share/forgesight/reviewer_v1"
export FS_DATA="$HOME/datasets/mvtec_ad"
export FS_SHA="$(cut -d ' ' -f 1 forgesight-reviewer-v1.tar.sha256)"
/usr/bin/python3 -B app/scripts/reviewer_bundle.py install \
  --bundle forgesight-reviewer-v1.tar --sha256 "$FS_SHA" \
  --prefix "$FS_REVIEWER" --dataset-root "$FS_DATA"
```

Use a fresh prefix; an existing prefix/shortcut or occupied port is refused.
Installation uses the bundled interpreter/wheels offline, creates separate vision
and Llama environments, verifies vision imports and a bounded two-token native Llama generation, and generates absolute paths in `station.json`. Dataset files remain
external through a link. The icon is the application SVG `forgesight/web/icon.svg`.
It adds **ForgeSight** to the application menu and desktop. Select **Allow
Launching** if the desktop requests trust, then double-click **ForgeSight**.
Cold startup may take up to six minutes; subsequent clicks reuse the station.
Closing the window leaves the backend available. Run from the installed user's
graphical session; no terminal commands are needed for subsequent launches.

The app opens the inspection screen with both part recipes, camera preview,
saved-image/upload inspection, local history/report tools and local advice.
Advice may correctly return unavailable/checklist fallback when validation
rejects generation; startup/model loading does not establish advice quality.
Logs: `~/.cache/forgesight-desktop/8770/station.log` and `browser.log`.
New inspection history: `$FS_REVIEWER/runs/`. Startup failures use a desktop
notification when available. Preserve these files for review.

```sh
# Recheck installed files and runtime/generation without starting a station:
/usr/bin/python3 -B "$FS_REVIEWER/app/scripts/reviewer_bundle.py" check --prefix "$FS_REVIEWER"
# Terminal equivalent of clicking the icon, useful for diagnostics:
/usr/bin/python3 -B "$FS_REVIEWER/app/scripts/jetson_desktop_launcher.py" \
  --station "$FS_REVIEWER/station.json"
```

For an additional isolated installation, use a fresh prefix and port, and
`--no-shortcuts` to preserve an existing icon. `--camera-index` selects another
camera; do not share one camera between running stations. Removing menu/desktop
shortcuts does not delete runtime files or inspection history. Installation
reports are saved as `reviewer_installation.json`; actual startup and reviewer
click acceptance are separate from the import checks.

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
A source-only clone still needs the authorized reviewer package above. Training
reproduction is separate from replaying the exact released inference artifacts.

### Maintainer: reproduce the application package

Use authorized staged copies of `stage/python.tar.gz`, `stage/inputs.tar` and
`model/` from the verified runtime; `pack` never copies a live venv or inspection
history. Run from the source checkout:

```sh
python3 -B scripts/reviewer_bundle.py pack \
  --runtime /path/to/authorized-runtime-inputs \
  --model-terms /path/to/llama3_2_1b \
  --checkpoint artifacts/vision/calibration_v1/efficientad_s_calibrated.pth \
  --output artifacts/submission/forgesight-reviewer-v1.tar
```

Keep the generated `.tar.sha256` and `.tar.manifest.json` with the archive.
The manifest identifies working-tree source bytes and individual asset hashes;
archive metadata/order is normalized. A new package requires a fresh output
filename. Attach all three files to the submission or approved release. Do not
commit model/runtime binaries into source Git. Re-training/exporting instead of
using these exact artifacts follows the technical report's archived methods and
produces a new experiment requiring new hash/accuracy/device validation.
