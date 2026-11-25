# Setup

## Reviewer application: install once, then click ForgeSight

Use the [README download and installation steps](../README.md#download-and-install-on-jetson)
for the published [v1.0.0 release](https://github.com/honey-cake7/ForgeSight/releases/tag/v1.0.0).
Download the application `.tar`, its `.tar.sha256` checksum and `.tar.manifest.json`.
GitHub's source-code ZIP/tar.gz files do not contain trained models or runtimes.
[Watch the two-minute demonstration](https://youtu.be/NZccgnyDJrY).

The application package contains source, both trained vision exports, the
calibrated PC checkpoint/policy, matching Llama PTE/tokenizer/manifests and terms,
and offline ARM Python/wheelhouse inputs. Datasets and prior inspection history
are excluded. The installer creates private vision/Llama environments and tests
bounded native generation. [Device verification](../manifests/reviewer_application_v1.json)
records the tested scope; it is not a wiped-OS or new camera-accuracy evaluation.

## Requirements

| Item | Supported baseline |
| --- | --- |
| Device | Jetson Orin Nano 8GB, ARM64 |
| System | Ubuntu 20.04.6 / L4T 35.6.1; system Python 3.8.10 |
| Desktop | Graphical session with Chromium or GTK3/WebKit2 4.0/PyGObject |
| Controller packages | System OpenCV, NumPy and Pillow |
| Storage | At least 8 GiB free before installation |
| Camera | USB camera; default index 0, C270 used in recorded checks |

Other JetPack/Python baselines need separate validation. The installer uses no
sudo and preserves system Python, CUDA and power settings. An administrator can
supply missing controller/viewer packages on the recorded Ubuntu baseline:

```sh
sudo apt-get install python3-opencv python3-numpy python3-pil python3-gi gir1.2-gtk-3.0 gir1.2-webkit2-4.0
```

Obtain the original **metal_nut** and **screw** categories from
[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), accept its
terms, and retain each category's `train`, `test` and `ground_truth` directories.
The README assumes `$HOME/datasets/mvtec_ad/` contains both category folders;
change `FS_DATA` if needed. The installer checks image/mask hashes. Resized images
and MVTec AD 2 are different inputs. Review the Llama license/use policy supplied
in the package. Installing from the verified wheelhouse and running inference
need no cloud service or model-access token.

## Launch, files and logs

After successful installation, use **ForgeSight** on the desktop/application
menu. Select **Allow Launching** if prompted. The shortcut uses the application
SVG logo. Installation takes several minutes; cold startup allows up to six
minutes. Run from the installed user's normal graphical session.

For the README's default installation:

```sh
export FS_REVIEWER="$HOME/.local/share/forgesight/reviewer_v1"
/usr/bin/python3 -B "$FS_REVIEWER/app/scripts/jetson_desktop_launcher.py" \
  --station "$FS_REVIEWER/station.json"
```

| Path | Contents |
| --- | --- |
| `$FS_REVIEWER/station.json` | Absolute startup command, port and application directory |
| `$FS_REVIEWER/app/` | Installed source, policies and vision models |
| `$FS_REVIEWER/model/` | Llama PTE, tokenizer, provenance and model terms |
| `$FS_REVIEWER/venv/` / `vision-venv/` | Separate Llama / vision environments |
| `$FS_REVIEWER/runs/` | New inspection artifacts, `inspections.sqlite3`, advice and operation logs |
| `$FS_REVIEWER/reviewer_installation.json` | Installation result and system-preservation check |
| `$FS_REVIEWER/runtime_probe.json` | Bounded native generation evidence |
| `~/.cache/forgesight-desktop/<port>/` | `station.log` and `browser.log`; default port 8770 |

The page is also available at `http://127.0.0.1:8770` for the default port.
Closing the window leaves the backend running. The release enables both recipes
and automatic grounded advice; see [USER_GUIDE](USER_GUIDE.md) for operator steps
and advice/fallback states.

Recheck installed files and native generation without starting a station:

```sh
/usr/bin/python3 -B "$FS_REVIEWER/app/scripts/reviewer_bundle.py" check --prefix "$FS_REVIEWER"
```

## Installation troubleshooting

| Symptom | Action |
| --- | --- |
| Checksum lists a filename ending in `\r` | An older copied sidecar has Windows line endings. Run `sed -i 's/\r$//' forgesight-reviewer-v1.tar.sha256`, then verify again. Current release checksums use LF. |
| Missing checksum/archive | Run from the download folder and confirm all three release files are present. Execute the commands directly, without wrapping them in `echo`. |
| Dataset missing or hash mismatch | Check `FS_DATA`, both category folders and original image/mask bytes. |
| Prefix already exists | Preserve the existing installation/history and choose a fresh prefix; failed partial installations are not overwritten. |
| Shortcut or port already exists | Stop the corresponding old station before sharing its camera. For another install, choose a fresh prefix/free port and add `--no-shortcuts`; this creates no new icon. |
| Window does not open | Check the station/browser logs and viewer prerequisites. Use the normal desktop session; isolated `dbus-run-session` diagnostics may emit GVFS/portal/keyring warnings. |
| Camera or advice unavailable | Use the [operator troubleshooting guide](USER_GUIDE.md#troubleshooting); preserve inspection evidence and logs. |

## Source and PC development

```sh
git clone https://github.com/honey-cake7/ForgeSight.git
cd ForgeSight
```

A clone excludes binary assets. After verifying/extracting the release, copy
`app/artifacts/vision/calibration_v1/` to this checkout's
`artifacts/vision/calibration_v1/` for PC inference. Keep `score_policy.json` with
the checkpoint. For source-based Jetson work, the matching exports are in the
release's `app/plugins/{metal_nut,screw}/`; Llama assets are under `model/`.
Original datasets belong at `data/mvtec_ad/{metal_nut,screw}/` in the checkout.
Model bindings are recorded in the [calibration manifest](../manifests/efficientad_s_calibration_v1.json)
and [application manifest](../manifests/reviewer_application_v1.json).

The desktop app lock targets Linux x86_64, Python 3.12.14 and the recorded CUDA
stack. Root `requirements.txt` provides only the early Pillow environment.
With compatible Python and uv available, create a fresh environment:

```sh
export FS_APP_ENV="$HOME/.cache/forgesight/app-dev-v1/venv"
uv venv --python 3.12.14 "$FS_APP_ENV"
uv pip sync --python "$FS_APP_ENV/bin/python" requirements/pylock.app.toml
uv pip check --python "$FS_APP_ENV/bin/python"
export FS_APP_PY="$FS_APP_ENV/bin/python"
"$FS_APP_PY" -B -m forgesight.data_manifest --check
"$FS_APP_PY" -B -m forgesight.operator_app
```

Open `http://127.0.0.1:8765`. The PC app defaults to CUDA; add `--device cpu`
for CPU inference. It uses the PC checkpoint, not the Jetson worker environments.
Tests require their locked dependencies and assets:

```sh
"$FS_APP_PY" -B -m unittest discover -s tests -q
```

Manual source-based Jetson commands are retained in the archived
[live-station guide](archive/2026-11-25/JETSON_LIVE_UI.md),
[runtime setup](archive/2026-11-25/JETSON_RUNTIME.md) and
[vision setup](archive/2026-11-25/JETSON_VISION.md). Those are lab snapshots;
use the release installer for reviewer deployment. Training/export methods and
holdout-exposure limits are indexed in [TECHNICAL_REPORT](TECHNICAL_REPORT.md#reproducibility).

## Rebuild the application package

Use coherent `stage/python.tar.gz`, `stage/inputs.tar`, `model/`, trained plugin
exports and the calibrated checkpoint from the verified release/runtime. `pack`
does not copy a live venv or inspection history. Run from the source checkout:

```sh
python3 -B scripts/reviewer_bundle.py pack \
  --runtime /path/to/extracted-runtime-inputs \
  --model-terms /path/to/llama-model-terms \
  --checkpoint artifacts/vision/calibration_v1/efficientad_s_calibrated.pth \
  --output artifacts/submission/forgesight-reviewer-new.tar
```

Use a fresh output filename and retain its generated checksum/asset manifest.
New source/assets require new device acceptance; the v1.0.0 evidence applies to
its pinned archive. Re-training produces a new experiment and need not reproduce
identical weights. Keep model binaries out of Git and preserve holdout records.
