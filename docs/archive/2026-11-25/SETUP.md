> Historical snapshot archived on 25 November 2026. This page preserves development history; use the [current documentation](../../../README.md) for supported instructions.

# Setup and required assets

Reviewed against source `081ba00`, 24 November 2026. This collects existing
workflows; it is not a new clean-machine or Jetson deployment proof. WSL access
was denied during the initial audit; a later feature audit ran focused tests in
the existing isolated WSL environment. Do not install into Jetson system Python
or change global packages.

## Source and platform

```sh
git clone https://github.com/honey-cake7/ForgeSight.git
cd ForgeSight
```

After the documentation branch is published, select the reviewed revision
explicitly. A default clone may not include this branch. Online training and
downloads are separate from offline inference.

## Assets excluded from Git

| Asset | Destination / binding | Acquisition and checks |
| --- | --- | --- |
| Original MVTec AD metal_nut images | `data/mvtec_ad/metal_nut/` | Obtain through the official source linked in [DATASET](DATASET.md); follow its license and frozen split. `manifests/metal_nut_v1.json` binds image hashes. |
| Calibrated PC checkpoint and policy | `artifacts/vision/calibration_v1/efficientad_s_calibrated.pth` and sibling `score_policy.json` | Restore an authorized artifact copy, or follow [training](VISION_TRAINING.md) then [calibration](VISION_CALIBRATION.md). The app checks both against `manifests/efficientad_s_calibration_v1.json`. New training does not guarantee identical bytes. |
| Teacher and training inputs | Paths in `manifests/vision_training_assets.json` | Required for training/model reconstruction; sources and hashes in [VISION_TRAINING](VISION_TRAINING.md). The PC checkpoint loader uses recorded local teacher paths; those assets/path settings must also be staged. |
| Jetson TorchScript export | `$HOME/.local/share/forgesight/vision_cpu_v1/efficientad_s_calibrated.ts` | Transfer the trusted export or export the verified PC bundle below. SHA-256: `8766c41a51a5444f57f564134427bbd2d2d95dd6ec5ff08b24698f00f7f55dcd`. |
| Llama source, PTE and tokenizer | Publisher assets under ignored model paths; deployed bundle under `$HOME/.local/share/forgesight/llama_cpu_v1` | Obtain publisher access for the pinned model revision; follow [EDGE_RUNTIME](EDGE_RUNTIME.md) and [JETSON_RUNTIME](JETSON_RUNTIME.md). Transfer matching stage files/manifests and native libraries. Historical exports and later device records have different PTE hashes; do not mix bundles. |
| Optional plugin exports | `plugins/metal_nut/efficientad_s.ts`, `plugins/screw/efficientad_s.ts` | Follow [SCREW_PLUGIN](SCREW_PLUGIN.md); each recipe pins its model. Stage original screw images separately for replay. |

There is no public model release or self-contained reviewer asset package
linked here. An authorized local artifact handoff is currently needed for the
exact tested bytes. This remains a submission reproducibility gap.

## PC/WSL environment and replay

The desktop app lock targets **Linux x86_64, Python 3.12.14 and CUDA**, not native
Windows or Jetson. Use the prepared environment or create a new one with an
available compatible Python and uv. The following adapts the recorded workflow;
it was not reproduced on a new machine during this update:

```sh
# Ubuntu/WSL; compatible Python and uv must already be available.
export FS_APP_ENV="$HOME/.cache/forgesight/app-readme-v1/venv"
uv venv --python 3.12.14 "$FS_APP_ENV"
uv pip sync --python "$FS_APP_ENV/bin/python" requirements/pylock.app.toml
uv pip check --python "$FS_APP_ENV/bin/python"
export FS_APP_PY="$FS_APP_ENV/bin/python"
```

Choose a new directory, preserving existing environments. The configured lab
PC uses `/root/.cache/forgesight/app-v1/venv/bin/python`. See
[ENVIRONMENT](ENVIRONMENT.md) for the recorded stack/source-only dependencies.
Root `requirements.txt` is only the early Pillow environment. Vision, exporter,
UI QA and ARM Llama locks serve different environments.

After staging the required assets:

```sh
"$FS_APP_PY" -B -m forgesight.data_manifest --check
"$FS_APP_PY" -B -m forgesight.operator_app
```

Open `http://127.0.0.1:8765`, enter a new declared part ID and choose a replay
example. Default model device is CUDA; add `--device cpu` to select CPU.
Default history is `artifacts/runtime/inspections.sqlite3`.
[PC operator behavior/API](OPERATOR_APP.md).

## Jetson preparation and launch

Keep the established prefixes separate:

- `~/.local/share/forgesight/vision_cpu_v1/venv`: isolated CPU Torch vision.
- `~/.local/share/forgesight/llama_cpu_v1`: standalone Python, ExecuTorch
  environment, staged model/tokenizer and native libraries.
- `/usr/bin/python3`: existing controller/UI interpreter. Check its already
  audited OpenCV, Pillow and NumPy; do not install into it from this guide.

Follow [Jetson runtime](JETSON_RUNTIME.md) and [vision setup](JETSON_VISION.md)
for compatible ARM interpreter/wheelhouse preparation. Copy the matching
`forgesight/`, `configs/`, `guidance/` and required `manifests/` into the source
root; stage dataset images at their original paths. Optional plugin mode also
needs `plugins/` and both pinned exports.

On the prepared PC, export the verified calibrated bundle to a new output:

```sh
"$FS_APP_PY" -B -m forgesight.vision_edge_export \
  --output artifacts/vision/edge_export_v1/efficientad_s_calibrated.ts
```

Preserve existing artifacts. Compare the SHA-256 before transfer; a different
export needs its own parity/provenance check, not a modified frozen reference.

On the prepared Jetson, from the matching source root:

```sh
export FS_VISION="$HOME/.local/share/forgesight/vision_cpu_v1"
sha256sum "$FS_VISION/efficientad_s_calibrated.ts"
/usr/bin/python3 -B -m forgesight.jetson_live_ui \
  --vision-python "$FS_VISION/venv/bin/python" \
  --model "$FS_VISION/efficientad_s_calibrated.ts" --port 8770
```

Open `http://127.0.0.1:8770` on the Jetson. For connected PC viewing use the
[SSH tunnel instructions](JETSON_LIVE_UI.md); a tunnel is not disconnected
proof. Outputs default to `runs/live_inspections_v1/` relative to the source
root. Existing runtime bundle staging must be complete before requesting advice.

Optional modes have extra requirements:

- Four-type assessment: `--defect-type-profile configs/defect_type_mvtec_four_v1.json`.
  Read [advisory routing](FORGEASSIST.md) before using assessed records.
- Grounded automatic advice: `--grounded-auto-advice` and a dedicated
  `--llama-worker-socket`; see [GROUNDED_ADVICE](GROUNDED_ADVICE.md).
- Plugin demo: `--plugins-root plugins`, both `--vision-worker-socket` and
  `--llama-worker-socket`, plus the recipe assets; see [SCREW_PLUGIN](SCREW_PLUGIN.md).

These options exist in current source; no new combined-device acceptance run
was performed for this documentation change.

## Verification

In the prepared app environment, from the repository root:

```sh
"$FS_APP_PY" -B -m unittest discover -s tests -q
```

Some checks require compatible dependencies/assets. Device probes run on the
Jetson separately; record source hashes, runtime versions, inputs and timing
boundaries. Further instructions:

- [Cached holdout metric reproduction](VISION_EVALUATION.md): summarize saved
  predictions; do not repeat final-holdout inference or clear exposure records.
- [Training](VISION_TRAINING.md), [calibration](VISION_CALIBRATION.md),
  [export/parity](JETSON_VISION.md) and [offline gates](STAGE10_VERIFICATION.md).
- [Storage queries, restart and review](STORAGE.md).

Before submission, reproduce setup from a clean checkout with an authorized
asset package and verify the intended device build. Those checks remain open.
Stop the server with Ctrl+C when finished.
