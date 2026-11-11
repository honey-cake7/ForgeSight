# Stage 2: Llama through ExecuTorch on the actual Jetson

Initial review: 2026-11-05 (Asia/Calcutta). Stage 2 device proof remains
BLOCKED - NOT VERIFIED. Substep 2.7 completed real desktop generation after
actual export; no Jetson generation, device benchmark, or disconnected run has
been performed. Earlier substep sections retain their historical snapshots.

The first authenticated Jetson inventory completed on 2026-11-11. The actual
Orin Nano 8GB target uses ARM64, Ubuntu 20.04.6/L4T 35.6.1, glibc 2.31 and
Python 3.8.10. This closes identification, not model execution. No target model,
Torch/ExecuTorch installation, export or generation occurred. See
`docs/HARDWARE.md` and `manifests/jetson_device_audit_v1.json`.

**Substep 2.8:** read-only checks matched all 33 selected ARM core ELF files
against ten actual target system libraries, and resolved 49 packages for isolated
CPython 3.12.14. CPU wheels include the native runner/custom/quantized libraries.
The existing system Python/CUDA are preserved; no target installation occurred.
See [the complete recipe and trade-offs](JETSON_RUNTIME.md) and
`manifests/jetson_runtime_compatibility_v1.json`.

The separate Python interpreter archive initially failed to download. A later
authorized retry acquired the exact pinned bytes and passed SHA-256 verification.
Its ten ARM64 ELF files and the original 33 core binaries match the checked
definitions from 12 actual target libraries. See
`manifests/jetson_python_archive_v1.json`. This clears static preparation, not
execution: no new interpreter was run or installed on Jetson. Next Substep 2.9
creates the isolated runtime and tests its interpreter/dependencies/imports.
The [pinned package's Python floor](https://pypi.org/pypi/executorch/1.5.1/json)
still excludes system Python 3.8. A native C++ build remains a fallback; its
[CMake minimum](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/CMakeLists.txt)
is 3.24, but the selected prebuilt candidate does not require rebuilding on-device.
Do not disable the desktop-only guard and call that ARM support. Actual native
generation, memory/latency under load and disconnected operation remain open.

The [organizer reply shared by the user](ORGANIZER_CLARIFICATION.md) accepts
Jetson Orin platform-family eligibility for PS5. Mandatory on-device Llama via
ExecuTorch and local defect/quality-control operation remain unchanged.

## 2.1: Candidate and prerequisites

`configs/edge_llama.json` pins a preparation candidate, not a working deployment:

| Item | Selected reference |
| --- | --- |
| Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Model revision | `9213176726f574b556790deb65791e0c5aa438b6` |
| Model format for native exporter | Original checkpoint, params, and tokenizer |
| ExecuTorch | Release `v1.5.1`, commit `3b60683923245cf472b7323426920e15623ba361` |
| Backend | XNNPACK on ARM64 CPU |
| Candidate quantization | Dynamic INT8 activations / INT4 weights, group 128; embedding 4-bit/group 32 |
| Initial lengths | Export sequence 128; context 512, subject to actual validation |

Primary source review:

- [Model API](https://huggingface.co/api/models/meta-llama/Llama-3.2-1B-Instruct)
  supplied the revision, original file list, and `manual` access gate. The user
  initially reported having no Meta Llama access. In Substep 2.4 the user obtained
  access and downloaded the pinned source assets. No credentials were requested
  or inspected by the verification tools.
- [Model license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/9213176726f574b556790deb65791e0c5aa438b6/LICENSE.txt)
  is the Llama 3.2 Community License with its Acceptable Use Policy, rather than
  the dataset's Creative Commons license. Review attribution/notice, license-copy,
  display, and redistribution conditions before shipping model artifacts.
  Access/license acceptance must occur through the user's publisher account.
- [ExecuTorch release](https://github.com/pytorch/executorch/releases/tag/v1.5.1)
  and its [tag reference](https://api.github.com/repos/pytorch/executorch/git/ref/tags/v1.5.1)
  supplied the pinned version and commit. PyPI's top-level API reported 1.6.0
  while GitHub's latest-release API reported 1.5.1. Record this source difference;
  use the pinned 1.5.1 source rather than combine latest instructions arbitrarily.
- [Pinned package metadata](https://pypi.org/pypi/executorch/1.5.1/json) lists
  CPython 3.12 Linux x86_64 and AArch64 wheels, Python >=3.10/<3.15, and newer
  numerical/export dependencies than the existing shared PyTorch environment.
  Package availability alone does not establish Jetson ABI or runtime support.
- [Pinned XNNPACK example](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/examples/models/llama/config/llama_xnnpack.yaml)
  provides the quantization/backend basis; the candidate retains those settings.
  [Pinned export schema](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/extension/llm/export/config/llm_config.py)
  and [export entry point](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/extension/llm/export/export_llm.py)
  are the authority for recipe keys, not snippets targeting a different revision.

Use a separate export environment when dependencies and model access are ready;
do not upgrade the image `.venv` or shared `/root/pytorch-env`. There is no tested
export-environment lock or completed installation yet.

## Why this route

The smallest proposed instruct model reduces the initial memory burden compared
with 3B, without promising it fits the unknown Jetson variant. INT4 may lower
weight memory but still requires tokenizer, KV cache, runner, vision, and OS
memory. Generated quality and quantization effects need actual tests.

CPU/XNNPACK is the first feasibility route documented for ARM64. GPU execution
and native binary compatibility remain separate investigations; do not assume
Jetson's GPU is used by this choice. Source export and target execution may
happen on different machines. Build the target runner for the actual Jetson
userspace/toolchain rather than copy a WSL x86 binary or Raspberry Pi setup.

The [official ARM tutorial](https://docs.pytorch.org/executorch/stable/raspberry_pi_llama_tutorial.html)
is a Raspberry Pi reference, not Jetson validation. Its toolchain and system-
library modification instructions are not adopted. No JetPack/glibc upgrade,
CPU affinity, or performance target has been selected without device facts.

## Read-only preflight

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.edge_preflight --require-jetson
```

On this PC it exits 2 with `NOT_VERIFIED_JETSON_ORIN_NANO`, records x86_64,
Python 3.12.14, absent ExecuTorch/export packages in the image interpreter, and
`inference_status=NOT_RUN`. The actual local report is ignored at
`artifacts/edge/pc_preflight.json`. Four parsing/guard tests passed. Their
synthetic host facts are unit tests, not Jetson evidence.

Later, on the Jetson in the project checkout:

```sh
python3 -B -m forgesight.edge_preflight --require-jetson --output artifacts/edge/jetson_preflight.json
```

A matching ARM64/device-tree/L4T identity is only a candidate for execution.
Verify exact module/RAM, JetPack packages, available memory/storage, and camera
separately. Preflight never claims successful inference or network disconnection.
It reads route tables without contacting an external service.

## Current blockers and next tasks

1. Device unavailable: exact Jetson facts, target build, and generated text absent.
2. Source model access/download blocker cleared in 2.4; user-confirmed access
   and all five publisher-matching local source files are recorded below.
3. Export environment, dependency closure, model export, and runner build not run.

Substep 2.2 prepares and validates source-pinned recipe files. Substep 2.3
provides a device-only capture harness. Those tools can be tested on the PC,
but Stage 2 cannot be DONE until actual Llama text is generated through
ExecuTorch on the Jetson and its provenance/measurements are reviewed.

## 2.2: Source-pinned export/build recipe, not an executed export

`forgesight/edge_recipe.py` renders two ignored files under
`artifacts/edge/export_recipe_v1/`: `export_config.yaml` and `recipe.json`.
The YAML file uses valid JSON syntax (a YAML subset), with the candidate's
checkpoint paths, XNNPACK/quantization settings, BOS/EOS metadata, and lengths.
The JSON recipe keeps commands as argument lists plus working directories,
including checkout/download, export, and native target build steps. It never
executes those commands or installs anything.

`configs/edge_upstream_schema.json` records field names parsed with Python's
AST parser from the pinned upstream dataclasses, their source commit, and the
exact source-file fingerprint. The upstream code was read, not imported or
executed. Recipe validation checks these keys and basic length/backend
constraints; it is not a full Hydra instantiation or successful export test.

The reviewed [native runner instructions](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/examples/models/llama/README.md)
use the `llm-release` and `llama-release` CMake workflows. Their names were
independently checked in both pinned CMake preset files. They remain NOT RUN
and require an appropriate native Jetson toolchain, dependency environment,
submodules, and CMake supporting those presets. No Pi toolchain is substituted.

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.edge_recipe
```

The observed output is `PREPARATION_ONLY_NOT_RUN`, with missing licensed model
files, absent matching source checkout, and no installed export environment.
`ready_to_claim_export` is false. Six recipe tests passed, covering schema
keys, source revision mismatch, lengths, backend substitution, path arguments,
missing input reporting, and existing-output preservation. No `.pte` was created.
The output directory must be new; `--output` can select another ignored path.

After model access, download the original checkpoint/params/tokenizer from the
pinned model revision and retain license notices locally. Verify their actual
hashes, review dependencies in a separate export environment, and validate the
recipe with that installed revision before exporting. Native CPU runner build
and ARM64 ABI checks happen on/for the actual Jetson. This preparation recipe
does not replace the still-missing reproducible installed dependency lock.

## 2.3: Device-only capture, pending real execution

`forgesight/edge_capture.py` collects the same preflight and refuses execution
unless it identifies a Linux ARM64 Jetson Orin Nano candidate. It also checks
the pinned source checkout, nonempty local artifacts, `.pte` extension, and a
native little-endian AArch64 ELF runner. There is no desktop override, alternate
LLM runtime, hosted call, or simulated model response.

On a future real run it fingerprints runner/model/tokenizer/prompt, passes an
argument list directly to the native runner, and saves raw stdout/stderr,
return code, elapsed process wall time, sampled RSS, and preflight before/after.
It bounds execution by timeout and log size. The pinned runner source was
checked for `model_path`, `tokenizer_path`, `prompt_file`, `max_new_tokens`,
`temperature`, and `num_bos`; deprecated generation-length flags are avoided.
The basic prompt is only a runtime smoke test, not the Stage 9 advisory design.

Wall time includes process startup/model loading and is not token latency or
vision latency. RSS is sampled every 50 ms, can miss peaks, and excludes other
processes. A route-table snapshot and operator-reported network state do not
prove offline operation; `offline_status` remains NOT VERIFIED. Exact module,
memory sufficiency, tokenizer behavior, model/export provenance, linked backend,
generated content, and device/runtime performance still require actual review.

The report records candidate/requested model IDs, not an automatic inspection
of model identity inside the artifact. A source checkout's HEAD does not attest
how an arbitrary binary was built. Match export/build logs, input/checkpoint
hashes and revisions to the artifact hashes when actual assets exist. Nonempty
stdout can contain echoed prompts or diagnostics, so even a successful process
is only CAPTURED_NOT_VERIFIED; it cannot automatically pass the Stage 2 gate.

Actual PC refusal test:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.edge_capture --output artifacts/edge/pc_capture_attempt_v1
```

Observed exit 2 and ignored `run.json`: BLOCKED_NOT_VERIFIED, missing target
identity/source checkout/runner/model/tokenizer, `process=null`, no generated
text, and Stage 2 NOT VERIFIED. No runner process was started.

Only after device audit, model access, real export, and native runner build,
run on the Jetson with paths appropriate to that checkout:

```sh
python3 -B -m forgesight.edge_capture --output artifacts/edge/jetson_capture_v1 --network-state connected
```

Use the actual observed network state; `physically_disconnected` is an operator
declaration, not a software-certified offline result. Output directories must
be new. Failure, timeout, or empty stdout remains FAILED_NOT_VERIFIED; a captured
run still needs its actual text/provenance reviewed. Stage 10 remains the full
offline system gate.

Six new capture tests passed; all 52 project tests passed together. The process
tests use real generic Python children for log/exit/timeout behavior, with no
pretend Llama generation, model artifacts, or Jetson success report.

## Stage 2 gate after these three preparation steps

Preparation tooling and downloaded source assets are verified, but installed
dependency closure, export, target compilation, real generated text, and measured on-device
execution remain BLOCKED or NOT RUN. Stage 2 is not DONE. Keep this gate open
while proceeding with independent PC vision work only after review, as allowed
by the roadmap. The next independent task is Stage 3.1: inspect compatible
PatchCore/EfficientAD baseline options for this PC without consuming the final
holdout. Resume actual Stage 2 execution when model/device access is available.

## 2.4: Actual source-model acquisition verified

The user confirmed access approval and completed the pinned download into
`models/llama3_2_1b/`. The verifier checked public metadata at the exact model
revision, not a moving branch, without reading authentication tokens. All five
files matched the publisher's sizes and object fingerprints. Large LFS objects
were compared using their published SHA-256; small files were compared using
their Git blob IDs. SHA-256 was additionally recorded for every local file.

| File | Actual bytes | Publisher check |
| --- | --- | --- |
| `original/consolidated.00.pth` | 2,471,677,246 | LFS SHA-256 matches |
| `original/params.json` | 220 | Git blob matches |
| `original/tokenizer.model` | 2,183,982 | LFS SHA-256 matches |
| `LICENSE.txt` | 7,712 | Git blob matches |
| `USE_POLICY.md` | 6,021 | Git blob matches |

Checkpoint SHA-256:
`fc17d497df5e4175b3a8acb4f5865b26f7fc1b009b25bef814b95fde10e8a1f3`.
Tokenizer SHA-256:
`82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55`.
Parameters describe 16 layers, hidden dimension 2048, 32 attention heads,
eight KV heads, and vocabulary size 128,256. The tokenizer contains 128,000
valid base64 token entries with unique tokens and contiguous ranks; reserved
special tokens are separate. No checkpoint deserialization or inference was
used to verify these files.

`manifests/llama3_2_1b_assets.json` commits metadata/fingerprints only. The
weights, tokenizer, license/policy copies, Hub cache, and verification reports
remain local under ignored paths. `configs/edge_llama.json` now records access
granted and source assets verified. This does not prove tokenization during
generation, export correctness, quantized quality, memory fit, or edge execution.

Repeat offline verification against the publisher-checked committed reference:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.edge_assets
```

Observed result: five files verified, 128,000 tokenizer base entries,
`ASSETS_VERIFIED_NOT_EXPORTED`, and inference NOT_RUN. Six new tests cover
integrity success, same-size corruption, missing files, path escape, duplicate
tokenizer ranks, and noncontiguous ranks. All 58 project tests passed.

A new ignored recipe at `artifacts/edge/export_recipe_assets_verified/` reflects
the current assets without overwriting the earlier blocked recipe. It reports
an empty missing-model-files list and no access blocker. Actual source checkout,
installed export environment/dependency lock, export, native build, and Jetson
execution are still unverified. Next is a separate Stage 2.5 export-environment
setup/compatibility check; it has not been started by this verification step.

## 2.5: Isolated export environment

This substep supersedes the earlier environment-not-installed status. No model
conversion or runner build was performed. Actual new workstation paths:

- Environment: `/root/.cache/forgesight/et1.5.1/venv`
- Source: `/root/.cache/forgesight/et1.5.1/executorch`
- Source HEAD: `3b60683923245cf472b7323426920e15623ba361`

These live on WSL's Linux filesystem. Complete before/after `pip freeze --all`
inventories matched for both the image `.venv` and shared `/root/pytorch-env`.
Submodules were not initialized: native runner compilation has separate
prerequisites. CMake is installed only in the new venv.

| Tool | Observed version |
| --- | --- |
| CPython / uv | 3.12.14 / 0.12.15 |
| ExecuTorch | 1.5.1+cpu |
| PyTorch / torchvision | 2.14.0+cpu / 0.29.0+cpu |
| TorchAO | 0.18.0 |
| transformers | 5.0.0rc1 (pinned upstream example requirement) |
| torchtune | Git `3c1872ace149f03ef4ffec765d3f0a5fd0399497` (reports 0.0.0) |
| hydra-core | 1.3.2 |
| CMake / Ninja | 3.31.10 / 1.13.2 |

The release's [dependency selection](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/install_requirements.py)
and [Llama requirements](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/examples/models/llama/install_requirements.sh)
inform `requirements/export.in`. We selected text-export dependencies and CPU
wheels rather than running the broad upstream development installer. Some
transitive packages support other backends; their presence proves no backend.

`requirements/pylock.export.toml` records 86 resolved dependencies, distribution
hashes, and torchtune's Git commit. uv's pylock support is experimental in the
tested version. This lock targets Linux x86_64 / CPython 3.12, not Windows or
ARM64. Python, OS, compiler, Git, uv, and isolated source-build tooling are
outside its reproducibility scope. Bundled pip is separate bootstrap tooling.

To reproduce, open Ubuntu WSL in the project directory. Use a **fresh** root;
the existing workstation root below must not be overwritten. This uses the
workstation's existing Python and uv:

```bash
export FORGESIGHT_EXPORT_ROOT=/root/.cache/forgesight/et1.5.1
test ! -e "$FORGESIGHT_EXPORT_ROOT" || { echo 'Path exists; inspect or choose another root.'; exit 1; }
mkdir -p "$FORGESIGHT_EXPORT_ROOT"
/root/pytorch-env/bin/python3 -m venv "$FORGESIGHT_EXPORT_ROOT/venv"
git clone --depth 1 --branch v1.5.1 https://github.com/pytorch/executorch.git "$FORGESIGHT_EXPORT_ROOT/executorch"
test "$(git -C "$FORGESIGHT_EXPORT_ROOT/executorch" rev-parse HEAD)" = 3b60683923245cf472b7323426920e15623ba361
/root/.local/bin/uv pip sync --python "$FORGESIGHT_EXPORT_ROOT/venv/bin/python" requirements/pylock.export.toml
/root/.local/bin/uv pip check --python "$FORGESIGHT_EXPORT_ROOT/venv/bin/python"
```

Check each command succeeds before continuing. Normal setup installs the
committed lock. For deliberate updates only, regenerate with the tested uv:
`uv pip compile requirements/export.in --python <export-python> --index-url https://pypi.org/simple --format pylock.toml --output-file requirements/pylock.export.toml`,
review changes and repeat the fresh-install check.

Actual verification commands run from PowerShell in the project directory:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/et1.5.1/venv/bin/python -B -m forgesight.edge_recipe --checkout /root/.cache/forgesight/et1.5.1/executorch --output artifacts/edge/export_recipe_env_verified
wsl -d Ubuntu --exec /root/.cache/forgesight/et1.5.1/venv/bin/python -B scripts/check_export_environment.py --checkout /root/.cache/forgesight/et1.5.1/executorch --config artifacts/edge/export_recipe_env_verified/export_config.yaml --output artifacts/edge/export_environment_v1
```

Those evidence directories now exist; repeats must use new output names.
The check validates our recipe through upstream OmegaConf into LlmConfig,
matches installed and checkout schema hashes to the committed reference,
performs CPU matrix multiplication (11.0), and runs the pinned source exporter's
actual `--help` (exit 0). It neither loads a checkpoint nor calls conversion.
Report: `EXPORT_ENVIRONMENT_VERIFIED_NOT_EXPORTED`; export/inference NOT_RUN.

TorchAO reports failures loading optional `_C_mxfp8` and `_C_cutlass_90a` GPU
libraries, plus upstream warnings. Raw import and CLI logs are preserved in the
ignored evidence directory. CPU startup succeeded; 8da4w quantization,
serialization, operator coverage, and model correctness remain untested.
Do not infer conversion compatibility from dependency metadata checks.

A fresh temporary venv installed solely from the committed lock and repeated
dependency and real exporter checks, then was automatically removed. Evidence:
`artifacts/edge/export_environment_lock_recheck/` and
`artifacts/edge/export_lock_recheck.log`. All 59 project tests passed in the
preserved image environment, including recognition of the official +cpu suffix
and rejection of other release/prerelease versions.

Next is Stage 2.6: attempt actual conversion with retained logs and record a
`.pte` hash only after success. Native runner build, quality, offline operation,
and performance remain unverified. Jetson Stage 2 is still BLOCKED - NOT VERIFIED.

## 2.6: Actual model export

The real pinned checkpoint was converted with the isolated environment and the
reviewed XNNPACK/8da4w configuration. This supersedes earlier no-export status.
Embedding quantization is 4-bit with group size 32; linear group size is 128.
Sequence/context limits remain 128/512. This is a candidate configuration,
not established operating capacity or a quality-validated assistant.

| Observed export evidence | Result |
| --- | --- |
| Exporter return code / completion | 0 / EXITED |
| Desktop conversion wall time | 248.46 seconds |
| Sampled export-process peak RSS | 8,237,948 KiB (50 ms sampling; excludes other processes and may miss peaks) |
| PTE bytes | 802,247,040 |
| PTE SHA-256 | `b82d0de5e0785269112d818be5a6c675225aab02c4e2a3b6c98d9e599da86739` |
| Program metadata load | Actual ExecuTorch InternalConsistency load succeeded |
| Serialized forward delegate | XnnpackBackend |
| Program metadata bytes / method count | 167,680 / 10 |
| Generation / quality / Jetson execution | NOT RUN / NOT VERIFIED / BLOCKED - NOT VERIFIED |

The artifact is local at
`artifacts/edge/export_attempt_01/exported/llama3_2_1b_xnnpack.pte`.
Only metadata is committed in `manifests/llama3_2_1b_export.json`, including
package versions, input/config/lock hashes, and the actual process result.
The artifact, raw logs, original assets, and provenance snapshots stay ignored.

The recorded invocation ran in the pinned ExecuTorch checkout, with
`OMP_NUM_THREADS=8`, a 1,800-second timeout and a 16 MiB log limit, using
`capture_process` from `forgesight.edge_capture`. Its exact argv is in the
manifest. Local evidence is under `artifacts/edge/export_attempt_01/`:
`conversion/stdout.log`, `conversion/stderr.log`, `invocation.json`,
`conversion_result.json`, `provenance.json`, and artifact verification reports.
The optional TorchAO GPU-library and upstream deprecation warnings were
retained. No dependency changes or source patches were needed for conversion.
Both original development environments' inventories still match their baseline.

### Output-path correction

The first config used `output_name: llama3_2_1b_xnnpack.pte`. In the pinned
upstream `_get_output_filename`, a name ending in .pte is treated as a verbatim
path and bypasses output_dir. The artifact landed in the source checkout.
That original remains preserved; the project copy has identical size/hash.
`artifact_copy.json` records both paths. The earlier config was not rewritten.

Future recipes now use `output_name: llama3_2_1b_xnnpack` with the same output_dir.
The real upstream filename resolver confirmed the intended full path in
`artifacts/edge/export_recipe_output_path_fixed/upstream_output_path_check.txt`.
Conversion was not rerun merely to test the filename change.

For a deliberate future re-export, generate a recipe in a fresh output
directory, then run its recorded command with the export venv, in the pinned
checkout, with OMP_NUM_THREADS=8. For example, from PowerShell in this project:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/et1.5.1/venv/bin/python -B -m forgesight.edge_recipe --checkout /root/.cache/forgesight/et1.5.1/executorch --output artifacts/edge/export_attempt_02
```

Then in Ubuntu WSL, from the ExecuTorch checkout:

```bash
OMP_NUM_THREADS=8 /root/.cache/forgesight/et1.5.1/venv/bin/python -B -m extension.llm.export.export_llm --config '/mnt/c/Users/RF AND SIMULATION/Desktop/honey/ForgeSight/artifacts/edge/export_attempt_02/export_config.yaml'
```

Choose a new attempt name and retain logs for every re-export. That example
shows how to reproduce conversion; attempt_02 was not run in this substep.
A new export needs its own reviewed fingerprint; do not overwrite this reference
merely because another binary differs.

### Verify the recorded artifact

From PowerShell in this project (choose a new evidence filename on repeats):

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/et1.5.1/venv/bin/python -B -m forgesight.edge_artifact --model artifacts/edge/export_attempt_01/exported/llama3_2_1b_xnnpack.pte --output artifacts/edge/export_attempt_01/artifact_recheck.json
```

The verifier checks the committed model/runtime reference, byte count, streaming
hash, and pinned ET12 schema identifier. Bounded metadata parsing uses private
helpers from this specific release; upgrades require review. It checks an
XNNPACK forward delegate, then calls the real Runtime.load_program with default
InternalConsistency verification. Methods include forward, BOS/EOS metadata,
context/sequence lengths, layer count, vocabulary, and KV-cache flags.

Observed status: `ARTIFACT_LOADED_NOT_INFERENCE_VERIFIED`. Listing methods does
not load/execute forward or establish operator coverage on ARM. No generation
method, including token-metadata methods, was executed. All 63 project tests
pass; four new rejection tests use generic invalid files, not pretend models.
The corrected filename was also checked through upstream's actual resolver.

Next, Stage 2.7 tests actual desktop token generation through ExecuTorch with
the matching tokenizer and exported model. PC output cannot close the mandatory
Jetson/device or offline gates. Quantized quality and real operator advice
require later evaluation even after a generation smoke test succeeds.

## 2.7: Real desktop generation

The recorded PTE now produces actual text through ExecuTorch's native C++
TextLLMRunner, called via its Python bindings on Linux x86_64 WSL. No other
model or runtime was substituted and no new packages or source build were
needed. This supersedes earlier no-generation snapshots. Jetson proof is
still BLOCKED - NOT VERIFIED.

### Runtime compatibility found by the actual attempt

The first attempt, `artifacts/edge/pc_generation_attempt_01/`, exited 1 before
generating text. The runtime reported 49 instructions missing registrations
across three operator types: quantized_decomposed::embedding_4bit.dtype_out,
llama::update_cache.out, and llama::custom_sdpa.out. Metadata loading alone had
not tested these operators. All failed-attempt logs and its empty output remain
preserved; no success was inferred from loading the program metadata.

The wheel contains the needed kernels but does not auto-load both libraries
for TextLLMRunner. The worker now registers them using torch.ops.load_library:

- `lib/libexecutorch_kernels_quantized.so`
- `extension/llm/custom_ops/libcustom_ops_aot_lib.so`

Both are located under the installed executorch package, not arbitrary system
paths. Their static registrations feed the wheel's shared ExecuTorch runtime.
Native registry inspection changed all three missing operator types from absent
to present, and the following actual generation succeeded. The native runner
binding and both kernel libraries have sizes/hashes in the smoke manifest.
The exported model/tokenizer retain their earlier verified fingerprints.

Relevant upstream source is pinned to the existing ExecuTorch commit:
[native bindings](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/extension/llm/runner/pybindings.cpp),
[runner/helper](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/extension/llm/runner/llm_runner_helper.cpp),
and [custom-kernel build](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/extension/llm/custom_ops/CMakeLists.txt).
The simple user/assistant prompt follows Meta's
[Llama 3 chat format](https://github.com/meta-llama/llama-models/blob/main/models/llama3/chat_format.py).
The installed C++ TikToken tokenizer was checked to encode the control markers
as 128000/128006/128007/128009, with exactly one BOS. num_bos/num_eos are zero
because the formatted prompt already contains its control tokens.

### Actual input and output

Prompt from `configs/edge_smoke_prompt.txt`:

> In one sentence, explain why stable lighting helps compare two inspection images. Do not claim that you inspected a part.

Actual answer (the raw evidence additionally retains `<|eot_id|>`):

> Stable lighting helps compare two inspection images by ensuring that the lighting conditions are consistent, allowing the comparison of the images to be more accurate and reliable.

This was reviewed as a coherent general answer with no claim of inspecting a
part. It is a functional smoke test, not advice-quality validation or evidence
of a defect-inspection application. Temperature was zero and echo disabled;
the runner generated the response from the PTE rather than repeating the input.

Final evidence: `artifacts/edge/pc_generation_verified/`. The CLI deliberately
reports `PC_GENERATION_CAPTURED_REQUIRES_REVIEW`; after reviewing the real
transcript, `manifests/llama3_2_1b_pc_smoke.json` records the scoped result
`PC_SMOKE_VERIFIED_NOT_JETSON_PROOF`. That committed record includes the actual
prompt, raw generated text, binary fingerprints, config, token IDs, native
statistics, process result, and source/reference hashes. Earlier attempts and
raw logs remain local/ignored.

| Final single-run desktop observation | Result |
| --- | --- |
| Return code / completion | 0 / EXITED |
| Prompt / native-reported generated tokens | 33 / 30 |
| Token callback pieces | 31, including the raw end-of-turn marker; not treated as the native token counter |
| Maximum new tokens / context | 64 / 512 |
| Native model-load time | 2.902 seconds |
| Native inference interval | 1.154 seconds |
| Python generate call | 4.077 seconds, including lazy loading |
| Whole captured process | 8.893 seconds, including imports and input verification |
| Sampled process peak RSS | 1,218,376 KiB; 50 ms sampling, may miss peaks and excludes other processes |

These are one desktop run's measurements, not p50/p95 or predicted Jetson
performance. No warmup or controlled benchmark was performed. Timing scopes
differ and must not be interchanged. Eight PyTorch threads and OMP_NUM_THREADS=8
were requested; the manifest does not certify the size of every native pool.

### Repeat the bounded PC smoke test

From PowerShell in this project, choose a new output directory:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/et1.5.1/venv/bin/python -B -m forgesight.edge_pc_smoke --output artifacts/edge/pc_generation_recheck
```

The parent captures a worker process with a 300-second default timeout and
16 MiB log bound. It retains stdout/stderr, request, generated.txt, inputs,
native statistics, and run.json. Existing evidence directories are refused.
The worker is explicitly desktop-only; the original device capture still
requires the actual identified Jetson. The tokenizer checks enforce the exported
prefill/context bounds and the native prompt count must match the precheck.

HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are set in the worker; all model and
tokenizer inputs are local. Those flags do not prove network isolation or the
mandatory disconnected Jetson run. The native loader first tries JSON and logs
a parse error before successfully selecting TikToken; WSL also reports failed
ARM CPU-information probes. These raw warnings are preserved, not hidden.

All 67 project tests passed, including four new prompt/control/token-budget
guard tests without simulated generation. Package compatibility passes and both
original environment inventories remain unchanged. Advice quality, device
deployment, camera/vision, and offline end-to-end evidence remain open.

Next is Stage 3.1: check PatchCore/EfficientAD compatibility and select a vision
baseline using the frozen training/validation policy, without opening the final
holdout. Fit/train follows the selection in a separate substep. Stage 2's Jetson
gate stays open under the roadmap's explicit hardware-blocker rule.
