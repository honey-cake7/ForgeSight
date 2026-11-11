# Jetson Llama runtime recipe — Substep 2.8

Status: **interpreter archive and core static checks match; nothing installed
on Jetson**. This is a recipe and dependency investigation, not proof
of imports, generation, performance or offline operation.

## Selected first route

Use a separate ARM64 CPython 3.12.14 environment with ExecuTorch 1.5.1+cpu,
PyTorch 2.14.0+cpu, pytorch-tokenizers 1.6.0 and TorchAO 0.18.0. Keep the
existing Llama 3.2 1B artifact, 128-token prefill bound and 512-token context.
The first bounded generation will use at most 64 new tokens. Do not substitute
another Llama runtime or copy the desktop x86_64 binary environment.

This route follows the already working PC native bindings, with matching
versions and operator libraries. CPU/XNNPACK advice can live separately from
the later CUDA vision environment. It is a feasibility choice, not a claim of
better Jetson latency. Thread count and combined CPU/RAM pressure need real
measurement; the desktop eight-thread setting is not a target budget.

The system Python remains `/usr/bin/python3` 3.8.10, Ubuntu remains 20.04.6,
L4T remains 35.6.1 and CUDA remains 11.4. All target commands in this step read
library metadata. The deleted manual-audit folder was not recreated.

| Option | Trade-off | Decision |
| --- | --- | --- |
| Isolated Python + verified ARM CPU wheels | Closest to tested PC bindings; no native build needed if actual imports work; brings Torch and export-related package dependencies | First candidate |
| Native C++ ExecuTorch runner | Could reduce Python/export-package overhead, but needs additional build/toolchain and adapter work | Fallback if runtime or resource tests fail |
| Change JetPack/Ubuntu | Changes a working camera/driver baseline; system migration has wider consequences | Not selected; static findings do not establish a need |

## Actual evidence

Four official ARM64 wheels were downloaded to the PC and checked against
publisher SHA-256 values: CPU ExecuTorch, CPU Torch, pytorch-tokenizers and the
plain PyPI ExecuTorch comparison wheel. No ARM wheel was installed or executed
on the PC or Jetson. The selected CPU wheel contains the actual native runner,
XNNPACK delegate, quantized kernels and LLM custom-operator library.

The reusable probe inspected **33 ELF binaries** in the three selected core
wheels. Each reports AArch64 machine ID 183. Its exact required GNU library
versions were compared with definitions read from **ten real Jetson system
libraries**, in an authenticated read-only SSH session. All checked requirements
matched. The highest core GLIBC requirement is 2.28; target libc is 2.31. No
CUDA library dependency was observed in these selected core ELF dependencies.

This is necessary static evidence only. It does not prove loader search paths,
CPU-instruction compatibility, Python imports, operator registration, model
execution, combined memory fit or performance. Other resolved dependencies
have compatible wheel tags according to the target resolver, but their complete
ELF/runtime behavior was not checked by this core probe. The interpreter archive
was subsequently acquired and inspected as described below; its executable has
not been run.

`requirements/pylock.jetson_llama.toml` records **49 resolved packages** for
CPython 3.12.14 and `aarch64-manylinux_2_31`. The hash-locked requirements file
provides pip-compatible installation input. These are CPU runtime dependencies;
they must not replace the Jetson's CUDA vision packages or system packages.

One dependency, antlr4-python3-runtime 4.9.3, is source-only in the lock. Its
publisher-hashed source successfully built into a `py3-none-any` wheel with no
native binaries in a new isolated PC environment, using the pinned setuptools
84.2.1 and wheel 0.45.1 build tools. The five existing environments are preserved.
No target build/install was attempted. A future target installation must use the
locked build tools and `--no-build-isolation`, so build-tool selection is controlled.

The interpreter asset is pinned to CPython 3.12.14, standalone build 20260901,
SHA-256 `577b4bec0793ad1ff0cbff9adbd0df078eddde38a4c41bf5d83ad381a85ee39d`.
Official release metadata confirms its name and 29,199,399-byte size. The initial
Astral URL returned HTTP 403 and canonical GitHub attempts timed out. Those
failures and the initial SSH banner timeout remain retained, not counted as
successful checks. After the user restored network connectivity, the canonical
GitHub download completed and matched the exact size and pinned SHA-256.
An initial CPU-wheel URL encoding error was corrected; all four wheel hashes
then passed.

The interpreter follow-up checked all 4,533 archive paths and 1,048 links for
prefix containment. Its ten ELF files are ARM64; `venv` and `ensurepip` source
files are present. The extra required `libcrypt.so.1` and `libutil.so.1` were
read from the actual Jetson after SSH connectivity was restored. Their required
GLIBC_2.17 definitions are present. Combined with the original core probe, all
43 checked native files match 12 system-library inventories. The archive's
literal `$ORIGIN/../lib/libpython3.12.so.1.0` dependency resolves to the included
library within the archive. This is static resolution, not a loader execution test.

The target still reports Python 3.8.10. Its package inventory and the hashes of
the previous ten system libraries match the earlier read-only probe. No new
interpreter/package/model was installed, executed or copied to the Jetson; only
in-memory system-Python queries ran. `manifests/jetson_python_archive_v1.json`
records this follow-up. Old failure evidence and the initial compatibility
manifest remain historical snapshots. The broad PC tests were not rerun because
runtime code is unchanged; the last full suite result is 140 passing tests.

## Files and reproducible PC checks

- `configs/jetson_llama_runtime_v1.json`: exact candidate, limits, hashes and gates.
- `requirements/jetson_llama.in`: four direct runtime requirements.
- `requirements/pylock.jetson_llama.toml`: ARM-targeted dependency resolution.
- `requirements/jetson_llama.lock.txt`: hash-locked runtime installation input.
- `requirements/jetson_build_tools.in` and `.lock.txt`: pinned source-build tools.
- `forgesight/edge_compatibility.py`: static inspector and read-only target collector.
- `manifests/jetson_runtime_compatibility_v1.json`: actual results and limitations.

On the configured PC, choose a fresh output directory:

```powershell
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.edge_compatibility --wheels artifacts/edge/jetson_compatibility_v1/downloaded_wheels.json --target artifacts/edge/jetson_compatibility_v1/target_libraries.json --output artifacts/edge/jetson_compatibility_v1/recheck-001
wsl -d Ubuntu --exec /root/.cache/forgesight/app-v1/venv/bin/python -B -m unittest tests.test_edge_compatibility -q
```

The first command parses wheel binaries with `readelf` on the PC; it does not
execute them. The target collector (`--collect-target`) runs from memory with
Python 3.8-compatible source, reads loader-cache/version metadata, and writes no
target files. Missing libraries, symbol versions, required runner files, wrong
architecture, wheel hash changes and unexpected CUDA dependencies fail checks.

Local evidence is under `artifacts/edge/jetson_compatibility_v1/`, including
downloaded wheels, target-library data, ELF reports, Python release metadata,
download failures and the isolated pure-Python build log. Raw artifacts stay
outside Git; no model/data or credential values are committed.

## Planned installation sequence — not executed

The exact Python archive and checked native requirements are now verified.
The next substep can prepare the isolated environment in this sequence:

1. Use a **fresh** user-owned prefix beneath `~/.local/share/forgesight/`.
   Refuse an existing prefix; never overwrite it or change `/usr/bin/python3`.
2. Extract the verified interpreter there and invoke its absolute executable
   to create a separate venv. Verify Python 3.12.14, AArch64 and isolation.
3. Stage the committed hash locks. From that isolated venv, the planned pip
   operations are:

   ```sh
   python -m pip install --require-hashes -r jetson_build_tools.lock.txt
   python -m pip install --require-hashes --no-build-isolation -r jetson_llama.lock.txt
   python -m pip check
   ```

   Here `python` must be the isolated venv executable. The commands above have
   not been run on the Jetson and are not a system-Python installation recipe.
4. Test CPU Torch, native bindings, tokenizer and actual operator registration.
   Record failures and memory before model transfer; don't install the GPU
   vision stack into this CPU-only environment.
5. Add an explicit ARM/device adapter with provenance and limits. Keep the
   desktop worker's x86 guard intact. Only then transfer the verified PTE and
   tokenizer and test bounded Llama generation through native ExecuTorch.

Next exact action, **Substep 2.9**: create/check only the isolated CPU
runtime. Full model generation is its own proof gate. Stop for review after
this compatibility substep. Stage 2 remains incomplete.

## Primary sources

- [Pinned ExecuTorch package metadata](https://pypi.org/pypi/executorch/1.5.1/json)
- [Official CPU wheel index](https://download.pytorch.org/whl/cpu/executorch/)
- [Pinned LLM Python bindings](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/extension/llm/runner/__init__.py)
- [Pinned native build](https://github.com/pytorch/executorch/blob/3b60683923245cf472b7323426920e15623ba361/CMakeLists.txt)
- [Python standalone release](https://github.com/astral-sh/python-build-standalone/releases/tag/20260901)
- [Release asset metadata](https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/20260901)

The Raspberry Pi tutorial is an ARM reference, not Jetson proof; its system-
library replacement instructions are not adopted. No on-device generation,
latency, camera inference or disconnected operation is established by this recipe.
