"""Actual Jetson isolated-runtime imports and registration; no model is loaded."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import resource
import site
import sys
import time


REQUIRED_OPS = ('quantized_decomposed::embedding_4bit.dtype_out',
                'llama::update_cache.out', 'llama::custom_sdpa.out')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def probe(prefix):
    started = time.monotonic()
    prefix = prefix.resolve()
    if (platform.system() != 'Linux' or platform.machine() != 'aarch64'
            or 'jetson orin nano' not in Path('/proc/device-tree/model').read_text().lower()):
        raise ValueError('This probe requires the actual Jetson Orin Nano.')
    if (platform.python_version() != '3.12.14' or Path(sys.prefix).resolve() != prefix/'venv'
            or Path(sys.base_prefix).resolve() != prefix/'python' or site.ENABLE_USER_SITE is not False):
        raise ValueError('Unexpected interpreter or environment isolation.')
    expected = json.loads((prefix/'stage/package_versions.json').read_text())
    installed = {re.sub(r'[-_.]+', '-', d.metadata['Name']).lower(): d.version
                 for d in importlib.metadata.distributions()}
    for name, version in expected.items():
        if installed.get(name) != version:
            raise ValueError('Wrong or missing package: ' + name)

    import torch
    import executorch
    from executorch.runtime import Runtime
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
    import executorch.extension.llm.runner._llm_runner as native
    import pytorch_tokenizers
    from pytorch_tokenizers import CppTiktoken

    for module in (torch, executorch, native, pytorch_tokenizers):
        location = Path(module.__file__) if getattr(module, '__file__', None) else Path(next(iter(module.__path__)))
        if prefix/'venv' not in location.resolve().parents:
            raise ValueError('Module loaded outside the isolated environment: ' + str(location))
    if torch.version.cuda is not None:
        raise ValueError('Expected CPU-only Torch, separate from CUDA vision.')
    torch.set_num_threads(1)
    result = (torch.arange(4, dtype=torch.float32).reshape(2, 2) + 1).sum().item()
    if result != 10.0:
        raise ValueError('CPU tensor arithmetic failed.')
    runtime = Runtime.get()
    before = {name: name in runtime.operator_registry.operator_names for name in REQUIRED_OPS}
    libraries = []
    for relative in ('lib/libexecutorch_kernels_quantized.so',
                     'extension/llm/custom_ops/libcustom_ops_aot_lib.so'):
        path = Path(next(iter(executorch.__path__)))/relative
        torch.ops.load_library(str(path))
        libraries.append({'relative_path': relative, 'sha256': sha256(path), 'bytes': path.stat().st_size})
    after = {name: name in runtime.operator_registry.operator_names for name in REQUIRED_OPS}
    backends = runtime.backend_registry.registered_backend_names
    if callable(backends):
        backends = backends()
    tokenizer = CppTiktoken()
    config = GenerationConfig(max_new_tokens=1)
    report = {'status': 'JETSON_RUNTIME_IMPORTS_AND_REGISTRATION_VERIFIED_NOT_GENERATION',
              'scope': 'ACTUAL_ARM64_ENVIRONMENT_SMOKE_NO_MODEL_LOADED',
              'host': {'machine': platform.machine(), 'python': platform.python_version(),
                       'executable': sys.executable, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
                       'user_site_enabled': site.ENABLE_USER_SITE},
              'packages': installed, 'tensor_cpu_sum': result, 'torch_cuda_build': torch.version.cuda,
              'torch_threads_for_probe_only': torch.get_num_threads(),
              'native_runner': {'path': native.__file__, 'sha256': sha256(native.__file__)},
              'kernel_libraries': libraries, 'operators_before_registration': before,
              'operators_after_registration': after, 'registered_backends': sorted(backends),
              'native_classes_imported': [TextLLMRunner.__name__, type(config).__name__, type(tokenizer).__name__],
              'elapsed_seconds': time.monotonic()-started,
              'process_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'memory_scope': 'This one import/tensor probe only; excludes future Llama/vision/UI memory.',
              'model_loaded': False, 'llama_generation_verified': False,
              'camera_vision_verified': False, 'offline_network_disconnection_verified': False}
    if not all(after.values()) or 'XnnpackBackend' not in backends:
        report['status'] = 'KERNEL_OR_BACKEND_REGISTRATION_INCOMPLETE'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing native probe evidence.')
    report = probe(args.prefix)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(report['status'])
    if report['status'] != 'JETSON_RUNTIME_IMPORTS_AND_REGISTRATION_VERIFIED_NOT_GENERATION':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
