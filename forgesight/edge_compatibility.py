"""Static ARM wheel/library checks; never install packages or execute wheel binaries.

Python 3.8-compatible so --collect-target can run in memory on the audited Jetson.
Passing static checks is not proof of imports, kernel execution or generation.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import zipfile


SYSTEM_LIBRARIES = ('ld-linux-aarch64.so.1', 'libatomic.so.1', 'libc.so.6',
                    'libdl.so.2', 'libgcc_s.so.1', 'libm.so.6', 'libpthread.so.0',
                    'librt.so.1', 'libstdc++.so.6', 'libz.so.1')
REQUIRED_MEMBERS = (
    'executorch/extension/llm/runner/_llm_runner.cpython-312-aarch64-linux-gnu.so',
    'executorch/extension/llm/custom_ops/libcustom_ops_aot_lib.so',
    'executorch/lib/libexecutorch_kernels_quantized.so',
    'executorch/lib/libexecutorch_backend_xnnpack.so',
    'pytorch_tokenizers/pytorch_tokenizers_cpp.cpython-312-aarch64-linux-gnu.so',
    'torch/lib/libtorch_cpu.so',
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def elf_machine(header):
    if len(header) < 20 or header[:4] != b'\x7fELF' or header[4] != 2 or header[5] not in (1, 2):
        raise ValueError('Expected a 64-bit ELF header.')
    return int.from_bytes(header[18:20], 'little' if header[5] == 1 else 'big')


def definitions(text):
    if 'Version definition section' not in text:
        return []
    section = text.split('Version definition section', 1)[1].split('Version needs section', 1)[0]
    return sorted(set(re.findall(r'Name:\s+(\S+)', section)))


def requirements(text):
    if 'Version needs section' not in text:
        return {}
    current, result = None, {}
    for line in text.split('Version needs section', 1)[1].splitlines():
        match = re.search(r'File:\s+(\S+)', line)
        if match:
            current = match[1]
            result.setdefault(current, [])
        match = re.search(r'Name:\s+(\S+)', line)
        if match and current:
            result[current].append(match[1])
    return {name: sorted(set(values)) for name, values in result.items()}


def readelf(path, option):
    return subprocess.check_output(['readelf', option, str(path)], text=True, timeout=30)


def collect_target():
    """Only inspect the loader cache/library files and Python metadata; no device writes."""
    loader = shutil.which('ldconfig') or '/sbin/ldconfig'
    cache = subprocess.check_output([loader, '-p'], text=True, timeout=10)
    libraries = {}
    for name in SYSTEM_LIBRARIES:
        matches = re.findall(r'^\s*' + re.escape(name) + r'\s+.*?=>\s+(\S+)', cache, re.MULTILINE)
        if not matches:
            libraries[name] = {'error': 'NOT_IN_LOADER_CACHE'}
            continue
        path = Path(matches[0]).resolve()
        try:
            with path.open('rb') as stream:
                machine = elf_machine(stream.read(20))
            libraries[name] = {'path': str(path), 'sha256': digest(path), 'machine': machine,
                               'provided_versions': definitions(readelf(path, '--version-info'))}
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            libraries[name] = {'path': str(path), 'error': str(exc)}
    import importlib.metadata
    packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())
    return {'scope': 'READ_ONLY_TARGET_LIBRARY_PROBE_NOT_MODEL_EXECUTION',
            'machine': platform.machine(), 'python': platform.python_version(),
            'python_executable': sys.executable, 'libc': list(platform.libc_ver()),
            'libraries': libraries, 'system_python_packages': packages,
            'packages_installed': False, 'target_files_written': False,
            'model_executed': False}


def assess(probes, target):
    errors = []
    if target.get('machine') != 'aarch64':
        errors.append('Target is not the audited aarch64 architecture.')
    members = {row['member'] for row in probes}
    errors.extend('Missing required runtime member: ' + name for name in REQUIRED_MEMBERS if name not in members)
    bundled = {}
    for row in probes:
        if row['machine'] != 183:
            errors.append('Wrong ELF architecture: ' + row['member'])
        bundled.setdefault(PurePosixPath(row['member']).name, []).append(row)
    external = set()
    for row in probes:
        for soname in row['needed']:
            required = set(row['required_versions'].get(soname, []))
            if soname in bundled:
                candidates = bundled[soname]
                available = any(required <= set(candidate['provided_versions']) for candidate in candidates)
            else:
                external.add(soname)
                candidate = target['libraries'].get(soname, {})
                available = (not candidate.get('error') and candidate.get('machine') == 183
                             and required <= set(candidate.get('provided_versions', [])))
            if not available:
                errors.append('{} needs unavailable library/version {} {}'.format(row['member'], soname, sorted(required)))
            if any(token in soname.lower() for token in ('libcuda', 'libcudart', 'libcublas')):
                errors.append('Unexpected CUDA dependency in the CPU reference runtime: ' + soname)
    return {'status': 'STATIC_CHECKS_MATCH_NOT_RUNTIME_PROOF' if not errors else 'STATIC_CHECKS_FAILED',
            'errors': sorted(set(errors)), 'elf_files_checked': len(probes),
            'system_libraries_required': sorted(external),
            'imports_verified': False, 'kernel_execution_verified': False,
            'llama_generation_verified': False, 'memory_or_latency_verified': False,
            'limitation': 'Library/version presence is necessary only. Loader search paths, CPU instructions, Python imports, operator registration and the full model must be tested on the real target.'}


def inspect(wheel_records, root, output, target):
    if output.exists():
        raise FileExistsError('Preserve existing probe outputs; choose a new directory.')
    output.mkdir(parents=True, exist_ok=False)
    probes = []
    for record in wheel_records:
        if record['label'] == 'pypi_executorch':
            continue  # Comparison artifact only; recipe uses the same +cpu source as the PC.
        if (Path(record['filename']).name != record['filename']
                or not record['filename'].endswith('.whl')
                or not re.fullmatch(r'[A-Za-z0-9_]+', record['label'])):
            raise ValueError('Wheel reference must name a file within the cache.')
        wheel_path = root / record['filename']
        if digest(wheel_path) != record['sha256']:
            raise ValueError('Wheel checksum differs from publisher reference: ' + record['filename'])
        with zipfile.ZipFile(wheel_path) as archive:
            for entry in archive.infolist():
                if not (entry.filename.endswith('.so') or '.so.' in entry.filename):
                    continue
                member = PurePosixPath(entry.filename)
                if (member.is_absolute() or '..' in member.parts or '\\' in entry.filename
                        or re.match(r'^[A-Za-z]:', entry.filename)):
                    raise ValueError('Unsafe archive member.')
                path = output / record['label'] / member
                if output.resolve() not in path.resolve().parents:
                    raise ValueError('Unsafe archive member.')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(archive.read(entry))
                with path.open('rb') as stream:
                    header = stream.read(20)
                if header[:4] != b'\x7fELF':
                    continue
                dynamic, versions = readelf(path, '-d'), readelf(path, '--version-info')
                probes.append({'member': entry.filename, 'wheel': record['label'],
                               'sha256': digest(path), 'machine': elf_machine(header),
                               'needed': re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic),
                               'required_versions': requirements(versions),
                               'provided_versions': definitions(versions),
                               'rpath': re.findall(r'\((?:RPATH|RUNPATH)\).*?\[(.*?)\]', dynamic)})
    report = {'scope': 'PC_STATIC_INSPECTION_OF_AUTHENTICATED_TARGET_AND_ARM_WHEELS',
              'target': target, 'probes': probes, 'assessment': assess(probes, target)}
    with (output/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-target', action='store_true')
    parser.add_argument('--wheels', type=Path)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.collect_target:
        print(json.dumps(collect_target(), indent=2))
        return
    if not all((args.wheels, args.target, args.output)):
        parser.error('Use --collect-target or provide --wheels, --target and --output.')
    report = inspect(json.loads(args.wheels.read_text()), args.wheels.parent,
                     args.output, json.loads(args.target.read_text()))
    print(json.dumps(report['assessment'], indent=2))
    if report['assessment']['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
