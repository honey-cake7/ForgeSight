"""Guarded isolated Jetson CPU runtime setup; no model transfer or generation.

Bootstrap commands work with system Python 3.8. Persistent installation files
stay in a fresh user-owned prefix; pip may use OS temporary build storage.
An existing unmarked directory is never reused.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import uuid


BASELINE_FILES = ('/etc/os-release', '/etc/nv_tegra_release', '/etc/nvpmodel.conf',
                  '/usr/bin/python3', '/usr/local/cuda/version.json')
MARKER = '.forgesight-runtime.json'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, document):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_prefix(prefix, home):
    home = Path(home).resolve()
    prefix = Path(prefix)
    if prefix.is_symlink():
        raise ValueError('A symlink cannot be a runtime prefix.')
    resolved = prefix.resolve()
    try:
        parts = resolved.relative_to(home).parts
    except ValueError as exc:
        raise ValueError('Runtime must stay within the user home.') from exc
    if len(parts) != 4 or parts[:3] != ('.local', 'share', 'forgesight'):
        raise ValueError('Use one fresh directory beneath ~/.local/share/forgesight/.')
    return resolved


def snapshot_system():
    code = "import importlib.metadata,json,platform;print(json.dumps({'python':platform.python_version(),'packages':sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())}))"
    result = json.loads(subprocess.check_output(['/usr/bin/python3', '-B', '-c', code], text=True, timeout=30))
    result['file_hashes'] = {path: digest(Path(path)) for path in BASELINE_FILES}
    return result


def initialize(prefix):
    prefix = validate_prefix(prefix, Path.home())
    if prefix.exists():
        raise FileExistsError('Runtime prefix already exists; preserve and inspect it.')
    model = Path('/proc/device-tree/model').read_text().lower()
    if platform.system() != 'Linux' or platform.machine() != 'aarch64' or 'jetson orin nano' not in model:
        raise ValueError('Setup requires the actual identified Jetson Orin Nano.')
    if os.geteuid() == 0:
        raise ValueError('Run as the normal device user, without sudo.')
    before = snapshot_system()
    if before['python'] != '3.8.10' or shutil.disk_usage(Path.home()).free < 8 * 1024**3:
        raise ValueError('Unexpected baseline Python or insufficient free storage; inspect first.')
    prefix.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in ('stage', 'logs', 'cache'):
        (prefix/name).mkdir()
    write_json(prefix/'system_before.json', before)
    write_json(prefix/MARKER, {'schema_version': 1, 'purpose': 'FORGESIGHT_STAGE_2_9_CPU_RUNTIME',
                             'prefix': str(prefix), 'created_utc': datetime.now(timezone.utc).isoformat(),
                             'run_id': uuid.uuid4().hex})
    return prefix


def require_prefix(prefix):
    prefix = validate_prefix(prefix, Path.home())
    marker = json.loads((prefix/MARKER).read_text())
    if marker.get('purpose') != 'FORGESIGHT_STAGE_2_9_CPU_RUNTIME' or marker.get('prefix') != str(prefix):
        raise ValueError('Prefix is not a marked ForgeSight CPU runtime.')
    return prefix


def validate_member(member, destination, required_top=None):
    name = PurePosixPath(member.name)
    if (name.is_absolute() or '..' in name.parts or '\\' in member.name
            or re.match(r'^[A-Za-z]:', member.name) or member.isdev() or member.isfifo()):
        raise ValueError('Unsafe archive member.')
    if required_top and (not name.parts or name.parts[0] != required_top):
        raise ValueError('Unexpected archive root.')
    path = (destination/Path(*name.parts)).resolve()
    if path != destination.resolve() and destination.resolve() not in path.parents:
        raise ValueError('Archive path escapes its destination.')
    if member.issym() or member.islnk():
        target = member.linkname if member.islnk() else posixpath.join(str(name.parent), member.linkname)
        normalized = PurePosixPath(posixpath.normpath(target))
        if (normalized.is_absolute() or '..' in normalized.parts or '\\' in target
                or re.match(r'^[A-Za-z]:', member.linkname)
                or (required_top and (not normalized.parts or normalized.parts[0] != required_top))):
            raise ValueError('Archive link escapes its destination.')


def extract_checked(archive, destination, expected, required_top=None):
    if digest(archive) != expected:
        raise ValueError('Archive checksum mismatch.')
    with tarfile.open(archive) as package:
        members = package.getmembers()
        for member in members:
            validate_member(member, destination, required_top)
        for member in members:
            # Recheck after earlier members create links; strip special mode bits.
            validate_member(member, destination, required_top)
            member.mode &= 0o777
            package.extract(member, path=destination)


def bootstrap(prefix, archive_sha, bundle_sha):
    prefix = require_prefix(prefix)
    if (prefix/'python').exists() or (prefix/'stage/wheelhouse').exists():
        raise FileExistsError('Existing extracted files are preserved; do not bootstrap twice.')
    extract_checked(prefix/'stage/python.tar.gz', prefix, archive_sha, 'python')
    extract_checked(prefix/'stage/inputs.tar', prefix/'stage', bundle_sha)
    manifest = json.loads((prefix/'stage/transfer_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = prefix/'stage'/relative
        if (prefix/'stage').resolve() not in path.resolve().parents or digest(path) != expected:
            raise ValueError('Staged input differs from its transfer reference: ' + relative)
    write_json(prefix/'bootstrap.json', {'archive_sha256': archive_sha, 'bundle_sha256': bundle_sha,
                                        'staged_files': len(manifest['files'])})


def isolated_environment(prefix):
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith('PIP_') or name in ('PYTHONPATH', 'PYTHONHOME'):
            environment.pop(name)
    environment.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
                        'PIP_CONFIG_FILE': os.devnull, 'OMP_NUM_THREADS': '1',
                        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                        'HF_HOME': str(prefix/'cache/hf'), 'XDG_CACHE_HOME': str(prefix/'cache')})
    return environment


def make_offline_lock(text):
    """Use staged copies of the same hashed CPU wheels instead of HTTPS selectors."""
    for name, version in (('torch', '2.14.0+cpu'), ('executorch', '1.5.1+cpu')):
        replacement = name + '==' + version + ' ' + chr(92)
        text, count = re.subn(r'^' + name + r' @ [^\n]+', lambda _: replacement,
                             text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError('Expected one pinned direct CPU wheel for ' + name)
    if re.search(r'^\s*[^#\n]+ @ ', text, re.MULTILINE):
        raise ValueError('Unexpected direct URL remains in the offline lock.')
    return text


def run_logged(prefix, phase, argv, timeout=1200):
    log = prefix/'logs'/('{}-{}.log'.format(phase, uuid.uuid4().hex[:8]))
    print('PHASE:' + phase, flush=True)
    with log.open('xb') as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                cwd=prefix, env=isolated_environment(prefix), timeout=timeout)
    if result.returncode:
        raise RuntimeError('{} failed with exit {}; log {}'.format(phase, result.returncode, log))
    print('OK:' + phase, flush=True)
    return {'argv': argv, 'returncode': result.returncode, 'log': str(log), 'log_sha256': digest(log)}


def install(prefix):
    prefix = require_prefix(prefix)
    profile = json.loads((prefix/'stage/recipe.json').read_text())
    if not profile['python']['download_verified'] or profile['target']['architecture'] != 'aarch64':
        raise ValueError('Expected the verified ARM runtime recipe.')
    for relative, expected in json.loads((prefix/'stage/transfer_manifest.json').read_text())['files'].items():
        if digest(prefix/'stage'/relative) != expected:
            raise ValueError('Staged file changed: ' + relative)
    interpreter = prefix/'python/bin/python3.12'
    check = "import json,platform,sys;assert platform.machine()=='aarch64';assert platform.python_version()=='3.12.14';print(json.dumps({'python':platform.python_version(),'executable':sys.executable,'prefix':sys.prefix}))"
    steps = [run_logged(prefix, 'interpreter', [str(interpreter), '-B', '-c', check], 60)]
    if (prefix/'venv').exists():
        raise FileExistsError('Existing venv is preserved; inspect its state before a retry.')
    steps.append(run_logged(prefix, 'venv', [str(interpreter), '-B', '-m', 'venv', str(prefix/'venv')], 120))
    python = prefix/'venv/bin/python'
    pip = [str(python), '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check',
           '--cache-dir', str(prefix/'cache/pip')]
    common = ['install', '--no-index', '--find-links', str(prefix/'stage/wheelhouse'), '--require-hashes']
    steps.append(run_logged(prefix, 'build-tools', pip + common + ['--only-binary=:all:', '-r', str(prefix/'stage/build_tools.lock.txt')]))
    steps.append(run_logged(prefix, 'cpu-runtime', pip + common + ['--no-build-isolation', '-r', str(prefix/'stage/offline_runtime.lock.txt')]))
    steps.append(run_logged(prefix, 'pip-check', pip + ['check']))
    after = snapshot_system()
    if after != json.loads((prefix/'system_before.json').read_text()):
        raise RuntimeError('System Python/packages/configuration differ from the saved baseline.')
    report = {'status': 'ISOLATED_PACKAGES_INSTALLED_NATIVE_PROBE_PENDING', 'prefix': str(prefix),
              'steps': steps, 'system_baseline_unchanged': True, 'model_executed': False}
    write_json(prefix/'installation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('initialize', 'bootstrap', 'install'))
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--archive-sha')
    parser.add_argument('--bundle-sha')
    args = parser.parse_args()
    if args.action == 'initialize':
        print('DIRECTORY:' + str(initialize(args.prefix)))
    elif args.action == 'bootstrap':
        if not args.archive_sha or not args.bundle_sha:
            parser.error('Bootstrap requires both expected checksums.')
        bootstrap(args.prefix, args.archive_sha, args.bundle_sha)
        print('BOOTSTRAP_VERIFIED')
    else:
        report = install(args.prefix)
        print('INSTALL_STATUS:' + report['status'])


if __name__ == '__main__':
    main()
