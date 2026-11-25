#!/usr/bin/env python3
"""Build, verify and install the offline reviewer application on the recorded Jetson baseline.

Uses system Python's standard library; package installation targets fresh private venvs.
Run --help for the pack, verify, install and check commands.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ('llama3_2_1b_xnnpack.pte', 'tokenizer.model', 'edge_llama.json',
               'llama3_2_1b_export.json', 'llama3_2_1b_assets.json')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_files(root):
    folders = ('forgesight', 'scripts', 'tests', 'configs', 'guidance', 'plugins',
               'manifests', 'requirements', 'docs', 'assets')
    suffixes = {'.py', '.json', '.toml', '.txt', '.md', '.js', '.css', '.html', '.svg', '.png', '.sha256'}
    for folder in folders:
        for path in sorted((root / folder).rglob('*')):
            if (path.is_file() and not path.is_symlink() and path.suffix in suffixes
                    and '__pycache__' not in path.parts and path.name != 'record_timings.py'):
                yield path
    for name in ('README.md', 'requirements.txt', '.gitattributes', '.gitignore'):
        if (root / name).is_file():
            yield root / name


def safe_name(name):
    value = PurePosixPath(name)
    if (not name or value.is_absolute() or '..' in value.parts or '\\' in name
            or re.match(r'^[A-Za-z]:', name) or str(value) != name):
        raise ValueError('Unsafe archive path: ' + name)
    return value


def verify_bundle(path, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected) or sha256(path) != expected:
        raise ValueError('Bundle SHA-256 does not match the supplied release checksum.')
    with tarfile.open(path, 'r:') as archive:
        members = archive.getmembers()
        names = set()
        for member in members:
            safe_name(member.name)
            if member.name in names:
                raise ValueError('Duplicate bundle member: ' + member.name)
            names.add(member.name)
            if not member.isfile() or member.size > 4 * 1024**3:
                raise ValueError('Only bounded regular files are accepted.')
        index_member = archive.getmember('bundle.json')
        if index_member.size > 4 * 1024**2:
            raise ValueError('Oversized bundle index.')
        index = json.load(archive.extractfile(index_member))
        if index.get('schema_version') != 1 or names != set(index['files']) | {'bundle.json'}:
            raise ValueError('Bundle index and file list differ.')
        for member in members:
            if member.name == 'bundle.json':
                continue
            reference = index['files'][member.name]
            value = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(block)
            if member.size != reference['bytes'] or value.hexdigest() != reference['sha256']:
                raise ValueError('File hash/size mismatch: ' + member.name)
    return index


def station_config(prefix, dataset, port, camera=0):
    prefix, dataset = Path(prefix), Path(dataset)
    app = prefix / 'app'
    return {'schema_version': 1, 'port': port, 'cwd': str(app), 'command': [
        '/usr/bin/python3', '-B', '-m', 'forgesight.jetson_live_ui',
        '--vision-python', str(prefix / 'vision-venv/bin/python'),
        '--model', str(app / 'plugins/metal_nut/efficientad_s.ts'),
        '--plugins-root', str(app / 'plugins'), '--grounded-auto-advice',
        '--defect-type-profile', str(app / 'configs/defect_type_mvtec_four_v1.json'),
        '--dataset-root', str(dataset / 'metal_nut'), '--llama-root', str(prefix),
        '--runs-root', str(prefix / 'runs'), '--database', str(prefix / 'runs/inspections.sqlite3'),
        '--advice-root', str(prefix / 'runs/advice'),
        '--vision-worker-socket', str(prefix / 'vision.sock'),
        '--llama-worker-socket', str(prefix / 'llama.sock'),
        '--camera-index', str(camera), '--port', str(port)]}


def validate_socket_prefix(prefix):
    if len(str(prefix / 'vision-screw.sock').encode()) >= 104:
        raise ValueError('Prefix is too long for the part-recipe Unix worker sockets.')


def pack(args):
    root = args.source.resolve()
    files = {'app/' + p.relative_to(root).as_posix(): p for p in source_files(root)}
    for part in ('metal_nut', 'screw'):
        path = root / 'plugins' / part / 'efficientad_s.ts'
        recipe = json.loads((path.parent / 'recipe.json').read_text())
        if sha256(path) != recipe['model']['sha256']:
            raise ValueError('Recipe model hash differs: ' + part)
        files['app/plugins/' + part + '/efficientad_s.ts'] = path
    files['app/artifacts/vision/calibration_v1/efficientad_s_calibrated.pth'] = args.checkpoint
    for name in ('score_policy.json', 'calibration_report.json'):
        files['app/artifacts/vision/calibration_v1/' + name] = args.checkpoint.parent / name
    files['stage/python.tar.gz'] = args.runtime / 'stage/python.tar.gz'
    files['stage/inputs.tar'] = args.runtime / 'stage/inputs.tar'
    for name in MODEL_FILES:
        files['model/' + name] = args.runtime / 'model' / name
    files['model/edge_jetson_smoke.py'] = root / 'forgesight/edge_jetson_smoke.py'
    for name in ('LICENSE.txt', 'USE_POLICY.md'):
        files['model/' + name] = args.model_terms / name
    files['model/Notice.txt'] = root / 'assets/llama/Notice.txt'
    export = json.loads(files['model/llama3_2_1b_export.json'].read_text())
    # The application performs the export/profile/assets chain checks at startup.
    index = {'schema_version': 1, 'target': 'Jetson Orin Nano 8GB; Ubuntu 20.04.6 / L4T 35.6.1; system Python 3.8.10',
             'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
             'source_scope': 'Working-tree source bytes (including local submission changes), individually hashed; not a claim that HEAD contains all bundled changes.',
             'dataset_policy': 'MVTec images are not bundled; obtain the original metal_nut and screw datasets under publisher terms.',
             'model_id': export.get('model_id', 'meta-llama/Llama-3.2-1B-Instruct'),
             'files': {name: {'sha256': sha256(path), 'bytes': path.stat().st_size} for name, path in sorted(files.items())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a release bundle.')
    with tarfile.open(args.output, 'x:') as archive:
        data = (json.dumps(index, indent=2) + '\n').encode()
        member = tarfile.TarInfo('bundle.json'); member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
        for name, path in sorted(files.items()):
            member = tarfile.TarInfo(name); member.size = path.stat().st_size; member.mode = 0o644
            with path.open('rb') as stream:
                archive.addfile(member, stream)
    checksum = sha256(args.output)
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(checksum + '  ' + args.output.name + '\n')
    args.output.with_suffix(args.output.suffix + '.manifest.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'bundle': str(args.output), 'sha256': checksum, 'files': len(files), 'bytes': args.output.stat().st_size}))


def preflight():
    subprocess.run(['/usr/bin/python3', '-B', '-c', 'import cv2,numpy,PIL'], check=True)
    if not any(shutil.which(name) for name in ('chromium', 'chromium-browser', 'google-chrome', 'microsoft-edge')):
        subprocess.run(['/usr/bin/python3', '-B', '-c',
                        "import gi;gi.require_version('Gtk','3.0');gi.require_version('WebKit2','4.0');from gi.repository import Gtk,WebKit2"], check=True)


def verify_dataset(app, dataset):
    count = 0
    for part in ('metal_nut', 'screw'):
        manifest = json.loads((app / 'plugins' / part / 'dataset.json').read_text())
        for sample in manifest['samples']:
            for item in (sample, sample.get('mask')):
                if item is None:
                    continue
                relative = safe_name(item['path'])
                if sha256(dataset / part / str(relative)) != item['sha256']:
                    raise ValueError('Dataset hash mismatch: ' + part + '/' + str(relative))
                count += 1
    return count


def install(args):
    sys.path.insert(0, str(ROOT))
    from forgesight import edge_runtime_setup as runtime
    prefix = runtime.validate_prefix(args.prefix, Path.home())
    if prefix.exists():
        raise FileExistsError('Choose a fresh --prefix; existing installations and history are preserved.')
    validate_socket_prefix(prefix)
    dataset = args.dataset_root.resolve()
    for part in ('metal_nut', 'screw'):
        if not (dataset / part / 'test').is_dir() or not (dataset / part / 'train').is_dir():
            raise ValueError('Expected original train/test directories for ' + part)
    preflight()
    # Refuse occupied ports and existing user shortcuts before making a prefix.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    if not args.no_shortcuts:
        data_home = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share')))
        if (data_home / 'applications/forgesight.desktop').exists():
            raise FileExistsError('A ForgeSight shortcut already exists; use --no-shortcuts to preserve it.')
        desktop = args.desktop_dir
        if desktop is None:
            desktop = Path.home() / 'Desktop'
            if shutil.which('xdg-user-dir'):
                value = subprocess.check_output(['xdg-user-dir', 'DESKTOP'], text=True).strip()
                if value:
                    desktop = Path(value)
        if not desktop.is_dir() or desktop.resolve() == Path.home().resolve():
            raise ValueError('An existing desktop directory is required; use --desktop-dir or --no-shortcuts.')
        if (desktop / 'ForgeSight.desktop').exists():
            raise FileExistsError('A desktop shortcut already exists; use --no-shortcuts to preserve it.')
        if (data_home / 'icons/forgesight.svg').exists():
            raise FileExistsError('A ForgeSight icon already exists; use --no-shortcuts to preserve it.')
    index = verify_bundle(args.bundle, args.sha256)
    required = {'stage/python.tar.gz', 'stage/inputs.tar', 'model/edge_jetson_smoke.py',
                'model/Notice.txt', 'model/LICENSE.txt', 'model/USE_POLICY.md',
                'app/plugins/metal_nut/efficientad_s.ts', 'app/plugins/screw/efficientad_s.ts'} | {'model/' + n for n in MODEL_FILES}
    if not required <= set(index['files']):
        raise ValueError('Incomplete application runtime bundle.')
    for relative in ('scripts/reviewer_bundle.py', 'forgesight/edge_runtime_setup.py',
                     'plugins/metal_nut/dataset.json', 'plugins/screw/dataset.json'):
        if sha256(ROOT / relative) != index['files']['app/' + relative]['sha256']:
            raise ValueError('Use the source app/ extracted from this exact bundle: ' + relative)
    verify_dataset(ROOT, dataset)
    runtime.initialize(prefix)
    # No extractall: verified files are created exclusively beneath the fresh prefix.
    with tarfile.open(args.bundle, 'r:') as archive:
        for name in index['files']:
            destination = prefix / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(name) as source, destination.open('xb') as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
    (prefix / 'bundle.json').write_text(json.dumps(index, indent=2) + '\n')
    runtime.bootstrap(prefix, index['files']['stage/python.tar.gz']['sha256'], index['files']['stage/inputs.tar']['sha256'])
    runtime.install(prefix)
    python = prefix / 'python/bin/python3.12'
    subprocess.run([str(python), '-B', '-m', 'venv', str(prefix / 'vision-venv')], check=True)
    env = runtime.isolated_environment(prefix)
    vision_python = prefix / 'vision-venv/bin/python'
    # These two wheels are among the individually hash-verified staged inputs.
    subprocess.run([str(vision_python), '-B', '-m', 'pip', '--isolated', 'install', '--no-index',
                    '--find-links', str(prefix / 'stage/wheelhouse'), 'torch==2.14.0+cpu', 'numpy==2.5.4'], env=env, check=True)
    subprocess.run([str(vision_python), '-B', '-m', 'pip', '--isolated', 'check'], env=env, check=True)
    data_link = prefix / 'app/data/mvtec_ad'
    data_link.parent.mkdir(parents=True, exist_ok=True)
    data_link.symlink_to(dataset, target_is_directory=True)
    config = station_config(prefix, dataset, args.port, args.camera_index)
    station = prefix / 'station.json'; station.write_text(json.dumps(config, indent=2) + '\n')
    check(prefix)
    if not args.no_shortcuts:
        icons = data_home / 'icons'; icons.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prefix / 'app/forgesight/web/icon.svg', icons / 'forgesight.svg')
        launcher = prefix / 'app/scripts/jetson_desktop_launcher.py'
        command = ['/usr/bin/python3', '-B', str(launcher), '--station', str(station), '--install-icon']
        if args.desktop_dir:
            command += ['--desktop-dir', str(args.desktop_dir.resolve())]
        subprocess.run(command, check=True)
    if runtime.snapshot_system() != json.loads((prefix / 'system_before.json').read_text()):
        raise RuntimeError('System baseline changed during installation.')
    report = {'status': 'INSTALLED_IMPORTS_AND_ASSETS_VERIFIED', 'prefix': str(prefix),
              'bundle_sha256': args.sha256, 'dataset_root': str(dataset), 'station': str(station),
              'system_baseline_unchanged': True, 'application_startup': 'NOT_YET_TESTED',
              'desktop_human_click': 'NOT_YET_TESTED'}
    (prefix / 'reviewer_installation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


def check(prefix):
    prefix = prefix.resolve()
    index = json.loads((prefix / 'bundle.json').read_text())
    for name, info in index['files'].items():
        if sha256(prefix / name) != info['sha256']:
            raise ValueError('Installed file hash mismatch: ' + name)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
    app = prefix / 'app'
    subprocess.run([str(prefix / 'vision-venv/bin/python'), '-B', '-c',
                    'import torch,numpy;assert torch.__version__=="2.14.0+cpu";assert numpy.__version__=="2.5.4"'], cwd=app, env=env, check=True)
    probe = subprocess.run([str(prefix / 'venv/bin/python'), '-B', '-c',
                    'import json;from pathlib import Path;from forgesight.assist_edge_worker import build_jetson_runtime,Worker,warm_worker;'
                    'print("REVIEWER_LLAMA_PROBE="+json.dumps(warm_worker(Worker(build_jetson_runtime(Path('+repr(str(prefix))+'))))))'],
                    cwd=app, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=240)
    (prefix / 'logs/reviewer-native-check.stdout.log').write_text(probe.stdout)
    (prefix / 'logs/reviewer-native-check.stderr.log').write_text(probe.stderr)
    probe.check_returncode()
    evidence = [line.split('=', 1)[1] for line in probe.stdout.splitlines() if line.startswith('REVIEWER_LLAMA_PROBE=')]
    if len(evidence) != 1:
        raise ValueError('Native generation probe did not report evidence; inspect logs/.')
    (prefix / 'runtime_probe.json').write_text(evidence[0] + '\n')
    print('Application files, vision imports and bounded native Llama generation verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    builder = commands.add_parser('pack')
    builder.add_argument('--source', type=Path, default=ROOT)
    builder.add_argument('--runtime', type=Path, required=True, help='Authorized staging directory containing stage/ and model/.')
    builder.add_argument('--model-terms', type=Path, required=True)
    builder.add_argument('--checkpoint', type=Path, required=True)
    builder.add_argument('--output', type=Path, required=True)
    verifier = commands.add_parser('verify')
    installer = commands.add_parser('install')
    for sub in (verifier, installer):
        sub.add_argument('--bundle', type=Path, required=True)
        sub.add_argument('--sha256', required=True)
    installer.add_argument('--prefix', type=Path, default=Path.home()/'.local/share/forgesight/reviewer_v1')
    installer.add_argument('--dataset-root', type=Path, required=True, help='Directory containing original metal_nut/ and screw/.')
    installer.add_argument('--port', type=int, default=8770)
    installer.add_argument('--camera-index', type=int, default=0)
    installer.add_argument('--no-shortcuts', action='store_true')
    installer.add_argument('--desktop-dir', type=Path)
    checker = commands.add_parser('check'); checker.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'pack': pack(args)
        elif args.action == 'verify':
            index = verify_bundle(args.bundle, args.sha256)
            print('Verified {} files.'.format(len(index['files'])))
        elif args.action == 'install':
            if not 1024 <= args.port <= 65535: raise ValueError('Use an unprivileged port between 1024 and 65535.')
            install(args)
        else: check(args.prefix)
    except (OSError, ValueError, KeyError, tarfile.TarError, subprocess.SubprocessError) as error:
        print('ForgeSight setup: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
