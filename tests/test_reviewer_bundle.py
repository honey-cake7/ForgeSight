"""Reviewer handoff must reject damaged/escaping bundles before installation."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts import reviewer_bundle as bundle


class ReviewerBundleTests(unittest.TestCase):
    def test_longest_plugin_socket_checked(self):
        bundle.validate_socket_prefix(Path('/home/reviewer/.local/share/forgesight/v1'))
        with self.assertRaisesRegex(ValueError, 'worker sockets'):
            bundle.validate_socket_prefix(Path('/' + 'x' * 89))

    def make_bundle(self, folder, name='app/hello.txt', body=b'hello', digest=None):
        path = Path(folder) / 'bundle.tar'
        index = {'schema_version': 1, 'files': {name: {'sha256': digest or hashlib.sha256(body).hexdigest(), 'bytes': len(body)}}}
        with tarfile.open(path, 'w') as archive:
            for key, data in [('bundle.json', json.dumps(index).encode()), (name, body)]:
                info = tarfile.TarInfo(key); info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return path

    def test_corrupt_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_bundle(folder, digest='0' * 64)
            with self.assertRaisesRegex(ValueError, 'hash'):
                bundle.verify_bundle(path, bundle.sha256(path))

    def test_escape_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ['../escape', '/absolute', 'C:/escape', 'app/../../escape']:
                path = self.make_bundle(folder, name=name)
                with self.assertRaises(ValueError):
                    bundle.verify_bundle(path, bundle.sha256(path))
            path = self.make_bundle(folder)
            with tarfile.open(path, 'a') as archive:
                entry = tarfile.TarInfo('app/hello.txt');entry.size=5
                archive.addfile(entry,io.BytesIO(b'hello'))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                bundle.verify_bundle(path,bundle.sha256(path))

    def test_wrong_whole_bundle_hash_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_bundle(folder)
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                bundle.verify_bundle(path, '0' * 64)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_bundle(folder)
            with tarfile.open(path, 'a') as archive:
                entry = tarfile.TarInfo('app/link');entry.type=tarfile.SYMTYPE;entry.linkname='/etc/passwd'
                archive.addfile(entry)
            with self.assertRaisesRegex(ValueError, 'regular files'):
                bundle.verify_bundle(path,bundle.sha256(path))

    def test_dataset_requires_exact_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data';app=root/'app'
            for part in ['metal_nut','screw']:
                image=data/part/'test/good/000.png';image.parent.mkdir(parents=True);image.write_bytes(b'original')
                manifest=app/'plugins'/part/'dataset.json';manifest.parent.mkdir(parents=True)
                manifest.write_text(json.dumps({'samples':[{'path':'test/good/000.png','sha256':hashlib.sha256(b'original').hexdigest(),'mask':None}]}))
            self.assertEqual(bundle.verify_dataset(app,data),2)
            image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Dataset hash mismatch'):
                bundle.verify_dataset(app,data)

    def test_valid_bundle_and_station_use_only_supplied_paths(self):
        with tempfile.TemporaryDirectory(prefix='reviewer space ') as folder:
            path = self.make_bundle(folder)
            self.assertEqual(len(bundle.verify_bundle(path,bundle.sha256(path))['files']),1)
            prefix=Path(folder)/'install';data=Path(folder)/'mvtec'
            station=bundle.station_config(prefix,data,18770)
            self.assertEqual(station['cwd'],str(prefix/'app'))
            command=station['command']
            for option,expected in [('--llama-root',prefix),('--vision-python',prefix/'vision-venv/bin/python'),('--dataset-root',data/'metal_nut'),('--runs-root',prefix/'runs')]:
                self.assertEqual(command[command.index(option)+1],str(expected))
            self.assertIn('--plugins-root',command)
            self.assertNotIn('/home/nvidia',json.dumps(station))

    def test_source_pack_excludes_credentials_and_generated_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ['forgesight/a.py','forgesight/__pycache__/a.pyc','AGENTS.md','.env','scripts/tool.py','models/secret.pte']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test')
            selected=[p.relative_to(root).as_posix() for p in bundle.source_files(root)]
            self.assertEqual(sorted(selected),['forgesight/a.py','scripts/tool.py'])

if __name__=='__main__':unittest.main()
