"""Prefix and extraction safeguards; fixtures do not establish Jetson runtime proof."""

import hashlib
import io
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from forgesight.edge_runtime_setup import (extract_checked, initialize, isolated_environment,
                                          make_offline_lock, validate_member, validate_prefix)


class EdgeRuntimeSetupTests(unittest.TestCase):
    def test_prefix_must_be_one_user_owned_child_at_the_planned_location(self):
        with TemporaryDirectory() as directory:
            home = Path(directory)
            valid = home/'.local/share/forgesight/llama_cpu_v1'
            self.assertEqual(validate_prefix(valid, home), valid.resolve())
            for path in (home, home/'venv', home/'.local/share/forgesight',
                         home/'.local/share/forgesight/a/b', home.parent/'outside'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    validate_prefix(path, home)

    def test_prefix_symlinks_and_existing_directory_are_preserved(self):
        with TemporaryDirectory() as directory:
            home = Path(directory); parent = home/'.local/share/forgesight'; parent.mkdir(parents=True)
            prefix = parent/'runtime'; prefix.mkdir(); marker = prefix/'existing.txt'; marker.write_text('keep')
            with patch('forgesight.edge_runtime_setup.Path.home', return_value=home):
                with self.assertRaises(FileExistsError): initialize(prefix)
            self.assertEqual(marker.read_text(), 'keep')
            alias = parent/'alias'; alias.symlink_to(prefix, target_is_directory=True)
            with self.assertRaises(ValueError): validate_prefix(alias, home)

    def test_paths_links_and_special_files_cannot_escape_destination(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('../outside', '/outside', 'C:/outside', '..\\outside'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    validate_member(tarfile.TarInfo(name), root, 'python')
            for target in ('../../outside', '/outside', 'C:/outside'):
                member = tarfile.TarInfo('python/bin/link'); member.type = tarfile.SYMTYPE; member.linkname = target
                with self.subTest(target=target), self.assertRaises(ValueError): validate_member(member, root, 'python')
            fifo = tarfile.TarInfo('python/fifo'); fifo.type = tarfile.FIFOTYPE
            with self.assertRaises(ValueError): validate_member(fifo, root, 'python')
            good = tarfile.TarInfo('python/bin/python3'); good.type = tarfile.SYMTYPE; good.linkname = 'python3.12'
            validate_member(good, root, 'python')

    def test_all_members_are_checked_before_any_extraction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); destination = root/'dest'; destination.mkdir()
            archive = root/'input.tar'
            with tarfile.open(archive, 'w') as stream:
                first = tarfile.TarInfo('python/first'); first.size = 4
                stream.addfile(first, io.BytesIO(b'data'))
                stream.addfile(tarfile.TarInfo('../escape'))
            expected = hashlib.sha256(archive.read_bytes()).hexdigest()
            with self.assertRaises(ValueError): extract_checked(archive, destination, expected, 'python')
            self.assertEqual(list(destination.iterdir()), [])

    def test_checksum_failure_and_preexisting_parent_symlink_are_refused(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); destination = root/'dest'; destination.mkdir(); outside = root/'outside'; outside.mkdir()
            archive = root/'input.tar'
            with tarfile.open(archive, 'w') as stream: stream.addfile(tarfile.TarInfo('python/item'))
            with self.assertRaisesRegex(ValueError, 'checksum'): extract_checked(archive, destination, '0'*64, 'python')
            (destination/'python').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError): validate_member(tarfile.TarInfo('python/item'), destination, 'python')
            self.assertEqual(list(outside.iterdir()), [])

    def test_child_environment_excludes_system_python_injection_and_uses_private_caches(self):
        with patch.dict('os.environ', {'PYTHONPATH': '/system/old', 'PYTHONHOME': '/system/old', 'PIP_EXTRA_INDEX_URL': 'untrusted'}):
            result = isolated_environment(Path('/runtime'))
        self.assertNotIn('PYTHONPATH', result)
        self.assertNotIn('PYTHONHOME', result)
        self.assertNotIn('PIP_EXTRA_INDEX_URL', result)
        self.assertEqual(result['PYTHONNOUSERSITE'], '1')
        self.assertEqual(result['XDG_CACHE_HOME'], '/runtime/cache')

    def test_offline_lock_keeps_hashes_and_refuses_unknown_url_dependencies(self):
        text = 'torch @ https://example.test/torch.whl ' + chr(92) + '\n --hash=sha256:abc\n'
        text += 'executorch @ https://example.test/et.whl ' + chr(92) + '\n --hash=sha256:def\n'
        result = make_offline_lock(text)
        self.assertIn('torch==2.14.0+cpu ' + chr(92), result)
        self.assertIn('executorch==1.5.1+cpu ' + chr(92), result)
        self.assertEqual(result.count('--hash='), 2)
        self.assertIn('sha256:abc', result)
        with self.assertRaises(ValueError): make_offline_lock(text + 'other @ https://example.test/other.whl\n')
        with self.assertRaises(ValueError): make_offline_lock('torch==2.14.0+cpu\n')


if __name__ == '__main__':
    unittest.main()
