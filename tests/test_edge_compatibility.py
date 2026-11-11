"""Static compatibility guards. Synthetic ELF fixtures are not device execution proof."""

from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from forgesight.edge_compatibility import (REQUIRED_MEMBERS, assess, definitions,
                                         elf_machine, inspect, requirements)


class EdgeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.probes = [{'member': name, 'machine': 183, 'provided_versions': [],
                        'needed': ['libc.so.6'],
                        'required_versions': {'libc.so.6': ['GLIBC_2.28']}}
                       for name in REQUIRED_MEMBERS]
        self.target = {'machine': 'aarch64', 'libraries': {
            'libc.so.6': {'machine': 183, 'provided_versions': ['GLIBC_2.17', 'GLIBC_2.28', 'GLIBC_2.31']}}}

    def test_elf_architecture_is_read_from_bytes_not_filename(self):
        for machine in (62, 183):
            header = bytearray(20)
            header[:6] = b'\x7fELF\x02\x01'
            header[18:20] = machine.to_bytes(2, 'little')
            self.assertEqual(elf_machine(header), machine)
        for header in (b'', b'not an ELF header'.ljust(20, b' '), b'\x7fELF\x01\x01' + bytes(14)):
            with self.assertRaises(ValueError):
                elf_machine(header)

    def test_provided_versions_are_not_confused_with_required_versions(self):
        text = '''Version definition section '.gnu.version_d' contains 2 entries:
          Name: libsample.so
          Name: SAMPLE_1.0
        Version needs section '.gnu.version_r' contains 2 entries:
          File: libc.so.6 Cnt: 2
          Name: GLIBC_2.17 Flags: none
          Name: GLIBC_2.28 Flags: none
          File: libstdc++.so.6 Cnt: 1
          Name: CXXABI_1.3.11 Flags: none
        '''
        self.assertEqual(definitions(text), ['SAMPLE_1.0', 'libsample.so'])
        self.assertEqual(requirements(text), {'libc.so.6': ['GLIBC_2.17', 'GLIBC_2.28'],
                                              'libstdc++.so.6': ['CXXABI_1.3.11']})
        self.assertEqual(requirements('No version information found.'), {})

    def test_matching_static_checks_never_claim_execution(self):
        before = deepcopy(self.probes)
        result = assess(self.probes, self.target)
        self.assertEqual(result['status'], 'STATIC_CHECKS_MATCH_NOT_RUNTIME_PROOF')
        self.assertFalse(result['imports_verified'])
        self.assertFalse(result['llama_generation_verified'])
        self.assertEqual(self.probes, before)

    def test_missing_kernel_wrong_host_and_wrong_binary_are_refused(self):
        cases = [(self.probes[:-1], self.target)]
        wrong_binary = deepcopy(self.probes); wrong_binary[0]['machine'] = 62
        cases.append((wrong_binary, self.target))
        wrong_host = deepcopy(self.target); wrong_host['machine'] = 'x86_64'
        cases.append((self.probes, wrong_host))
        for probes, target in cases:
            with self.subTest(probes=probes, target=target):
                self.assertEqual(assess(probes, target)['status'], 'STATIC_CHECKS_FAILED')

    def test_version_must_be_provided_by_the_required_library(self):
        target = deepcopy(self.target)
        target['libraries']['libc.so.6']['provided_versions'] = ['GLIBC_2.17']
        target['libraries']['wrong_library.so'] = {'machine': 183, 'provided_versions': ['GLIBC_2.28']}
        self.assertTrue(assess(self.probes, target)['errors'])
        del target['libraries']['libc.so.6']
        self.assertTrue(assess(self.probes, target)['errors'])

    def test_cuda_dependency_is_not_accepted_for_cpu_reference_recipe(self):
        target = deepcopy(self.target)
        target['libraries']['libcudart.so'] = {'machine': 183, 'provided_versions': []}
        self.probes[0]['needed'].append('libcudart.so')
        self.assertTrue(any('Unexpected CUDA' in e for e in assess(self.probes, target)['errors']))

    def test_bad_wheel_hash_is_refused_without_altering_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); wheel = root/'fixture.whl'; wheel.write_bytes(b'changed archive')
            records = [{'label': 'fixture', 'filename': wheel.name, 'sha256': '0'*64}]
            with self.assertRaisesRegex(ValueError, 'checksum'):
                inspect(records, root, root/'probe', self.target)
            self.assertEqual(wheel.read_bytes(), b'changed archive')
            with self.assertRaises(FileExistsError):
                inspect(records, root, root/'probe', self.target)

    def test_wheel_member_cannot_escape_probe_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for i, name in enumerate(('../../escaped.so', 'C:/escaped.so', '/escaped.so', '..\\escaped.so')):
                wheel = root/('fixture{}.whl'.format(i))
                with zipfile.ZipFile(wheel, 'w') as archive:
                    archive.writestr(name, b'\x7fELF')
                records = [{'label': 'fixture', 'filename': wheel.name,
                            'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}]
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unsafe'):
                    inspect(records, root, root/('probe{}'.format(i)), self.target)
                self.assertFalse((root/'escaped.so').exists())


if __name__ == '__main__':
    unittest.main()
