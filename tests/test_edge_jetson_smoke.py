"""Guard tests for the ARM adapter; real generation is tested on the device."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from forgesight import edge_jetson_smoke as smoke
from forgesight.edge_jetson_smoke import chat_prompt, fingerprint, validate_budget, verify_target


class JetsonSmokeGuards(unittest.TestCase):
    def test_chat_prompt_has_one_bos_and_no_user_control_markers(self):
        value = chat_prompt(' Explain stable lighting. ')
        self.assertEqual(value.count('<|begin_of_text|>'), 1)
        self.assertTrue(value.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n'))
        for text in ('', '<|eot_id|>', 'a'*4097):
            with self.subTest(text=text[:12]), self.assertRaises(ValueError):
                chat_prompt(text)

    def test_optional_system_role_has_separate_header_and_rejects_control_markers(self):
        value = chat_prompt('Use [E1].', 'Cite the supplied evidence.')
        self.assertEqual(value.count('<|begin_of_text|>'), 1)
        self.assertEqual(value.count('<|start_header_id|>'), 3)
        self.assertEqual(value.count('<|eot_id|>'), 2)
        self.assertIn('<|start_header_id|>system<|end_header_id|>', value)
        for text in ('', '<|eot_id|>', 'a'*2049):
            with self.subTest(text=text[:12]), self.assertRaises(ValueError):
                chat_prompt('Use [E1].', text)

    def test_token_limits_follow_export_bounds(self):
        profile = {'max_seq_length': 128, 'max_context_length': 512}
        validate_budget(33, 64, profile)
        for prompt, answer in ((0, 64), (128, 1), (33, 65)):
            with self.subTest(prompt=prompt, answer=answer), self.assertRaises(ValueError):
                validate_budget(prompt, answer, profile)

    def test_hash_mismatch_is_refused_before_model_import(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'model.pte'
            path.write_bytes(b'checked bytes')
            expected = {'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(fingerprint(path, expected), expected)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                fingerprint(path, {**expected, 'sha256': '0'*64})
            with self.assertRaisesRegex(ValueError, 'wrong-sized'):
                fingerprint(path, {**expected, 'bytes': expected['bytes']+1})

    def test_desktop_is_not_accepted_as_jetson(self):
        with self.assertRaisesRegex(ValueError, 'Jetson Orin Nano'):
            verify_target(Path('/tmp/does-not-exist'))

    def verified_runtime_root(self, directory):
        root = Path(directory)
        model_dir = root / 'model'
        model_dir.mkdir(parents=True)
        model = model_dir / 'llama3_2_1b_xnnpack.pte'
        model.write_bytes(b'checked model bytes')
        tokenizer = model_dir / 'tokenizer.model'
        tokenizer.write_bytes(b'checked tokenizer bytes')
        profile_path = model_dir / 'edge_llama.json'
        assets_path = model_dir / 'llama3_2_1b_assets.json'
        profile = {'model_id': 'llama3_2_1b', 'model_revision': 'rev-1',
                   'executorch_commit': 'commit-1', 'backend': 'xnnpack',
                   'max_seq_length': 128, 'max_context_length': 512}
        profile_path.write_text(json.dumps(profile))
        assets = {'model_id': 'llama3_2_1b', 'model_revision': 'rev-1',
                  'executorch_commit': 'commit-1',
                  'files': {'original/tokenizer.model': {
                      'bytes': tokenizer.stat().st_size,
                      'sha256': smoke.sha256(tokenizer)}}}
        assets_path.write_text(json.dumps(assets))
        export = {'model_id': 'llama3_2_1b', 'model_revision': 'rev-1',
                  'executorch_commit': 'commit-1', 'backend_requested': 'xnnpack',
                  'quantization': {'qmode': '8da4w'},
                  'inputs_sha256': {
                      'configs/edge_llama.json': smoke.sha256(profile_path),
                      'manifests/llama3_2_1b_assets.json': smoke.sha256(assets_path)},
                  'artifact': {'bytes': model.stat().st_size,
                               'sha256': smoke.sha256(model)}}
        export_path = model_dir / 'llama3_2_1b_export.json'
        export_path.write_text(json.dumps(export))
        return root, {'model': model, 'tokenizer': tokenizer, 'profile': profile_path,
                      'export': export_path, 'assets': assets_path}

    def test_verify_inputs_does_not_require_a_prompt_file(self):
        # The persistent advisory worker verifies device and assets once, before
        # any per-request prompt exists; verification must not demand one.
        with TemporaryDirectory() as directory:
            root, files = self.verified_runtime_root(directory)
            request = {'root': str(root), 'model': str(files['model']),
                       'tokenizer': str(files['tokenizer']), 'profile': str(files['profile']),
                       'export': str(files['export']), 'assets': str(files['assets'])}
            with mock.patch.object(smoke, 'verify_target', return_value={'machine': 'test'}):
                _host, _profile, resolved, _model_fp, _tokenizer_fp = smoke.verify_inputs(request)
            self.assertNotIn('prompt', resolved)

    def test_verify_inputs_still_resolves_prompt_when_supplied(self):
        with TemporaryDirectory() as directory:
            root, files = self.verified_runtime_root(directory)
            prompt = root / 'prompt.txt'
            prompt.write_text('Explain stable lighting.')
            request = {'root': str(root), 'model': str(files['model']),
                       'tokenizer': str(files['tokenizer']), 'profile': str(files['profile']),
                       'export': str(files['export']), 'assets': str(files['assets']),
                       'prompt': str(prompt)}
            with mock.patch.object(smoke, 'verify_target', return_value={'machine': 'test'}):
                _host, _profile, resolved, _model_fp, _tokenizer_fp = smoke.verify_inputs(request)
            self.assertEqual(resolved['prompt'], prompt.resolve())


if __name__ == '__main__':
    unittest.main()
