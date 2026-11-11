"""Guard tests for the ARM adapter; real generation is tested on the device."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from forgesight.edge_jetson_smoke import chat_prompt, fingerprint, validate_budget, verify_target


class JetsonSmokeGuards(unittest.TestCase):
    def test_chat_prompt_has_one_bos_and_no_user_control_markers(self):
        value = chat_prompt(' Explain stable lighting. ')
        self.assertEqual(value.count('<|begin_of_text|>'), 1)
        self.assertTrue(value.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n'))
        for text in ('', '<|eot_id|>', 'a'*4097):
            with self.subTest(text=text[:12]), self.assertRaises(ValueError):
                chat_prompt(text)

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


if __name__ == '__main__':
    unittest.main()
