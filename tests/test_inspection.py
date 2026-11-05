import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from forgesight.inspection import CaptureGate, inspect_one_image


class CaptureGateTests(unittest.TestCase):
    def test_repeat_frames_and_distinct_parts_with_identical_pixels(self):
        gate, calls = CaptureGate(), []
        def inspect(path, part):
            calls.append((path, part))
            return part
        with self.assertRaises(ValueError):
            gate.on_frame(Path('same.png'), inspect)
        for part in ['one', 'two']:
            gate.arm(part)
            self.assertEqual(gate.on_frame(Path('same.png'), inspect), part)
            self.assertIsNone(gate.on_frame(Path('different_lighting.png'), inspect))
        self.assertEqual(len(calls), 2)
        self.assertEqual(gate.suppressed_frames, 2)
        with self.assertRaises(ValueError):
            gate.arm('one')

    def test_no_silent_drop_or_repeated_failed_attempt(self):
        gate = CaptureGate()
        gate.arm('one')
        with self.assertRaises(ValueError):
            gate.arm('two')
        def broken(*args):
            raise OSError('output unavailable')
        with self.assertRaises(OSError):
            gate.on_frame(Path('bad'), broken)
        self.assertIsNone(gate.on_frame(Path('bad'), broken))

    def test_invalid_image_is_explicit_saved_error(self):
        # Metadata-only test double; no synthetic detector success is asserted.
        predictor = SimpleNamespace(reference={'calibrated_checkpoint': {'sha256': 'test'},
                                              'policy_file_sha256': 'test'},
                                    policy={'threshold': 0.2}, device='test')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            record = inspect_one_image(root/'missing.png', 'bad', predictor, root/'out')
            self.assertEqual(record.status, 'ERROR')
            self.assertIsNone(record.score)
            self.assertEqual(record.error['type'], 'FileNotFoundError')
            self.assertTrue((root/'out'/f'{record.inspection_id}.json').is_file())


if __name__ == '__main__':
    unittest.main()
