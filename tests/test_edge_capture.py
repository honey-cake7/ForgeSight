"""Process/guard tests use generic children, not fake Llama output or proof."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from forgesight.edge_capture import capture_process, capture_run, file_sha256, require_aarch64_elf


class CaptureTests(unittest.TestCase):
    def test_hash_and_non_arm_executable_rejection(self):
        with tempfile.TemporaryDirectory(prefix="forgesight-capture-") as directory:
            path = Path(directory) / "not-a-runner"
            path.write_bytes(b"not an ELF executable")
            self.assertEqual(len(file_sha256(path)), 64)
            with self.assertRaisesRegex(ValueError, "AArch64 ELF"):
                require_aarch64_elf(path)

    def test_generic_process_stdout_is_captured_without_model_claim(self):
        with tempfile.TemporaryDirectory(prefix="forgesight-process-") as directory:
            output = Path(directory)
            result = capture_process([sys.executable, "-c", "print('generic process capture test')"], output, 5)
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["completion_reason"], "EXITED")
            self.assertEqual((output / "stdout.log").read_text().strip(), "generic process capture test")
            self.assertNotIn("stage_2_gate", result)

    def test_nonzero_exit_is_recorded(self):
        with tempfile.TemporaryDirectory(prefix="forgesight-process-") as directory:
            result = capture_process([sys.executable, "-c", "raise SystemExit(7)"], Path(directory), 5)
            self.assertEqual(result["returncode"], 7)

    def test_timeout_stops_a_real_generic_child(self):
        with tempfile.TemporaryDirectory(prefix="forgesight-process-") as directory:
            result = capture_process([sys.executable, "-c", "import time; time.sleep(10)"], Path(directory), 0.1)
            self.assertEqual(result["completion_reason"], "TIMEOUT")
            self.assertNotEqual(result["returncode"], 0)
            self.assertLess(result["process_wall_seconds"], 5)

    def test_invalid_timeout_is_rejected_without_logs(self):
        with tempfile.TemporaryDirectory(prefix="forgesight-process-") as directory:
            with self.assertRaises(ValueError):
                capture_process([sys.executable], Path(directory), 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_desktop_gate_blocks_before_any_runner_process(self):
        profile = json.loads(Path("configs/edge_llama.json").read_text())
        with tempfile.TemporaryDirectory(prefix="forgesight-blocked-") as directory:
            base = Path(directory)
            output = base / "attempt"
            with patch("forgesight.edge_capture.collect_preflight", return_value={"host_kind": "NOT_VERIFIED_JETSON_ORIN_NANO"}), \
                    patch("forgesight.edge_capture.capture_process") as process:
                report = capture_run(profile, base / "runner", base / "model.pte", base / "tokenizer.model",
                                     Path("configs/edge_smoke_prompt.txt"), base / "checkout", output)
            process.assert_not_called()
            self.assertEqual(report["status"], "BLOCKED_NOT_VERIFIED")
            self.assertEqual(report["stage_2_gate"], "NOT_VERIFIED")
            self.assertIsNone(report["process"])
            self.assertIsNone(report["raw_stdout_text"])
            self.assertEqual({path.name for path in output.iterdir()}, {"run.json"})
            with self.assertRaises(FileExistsError):
                capture_run(profile, base / "r", base / "m", base / "t", base / "p", base / "c", output)


if __name__ == "__main__":
    unittest.main()
