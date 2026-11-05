"""Preflight parsing/guard tests; synthetic host facts are not hardware proof."""

from pathlib import Path
import tempfile
import unittest

from forgesight.edge_preflight import identify_host, memory_kib, optional_text


class PreflightTests(unittest.TestCase):
    def test_desktop_and_incomplete_target_facts_do_not_pass(self):
        for system, arch, model, l4t in [
            ("Linux", "x86_64", None, None),
            ("Windows", "aarch64", "NVIDIA Jetson Orin Nano", "# R36"),
            ("Linux", "aarch64", "Raspberry Pi 5", None),
            ("Linux", "aarch64", "NVIDIA Jetson Orin Nano", None),
            ("Linux", "aarch64", "NVIDIA Jetson AGX Orin", "# R36"),
        ]:
            with self.subTest(system=system, arch=arch, model=model):
                self.assertEqual(identify_host(system, arch, model, l4t), "NOT_VERIFIED_JETSON_ORIN_NANO")

    def test_complete_identification_is_only_a_candidate_not_inference(self):
        self.assertEqual(identify_host("Linux", "aarch64", "NVIDIA Jetson Orin Nano Developer Kit",
                                       "# R36 (release), REVISION: 4.4"), "JETSON_ORIN_NANO_CANDIDATE")

    def test_memory_units_are_explicit_and_missing_fields_stay_unknown(self):
        facts = memory_kib("MemTotal: 8000000 kB\nMemAvailable: 4000000 kB\nSwapTotal: 0 kB\n")
        self.assertEqual(facts["MemTotal"], 8000000)
        self.assertEqual(facts["SwapTotal"], 0)
        self.assertIsNone(facts["SwapFree"])
        self.assertTrue(all(value is None for value in memory_kib(None).values()))

    def test_missing_device_files_record_errors_and_null_termination_is_removed(self):
        with tempfile.TemporaryDirectory(prefix="forgesight-preflight-") as directory:
            path = Path(directory) / "model"
            self.assertIsNone(optional_text(path)["value"])
            self.assertEqual(optional_text(path)["error"], "FileNotFoundError")
            path.write_bytes(b"NVIDIA Jetson Orin Nano\x00")
            self.assertEqual(optional_text(path)["value"], "NVIDIA Jetson Orin Nano")


if __name__ == "__main__":
    unittest.main()
