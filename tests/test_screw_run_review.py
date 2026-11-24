"""Post-training preparation must not execute or disturb the running job."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from forgesight.screw_run_review import load_protocol, review_status, postprocess_commands
from forgesight.screw_evaluate import verify_export


class ScrewRunReviewTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(Path("configs/screw_30k_review_v2.json"))

    def input_hashes(self):
        paths = [self.protocol[k] for k in ("training_recipe", "model_profile", "dataset_manifest")]
        paths.append("forgesight/vision_train.py")
        return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}

    def test_outputs_are_isolated_and_commands_use_cpu(self):
        commands = postprocess_commands(self.protocol)
        self.assertEqual(len(commands), 4)
        for command in commands[:2]:
            self.assertEqual(command[command.index("--device") + 1], "cpu")
        for key in ("reload_report", "calibration_directory", "export_model", "export_parity"):
            self.assertTrue(self.protocol["paths"][key].startswith("artifacts/vision/screw_30k_v2/"))
        self.assertNotIn("--allow-final-holdout", str(commands))

    def test_incomplete_or_partial_progress_is_never_ready(self):
        with tempfile.TemporaryDirectory(dir="artifacts") as directory:
            progress = Path(directory) / "progress.json"
            report = Path(directory) / "training_report.json"
            protocol = deepcopy(self.protocol)
            protocol["paths"].update(progress=str(progress), training_report=str(report),
                                     run_config=str(Path(directory) / "run_config.json"))
            progress.write_text(json.dumps({"step": 1500, "target": 30000}))
            self.assertEqual(review_status(protocol)["training_status"], "TRAINING_IN_PROGRESS")
            self.assertFalse(review_status(protocol)["postprocessing_ready"])
            progress.write_text('{"step":')
            self.assertFalse(review_status(protocol)["postprocessing_ready"])

    def test_only_complete_verified_budget_report_is_ready(self):
        with tempfile.TemporaryDirectory(dir="artifacts") as directory:
            report = Path(directory) / "training_report.json"
            protocol = deepcopy(self.protocol)
            run_config = Path(directory) / "run_config.json"
            protocol["paths"].update(training_report=str(report),
                                     run_config=str(run_config))
            run_config.write_text(json.dumps({"input_sha256": self.input_hashes()}))
            saved = {"status": "TRAINED_UNCALIBRATED_NOT_ACCURACY_VERIFIED", "completed_steps": 30000,
                     "training_config": json.loads(Path(protocol["training_recipe"]).read_text()),
                     "input_sha256": self.input_hashes(),
                     "checkpoints": [{"step": n} for n in (10000, 20000, 30000)]}
            report.write_text(json.dumps(saved))
            result = review_status(protocol)
            self.assertTrue(result["postprocessing_ready"])
            self.assertEqual(result["acceptance_status"], "BLOCKED_MISSING_FRESH_ACCEPTANCE_DATA")
            saved["completed_steps"] = 10000
            report.write_text(json.dumps(saved))
            self.assertFalse(review_status(protocol)["postprocessing_ready"])

    def test_exposed_holdout_and_canonical_output_paths_are_rejected(self):
        with tempfile.TemporaryDirectory(dir="artifacts") as directory:
            for field, value in (("development_manifest", "manifests/screw_v1.json"),
                                  ("acceptance_manifest", "manifests/screw_v1.json")):
                changed = deepcopy(self.protocol)
                changed[field] = value
                path = Path(directory) / "protocol.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_protocol(path)
            changed = deepcopy(self.protocol)
            changed["paths"]["export_model"] = "plugins/screw/efficientad_s.ts"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                load_protocol(path)

    def test_custom_export_parity_never_overwrites_existing_evidence(self):
        with tempfile.TemporaryDirectory(dir="artifacts") as directory:
            existing = Path(directory) / "parity.json"
            existing.write_text("preserve")
            with self.assertRaises(FileExistsError):
                verify_export(parity_path=existing)
            self.assertEqual(existing.read_text(), "preserve")

    def test_changed_training_input_blocks_postprocessing(self):
        with tempfile.TemporaryDirectory(dir="artifacts") as directory:
            run_config = Path(directory) / "run_config.json"
            run_config.write_text(json.dumps({"input_sha256": {
                self.protocol["training_recipe"]: "0" * 64}}))
            protocol = deepcopy(self.protocol)
            protocol["paths"]["run_config"] = str(run_config)
            status = review_status(protocol)
            self.assertEqual(status["training_status"], "BLOCKED_TRAINING_INPUT_CHANGED")
            self.assertFalse(status["postprocessing_ready"])

    def test_missing_malformed_or_disagreeing_hash_evidence_cannot_be_ready(self):
        with tempfile.TemporaryDirectory(dir="artifacts") as directory:
            report = Path(directory) / "training_report.json"
            run_config = Path(directory) / "run_config.json"
            protocol = deepcopy(self.protocol)
            protocol["paths"].update(training_report=str(report), run_config=str(run_config))
            saved = {"status": "TRAINED_UNCALIBRATED_NOT_ACCURACY_VERIFIED", "completed_steps": 30000,
                     "training_config": json.loads(Path(protocol["training_recipe"]).read_text()),
                     "input_sha256": self.input_hashes(),
                     "checkpoints": [{"step": n} for n in (10000, 20000, 30000)]}
            report.write_text(json.dumps(saved))
            self.assertFalse(review_status(protocol)["postprocessing_ready"])
            for content in ('{"input_sha256":', '{}', '{"input_sha256": {}}'):
                run_config.write_text(content)
                self.assertFalse(review_status(protocol)["postprocessing_ready"])
            run_config.write_text(json.dumps({"input_sha256": self.input_hashes()}))
            saved["input_sha256"].pop(protocol["training_recipe"])
            report.write_text(json.dumps(saved))
            self.assertFalse(review_status(protocol)["postprocessing_ready"])


if __name__ == "__main__":
    unittest.main()
