"""Frozen screw calibration and evaluation gates; no simulated accuracy claims."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from forgesight.vision_calibrate import calibration_split, tensor_for
from forgesight.vision_recipe import read_manifest
from forgesight.screw_evaluate import acceptance_gate, holdout_rows, claim_holdout


class ScrewEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.document = read_manifest(Path("manifests/screw_v1.json"))
        self.policy = json.loads(Path("configs/vision_score_policy.json").read_text())
        self.policy.update(category="screw", map_fit_count=64,
                           threshold_role="threshold_normal", threshold_count=20)

    def test_calibration_uses_only_frozen_normal_roles(self):
        maps, threshold = calibration_split(self.document, self.policy)
        self.assertEqual((len(maps), len(threshold)), (64, 20))
        self.assertTrue(all(r["path"].startswith("train/good/") for r in maps))
        self.assertTrue(all(r["path"].startswith("test/good/") for r in threshold))
        reserved = {r["path"] for r in maps + threshold}
        self.assertFalse(reserved & {r["path"] for r in holdout_rows(self.document)})

    def test_calibration_cannot_reassign_holdout_or_use_wrong_policy(self):
        for key, value in (("threshold_role", "final_holdout"), ("map_fit_count", 32),
                           ("category", "metal_nut"), ("final_holdout_access", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                calibration_split(self.document, dict(self.policy, **{key: value}))
        changed = deepcopy(self.document)
        next(r for r in changed["samples"] if r["role"] == "threshold_normal")["role"] = "final_holdout"
        with self.assertRaises(ValueError):
            calibration_split(changed, self.policy)

    def test_holdout_contains_only_21_normals_and_29_defects(self):
        rows = holdout_rows(self.document)
        self.assertEqual(len(rows), 50)
        self.assertEqual(sum(r["label"] == "good" for r in rows), 21)

    def test_gate_requires_all_three_targets_and_finite_metrics(self):
        metrics = {"image_auroc": .95, "recall": .95, "false_alarm_rate": .10}
        self.assertTrue(acceptance_gate(metrics)["passed"])
        for key, value in (("image_auroc", .949), ("recall", .949),
                           ("false_alarm_rate", .101), ("recall", float("nan"))):
            self.assertFalse(acceptance_gate(dict(metrics, **{key: value}))["passed"])

    def test_holdout_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exposure.json"
            claim_holdout(path, {"status": "STARTED"})
            with self.assertRaises(FileExistsError):
                claim_holdout(path, {"status": "RESTARTED"})
            self.assertEqual(json.loads(path.read_text())["status"], "STARTED")

    def test_grayscale_preprocessing_matches_explicit_rgb(self):
        import torch
        from torchvision.transforms import Compose, Resize, ToTensor
        from forgesight.data_loader import load_verified_image
        row = next(r for r in self.document["samples"] if r["role"] == "validation_normal")
        root = Path("data/mvtec_ad/screw")
        with load_verified_image(root, row) as image:
            expected = Compose([Resize((256, 256), antialias=True), ToTensor()])(
                image.convert("RGB")).unsqueeze(0)
        self.assertTrue(torch.equal(tensor_for(row, root, "cpu"), expected))


if __name__ == "__main__":
    unittest.main()
