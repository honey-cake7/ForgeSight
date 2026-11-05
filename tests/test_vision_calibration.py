"""Calibration partition/rank/bundle guards, without simulated detector success."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from forgesight.vision_calibrate import calibration_split, classify_score, rank_threshold, validate_quantiles
from forgesight.vision_calibration_bundle import load_calibrated_bundle, validate_bundle_policy


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(Path("configs/vision_score_policy.json").read_text())
        self.document = {"counts": {"validation_normal": 44}, "samples": [
            {"role": "validation_normal", "label": "good", "source_split": "train", "mask": None,
             "path": f"train/good/{i:03d}.png"} for i in range(44)]}

    def test_subsets_are_disjoint_exhaustive_and_order_independent(self):
        maps, threshold = calibration_split(self.document, self.policy)
        shuffled = dict(self.document, samples=list(reversed(self.document["samples"])))
        self.assertEqual((maps, threshold), calibration_split(shuffled, self.policy))
        self.assertEqual((len(maps), len(threshold)), (22, 22))
        self.assertFalse({r["path"] for r in maps} & {r["path"] for r in threshold})

    def test_role_change_duplicates_and_degenerate_split_are_rejected(self):
        for key, value in (("validation_role", "final_holdout"), ("final_holdout_access", True),
                           ("map_fit_count", 44), ("map_fit_count", 0)):
            with self.assertRaises(ValueError):
                calibration_split(self.document, dict(self.policy, **{key: value}))
        duplicate = deepcopy(self.document)
        duplicate["samples"][0]["path"] = duplicate["samples"][1]["path"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            calibration_split(duplicate, self.policy)

    def test_finite_sample_rank_has_correct_boundary(self):
        result = rank_threshold(list(range(22)), 0.05)
        self.assertEqual(result["order_statistic_rank"], 22)
        self.assertEqual(result["threshold"], 21)
        self.assertAlmostEqual(result["rank_tail_fraction_under_assumptions"], 1 / 23)
        self.assertEqual(rank_threshold(list(range(44)), 0.05)["threshold"], 42)

    def test_insufficient_normals_cannot_promise_one_percent(self):
        with self.assertRaisesRegex(ValueError, "Too few"):
            rank_threshold(list(range(22)), 0.01)
        self.assertEqual(rank_threshold(list(range(99)), 0.01)["order_statistic_rank"], 99)

    def test_invalid_samples_or_alpha_are_rejected(self):
        for scores, alpha in (([], 0.05), ([math.nan], 0.05), ([1], 0), ([1], 1), ([1], math.nan)):
            with self.assertRaises(ValueError):
                rank_threshold(scores, alpha)

    def test_equality_is_not_flagged_and_nan_is_an_error(self):
        self.assertEqual(classify_score(-0.2, -0.2), "NORMAL_CANDIDATE")
        self.assertEqual(classify_score(-0.19, -0.2), "ANOMALY_CANDIDATE")
        for score, threshold in ((math.nan, 1), (1, math.inf)):
            with self.assertRaises(ValueError):
                classify_score(score, threshold)

    def test_quantile_ranges_and_candidate_semantics_are_guarded(self):
        for qb in (1, 0, math.nan):
            with self.assertRaises(ValueError):
                validate_quantiles({"qa_st": 1, "qb_st": qb, "qa_ae": 1, "qb_ae": 2})
        policy = dict(self.policy, threshold=0.1, above_threshold="REJECT")
        with self.assertRaisesRegex(ValueError, "semantics"):
            validate_bundle_policy(policy)

    def test_corrupted_policy_is_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "score_policy.json").write_text("corrupted policy bytes")
            reference = {"status": "CALIBRATED_POLICY_FROZEN_HOLDOUT_NOT_EVALUATED",
                         "calibrated_checkpoint": {"path": str(root / "absent.pth")},
                         "policy_file_sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "policy file fingerprint"):
                load_calibrated_bundle(reference)


if __name__ == "__main__":
    unittest.main()
