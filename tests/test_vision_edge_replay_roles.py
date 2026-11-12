"""The offline runtime check must not consume held-out evaluation images."""

import unittest

from forgesight.vision_edge_replay import select_replay_sample


class ReplayRoleTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"path": "test/bent/017.png", "sha256": "dev", "role": "development_preview"},
            {"path": "train/good/000.png", "sha256": "fit", "role": "fit"},
            {"path": "train/good/001.png", "sha256": "other", "role": "fit"},
            {"path": "test/good/001.png", "sha256": "held", "role": "final_holdout"},
        ]

    def test_development_preview_allowed_by_default(self):
        sample = select_replay_sample(self.samples, "dev", "/data/test/bent/017.png")
        self.assertEqual(sample["role"], "development_preview")

    def test_only_explicit_named_fit_reference_allowed(self):
        with self.assertRaises(ValueError):
            select_replay_sample(self.samples, "fit", "/data/train/good/000.png")
        sample = select_replay_sample(self.samples, "fit", "/data/train/good/000.png", True)
        self.assertEqual(sample["path"], "train/good/000.png")
        with self.assertRaises(ValueError):
            select_replay_sample(self.samples, "other", "/data/train/good/001.png", True)

    def test_holdout_and_wrong_path_rejected(self):
        with self.assertRaises(ValueError):
            select_replay_sample(self.samples, "held", "/data/test/good/001.png", True)
        with self.assertRaises(ValueError):
            select_replay_sample(self.samples, "dev", "/data/test/bent/999.png", True)


if __name__ == "__main__":
    unittest.main()
