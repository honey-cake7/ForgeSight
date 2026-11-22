"""Regression coverage for the opt-in, advisory-only defect assessment."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from forgesight.defect_type import assess_response, load_profile, predict_type, response_features
from forgesight.regions import response_regions


ROOT = Path(__file__).resolve().parents[1]


class DefectTypeTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile(ROOT / "configs/defect_type_preview_v1.json")
        self.values = np.zeros((80, 80), dtype=np.float32)
        self.values[20:45, 30:48] = 1
        self.geometry = response_regions(self.values, self.profile["image_threshold"])
        pixels = np.full((80, 80, 3), 128, dtype=np.uint8)
        pixels[self.values != 0] = [180, 100, 100]
        self.image = Image.fromarray(pixels)
        self.record = {"source": {"mode": "SAVED_IMAGE_REPLAY"}, "candidate": "ANOMALY_CANDIDATE",
                       "model": {"export_sha256": self.profile["model_sha256"],
                                 "image_threshold": self.profile["image_threshold"]},
                       "geometry": self.geometry, "timings": {}, "predicted_defect_type": "unknown",
                       "decision": {"grade": "REVIEW"}}

    def test_chroma_cue_preserves_authoritative_fields(self):
        prior = deepcopy(self.record)
        assess_response(self.record, self.image, self.values, self.profile)
        self.assertEqual(self.record.pop("assessment")["defect_type"], "discoloration")
        self.assertGreaterEqual(self.record["timings"].pop("defect_type_ms"), 0)
        self.assertEqual(self.record, prior)

    def test_skip_ineligible_and_mismatched_detector_abstains(self):
        assess_response(self.record, self.image, self.values, None)
        self.assertNotIn("assessment", self.record)
        self.record["source"]["mode"] = "LIVE_CAMERA"
        assess_response(self.record, self.image, self.values, self.profile)
        self.assertNotIn("assessment", self.record)
        self.record["source"]["mode"] = "SAVED_IMAGE_REPLAY"
        self.record["candidate"] = "NORMAL_CANDIDATE"
        assess_response(self.record, self.image, self.values, self.profile)
        self.assertNotIn("assessment", self.record)
        self.record["candidate"] = "ANOMALY_CANDIDATE"
        self.record["model"]["image_threshold"] = 0.9
        assess_response(self.record, self.image, self.values, self.profile)
        self.assertEqual(self.record["assessment"]["defect_type"], "unknown")
        self.assertIsNone(self.record["assessment"]["features"])

    def test_diagonal_region_elongation_and_missing_region(self):
        values = np.zeros_like(self.values)
        values[np.arange(20, 60), np.arange(20, 60)] = 1
        geometry = response_regions(values, 0.5)
        features = response_features(self.image, values, geometry)
        self.assertGreater(features["elongation"], 10)
        self.assertIsNone(response_features(self.image, values, {"regions": []}))

    def test_tree_load_prediction_and_validation(self):
        profile = {**self.profile, "kind": "decision_tree",
                   "tree": {"feature": "elongation", "threshold": 3,
                            "left": {"label": "unknown"}, "right": {"label": "scratch"}}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tree.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            exported = load_profile(path)
            self.assertEqual(predict_type({"elongation": 3, "chroma_difference": 1}, exported), "unknown")
            self.assertEqual(predict_type({"elongation": 4, "chroma_difference": 1}, exported), "scratch")
            self.assertEqual(predict_type(None, exported), "unknown")
            profile["tree"]["right"]["label"] = "invented"
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_profile(path)
