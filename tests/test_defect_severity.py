"""Check policy boundaries, location and abstention."""

import unittest

from forgesight.defect_severity import POLICY_ID, assess_severity


class DefectSeverityTests(unittest.TestCase):
    def assessment(self, defect_type, *, span=14, area=19, location="annular face", outline=(100, 1000)):
        region = {"region_id": 1, "area_px": area, "bbox_diagonal_px": span,
                  "approximate_nut_area": location}
        return assess_severity(defect_type, {"regions": [region]}, outline)

    def test_inclusive_thresholds_and_type_specific_metrics(self):
        for defect_type, key, cases in (
            ("scratch", "span", ((14, "LOW"), (15, "MEDIUM"), (34, "MEDIUM"), (35, "HIGH"))),
            ("discoloration", "area", ((19, "LOW"), (20, "MEDIUM"), (79, "MEDIUM"), (80, "HIGH"))),
            ("bent", "area", ((49, "LOW"), (50, "MEDIUM"), (149, "MEDIUM"), (150, "HIGH"))),
        ):
            for size, level in cases:
                with self.subTest(defect_type=defect_type, size=size):
                    outcome = self.assessment(defect_type, **{key: size})
                    self.assertEqual(outcome["policy_id"], POLICY_ID)
                    self.assertEqual(outcome["level"], level)

    def test_inner_rim_escalates_scratch_and_bent_only(self):
        location = "inner rim / annular face"
        for label in ("scratch", "bent", "discoloration"):
            expected = "LOW" if label == "discoloration" else "MEDIUM"
            with self.subTest(label=label):
                result = self.assessment(label, location=location)
                self.assertEqual(result["level"], expected)
                self.assertEqual(result["evidence"]["size_band"], "LOW")
                self.assertEqual(result["evidence"]["rim_escalation"], expected != "LOW")
        self.assertEqual(self.assessment("scratch", location=location, span=40)["level"], "HIGH")

    def test_abstention_reasons(self):
        for result, reason in (
            (self.assessment("flip"), "ORIENTATION_ISSUE"),
            (self.assessment("unknown"), "UNKNOWN_DEFECT_TYPE"),
            (self.assessment("scratch", outline=None), "LOCALIZATION_UNAVAILABLE"),
            (assess_severity("bent", {"regions": []}, (100, 1000)), "LOCALIZATION_UNAVAILABLE"),
            (self.assessment("scratch", location=None), "LOCALIZATION_UNCERTAIN"),
            (self.assessment("scratch", location="central opening"), "LOCALIZATION_UNCERTAIN"),
            (self.assessment("scratch", location="area undetermined; response spans multiple areas"),
             "LOCALIZATION_UNCERTAIN"),
            (self.assessment("scratch", span=101), "RESPONSE_EXTENT_UNCERTAIN"),
        ):
            with self.subTest(reason=reason):
                self.assertEqual(result["level"], "UNASSESSED")
                self.assertEqual(result["reason"], reason)
                self.assertNotIn("evidence", result)
