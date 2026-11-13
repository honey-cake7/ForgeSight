"""Acceptance-gate logic tests: warm vs cold advisory comparison with
synthetic reports. The real reports are produced on the Jetson."""

import copy
import unittest

from forgesight.assist_worker_acceptance import compare_reports

NOTE = ("[E1] Anomaly flagged; grade REVIEW; defect type and cause unknown. "
        "[G1] Prototype checklist: hold for human review; check image clarity and lighting.")


def report(mode, note=NOTE, wall=None, status="JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW"):
    return {"status": status, "llama_runtime_mode": mode,
            "inspection_sha256": "i" * 64, "decision_sha256": "d" * 64,
            "authoritative_grade": "REVIEW",
            "inspection_and_sqlite_unchanged": True,
            "controller_runtime_wall_ms": wall,
            "native": {"process_wall_seconds": None},
            "advisory": {"response": {"generated_note": note}}}


class AcceptanceTests(unittest.TestCase):
    def test_identical_verified_notes_accept(self):
        verdict = compare_reports(report("PERSISTENT_WORKER", wall=3000.0),
                                  report("COLD_SUBPROCESS", wall=21000.0))
        self.assertTrue(verdict["adoption_accepted"])
        self.assertEqual(verdict["warm_wall"], 3000.0)

    def test_different_note_rejects(self):
        # At temperature 0 a different warm note means state leaked across
        # requests or the runtime diverged: adoption is blocked.
        verdict = compare_reports(report("PERSISTENT_WORKER", note="[E1] different wording entirely [G1]"),
                                  report("COLD_SUBPROCESS"))
        self.assertFalse(verdict["checks"]["notes_byte_identical"])
        self.assertFalse(verdict["adoption_accepted"])

    def test_unverified_either_side_rejects(self):
        verdict = compare_reports(report("PERSISTENT_WORKER", status="ADVICE_UNAVAILABLE"),
                                  report("COLD_SUBPROCESS"))
        self.assertFalse(verdict["adoption_accepted"])

    def test_different_inspection_rejects(self):
        cold = report("COLD_SUBPROCESS")
        cold["inspection_sha256"] = "x" * 64
        verdict = compare_reports(report("PERSISTENT_WORKER"), cold)
        self.assertFalse(verdict["adoption_accepted"])

    def test_mutated_grade_rejects(self):
        warm = report("PERSISTENT_WORKER")
        warm["authoritative_grade"] = "PASS"
        verdict = compare_reports(warm, report("COLD_SUBPROCESS"))
        self.assertFalse(verdict["adoption_accepted"])

    def test_missing_immutability_recheck_rejects(self):
        warm = report("PERSISTENT_WORKER")
        warm["inspection_and_sqlite_unchanged"] = False
        verdict = compare_reports(warm, report("COLD_SUBPROCESS"))
        self.assertFalse(verdict["adoption_accepted"])


if __name__ == "__main__":
    unittest.main()
