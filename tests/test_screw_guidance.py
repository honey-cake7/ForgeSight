"""Screw advice uses bounded check choices and never changes quality decisions."""
import json
from copy import deepcopy
from pathlib import Path
import unittest


class ScrewGuidanceTests(unittest.TestCase):
    def documents(self):
        root = Path('plugins/screw')
        return (json.loads((root / 'grade_policy.json').read_text(encoding='utf-8')),
                json.loads((root / 'guidance.json').read_text(encoding='utf-8')))

    def test_all_real_labels_have_reviewed_manual_inspection_choices(self):
        from forgesight.screw_guidance import validate_guidance
        _, guidance = self.documents()
        validate_guidance(guidance)
        self.assertEqual(set(guidance['checks']), {'manipulated_front', 'scratch_head', 'scratch_neck',
                         'thread_side', 'thread_top', 'unknown'})
        self.assertEqual(guidance['approval_status'], 'PROTOTYPE_PENDING_SITE_QC_APPROVAL')
        self.assertFalse(guidance['authorizes_release_or_repair'])
        for choices in guidance['checks'].values():
            self.assertEqual(set(choices), {'A', 'B'})

    def test_choice_guard_never_accepts_unbounded_generated_actions(self):
        from forgesight.screw_guidance import advice_request, resolve_choice
        _, guidance = self.documents()
        request = advice_request('thread_side', guidance)
        self.assertEqual(request['max_new_tokens'], 2)
        self.assertIn('A.', request['prompt_text'])
        self.assertIn('B.', request['prompt_text'])
        accepted = resolve_choice('A', 'thread_side', guidance)
        self.assertEqual(accepted['status'], 'REVIEWED_CHOICE')
        self.assertEqual(accepted['text'], guidance['checks']['thread_side']['A'])
        self.assertEqual(accepted['quality_grade_authority'], 'DETERMINISTIC_RULES_ONLY')
        for reply in ('Release part', 'A and B', 'AB', 'B. Repair the thread', ''):
            result = resolve_choice(reply, 'thread_side', guidance)
            self.assertEqual(result['status'], 'FALLBACK_MANUAL_INSPECTION')
            self.assertEqual(result['text'], guidance['fallback'])
        with self.assertRaises(ValueError):
            advice_request('invented_label', guidance)

    def test_grade_is_deterministic_and_production_always_requires_review(self):
        from forgesight.screw_guidance import grade_screw
        policy, _ = self.documents()
        threshold = policy['image_threshold']
        model = {'export_sha256': policy['model_sha256'], 'image_threshold': threshold}
        call = lambda score, benchmark: grade_screw(score, policy, benchmark=benchmark, model=model)
        self.assertEqual(call(threshold + .1, True)['grade'], 'REVIEW')
        self.assertEqual(call(threshold * .5, True)['grade'], 'PASS')
        self.assertEqual(call(threshold, True)['grade'], 'REVIEW')
        self.assertEqual(call(threshold * .5, False)['grade'], 'REVIEW')
        self.assertEqual(call(float('nan'), True)['grade'], 'REVIEW')
        self.assertEqual(call(.1, True), call(.1, True))

    def test_guidance_changes_and_misdeclared_grade_policy_are_refused(self):
        from forgesight.screw_guidance import advice_request, grade_screw
        policy, guidance = self.documents()
        changed = deepcopy(guidance)
        changed['checks']['unknown']['A'] = 'Release this part.'
        with self.assertRaises(ValueError):
            advice_request('unknown', changed)
        with self.assertRaises(ValueError):
            grade_screw(.1, dict(policy, normal_outside_margin_grade='REJECT'), benchmark=True)
        with self.assertRaises(ValueError):
            grade_screw(.1, policy, benchmark='false')

    def test_benchmark_pass_requires_matching_detector_hash_and_threshold(self):
        from forgesight.screw_guidance import grade_screw
        policy, _ = self.documents()
        self.assertEqual(grade_screw(.1, policy, benchmark=True)['grade'], 'REVIEW')
        model = {'export_sha256': policy['model_sha256'], 'image_threshold': policy['image_threshold']}
        self.assertEqual(grade_screw(.1, policy, benchmark=True, model=model)['grade'], 'PASS')
        for bad in (dict(model, export_sha256='b' * 64), dict(model, image_threshold=.5)):
            self.assertEqual(grade_screw(.1, policy, benchmark=True, model=bad)['grade'], 'REVIEW')
        tampered_policy = dict(policy, model_sha256='b' * 64, image_threshold=.5)
        self.assertEqual(grade_screw(.3, tampered_policy, benchmark=True, model=model)['grade'], 'REVIEW')


if __name__ == '__main__':
    unittest.main()
