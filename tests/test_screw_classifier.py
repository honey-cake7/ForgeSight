"""Classifier fitting uses frozen roles and independent validation abstention."""
from copy import deepcopy
import json
import unittest

from forgesight.defect_type import SCREW_FEATURES, SCREW_TYPES, predict_type


class ScrewClassifierTests(unittest.TestCase):
    def test_selection_cannot_gain_accuracy_by_dropping_a_native_class(self):
        from forgesight.screw_classifier import candidate_rank
        complete = {'supported_labels': list(SCREW_TYPES), 'validation_correct': 16,
                    'leaf_count': 6, 'feature_count': 18}
        missing = dict(complete, supported_labels=list(SCREW_TYPES[:-1]), validation_correct=17, leaf_count=4)
        self.assertGreater(candidate_rank(complete), candidate_rank(missing))

    def rows(self):
        result = {'classifier_fit': [], 'classifier_validation': []}
        for index, label in enumerate(SCREW_TYPES):
            for role, count in [('classifier_fit', 6), ('classifier_validation', 2)]:
                for i in range(count):
                    features = dict.fromkeys(SCREW_FEATURES, 0.)
                    features['peak_axial'] = index * 10 + i * .01
                    result[role].append({'role': role, 'label': label, 'flagged': True,
                        'features': features, 'source_path': f'{role}/{label}/{i}.png',
                        'source_sha256': f'{role}:{label}:{i}'})
        return result

    def test_tree_exports_all_native_classes_and_matches_fit(self):
        from forgesight.screw_classifier import fit_profile
        profile, metrics = fit_profile(self.rows(), 'a' * 64, .5, 'b' * 64)
        self.assertEqual(profile['kind'], 'screw_decision_tree')
        self.assertEqual(profile['labels'], list(SCREW_TYPES))
        self.assertFalse(profile['fresh_acceptance_verified'])
        self.assertEqual(metrics['selection']['data_role'], 'classifier_validation')
        self.assertEqual(len(metrics['selection']['candidates']), 27)
        self.assertFalse(metrics['selection']['final_holdout_used'])
        json.dumps(metrics, allow_nan=False)
        for row in self.rows()['classifier_validation']:
            self.assertEqual(predict_type(row['features'], profile), row['label'])
        self.assertEqual(metrics['validation_after_leaf_abstention']['correct'], 10)

    def test_validation_conflict_abstains_and_training_cannot_use_holdout(self):
        from forgesight.screw_classifier import fit_profile
        rows = self.rows()
        rows['classifier_validation'][0]['label'] = 'thread_top'
        rows['classifier_validation'][1]['label'] = 'thread_top'
        profile, _ = fit_profile(rows, 'a' * 64, .5, 'b' * 64)
        self.assertEqual(predict_type(rows['classifier_fit'][0]['features'], profile), 'unknown')
        bad = self.rows()
        bad['classifier_fit'][0]['role'] = 'final_holdout'
        with self.assertRaises(ValueError):
            fit_profile(bad, 'a' * 64, .5, 'b' * 64)

    def test_mixed_validation_support_is_reported_without_hiding_provisional_labels(self):
        from forgesight.screw_classifier import fit_profile
        rows = self.rows()
        rows['classifier_validation'][0]['label'] = 'thread_top'
        profile, metrics = fit_profile(rows, 'a' * 64, .5, 'b' * 64)
        self.assertEqual(predict_type(rows['classifier_fit'][0]['features'], profile), 'manipulated_front')
        summary = metrics['validation_after_leaf_abstention']
        self.assertEqual(summary['correct'], 9)
        self.assertEqual(summary['incorrect_type_calls'], 1)
        bad = self.rows()
        bad['classifier_validation'][0]['source_sha256'] = bad['classifier_fit'][0]['source_sha256']
        with self.assertRaises(ValueError):
            fit_profile(bad, 'a' * 64, .5, 'b' * 64)

    def test_detector_misses_and_unusable_geometry_count_as_coverage_losses(self):
        from forgesight.screw_classifier import fit_profile
        rows = self.rows()
        rows['classifier_validation'][0].update(flagged=False, features=None)
        _, metrics = fit_profile(rows, 'a' * 64, .5, 'b' * 64)
        self.assertEqual(metrics['validation_after_leaf_abstention']['images'], 10)
        self.assertEqual(metrics['validation_after_leaf_abstention']['not_flagged'], 1)
        bad = self.rows()
        for row in bad['classifier_fit']:
            if row['label'] == 'thread_top':
                row['features'] = None
        with self.assertRaises(ValueError):
            fit_profile(bad, 'a' * 64, .5, 'b' * 64)


if __name__ == '__main__':
    unittest.main()
