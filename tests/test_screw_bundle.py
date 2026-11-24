"""The development plugin binds its JSON assets without promoting unsafe readiness."""
import hashlib
import json
from pathlib import Path
import unittest


class ScrewBundleTests(unittest.TestCase):
    def test_guidance_verification_is_current_or_explicitly_historical(self):
        from forgesight.assist_contract import canonical_hash
        root = Path('plugins/screw')
        vision = json.loads((root / 'vision.json').read_text(encoding='utf-8'))
        binding = vision['guidance']
        current = canonical_hash(json.loads((root / binding['file']).read_text(encoding='utf-8')))
        reference = binding.get('verification_reference')
        if reference:
            report = json.loads(Path(reference).read_text(encoding='utf-8'))
            hashes = {row['guidance_sha256'] for row in report['validated_choices'].values()}
            self.assertEqual(hashes, {current}, 'Active verification must cover the current guidance')
        else:
            self.assertEqual(binding['verification_status'], 'NOT_VERIFIED_FOR_CURRENT_GUIDANCE')
            historical = Path(binding['historical_verification_reference'])
            self.assertEqual(hashlib.sha256(historical.read_bytes()).hexdigest(),
                             binding['historical_verification_reference_sha256'])
            self.assertFalse(vision['gate_2']['llama_choice_checks_passed'])

    def test_reference_paths_are_portable_between_windows_and_jetson(self):
        vision = json.loads(Path('plugins/screw/vision.json').read_text())
        for key in ('development_selection', 'dataset_manifest', 'training_reference',
                    'calibration_reference', 'export_verification', 'diagnostic_reference'):
            self.assertNotIn('\\', vision[key], key)

    def test_all_packaged_json_components_match_their_pinned_hashes(self):
        root = Path('plugins/screw')
        vision = json.loads((root / 'vision.json').read_text())
        for name in ('score_policy', 'classifier', 'grade_policy', 'guidance'):
            component = vision[name]
            self.assertIsInstance(component, dict, name)
            self.assertEqual(Path(component['file']).name, component['file'])
            payload = (root / component['file']).read_bytes()
            self.assertNotIn(b'\r\n', payload, name)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), component['sha256'], name)
        policy = json.loads((root / vision['score_policy']['file']).read_text())
        grade = json.loads((root / vision['grade_policy']['file']).read_text())
        classifier = json.loads((root / vision['classifier']['file']).read_text())
        self.assertEqual(policy['threshold'], grade['image_threshold'])
        self.assertEqual(policy['threshold'], classifier['image_threshold'])
        self.assertEqual(grade['model_sha256'], vision['model']['sha256'])
        self.assertEqual(classifier['model_sha256'], vision['model']['sha256'])
        self.assertEqual((root / 'efficientad_s.ts.sha256').read_text().strip(), vision['model']['sha256'])

    def test_operator_development_selection_does_not_claim_passed_acceptance(self):
        root = Path('plugins/screw')
        vision = json.loads((root / 'vision.json').read_text())
        self.assertFalse(vision['ready_for_station'])
        self.assertFalse(vision['gate_1']['passed'])
        self.assertFalse(vision['gate_2']['passed'])
        self.assertEqual(vision['physical_camera_accuracy'], 'NOT_VERIFIED')
        selection = Path(vision['development_selection'])
        self.assertEqual(hashlib.sha256(selection.read_bytes()).hexdigest(), vision['development_selection_sha256'])
        self.assertFalse(json.loads(selection.read_text())['acceptance_gate_passed'])


if __name__ == '__main__':
    unittest.main()
