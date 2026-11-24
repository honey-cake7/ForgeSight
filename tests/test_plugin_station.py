"""Recipe integrity and atomic demo switching without model quality claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace
import unittest


class PluginTests(unittest.TestCase):
    def test_live_template_exposes_selector_without_removing_existing_controls(self):
        html = Path('forgesight/web/jetson_live.html').read_text()
        self.assertIn('id="active-plugin"', html)
        for control in ('camera-start', 'camera-stop', 'theme-toggle', 'station-sidebar', 'view-title'):
            self.assertIn('id="' + control + '"', html)
        self.assertIn('/api/plugin/select', Path('forgesight/web/jetson_live.js').read_text())

    def fixture(self, root, name):
        folder = root / name
        folder.mkdir()
        payloads = {'model.ts': name.encode(), 'policy.json': b'{"threshold":0.2}',
                    'grade.json': b'{"image_threshold":0.2}', 'type.json': b'{}',
                    'guidance.json': b'{}', 'dataset.json': b'{"samples":[]}'}
        for filename, data in payloads.items():
            (folder / filename).write_bytes(data)
        component = lambda filename: {'file': filename, 'sha256': hashlib.sha256(payloads[filename]).hexdigest()}
        metadata = {'schema_version': 1, 'plugin_id': name, 'label': name.replace('_', ' ').upper(),
                    'model': component('model.ts'), 'score_policy': component('policy.json'),
                    'grade_policy': component('grade.json'), 'classifier': component('type.json'),
                    'guidance': component('guidance.json'), 'dataset': component('dataset.json'),
                    'benchmark_size_px': [1024, 1024] if name == 'screw' else [700, 700],
                    'replay_roles': ['classifier_validation'] if name == 'screw' else ['development_preview']}
        (folder / 'recipe.json').write_text(json.dumps(metadata))
        return folder

    def test_registry_verifies_every_asset_and_rejects_path_escape(self):
        from forgesight.plugin_recipe import PluginRegistry
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.fixture(root, 'metal_nut')
            self.fixture(root, 'screw')
            registry = PluginRegistry(root)
            self.assertEqual(set(registry.recipes), {'metal_nut', 'screw'})
            self.assertEqual(registry.recipes['screw'].threshold, .2)
            (folder / 'grade.json').write_text('{}')
            with self.assertRaises(ValueError):
                PluginRegistry(root)
            metadata = json.loads((folder / 'recipe.json').read_text())
            metadata['grade_policy']['file'] = '../outside.json'
            (folder / 'recipe.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                PluginRegistry(root)

    def test_atomic_switch_preserves_llama_and_rejects_busy_or_dead_workers(self):
        from forgesight.plugin_recipe import PluginRegistry
        from forgesight.jetson_live_ui import LiveService
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('metal_nut', 'screw'):
                self.fixture(root, name)
            service = LiveService.__new__(LiveService)
            service.lock = Lock()
            service.busy = False
            service.closed = False
            service.worker_respawning = False
            service.plugin_registry = PluginRegistry(root)
            service.plugin_workers = {key: SimpleNamespace(poll=lambda: None) for key in service.plugin_registry.recipes}
            service.plugin_sockets = {key: root / (key + '.sock') for key in service.plugin_registry.recipes}
            service.plugin_advisors = dict.fromkeys(service.plugin_registry.recipes, None)
            service.plugin_dataset_roots = dict.fromkeys(service.plugin_registry.recipes, root)
            service.args = SimpleNamespace()
            service.advice_worker = object()
            llama = service.advice_worker
            service.latest = {'old': True}
            service.active_plugin = 'metal_nut'
            result = service.select_plugin('screw')
            self.assertEqual(result['active_plugin'], 'screw')
            self.assertLess(result['switch_ms'], 500)
            self.assertIs(service.advice_worker, llama)
            self.assertEqual(service.args.model, root / 'screw/model.ts')
            self.assertIsNone(service.latest)
            service.busy = True
            with self.assertRaises(ValueError):
                service.select_plugin('metal_nut')
            self.assertEqual(service.active_plugin, 'screw')
            service.busy = False
            service.plugin_workers['metal_nut'] = SimpleNamespace(poll=lambda: 1)
            with self.assertRaises(ValueError):
                service.select_plugin('metal_nut')
            with self.assertRaises(ValueError):
                service.select_plugin('../screw')


if __name__ == '__main__':
    unittest.main()
