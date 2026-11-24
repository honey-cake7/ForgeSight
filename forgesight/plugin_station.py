"""Switch already-loaded vision workers and immutable recipe snapshots."""
from copy import copy
import json
from pathlib import Path
import time


class PluginStation:
    def _apply_plugin(self, name):
        recipe = self.plugin_registry.recipes[name]
        self.active_recipe = recipe
        self.active_plugin = name
        self.args.model = recipe.paths['model']
        self.args.model_sha256 = recipe.model_sha256
        self.args.grade_policy = recipe.paths['grade_policy']
        self.args.defect_type_profile = recipe.paths['classifier']
        self.args.guidance = recipe.paths['guidance']
        self.args.dataset_manifest = recipe.paths['dataset']
        self.args.dataset_root = self.plugin_dataset_roots[name]
        self.args.vision_worker_socket = self.plugin_sockets[name]
        self.defect_type_profile = recipe.documents['classifier']
        self.vision_worker = self.plugin_workers[name]
        self.grounded_advice = self.plugin_advisors[name]
        roles = recipe.metadata['replay_roles']
        self.replay_samples = {r['path']: r for r in recipe.documents['dataset']['samples'] if r['role'] in roles}
        # Local files are audited once at boot, never hashed on the switch path.
        self.local_replay_samples = getattr(self, 'plugin_local_samples', {}).get(name, [])

    def select_plugin(self, name):
        started = time.perf_counter()
        with self.lock:
            registry = getattr(self, 'plugin_registry', None)
            if registry is None or not isinstance(name, str) or name not in registry.recipes:
                raise ValueError('Choose an available part recipe.')
            if self.busy or self.closed or self.worker_respawning:
                raise ValueError('Wait for the current inspection or worker recovery.')
            worker = self.plugin_workers[name]
            if worker.poll() is not None:
                raise ValueError('The selected vision worker is unavailable; restart the station.')
            if name != self.active_plugin:
                self._apply_plugin(name)
                self.latest = self.latest_root = self.latest_source = None
            return {'active_plugin': name, 'label': registry.recipes[name].label,
                    'switch_ms': (time.perf_counter() - started) * 1000, 'demo_only': True}

    def plugin_state(self):
        registry = getattr(self, 'plugin_registry', None)
        if registry is None:
            return None
        with self.lock:
            return {'active_plugin': self.active_plugin, 'demo_only': True,
                    'label': self.active_recipe.label,
                    'plugins': [{'id': name, 'label': r.label,
                                 'benchmark_size_px': r.metadata['benchmark_size_px'],
                                 'ready': self.plugin_workers[name].poll() is None}
                                for name, r in registry.recipes.items()]}

    def _start_plugins(self):
        from forgesight.plugin_recipe import PluginRegistry, digest
        from forgesight.jetson_live_ui import launch_vision_worker
        from forgesight.grounded_live import GroundedLive
        from forgesight.screw_live import ScrewLive
        self.plugin_registry = PluginRegistry(self.args.plugins_root)
        baseline_root = self.args.dataset_root
        socket = self.args.vision_worker_socket
        self.plugin_workers = {'metal_nut': self.vision_worker}
        self.plugin_sockets = {'metal_nut': socket}
        self.plugin_dataset_roots = {'metal_nut': baseline_root,
            'screw': self.args.plugins_root.parent / 'data/mvtec_ad/screw'}
        self.plugin_advisors = {'metal_nut': self.grounded_advice}
        self.plugin_local_samples = {}
        for name, recipe in self.plugin_registry.recipes.items():
            if name == 'metal_nut':
                if recipe.model_sha256 != self.args.model_sha256:
                    raise ValueError('Baseline recipe differs from the station model.')
            else:
                args = copy(self.args)
                args.model = recipe.paths['model']
                args.model_sha256 = recipe.model_sha256
                args.vision_worker_socket = socket.with_name(socket.stem + '-' + name + socket.suffix)
                self.plugin_sockets[name] = args.vision_worker_socket
                self.plugin_workers[name] = launch_vision_worker(args)
                self.plugin_advisors[name] = ScrewLive(self.advice_queue, self.advice_root / 'grounded',
                    self.args.llama_worker_socket, self.args.llama_timeout_seconds, recipe,
                    llama_root=self.args.llama_root)
            dataset_root = self.plugin_dataset_roots[name]
            self.plugin_local_samples[name] = [r['path'] for r in recipe.documents['dataset']['samples']
                if r['role'] in recipe.metadata['replay_roles'] and (dataset_root / r['path']).is_file()
                and digest(dataset_root / r['path']) == r['sha256']]
        self.active_plugin = 'metal_nut'
        self._apply_plugin('metal_nut')
