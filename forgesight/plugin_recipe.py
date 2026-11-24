"""Hash-bound local recipe bundles for the opt-in dual-part demonstration."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PluginRecipe:
    plugin_id: str
    label: str
    metadata: dict
    paths: dict
    documents: dict

    @property
    def threshold(self):
        return self.documents['score_policy']['threshold']

    @property
    def model_sha256(self):
        return self.metadata['model']['sha256']


class PluginRegistry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.recipes = {}
        for name in ('metal_nut', 'screw'):
            folder = self.root / name
            metadata = json.loads((folder / 'recipe.json').read_text(encoding='utf-8'))
            if metadata.get('schema_version') != 1 or metadata.get('plugin_id') != name:
                raise ValueError('Invalid part recipe identity.')
            paths, documents = {}, {}
            for component in ('model', 'score_policy', 'grade_policy', 'classifier', 'guidance', 'dataset'):
                entry = metadata[component]
                filename = entry['file']
                if not isinstance(filename, str) or Path(filename).name != filename or '\\' in filename:
                    raise ValueError('Recipe assets must be files inside their bundle.')
                path = (folder / filename).resolve()
                if path.parent != folder.resolve():
                    raise ValueError('Recipe asset leaves its bundle: ' + name + '/' + component)
                if not path.is_file():
                    raise ValueError('Missing recipe asset: ' + str(path))
                if digest(path) != entry['sha256']:
                    raise ValueError('Recipe asset hash mismatch: ' + name + '/' + component)
                paths[component] = path
                if component != 'model':
                    documents[component] = json.loads(path.read_text(encoding='utf-8'))
            threshold = documents['score_policy']['threshold']
            if (type(threshold) not in (int, float) or not math.isfinite(threshold)
                    or documents['grade_policy']['image_threshold'] != threshold):
                raise ValueError('Recipe threshold/policy mismatch.')
            classifier = documents['classifier']
            if classifier and (classifier.get('model_sha256') != metadata['model']['sha256']
                               or classifier.get('image_threshold') != threshold):
                raise ValueError('Recipe classifier/model mismatch.')
            self.recipes[name] = PluginRecipe(name, metadata['label'], metadata, paths, documents)
