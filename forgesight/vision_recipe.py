"""Select a category-specific manifest validator for offline training."""
import hashlib
import json

from forgesight.data_manifest import read_manifest as read_metal_nut_manifest


def read_manifest(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Manifest checksum mismatch.")
    document = json.loads(data)
    if document.get("category") == "screw":
        from forgesight.screw_recipe import validate_manifest
        validate_manifest(document)
        return document
    return read_metal_nut_manifest(path)
