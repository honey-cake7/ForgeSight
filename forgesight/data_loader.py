"""Load unchanged image/mask pairs from a verified, role-restricted manifest."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from forgesight.data_manifest import DEFAULT_MANIFEST, DEFAULT_ROOT, ROLES, read_manifest, safe_path


@dataclass(frozen=True)
class InspectionInput:
    source_path: str
    role: str
    ground_truth_type: str
    image: Image.Image
    mask: Image.Image
    mask_origin: str


def load_verified_image(root: Path, record: dict) -> Image.Image:
    path = safe_path(root, record["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"File hash mismatch: {record['path']}")
    with Image.open(BytesIO(data)) as source:
        source.load()
        if source.mode != record["mode"] or list(source.size) != record["size"]:
            raise ValueError(f"Geometry/mode mismatch: {record['path']}")
        if hashlib.sha256(source.tobytes()).hexdigest() != record["pixel_sha256"]:
            raise ValueError(f"Decoded-pixel hash mismatch: {record['path']}")
        return source.copy()


class MetalNutDataset:
    """A single role, preserving source resolution and separating ground truth."""

    def __init__(self, root: Path = DEFAULT_ROOT, manifest: Path = DEFAULT_MANIFEST,
                 role: str = "fit", *, allow_final_holdout: bool = False):
        if role not in ROLES:
            raise ValueError(f"Unknown dataset role: {role}")
        if role == "final_holdout" and not allow_final_holdout:
            raise PermissionError("Final holdout access requires explicit evaluation authorization.")
        self.root = Path(root)
        self.role = role
        document = read_manifest(Path(manifest))
        self._rows = tuple(row for row in document["samples"] if row["role"] == role)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> InspectionInput:
        row = self._rows[index]
        image = load_verified_image(self.root, row)
        try:
            if row["mask"] is None:
                mask = Image.new("L", image.size, 0)
                origin = "implicit_normal"
            else:
                mask = load_verified_image(self.root, row["mask"])
                values = {value for _, value in mask.getcolors(256)}
                if not values <= {0, 255} or not any(values):
                    mask.close()
                    raise ValueError(f"Invalid anomaly mask: {row['path']}")
                origin = "dataset_annotation"
            return InspectionInput(row["path"], row["role"], row["label"], image, mask, origin)
        except Exception:
            image.close()
            raise

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def summarize(dataset: MetalNutDataset) -> dict:
    labels = Counter()
    nonempty = 0
    for sample in dataset:
        with sample.image, sample.mask:
            if sample.image.size != sample.mask.size:
                raise ValueError(f"Image/mask size mismatch: {sample.source_path}")
            labels[sample.ground_truth_type] += 1
            nonempty += sample.mask.getbbox() is not None
    return {"role": dataset.role, "samples_loaded": len(dataset),
            "ground_truth_counts": dict(sorted(labels.items())), "nonempty_masks": nonempty}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--role", choices=ROLES, default="fit")
    parser.add_argument("--allow-final-holdout", action="store_true",
                        help="Explicit evaluation access; leave off during development.")
    args = parser.parse_args()
    try:
        dataset = MetalNutDataset(args.root, args.manifest, args.role,
                                 allow_final_holdout=args.allow_final_holdout)
        print(json.dumps(summarize(dataset), indent=2))
        print("PASS: verified image/mask loading. No predictions or visual previews.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
