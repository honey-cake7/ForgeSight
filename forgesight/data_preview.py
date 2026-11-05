"""Display fitting/development annotations without opening final-holdout images."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from forgesight.data_loader import MetalNutDataset
from forgesight.data_manifest import DEFAULT_MANIFEST, DEFAULT_ROOT


def annotation_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Tint annotated pixels red; leave all unannotated pixels unchanged."""
    if image.mode != "RGB" or mask.mode != "L" or image.size != mask.size:
        raise ValueError("Overlay requires aligned RGB image and grayscale mask.")
    values = {value for _, value in mask.getcolors(256)}
    if not values <= {0, 255}:
        raise ValueError("Overlay mask must use binary 0/255 values.")
    with Image.new("RGB", image.size, (255, 0, 0)) as red:
        with Image.blend(image, red, 0.5) as tinted:
            return Image.composite(tinted, image, mask)


def create_preview(root: Path, manifest: Path, output: Path) -> dict:
    normal = MetalNutDataset(root, manifest, "fit")
    development = MetalNutDataset(root, manifest, "development_preview")
    if output.exists():
        raise FileExistsError(f"Preview directory exists; preserve it or choose a new path: {output}")
    tile, gap, header, row_height, footer = 300, 16, 110, 338, 68
    width = tile * 3 + gap * 4
    height = header + row_height * (1 + len(development)) + footer
    records = []
    with Image.new("RGB", (width, height), "white") as sheet:
        draw = ImageDraw.Draw(sheet)
        title_font = ImageFont.load_default(size=26)
        font = ImageFont.load_default(size=16)
        small_font = ImageFont.load_default(size=14)
        draw.text((gap, 12), "ForgeSight | Dataset inspection", fill="black", font=title_font)
        draw.text((gap, 45), "GROUND TRUTH ANNOTATIONS - no model predictions", fill="#a02020", font=font)
        for column, title in enumerate(("Source image", "Annotation mask", "Red annotation overlay")):
            draw.text((gap + column * (tile + gap), 80), title, fill="black", font=font)
        selections = [(normal, 0)] + [(development, index) for index in range(len(development))]
        for row_index, (dataset, index) in enumerate(selections):
            sample = dataset[index]
            with sample.image, sample.mask:
                if sample.role not in {"fit", "development_preview"}:
                    raise PermissionError("Preview rejects validation and final-holdout samples.")
                y = header + row_index * row_height
                draw.text((gap, y), f"{sample.ground_truth_type} | {sample.source_path} | {sample.role}",
                          fill="black", font=font)
                with annotation_overlay(sample.image, sample.mask) as overlay:
                    for column, (source, resample) in enumerate((
                            (sample.image, Image.Resampling.BICUBIC),
                            (sample.mask, Image.Resampling.NEAREST),
                            (overlay, Image.Resampling.BICUBIC))):
                        with source.resize((tile, tile), resample) as thumbnail:
                            sheet.paste(thumbnail, (gap + column * (tile + gap), y + 28))
                records.append({"path": sample.source_path, "role": sample.role,
                                "ground_truth_type": sample.ground_truth_type,
                                "mask_origin": sample.mask_origin, "source_size": list(sample.image.size),
                                "annotation_area_pixels": sample.mask.histogram()[255],
                                "annotation_bbox_pixels": sample.mask.getbbox()})
        y = height - footer
        draw.text((gap, y), "MVTec AD / MVTec Software GmbH | CC BY-NC-SA 4.0", fill="#333333", font=small_font)
        draw.text((gap, y + 20), "Images resized for display; overlays added. Source inputs remain 700x700.",
                  fill="#333333", font=small_font)
        draw.text((gap, y + 40), "No final-holdout images are shown. See preview.json for provenance.",
                  fill="#333333", font=small_font)
        output.mkdir(parents=True, exist_ok=False)
        with (output / "contact_sheet.png").open("xb") as stream:
            sheet.save(stream, format="PNG")
    report = {"schema_version": 1, "kind": "dataset_ground_truth_preview",
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "samples": records, "holdout_images_displayed": 0,
              "prediction_model": None, "display_tile_size": [tile, tile],
              "attribution": {"dataset": "MVTec AD", "creator": "MVTec Software GmbH",
                              "authors": ["Paul Bergmann", "Michael Fauser", "David Sattlegger", "Carsten Steger"],
                              "paper": "MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection, CVPR 2019",
                              "source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
                              "license": "CC BY-NC-SA 4.0",
                              "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                              "changes": "Resized display tiles and added red ground-truth overlays."}}
    with (output / "preview.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/development_preview_v1"))
    args = parser.parse_args()
    try:
        report = create_preview(args.root, args.manifest, args.output)
        print(json.dumps({"samples": report["samples"], "holdout_images_displayed": 0,
                          "output": str(args.output)}, indent=2))
        print("PASS: ground-truth previews saved; no model predictions or calibrated measurements.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
