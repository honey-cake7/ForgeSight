"""Prepare one RGB image for the frozen 256-pixel EfficientAD input on Jetson."""

import argparse
import hashlib
import json


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--tensor", required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image

    with Image.open(args.image) as source:
        image = source.convert("RGB").resize((256, 256), Image.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
    tensor = np.transpose(array, (2, 0, 1)).astype(np.float32)[None] / 255.0
    with open(args.tensor, "xb") as stream:
        np.save(stream, tensor, allow_pickle=False)
    print(json.dumps({"image_sha256": sha256(args.image), "tensor_sha256": sha256(args.tensor),
                      "tensor_shape": list(tensor.shape), "color": "RGB", "resize": "Pillow bilinear 256x256"}))


if __name__ == "__main__":
    main()
