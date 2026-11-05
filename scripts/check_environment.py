"""Check environment isolation and PNG codecs; this does not inspect defects."""

from io import BytesIO
import platform
import site
import sys


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {sys.platform} {platform.machine()}")
    print(f"Executable: {sys.executable}")
    if sys.prefix == sys.base_prefix or site.ENABLE_USER_SITE:
        raise SystemExit("FAIL: use the project's isolated .venv/bin/python.")
    print("Isolation: PASS (virtual environment; user site disabled)")

    try:
        from PIL import Image, __version__ as pillow_version
    except ImportError as error:
        raise SystemExit("FAIL: install requirements.txt in the project venv.") from error

    print(f"Pillow: {pillow_version}")
    pixels = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30, 255, 255, 255, 0, 0, 0))
    with Image.frombytes("RGB", (3, 2), pixels) as source, BytesIO() as encoded:
        source.save(encoded, format="PNG")
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            decoded.load()
            if (decoded.format, decoded.mode, decoded.size, decoded.tobytes()) != (
                "PNG", "RGB", (3, 2), pixels
            ):
                raise SystemExit("FAIL: PNG decoding changed the known pixels.")
    print("PNG round trip: PASS (3x2 RGB; all pixel bytes preserved)")
    print("Environment smoke check: PASS; no defect model or Jetson was tested.")


if __name__ == "__main__":
    main()
