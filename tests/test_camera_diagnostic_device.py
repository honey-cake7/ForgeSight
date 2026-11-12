"""An unavailable capture device must fail before creating a frame."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from forgesight.vision_edge_camera_diagnostic import capture_frame


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed")
class CameraDeviceTests(unittest.TestCase):
    def test_unavailable_device_creates_no_frame(self):
        with TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.png"
            with self.assertRaisesRegex(RuntimeError, "did not open"):
                capture_frame(frame, 99)
            self.assertFalse(frame.exists())


if __name__ == "__main__":
    unittest.main()
