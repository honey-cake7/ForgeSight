"""Exercise the loopback inspection contract without claiming camera proof."""

from http.server import ThreadingHTTPServer
import hashlib
from io import BytesIO
import json
from pathlib import Path
from threading import Lock, Thread
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from forgesight.jetson_live_ui import LiveService, handler_for, vision_executable


class StubCamera:
    def preview(self):
        return b"\xff\xd8\xff\xd9"


class StubService:
    def __init__(self):
        self.camera = StubCamera()
        self.requests = []

    def state(self):
        return {"camera": {"ready": True, "message": "Test frame"},
                "busy": False, "latest": None, "history": [],
                "advice": {"status": "NOT_ELIGIBLE", "advisory": None},
                "replay_samples": ["test/bent/017.png"],
                "local_replay_samples": ["test/bent/017.png"]}

    def image(self, name):
        return None

    def inspect(self, part_id, confirmed, source_kind):
        if confirmed is not True:
            raise ValueError("Operator declaration required")
        self.requests.append((part_id, source_kind))

    def replay(self, replay_id, dataset_path, data):
        self.requests.append((replay_id, dataset_path, data))

    def replay_local(self, replay_id, dataset_path):
        self.requests.append(("local", replay_id, dataset_path))

    def upload_diagnostic(self, upload_id, data, benchmark=False):
        self.requests.append(("diagnostic", upload_id, data, benchmark))

    def request_advice(self):
        self.requests.append(("advice",))
        return "PENDING"


class LiveHttpTests(unittest.TestCase):
    def test_venv_executable_symlink_is_preserved(self):
        with TemporaryDirectory() as folder:
            venv = Path(folder) / "venv" / "bin"
            venv.mkdir(parents=True)
            base = Path(folder) / "base-python"
            base.write_text("test executable")
            link = venv / "python"
            link.symlink_to(base)
            self.assertEqual(vision_executable(link), link)
            self.assertNotEqual(vision_executable(link), link.resolve())

    def setUp(self):
        self.service = StubService()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.service))
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_page_preview_state_and_same_origin_capture(self):
        self.assertIn(b"Camera preview", urlopen(self.base).read())
        self.assertEqual(urlopen(self.base + "/api/frame.jpg").headers.get_content_type(), "image/jpeg")
        self.assertTrue(json.loads(urlopen(self.base + "/api/state").read())["camera"]["ready"])
        request = Request(self.base + "/api/inspect", method="POST",
                          data=json.dumps({"part_id": "SCREEN-001", "confirmed": True,
                                           "source_kind": "screen_photo"}).encode(),
                          headers={"Origin": self.base, "Content-Type": "application/json"})
        self.assertEqual(urlopen(request).status, 202)
        self.assertEqual(self.service.requests, [("SCREEN-001", "screen_photo")])

    def test_cross_origin_and_missing_declaration_are_rejected(self):
        for origin, confirmed, code in [("http://elsewhere.invalid", True, 403),
                                        (self.base, False, 409)]:
            with self.subTest(origin=origin, confirmed=confirmed):
                request = Request(self.base + "/api/inspect", method="POST",
                                  data=json.dumps({"part_id": "NUT-002", "confirmed": confirmed,
                                                   "source_kind": "physical_nut"}).encode(),
                                  headers={"Origin": origin, "Content-Type": "application/json"})
                with self.assertRaises(HTTPError) as raised:
                    urlopen(request)
                self.assertEqual(raised.exception.code, code)
        self.assertEqual(self.service.requests, [])

    def test_original_png_upload_route_and_size_guard(self):
        page = urlopen(self.base).read()
        self.assertNotIn(b"Test an original MVTec defect image", page)
        self.assertIn(b"Upload your own test PNG", page)
        request = Request(self.base + "/api/replay", method="POST", data=b"png-fixture",
                          headers={"Origin": self.base, "Content-Type": "image/png",
                                   "X-Replay-Id": "BENT-001",
                                   "X-Dataset-Path": "test/bent/017.png"})
        self.assertEqual(urlopen(request).status, 202)
        self.assertEqual(self.service.requests[-1],
                         ("BENT-001", "test/bent/017.png", b"png-fixture"))
        request = Request(self.base + "/api/replay", method="POST", data=b"wrong-origin",
                          headers={"Origin": "http://elsewhere.invalid", "Content-Type": "image/png"})
        with self.assertRaises(HTTPError) as rejected:
            urlopen(request)
        self.assertEqual(rejected.exception.code, 403)
        request = Request(self.base + "/api/replay", method="POST", data=b"wrong-type",
                          headers={"Origin": self.base, "Content-Type": "application/octet-stream"})
        with self.assertRaises(HTTPError) as rejected:
            urlopen(request)
        self.assertEqual(rejected.exception.code, 409)
        self.assertEqual(len(self.service.requests), 1)

    def test_jetson_local_replay_route_needs_no_file_upload(self):
        self.assertNotIn(b"Analyze Jetson copy", urlopen(self.base).read())
        request = Request(self.base + "/api/replay-local", method="POST",
                          data=json.dumps({"replay_id": "BENT-LOCAL-001",
                                           "dataset_path": "test/bent/017.png"}).encode(),
                          headers={"Origin": self.base, "Content-Type": "application/json"})
        self.assertEqual(urlopen(request).status, 202)
        self.assertEqual(self.service.requests[-1],
                         ("local", "BENT-LOCAL-001", "test/bent/017.png"))
        request = Request(self.base + "/api/replay-local", method="POST",
                          data=b"{}", headers={"Origin": "http://elsewhere.invalid",
                                                 "Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as rejected:
            urlopen(request)
        self.assertEqual(rejected.exception.code, 403)

    def test_own_png_diagnostic_route_accepts_binary_upload(self):
        self.assertIn(b"Upload your own test PNG", urlopen(self.base).read())
        request = Request(self.base + "/api/upload-diagnostic", method="POST",
                          data=b"png-fixture", headers={"Origin": self.base,
                          "Content-Type": "image/png", "X-Upload-Id": "USER-001"})
        self.assertEqual(urlopen(request).status, 202)
        self.assertEqual(self.service.requests[-1],
                         ("diagnostic", "USER-001", b"png-fixture", False))
        benchmark = Request(self.base + "/api/upload-diagnostic", method="POST",
                            data=b"benchmark-fixture", headers={"Origin": self.base,
                            "Content-Type": "image/png", "X-Upload-Id": "BENCH-001",
                            "X-Mvtec-Benchmark": "yes"})
        self.assertEqual(urlopen(benchmark).status, 202)
        self.assertEqual(self.service.requests[-1],
                         ("diagnostic", "BENCH-001", b"benchmark-fixture", True))
        request = Request(self.base + "/api/upload-diagnostic", method="POST",
                          data=b"forbidden", headers={"Origin": "http://elsewhere.invalid",
                          "Content-Type": "image/png", "X-Upload-Id": "USER-002"})
        with self.assertRaises(HTTPError) as rejected:
            urlopen(request)
        self.assertEqual(rejected.exception.code, 403)

    def test_same_origin_advice_request_is_separate_from_capture(self):
        self.assertIn(b"Generate Llama advice", urlopen(self.base).read())
        request = Request(self.base + "/api/advice", method="POST", data=b"",
                          headers={"Origin": self.base})
        self.assertEqual(urlopen(request).status, 202)
        self.assertEqual(self.service.requests[-1], ("advice",))
        request = Request(self.base + "/api/advice", method="POST", data=b"x",
                          headers={"Origin": self.base})
        with self.assertRaises(HTTPError) as rejected:
            urlopen(request)
        self.assertEqual(rejected.exception.code, 400)


class ReplayIntakeTests(unittest.TestCase):
    def test_only_exact_manifest_preview_bytes_are_accepted(self):
        with TemporaryDirectory() as folder:
            service = LiveService.__new__(LiveService)
            service.args = SimpleNamespace(runs_root=Path(folder))
            service.replay_samples = {"test/bent/017.png": {
                "sha256": hashlib.sha256(b"original-png-fixture").hexdigest()}}
            service.lock = Lock()
            service.busy = False
            service.latest = None
            service.store = SimpleNamespace(has_part=lambda _part_id: False)
            for path, content in [("test/bent/017.png", b"modified"),
                                  ("test/bent/000.png", b"original-png-fixture")]:
                with self.subTest(path=path):
                    with self.assertRaises(ValueError):
                        service.replay("BENT-001", path, content)
            self.assertEqual(list(Path(folder).iterdir()), [])
            with patch("forgesight.jetson_live_ui.Thread") as worker:
                service.replay("BENT-001", "test/bent/017.png", b"original-png-fixture")
            worker.return_value.start.assert_called_once()
            saved = list(Path(folder).glob("replay_*/upload/test/bent/017.png"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].read_bytes(), b"original-png-fixture")
            self.assertEqual(service.latest["source_mode"], "SAVED_IMAGE_REPLAY")

    def test_local_replay_uses_only_checked_jetson_file(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / "dataset"
            image = dataset / "test/bent/017.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"original-png-fixture")
            service = LiveService.__new__(LiveService)
            service.args = SimpleNamespace(runs_root=root / "runs", dataset_root=dataset)
            service.replay_samples = {"test/bent/017.png": {
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}}
            service.local_replay_samples = ["test/bent/017.png"]
            service.lock = Lock()
            service.busy = False
            service.latest = None
            service.store = SimpleNamespace(has_part=lambda _part_id: False)
            with self.assertRaises(ValueError):
                service.replay_local("LOCAL-001", "test/bent/000.png")
            image.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                service.replay_local("LOCAL-001", "test/bent/017.png")
            image.write_bytes(b"original-png-fixture")
            with patch("forgesight.jetson_live_ui.Thread") as worker:
                service.replay_local("LOCAL-001", "test/bent/017.png")
            worker.return_value.start.assert_called_once()
            self.assertEqual(len(list((root / "runs").glob("replay_*/upload/test/bent/017.png"))), 1)

    def test_user_png_diagnostic_accepts_non_manifest_rgb_png(self):
        with TemporaryDirectory() as folder:
            service = LiveService.__new__(LiveService)
            service.args = SimpleNamespace(runs_root=Path(folder))
            service.lock = Lock()
            service.busy = False
            service.latest = None
            service.store = SimpleNamespace(has_part=lambda _part_id: False)
            png = BytesIO()
            Image.new("RGB", (700, 700), "gray").save(png, format="PNG")
            with self.assertRaises(ValueError):
                service.upload_diagnostic("USER-001", b"not-png")
            with patch("forgesight.jetson_live_ui.Thread") as worker:
                service.upload_diagnostic("USER-001", png.getvalue())
            worker.return_value.start.assert_called_once()
            saved = list(Path(folder).glob("user_upload_*/upload.png"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].read_bytes(), png.getvalue())
            self.assertEqual(service.latest["source_mode"], "USER_UPLOADED_IMAGE_DIAGNOSTIC")

    def test_benchmark_upload_requires_original_size_and_declares_source(self):
        with TemporaryDirectory() as folder:
            service = LiveService.__new__(LiveService)
            service.args = SimpleNamespace(runs_root=Path(folder))
            service.lock = Lock()
            service.busy = False
            service.latest = None
            service.store = SimpleNamespace(has_part=lambda _part_id: False)
            small = BytesIO()
            Image.new("RGB", (640, 480), "gray").save(small, format="PNG")
            with self.assertRaises(ValueError):
                service.upload_diagnostic("BENCH-001", small.getvalue(), True)
            full = BytesIO()
            Image.new("RGB", (700, 700), "gray").save(full, format="PNG")
            with patch("forgesight.jetson_live_ui.Thread") as worker:
                service.upload_diagnostic("BENCH-001", full.getvalue(), True)
            worker.return_value.start.assert_called_once()
            self.assertEqual(service.latest["source_mode"], "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD")


if __name__ == "__main__":
    unittest.main()
