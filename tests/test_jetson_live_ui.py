"""Exercise the loopback inspection contract without claiming camera proof."""

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from forgesight.jetson_live_ui import handler_for, vision_executable


class StubCamera:
    def preview(self):
        return b"\xff\xd8\xff\xd9"


class StubService:
    def __init__(self):
        self.camera = StubCamera()
        self.requests = []

    def state(self):
        return {"camera": {"ready": True, "message": "Test frame"},
                "busy": False, "latest": None, "history": []}

    def image(self, name):
        return None

    def inspect(self, part_id, confirmed, source_kind):
        if confirmed is not True:
            raise ValueError("Operator declaration required")
        self.requests.append((part_id, source_kind))


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


if __name__ == "__main__":
    unittest.main()
