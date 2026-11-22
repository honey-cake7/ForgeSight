"""Exercise the loopback inspection contract without claiming camera proof."""

from http.server import ThreadingHTTPServer
import hashlib
import logging
from io import BytesIO
import json
from pathlib import Path
from threading import Lock, Thread
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from forgesight.jetson_live_ui import CameraFeed, LiveService, handler_for, vision_executable
from forgesight.storage import InspectionStore


class StubCamera:
    def __init__(self):
        self.running = True
        self.calls = []
        self.fail_controls = False

    def preview(self):
        return b"\xff\xd8\xff\xd9"

    def state(self):
        return {"ready": self.running, "running": self.running,
                "message": "Live test preview" if self.running else "Camera stopped by operator"}

    def turn_on(self):
        if self.fail_controls:
            raise RuntimeError("test control failure")
        self.calls.append("start")
        self.running = True

    def turn_off(self):
        if self.fail_controls:
            raise RuntimeError("test control failure")
        self.calls.append("stop")
        self.running = False


class StubService:
    def __init__(self):
        self.camera = StubCamera()
        self.requests = []

    def state(self):
        return {"camera": self.camera.state(),
                "busy": False, "latest": None, "history": [],
                "advice": {"status": "NOT_ELIGIBLE", "advisory": None},
                "replay_samples": ["test/bent/017.png"],
                "local_replay_samples": ["test/bent/017.png"]}

    def inspection_image(self, inspection_id, name):
        return 404, None

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

    def manual_review(self, inspection_id, actor, outcome, reason):
        self.requests.append(("review", inspection_id, actor, outcome, reason))
        return {"verification_status": "UNVERIFIED_OPERATOR_CLAIM"}


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

    def test_camera_controls_change_server_state_and_require_same_origin(self):
        page = urlopen(self.base).read()
        self.assertIn(b'id="camera-start"', page)
        self.assertIn(b'id="camera-stop"', page)
        self.assertTrue(json.loads(urlopen(self.base + "/api/state").read())["camera"]["running"])

        stop = Request(self.base + "/api/camera/stop", method="POST", data=b"",
                       headers={"Origin": self.base})
        response = urlopen(stop)
        self.assertEqual(response.status, 200)
        self.assertFalse(json.loads(response.read())["running"])
        self.assertEqual(self.service.camera.calls, ["stop"])

        start = Request(self.base + "/api/camera/start", method="POST", data=b"",
                        headers={"Origin": self.base})
        self.assertTrue(json.loads(urlopen(start).read())["running"])
        self.assertEqual(self.service.camera.calls, ["stop", "start"])

        cross_origin = Request(self.base + "/api/camera/stop", method="POST", data=b"",
                               headers={"Origin": "http://elsewhere.invalid"})
        with self.assertRaises(HTTPError) as rejected:
            urlopen(cross_origin)
        self.assertEqual(rejected.exception.code, 403)
        self.assertTrue(self.service.camera.running)

        invalid_body = Request(self.base + "/api/camera/stop", method="POST", data=b"x",
                               headers={"Origin": self.base})
        with self.assertRaises(HTTPError) as rejected:
            urlopen(invalid_body)
        self.assertEqual(rejected.exception.code, 400)

        self.service.camera.fail_controls = True
        failed = Request(self.base + "/api/camera/stop", method="POST", data=b"",
                         headers={"Origin": self.base})
        with self.assertRaises(HTTPError) as rejected:
            urlopen(failed)
        self.assertEqual(rejected.exception.code, 503)
        self.assertIn(b"test control failure", rejected.exception.read())

    def test_camera_feed_stop_releases_and_start_reopens_capture(self):
        released = []

        class Frame:
            shape = (480, 640, 3)

            def copy(self):
                return self

        class Encoded:
            def tobytes(self):
                return b"jpeg"

        class Capture:
            def __init__(self, index, backend):
                self.was_released = False
                released.append(self)

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                time.sleep(0.005)
                return True, Frame()

            def release(self):
                self.was_released = True

        fake_cv2 = SimpleNamespace(
            CAP_V4L2=1, CAP_PROP_FOURCC=2, CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5, COLOR_BGR2GRAY=6,
            IMWRITE_JPEG_QUALITY=7, VideoCapture=Capture,
            VideoWriter_fourcc=lambda *args: 8,
            cvtColor=lambda frame, code: SimpleNamespace(mean=lambda: 200),
            imencode=lambda *args: (True, Encoded()))
        camera = CameraFeed()

        def wait_for(predicate):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if predicate():
                    return True
                time.sleep(0.01)
            return False

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            camera.start()
            try:
                self.assertTrue(wait_for(lambda: camera.state()["ready"]))
                camera.turn_off()
                self.assertTrue(released[0].was_released)
                self.assertFalse(camera.state()["running"])

                camera.turn_on()
                self.assertTrue(wait_for(lambda: len(released) == 2 and camera.state()["ready"]))
                camera.turn_off()
                self.assertTrue(released[1].was_released)
            finally:
                camera.close()

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
        self.assertIn(b'id="diagnostic-file"', page)
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
        self.assertIn(b'id="diagnostic-file"', urlopen(self.base).read())
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

    def test_manual_review_route_requires_same_origin_and_bounded_json(self):
        self.assertIn(b'id="review-submit"', urlopen(self.base).read())
        body = json.dumps({"inspection_id": "fixture", "actor": "Operator-1",
                           "outcome": "visible_issue", "reason": "Visible response"}).encode()
        request = Request(self.base + "/api/manual-review", method="POST", data=body,
                          headers={"Origin": self.base, "Content-Type": "application/json"})
        self.assertEqual(urlopen(request).status, 201)
        self.assertEqual(self.service.requests[-1],
                         ("review", "fixture", "Operator-1", "visible_issue", "Visible response"))
        request = Request(self.base + "/api/manual-review", method="POST", data=body,
                          headers={"Origin": "http://elsewhere.invalid", "Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as rejected:
            urlopen(request)
        self.assertEqual(rejected.exception.code, 403)


class RefusalSupportIdTests(unittest.TestCase):
    """A refused/failed POST carries a support ID the operation log repeats."""

    def setUp(self):
        from forgesight.operation_log import operation_logger
        self.logger = operation_logger()
        self.records = []
        handler = logging.Handler()
        handler.emit = self.records.append
        self.logger.addHandler(handler)
        self.addCleanup(self.logger.removeHandler, handler)
        self.service = StubService()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.service))
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.base = "http://127.0.0.1:%d" % self.server.server_port

    def post(self, path, payload):
        request = Request(self.base + path, method="POST",
                          data=json.dumps(payload).encode(),
                          headers={"Origin": self.base,
                                   "Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)
        return raised.exception

    def test_refusal_body_carries_support_id_matching_the_log(self):
        error = self.post("/api/inspect", {"part_id": "NUT-002", "confirmed": False,
                                           "source_kind": "physical_nut"})
        self.assertEqual(error.code, 409)
        body = json.loads(error.read())
        self.assertIn("Operator declaration required", body["error"])
        support_id = body["support_id"]
        self.assertRegex(support_id, r"^[0-9a-f]{8}$")
        logged = [r.getMessage() for r in self.records
                  if "support_id=" + support_id in r.getMessage()]
        self.assertTrue(logged)
        self.assertIn("/api/inspect", logged[0])

    def test_internal_failure_is_500_generic_with_support_id(self):
        def broken(*args):
            raise OSError("disk full while writing token=secret99 payload")
        self.service.inspect = broken
        error = self.post("/api/inspect", {"part_id": "NUT-002", "confirmed": True,
                                           "source_kind": "screen_photo"})
        self.assertEqual(error.code, 500)
        body = json.loads(error.read())
        # No internals and no secret-shaped content reach the client.
        self.assertNotIn("disk full", body["error"])
        self.assertNotIn("secret99", json.dumps(body))
        support_id = body["support_id"]
        self.assertRegex(support_id, r"^[0-9a-f]{8}$")
        logged = [r for r in self.records
                  if "support_id=" + support_id in r.getMessage()]
        self.assertTrue(logged)
        self.assertEqual(logged[0].levelno, logging.ERROR)
        self.assertIn("/api/inspect", logged[0].getMessage())
        self.assertNotIn("?", logged[0].getMessage())
        # The log line has the detail, scrubbed.
        self.assertIn("disk full", logged[0].getMessage())
        self.assertNotIn("secret99", logged[0].getMessage())

    def test_sqlite_failure_is_500_not_a_dropped_connection(self):
        import sqlite3
        def broken(*args):
            raise sqlite3.OperationalError("database is locked")
        self.service.inspect = broken
        error = self.post("/api/inspect", {"part_id": "NUT-003", "confirmed": True,
                                           "source_kind": "screen_photo"})
        self.assertEqual(error.code, 500)
        body = json.loads(error.read())
        self.assertRegex(body.get("support_id", ""), r"^[0-9a-f]{8}$")

    def test_advice_refusal_carries_support_id(self):
        def refuse():
            raise ValueError("Llama advice is available after a flagged result")
        self.service.request_advice = refuse
        request = Request(self.base + "/api/advice", method="POST", data=b"",
                          headers={"Origin": self.base, "Content-Length": "0"})
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)
        self.assertEqual(raised.exception.code, 409)
        body = json.loads(raised.exception.read())
        self.assertRegex(body.get("support_id", ""), r"^[0-9a-f]{8}$")


    def test_advice_internal_type_error_is_500_not_disconnect(self):
        def broken():
            raise TypeError("unexpected None in advice path")
        self.service.request_advice = broken
        request = Request(self.base + "/api/advice", method="POST", data=b"",
                          headers={"Origin": self.base, "Content-Length": "0"})
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)
        self.assertEqual(raised.exception.code, 500)
        body = json.loads(raised.exception.read())
        self.assertRegex(body.get("support_id", ""), r"^[0-9a-f]{8}$")
        self.assertNotIn("None", body["error"])


class ManualReviewStoreTests(unittest.TestCase):
    def test_operator_findings_are_append_only_claims_and_do_not_change_grade(self):
        with TemporaryDirectory() as folder:
            store = InspectionStore(Path(folder) / "inspections.sqlite3")
            original = json.loads(Path("manifests/graded_inspection_replay_v1.json").read_text())["records"][0]
            store.save_inspection(original)
            service = LiveService.__new__(LiveService)
            service.store = store
            service.lock = Lock()
            service.busy = False
            service.latest = {"inspection_id": original["inspection_id"], "status": "ANALYZED"}
            for outcome, grade in (("visible_issue", "REJECT"), ("uncertain_or_recapture", "REVIEW")):
                review = service.manual_review(original["inspection_id"], "Operator-1", outcome,
                                               "Checked the saved original and overlay")
                self.assertEqual(review["proposed_grade"], grade)
                self.assertEqual(review["verification_status"], "UNVERIFIED_OPERATOR_CLAIM")
            self.assertEqual(len(store.corrections(original["inspection_id"])), 2)
            self.assertEqual(store.get_inspection(original["inspection_id"]), original)
            with self.assertRaises(ValueError):
                service.manual_review("stale-id", "Operator-1", "visible_issue", "Checked image")
            with self.assertRaises(ValueError):
                service.manual_review(original["inspection_id"], "Operator-1", "fabricated", "Checked image")


class MediaBindingTests(unittest.TestCase):
    """Evidence is served by inspection ID from the stored record, with digests."""

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.store = InspectionStore(self.root / "inspections.sqlite3")
        records = json.loads(
            Path("manifests/graded_inspection_replay_v1.json").read_text())["records"]
        self.ids = []
        for index, tag in ((0, "a"), (1, "b")):
            record = json.loads(json.dumps(records[index]))
            folder = self.runs / tag
            folder.mkdir()
            overlay = folder / "response_overlay.png"
            overlay.write_bytes(tag.encode() * 100)
            source = folder / "source.png"
            source.write_bytes(("source-" + tag).encode() * 100)
            record["artifacts"]["response_overlay"] = str(overlay)
            record["artifacts"]["response_overlay_sha256"] = hashlib.sha256(
                overlay.read_bytes()).hexdigest()
            record["source"]["path"] = str(source)
            record["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            self.store.save_inspection(record)
            self.ids.append(record["inspection_id"])
        service = LiveService.__new__(LiveService)
        service.store = self.store
        service.args = SimpleNamespace(runs_root=self.runs)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(service))
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def get(self, path):
        try:
            response = urlopen(self.base + path)
            return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def test_media_is_bound_to_the_requested_inspection(self):
        for index, tag in ((0, "a"), (1, "b")):
            status, body = self.get(f"/api/inspection/{self.ids[index]}/overlay.png?t=1")
            self.assertEqual((status, body), (200, tag.encode() * 100))
            status, body = self.get(f"/api/inspection/{self.ids[index]}/source.png")
            self.assertEqual((status, body), (200, ("source-" + tag).encode() * 100))

    def test_changed_evidence_is_refused_not_served(self):
        overlay = self.runs / "a" / "response_overlay.png"
        overlay.write_bytes(b"tampered")
        status, body = self.get(f"/api/inspection/{self.ids[0]}/overlay.png")
        self.assertEqual(status, 409)
        self.assertIn(b"changed", body)
        # The other inspection's untouched evidence still serves.
        status, _ = self.get(f"/api/inspection/{self.ids[1]}/overlay.png")
        self.assertEqual(status, 200)

    def test_non_canonical_uuid_forms_are_404_not_a_dropped_connection(self):
        # 32-hex-without-hyphens and braced forms parse as UUIDs, but the
        # store keys on the canonical hyphenated form. Serving must 404,
        # never raise through the HTTP layer (which drops the connection).
        canonical = self.ids[0]
        for variant in (canonical.replace("-", ""),
                        "{" + canonical + "}",
                        canonical.upper()):
            with self.subTest(variant=variant):
                status, _ = self.get(f"/api/inspection/{variant}/source.png")
                self.assertEqual(status, 404)

    def test_unknown_malformed_and_out_of_root_references_are_not_found(self):
        status, _ = self.get("/api/inspection/00000000-0000-0000-0000-000000000000/source.png")
        self.assertEqual(status, 404)
        status, _ = self.get("/api/inspection/not-a-uuid/source.png")
        self.assertEqual(status, 404)
        status, _ = self.get("/api/overlay.png")
        self.assertEqual(status, 404)
        record = json.loads(json.dumps(
            self.store.get_inspection(self.ids[0])))
        record["inspection_id"] = "11111111-1111-1111-1111-111111111111"
        record["part_id"] = "ESCAPE-001"
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside")
        record["source"]["path"] = str(outside)
        record["source"]["sha256"] = hashlib.sha256(b"outside").hexdigest()
        self.store.save_inspection(record)
        status, _ = self.get(f"/api/inspection/{record['inspection_id']}/source.png")
        self.assertEqual(status, 404)


class ReplayIntakeTests(unittest.TestCase):
    def test_camera_and_upload_share_the_automatic_counter(self):
        with TemporaryDirectory() as folder:
            service = LiveService.__new__(LiveService)
            root = Path(folder)
            service.args = SimpleNamespace(runs_root=root / "runs")
            service.camera = SimpleNamespace(snapshot=lambda: object())
            service.store = InspectionStore(root / "inspections.sqlite3")
            service.lock = Lock()
            service.busy = False
            service.latest = None
            png = BytesIO()
            Image.new("RGB", (700, 700), "gray").save(png, format="PNG")
            with patch("forgesight.jetson_live_ui.Thread"):
                self.assertEqual(service.inspect(None, True, "screen_photo"), "INS-000001")
                with self.assertRaisesRegex(ValueError, "already running"):
                    service.upload_diagnostic(None, png.getvalue())
                service.busy = False
                self.assertEqual(service.upload_diagnostic(None, png.getvalue()), "INS-000002")
            self.assertEqual(service.latest["part_id"], "INS-000002")

    def test_early_worker_failure_is_saved_and_logged_without_secret(self):
        with TemporaryDirectory() as folder:
            service = LiveService.__new__(LiveService)
            root = Path(folder)
            service.args = SimpleNamespace(runs_root=root / "runs",
                                           grade_policy=Path("configs/forgegrade_benchmark_v1.json"))
            service.store = InspectionStore(root / "inspections.sqlite3")
            service.lock = Lock()
            service.busy = False
            service.latest = None
            png = BytesIO()
            Image.new("RGB", (64, 64), "gray").save(png, format="PNG")
            with patch.object(service, "_run_upload", side_effect=RuntimeError("Worker failed: TOKEN=private")), \
                    self.assertLogs("forgesight.operations", level="ERROR") as log:
                self.assertEqual(service.upload_diagnostic(None, png.getvalue()), "INS-000001")
                deadline = time.monotonic() + 5
                while service.busy and time.monotonic() < deadline:
                    time.sleep(0.01)
            self.assertFalse(service.busy)
            latest = service.latest
            record = service.store.get_inspection(latest["inspection_id"])
            self.assertEqual(record["status"], "ERROR")
            self.assertEqual(record["decision"]["grade"], "REVIEW")
            self.assertEqual(record["error"]["message"], "Worker failed: TOKEN=[redacted]")
            self.assertNotIn("private", json.dumps(latest))
            self.assertIn("support_id=" + latest["support_id"], " ".join(log.output))

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
