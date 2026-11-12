"""Loopback live C270 preview and one-part EfficientAD inspection UI for Jetson."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Lock, Thread
import time
from uuid import uuid4

from forgesight.storage import InspectionStore
from forgesight.vision_edge_live_once import MODEL_SHA256, sha256


WEB_ROOT = Path(__file__).with_name("web")
PART_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
DATASET_MANIFEST_SHA256 = "14b11ebd03d5107c7ef3478e1c4d20887237e2fbef58601ac043014b8f0e340f"


def vision_executable(path):
    """Make the venv path absolute without following its Python symlinks."""
    return Path(os.path.abspath(path))


class CameraFeed:
    def __init__(self, index=0):
        self.index = index
        self.lock = Lock()
        self.stop = Event()
        self.frame = None
        self.jpeg = None
        self.updated = 0.0
        self.error = "Starting camera"
        self.thread = Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)

    def snapshot(self):
        with self.lock:
            if self.frame is None or time.monotonic() - self.updated > 2:
                raise RuntimeError(self.error or "Camera frame is stale")
            return self.frame.copy()

    def preview(self):
        with self.lock:
            if self.jpeg is None or time.monotonic() - self.updated > 2:
                return None
            return self.jpeg

    def state(self):
        with self.lock:
            ready = self.jpeg is not None and time.monotonic() - self.updated <= 2
            return {"ready": ready, "message": "Live C270 preview" if ready else self.error}

    def _loop(self):
        import cv2

        while not self.stop.is_set():
            capture = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            try:
                if not capture.isOpened():
                    raise RuntimeError("Camera could not open; check /dev/video0 and USB connection")
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_FPS, 30)
                next_encode = 0.0
                while not self.stop.is_set():
                    good, frame = capture.read()
                    if not good or frame is None or frame.shape != (480, 640, 3):
                        raise RuntimeError("Camera stopped delivering 640x480 color frames")
                    if time.monotonic() < next_encode:
                        continue
                    next_encode = time.monotonic() + 0.20
                    if float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) < 20.0:
                        with self.lock:
                            self.error = "Camera exposure is settling or scene is too dark"
                            self.jpeg = None
                            self.frame = None
                        continue
                    good, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not good:
                        raise RuntimeError("Camera JPEG encoding failed")
                    with self.lock:
                        self.frame = frame.copy()
                        self.jpeg = encoded.tobytes()
                        self.updated = time.monotonic()
                        self.error = ""
            except Exception as error:
                with self.lock:
                    self.frame = None
                    self.jpeg = None
                    self.error = str(error)
            finally:
                capture.release()
            self.stop.wait(1)


class LiveService:
    def __init__(self, args, camera):
        self.args = args
        self.camera = camera
        if not args.vision_python.is_file() or sha256(args.model) != MODEL_SHA256:
            raise ValueError("Jetson vision runtime or frozen model is missing/mismatched")
        if not args.grade_policy.is_file():
            raise ValueError("ForgeGrade policy is missing")
        if sha256(args.dataset_manifest) != DATASET_MANIFEST_SHA256:
            raise ValueError("Frozen MVTec manifest is missing or mismatched")
        manifest = json.loads(args.dataset_manifest.read_text())
        self.replay_samples = {row["path"]: row for row in manifest["samples"]
                               if row["role"] == "development_preview"}
        self.store = InspectionStore(args.database)
        self.lock = Lock()
        self.busy = False
        self.latest = None
        self.latest_root = None
        self.latest_source = None

    def state(self):
        with self.lock:
            latest = self.latest
            busy = self.busy
        return {"camera": self.camera.state(), "busy": busy, "latest": latest,
                "replay_samples": sorted(self.replay_samples),
                "history": [{"part_id": r["part_id"], "inspection_id": r["inspection_id"],
                             "source_mode": r["source"]["mode"],
                             "grade": r["decision"]["grade"], "status": r["status"],
                             "timestamp_utc": r["timestamp_utc"]}
                            for r in self.store.history(limit=10)]}

    def image(self, name):
        with self.lock:
            root = self.latest_root
            source = self.latest_source
        if root is None:
            return None
        path = root / "response_overlay.png" if name == "overlay" else source
        return path.read_bytes() if path.is_file() else None

    def replay(self, replay_id, dataset_path, data):
        if not isinstance(replay_id, str) or not PART_ID.fullmatch(replay_id):
            raise ValueError("Enter a new replay ID using letters, numbers, - or _")
        sample = self.replay_samples.get(dataset_path)
        if sample is None or not isinstance(data, bytes) or not 0 < len(data) <= MAX_UPLOAD_BYTES:
            raise ValueError("Choose one permitted MVTec development-preview PNG")
        if hashlib.sha256(data).hexdigest() != sample["sha256"]:
            raise ValueError("Uploaded file does not match the selected original MVTec PNG")
        with self.lock:
            if self.busy:
                raise ValueError("An inspection is already running")
            if self.store.has_part(replay_id):
                raise ValueError("This replay ID is already recorded; enter a new one")
            root = self.args.runs_root / ("replay_" + uuid4().hex)
            image = root / "upload" / dataset_path
            image.parent.mkdir(parents=True, exist_ok=False)
            with image.open("xb") as stream:
                stream.write(data)
            self.busy = True
            self.latest = {"status": "PROCESSING", "part_id": replay_id,
                           "source_mode": "SAVED_IMAGE_REPLAY"}
        Thread(target=self._run_replay, args=(replay_id, image, sample, root), daemon=True).start()

    def _run_replay(self, replay_id, image, sample, root):
        try:
            command = [sys.executable, "-B", "-m", "forgesight.vision_edge_replay",
                       "--image", str(image), "--image-sha256", sample["sha256"],
                       "--dataset-manifest", str(self.args.dataset_manifest),
                       "--part-id", replay_id,
                       "--vision-python", str(self.args.vision_python),
                       "--model", str(self.args.model), "--model-sha256", MODEL_SHA256,
                       "--grade-policy", str(self.args.grade_policy),
                       "--database", str(self.args.database),
                       "--output-dir", str(root / "result")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self._publish_result(replay_id, root / "result", image, result)
        except Exception as error:
            with self.lock:
                self.latest_root = root
                self.latest_source = image
                self.latest = {"status": "ERROR", "part_id": replay_id, "error": str(error)}
        finally:
            with self.lock:
                self.busy = False

    def _publish_result(self, part_id, result_root, image, result):
        record_path = result_root / "record.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text())
            if self.store.get_inspection(record["inspection_id"]) != record:
                raise RuntimeError("Inspection record was not confirmed in SQLite")
            if result.returncode and record["status"] == "ANALYZED":
                raise RuntimeError("Inspection process failed after writing an analyzed record")
            latest = {"status": record["status"], "part_id": part_id,
                      "source_mode": record["source"]["mode"],
                      "dataset_path": record["source"].get("dataset_path"),
                      "inspection_id": record["inspection_id"], "grade": record["decision"]["grade"],
                      "reason": record["decision"]["reason"],
                      "action": record["decision"]["operator_action"],
                      "score": record["score"], "candidate": record["candidate"],
                      "threshold": record["model"]["image_threshold"],
                      "response_area_px": (record["geometry"] or {}).get("total_area_px"),
                      "source_area_px": record["source"]["size_px"][0] * record["source"]["size_px"][1],
                      "model_forward_ms": record["timings"].get("model_forward_ms"),
                      "inspection_ms": (record["timings"].get("capture_file_to_record_ms")
                                        or record["timings"].get("replay_before_record_write_ms")),
                      "error": record["error"]}
        else:
            latest = {"status": "ERROR", "part_id": part_id,
                      "error": (result.stderr or result.stdout)[-1000:] or "Inspection process failed"}
        with self.lock:
            self.latest_root = result_root
            self.latest_source = image
            self.latest = latest

    def inspect(self, part_id, confirmed, source_kind):
        if (not isinstance(part_id, str) or not PART_ID.fullmatch(part_id)
                or confirmed is not True or source_kind not in ("physical_nut", "screen_photo")):
            raise ValueError("Enter a new capture ID, choose the source, and confirm it is in view")
        with self.lock:
            if self.busy:
                raise ValueError("An inspection is already running")
            if self.store.has_part(part_id):
                raise ValueError("This part ID was already inspected; use a new ID only for a distinct part")
            frame = self.camera.snapshot()
            self.busy = True
            self.latest = {"status": "PROCESSING", "part_id": part_id}
        Thread(target=self._run, args=(part_id, frame, source_kind), daemon=True).start()

    def _run(self, part_id, frame, source_kind):
        import cv2

        root = self.args.runs_root / ("part_" + uuid4().hex)
        try:
            root.mkdir(parents=True, exist_ok=False)
            image = root / "frame.png"
            good, encoded = cv2.imencode(".png", frame)
            if not good:
                raise RuntimeError("Camera frame could not be saved")
            with image.open("xb") as stream:
                stream.write(encoded.tobytes())
            command = [sys.executable, "-B", "-m", "forgesight.vision_edge_live_once",
                       "--image", str(image), "--part-id", part_id,
                       "--vision-python", str(self.args.vision_python),
                       "--model", str(self.args.model), "--model-sha256", MODEL_SHA256,
                       "--grade-policy", str(self.args.grade_policy),
                       "--database", str(self.args.database),
                       "--output-dir", str(root / "result"),
                       "--camera-index", str(self.args.camera_index),
                       "--operator-declared-part" if source_kind == "physical_nut" else "--screen-photo-diagnostic"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self._publish_result(part_id, root / "result", image, result)
        except Exception as error:
            with self.lock:
                self.latest_root = root
                self.latest_source = root / "frame.png"
                self.latest = {"status": "ERROR", "part_id": part_id, "error": str(error)}
        finally:
            with self.lock:
                self.busy = False


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def log_request(self, code='-', size='-'):
            if not isinstance(code, int) or code >= 400:
                super().log_request(code, size)

        def send(self, status, content, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def allowed_host(self):
            return self.headers.get("Host") in ("127.0.0.1:" + str(self.server.server_port),
                                                "localhost:" + str(self.server.server_port))

        def do_GET(self):
            if not self.allowed_host():
                self.send(403, b"Forbidden", "text/plain")
                return
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                self.send(200, json.dumps(service.state(), allow_nan=False).encode(), "application/json")
            elif path == "/api/frame.jpg":
                frame = service.camera.preview()
                self.send(200 if frame else 503, frame or b"Camera unavailable", "image/jpeg" if frame else "text/plain")
            elif path in ("/api/overlay.png", "/api/source.png"):
                data = service.image("overlay" if path.endswith("overlay.png") else "source")
                self.send(200 if data else 404, data or b"Image unavailable", "image/png" if data else "text/plain")
            elif path in ("/", "/app.js", "/style.css"):
                file = {"/": "jetson_live.html", "/app.js": "jetson_live.js",
                        "/style.css": "jetson_live.css"}[path]
                mime = {"/": "text/html", "/app.js": "application/javascript",
                        "/style.css": "text/css"}[path]
                self.send(200, (WEB_ROOT / file).read_bytes(), mime + "; charset=utf-8")
            else:
                self.send(404, b"Not found", "text/plain")

        def do_POST(self):
            if (not self.allowed_host() or self.headers.get("Origin") !=
                    "http://" + self.headers.get("Host", "")):
                self.send(403, b"Forbidden", "text/plain")
                return
            if self.path not in ("/api/inspect", "/api/replay"):
                self.send(400, b"Invalid request", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if self.path == "/api/inspect":
                    if self.headers.get("Content-Type") != "application/json" or not 0 < length <= 512:
                        raise ValueError("Invalid camera request")
                    body = json.loads(self.rfile.read(length))
                    service.inspect(body.get("part_id"), body.get("confirmed"), body.get("source_kind"))
                else:
                    if self.headers.get("Content-Type") != "image/png" or not 0 < length <= MAX_UPLOAD_BYTES:
                        raise ValueError("Upload one original PNG under 8 MiB")
                    service.replay(self.headers.get("X-Replay-Id"),
                                   self.headers.get("X-Dataset-Path"), self.rfile.read(length))
            except (ValueError, RuntimeError, TypeError, AttributeError) as error:
                self.send(409, json.dumps({"error": str(error)}).encode(), "application/json")
                return
            self.send(202, b'{"status":"PROCESSING"}', "application/json")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-python", type=Path, default=Path("venv/bin/python"))
    parser.add_argument("--model", type=Path, default=Path("efficientad_s_calibrated.ts"))
    parser.add_argument("--grade-policy", type=Path, default=Path("configs/forgegrade_v1.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("manifests/metal_nut_v1.json"))
    parser.add_argument("--database", type=Path, default=Path("runs/live_inspections_v1/inspections.sqlite3"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/live_inspections_v1"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged port")
    args.database = args.database.resolve()
    args.runs_root = args.runs_root.resolve()
    args.vision_python = vision_executable(args.vision_python)
    args.model = args.model.resolve()
    args.grade_policy = args.grade_policy.resolve()
    args.dataset_manifest = args.dataset_manifest.resolve()
    camera = CameraFeed(args.camera_index)
    service = LiveService(args, camera)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(service))
    camera.start()
    try:
        print("ForgeSight live camera UI: http://127.0.0.1:%d" % args.port, flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        camera.close()


if __name__ == "__main__":
    main()
