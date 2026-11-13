"""Loopback live C270 preview and one-part EfficientAD inspection UI for Jetson."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Lock, Thread
import time
from uuid import uuid4

from PIL import Image

from forgesight.assist_display import advice_for
from forgesight.assist_queue import AdvisoryQueue
from forgesight.storage import InspectionStore, new_correction
from forgesight.vision_edge_live_once import MODEL_SHA256, sha256


WEB_ROOT = Path(__file__).with_name("web")
PART_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
DATASET_MANIFEST_SHA256 = "14b11ebd03d5107c7ef3478e1c4d20887237e2fbef58601ac043014b8f0e340f"


def vision_executable(path):
    """Make the venv path absolute without following its Python symlinks."""
    return Path(os.path.abspath(path))


def launch_vision_worker(args, worker_script=None):
    """Start the persistent vision worker and wait for its READY line.

    stderr is drained on a daemon thread into a small ring buffer: a worker
    whose stderr pipe fills (torch warnings are chatty) would otherwise block
    mid-inference with no error. The buffer tail is included in boot errors.
    """
    from collections import deque
    script = worker_script or (Path(__file__).resolve().parent / "vision_edge_worker.py")
    command = [str(args.vision_python), "-B", str(script),
               "--model", str(args.model), "--model-sha256", MODEL_SHA256,
               "--socket", str(args.vision_worker_socket),
               "--threads", str(args.vision_worker_threads),
               "--idle-timeout-seconds", str(args.vision_worker_idle_timeout_seconds)]
    worker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_tail = deque(maxlen=50)

    def drain_stderr():
        for line in worker.stderr:
            stderr_tail.append(line.rstrip())

    Thread(target=drain_stderr, daemon=True).start()
    ready_holder = []

    def read_ready():
        ready_holder.append(worker.stdout.readline())

    reader = Thread(target=read_ready, daemon=True)
    reader.start()
    reader.join(timeout=120)
    ready = ready_holder[0] if ready_holder else ""
    try:
        status = json.loads(ready)
    except ValueError:
        status = {}
    if status.get("status") != "WORKER_READY":
        worker.terminate()
        detail = "; worker stderr tail: " + " | ".join(stderr_tail) if stderr_tail else ""
        raise ValueError(
            "Vision worker failed to start with the frozen model within 120 seconds" + detail)
    return worker


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
        self.local_replay_samples = sorted(
            path for path, row in self.replay_samples.items()
            if (args.dataset_root / path).is_file()
            and sha256(args.dataset_root / path) == row["sha256"])
        self.store = InspectionStore(args.database)
        self.vision_worker = None
        self.worker_respawn_lock = Lock()
        self.worker_respawning = False
        if args.vision_worker_socket is not None:
            self.vision_worker = launch_vision_worker(args)
        self.lock = Lock()
        self.busy = False
        self.latest = None
        self.latest_root = None
        self.latest_source = None
        self.advice_root = args.advice_root
        self.advice_queue = AdvisoryQueue()
        self.advice_futures = {}
        self.llama_ready = (args.guidance.is_file()
                            and (args.llama_root / "venv/bin/python").is_file()
                            and (args.llama_root / "model/llama3_2_1b_xnnpack.pte").is_file())

    def _inspect_inline(self, image, part_id, root, mode_flags):
        """Run the same inspection in this long-lived process; worker mode only.

        Identical gates and artifacts to the vision_edge_live_once.py subprocess,
        without paying a fresh interpreter plus numpy/Pillow import per image.
        """
        import argparse
        from forgesight import vision_edge_live_once as live_once
        args = argparse.Namespace(
            image=image, part_id=part_id,
            vision_python=self.args.vision_python, model=self.args.model,
            model_sha256=MODEL_SHA256, grade_policy=self.args.grade_policy,
            database=self.args.database, output_dir=root / "result",
            camera_index=self.args.camera_index,
            vision_worker_socket=self.args.vision_worker_socket,
            overlay_format=self.args.overlay_format,
            **mode_flags)
        record = live_once.inspect(args)

        class Done:
            returncode = 0 if record["status"] == "ANALYZED" else 1
            stdout = stderr = ""
        self._publish_result(part_id, root / "result", image, Done())
        if record["timings"].get("vision_mode") == "COLD_PROCESS_AFTER_WORKER_UNAVAILABLE":
            # The worker died (idle timeout or crash). Bring it back so the next
            # inspection is warm instead of silently running cold forever.
            self._respawn_vision_worker()

    def _respawn_vision_worker(self):
        with self.worker_respawn_lock:
            if self.worker_respawning:
                return
            self.worker_respawning = True

        def respawn():
            try:
                old = self.vision_worker
                if old is not None and old.poll() is None:
                    old.terminate()
                    try:
                        old.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        old.kill()
                self.vision_worker = launch_vision_worker(self.args)
            except Exception:
                # Next inspection cold-falls-back again and retries the respawn.
                pass
            finally:
                with self.worker_respawn_lock:
                    self.worker_respawning = False

        Thread(target=respawn, daemon=True).start()

    def close(self):
        self.advice_queue.close()
        if self.vision_worker is not None:
            self.vision_worker.terminate()
            try:
                self.vision_worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.vision_worker.kill()

    def advice_state(self):
        with self.lock:
            latest, root = self.latest, self.latest_root
            future = self.advice_futures.get(latest.get("inspection_id")) if latest else None
        if not self.llama_ready:
            return {"status": "NOT_CONNECTED", "advisory": None}
        if (not latest or latest.get("status") != "ANALYZED"
                or latest.get("candidate") != "ANOMALY_CANDIDATE"
                or latest.get("source_mode") not in
                ("SAVED_IMAGE_REPLAY", "USER_UPLOADED_IMAGE_DIAGNOSTIC",
                 "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD")
                or root is None):
            return {"status": "NOT_ELIGIBLE", "advisory": None}
        record_path = root / "record.json"
        if not record_path.is_file():
            return {"status": "ADVICE_UNAVAILABLE", "advisory": None}
        record = json.loads(record_path.read_text())
        advice = advice_for(record, self.advice_root)
        if future is not None and not future.done():
            return {"status": "PENDING", "advisory": None}
        if future is not None and advice["status"] in ("NOT_REQUESTED", "PENDING"):
            return {"status": "ADVICE_UNAVAILABLE", "advisory": None}
        return advice

    def request_advice(self):
        with self.lock:
            latest, root = self.latest, self.latest_root
            if (not self.llama_ready or self.busy or not latest
                    or latest.get("status") != "ANALYZED"
                    or latest.get("candidate") != "ANOMALY_CANDIDATE"
                    or latest.get("source_mode") not in
                    ("SAVED_IMAGE_REPLAY", "USER_UPLOADED_IMAGE_DIAGNOSTIC",
                     "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD")
                    or root is None):
                raise ValueError("Llama advice is available after a flagged saved-image result")
            record_path = root / "record.json"
            record = json.loads(record_path.read_text())
            if (record["inspection_id"] != latest["inspection_id"]
                    or self.store.get_inspection(record["inspection_id"]) != record):
                raise ValueError("Inspection record does not match SQLite")
            output = self.advice_root / record["inspection_id"]
            if output.exists() or record["inspection_id"] in self.advice_futures:
                return "ALREADY_REQUESTED"
            command = [sys.executable, "-B", "-m", "forgesight.assist_jetson_replay_once",
                       "--record", str(record_path), "--database", str(self.args.database),
                       "--guidance", str(self.args.guidance),
                       "--llama-root", str(self.args.llama_root),
                       "--output-dir", str(output), "--llama-timeout-seconds",
                       str(self.args.llama_timeout_seconds)]
            future = self.advice_queue.submit(lambda: subprocess.run(
                command, capture_output=True, text=True,
                timeout=self.args.llama_timeout_seconds + 45))
            if future is None:
                return "BUSY"
            self.advice_futures[record["inspection_id"]] = future
            return "PENDING"

    def state(self):
        with self.lock:
            latest = self.latest
            busy = self.busy
        reviews = (self.store.corrections(latest["inspection_id"])
                   if latest and latest.get("inspection_id") else [])
        return {"camera": self.camera.state(), "busy": busy, "latest": latest,
                "manual_reviews": reviews,
                "advice": self.advice_state(),
                "replay_samples": sorted(self.replay_samples),
                "local_replay_samples": self.local_replay_samples,
                "history": [{"part_id": r["part_id"], "inspection_id": r["inspection_id"],
                             "source_mode": r["source"]["mode"],
                             "grade": r["decision"]["grade"], "status": r["status"],
                             "timestamp_utc": r["timestamp_utc"]}
                            for r in self.store.history(limit=10)]}

    def manual_review(self, inspection_id, actor, outcome, reason):
        choices = {"no_visible_issue": "PASS", "visible_issue": "REJECT",
                   "uncertain_or_recapture": "REVIEW"}
        if (not isinstance(inspection_id, str) or not isinstance(actor, str)
                or not re.fullmatch(r"[A-Za-z0-9 _.-]{1,64}", actor)
                or not isinstance(reason, str) or not 3 <= len(reason.strip()) <= 500
                or outcome not in choices):
            raise ValueError("Choose a review finding and enter your name and a short observation")
        with self.lock:
            if (self.busy or not self.latest or self.latest.get("inspection_id") != inspection_id
                    or self.latest.get("status") != "ANALYZED"):
                raise ValueError("Review the latest completed inspection")
            original = self.store.get_inspection(inspection_id)
            if original is None:
                raise ValueError("Inspection record is missing")
            correction = new_correction(inspection_id, actor.strip(), reason.strip(), choices[outcome])
            correction["review_outcome"] = outcome
            self.store.save_correction(correction)
            return correction

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

    def replay_local(self, replay_id, dataset_path):
        if dataset_path not in self.local_replay_samples:
            raise ValueError("Selected original image is not available on the Jetson")
        image = self.args.dataset_root / dataset_path
        if not image.is_file() or image.stat().st_size > MAX_UPLOAD_BYTES:
            raise ValueError("Selected Jetson image is missing or too large")
        self.replay(replay_id, dataset_path, image.read_bytes())

    def upload_diagnostic(self, upload_id, data, benchmark=False):
        if type(benchmark) is not bool:
            raise ValueError("Invalid benchmark declaration")
        if not isinstance(upload_id, str) or not PART_ID.fullmatch(upload_id):
            raise ValueError("Enter a new upload ID using letters, numbers, - or _")
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_UPLOAD_BYTES:
            raise ValueError("Choose one PNG under 8 MiB")
        try:
            with Image.open(BytesIO(data)) as source:
                width, height = source.size
                if (source.format != "PNG" or source.mode != "RGB"
                        or not 64 <= width <= 1024 or not 64 <= height <= 1024):
                    raise ValueError("Expected an RGB PNG between 64 and 1024 pixels per side")
                if benchmark and (width, height) != (700, 700):
                    raise ValueError("MVTec benchmark mode requires an original-size 700x700 PNG")
                source.load()
        except OSError as error:
            raise ValueError("The selected file is not a readable PNG") from error
        with self.lock:
            if self.busy:
                raise ValueError("An inspection is already running")
            if self.store.has_part(upload_id):
                raise ValueError("This upload ID is already recorded; enter a new one")
            root = self.args.runs_root / ("user_upload_" + uuid4().hex)
            root.mkdir(parents=True, exist_ok=False)
            image = root / "upload.png"
            with image.open("xb") as stream:
                stream.write(data)
            self.busy = True
            mode = "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD" if benchmark else "USER_UPLOADED_IMAGE_DIAGNOSTIC"
            self.latest = {"status": "PROCESSING", "part_id": upload_id, "source_mode": mode}
        Thread(target=self._run_upload, args=(upload_id, image, root, benchmark), daemon=True).start()

    def _run_upload(self, upload_id, image, root, benchmark):
        try:
            if self.args.vision_worker_socket is not None:
                self._inspect_inline(image, upload_id, root,
                                     {"mvtec_benchmark_upload": benchmark,
                                      "uploaded_image_diagnostic": not benchmark,
                                      "operator_declared_part": False,
                                      "screen_photo_diagnostic": False,
                                      "diagnostic_no_part": False})
                return
            command = [sys.executable, "-B", "-m", "forgesight.vision_edge_live_once",
                       "--image", str(image), "--part-id", upload_id,
                       "--vision-python", str(self.args.vision_python),
                       "--model", str(self.args.model), "--model-sha256", MODEL_SHA256,
                       "--grade-policy", str(self.args.grade_policy),
                       "--database", str(self.args.database),
                       "--output-dir", str(root / "result"),
                       "--mvtec-benchmark-upload" if benchmark else "--uploaded-image-diagnostic"]
            if self.args.vision_worker_socket is not None:
                command += ["--vision-worker-socket", str(self.args.vision_worker_socket)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self._publish_result(upload_id, root / "result", image, result)
        except Exception as error:
            with self.lock:
                self.latest_root = root
                self.latest_source = image
                self.latest = {"status": "ERROR", "part_id": upload_id,
                               "source_mode": ("USER_DECLARED_MVTEC_METAL_NUT_UPLOAD" if benchmark
                                               else "USER_UPLOADED_IMAGE_DIAGNOSTIC"),
                               "error": str(error)}
        finally:
            with self.lock:
                self.busy = False

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
            if self.args.vision_worker_socket is not None:
                command += ["--vision-worker-socket", str(self.args.vision_worker_socket)]
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
                      "response_regions": (record["geometry"] or {}).get("regions", []),
                      "zone_method": (record["geometry"] or {}).get("zone_method"),
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
            if self.args.vision_worker_socket is not None:
                self._inspect_inline(image, part_id, root,
                                     {"mvtec_benchmark_upload": False,
                                      "uploaded_image_diagnostic": False,
                                      "operator_declared_part": source_kind == "physical_nut",
                                      "screen_photo_diagnostic": source_kind != "physical_nut",
                                      "diagnostic_no_part": False})
                return
            command = [sys.executable, "-B", "-m", "forgesight.vision_edge_live_once",
                       "--image", str(image), "--part-id", part_id,
                       "--vision-python", str(self.args.vision_python),
                       "--model", str(self.args.model), "--model-sha256", MODEL_SHA256,
                       "--grade-policy", str(self.args.grade_policy),
                       "--database", str(self.args.database),
                       "--output-dir", str(root / "result"),
                       "--camera-index", str(self.args.camera_index),
                       "--operator-declared-part" if source_kind == "physical_nut" else "--screen-photo-diagnostic"]
            if self.args.vision_worker_socket is not None:
                command += ["--vision-worker-socket", str(self.args.vision_worker_socket)]
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
            if self.path == "/api/advice":
                if self.headers.get("Content-Length", "0") != "0":
                    self.send(400, b"Invalid request", "text/plain")
                    return
                try:
                    status = service.request_advice()
                except ValueError as error:
                    self.send(409, json.dumps({"error": str(error)}).encode(), "application/json")
                    return
                self.send(202 if status == "PENDING" else 409,
                          json.dumps({"status": status}).encode(), "application/json")
                return
            if self.path not in ("/api/inspect", "/api/replay", "/api/replay-local",
                                 "/api/upload-diagnostic", "/api/manual-review"):
                self.send(400, b"Invalid request", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if self.path == "/api/manual-review":
                    if self.headers.get("Content-Type") != "application/json" or not 0 < length <= 1024:
                        raise ValueError("Invalid manual review request")
                    body = json.loads(self.rfile.read(length))
                    review = service.manual_review(body.get("inspection_id"), body.get("actor"),
                                                   body.get("outcome"), body.get("reason"))
                    self.send(201, json.dumps(review).encode(), "application/json")
                    return
                elif self.path == "/api/inspect":
                    if self.headers.get("Content-Type") != "application/json" or not 0 < length <= 512:
                        raise ValueError("Invalid camera request")
                    body = json.loads(self.rfile.read(length))
                    service.inspect(body.get("part_id"), body.get("confirmed"), body.get("source_kind"))
                elif self.path == "/api/replay":
                    if self.headers.get("Content-Type") != "image/png" or not 0 < length <= MAX_UPLOAD_BYTES:
                        raise ValueError("Upload one original PNG under 8 MiB")
                    service.replay(self.headers.get("X-Replay-Id"),
                                   self.headers.get("X-Dataset-Path"), self.rfile.read(length))
                elif self.path == "/api/replay-local":
                    if self.headers.get("Content-Type") != "application/json" or not 0 < length <= 512:
                        raise ValueError("Invalid local replay request")
                    body = json.loads(self.rfile.read(length))
                    service.replay_local(body.get("replay_id"), body.get("dataset_path"))
                else:
                    if self.headers.get("Content-Type") != "image/png" or not 0 < length <= MAX_UPLOAD_BYTES:
                        raise ValueError("Upload one RGB PNG under 8 MiB")
                    benchmark = self.headers.get("X-Mvtec-Benchmark")
                    if benchmark not in (None, "yes", "no"):
                        raise ValueError("Invalid benchmark declaration")
                    service.upload_diagnostic(self.headers.get("X-Upload-Id"), self.rfile.read(length),
                                              benchmark == "yes")
            except (ValueError, RuntimeError, TypeError, AttributeError) as error:
                self.send(409, json.dumps({"error": str(error)}).encode(), "application/json")
                return
            self.send(202, b'{"status":"PROCESSING"}', "application/json")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-python", type=Path, default=Path("venv/bin/python"))
    parser.add_argument("--model", type=Path, default=Path("efficientad_s_calibrated.ts"))
    parser.add_argument("--grade-policy", type=Path, default=Path("configs/forgegrade_benchmark_v1.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("manifests/metal_nut_v1.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec_ad/metal_nut"))
    parser.add_argument("--guidance", type=Path, default=Path("guidance/prototype_inspection_v1.json"))
    parser.add_argument("--llama-root", type=Path,
                        default=Path("/home/nvidia/.local/share/forgesight/llama_cpu_v1"))
    parser.add_argument("--advice-root", type=Path,
                        default=Path("runs/live_inspections_v1/advice"))
    parser.add_argument("--llama-timeout-seconds", type=float, default=75)
    parser.add_argument("--database", type=Path, default=Path("runs/live_inspections_v1/inspections.sqlite3"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/live_inspections_v1"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--vision-worker-socket", type=Path, default=None,
                        help="Keep one hash-checked EfficientAD worker process alive at this Unix socket; inspections reuse it instead of a fresh model process each time.")
    parser.add_argument("--vision-worker-threads", type=int, default=2)
    parser.add_argument("--overlay-format", choices=("png", "jpeg"), default="png",
                        help="Overlay encoding for inline inspections; jpeg is much faster to encode, lossy pixels.")
    parser.add_argument("--vision-worker-idle-timeout-seconds", type=float, default=3600,
                        help="Worker exits after this long without a request, freeing model memory; default 3600.")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged port")
    if not 0 < args.llama_timeout_seconds <= 240:
        parser.error("Use a Llama timeout between 0 and 240 seconds")
    if not 1 <= args.vision_worker_threads <= 4:
        parser.error("Use one to four vision worker threads")
    if args.vision_worker_socket is not None and args.vision_worker_idle_timeout_seconds < 60:
        parser.error("Use a vision worker idle timeout of at least 60 seconds")
    args.database = args.database.resolve()
    args.runs_root = args.runs_root.resolve()
    args.vision_python = vision_executable(args.vision_python)
    args.model = args.model.resolve()
    args.grade_policy = args.grade_policy.resolve()
    args.dataset_manifest = args.dataset_manifest.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.guidance = args.guidance.resolve()
    args.llama_root = args.llama_root.resolve()
    args.advice_root = args.advice_root.resolve()
    if args.vision_worker_socket is not None:
        # The worker itself refuses only a live socket and reclaims a stale one.
        args.vision_worker_socket = args.vision_worker_socket.resolve()
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
        service.close()


if __name__ == "__main__":
    main()
