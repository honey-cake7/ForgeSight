"""Loopback live C270 preview and one-part EfficientAD inspection UI for Jetson."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
from threading import Event, Lock, Thread
import time
from uuid import UUID, uuid4

from PIL import Image

from forgesight.assist_display import advice_for
from forgesight.assist_queue import AdvisoryQueue
from forgesight.operation_log import configure_operation_log, operation_logger, scrub
from forgesight.defect_type import load_profile
from forgesight.storage import InspectionStore, new_correction
from forgesight.vision_edge_live_once import (
    CHILD_TIMEOUT_SECONDS, MODEL_SHA256, WORKER_REQUEST_TIMEOUT_SECONDS, sha256)

# The worst legitimate inline/replay path is a worker request that hangs to
# its full client budget, then a cold fallback child with its own full
# budget, plus preprocess/hash/grade overhead. The outer subprocess timeout
# must exceed that sum or it kills the cold retry mid-run, leaving a
# partial run directory and no record.
VISION_OUTER_TIMEOUT_SECONDS = (WORKER_REQUEST_TIMEOUT_SECONDS
                                + CHILD_TIMEOUT_SECONDS + 60)

_log = operation_logger()


WEB_ROOT = Path(__file__).with_name("web")
PART_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
LOW_DISK_BYTES = 1024 ** 3  # warn below 1 GiB free
RUN_DIR_PREFIXES = ("part_", "replay_", "user_upload_")
STORAGE_SCAN_MAX_ENTRIES = 5000    # directory entries examined per pass
STORAGE_SCAN_MAX_DIRS = 200        # run dirs reconciled per pass
STORAGE_SCAN_MAX_RECORD_BYTES = 1024 * 1024
DATASET_MANIFEST_SHA256 = "14b11ebd03d5107c7ef3478e1c4d20887237e2fbef58601ac043014b8f0e340f"


def _path_is_relative_to(path, parent):
    """Return pathlib containment using APIs available in Python 3.8."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


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


def launch_advice_worker(args, python_command=None):
    """Start the persistent advisory worker and wait for its READY line.

    Runs the Llama venv Python with the repo root on PYTHONPATH so the venv
    (which only holds the runtime packages) can import forgesight. A boot
    failure raises; the caller keeps the cold subprocess path available.
    """
    from collections import deque
    command = (python_command if python_command is not None
               else [str(args.llama_root / "venv/bin/python"), "-B",
                     "-m", "forgesight.assist_edge_worker"])
    command = list(command) + [
               "--socket", str(args.llama_worker_socket),
               "--llama-root", str(args.llama_root),
               "--idle-timeout-seconds", str(args.llama_worker_idle_timeout_seconds)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    worker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=env)
    stderr_tail = deque(maxlen=50)

    def drain_stderr():
        for line in worker.stderr:
            stderr_tail.append(line.rstrip())

    drain = Thread(target=drain_stderr, daemon=True)
    drain.start()
    stdout_tail = deque(maxlen=50)
    ready_holder = []

    def read_ready():
        # ExecuTorch writes diagnostic lines to stdout before the worker's
        # readiness JSON. Keep draining those lines so the pipe cannot fill,
        # and accept only the explicit protocol record as readiness.
        for line in worker.stdout:
            stdout_tail.append(line.rstrip()[-1000:])
            try:
                status = json.loads(line)
            except ValueError:
                continue
            if isinstance(status, dict) and status.get("status") == "WORKER_READY":
                ready_holder.append(status)
                return

    reader = Thread(target=read_ready, daemon=True)
    reader.start()
    reader.join(timeout=180)
    status = ready_holder[0] if ready_holder else {}
    if status.get("status") != "WORKER_READY":
        exited = worker.poll()
        worker.terminate()
        drain.join(timeout=2)  # collect the worker's own error output first
        reader.join(timeout=2)
        detail = ""
        if stderr_tail:
            detail += "; worker stderr tail: " + " | ".join(stderr_tail)
        if stdout_tail:
            detail += "; worker stdout tail: " + " | ".join(stdout_tail)
        if exited is not None:
            raise ValueError(
                "Advisory worker exited during startup (code " + str(exited) +
                ") before reporting ready" + detail)
        raise ValueError(
            "Advisory worker failed to verify and load the model within 180 seconds" + detail)
    return worker


class CameraFeed:
    def __init__(self, index=0):
        self.index = index
        self.lock = Lock()
        self.control_lock = Lock()
        self.stop = Event()
        self.enabled = Event()
        self.enabled.set()  # Existing behavior: camera starts with station
        self.capture_released = Event()
        self.capture_released.set()
        self.frame = None
        self.jpeg = None
        self.updated = 0.0
        self.error = "Starting camera"
        self.thread = Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.enabled.set()  # Wake the capture loop for shutdown
        self.thread.join(timeout=3)

    def turn_on(self):
        with self.control_lock:
            with self.lock:
                if self.stop.is_set():
                    raise RuntimeError("Camera service is shutting down")
                if self.enabled.is_set():
                    return
            if not self.capture_released.wait(timeout=3):
                raise RuntimeError("Timed out waiting for camera device release")
            with self.lock:
                if self.stop.is_set():
                    raise RuntimeError("Camera service is shutting down")
                if self.enabled.is_set():
                    return
                self.frame = None
                self.jpeg = None
                self.updated = 0.0
                self.error = "Starting camera"
                self.enabled.set()

    def turn_off(self):
        with self.control_lock:
            with self.lock:
                self.enabled.clear()
                self.frame = None
                self.jpeg = None
                self.updated = 0.0
                self.error = "Camera stopped by operator"
            if not self.capture_released.wait(timeout=3):
                with self.lock:
                    self.error = "Timed out waiting for camera device release"
                raise RuntimeError(self.error)

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
            running = self.enabled.is_set() and not self.stop.is_set()
            ready = running and self.jpeg is not None and time.monotonic() - self.updated <= 2
            return {"ready": ready, "running": running,
                    "message": "Live C270 preview" if ready else self.error}

    def _loop(self):
        import cv2

        while not self.stop.is_set():
            if not self.enabled.wait(timeout=0.1) or self.stop.is_set():
                continue
            self.capture_released.clear()
            capture = None
            try:
                if not self.enabled.is_set() or self.stop.is_set():
                    continue
                capture = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
                if not capture.isOpened():
                    raise RuntimeError("Camera could not open; check /dev/video0 and USB connection")
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_FPS, 30)
                next_encode = 0.0
                while not self.stop.is_set() and self.enabled.is_set():
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
                        if not self.enabled.is_set() or self.stop.is_set():
                            continue
                        self.frame = frame.copy()
                        self.jpeg = encoded.tobytes()
                        self.updated = time.monotonic()
                        self.error = ""
            except Exception as error:
                with self.lock:
                    self.frame = None
                    self.jpeg = None
                    if self.enabled.is_set() and not self.stop.is_set():
                        self.error = str(error)
            finally:
                try:
                    if capture is not None:
                        capture.release()
                finally:
                    self.capture_released.set()
            # Do not impose a 1-second restart delay after an intentional stop.
            if self.enabled.is_set():
                self.stop.wait(1)


ADVICE_FUTURES_COMPLETED_KEPT = 32
WORKER_RESPAWN_MIN_INTERVAL_SECONDS = 60


class LiveService:
    def __init__(self, args, camera):
        self.args = args
        self.camera = camera
        profile_path = getattr(args, "defect_type_profile", None)
        self.defect_type_profile = load_profile(profile_path) if profile_path else None
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
            try:
                self.vision_worker = launch_vision_worker(args)
            except Exception:
                _log.exception("Vision worker failed to start at boot.")
                raise
            _log.info("Vision worker started at boot.")
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
        self.advice_worker = None
        self.advice_worker_boot_error = None
        self.advice_worker_respawning = False
        self.advice_worker_respawn_lock = Lock()
        self.advice_recovered = set()  # inspection ids whose wedge recovery already ran
        self.closed = False
        self._advice_respawn_last = None  # no attempt yet; never suppress the first
        self._vision_respawn_last = None
        if self.llama_ready and getattr(args, "llama_worker_socket", None) is not None:
            try:
                self.advice_worker = launch_advice_worker(args)
            except ValueError as error:
                # Advice is optional: keep serving inspections cold; the boot
                # error stays visible in advice_state instead of crashing the UI.
                self.advice_worker_boot_error = str(error)
                _log.error("Advisory worker failed to start at boot; requests "
                           "cold-fall back and retry: %s", scrub(error))
            else:
                _log.info("Advisory worker started at boot.")

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
            defect_type_profile=getattr(self, "defect_type_profile", None),
            **mode_flags)
        record = live_once.inspect(args)

        class Done:
            returncode = 0 if record["status"] == "ANALYZED" else 1
            stdout = stderr = ""
        self._publish_result(part_id, root / "result", image, Done())
        self._rewarm_after_cold_fallback(record)

    def _rewarm_after_cold_fallback(self, record):
        if (record.get("timings") or {}).get("vision_mode") == "COLD_PROCESS_AFTER_WORKER_UNAVAILABLE":
            # The worker died (idle timeout or crash). Bring it back so the next
            # inspection is warm instead of silently running cold forever.
            _log.warning("Inspection %s ran cold after the vision worker became unavailable; respawning.",
                         record.get("inspection_id"))
            self._respawn_vision_worker()

    def _respawn_advice_worker(self):
        with self.advice_worker_respawn_lock:
            if self.advice_worker_respawning or getattr(self, "closed", False):
                return
            now = time.monotonic()
            last = getattr(self, "_advice_respawn_last", None)
            if (last is not None
                    and now - last < WORKER_RESPAWN_MIN_INTERVAL_SECONDS):
                # A permanently broken worker must not earn a fresh launch
                # attempt on every request; the cold path keeps serving.
                return
            self.advice_worker_respawning = True
            self._advice_respawn_last = now

        def respawn():
            try:
                old = self.advice_worker
                if old is not None and old.poll() is None:
                    old.terminate()
                    try:
                        old.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        old.kill()
                if self.llama_ready:
                    with self.advice_worker_respawn_lock:
                        if getattr(self, "closed", False):
                            return
                    new = launch_advice_worker(self.args)
                    with self.advice_worker_respawn_lock:
                        if getattr(self, "closed", False):
                            # close() ran while the worker was launching; do
                            # not leave a live worker behind a closed service.
                            new.terminate()
                            try:
                                new.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                new.kill()
                            return
                        self.advice_worker = new
                        self.advice_worker_boot_error = None
                    _log.info("Advisory worker respawned.")
            except Exception:
                # Next request cold-falls-back again and retries the respawn
                # after the backoff interval.
                _log.exception("Advisory worker respawn failed; requests cold-fall back and retry.")
            finally:
                with self.advice_worker_respawn_lock:
                    self.advice_worker_respawning = False

        Thread(target=respawn, daemon=True).start()

    def _respawn_vision_worker(self):
        with self.worker_respawn_lock:
            if self.worker_respawning or getattr(self, "closed", False):
                return
            now = time.monotonic()
            last = getattr(self, "_vision_respawn_last", None)
            if (last is not None
                    and now - last < WORKER_RESPAWN_MIN_INTERVAL_SECONDS):
                return
            self.worker_respawning = True
            self._vision_respawn_last = now

        def respawn():
            try:
                old = self.vision_worker
                if old is not None and old.poll() is None:
                    old.terminate()
                    try:
                        old.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        old.kill()
                with self.worker_respawn_lock:
                    if getattr(self, "closed", False):
                        return
                new = launch_vision_worker(self.args)
                with self.worker_respawn_lock:
                    if getattr(self, "closed", False):
                        new.terminate()
                        try:
                            new.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            new.kill()
                        return
                    self.vision_worker = new
                _log.info("Vision worker respawned.")
            except Exception:
                # Next inspection cold-falls-back again and retries the
                # respawn after the backoff interval.
                _log.exception("Vision worker respawn failed; inspections cold-fall back and retry.")
            finally:
                with self.worker_respawn_lock:
                    self.worker_respawning = False

        Thread(target=respawn, daemon=True).start()

    def close(self):
        # Block new respawns first; a respawn mid-launch terminates its new
        # worker when it sees this flag instead of leaking past close().
        self.closed = True
        self.advice_queue.close()
        if self.advice_worker is not None:
            self.advice_worker.terminate()
            try:
                self.advice_worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.advice_worker.kill()
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
        worker_available = (self.advice_worker is not None
                            and self.advice_worker.poll() is None)
        mode = {"llama_runtime": "PERSISTENT_WORKER" if worker_available else "COLD_SUBPROCESS",
                "advice_worker_available_now": worker_available,
                "advice_worker_boot_error": self.advice_worker_boot_error}
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
        if future is not None and future.done():
            self._recover_finished_advice(latest["inspection_id"], future)
        if future is not None and not future.done():
            return {"status": "PENDING", "advisory": None, **mode}
        if future is not None and advice["status"] in ("NOT_REQUESTED", "PENDING"):
            return {"status": "ADVICE_UNAVAILABLE", "advisory": None, **mode}
        if isinstance(advice, dict):
            recorded = advice.pop("llama_runtime_mode", None)
            advice = {**advice, **mode}
            if advice.get("status") == "GENERATED_REQUIRES_HUMAN_REVIEW":
                # A saved advisory carries the runtime that produced it; the
                # process alive now may not be the one that generated it. A
                # legacy report without a recorded mode is honestly unknown,
                # never labelled from current worker health.
                advice["llama_runtime"] = (recorded if recorded in (
                    "PERSISTENT_WORKER", "COLD_SUBPROCESS",
                    "PERSISTENT_WORKER_UNREACHABLE_COLD_FALLBACK") else "NOT_RECORDED")
        return advice

    def _recover_finished_advice(self, inspection_id, future):
        """One bounded recovery action per finished advisory request.

        When the controller flagged the persistent worker as wedged (alive
        but no longer answering), replace it so the next request is warm
        again. Best-effort: the report file on disk stays authoritative.
        """
        recovered = getattr(self, "advice_recovered", None)
        if recovered is None:
            recovered = self.advice_recovered = set()
        if len(recovered) >= 64:
            # Bounded state: only the latest inspection's finished future is
            # ever consulted, so old ids carry no information.
            recovered.clear()
        if inspection_id in recovered:
            return
        recovered.add(inspection_id)
        try:
            result = future.result()
        except Exception as error:
            # A killed or crashed replay carries no summary; the outcome
            # still belongs in the operation log.
            _log.error("Advice request for inspection %s could not be collected: %s",
                       inspection_id, scrub(error))
            return
        if isinstance(result, dict):
            # AdvisoryQueue converts a raised worker (for example a subprocess
            # timeout) into this outcome dict instead of a CompletedProcess.
            _log.error("Advice request for inspection %s failed: status=%s error=%s",
                       inspection_id, result.get("status"),
                       scrub(str(result.get("error", ""))))
            return
        summary = {}
        if result is not None and result.stdout:
            try:
                parsed = json.loads(result.stdout)
                if isinstance(parsed, dict):
                    summary = parsed
            except ValueError:
                pass
        returncode = getattr(result, "returncode", None)
        status = summary.get("status") or "NO_REPORT"
        if returncode == 0:
            _log.info("Advice request for inspection %s finished: status=%s",
                      inspection_id, status)
        else:
            stderr_tail = ""
            if result is not None and result.stderr:
                stderr_tail = result.stderr[-400:]
            _log.error("Advice request for inspection %s failed: returncode=%s "
                       "status=%s stderr_tail=%s",
                       inspection_id, returncode, status, scrub(stderr_tail))
        if summary.get("persistent_worker_unhealthy"):
            self._respawn_advice_worker()

    def _prune_advice_futures(self):
        """Bound advice_futures growth; caller holds self.lock.

        Completed futures are retained only so a late /api/state read can
        resolve a finished request. Without a bound the dict gains one entry
        per inspection for the life of the process, leaking every Future and
        its captured subprocess result. Keep the newest few completed
        entries; in-flight futures are never evicted.
        """
        completed = [key for key, future in self.advice_futures.items()
                     if future.done()]
        for key in completed[:-ADVICE_FUTURES_COMPLETED_KEPT]:
            self.advice_futures.pop(key, None)

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
            pending = self.advice_futures.get(record["inspection_id"])
            if pending is not None and not pending.done():
                return "ALREADY_REQUESTED"
            if output.exists():
                report_path = output / "advice_report.json"
                if report_path.is_file():
                    # Only a validated terminal success for THIS inspection
                    # blocks another request: a bare claimed status without
                    # provenance is withheld by advice_for and must not block
                    # retry forever. Failed, unreadable or non-object reports
                    # are quarantined aside the same way as incomplete output,
                    # preserved, never deleted.
                    if advice_for(record, self.advice_root)["status"] == (
                            "GENERATED_REQUIRES_HUMAN_REVIEW"):
                        return "ALREADY_REQUESTED"
                    quarantine = output.with_name(
                        output.name + ".failed-"
                        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                        + "-" + uuid4().hex[:8])
                    os.replace(output, quarantine)
                else:
                    # An earlier attempt died before publishing its report (outer
                    # kill, crash, strict-JSON failure). Its files are incomplete
                    # and must not masquerade as a finished advisory, but they
                    # must not block a fresh request either: quarantine them
                    # aside, never delete. The suffix is unique per quarantine so
                    # repeated retries within one second cannot collide; a
                    # symlinked output moves the link itself, never its target.
                    quarantine = output.with_name(
                        output.name + ".incomplete-"
                        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                        + "-" + uuid4().hex[:8])
                    os.replace(output, quarantine)
            command = [sys.executable, "-B", "-m", "forgesight.assist_jetson_replay_once",
                       "--record", str(record_path), "--database", str(self.args.database),
                       "--guidance", str(self.args.guidance),
                       "--llama-root", str(self.args.llama_root),
                       "--output-dir", str(output), "--llama-timeout-seconds",
                       str(self.args.llama_timeout_seconds),
                       # Absolute deadline on the same clock the outer kill
                       # uses, computed at spawn time: interpreter and import
                       # time count against the shared budget too.
                       "--request-deadline-monotonic",
                       str(time.monotonic() + self.args.llama_timeout_seconds + 45)]
            if getattr(self.args, "llama_worker_socket", None) is not None:
                if self.advice_worker is not None and self.advice_worker.poll() is None:
                    command += ["--llama-worker-socket", str(self.args.llama_worker_socket)]
                else:
                    # Dead or never started: this request takes the cold path
                    # (which re-verifies every asset) while the worker respawns.
                    self._respawn_advice_worker()
            future = self.advice_queue.submit(lambda: subprocess.run(
                command, capture_output=True, text=True,
                timeout=self.args.llama_timeout_seconds + 45))
            if future is None:
                _log.warning("Advice request for inspection %s could not be queued (busy).",
                             record["inspection_id"])
                return "BUSY"
            self.advice_futures[record["inspection_id"]] = future
            self._prune_advice_futures()
            _log.info("Advice request for inspection %s submitted (worker=%s).",
                      record["inspection_id"],
                      self.advice_worker is not None and self.advice_worker.poll() is None)
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
                             "assessment": r.get("assessment"),
                             "timestamp_utc": r["timestamp_utc"]}
                            for r in self.store.history(limit=10)],
                "storage": self.storage_health()}

    def storage_health(self, *, now=None):
        """Read-only runs_root/SQLite reconciliation plus free-disk check.

        Orphan classes: "partial" (killed before record.json),
        "unconfirmed" (record.json without a SQLite row), "mismatch"
        (stored record differs from the disk record) and
        "stored_unreadable" (the SQLite row itself fails its checksum).
        Everything is identified and reported, never deleted: evidence
        retention is an operator policy. The scan is bounded (entry cap,
        record size cap, no symlink traversal) and single-flight; results
        cache for 60 seconds so /api/state polls stay cheap. A store or
        filesystem failure degrades the diagnostic to UNAVAILABLE or
        INCOMPLETE instead of taking /api/state down or lying zero.
        """
        now = time.monotonic() if now is None else now
        lock = getattr(self, "_storage_health_lock", None)
        if lock is None:
            lock = self._storage_health_lock = Lock()
        with lock:
            cached = getattr(self, "_storage_health", None)
            if cached is not None and now - self._storage_health_at < 60:
                return cached
            runs_root = Path(self.args.runs_root)
            health = {"status": "OK", "orphan_count": 0, "orphan_runs": [],
                      "orphans": {}, "truncated": False,
                      "disk_free_bytes": None, "low_disk": None}
            try:
                free = shutil.disk_usage(runs_root).free
                health["disk_free_bytes"] = free
                health["low_disk"] = free < LOW_DISK_BYTES
            except OSError as error:
                _log.warning("Could not measure free space under %s: %s",
                             runs_root, scrub(error))
            scanned = 0
            orphans = {}
            try:
                with os.scandir(runs_root) as listing:
                    for examined, entry in enumerate(listing, 1):
                        if examined > STORAGE_SCAN_MAX_ENTRIES:
                            health["status"] = "INCOMPLETE"
                            health["truncated"] = True
                            break
                        try:
                            if (not entry.is_dir(follow_symlinks=False)
                                    or not entry.name.startswith(RUN_DIR_PREFIXES)):
                                continue
                        except OSError:
                            continue
                        if scanned >= STORAGE_SCAN_MAX_DIRS:
                            health["status"] = "INCOMPLETE"
                            health["truncated"] = True
                            break
                        scanned += 1
                        kind = self._classify_run_dir(Path(entry.path))
                        if kind == "unavailable":
                            health["status"] = "UNAVAILABLE"
                            break
                        if kind is not None:
                            orphans[entry.name] = kind
            except OSError as error:
                health["status"] = "UNAVAILABLE"
                _log.warning("Could not list runs root %s: %s",
                             runs_root, scrub(error))
            health["orphan_count"] = len(orphans)
            health["orphan_runs"] = sorted(orphans)[:20]
            health["orphans"] = dict(sorted(orphans.items())[:20])
            if health["orphan_count"] and health["status"] == "OK":
                health["status"] = "OK"  # orphans are findings, not scan failure
            previous = cached or {}
            changed = (previous.get("orphan_count"), previous.get("low_disk"),
                       previous.get("status")) != (
                           health["orphan_count"], health["low_disk"],
                           health["status"])
            if changed and orphans:
                _log.warning("%d orphan run dir(s) under %s (%s): %s",
                             len(orphans), runs_root, health["status"],
                             ", ".join("%s=%s" % pair
                                       for pair in sorted(orphans.items())[:20]))
            if changed and health["low_disk"]:
                _log.warning("Low disk space under %s: %d bytes free.",
                             runs_root, health["disk_free_bytes"])
            if changed and health["status"] == "UNAVAILABLE":
                _log.warning("Storage reconciliation unavailable under %s.",
                             runs_root)
            self._storage_health, self._storage_health_at = health, now
            return health

    def _classify_run_dir(self, folder):
        """Orphan kind for one run dir, None when confirmed, 'unavailable'
        when the store itself cannot answer."""
        record_path = folder / "result" / "record.json"
        try:
            # Reject link escapes before any read: a symlinked result dir or
            # record file, or a path that resolves outside runs_root, must
            # never be opened or looked up.
            if ((folder / "result").is_symlink() or record_path.is_symlink()
                    or not _path_is_relative_to(
                        record_path.resolve(), Path(self.args.runs_root).resolve())):
                return "invalid"
            if not record_path.is_file():
                return "partial"
            if record_path.stat().st_size > STORAGE_SCAN_MAX_RECORD_BYTES:
                return "partial"  # never written by the bounded pipeline
            record = json.loads(record_path.read_text())
            inspection_id = record["inspection_id"]
        except (ValueError, KeyError, TypeError, OSError):
            return "partial"
        try:
            stored = self.store.get_inspection(inspection_id)
        except sqlite3.Error:
            return "unavailable"
        except (ValueError, KeyError, TypeError):
            return "stored_unreadable"
        if stored is None:
            return "unconfirmed"
        if stored != record:
            return "mismatch"
        return None

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

    def inspection_image(self, inspection_id, name):
        """Evidence bytes bound to one stored inspection, never the global latest.

        The record in SQLite is the authority for which file and which digest
        belong to an inspection. Serving whatever the latest run directory
        holds could show one inspection's decision with another's image, and
        re-reading changed bytes would misrepresent the original evidence.
        Returns (status, data): 200 with bytes, 404 when unknown/unavailable,
        409 when the bytes no longer match the recorded digest.
        """
        try:
            parsed = UUID(str(inspection_id))
        except (ValueError, AttributeError, TypeError):
            return 404, None
        if str(parsed) != inspection_id:
            # UUID() also accepts 32-hex-without-hyphens, braced and uppercase
            # forms; the store keys on the canonical hyphenated form and
            # raises on the others. Treat any non-canonical form as unknown.
            return 404, None
        try:
            record = self.store.get_inspection(inspection_id)
        except (ValueError, KeyError, TypeError):
            # A lookup failure is an unknown inspection, never a 500.
            return 404, None
        if record is None:
            return 404, None
        if name == "overlay":
            reference = record["artifacts"].get("response_overlay")
            expected = record["artifacts"].get("response_overlay_sha256")
        else:
            reference = record["source"].get("path")
            expected = record["source"].get("sha256")
        if not reference:
            return 404, None
        path = Path(reference).resolve()
        runs_root = Path(self.args.runs_root).resolve()
        if (not _path_is_relative_to(path, runs_root)
                or path.suffix.lower() not in (".png", ".jpg", ".jpeg")):
            return 404, None
        if not path.is_file():
            return 404, None
        data = path.read_bytes()
        if expected and hashlib.sha256(data).hexdigest() != expected:
            return 409, None
        return 200, data

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
            if benchmark and getattr(self.args, "defect_type_profile", None):
                command += ["--defect-type-profile", str(self.args.defect_type_profile)]
            if self.args.vision_worker_socket is not None:
                command += ["--vision-worker-socket", str(self.args.vision_worker_socket)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=VISION_OUTER_TIMEOUT_SECONDS)
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
            if getattr(self.args, "defect_type_profile", None):
                command += ["--defect-type-profile", str(self.args.defect_type_profile)]
            if self.args.vision_worker_socket is not None:
                command += ["--vision-worker-socket", str(self.args.vision_worker_socket)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=VISION_OUTER_TIMEOUT_SECONDS)
            self._publish_result(replay_id, root / "result", image, result)
            record_path = root / "result" / "record.json"
            if record_path.is_file():
                # Same cold-fallback recovery as inline inspections: a dead
                # worker must be replaced after a replay too, or replay-only
                # sessions run every later inspection on the slow cold path.
                self._rewarm_after_cold_fallback(json.loads(record_path.read_text()))
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
        record = None
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
                      "assessment": record.get("assessment"),
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
        if record is not None:
            _log.info("Inspection %s (%s) finished: status=%s grade=%s vision_mode=%s",
                      record["inspection_id"], part_id, latest.get("status"),
                      latest.get("grade"), record.get("timings", {}).get("vision_mode"))
        else:
            _log.error("Inspection for part %s failed before a record was confirmed: %s",
                       part_id, latest.get("error"))

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
            result = subprocess.run(command, capture_output=True, text=True, timeout=VISION_OUTER_TIMEOUT_SECONDS)
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
            elif path.startswith("/api/inspection/"):
                parts = path.split("/")
                kind = parts[4] if len(parts) == 5 else ""
                if kind not in ("source.png", "overlay.png"):
                    self.send(404, b"Not found", "text/plain")
                    return
                status, data = service.inspection_image(
                    parts[3], "overlay" if kind.startswith("overlay") else "source")
                if status == 409:
                    self.send(409, b"Saved image changed; original inspection evidence does not match.",
                              "text/plain")
                else:
                    self.send(200 if data else 404, data or b"Image unavailable",
                              "image/png" if data else "text/plain")
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
            if self.path in ("/api/camera/start", "/api/camera/stop"):
                if self.headers.get("Content-Length", "0") != "0":
                    self.send(400, b"Invalid camera control request", "text/plain")
                    return
                try:
                    if self.path == "/api/camera/start":
                        service.camera.turn_on()
                    else:
                        service.camera.turn_off()
                except RuntimeError as error:
                    self.send(503, str(error).encode(), "text/plain")
                    return
                self.send(200, json.dumps(service.camera.state()).encode(), "application/json")
                return
            if self.path == "/api/advice":
                if self.headers.get("Content-Length", "0") != "0":
                    self.send(400, b"Invalid request", "text/plain")
                    return
                try:
                    status = service.request_advice()
                except ValueError as error:
                    code, body = _refusal_body("/api/advice", error, safe=True)
                    self.send(code, body.encode(), "application/json")
                    return
                except (OSError, RuntimeError, TypeError, AttributeError,
                        sqlite3.Error) as error:
                    code, body = _refusal_body("/api/advice", error, safe=False)
                    self.send(code, body.encode(), "application/json")
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
            except ValueError as error:
                code, body = _refusal_body(self.path, error, safe=True)
                self.send(code, body.encode(), "application/json")
                return
            except (RuntimeError, TypeError, AttributeError, OSError,
                    sqlite3.Error) as error:
                # SQLite and filesystem failures must never drop the
                # connection or leak internals to the client.
                code, body = _refusal_body(self.path, error, safe=False)
                self.send(code, body.encode(), "application/json")
                return
            self.send(202, b'{"status":"PROCESSING"}', "application/json")

    return Handler


def _refusal_body(path, error, *, safe):
    """Error body with a support ID; the log line with the same ID has detail.

    Validation messages are operator-facing by convention and returned with
    a 409; that is a policy for known validation paths, not a guarantee that
    every ValueError text is safe. Anything else is an internal failure: the
    client gets a generic
    500 message and the support ID, never raw internals; the full scrubbed
    detail is in the operation log under the same ID. The endpoint is
    normalized (no query string) before logging.
    """
    support_id = uuid4().hex[:8]
    endpoint = path.split("?", 1)[0]
    if safe:
        _log.warning("POST %s refused (support_id=%s): %s",
                     endpoint, support_id, scrub(error))
        return 409, json.dumps({"error": str(error), "support_id": support_id})
    _log.error("POST %s failed (support_id=%s): %s",
               endpoint, support_id, scrub(error))
    return 500, json.dumps({
        "error": "Station failed to complete the request; quote the support "
                 "ID when reporting it.",
        "support_id": support_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-python", type=Path, default=Path("venv/bin/python"))
    parser.add_argument("--model", type=Path, default=Path("efficientad_s_calibrated.ts"))
    parser.add_argument("--grade-policy", type=Path, default=Path("configs/forgegrade_benchmark_v1.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("manifests/metal_nut_v1.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mvtec_ad/metal_nut"))
    parser.add_argument("--defect-type-profile", type=Path,
                        help="Enable provisional defect assessment for benchmark images.")
    parser.add_argument("--guidance", type=Path, default=Path("guidance/prototype_inspection_v1.json"))
    parser.add_argument("--llama-worker-socket", type=Path, default=None,
                        help="Optional persistent advisory worker socket; omit for the per-request cold path")
    parser.add_argument("--llama-worker-idle-timeout-seconds", type=float, default=900,
                        help="Idle advisory worker exit so its memory is released (default 900)")
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
    if not 60 <= args.llama_worker_idle_timeout_seconds <= 3600:
        parser.error("Advisory worker idle timeout must be 60-3600 seconds.")
    if not 0 < args.llama_timeout_seconds <= 240:
        parser.error("Use a Llama timeout between 0 and 240 seconds")
    if not 1 <= args.vision_worker_threads <= 4:
        parser.error("Use one to four vision worker threads")
    if args.vision_worker_socket is not None and args.vision_worker_idle_timeout_seconds < 60:
        parser.error("Use a vision worker idle timeout of at least 60 seconds")
    args.database = args.database.resolve()
    args.runs_root = args.runs_root.resolve()
    configure_operation_log(args.runs_root / "logs")
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
