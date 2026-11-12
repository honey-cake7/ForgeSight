"""Loopback saved-replay advice status page for the isolated Jetson vision runtime."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys

from forgesight.assist_display import advice_for
from forgesight.assist_queue import AdvisoryQueue
from forgesight.storage import InspectionStore


WEB_ROOT = Path(__file__).with_name("web")


class ReplayAdviceService:
    def __init__(self, record_path, database, advice_root, runner):
        self.record_path = Path(record_path)
        self.record = json.loads(self.record_path.read_text())
        self.database = Path(database)
        self.advice_root = Path(advice_root)
        if (self.record.get("source", {}).get("mode") != "SAVED_IMAGE_REPLAY"
                or self.record["source"].get("physical_part") is not False):
            raise ValueError("Only a saved-image replay is permitted.")
        if InspectionStore(self.database, create=False).get_inspection(self.record["inspection_id"]) != self.record:
            raise ValueError("The saved inspection does not match SQLite.")
        self.runner = runner
        self.queue = AdvisoryQueue()
        self.future = None

    @property
    def output(self):
        return self.advice_root / self.record["inspection_id"]

    def state(self):
        advice = advice_for(self.record, self.advice_root)
        if self.future is not None and not self.future.done():
            advice = {"status": "PENDING", "advisory": None}
        elif self.future is not None and advice["status"] in ("PENDING", "NOT_REQUESTED"):
            advice = {"status": "ADVICE_UNAVAILABLE", "advisory": None}
        return {"inspection": {
            "inspection_id": self.record["inspection_id"],
            "grade": self.record["decision"]["grade"],
            "operator_action": self.record["decision"]["operator_action"],
            "source_mode": "SAVED_IMAGE_REPLAY", "physical_part_inspected": False},
            "advice": advice}

    def request(self):
        if self.future is not None:
            return "BUSY" if not self.future.done() else "ALREADY_REQUESTED"
        if self.output.exists():
            return "ALREADY_REQUESTED"
        future = self.queue.submit(lambda: self.runner(self.output))
        if future is None:
            return "BUSY"
        self.future = future
        return "PENDING"

    def close(self):
        self.queue.close()


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def allowed_host(self):
            return self.headers.get("Host") in ("127.0.0.1:" + str(self.server.server_port),
                                                "localhost:" + str(self.server.server_port))

        def do_GET(self):
            if not self.allowed_host():
                self.send(403, b"Forbidden", "text/plain; charset=utf-8")
            elif self.path == "/api/state":
                self.send(200, json.dumps(service.state()).encode(), "application/json")
            elif self.path == "/":
                self.send(200, (WEB_ROOT / "jetson_advice.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/app.js":
                self.send(200, (WEB_ROOT / "jetson_advice.js").read_bytes(), "application/javascript; charset=utf-8")
            elif self.path == "/style.css":
                self.send(200, (WEB_ROOT / "jetson_advice.css").read_bytes(), "text/css; charset=utf-8")
            else:
                self.send(404, b"Not found", "text/plain; charset=utf-8")

        def do_POST(self):
            if (not self.allowed_host() or self.headers.get("Origin") !=
                    "http://" + self.headers.get("Host", "")):
                self.send(403, b"Forbidden", "text/plain; charset=utf-8")
            elif self.path != "/api/advice" or self.headers.get("Content-Length", "0") != "0":
                self.send(400, b"Invalid request", "text/plain; charset=utf-8")
            else:
                status = service.request()
                code = 202 if status == "PENDING" else 409
                self.send(code, json.dumps({"status": status}).encode(), "application/json")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("record", "database", "guidance", "llama-root", "advice-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--llama-timeout-seconds", type=float, default=240)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    if not 0 < args.llama_timeout_seconds <= 240 or not 1024 <= args.port <= 65535:
        parser.error("Use timeout 0-240 seconds and an unprivileged port.")

    def runner(output):
        command = [sys.executable, "-B", "-m", "forgesight.assist_jetson_replay_once",
                   "--record", str(args.record), "--database", str(args.database),
                   "--guidance", str(args.guidance), "--llama-root", str(args.llama_root),
                   "--output-dir", str(output), "--llama-timeout-seconds",
                   str(args.llama_timeout_seconds)]
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=args.llama_timeout_seconds + 45)

    service = ReplayAdviceService(args.record, args.database, args.advice_root, runner)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
