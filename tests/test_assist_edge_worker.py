"""Tests for the persistent advisory worker with a stand-in runtime.

The ExecuTorch runner is Jetson-only, so a scripted stand-in stands in for
it here. These tests cover the worker's own contract: startup-once
verification, per-request independence (no reply cache, no cross-request
state), prompt/token bounds, the total read deadline against stalled and
dripping clients, artifact-change fail-safe, and idle-timeout lifecycle.
"""

import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

from forgesight import assist_edge_worker as aew

PROFILE = {"max_seq_length": 512, "max_context_length": 640}


class StandInRuntime:
    """Deterministic fake: tokenizes by whitespace, echoes a fixed note whose
    content derives only from the current prompt, and records every call."""

    def __init__(self, artifact):
        self.artifact = Path(artifact)
        self.profile = PROFILE
        self.evidence = {"artifact": {"sha256": "standin"}}
        self.generate_calls = 0
        self.seen_prompts = []

    def artifact_state(self):
        info = self.artifact.stat()
        return (str(self.artifact), info.st_size, info.st_mtime_ns, info.st_ino)

    def token_count(self, text):
        return len(text.split())

    def generate(self, formatted, max_new_tokens):
        self.generate_calls += 1
        self.seen_prompts.append(formatted)
        user_line = [line for line in formatted.splitlines()
                     if line and not line.startswith("<|")][-1]
        text = f"note for: {user_line}"
        tokens = min(len(text.split()), max_new_tokens)
        return text, {"prompt_tokens": len(formatted.split()),
                      "generated_tokens": tokens}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.artifact = Path(self.dir.name) / "model.pte"
        self.artifact.write_bytes(b"stand-in model bytes")
        self.runtime = StandInRuntime(self.artifact)
        self.worker = aew.Worker(self.runtime)

    def tearDown(self):
        self.dir.cleanup()

    def ask(self, prompt, tokens=32, system=None):
        request = {"prompt_text": prompt, "max_new_tokens": tokens}
        if system is not None:
            request["system_text"] = system
        return self.worker.handle(request)

    def test_startup_once_and_no_reply_cache(self):
        first = self.ask("hold part for review")
        second = self.ask("check lighting and clarity")
        self.assertEqual(self.runtime.generate_calls, 2)
        self.assertIn("hold part", first["generated_text"])
        self.assertIn("check lighting", second["generated_text"])
        self.assertNotIn("hold part", second["generated_text"])
        # Same prompt twice still generates twice: no output is cached.
        self.ask("hold part for review")
        self.assertEqual(self.runtime.generate_calls, 3)

    def test_no_cross_request_state_in_prompts(self):
        self.ask("alpha request")
        self.ask("bravo request")
        self.assertNotIn("alpha", self.runtime.seen_prompts[1].split("bravo")[0].replace("<|begin_of_text|>", ""))
        self.assertIn("bravo request", self.runtime.seen_prompts[1])

    def test_prompt_contract_bounds(self):
        with self.assertRaises(ValueError):
            self.ask("x" * 4097)
        with self.assertRaises(ValueError):
            self.ask("valid", system="y" * 2049)
        with self.assertRaises(ValueError):
            self.ask("prompt with <|bad|> marker")
        with self.assertRaises(ValueError):
            self.ask("valid", tokens=0)
        with self.assertRaises(ValueError):
            self.ask("valid", tokens=65)
        with self.assertRaises(ValueError):
            self.ask("valid", tokens=True)

    def test_token_budget_against_profile(self):
        # Tight context bound so the sum rule is reachable with <= 64 new tokens.
        self.runtime.profile = {"max_seq_length": 512, "max_context_length": 540}
        near_limit = " ".join(["word"] * 500)   # ~502 stand-in tokens
        with self.assertRaises(ValueError):
            self.ask(near_limit, tokens=64)      # 502 + 64 > 540 context bound
        self.ask(near_limit, tokens=32)          # 502 + 32 <= 540 passes
        with self.assertRaises(ValueError):
            self.ask(" ".join(["word"] * 520), tokens=1)  # over the prefill bound itself

    def test_stats_mismatch_rejected(self):
        def lying_generate(formatted, max_new_tokens):
            return "text", {"prompt_tokens": 999, "generated_tokens": 1}
        self.runtime.generate = lying_generate
        with self.assertRaises(ValueError):
            self.ask("hold part for review")

    def test_artifact_change_fails_closed(self):
        self.ask("hold part for review")
        time.sleep(0.01)
        self.artifact.write_bytes(b"tampered replacement")
        with self.assertRaises(RuntimeError) as caught:
            self.ask("hold part for review")
        self.assertIn("restart", str(caught.exception))


class SocketTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.artifact = self.root / "model.pte"
        self.artifact.write_bytes(b"stand-in model bytes")
        self.runtime = StandInRuntime(self.artifact)
        self.socket_path = self.root / "assist.sock"

    def tearDown(self):
        self.dir.cleanup()

    def start_server(self, **kwargs):
        worker = aew.Worker(self.runtime)
        thread = threading.Thread(
            target=aew.serve,
            args=(self.socket_path, worker),
            kwargs=kwargs, daemon=True)
        thread.start()
        for _ in range(100):
            if self.socket_path.exists():
                return thread
            time.sleep(0.01)
        raise RuntimeError("server did not start")

    def test_end_to_end_and_idle_lifecycle(self):
        thread = self.start_server(idle_timeout_seconds=0.3)
        report = aew.request(self.socket_path,
                             {"prompt_text": "hold part for review", "max_new_tokens": 16})
        self.assertEqual(report["status"], "GENERATED_BY_PERSISTENT_WORKER")
        self.assertEqual(report["worker_mode"], "PERSISTENT_PROCESS")
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(self.socket_path.exists())

    def test_stalled_then_dripping_clients_do_not_block(self):
        self.start_server(request_read_timeout_seconds=0.4)
        # Stalled: connects, sends nothing.
        stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stalled.connect(str(self.socket_path))
        time.sleep(0.6)
        stalled.close()
        # Dripping: one byte, then silence past the total deadline.
        drip = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        drip.connect(str(self.socket_path))
        drip.sendall(b"{")
        time.sleep(0.6)
        drip.close()
        report = aew.request(self.socket_path,
                             {"prompt_text": "check lighting", "max_new_tokens": 16},
                             timeout_seconds=5)
        self.assertIn("check lighting", report["generated_text"])

    def test_malformed_and_oversized_requests_error_not_crash(self):
        self.start_server()
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(str(self.socket_path))
        connection.sendall(b"not json\n")
        reply = json.loads(connection.recv(65536).split(b"\n")[0])
        connection.close()
        self.assertEqual(reply["status"], "ERROR")
        report = aew.request(self.socket_path,
                             {"prompt_text": "still alive", "max_new_tokens": 8},
                             timeout_seconds=5)
        self.assertIn("still alive", report["generated_text"])

    def test_busy_socket_path_rejected(self):
        self.start_server()
        with self.assertRaises(ValueError):
            aew.serve(self.socket_path, aew.Worker(self.runtime))


if __name__ == "__main__":
    unittest.main()


class ControllerWiringTests(unittest.TestCase):
    """assist_jetson_replay_once --llama-worker-socket end to end against a
    fake worker on a real socket: proves the warm path is selected, no
    subprocess is spawned, the host contract (parse_advisory, immutability
    re-check) still gates the output, and the report records worker mode."""

    REVIEW_NOTE = ("[E1] Anomaly flagged; grade REVIEW; defect type and cause unknown. "
                   "[G1] Prototype checklist: hold for human review; check image clarity and lighting.")

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        record = json.loads(Path("manifests/operator_app_verification_v1.json").read_text()
                            )["browser_proof"]["inspections"][1]["record"]
        record["model"]["jetson_verified"] = True
        record["source"]["physical_part"] = False
        self.record_path = root / "record.json"
        self.record_path.write_text(json.dumps(record, indent=2) + "\n")
        self.database = root / "inspections.db"
        from forgesight.storage import InspectionStore
        InspectionStore(self.database).save_inspection(record)
        self.output = root / "advice_out"
        self.socket_path = root / "assist.sock"

    def tearDown(self):
        self.dir.cleanup()

    def fake_worker(self, reply_report):
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen(1)
        connection, _ = server.accept()
        with connection:
            raw = b""
            while b"\n" not in raw:
                raw += connection.recv(65536)
            request = json.loads(raw.split(b"\n")[0])
            reply_report["request_seen"] = request
            payload = {"status": "OK", "report": {
                "status": "GENERATED_BY_PERSISTENT_WORKER",
                "worker_mode": "PERSISTENT_PROCESS",
                "worker_started_utc": "2026-11-14T00:00:00Z",
                "runtime_evidence": {"artifact": {"sha256": "m" * 64},
                                     "tokenizer": {"sha256": "t" * 64}},
                "prompt_token_count": 40,
                "stats": {"prompt_tokens": 40, "generated_tokens": 33},
                "generated_text": self.REVIEW_NOTE}}
            connection.sendall(json.dumps(payload).encode() + b"\n")
        server.close()

    def run_controller(self):
        import sys
        from unittest import mock
        from forgesight import assist_jetson_replay_once as replay
        argv = ["prog", "--record", str(self.record_path),
                "--database", str(self.database),
                "--guidance", "guidance/prototype_inspection_v1.json",
                "--llama-root", str(Path(self.dir.name) / "unused_llama_root"),
                "--output-dir", str(self.output),
                "--llama-worker-socket", str(self.socket_path)]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch("subprocess.run", side_effect=AssertionError("cold subprocess spawned")):
            replay.main()

    def test_worker_path_end_to_end(self):
        seen = {}
        thread = threading.Thread(target=self.fake_worker, args=(seen,), daemon=True)
        thread.start()
        self.run_controller()
        thread.join(timeout=10)
        report = json.loads((self.output / "advice_report.json").read_text())
        self.assertEqual(report["status"], "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW")
        self.assertEqual(report["llama_runtime_mode"], "PERSISTENT_WORKER")
        self.assertTrue(report["inspection_and_sqlite_unchanged"])
        self.assertEqual(report["native"]["model_sha256"], "m" * 64)
        self.assertIsNone(report["native"]["process_wall_seconds"])
        self.assertEqual(report["advisory"]["authoritative_grade"], "REVIEW")
        # The worker received only prompt/system text and the token budget:
        # the inspection record, paths and guidance never leave the host.
        self.assertEqual(set(seen["request_seen"]),
                         {"prompt_text", "system_text", "max_new_tokens"})
