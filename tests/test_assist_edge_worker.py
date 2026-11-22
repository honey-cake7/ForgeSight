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
        self.events = []

    def reset(self):
        self.events.append("reset")

    def artifact_state(self):
        info = self.artifact.stat()
        return (str(self.artifact), info.st_size, info.st_mtime_ns, info.st_ino)

    def token_count(self, text):
        return len(text.split())

    def generate(self, formatted, max_new_tokens):
        self.events.append("generate")
        self.generate_calls += 1
        self.seen_prompts.append(formatted)
        user_line = [line for line in formatted.splitlines()
                     if line and not line.startswith("<|")][-1]
        text = f"note for: {user_line}"
        tokens = min(len(text.split()), max_new_tokens)
        return text, {"prompt_tokens": len(formatted.split()),
                      "generated_tokens": tokens}


class DecodeCountRuntime(StandInRuntime):
    """Follows the real runner's counting contract: prefill emits the first
    token and stats["generated_tokens"] counts only the max_new_tokens - 1
    decode steps (so a 1-token request reports 0)."""

    def generate(self, formatted, max_new_tokens):
        self.events.append("generate")
        self.generate_calls += 1
        self.seen_prompts.append(formatted)
        decoded = max(0, min(5, max_new_tokens - 1))
        return "note", {"prompt_tokens": len(formatted.split()),
                        "generated_tokens": decoded}


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

    def test_reset_precedes_every_generate(self):
        self.ask("hold part for review")
        self.ask("check lighting and clarity")
        self.assertEqual(self.runtime.events, ["reset", "generate", "reset", "generate"])

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
            self.ask(" ".join(["word"] * 545), tokens=1)  # over the context envelope itself

    def test_prompt_above_max_seq_length_uses_chunked_prefill(self):
        # max_seq_length is the prefill chunk size, not a prompt bound: the
        # native runner chunks longer prompts. The KV context envelope is the
        # real bound (prompt + max_new_tokens <= max_context_length), kept
        # because the native decode cap ignores consumed prompt position.
        self.runtime.profile = {"max_seq_length": 128, "max_context_length": 512}
        prompt = " ".join(["word"] * 300)   # ~302 stand-in tokens, above 128
        result = self.ask(prompt, tokens=8)  # 302 + 8 <= 512: accepted
        self.assertIn("generated_text", result)
        with self.assertRaises(ValueError):
            self.ask(" ".join(["word"] * 510), tokens=8)  # ~512 + 8 > 512

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

    def start_server(self, worker=None, **kwargs):
        if worker is None:
            worker = aew.Worker(self.runtime)
        thread = threading.Thread(
            target=aew.serve,
            args=(self.socket_path, worker),
            kwargs=kwargs, daemon=True)
        thread.start()
        # The socket file appears at bind(), before listen(): probe-connect
        # (retried) so callers never race the listen backlog.
        for _ in range(200):
            if not self.socket_path.exists():
                time.sleep(0.01)
                continue
            try:
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                probe.connect(str(self.socket_path))
                probe.close()
                return thread
            except (ConnectionRefusedError, FileNotFoundError):
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

        def wait_dropped(connection):
            # Deterministic: wait until the worker closes the connection.
            connection.settimeout(5)
            while connection.recv(65536):
                pass

        # Stalled: connects, sends nothing; the total deadline drops it.
        stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stalled.connect(str(self.socket_path))
        wait_dropped(stalled)
        stalled.close()
        # Dripping: one byte, then silence past the total deadline.
        drip = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        drip.connect(str(self.socket_path))
        drip.sendall(b"{")
        wait_dropped(drip)
        drip.close()
        report = aew.request(self.socket_path,
                             {"prompt_text": "check lighting", "max_new_tokens": 16},
                             timeout_seconds=5)
        self.assertIn("check lighting", report["generated_text"])

    def test_reply_write_uses_fresh_timeout_after_slow_read(self):
        # Deterministic leftover-vs-fresh discriminator: a request dripped in
        # over ~0.6s of the 0.8s read deadline leaves ~0.2s on the socket. The
        # reply (2 MB) far exceeds the socket buffers and the client delays
        # reading 0.5s, so sendall blocks. Leftover budget loses the reply
        # (timeout at ~0.2s, before the client reads at ~0.5s); the restored
        # fresh timeout covers the blocked write and the full reply arrives.
        class PadWorker(aew.Worker):
            def handle(self, request):
                report = super().handle(request)
                report["pad"] = "x" * 2_000_000
                return report

        self.start_server(worker=PadWorker(self.runtime),
                          request_read_timeout_seconds=0.8)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            client.connect(str(self.socket_path))
            body = json.dumps({"prompt_text": "hi", "max_new_tokens": 4},
                              separators=(",", ":")).encode() + b"\n"
            for byte in body:
                client.sendall(bytes([byte]))
                time.sleep(0.015)
            time.sleep(0.5)  # server sendall blocks on full buffers
            client.settimeout(10)
            chunks = b""
            while b"\n" not in chunks:
                block = client.recv(65536)
                if not block:
                    break
                chunks += block
        finally:
            client.close()
        payload = json.loads(chunks.split(b"\n", 1)[0].decode("utf-8"))
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(len(payload["report"]["pad"]), 2_000_000)

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

    REVIEW_NOTE = ("An anomaly is flagged; type and cause are unknown [E1]. "
                   "Severity HIGH, prototype pixel-response severity, not physical-damage severity. "
                   "Anomaly score 0.682809 vs threshold 0.227766. "
                   "Response area 9015 px across 2 regions; largest region width 139 px height 116 px "
                   "at x 512 y 361 in the right area of the frame. "
                   "Hold for human review; check clarity and lighting; compare the highlighted "
                   "region with a known-good part [G1].")

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

    def fake_worker(self, reply_report, evidence=None):
        evidence = evidence or {"artifact": {"sha256": "1" * 64},
                                "tokenizer": {"sha256": "2" * 64}}
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
                "runtime_evidence": evidence,
                "prompt_token_count": 40,
                "stats": {"prompt_tokens": 40, "generated_tokens": 33},
                "generated_text": self.REVIEW_NOTE}}
            connection.sendall(json.dumps(payload).encode() + b"\n")
        server.close()

    def run_controller(self, llama_root=None):
        import sys
        from unittest import mock
        from forgesight import assist_jetson_replay_once as replay
        argv = ["prog", "--record", str(self.record_path),
                "--database", str(self.database),
                "--guidance", "guidance/prototype_inspection_v3.json",
                "--llama-root", str(llama_root or Path(self.dir.name) / "unused_llama_root"),
                "--output-dir", str(self.output),
                "--llama-worker-socket", str(self.socket_path)]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch("subprocess.run", side_effect=AssertionError("cold subprocess spawned")):
            replay.main()

    def shipped_llama_root(self):
        """A llama_root carrying the real shipped manifests (not synthetic hashes)."""
        import shutil
        llama_root = Path(self.dir.name) / "shipped_llama"
        (llama_root / "model").mkdir(parents=True)
        for name in ("llama3_2_1b_export.json", "llama3_2_1b_assets.json"):
            shutil.copy(Path("manifests") / name, llama_root / "model" / name)
        return llama_root

    def shipped_evidence(self, artifact_sha=None):
        export = json.loads(Path("manifests/llama3_2_1b_export.json").read_text())
        assets = json.loads(Path("manifests/llama3_2_1b_assets.json").read_text())
        return {"artifact": {"sha256": artifact_sha or export["artifact"]["sha256"]},
                "tokenizer": {"sha256": assets["files"]["original/tokenizer.model"]["sha256"]}}

    def test_shipped_manifest_gate_accepts_real_evidence(self):
        # The pinned-asset gate must accept the hashes the shipped manifests
        # actually record: the exported PTE hash lives in the export manifest,
        # the tokenizer hash in the source-assets manifest it pins by hash.
        seen = {}
        thread = threading.Thread(target=self.fake_worker,
                                  args=(seen, self.shipped_evidence()), daemon=True)
        thread.start()
        self.run_controller(llama_root=self.shipped_llama_root())
        thread.join(timeout=10)
        report = json.loads((self.output / "advice_report.json").read_text())
        self.assertEqual(report["status"], "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW")
        self.assertEqual(report["llama_runtime_mode"], "PERSISTENT_WORKER")
        self.assertEqual(report["native"]["model_sha256"], self.shipped_evidence()["artifact"]["sha256"])

    def test_shipped_manifest_gate_rejects_unpinned_evidence(self):
        # Same shipped manifests, but the worker reports an artifact hash the
        # export manifest does not pin: hard failure, never a cold fallback.
        args_seen = {}
        thread = threading.Thread(target=self.fake_worker,
                                  args=(args_seen, self.shipped_evidence(artifact_sha="3" * 64)),
                                  daemon=True)
        thread.start()
        with self.assertRaises(SystemExit):  # withheld advice exits nonzero
            self.run_controller(llama_root=self.shipped_llama_root())
        thread.join(timeout=10)
        report = json.loads((self.output / "advice_report.json").read_text())
        self.assertEqual(report["status"], "ADVICE_UNAVAILABLE")
        self.assertIn("pinned asset manifest", report["runtime_error"])
        self.assertNotIn("persistent_worker_transport_error", report)
        self.assertEqual(report["llama_runtime_mode"], "PERSISTENT_WORKER")

    def test_shipped_manifest_gate_fails_closed_on_inconsistent_pair(self):
        # A source-assets manifest whose hash differs from the one the export
        # manifest pins is tampering, not "manifest unavailable": fail closed.
        llama_root = self.shipped_llama_root()
        assets_path = llama_root / "model" / "llama3_2_1b_assets.json"
        assets_path.write_text(assets_path.read_text() + "\n")
        seen = {}
        thread = threading.Thread(target=self.fake_worker,
                                  args=(seen, self.shipped_evidence()), daemon=True)
        thread.start()
        with self.assertRaises(SystemExit):  # inconsistent manifests fail closed
            self.run_controller(llama_root=llama_root)
        thread.join(timeout=10)
        report = json.loads((self.output / "advice_report.json").read_text())
        self.assertEqual(report["status"], "ADVICE_UNAVAILABLE")
        self.assertIn("inconsistent", report["runtime_error"])
        self.assertNotIn("persistent_worker_transport_error", report)

    def test_worker_path_end_to_end(self):
        seen = {}
        thread = threading.Thread(target=self.fake_worker,
                                  args=(seen, self.shipped_evidence()), daemon=True)
        thread.start()
        self.run_controller(llama_root=self.shipped_llama_root())
        thread.join(timeout=10)
        report = json.loads((self.output / "advice_report.json").read_text())
        self.assertEqual(report["status"], "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW")
        self.assertEqual(report["llama_runtime_mode"], "PERSISTENT_WORKER")
        from forgesight.assist_display import advice_for
        record = json.loads(self.record_path.read_text())
        shown = advice_for(record, Path(self.dir.name) / "advice_root")
        self.assertEqual(shown["status"], "NOT_REQUESTED")  # sanity: wrong root
        import shutil
        root = Path(self.dir.name) / "advice_root"
        shutil.copytree(self.output, root / record["inspection_id"])
        shown = advice_for(record, root)
        self.assertEqual(shown["status"], "GENERATED_REQUIRES_HUMAN_REVIEW")
        self.assertEqual(shown["advisory"]["note"], self.REVIEW_NOTE)
        self.assertTrue(report["inspection_and_sqlite_unchanged"])
        self.assertEqual(report["native"]["model_sha256"], self.shipped_evidence()["artifact"]["sha256"])
        self.assertIsNone(report["native"]["process_wall_seconds"])
        self.assertEqual(report["advisory"]["authoritative_grade"], "REVIEW")
        # The worker received only prompt/system text and the token budget:
        # the inspection record, paths and guidance never leave the host.
        self.assertEqual(set(seen["request_seen"]),
                         {"prompt_text", "system_text", "max_new_tokens"})

    def reply_server(self, body):
        """Accept one request and send back the exact bytes given."""
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen(1)
        connection, _ = server.accept()
        with connection:
            data = b""
            while b"\n" not in data:
                block = connection.recv(65536)
                if not block:
                    break
                data += block
            connection.sendall(body)
        server.close()

    def run_controller_with_mocked_cold(self):
        """Run the controller with the cold subprocess path mocked; return (cold, report)."""
        import sys
        from unittest import mock
        from forgesight import assist_jetson_replay_once as replay
        argv = ["prog", "--record", str(self.record_path),
                "--database", str(self.database),
                "--guidance", "guidance/prototype_inspection_v3.json",
                "--llama-root", str(Path(self.dir.name) / "unused_llama_root"),
                "--output-dir", str(self.output),
                "--llama-worker-socket", str(self.socket_path)]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(replay, "_generate_via_subprocess") as cold:
            with self.assertRaises(SystemExit):  # cold path is mocked, so no advice completes
                replay.main()
        return cold, json.loads((self.output / "advice_report.json").read_text())

    def test_truncated_reply_falls_back_cold(self):
        # Transport corruption (a reply cut mid-JSON) is not a validation
        # failure: it takes the transport-only cold fallback.
        thread = threading.Thread(target=self.reply_server,
                                  args=(b'{"status": "OK", "report": {"status": "GENERATED_BY_PERSIST',),
                                  daemon=True)
        thread.start()
        cold, report = self.run_controller_with_mocked_cold()
        thread.join(timeout=10)
        cold.assert_called_once()
        self.assertEqual(report["llama_runtime_mode"], "PERSISTENT_WORKER_UNREACHABLE_COLD_FALLBACK")
        self.assertIn("persistent_worker_transport_error", report)
        self.assertIsNone(report["validation_error"])

    def test_non_object_reply_fails_closed(self):
        # Syntactically valid JSON that is not the protocol object is not
        # transport corruption: hard failure, no fallback.
        thread = threading.Thread(target=self.reply_server, args=(b'[1, 2, 3]\n',), daemon=True)
        thread.start()
        cold, report = self.run_controller_with_mocked_cold()
        thread.join(timeout=10)
        cold.assert_not_called()
        self.assertIn("not a JSON object", report["runtime_error"])
        self.assertNotIn("persistent_worker_transport_error", report)

    def test_statusless_reply_fails_closed(self):
        # A valid JSON object without the protocol status is untrusted
        # content, not transport corruption: hard failure, no fallback.
        thread = threading.Thread(target=self.reply_server, args=(b'{"unexpected": true}\n',), daemon=True)
        thread.start()
        cold, report = self.run_controller_with_mocked_cold()
        thread.join(timeout=10)
        cold.assert_not_called()
        self.assertIn("Advisory worker error", report["runtime_error"])
        self.assertNotIn("persistent_worker_transport_error", report)

    def test_transport_failure_falls_back_cold(self):
        # No worker listening: connect raises FileNotFoundError, the cold
        # subprocess path runs instead, and the mode is recorded honestly.
        import sys
        from unittest import mock
        from forgesight import assist_jetson_replay_once as replay
        argv = ["prog", "--record", str(self.record_path),
                "--database", str(self.database),
                "--guidance", "guidance/prototype_inspection_v3.json",
                "--llama-root", str(Path(self.dir.name) / "unused_llama_root"),
                "--output-dir", str(self.output),
                "--llama-worker-socket", str(self.socket_path)]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(replay, "_generate_via_subprocess") as cold:
            with self.assertRaises(SystemExit):  # cold path is mocked, so no advice completes
                replay.main()
        cold.assert_called_once()
        report = json.loads((self.output / "advice_report.json").read_text())
        self.assertEqual(report["llama_runtime_mode"], "PERSISTENT_WORKER_UNREACHABLE_COLD_FALLBACK")
        self.assertIn("persistent_worker_transport_error", report)

    def test_worker_error_never_falls_back(self):
        # A worker-reported ERROR (validation or artifact tamper) is a hard
        # failure: no cold retry that could mask it.
        def error_server():
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(self.socket_path))
            server.listen(1)
            connection, _ = server.accept()
            with connection:
                while b"\n" not in (raw := connection.recv(65536)):
                    if not raw:
                        break
                payload = {"status": "ERROR",
                           "error": {"type": "RuntimeError",
                                     "message": "Model/tokenizer/runtime artifact changed; restart the worker."}}
                connection.sendall(json.dumps(payload).encode() + b"\n")
            server.close()

        import sys
        from unittest import mock
        from forgesight import assist_jetson_replay_once as replay
        thread = threading.Thread(target=error_server, daemon=True)
        thread.start()
        argv = ["prog", "--record", str(self.record_path),
                "--database", str(self.database),
                "--guidance", "guidance/prototype_inspection_v3.json",
                "--llama-root", str(Path(self.dir.name) / "unused_llama_root"),
                "--output-dir", str(self.output),
                "--llama-worker-socket", str(self.socket_path)]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(replay, "_generate_via_subprocess") as cold:
            with self.assertRaises(SystemExit):
                replay.main()
        thread.join(timeout=10)
        cold.assert_not_called()
        report = json.loads((self.output / "advice_report.json").read_text())
        self.assertEqual(report["status"], "ADVICE_UNAVAILABLE")
        self.assertIn("artifact changed", report["runtime_error"])


class WarmupTests(unittest.TestCase):
    """Boot warmup absorbs the lazy first-generation cost before READY."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.artifact = Path(self.dir.name) / "model.pte"
        self.artifact.write_bytes(b"stand-in model bytes")
        self.runtime = StandInRuntime(self.artifact)
        self.worker = aew.Worker(self.runtime)

    def tearDown(self):
        self.dir.cleanup()

    def test_warmup_generates_once_through_the_guarded_path(self):
        evidence = aew.warm_worker(self.worker)
        self.assertEqual(self.runtime.generate_calls, 1)
        # The warmup goes through handle(): runner state was reset first, and
        # the two-token warmup bound reached the runtime.
        self.assertEqual(self.runtime.events, ["reset", "generate"])
        self.assertGreaterEqual(evidence["warmup_generated_tokens"], 1)
        self.assertGreaterEqual(evidence["warmup_generate_ms"], 0.0)
        self.assertGreater(evidence["warmup_prompt_tokens"], 0)

    def test_warmup_passes_under_the_real_decode_count_contract(self):
        # Red before the fix: a 1-token warmup reports 0 decoded tokens on the
        # real runner and the strict check rejected it, so READY never came.
        worker = aew.Worker(DecodeCountRuntime(self.artifact))
        evidence = aew.warm_worker(worker)
        self.assertEqual(evidence["warmup_generated_tokens"], 1)

    def test_warmup_bound_stays_inside_the_exported_context(self):
        prompt = aew.chat_prompt(aew.WARMUP_REQUEST["prompt_text"],
                                 aew.WARMUP_REQUEST["system_text"])
        tokens = self.runtime.token_count(prompt)
        self.assertGreaterEqual(aew.WARMUP_REQUEST["max_new_tokens"], 2)
        self.assertLessEqual(aew.WARMUP_REQUEST["max_new_tokens"], aew.MAX_NEW_TOKENS)
        self.assertLess(tokens, PROFILE["max_seq_length"])
        self.assertLessEqual(tokens + aew.WARMUP_REQUEST["max_new_tokens"],
                             PROFILE["max_context_length"])

    def test_warmup_rejected_before_runtime_when_only_one_slot_fits(self):
        # The runner can clamp the budget to the remaining context. The guard
        # must refuse before generating, not rely on the native clamp.
        runtime = DecodeCountRuntime(self.artifact)
        prompt = aew.chat_prompt(aew.WARMUP_REQUEST["prompt_text"],
                                 aew.WARMUP_REQUEST["system_text"])
        runtime.profile = {"max_seq_length": 512,
                           "max_context_length": runtime.token_count(prompt) + 1}
        worker = aew.Worker(runtime)
        with self.assertRaisesRegex(ValueError, "exceeds the exported context"):
            aew.warm_worker(worker)
        self.assertEqual(runtime.generate_calls, 0)

    def test_zero_decoded_tokens_still_fails_closed(self):
        # Strictness is unchanged: a runner that reports no generated tokens
        # (for example a stand-in that stops at prefill) never reports READY.
        worker = aew.Worker(DecodeCountRuntime(self.artifact))
        with self.assertRaisesRegex(ValueError, "token statistics"):
            worker.handle({"prompt_text": "x", "max_new_tokens": 1})

    def test_first_served_request_after_warmup_is_fresh(self):
        aew.warm_worker(self.worker)
        report = self.worker.handle({"prompt_text": "hold part for review",
                                     "max_new_tokens": 32})
        # A second reset preceded the real request; the discarded warmup
        # output cannot leak into it.
        self.assertEqual(self.runtime.events, ["reset", "generate", "reset", "generate"])
        self.assertIn("hold part", report["generated_text"])
        self.assertNotIn("warmup", report["generated_text"].lower())

    def test_warmup_failure_prevents_ready(self):
        def failing_generate(formatted, max_new_tokens):
            raise RuntimeError("native load failed")
        self.runtime.generate = failing_generate
        with self.assertRaises(RuntimeError):
            aew.warm_worker(self.worker)
