"""Warm replay provenance gate: an unverified worker response must never be
attested as a Jetson-verified advisory (jetson_verified=True).

An impostor same-uid process can bind the worker socket path once the real
worker exits. Before this gate, any such server returning parseable text -
with no status, no worker mode and no verified artifact/tokenizer evidence -
was accepted and marked jetson_verified=True. These tests run the real
_generate_via_worker against a stub socket server; no model executes.
"""

import json
import hashlib
import socket
import threading
import unittest
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from forgesight.assist_contract import prepare_advisory
from forgesight.assist_jetson_replay_once import _generate_via_worker

NOTE = ("[E1] Anomaly flagged; grade REVIEW; defect type and cause unknown. "
        "[G1] Prototype checklist: hold for human review; check image clarity and lighting.")
VERIFIED = "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW"


def serve_once(sock_path, worker_report):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    connection, _ = server.accept()
    with connection:
        data = b""
        while b"\n" not in data:
            block = connection.recv(65536)
            if not block:
                break
            data += block
        connection.sendall(json.dumps({"status": "OK", "report": worker_report}).encode() + b"\n")
    server.close()


def write_pinned_manifests(model_dir, artifact_sha, tokenizer_sha):
    """A consistent synthetic manifest pair in the real shipped schema: the
    export manifest pins the artifact and, by hash, the source-assets manifest
    that pins the tokenizer."""
    assets_bytes = json.dumps({"files": {"original/tokenizer.model": {"sha256": tokenizer_sha}}}).encode()
    export = {"artifact": {"sha256": artifact_sha},
              "inputs_sha256": {"manifests/llama3_2_1b_assets.json":
                                hashlib.sha256(assets_bytes).hexdigest()}}
    (model_dir / "llama3_2_1b_assets.json").write_bytes(assets_bytes)
    (model_dir / "llama3_2_1b_export.json").write_text(json.dumps(export))


class WarmPathProvenanceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        record = deepcopy(json.loads(Path('manifests/operator_app_verification_v1.json').read_text())
                          ['browser_proof']['inspections'][1]['record'])
        guidance = json.loads(Path('guidance/prototype_inspection_v1.json').read_text())
        self.context = prepare_advisory(record, guidance)

    def tearDown(self):
        self._tmp.cleanup()

    def run_warm(self, name, worker_report):
        args = Namespace(llama_worker_socket=self.dir / f"{name}.sock",
                         llama_timeout_seconds=30, llama_root=None)
        report = {}
        root = self.dir / name
        root.mkdir()
        thread = threading.Thread(target=serve_once,
                                  args=(args.llama_worker_socket, worker_report), daemon=True)
        thread.start()
        _generate_via_worker(args, self.context, root, report, "SAVED_IMAGE_REPLAY")
        thread.join(timeout=5)
        return report, root

    def test_bare_impostor_response_is_refused(self):
        report, root = self.run_warm("bare", {"generated_text": NOTE})
        self.assertIsNone(report.get("advisory"))
        self.assertIn("runtime_error", report)
        self.assertFalse((root / "advisory.json").exists())

    def test_fabricated_provenance_strings_are_refused(self):
        # Status/mode strings alone are self-asserted; without the worker's
        # verified startup evidence they must not earn jetson_verified.
        report, root = self.run_warm("fabricated",
                                     {"status": "GENERATED_BY_PERSISTENT_WORKER",
                                      "worker_mode": "PERSISTENT_PROCESS",
                                      "generated_text": NOTE})
        self.assertIsNone(report.get("advisory"))
        self.assertIn("runtime_error", report)
        self.assertFalse((root / "advisory.json").exists())

    def test_malformed_hash_is_refused(self):
        report, root = self.run_warm("badhash",
                                     {"status": "GENERATED_BY_PERSISTENT_WORKER",
                                      "worker_mode": "PERSISTENT_PROCESS",
                                      "worker_started_utc": "2026-11-14T00:00:00Z",
                                      "runtime_evidence": {"artifact": {"sha256": "not-a-hash"},
                                                           "tokenizer": {"sha256": "b" * 64}},
                                      "generated_text": NOTE})
        self.assertIsNone(report.get("advisory"))
        self.assertIn("runtime_error", report)
        self.assertFalse((root / "advisory.json").exists())

    def test_malformed_start_time_is_refused(self):
        report, root = self.run_warm("badtime",
                                     {"status": "GENERATED_BY_PERSISTENT_WORKER",
                                      "worker_mode": "PERSISTENT_PROCESS",
                                      "worker_started_utc": "yesterday",
                                      "runtime_evidence": {"artifact": {"sha256": "a" * 64},
                                                           "tokenizer": {"sha256": "b" * 64}},
                                      "generated_text": NOTE})
        self.assertIsNone(report.get("advisory"))
        self.assertIn("runtime_error", report)
        self.assertFalse((root / "advisory.json").exists())

    def test_pinned_manifest_mismatch_is_refused(self):
        llama_root = self.dir / "llama"
        (llama_root / "model").mkdir(parents=True)
        write_pinned_manifests(llama_root / "model", "c" * 64, "d" * 64)
        args = Namespace(llama_worker_socket=self.dir / "pin.sock",
                         llama_timeout_seconds=30, llama_root=llama_root)
        report = {}
        root = self.dir / "pin-out"
        root.mkdir()
        thread = threading.Thread(target=serve_once,
                                  args=(args.llama_worker_socket,
                                        {"status": "GENERATED_BY_PERSISTENT_WORKER",
                                         "worker_mode": "PERSISTENT_PROCESS",
                                         "worker_started_utc": "2026-11-14T00:00:00Z",
                                         "runtime_evidence": {"artifact": {"sha256": "a" * 64},
                                                              "tokenizer": {"sha256": "b" * 64}},
                                         "generated_text": NOTE}),
                                  daemon=True)
        thread.start()
        _generate_via_worker(args, self.context, root, report, "SAVED_IMAGE_REPLAY")
        thread.join(timeout=5)
        self.assertIsNone(report.get("advisory"))
        self.assertIn("pinned asset manifest", report.get("runtime_error", ""))
        self.assertFalse((root / "advisory.json").exists())

    def test_pinned_manifest_match_is_accepted(self):
        llama_root = self.dir / "llama-ok"
        (llama_root / "model").mkdir(parents=True)
        write_pinned_manifests(llama_root / "model", "a" * 64, "b" * 64)
        args = Namespace(llama_worker_socket=self.dir / "pin-ok.sock",
                         llama_timeout_seconds=30, llama_root=llama_root)
        report = {}
        root = self.dir / "pin-ok-out"
        root.mkdir()
        thread = threading.Thread(target=serve_once,
                                  args=(args.llama_worker_socket,
                                        {"status": "GENERATED_BY_PERSISTENT_WORKER",
                                         "worker_mode": "PERSISTENT_PROCESS",
                                         "worker_started_utc": "2026-11-14T00:00:00Z",
                                         "runtime_evidence": {"artifact": {"sha256": "a" * 64},
                                                              "tokenizer": {"sha256": "b" * 64}},
                                         "generated_text": NOTE,
                                         "prompt_token_count": 40, "stats": {}}),
                                  daemon=True)
        thread.start()
        _generate_via_worker(args, self.context, root, report, "SAVED_IMAGE_REPLAY")
        thread.join(timeout=5)
        self.assertEqual(report.get("status"), VERIFIED)
        self.assertTrue(report["advisory"]["jetson_verified"])

    def test_full_worker_evidence_is_accepted(self):
        report, root = self.run_warm("full",
                                     {"status": "GENERATED_BY_PERSISTENT_WORKER",
                                      "worker_mode": "PERSISTENT_PROCESS",
                                      "worker_started_utc": "2026-11-14T00:00:00Z",
                                      "runtime_evidence": {"artifact": {"sha256": "a" * 64},
                                                           "tokenizer": {"sha256": "b" * 64}},
                                      "generated_text": NOTE,
                                      "prompt_token_count": 40, "stats": {}})
        self.assertEqual(report.get("status"), VERIFIED)
        self.assertTrue(report["advisory"]["jetson_verified"])
        self.assertEqual(report["native"]["model_sha256"], "a" * 64)
        self.assertTrue((root / "advisory.json").exists())


if __name__ == "__main__":
    unittest.main()
