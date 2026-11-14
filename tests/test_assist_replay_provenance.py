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

    def run_warm(self, name, worker_report, llama_root=None):
        args = Namespace(llama_worker_socket=self.dir / f"{name}.sock",
                         llama_timeout_seconds=30, llama_root=llama_root)
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

    VALID_EVIDENCE = {"status": "GENERATED_BY_PERSISTENT_WORKER",
                      "worker_mode": "PERSISTENT_PROCESS",
                      "worker_started_utc": "2026-11-14T00:00:00Z",
                      "runtime_evidence": {"artifact": {"sha256": "a" * 64},
                                           "tokenizer": {"sha256": "b" * 64}},
                      "generated_text": NOTE}

    def manifest_root(self, name, export=None, assets=None):
        """A llama_root whose model dir holds exactly the given file bodies."""
        root = self.dir / (name + "-manifests")
        (root / "model").mkdir(parents=True)
        if export is not None:
            (root / "model" / "llama3_2_1b_export.json").write_text(export)
        if assets is not None:
            (root / "model" / "llama3_2_1b_assets.json").write_text(assets)
        return root

    def assert_gate_fails_closed(self, name, llama_root, needle="Pinned"):
        report, root = self.run_warm(name, dict(self.VALID_EVIDENCE), llama_root=llama_root)
        self.assertIsNone(report.get("advisory"))
        self.assertIn(needle, report.get("runtime_error", ""))
        self.assertNotIn("persistent_worker_transport_error", report)
        self.assertFalse((root / "advisory.json").exists())

    def test_no_root_keeps_test_only_presence_semantics(self):
        from forgesight.assist_jetson_replay_once import _pinned_asset_hashes
        self.assertIsNone(_pinned_asset_hashes(None))

    def test_missing_export_manifest_fails_closed(self):
        self.assert_gate_fails_closed("m-exp", self.manifest_root(
            "m-exp", assets=Path("manifests/llama3_2_1b_assets.json").read_text()))

    def test_missing_assets_manifest_fails_closed(self):
        self.assert_gate_fails_closed("m-ass", self.manifest_root(
            "m-ass", export=Path("manifests/llama3_2_1b_export.json").read_text()))

    def test_malformed_export_manifest_fails_closed(self):
        self.assert_gate_fails_closed("m-badexp", self.manifest_root(
            "m-badexp", export="{not json",
            assets=Path("manifests/llama3_2_1b_assets.json").read_text()))

    def test_malformed_assets_manifest_fails_closed(self):
        self.assert_gate_fails_closed("m-badass", self.manifest_root(
            "m-badass", export=Path("manifests/llama3_2_1b_export.json").read_text(),
            assets="{broken"))

    def test_non_object_manifest_schema_fails_closed(self):
        self.assert_gate_fails_closed("m-list", self.manifest_root(
            "m-list", export="[]",
            assets=Path("manifests/llama3_2_1b_assets.json").read_text()))

    def test_missing_chain_entry_fails_closed(self):
        export = json.loads(Path("manifests/llama3_2_1b_export.json").read_text())
        del export["inputs_sha256"]["manifests/llama3_2_1b_assets.json"]
        self.assert_gate_fails_closed("m-nochain", self.manifest_root(
            "m-nochain", export=json.dumps(export),
            assets=Path("manifests/llama3_2_1b_assets.json").read_text()),
            needle="does not pin")

    def test_invalid_artifact_hash_format_fails_closed(self):
        export = json.loads(Path("manifests/llama3_2_1b_export.json").read_text())
        export["artifact"]["sha256"] = "not-a-sha"
        self.assert_gate_fails_closed("m-badfmt", self.manifest_root(
            "m-badfmt", export=json.dumps(export),
            assets=Path("manifests/llama3_2_1b_assets.json").read_text()),
            needle="no valid artifact hash")

    def test_assets_without_valid_hashes_fails_closed(self):
        export = json.loads(Path("manifests/llama3_2_1b_export.json").read_text())
        assets = json.dumps({"files": {}})
        export["inputs_sha256"]["manifests/llama3_2_1b_assets.json"] = (
            hashlib.sha256(assets.encode()).hexdigest())
        self.assert_gate_fails_closed("m-empty", self.manifest_root(
            "m-empty", export=json.dumps(export), assets=assets),
            needle="no valid tokenizer hash")

    def test_unreadable_manifest_fails_closed(self):
        root = self.manifest_root(
            "m-unread", export=Path("manifests/llama3_2_1b_export.json").read_text(),
            assets=Path("manifests/llama3_2_1b_assets.json").read_text())
        (root / "model" / "llama3_2_1b_export.json").chmod(0)
        self.assert_gate_fails_closed("m-unread", root)

    def test_non_dict_inputs_sha256_fails_closed_not_crash(self):
        export = json.loads(Path("manifests/llama3_2_1b_export.json").read_text())
        export["inputs_sha256"] = ["truthy", "but", "not", "a", "mapping"]
        self.assert_gate_fails_closed("m-inputslist", self.manifest_root(
            "m-inputslist", export=json.dumps(export),
            assets=Path("manifests/llama3_2_1b_assets.json").read_text()),
            needle="does not pin")

    def test_non_dict_artifact_fails_closed_not_crash(self):
        export = json.loads(Path("manifests/llama3_2_1b_export.json").read_text())
        export["artifact"] = ["truthy", "not", "a", "mapping"]
        self.assert_gate_fails_closed("m-artlist", self.manifest_root(
            "m-artlist", export=json.dumps(export),
            assets=Path("manifests/llama3_2_1b_assets.json").read_text()),
            needle="no valid artifact hash")

    def test_tokenizer_matching_another_asset_entry_is_refused(self):
        # The gate binds the tokenizer to the exact original/tokenizer.model
        # entry, like verify_inputs does at worker startup: evidence quoting
        # a DIFFERENT pinned source asset's hash is not tokenizer evidence.
        export = json.loads(Path("manifests/llama3_2_1b_export.json").read_text())
        assets = json.loads(Path("manifests/llama3_2_1b_assets.json").read_text())
        params_sha = assets["files"]["original/params.json"]["sha256"]
        evidence = json.loads(json.dumps(self.VALID_EVIDENCE))
        evidence["runtime_evidence"] = {
            "artifact": {"sha256": export["artifact"]["sha256"]},
            "tokenizer": {"sha256": params_sha}}
        report, root = self.run_warm("m-tokid", evidence, llama_root=self.manifest_root(
            "m-tokid", export=json.dumps(export),
            assets=Path("manifests/llama3_2_1b_assets.json").read_text()))
        self.assertIsNone(report.get("advisory"))
        self.assertIn("does not match the pinned asset manifest",
                      report.get("runtime_error", ""))
        self.assertFalse((root / "advisory.json").exists())

    def test_tokenizer_entry_missing_fails_closed(self):
        export = json.loads(Path("manifests/llama3_2_1b_export.json").read_text())
        assets = json.loads(Path("manifests/llama3_2_1b_assets.json").read_text())
        del assets["files"]["original/tokenizer.model"]
        assets = json.dumps(assets)
        export["inputs_sha256"]["manifests/llama3_2_1b_assets.json"] = (
            hashlib.sha256(assets.encode()).hexdigest())
        self.assert_gate_fails_closed("m-notok", self.manifest_root(
            "m-notok", export=json.dumps(export), assets=assets),
            needle="no valid tokenizer hash")

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
