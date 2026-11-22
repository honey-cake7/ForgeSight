"""Generate one separate, evidence-bound Jetson ExecuTorch advisory for a saved replay."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from forgesight.assist_contract import AdvisoryRefusal, canonical_hash, parse_advisory, prepare_advisory
from forgesight.storage import InspectionStore


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Shared-request-deadline accounting: the caller (the live UI) kills this whole
# process at the absolute --request-deadline-monotonic instant it passes at
# spawn. Generation attempts are capped so the final report write still fits
# inside that kill; starting a generation that cannot finish would only leave
# another half-written output.
REPORT_WRITE_RESERVE_SECONDS = 5
MIN_VIABLE_GENERATION_SECONDS = 5


def _reject_non_finite(value):
    """json.loads parse_constant hook: NaN/Infinity in native output is not
    evidence. Python's json accepts them by default; reject them so they can
    never reach the strict report write (allow_nan=False) or the advisory."""
    raise ValueError(f"Non-finite number in native output: {value}")


def _generation_budget(deadline, cap):
    """Seconds one generation attempt may take within the shared deadline.

    Returns None when the remaining budget cannot fit even a bounded attempt:
    the caller then records the skip honestly instead of starting a run the
    outer kill would terminate mid-write.
    """
    if deadline is None:
        return cap
    remaining = deadline - time.monotonic() - REPORT_WRITE_RESERVE_SECONDS
    if remaining < MIN_VIABLE_GENERATION_SECONDS:
        return None
    return min(cap, remaining)


def _valid_sha256(value):
    return isinstance(value, str) and _SHA256_RE.match(value) is not None


def _valid_utc_timestamp(value):
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return False
    return True


def _pinned_asset_hashes(llama_root):
    """Expected (artifact, tokenizer) hash sets from the pinned manifests.

    Same trust root the cold path verifies against: the exported artifact hash
    comes from the export manifest, and the source-assets manifest (which pins
    the tokenizer) is itself pinned by hash from the export manifest. Returns
    None only when llama_root is None - a unit-test path; the CLI always
    supplies a root. With a root supplied, any missing, unreadable, malformed
    or inconsistent manifest fails closed: the gate is never silently weakened.
    """
    if llama_root is None:
        return None
    model_dir = Path(llama_root) / "model"
    try:
        export = json.loads((model_dir / "llama3_2_1b_export.json").read_text())
        assets_bytes = (model_dir / "llama3_2_1b_assets.json").read_bytes()
        assets = json.loads(assets_bytes)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Pinned export/assets manifests are missing or unreadable: {error}") from error
    if not isinstance(export, dict) or not isinstance(assets, dict):
        raise RuntimeError("Pinned export/assets manifests have an unexpected schema.")
    inputs = export.get("inputs_sha256")
    pinned_assets_file = (inputs.get("manifests/llama3_2_1b_assets.json")
                          if isinstance(inputs, dict) else None)
    if not _valid_sha256(pinned_assets_file):
        raise RuntimeError("Pinned export manifest does not pin the source-assets manifest.")
    if hashlib.sha256(assets_bytes).hexdigest() != pinned_assets_file:
        raise RuntimeError("Pinned export/assets manifests are inconsistent.")
    artifact_entry = export.get("artifact")
    artifact = artifact_entry.get("sha256") if isinstance(artifact_entry, dict) else None
    if not _valid_sha256(artifact):
        raise RuntimeError("Pinned export manifest has no valid artifact hash.")
    files = assets.get("files")
    tokenizer_entry = files.get("original/tokenizer.model") if isinstance(files, dict) else None
    tokenizer = tokenizer_entry.get("sha256") if isinstance(tokenizer_entry, dict) else None
    if not _valid_sha256(tokenizer):
        raise RuntimeError("Pinned source-assets manifest has no valid tokenizer hash.")
    return {artifact}, {tokenizer}


def _cold_fallback_within_budget(args, context, root, report, source_mode, deadline, mode):
    """Run the trusted cold path, bounded by the shared request deadline.

    A fresh full cold budget after a late worker failure can overrun the
    caller's outer kill, leaving no report and an output dir that blocks
    retry. The fallback therefore gets only the remaining budget; when that
    cannot fit a verified cold generation, the skip is recorded instead.
    """
    if deadline is None:
        # No shared deadline supplied: legacy full-budget fallback, unchanged.
        _generate_via_subprocess(args, context, root, report, source_mode)
        report["llama_runtime_mode"] = mode
        return
    timeout = _generation_budget(deadline, args.llama_timeout_seconds + 30)
    if timeout is None:
        report["runtime_error"] = ("Cold fallback skipped: the remaining request "
                                   "budget cannot fit a verified cold generation.")
        report["cold_fallback_skipped"] = "SHARED_REQUEST_DEADLINE"
        return
    _generate_via_subprocess(args, context, root, report, source_mode,
                             timeout_seconds=timeout)
    report["llama_runtime_mode"] = mode


def _generate_via_worker(args, context, root, report, source_mode, deadline=None):
    """Warm path: reuse the verified persistent worker; no process spawn or
    model reload per advisory. parse_advisory and the immutability re-check
    in main() are unchanged."""
    from forgesight.assist_edge_worker import request as worker_request
    report["llama_runtime_mode"] = "PERSISTENT_WORKER"
    started = time.perf_counter()
    timeout = _generation_budget(deadline, args.llama_timeout_seconds + 30)
    if timeout is None:
        report["runtime_error"] = ("The remaining request budget cannot fit "
                                   "the advisory generation.")
        return
    try:
        report["runtime_requested"] = True
        native = worker_request(args.llama_worker_socket,
                                {"prompt_text": context["prompt_text"],
                                 "system_text": context["system_text"],
                                 "max_new_tokens": context.get("max_new_tokens", 64)},
                                timeout_seconds=timeout)
        report["controller_runtime_wall_ms"] = (time.perf_counter() - started) * 1000
        evidence = native.get("runtime_evidence") or {}
        # Attestation gate: only a response carrying the worker's verified
        # startup evidence may be marked jetson_verified. A bare or fabricated
        # reply (no status, no mode, no artifact/tokenizer hashes) is a hard
        # failure, never a fallback: the cold path stays the trusted route.
        # These are operational trusted-local checks, not cryptographic
        # verification: a same-uid impostor can fabricate evidence, so the
        # acceptance gate's warm-vs-cold cross-check remains the binding.
        artifact_sha = (evidence.get("artifact") or {}).get("sha256")
        tokenizer_sha = (evidence.get("tokenizer") or {}).get("sha256")
        started_utc = native.get("worker_started_utc")
        if (native.get("status") != "GENERATED_BY_PERSISTENT_WORKER"
                or native.get("worker_mode") != "PERSISTENT_PROCESS"
                or not isinstance(started_utc, str)
                or not _valid_utc_timestamp(started_utc)
                or not _valid_sha256(artifact_sha)
                or not _valid_sha256(tokenizer_sha)):
            raise RuntimeError("Persistent worker response lacks verified runtime evidence; refusing to attest it.")
        pinned = _pinned_asset_hashes(args.llama_root)
        if pinned is not None:
            artifact_pins, tokenizer_pins = pinned
            if artifact_sha not in artifact_pins or tokenizer_sha not in tokenizer_pins:
                raise RuntimeError("Persistent worker evidence does not match the pinned asset manifest.")
        report["native"] = {"status": native.get("status"),
                            "worker_mode": native.get("worker_mode"),
                            "worker_started_utc": native.get("worker_started_utc"),
                            "model_sha256": (evidence.get("artifact") or {}).get("sha256"),
                            "tokenizer_sha256": (evidence.get("tokenizer") or {}).get("sha256"),
                            "prompt_tokens": native.get("prompt_token_count"),
                            "generation_stats": [native.get("stats")],
                            "process_wall_seconds": None,
                            "sampled_peak_child_rss_kib": None,
                            "timing_note": "persistent worker: no per-request process, import or model load to measure"}
        advice = parse_advisory(native["generated_text"], context)
        advice["jetson_verified"] = True
        advice["source_mode"] = source_mode
        report["advisory"] = advice
        report["status"] = "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW"
        (root / "advisory.json").write_text(json.dumps(advice, indent=2) + "\n")
    except (ValueError, TypeError, KeyError) as error:
        report["validation_error"] = str(error)
    except (ConnectionError, FileNotFoundError) as error:
        # Transport failure only: the worker is gone or unreachable. The cold
        # path re-verifies every asset itself, so tampering is never masked.
        # A worker-reported ERROR (validation or artifact change) above is a
        # hard failure and never falls back.
        report["persistent_worker_transport_error"] = type(error).__name__ + ": " + str(error)
        _cold_fallback_within_budget(args, context, root, report, source_mode, deadline,
                                     "PERSISTENT_WORKER_UNREACHABLE_COLD_FALLBACK")
    except TimeoutError as error:
        # The worker accepted the connection but stopped answering: it is
        # alive (the UI's poll() never respawns it) yet wedged inside
        # generation. Treat it like a transport failure - cold-fall-back
        # within the remaining shared budget - and flag the wedge so the UI
        # replaces the worker instead of leaving advice stuck until restart.
        report["persistent_worker_transport_error"] = type(error).__name__ + ": " + str(error)
        report["persistent_worker_unhealthy"] = True
        _cold_fallback_within_budget(args, context, root, report, source_mode, deadline,
                                     "PERSISTENT_WORKER_UNHEALTHY_COLD_FALLBACK")
    except (OSError, RuntimeError) as error:
        report["runtime_error"] = type(error).__name__ + ": " + str(error)


def _generate_via_subprocess(args, context, root, report, source_mode, timeout_seconds=None):
    """Cold path: one fresh native process chain per advisory. timeout_seconds
    defaults to the legacy full budget; the warm-path fallback passes the
    remaining shared request budget instead."""
    report["llama_runtime_mode"] = "COLD_SUBPROCESS"
    if timeout_seconds is None:
        timeout_seconds = args.llama_timeout_seconds + 30
    llama = args.llama_root.resolve()
    model_dir = llama / "model"
    command = [str(llama / "venv/bin/python"), "-B", str(model_dir / "edge_jetson_smoke.py"),
               "--root", str(llama), "--model", str(model_dir / "llama3_2_1b_xnnpack.pte"),
               "--tokenizer", str(model_dir / "tokenizer.model"),
               "--profile", str(model_dir / "edge_llama.json"),
               "--export", str(model_dir / "llama3_2_1b_export.json"),
               "--assets", str(model_dir / "llama3_2_1b_assets.json"),
               "--prompt", str(root / "prompt.txt"), "--system-prompt", str(root / "system.txt"),
               "--output", str(root / "native"), "--threads", "2",
               "--max-new-tokens", str(context.get("max_new_tokens", 64)),
               "--timeout", str(args.llama_timeout_seconds)]
    started = time.perf_counter()
    try:
        report["runtime_requested"] = True
        process = subprocess.run(command, capture_output=True, text=True,
                                 timeout=timeout_seconds)
        report["controller_runtime_wall_ms"] = (time.perf_counter() - started) * 1000
        native_path = root / "native/run.json"
        if process.returncode != 0 or not native_path.is_file():
            report["runtime_error"] = "ExecuTorch runner did not complete successfully."
        else:
            try:
                # parse_constant rejects NaN/Infinity: a non-finite number is
                # not evidence, and it must never reach the strict report
                # write (allow_nan=False), where it used to raise mid-dump
                # and leave a partial advice_report.json behind.
                native = json.loads(native_path.read_text(),
                                    parse_constant=_reject_non_finite)
                worker = native.get("worker_result") or {}
                report["native"] = {"run_sha256": sha256(native_path),
                                    "generated_sha256": sha256(root / "native/generated.txt"),
                                    "status": native["status"],
                                    "process_wall_seconds": native["process_wall_seconds"],
                                    "sampled_peak_child_rss_kib": native["sampled_peak_child_rss_kib"],
                                    "model_sha256": worker.get("artifact", {}).get("sha256"),
                                    "tokenizer_sha256": worker.get("tokenizer", {}).get("sha256"),
                                    "prompt_tokens": worker.get("prompt_token_count"),
                                    "generation_stats": worker.get("stats")}
            except (ValueError, TypeError, KeyError) as error:
                report["runtime_error"] = ("ExecuTorch native output was unreadable or "
                                           "carried a non-finite number: "
                                           + type(error).__name__)
            else:
                if native["status"] != "JETSON_EXECUTORCH_LLAMA_GENERATION_VERIFIED":
                    report["runtime_error"] = "ExecuTorch generation was not verified."
                else:
                    try:
                        advice = parse_advisory(worker["generated_text_display"], context)
                        advice["jetson_verified"] = True
                        advice["source_mode"] = source_mode
                        report["advisory"] = advice
                        report["status"] = "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW"
                        (root / "advisory.json").write_text(json.dumps(advice, indent=2) + "\n")
                    except (ValueError, TypeError, KeyError) as error:
                        report["validation_error"] = str(error)
    except (OSError, subprocess.TimeoutExpired) as error:
        report["runtime_error"] = type(error).__name__ + ": " + str(error)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--guidance", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llama-timeout-seconds", type=float, default=240)
    parser.add_argument("--llama-worker-socket", type=Path, default=None,
                        help="Persistent advisory worker socket; omit for the cold subprocess path")
    parser.add_argument("--request-deadline-monotonic", type=float, default=None,
                        help="Absolute time.monotonic() instant by which generation must finish "
                             "so the final report write still fits before the caller's kill. "
                             "The live UI passes its own outer deadline (llama timeout + 45 s "
                             "from spawn); an absolute instant keeps interpreter and import "
                             "time inside the shared budget, which an inner duration could not. "
                             "Omit for the legacy per-attempt budgets.")
    args = parser.parse_args()
    if not 0 < args.llama_timeout_seconds <= 240:
        parser.error("Llama timeout must be greater than zero and at most 240 seconds.")
    if args.request_deadline_monotonic is not None and (
            not math.isfinite(args.request_deadline_monotonic)
            or args.request_deadline_monotonic <= 0):
        # A NaN or infinite deadline would silently disable the shared cap
        # (NaN comparisons are false; min(cap, inf) is the legacy cap).
        parser.error("Request deadline must be a finite, positive monotonic timestamp.")
    if args.output_dir.exists():
        parser.error("Choose a new advisory output directory.")
    if args.record.stat().st_size > 1024 * 1024:
        parser.error("Inspection record exceeds the metadata size bound.")
    original_bytes = args.record.read_bytes()
    record = json.loads(original_bytes)
    source_mode = record.get("source", {}).get("mode")
    if (source_mode not in ("SAVED_IMAGE_REPLAY", "USER_UPLOADED_IMAGE_DIAGNOSTIC",
                            "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD")
            or record["source"].get("physical_part") is not False
            or record.get("model", {}).get("jetson_verified") is not True):
        parser.error("Jetson advisory requires a saved replay or uploaded-image diagnostic record.")
    stored = InspectionStore(args.database, create=False).get_inspection(record["inspection_id"])
    if stored != record:
        parser.error("The saved inspection differs from the immutable SQLite record.")
    guidance = json.loads(args.guidance.read_text())
    root = args.output_dir.resolve()
    root.mkdir(parents=True)
    (root / "input_inspection.json").write_bytes(original_bytes)
    deadline = args.request_deadline_monotonic
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "ADVICE_UNAVAILABLE", "source_mode": source_mode,
              "persistent_worker_unhealthy": False,
              "physical_part_inspected": False, "inspection_id": record["inspection_id"],
              "inspection_sha256": canonical_hash(record), "inspection_file_sha256": sha256(args.record),
              "authoritative_grade": record["decision"]["grade"],
              "decision_sha256": canonical_hash(record["decision"]),
              "guidance_sha256": canonical_hash(guidance),
              "runtime_requested": False, "runtime_error": None, "validation_error": None,
              "input_refusal": None, "advisory": None, "native": None,
              "inspection_and_sqlite_unchanged": False, "async_queue_implemented": False,
              "offline_disconnection_verified": False}
    try:
        context = prepare_advisory(record, guidance)
    except AdvisoryRefusal as error:
        report["input_refusal"] = {"code": error.code, "message": str(error)}
        context = None
    if context is not None:
        (root / "evidence.json").write_text(json.dumps(context, indent=2) + "\n")
        (root / "prompt.txt").write_text(context["prompt_text"], encoding="utf-8")
        (root / "system.txt").write_text(context["system_text"], encoding="utf-8")
        if args.llama_worker_socket is not None:
            _generate_via_worker(args, context, root, report, source_mode, deadline=deadline)
        else:
            timeout = _generation_budget(deadline, args.llama_timeout_seconds + 30)
            if timeout is None:
                report["runtime_error"] = ("The remaining request budget cannot fit "
                                           "a verified cold generation.")
            else:
                _generate_via_subprocess(args, context, root, report, source_mode,
                                         timeout_seconds=timeout)
    if (args.record.read_bytes() != original_bytes
            or InspectionStore(args.database, create=False).get_inspection(record["inspection_id"]) != record):
        raise RuntimeError("Original inspection changed while generating advice.")
    report["inspection_and_sqlite_unchanged"] = True
    # Serialize fully before touching the filesystem, then publish
    # atomically: a strict-JSON failure or an interrupted write must never
    # leave a partial advice_report.json that advice_for would half-read.
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    temp_report = root / "advice_report.json.tmp"
    with temp_report.open("x") as stream:
        stream.write(payload)
    os.replace(temp_report, root / "advice_report.json")
    print(json.dumps({key: report[key] for key in ("status", "authoritative_grade", "advisory",
                                               "input_refusal", "runtime_error", "validation_error",
                                               "persistent_worker_unhealthy")}, indent=2))
    if report["status"] != "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
