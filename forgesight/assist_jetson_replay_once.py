"""Generate one separate, evidence-bound Jetson ExecuTorch advisory for a saved replay."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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

def _generate_via_worker(args, context, root, report, source_mode):
    """Warm path: reuse the verified persistent worker; no process spawn or
    model reload per advisory. parse_advisory and the immutability re-check
    in main() are unchanged."""
    from forgesight.assist_edge_worker import request as worker_request
    report["llama_runtime_mode"] = "PERSISTENT_WORKER"
    started = time.perf_counter()
    try:
        report["runtime_requested"] = True
        native = worker_request(args.llama_worker_socket,
                                {"prompt_text": context["prompt_text"],
                                 "system_text": context["system_text"],
                                 "max_new_tokens": 64},
                                timeout_seconds=args.llama_timeout_seconds + 30)
        report["controller_runtime_wall_ms"] = (time.perf_counter() - started) * 1000
        evidence = native.get("runtime_evidence") or {}
        report["native"] = {"worker_mode": native.get("worker_mode"),
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
    except (OSError, RuntimeError) as error:
        report["runtime_error"] = type(error).__name__ + ": " + str(error)


def _generate_via_subprocess(args, context, root, report, source_mode):
    """Cold path: one fresh native process chain per advisory (unchanged)."""
    report["llama_runtime_mode"] = "COLD_SUBPROCESS"
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
               "--max-new-tokens", "64", "--timeout", str(args.llama_timeout_seconds)]
    started = time.perf_counter()
    try:
        report["runtime_requested"] = True
        process = subprocess.run(command, capture_output=True, text=True,
                                 timeout=args.llama_timeout_seconds + 30)
        report["controller_runtime_wall_ms"] = (time.perf_counter() - started) * 1000
        native_path = root / "native/run.json"
        if process.returncode != 0 or not native_path.is_file():
            report["runtime_error"] = "ExecuTorch runner did not complete successfully."
        else:
            native = json.loads(native_path.read_text())
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
    args = parser.parse_args()
    if not 0 < args.llama_timeout_seconds <= 240:
        parser.error("Llama timeout must be greater than zero and at most 240 seconds.")
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
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "ADVICE_UNAVAILABLE", "source_mode": source_mode,
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
            _generate_via_worker(args, context, root, report, source_mode)
        else:
            _generate_via_subprocess(args, context, root, report, source_mode)
    if (args.record.read_bytes() != original_bytes
            or InspectionStore(args.database, create=False).get_inspection(record["inspection_id"]) != record):
        raise RuntimeError("Original inspection changed while generating advice.")
    report["inspection_and_sqlite_unchanged"] = True
    with (root / "advice_report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "authoritative_grade", "advisory",
                                               "input_refusal", "runtime_error", "validation_error")}, indent=2))
    if report["status"] != "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
