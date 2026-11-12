"""Queue one saved-replay Jetson advisory without blocking inspection reads."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from forgesight.assist_queue import AdvisoryQueue
from forgesight.storage import InspectionStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("record", "database", "guidance", "llama-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    inspection_id = record["inspection_id"]
    command = [sys.executable, "-B", "-m", "forgesight.assist_jetson_replay_once"]
    for name in ("record", "database", "guidance", "llama_root", "output_dir"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])

    queue = AdvisoryQueue()
    try:
        started = time.perf_counter()
        future = queue.submit(lambda: subprocess.run(command, capture_output=True,
                                                      text=True, timeout=300))
        submit_ms = (time.perf_counter() - started) * 1000
        busy_started = time.perf_counter()
        busy_refused = queue.submit(lambda: "duplicate") is None
        busy_ms = (time.perf_counter() - busy_started) * 1000
        stored = InspectionStore(args.database, create=False).get_inspection(inspection_id)
        readable_while_pending = stored == record and not future.done()
        result = future.result(timeout=305)
        report_path = args.output_dir / "advice_report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else None
        proof = {"schema_version": 1, "source_mode": "SAVED_IMAGE_REPLAY",
                 "submit_ms": submit_ms, "busy_refusal_ms": busy_ms,
                 "busy_refused": busy_refused,
                 "inspection_readable_while_pending": readable_while_pending,
                 "inspection_unchanged_after":
                 InspectionStore(args.database, create=False).get_inspection(inspection_id) == record,
                 "worker_returncode": result.returncode if isinstance(result, subprocess.CompletedProcess) else None,
                 "advice_status": report["status"] if report else "ADVICE_UNAVAILABLE",
                 "runtime_error": report["runtime_error"] if report else
                 (result.get("error") if isinstance(result, dict) else "NO_REPORT")}
        if report:
            with (args.output_dir / "queue_probe.json").open("x") as stream:
                json.dump(proof, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(proof, indent=2))
        if not (busy_refused and readable_while_pending and proof["inspection_unchanged_after"]
                and report and report["inspection_and_sqlite_unchanged"]):
            raise SystemExit(1)
    finally:
        queue.close()


if __name__ == "__main__":
    main()
