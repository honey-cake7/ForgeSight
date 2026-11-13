"""Device acceptance gate for the persistent advisory worker.

Compares one warm-path advisory (assist_jetson_replay_once with
--llama-worker-socket) against one cold-path advisory (without it) for the
SAME saved inspection on the Jetson. Adoption requires: identical inspection
hashes, both reports verified, and byte-identical generated notes - at
temperature 0 the persistent runner (reset per request) must reproduce the
cold process exactly; any difference points at state leaking across requests
or a runtime discrepancy and blocks adoption. Timing is recorded, never
asserted: this gate proves correctness, and the latency manifests prove speed.

Runs on the Jetson against real reports; the comparison logic is unit-tested
on PC with synthetic reports.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


VERIFIED = "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW"


def compare_reports(warm: dict, cold: dict) -> dict:
    checks = {}
    checks["same_inspection"] = (
        warm.get("inspection_sha256") is not None
        and warm.get("inspection_sha256") == cold.get("inspection_sha256"))
    checks["both_verified"] = warm.get("status") == cold.get("status") == VERIFIED
    warm_note = ((warm.get("advisory") or {}).get("response") or {}).get("generated_note")
    cold_note = ((cold.get("advisory") or {}).get("response") or {}).get("generated_note")
    checks["notes_byte_identical"] = warm_note is not None and warm_note == cold_note
    checks["grades_unchanged"] = (
        warm.get("authoritative_grade") == cold.get("authoritative_grade")
        and warm.get("decision_sha256") == cold.get("decision_sha256"))
    checks["immutability_rechecked"] = bool(
        warm.get("inspection_and_sqlite_unchanged") and cold.get("inspection_and_sqlite_unchanged"))
    warm_wall = (warm.get("controller_runtime_wall_ms")
                 or (warm.get("native") or {}).get("process_wall_seconds"))
    cold_wall = (cold.get("controller_runtime_wall_ms")
                 or (cold.get("native") or {}).get("process_wall_seconds"))
    verdict = {
        "checks": checks,
        "warm_note": warm_note, "cold_note": cold_note,
        "warm_mode": warm.get("llama_runtime_mode"),
        "cold_mode": cold.get("llama_runtime_mode"),
        "warm_wall": warm_wall, "cold_wall": cold_wall,
        "adoption_accepted": all(checks.values()),
    }
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-report", type=Path, required=True)
    parser.add_argument("--cold-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output manifest.")
    warm = json.loads(args.warm_report.read_text())
    cold = json.loads(args.cold_report.read_text())
    verdict = compare_reports(warm, cold)
    manifest = {"schema_version": 1,
                "status": "WARM_ADVISORY_ACCEPTED" if verdict["adoption_accepted"]
                else "WARM_ADVISORY_REJECTED",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "One saved inspection, warm vs cold advisory; correctness gate only",
                "verdict": verdict}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, allow_nan=False))
    if not verdict["adoption_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
