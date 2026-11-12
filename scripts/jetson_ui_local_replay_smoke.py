"""Verify the live UI can analyze a frozen PNG already stored on the Jetson."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from forgesight.storage import InspectionStore
from scripts.jetson_ui_upload_smoke import DATASET_PATH, EXPECTED_SHA256, get_json


def post(base, replay_id, dataset_path):
    request = Request(base + "/api/replay-local", method="POST",
                      data=json.dumps({"replay_id": replay_id,
                                       "dataset_path": dataset_path}).encode(),
                      headers={"Origin": base, "Content-Type": "application/json"})
    return urlopen(request, timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    root = args.prefix.resolve()
    base = "http://127.0.0.1:%d" % args.port
    initial = get_json(base + "/api/state")
    if DATASET_PATH not in initial["local_replay_samples"] or initial["busy"]:
        raise RuntimeError("Original Jetson PNG is unavailable or UI is busy")
    replay_id = "JETSON-COPY-" + uuid4().hex[:12]
    try:
        post(base, replay_id, "test/bent/000.png")
    except HTTPError as error:
        if error.code != 409:
            raise RuntimeError("Unexpected rejection for non-preview path") from error
    else:
        raise RuntimeError("Unapproved dataset path was accepted")
    started = time.perf_counter()
    with post(base, replay_id, DATASET_PATH) as response:
        if response.status != 202:
            raise RuntimeError("Jetson copy replay was not accepted")
    latest = None
    for _ in range(120):
        state = get_json(base + "/api/state")
        candidate = state["latest"]
        if candidate and candidate.get("part_id") == replay_id and not state["busy"]:
            latest = candidate
            break
        time.sleep(0.25)
    if (latest is None or latest["status"] != "ANALYZED"
            or latest["source_mode"] != "SAVED_IMAGE_REPLAY"
            or latest["dataset_path"] != DATASET_PATH
            or latest["score"] <= latest["threshold"]
            or latest["candidate"] != "ANOMALY_CANDIDATE"):
        raise RuntimeError("Jetson copy did not produce expected replay: " + repr(latest))
    with urlopen(base + "/api/source.png", timeout=10) as response:
        if hashlib.sha256(response.read()).hexdigest() != EXPECTED_SHA256:
            raise RuntimeError("Served source differs from original PNG")
    with urlopen(base + "/api/overlay.png", timeout=10) as response:
        overlay = response.read()
    if not overlay.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Saved overlay is not PNG")
    database = root / "runs/live_inspections_v1/inspections.sqlite3"
    record = InspectionStore(database, create=False).get_inspection(latest["inspection_id"])
    if (record is None or record["part_id"] != replay_id
            or record["source"]["physical_part"] is not False
            or record["score"] != latest["score"]):
        raise RuntimeError("SQLite record differs from the UI result")
    report = {"status": "JETSON_LOCAL_DATASET_REPLAY_VERIFIED",
              "source_mode": latest["source_mode"], "dataset_path": DATASET_PATH,
              "image_sha256": EXPECTED_SHA256, "non_preview_path_rejected": True,
              "replay_id": replay_id, "inspection_id": latest["inspection_id"],
              "score": latest["score"], "threshold": latest["threshold"],
              "candidate": latest["candidate"], "grade": latest["grade"],
              "response_area_px": latest["response_area_px"],
              "model_forward_ms": latest["model_forward_ms"],
              "request_to_state_ms": (time.perf_counter() - started) * 1000,
              "overlay_sha256": hashlib.sha256(overlay).hexdigest(),
              "sqlite_exact_record": True, "physical_part_inspected": False,
              "new_accuracy_evidence": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
