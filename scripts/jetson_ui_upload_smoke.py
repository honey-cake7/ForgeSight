"""Exercise the running loopback UI with one original development-preview PNG."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from forgesight.storage import InspectionStore


DATASET_PATH = "test/bent/017.png"
EXPECTED_SHA256 = "b49cc898143c46a8c3a8697411d04a078b36f60f4798ccc68aea33ea8fcac026"


def get_json(url):
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    root = args.prefix.resolve()
    image = root / "data/mvtec_ad/metal_nut" / DATASET_PATH
    content = image.read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Original image SHA-256 differs from frozen manifest")
    base = "http://127.0.0.1:%d" % args.port
    initial = get_json(base + "/api/state")
    if DATASET_PATH not in initial["replay_samples"] or initial["busy"]:
        raise RuntimeError("Running UI is busy or does not offer the intended sample")
    replay_id = "UPLOAD-BENT-" + uuid4().hex[:12]
    headers = {"Origin": base, "Content-Type": "image/png",
               "X-Replay-Id": replay_id, "X-Dataset-Path": DATASET_PATH}
    invalid = Request(base + "/api/replay", data=b"damaged-file", headers=headers,
                      method="POST")
    try:
        urlopen(invalid, timeout=10)
    except HTTPError as error:
        if error.code != 409:
            raise RuntimeError("Unexpected rejection status for mismatched bytes") from error
    else:
        raise RuntimeError("Mismatched image bytes were accepted")
    if get_json(base + "/api/state")["busy"]:
        raise RuntimeError("Rejected upload started a replay")
    start = time.perf_counter()
    request = Request(base + "/api/replay", data=content, headers=headers, method="POST")
    with urlopen(request, timeout=10) as response:
        if response.status != 202:
            raise RuntimeError("Original PNG upload was not accepted")
    latest = None
    for _ in range(120):
        state = get_json(base + "/api/state")
        candidate = state["latest"]
        if candidate and candidate.get("part_id") == replay_id and not state["busy"]:
            latest = candidate
            break
        time.sleep(0.25)
    if latest is None or latest["status"] != "ANALYZED":
        raise RuntimeError("Uploaded PNG did not produce an analyzed result: " + repr(latest))
    if (latest["source_mode"] != "SAVED_IMAGE_REPLAY" or latest["dataset_path"] != DATASET_PATH
            or latest["score"] <= latest["threshold"] or latest["candidate"] != "ANOMALY_CANDIDATE"):
        raise RuntimeError("Uploaded replay score or provenance did not match expectation")
    with urlopen(base + "/api/source.png", timeout=10) as response:
        if hashlib.sha256(response.read()).hexdigest() != EXPECTED_SHA256:
            raise RuntimeError("UI source image differs from original PNG")
    with urlopen(base + "/api/overlay.png", timeout=10) as response:
        overlay = response.read()
    if not overlay.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("UI did not serve a PNG response overlay")
    database = root / "runs/live_inspections_v1/inspections.sqlite3"
    record = InspectionStore(database, create=False).get_inspection(latest["inspection_id"])
    if (record is None or record["part_id"] != replay_id
            or record["source"]["physical_part"] is not False
            or record["score"] != latest["score"]):
        raise RuntimeError("SQLite record does not match the uploaded replay")
    report = {"status": "JETSON_UI_ORIGINAL_PNG_REPLAY_VERIFIED",
              "source_mode": latest["source_mode"], "dataset_path": DATASET_PATH,
              "image_sha256": EXPECTED_SHA256, "invalid_upload_rejected": True,
              "replay_id": replay_id, "inspection_id": latest["inspection_id"],
              "score": latest["score"], "threshold": latest["threshold"],
              "candidate": latest["candidate"], "grade": latest["grade"],
              "response_area_px": latest["response_area_px"],
              "source_area_px": latest["source_area_px"],
              "model_forward_ms": latest["model_forward_ms"],
              "upload_to_state_ms": (time.perf_counter() - start) * 1000,
              "overlay_sha256": hashlib.sha256(overlay).hexdigest(),
              "sqlite_exact_record": True, "physical_part_inspected": False,
              "new_accuracy_evidence": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
