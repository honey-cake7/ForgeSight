"""Optional read-only operator advisory for a saved ForgeSight inspection."""

import argparse
import json
from pathlib import Path
import re
import time

from . import assist_edge_worker
from .assist_contract import AdvisoryRefusal, DEFAULT_GUIDANCE, canonical_hash, prepare_advisory
from .assist_jetson_replay_once import (
    _pinned_asset_hashes, _valid_sha256, _valid_utc_timestamp,
)
from .grade import grade_record


RULES = Path(__file__).resolve().parent.parent / "guidance/grounded_advice_v1.json"
GUIDANCE = RULES.parent.parent / DEFAULT_GUIDANCE
SYSTEM = "Choose the best first inspection check. Reply with the exact check."
TYPE_NAMES = {
    "scratch": "scratch",
    "bent": "bent nut",
    "flip": "flip",
    "discoloration": "discoloration",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(record, rules):
    if rules.get("schema_version") != 1:
        raise ValueError("Unsupported grounded advisory rules version.")
    source_mode = record.get("source", {}).get("mode")
    diagnostic = source_mode in ("LIVE_CAMERA", "LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC")
    if diagnostic:
        decision = record["decision"]
        area = (record.get("geometry") or {}).get("total_area_px")
        if (record.get("status") != "ANALYZED"
                or record.get("candidate") != "ANOMALY_CANDIDATE"
                or record.get("assessment") is not None
                or type(area) is not int or area <= 0
                or decision != grade_record(record, decision["policy_snapshot"])):
            raise AdvisoryRefusal("UNSUPPORTED_CAMERA_EVIDENCE", "Camera advice is withheld.")
        facts = {
            "type": "unknown", "severity": "UNASSESSED", "location": "uncertain",
            "response_region_area_px": area, "extent": None,
        }
    else:
        mode = "vision_choice" if record.get("assessment") is not None else "composed"
        context = prepare_advisory(record, read_json(GUIDANCE), mode=mode)
        evidence = context["evidence"]["E1"]
        facts = evidence.get("vision_assessment")
        if facts is None:
            region = evidence["largest_region"]
            facts = {
                "type": "unknown", "severity": "UNASSESSED", "location": region["location"],
                "response_region_area_px": region["area_px"], "extent": None,
            }

    defect_type = facts["type"]
    location = facts["location"]
    normal_type = TYPE_NAMES.get(defect_type)
    severity_str = facts.get("severity", "UNASSESSED").lower()
    area_px = facts.get("response_region_area_px")
    area_str = f" ({area_px} px)" if area_px else ""
    loc_str = f" on {location}" if location and location != "uncertain" else ""
    if normal_type:
        reason = f"Vision flagged possible {normal_type}{loc_str}{area_str} with {severity_str} severity. Hold part in Quarantine Bin B for review; do not release without QC sign-off."
    else:
        reason = f"Vision flagged an anomaly{loc_str}{area_str} with {severity_str} severity. Hold part in Quarantine Bin B for review; do not release without QC sign-off."

    rule = defect_type
    if defect_type != "flip" and location == "uncertain":
        rule = "uncertain"
    elif defect_type in ("scratch", "bent") and "inner rim" in location.split(" / "):
        rule = "inner_rim"
    if diagnostic:
        rule = "screen_photo" if source_mode == "LIVE_CAMERA_SCREEN_PHOTO_DIAGNOSTIC" else "camera"
        reason = "The camera result is diagnostic and the defect is unconfirmed. Hold part for manual review."
    actions = rules["checks"][rule]
    notes = {normalized(action): f"{action} {reason}" for action in actions}
    grade = record["decision"]["grade"]
    prompt = (
        f"Metal nut inspection. Possible {normal_type or 'anomaly'} on {location}.\n"
        + "Checks:\n" + "\n".join(actions) + "\nWhich is the best first check?"
    )
    request = {"prompt_text": prompt, "system_text": SYSTEM, "max_new_tokens": 36}
    report = {
        "inspection_id": record["inspection_id"],
        "inspection_sha256": canonical_hash(record),
        "source_mode": source_mode,
        "grade": grade,
        "vision": facts,
        "rule": rule,
        "rules_sha256": canonical_hash(rules),
        "approval_status": rules["approval_status"],
    }
    return request, notes, report


def normalized(text):
    return " ".join(text.split()).rstrip(".").casefold()


def selected_check(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    end_marker = "<|eot_id|>"
    if text.endswith(end_marker):
        text = text[:-len(end_marker)].rstrip()
    # Some replies wrap the first check in a predictable field with or without Markdown.
    # Only that field can influence the displayed text.
    field = re.match(r"^The best first (?:inspection )?check is:\s*(?:\*\*)?([^\r\n*]+)",
                     text, flags=re.IGNORECASE)
    return normalized(field.group(1) if field else text)


def advise(record, rules, socket, timeout, llama_root=None):
    try:
        request, notes, report = prepare(record, rules)
    except AdvisoryRefusal as error:
        return {"status": "INSPECTION_WITHHELD", "reason_code": error.code}

    report.update(status="CHECKLIST_FALLBACK", advice=next(iter(notes.values())),
                  action_origin="PROTOTYPE_CHECKLIST", reason_origin="SAVED_VISION_EVIDENCE",
                  llama_runtime_mode="NOT_REQUESTED")
    started = time.perf_counter()
    try:
        if socket is None:
            raise ConnectionError("The persistent Llama worker is not ready.")
        report["llama_runtime_mode"] = "PERSISTENT_WORKER_UNAVAILABLE"
        reply = assist_edge_worker.request(socket, request, timeout_seconds=timeout,
                                           connect_retries=1)
        if not isinstance(reply, dict):
            raise ValueError("Persistent worker report must be an object.")
        evidence = reply.get("runtime_evidence") or {}
        if (not isinstance(evidence, dict)
                or not isinstance(evidence.get("artifact"), dict)
                or not isinstance(evidence.get("tokenizer"), dict)):
            raise ValueError("Persistent worker report lacks asset evidence objects.")
        artifact = evidence["artifact"].get("sha256")
        tokenizer = evidence["tokenizer"].get("sha256")
        if (reply.get("status") != "GENERATED_BY_PERSISTENT_WORKER"
                or reply.get("worker_mode") != "PERSISTENT_PROCESS"
                or not _valid_utc_timestamp(reply.get("worker_started_utc"))
                or not _valid_sha256(artifact) or not _valid_sha256(tokenizer)):
            raise ValueError("Persistent worker response lacks verified runtime evidence.")
        pinned = _pinned_asset_hashes(llama_root)
        if pinned is not None and (artifact not in pinned[0] or tokenizer not in pinned[1]):
            raise ValueError("Persistent worker evidence does not match the pinned assets.")
        report["worker"] = {"worker_mode": reply["worker_mode"],
                            "worker_started_utc": reply["worker_started_utc"],
                            "model_sha256": artifact, "tokenizer_sha256": tokenizer}
        report["llama_runtime_mode"] = "PERSISTENT_WORKER"
        report["prompt_tokens"] = reply["prompt_token_count"]
        report["generation_ms"] = reply["generate_ms"]
        matching_note = notes.get(selected_check(reply["generated_text"]))
        if reply.get("status") == "GENERATED_BY_PERSISTENT_WORKER" and matching_note:
            report.update(status="LLAMA_ADVICE", advice=matching_note,
                          action_origin="LLAMA_MATCHED_APPROVED_CHECK")
        else:
            report["fallback_reason"] = "UNAPPROVED_MODEL_OUTPUT"
    except TimeoutError as error:
        report["fallback_reason"] = "WORKER_TIMEOUT"
        report["persistent_worker_unhealthy"] = True
        report["worker_error"] = str(error)
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as error:
        report["fallback_reason"] = "WORKER_UNAVAILABLE"
        report["worker_error"] = str(error)
    report["advice_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=8)
    parser.add_argument("--llama-root", type=Path,
                        default=Path.home() / ".local/share/forgesight/llama_cpu_v1")
    args = parser.parse_args()
    if not 0 < args.timeout <= 30:
        parser.error("--timeout must be above 0 and at most 30 seconds.")
    try:
        report = advise(read_json(args.record), read_json(RULES), args.socket, args.timeout,
                        llama_root=args.llama_root)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"Cannot read inspection or procedure: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
