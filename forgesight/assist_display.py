"""Read separate local advisory reports without changing inspection records."""

import json
from pathlib import Path

from forgesight.assist_contract import canonical_hash


def advice_for(record, root):
    if root is None:
        return {"status": "NOT_CONNECTED", "advisory": None}
    folder = Path(root) / record["inspection_id"]
    if not folder.is_dir():
        return {"status": "NOT_REQUESTED", "advisory": None}
    path = folder / "advice_report.json"
    if not path.is_file():
        return {"status": "PENDING", "advisory": None}
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Oversized advice report")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("inspection_id") != record["inspection_id"]
                or report.get("inspection_sha256") != canonical_hash(record)
                or report.get("decision_sha256") != canonical_hash(record["decision"])
                or report.get("authoritative_grade") != record["decision"]["grade"]
                or report.get("inspection_and_sqlite_unchanged") is not True
                or report.get("source_mode") != record["source"]["mode"]
                or report.get("source_mode") not in ("SAVED_IMAGE_REPLAY", "USER_UPLOADED_IMAGE_DIAGNOSTIC",
                                                      "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD")
                or report.get("physical_part_inspected") is not False):
            raise ValueError("Advice does not match saved inspection")
        if report.get("status") != "JETSON_ADVICE_GENERATED_REQUIRES_HUMAN_REVIEW":
            return {"status": "ADVICE_UNAVAILABLE", "advisory": None}
        advisory = report["advisory"]
        if (not isinstance(advisory, dict)
                or advisory.get("inspection_id") != record["inspection_id"]
                or advisory.get("inspection_sha256") != canonical_hash(record)
                or advisory.get("decision_sha256") != canonical_hash(record["decision"])
                or advisory.get("authoritative_grade") != record["decision"]["grade"]
                or advisory.get("quality_grade_authority") != "DETERMINISTIC_RULES_ONLY"
                or advisory.get("guidance_approval") != "PROTOTYPE_NOT_FACTORY_APPROVED"
                or advisory.get("jetson_verified") is not True
                or advisory.get("source_mode") != record["source"]["mode"]
                or not isinstance(report.get("native"), dict)
                or report["native"].get("status") != "JETSON_EXECUTORCH_LLAMA_GENERATION_VERIFIED"):
            raise ValueError("Advisory provenance is invalid")
        response = advisory.get("response")
        if not isinstance(response, dict):
            raise ValueError("Advisory response is invalid")
        note = response.get("generated_note")
        guide_id = "G2" if record["source"]["mode"] == "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD" else "G1"
        if (not isinstance(note, str) or not note.strip() or len(note) > 2000
                or response.get("evidence_ids") != ["E1", guide_id]
                or "[E1]" not in note or "[" + guide_id + "]" not in note):
            raise ValueError("Advisory note is invalid")
        return {"status": "GENERATED_REQUIRES_HUMAN_REVIEW", "advisory": {
            "note": note, "evidence_ids": response["evidence_ids"],
            "guidance_approval": advisory.get("guidance_approval"),
            "source_mode": advisory.get("source_mode"),
            "jetson_reported": True}}
    except (OSError, ValueError, TypeError, KeyError):
        return {"status": "ADVICE_UNAVAILABLE", "advisory": None}
