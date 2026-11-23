"""Read/export saved station history without involving the inspection workers."""

import base64
from contextlib import closing
import html
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

from forgesight.assist_contract import canonical_hash
from forgesight.assist_display import advice_for
from forgesight.storage import new_correction


def action_file(service):
    return Path(service.args.runs_root) / "history-actions.sqlite3"


def excluded(service, kind):
    path = action_file(service)
    if not path.is_file():
        return set()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        return {key for key, in db.execute("SELECT item FROM hidden WHERE kind = ?", (kind,))}


def conceal(service, kind, item):
    # Keep original records, corrections and evidence intact. This only
    # controls what is visible in the operator history.
    with closing(sqlite3.connect(str(action_file(service)))) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS hidden "
                   "(kind TEXT NOT NULL, item TEXT NOT NULL, PRIMARY KEY(kind,item))")
        db.execute("INSERT OR IGNORE INTO hidden VALUES (?, ?)", (kind, item))


def current_reviews(service, inspection_id):
    rows = service.store.corrections(inspection_id)
    superseded = {row.get("supersedes_correction_id") for row in rows}
    omitted = excluded(service, "review") | superseded
    return [row for row in rows if row["correction_id"] not in omitted]


def inspection(service, key):
    if str(UUID(key)) != key or key in excluded(service, "inspection"):
        raise ValueError("This inspection is unavailable in History.")
    record = service.store.get_inspection(key)
    if record is None:
        raise ValueError("This inspection is unavailable in History.")
    return record


def list_records(service, page):
    if not 0 <= page < 1000000:
        raise ValueError("Invalid History page.")
    hidden = excluded(service, "inspection")
    size = 20
    offset = page * size
    found = []
    seen = 0
    for source_offset in range(0, (offset + size + len(hidden) + 1), 100):
        batch = service.store.history(limit=100, offset=source_offset)
        for record in batch:
            if record["inspection_id"] in hidden:
                continue
            if seen >= offset:
                found.append({
                    "inspection_id": record["inspection_id"], "part_id": record["part_id"],
                    "timestamp_utc": record["timestamp_utc"], "status": record["status"],
                    "grade": record["decision"]["grade"], "source_mode": record["source"]["mode"],
                    "assessment": record.get("assessment"),
                })
            seen += 1
            if len(found) > size:
                return {"rows": found[:size], "more": True}
        if len(batch) < 100:
            break
    return {"rows": found, "more": False}


def stored_advice(service, record):
    root = service.advice_root
    report = root / "grounded" / (record["inspection_id"] + ".json")
    try:
        with report.open(encoding="utf-8") as f:
            saved = json.loads(f.read(128 * 1024))
        if (saved.get("inspection_sha256") == canonical_hash(record)
                and saved.get("status") in ("LLAMA_ADVICE", "CHECKLIST_FALLBACK")
                and isinstance(saved.get("advice"), str)):
            return {"status": saved["status"], "note": saved["advice"]}
    except (OSError, ValueError, AttributeError):
        pass
    legacy = advice_for(record, root)
    note = (legacy.get("advisory") or {}).get("note")
    return {"status": legacy.get("status"), "note": note}


def details(service, record):
    return {"inspection_id": record["inspection_id"], "part_id": record["part_id"],
            "timestamp_utc": record["timestamp_utc"], "grade": record["decision"]["grade"],
            "status": record["status"], "assessment": record.get("assessment"),
            "advice": stored_advice(service, record),
            "reviews": current_reviews(service, record["inspection_id"])}


def export_report(service, record):
    item = details(service, record)
    escape = lambda value: html.escape(str(value))
    images = []
    for kind, heading in (("source", "Original image"), ("overlay", "Anomaly overlay")):
        status, data = service.inspection_image(record["inspection_id"], kind)
        if status == 200 and data:
            mime = "image/jpeg" if data.startswith(b"\xff\xd8") else "image/png"
            image = base64.b64encode(data).decode("ascii")
            images.append(f'<figure><figcaption>{heading}</figcaption><img style="max-width:100%" alt="{heading}" src="data:{mime};base64,{image}"></figure>')
        else:
            images.append(f"<figure><figcaption>{heading}: unavailable</figcaption></figure>")
    advice = item["advice"]
    notes = escape(advice["note"] or "No advice was saved.").replace("\n", "<br>")
    review = []
    for row in item["reviews"]:
        review.append(f"<li><strong>{escape(row['actor'])}</strong> | {escape(row['timestamp_utc'])} | "
                      f"{escape(row.get('review_outcome') or 'Operator finding')}<p>{escape(row['reason'])}</p></li>")
    assessment = item.get("assessment") or {}
    severity = (assessment.get("severity") or {}).get("level", "Unassessed")
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>ForgeSight inspection report</title>'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<style>body{font:16px/1.55 system-ui;color:#20333b;background:#e8eff0;margin:0}'
            'main{max-width:70rem;margin:auto;padding:2rem}h1,h2{line-height:1.2}'
            'figure{margin:0;padding:.8rem;background:white;border-radius:.5rem}'
            'figcaption{font-weight:700;margin-bottom:.5rem}img{width:100%;height:auto}'
            '.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}'
            'li{margin:.75rem 0}@media print{body{background:white;font-size:12px}'
            'main{padding:0}h2{break-after:avoid}li,.images{break-inside:avoid}'
            'figure{padding:.3rem}figure img{width:100%;height:48mm;object-fit:contain}'
            '.images{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>'
            '<body><main><h1>ForgeSight inspection report</h1>'
            f"<p><strong>Inspection:</strong> {escape(item['part_id'])} | <time>{escape(item['timestamp_utc'])}</time></p>"
            f"<h2>Vision</h2><p>Grade {escape(item['grade'])}, status {escape(item['status'])}. "
            f"Defect type: {escape(assessment.get('defect_type','Unknown'))}. Severity: {escape(severity)}.</p>"
            + '<div class="images">' + "".join(images) + '</div>'
            + f"<h2>Llama or checklist advice</h2><p>{escape(advice['status'] or 'Unavailable')}</p><p>{notes}</p>"
            + "<h2>Operator review</h2>"
            + ("<ul>" + "".join(review) + "</ul>" if review else "<p>No operator review was saved.</p>")
            + "</main></body></html>")


def read_request(service, url):
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) == 2:
        return list_records(service, int(parse_qs(parsed.query).get("page", ["0"])[0])), "application/json"
    if len(parts) not in (3, 4) or (len(parts) == 4 and parts[3] != "report.html"):
        raise ValueError("Unknown history view.")
    record = inspection(service, parts[2])
    if len(parts) == 4:
        return export_report(service, record), "text/html; charset=utf-8"
    return details(service, record), "application/json"


def change_request(service, body):
    action = body["action"]
    key = body["inspection_id"]
    with service.lock:
        if service.busy:
            raise ValueError("Wait until the inspection is finished.")
        record = inspection(service, key)
        if action == "delete":
            conceal(service, "inspection", key)
            return {"status": "REMOVED_FROM_HISTORY"}
        target = None
        if action in ("edit_review", "delete_review"):
            target = body["correction_id"]
            if not any(row["correction_id"] == target for row in current_reviews(service, key)):
                raise ValueError("That review is no longer current.")
        if action == "delete_review":
            conceal(service, "review", target)
            return {"status": "REVIEW_REMOVED"}
        choices = {"no_visible_issue": "PASS", "visible_issue": "REJECT",
                   "uncertain_or_recapture": "REVIEW"}
        actor, reason, outcome = body["actor"].strip(), body["reason"].strip(), body["outcome"]
        if (action not in ("edit_review", "add_review") or record["status"] != "ANALYZED"
                or not re.fullmatch(r"[A-Za-z0-9 _.-]{1,64}", actor)
                or not 3 <= len(reason) <= 500 or outcome not in choices):
            raise ValueError("Enter an operator ID, finding and observation.")
        correction = new_correction(key, actor, reason, choices[outcome])
        correction["review_outcome"] = outcome
        if target is not None:
            correction["supersedes_correction_id"] = target
        service.store.save_correction(correction)
        return {"status": "REVIEW_UPDATED"}
