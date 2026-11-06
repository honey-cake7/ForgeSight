# Local operator workflow — Stage 7

The operator screen now calls the real model, grading rules and SQLite save
path. It runs locally with FastAPI and Uvicorn and serves plain HTML/CSS/JS.
No frontend build, CDN, cloud service, hosted Site, camera or Llama advisory
worker is required for this stage. The Sites skill's local workflow was used;
this is local-only work under the project's offline architecture.

## Start and use it

From PowerShell in the project folder on this tested PC:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/app-v1/venv/bin/python -B -m forgesight.operator_app
```

Wait for `Application startup complete`, then open **http://127.0.0.1:8765**.
Keep that terminal open; Ctrl+C stops the server. The reviewed test server is
stopped at handoff. Launching uses the new app environment, not the old vision
environment, which does not contain FastAPI.

1. Enter a unique ID for a new, distinct declared part, such as `BENCH-001`.
2. Choose a saved photo and click **Inspect part**.
3. Review the actual source photo, highlighted response, unknown defect type,
   pixel area, provisional grade and deterministic operator action.
4. Browse/filter saved history and use **View** to reopen a record. Refreshing
   or restarting the server does not create another inspection.

The default working database is `artifacts/runtime/inspections.sqlite3`; outputs
are under `artifacts/runtime/operator/`. These may initially be empty. The proof
database is separate and contains the two verified browser inspections:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/app-v1/venv/bin/python -B -m forgesight.operator_app --database artifacts/app_stage7/proof_runtime/inspections.sqlite3 --output artifacts/app_stage7/proof_runtime
```

Appending to that proof database changes its recorded verification-time hash;
historical manifests and existing inspection JSON remain preserved. The CLI
also accepts `--port` and an explicit `--device cpu`. CPU mode is not a silent
fallback or an independently benchmarked deployment. The launcher binds to
loopback only, runs one server worker, and does not publish the app.

## What the screen does

The main task is inspecting one declared part. The source and response images
are side by side; saved history is below. The page is responsive and includes
keyboard labels, focus states and visible error/progress messages. On narrow
screens the history table scrolls within its container.

It clearly labels **SAVED-IMAGE REPLAY / PC**, provisional decisions, unknown
type, missing camera validation and disconnected Llama advice. Model response
area is pixels, not physical damage area or calibrated millimetres. It also
discloses the two known missed scratches. Counts come from SQLite, not hardcoded
factory statistics. Folder ground-truth labels are never reported as predicted
defect names. Guidance is the deterministic action saved with the grade, not
generated Llama text or a confirmed root cause.

The catalog contains one already exposed fitting normal plus the four frozen
development examples. It has no arbitrary server-path input, upload, live
camera or final-holdout selector. Unknown IDs/invalid requests are refused.
File/image failures in accepted inspection attempts are recorded as ERROR and
REVIEW by the existing real pipeline. Sample failures can be viewed in history;
missing images show a placeholder rather than a fabricated result.

The calibrated model loads once at startup, with explicit device selection.
One inspection runs at a time; concurrent submissions receive a clear busy
response instead of building an unbounded inference queue. Duplicate part IDs
are checked against SQLite before inference and stay blocked after restart.
Do not rename the same part to evade the gate. Database-write failures preserve
JSON and are reported as failures, using the Stage 6 manual import retry.

If model loading fails, health is degraded, history remains readable, and
inspection is disabled/refused. A missing model never produces a fake normal
score. Start-up model failures require fixing the cause and restarting.

## Local API

| Endpoint | Behavior |
| --- | --- |
| `GET /api/health` | Actual model availability/device and replay/advice status; 503 when model unavailable |
| `GET /api/config` | Read-only sample catalog, profile and frozen policy; no threshold tuning |
| `POST /api/inspect` | IDs only: `part_id`, `sample_id`; runs real inference/grade/save and returns the recorded result |
| `GET /api/latest` | Latest saved record, or null when empty |
| `GET /api/history` | Paginated saved records; optional grade/status filters |
| `GET /api/summary` | Original SQLite decision/status counts |
| `GET /api/inspections/{id}` | Full saved record plus permitted image URLs |
| `GET /api/inspections/{id}/source` | Recorded source image; checks saved SHA-256 |
| `GET /api/inspections/{id}/overlay` | Saved response overlay; new records include its SHA-256 |

Missing/changed media returns explicit 404/409 responses. Images are served only
from the configured data root or project artifacts and only through saved
record references. Older overlays without an overlay digest remain historical
references; the source hash is still checked. Mutations reject cross-origin
browser requests, and hostnames are limited to localhost/127.0.0.1. The HTML,
CSS, JS, icon and images are local. Swagger/ReDoc CDN pages are disabled;
the OpenAPI JSON remains available locally.

API responses preserve the original record and provide media links separately.
`request_handler_ms` measures through JSON/SQLite commit, excluding model startup,
HTTP transfer and browser rendering. The record's forward-only timer excludes
saving and display. Neither timer is a Jetson benchmark or production rate.

## Actual verification

All **119 tests pass**: the previous 112 plus seven API tests. They cover empty
and persisted history, degraded model state, invalid inputs, known-part blocking,
saved missing-file errors, restart reads, read-only configuration, filters,
media integrity and refusing concurrent work without a queue. Test doubles
never provide synthetic successful detector outputs. Synthetic image fixtures
test HTTP/hash handling only, not detector performance.

Headless Chromium on Windows clicked **Inspect part** against the real WSL
GPU server twice:

| Saved part | Actual photo/result | Saved decision |
| --- | --- | --- |
| UI-PROOF-A | Normal reference; score 0.0284541; 0 response pixels | Provisional PASS |
| UI-PROOF-B | Development photo 1; score 0.6828089; 9,015 response pixels; type unknown | REVIEW |

Scores, input hashes and pixel geometry exactly match Stage 5. The duplicate
POST was refused with 409, leaving two rows. Refresh, grade filter and history
View worked. The first browser run found mobile overflow from an absolute
hidden table label. Its two actual records and failure log were preserved;
the containing block was fixed, and layout QA resumed on those existing rows
without new inference. First-click/API wall timings were not retained from
that interrupted run; none are invented in the report.

The owned server was then gracefully stopped and restarted on the same database.
A fresh browser still displayed the same latest record, both rows and their
actual images. Desktop and 390-pixel mobile screenshots passed; browser reports
contain no JavaScript page errors or external HTTP requests. This is browser
network observation, **not** a disconnected-machine or offline Jetson test.
No new holdout inference or model/threshold tuning occurred.

Saved local evidence:

- `artifacts/app_stage7/browser_verified/operator_desktop.png`
- `artifacts/app_stage7/browser_verified/operator_mobile.png`
- `artifacts/app_stage7/browser_restart/history_after_server_restart.png`
- `artifacts/app_stage7/proof_runtime/` — actual database, JSON, maps and overlays
- `artifacts/app_stage7/` — preserved test/install/server/browser logs and reports

Committed metadata: `manifests/operator_app_verification_v1.json`. MVTec images,
screenshots containing those images, models and runtime databases remain local.

```powershell
# Full software suite uses the new app environment.
wsl -d Ubuntu --exec /root/.cache/forgesight/app-v1/venv/bin/python -B -m unittest discover -s tests -q
```

Browser QA uses a separate Windows environment and Chromium, not an app runtime
dependency. `scripts/verify_operator_browser.py` requires a fresh proof database
for its initial two clicks, a fresh output directory, and the saved proof IDs.
`--resume-existing` verifies an interrupted proof without new inference;
`--restart-only` verifies the two records after restarting the actual server.
The browser/server setup and locked environments are in `docs/ENVIRONMENT.md`.

Stage 7 is verified for the desktop saved-image workflow. Type classification,
industrial tolerances, the two missed scratches, camera/physical identity,
Jetson performance, on-device ExecuTorch advice and disconnected edge proof
remain open. **Next is Stage 8: actual Jetson and controlled camera validation.**
The Jetson was last reported unavailable; no physical-device command ran here.

Stage 9.1 later added a separate CLI advisory component, with 128 current tests
passing. It is not connected to this screen, so the displayed pending-advice
message remains accurate. See [the advisory proof and limits](FORGEASSIST.md).

## Minimal operator view

At the user's request, the main view now shows the original/highlighted photo,
result, defect type and one short next action. PASS remains visibly provisional;
unknown types remain Unknown and failed images show Not assessed. Source mode
is a small Saved photos / Prototype label. No unsupported cause or disconnected
advisory placeholder is shown as a result.

Response area, timing, full recorded action and error details are available under
Details. History is collapsed by default but still supports filtering/View.
Model profile, known misses, pending Llama/camera/Jetson validation and other
limitations are under About this prototype. The stored record, rules, backend
and source evidence remain unchanged. The short action is a display summary;
the original complete action remains saved and accessible.

Desktop and 390-pixel mobile browser checks verified the collapsed sections,
history filters, provisional PASS/unknown-type REVIEW and a presentation-only
replay of a prior actual ERROR record. QA used copies of two previously verified
records in an isolated database on port 8767. No inference was rerun. The user's
already running default app/process was left intact. Screenshots are under
`artifacts/ui_simplification_v1/browser_final/`; committed metadata is
`manifests/operator_ui_simplification_v1.json`.

For this presentation-only check, `verify_operator_browser.py` accepts
`--stored-records-only` with the isolated server URL and a new output directory.
The first attempt reached an existing app on port 8765 and failed its record-count
precondition before any inspection POST; the separate test instance avoided
touching that history. Startup/readiness and initial failure logs are preserved.

Future advisory UI should keep this hierarchy: show a brief supported next check
or precaution, and a possible cause only when source-backed and labelled as
unconfirmed unless verified. Put sources, audit/provenance and technical detail
behind expansion; do not invent values to fill empty fields.

Implementation references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/), and the
official PyPI distribution metadata pinned in the dependency locks.
