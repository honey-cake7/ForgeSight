# User guide

Open **ForgeSight** after [installation](../README.md#download-and-install-on-jetson).
The release enables both part recipes and automatic grounded advice. [SETUP](SETUP.md)
covers deployment; [TECHNICAL_REPORT](TECHNICAL_REPORT.md) holds measurement scope.

## Inspect a component or saved image

1. In plugin mode, choose **Metal nut** or **Screw** under **Part Recipe**.
   Switching is blocked while an inspection/recovery is active or a worker is
   unavailable. The previous latest-result view clears; saved history retains
   the original recipe.
2. Choose **Camera** for a physical component. **Start Camera** opens the C270;
   **Stop Camera** releases it. Position the part and select **Analyze** to save
   one frame. The current button declares a physical part. Screen-photo
   diagnostics require the backend's explicit source mode; the page has no
   separate screen-photo selector.
3. For saved images use **Upload PNG**. RGB PNGs are supported; screw mode also
   accepts native grayscale. Respect the size/dimension limits shown by the
   page. Declare original-size MVTec inputs only when appropriate; this declaration
   does not prove their provenance. Backend replay routes remain available but
   are not a separate replay button in the current station page.
4. Open **Results**. Compare Original, Highlighted Area, Compare slider,
   Heatmap or Contours. Read the grade, provisional type/severity, response
   extent and fixed next action. Overlays identify model response, not confirmed
   damage; pixel measurements are not millimeters.
5. Request manual advice for an eligible flagged record, or use automatic
   grounded advice when enabled. The grade is saved independently. Pending,
   busy, unavailable and checklist fallback have explicit states.
6. Use **Review** or **History** to save observations and reopen results.
   The UI supports light/dark preference, responsive navigation and keyboard
   labels. Closing the desktop window leaves the backend running; Ctrl+C stops
   a directly launched service in its own terminal.

Inspection numbers count accepted attempts. Repeated photos or frames do not
by themselves establish distinct physical parts. Camera results stay REVIEW
until their domain is validated. Never treat confidence as severity or an
operator claim as confirmed ground truth.

## Advice modes

The page's single **Generate Llama advice** button requests server mode `auto`.
The server selects `composed` for records without an assessment, or
`vision_choice` for a supported benchmark assessment with its profile enabled;
unsupported combinations fail visibly. Plugin/grounded mode starts advice
automatically instead. Standalone CLI callers select `--advisory-mode vision_choice`
for reviewed-check advice; [examples](archive/2026-11-25/FORGEASSIST.md#composed-advice-on-the-jetson)
are preserved in the archive. Normal or failed inspections skip automatic
generation. A busy or failed Llama worker
does not change the inspection's original grade. Checklist fallback is labeled
and is not presented as generated Llama text. Guidance is generic until site QC
procedures are approved.

## History, review and report export

History is paginated. **Open result** shows saved original/overlay images,
advice and current operator claims without another model forward. Add a review
with operator ID, finding and observation. Editing creates a superseding claim;
Delete review or Delete test hides the entry from the operator view and preserves
original audit data. Reviews never replace the model's original decision.

**Download report** produces self-contained HTML with available embedded images,
outcome, advice and reviews. It is printable. The GTK WebKit desktop viewer
supports **Save PDF** into **Desktop / ForgeSight Results**; ordinary browsers
download HTML. Missing source/overlay files are reported rather than fabricated.
These are individual inspection reports, separate from the project technical report.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Camera unavailable | Check the C270 connection and Start Camera; saved-image upload can be used independently. |
| Station busy / recipe change refused | Wait for the current inspection or worker recovery; avoid duplicate retries. |
| Advice unavailable or fallback | Retain the inspection and follow the displayed manual check; inspect runtime state/logs before retrying. |
| ERROR with support ID | Use that ID to trace the local operation log; accepted failures remain in history. |
| Missing/mismatched model or policy | Restore the matching authorized bundle; never bypass hash checks or install into system Python. |
| Missing images or storage warning | Check referenced files/free space and reported orphan findings; do not delete evidence automatically. |

Release records live under `<installation>/runs/`, including
`inspections.sqlite3`, advice and rotating `logs/`. Manual source launches default
to `runs/live_inspections_v1/`; `station.json` records the installed paths.
The local API provides state, camera controls, upload/replay, advice, history,
media and review routes; mutations check origin and bounds. PC API/CLI queries
also support history filters and summaries.

## Feature reference

These 38 entries describe implemented functionality, including optional modes.
Device acceptance and benchmark scope are reported separately in
[TECHNICAL_REPORT](TECHNICAL_REPORT.md).

### Inspection station and operator interface

| Feature | What it does |
| --- | --- |
| Live camera preview and controls | Shows the C270 feed; Start Camera opens capture and Stop Camera releases it without discarding the saved result. |
| Capture one inspection | Analyze saves one frame and creates a structured inspection. SQLite assigns persistent `INS-000001`-style attempt IDs; repeated frames are not automatically counted as new physical parts. |
| PNG upload | Accepts RGB PNGs and, in screw mode, native grayscale PNGs from the browser computer with size/dimension limits. The original-size MVTec declaration enables provisional benchmark grading; other inputs remain diagnostic. |
| Saved dataset replay | Backend upload/replay and Jetson-copy replay endpoints validate permitted dataset references/bytes. The separate PC replay app offers an exposed-example catalog; the current Jetson page presents Camera and Upload PNG tabs. |
| Four operator views | Inspect, Results, Review and History navigation; responsive layouts, collapsible sidebar, keyboard labels, progress/busy/error feedback and saved light/dark preference. |
| Image exploration | Original, Highlighted Area, interactive Compare slider, Heatmap and Contours views; response-value legend and access to the saved original. PNG is the default overlay format; optional JPEG is lossy. These visualize model response, not confirmed damage. |

### Detection, localization and ForgeGrade

| Feature | What it does |
| --- | --- |
| EfficientAD anomaly detection | Produces a real image score and source-resolution response maps from the frozen, calibrated model; threshold and model provenance are retained. |
| Response-region measurements | Finds connected response regions, largest/additional regions, pixel area, bounding boxes and a diagonal extent proxy. Saved benchmark nut layouts can receive approximate opening/rim/face/perimeter descriptions. |
| Metal-nut type assessment | Optional four-type profile reports bent, flip, scratch or discoloration and can abstain as `unknown`; no folder label is substituted for a prediction. |
| Screw type assessment | Separate screw tree supports `manipulated_front`, `scratch_head`, `scratch_neck`, `thread_side` and `thread_top`, with approximate head/neck/thread/tip geometry and unknown handling. |
| Deterministic decisions | ForgeGrade applies versioned PASS/REVIEW/REJECT rules and fixed next actions. Llama and operator review cannot overwrite the original grade. |
| Severity and uncertainty | Prototype pixel-response severity plus optional type-conditioned Low/Medium/High/Unassessed metadata; unknown, orientation and uncertain geometry are handled explicitly. |

### Part plugins and recipe switching

| Feature | What it does |
| --- | --- |
| Two bundled part recipes | `metal_nut` and `screw` each bind their own detector, score policy, grade policy, classifier, guidance and dataset manifest. |
| Verified recipe loading | Checks component SHA-256, bundle paths, threshold agreement and classifier/model binding before use; mismatched or missing assets are refused. |
| Preloaded recipe switching | The optional Part Recipe selector changes routing between already-loaded vision workers while retaining the shared Llama worker. It refuses switching during inspection/recovery or to an unavailable worker. |
| Per-inspection recipe provenance | New inspections use the selected recipe; saved records retain their original recipe/evidence. Switching clears the prior latest-result display instead of relabeling it. |

### ForgeAssist and grounded operator guidance

| Feature | What it does |
| --- | --- |
| Real local Llama advice | Llama 3.2 1B Instruct generates through on-device ExecuTorch/XNNPACK from saved structured inspection findings. |
| Manual and automatic modes | The manual button uses guarded server auto-routing between composed and reviewed-check advice. Optional grounded mode queues eligible flagged records automatically; normal/failed records skip generation. |
| Bounded background processing | Advice runs separately from vision; pending/busy states and deadlines prevent an unbounded request backlog. Saved grades remain available if advice fails. |
| Grounded checks and fallback | Part/type/region rules select allowed manual checks. Grounded modes validate the reply and use an identified checklist fallback for unsupported output or unavailable workers; default v5 can return ADVICE_UNAVAILABLE. |
| Saved advice provenance | Reports bind advice to the inspection and record runtime/model identity. Generated checks, host-supplied reasons and checklist fallback are distinguished. |
| Independent advisory tool | The standalone grounded-advice command reads saved evidence through a dedicated worker socket without camera access or database writes. |

### Inspection history, review and reporting

| Feature | What it does |
| --- | --- |
| Persistent SQLite records | Stores original records, source references, scores, geometry, versions, decisions and advice state; exact retries are idempotent and conflicting IDs are refused. |
| Restart-safe numbering | A persistent counter is shared by captures, uploads and replays; recognized older databases migrate while retaining inspection/correction rows. |
| Browse and reopen history | Paginated station history opens original/overlay, saved advice, type/severity and operator reviews without rerunning inference. API/CLI queries additionally support filters and summaries. |
| Add and edit operator findings | Saves actor, time, finding and observation as unverified claims. Editing appends a superseding correction; original decision/evidence remain intact. |
| Hide test entries and reviews | History deletion controls hide selected entries/reviews from the operator view; they do not erase the original inspection or claim from audit storage. |
| Download inspection reports | Exports self-contained HTML with available embedded original/overlay images, outcome, saved advice and reviews. Reports are printable; the GTK WebKit desktop viewer supports Save PDF to Desktop / ForgeSight Results. |
| Recover database-save failures | Retains inspection JSON for explicit import retry when SQLite commit fails, avoiding another model forward. |

### Offline runtime, reliability and deployment tools

| Feature | What it does |
| --- | --- |
| Offline edge operation | **Verified on Jetson** for recorded saved-image vision, grading, SQLite, real ExecuTorch advice and loopback-service workflows with Wi-Fi disconnected. |
| Persistent inference workers | Reuses verified vision/Llama runtimes across requests; model-state reset, ready handshakes, bounded transport and explicit timeout/fallback paths are implemented. |
| Worker lifecycle and recovery | Detects worker failures, uses supported cold fallbacks, respawns/rewarms workers, drains diagnostics and releases idle/owned workers; workers follow station lifetime rather than a short-lived launch thread. |
| Operational error tracing | Rotating local logs and support IDs link request, storage and worker failures. Messages are bounded and known credential patterns are redacted. |
| Local API and validated intake | Loopback routes expose camera controls, inspection/replay/upload, advice, state, history, media and review. Mutations check origin, requests are bounded and saved media is checked against its record. |
| Storage health | Reports free disk space, low-space warnings and bounded checks for partial/unconfirmed/mismatched run directories; reports findings without deleting evidence. |
| Desktop application launcher | Starts or reuses the configured local station, checks loopback readiness, opens a dedicated Chromium/GTK WebKit window and offers desktop/menu shortcuts with local startup/browser logs. |
| Llama activation and rollback | Tooling checks the PTE/profile/export hash chain, backs up coherent assets and supports activation/rollback while the app is stopped. Runtime acceptance remains a separate check. |
| Reproducibility and diagnostics | Dataset/model fingerprints, split/exposure records, separate runtime locks, export/parity scripts, offline benchmarks and a bounded read-only diagnostic bundle collector. |

Further reading: [setup](SETUP.md), [technical report](TECHNICAL_REPORT.md), [current status](STATUS.md), [historical guides](archive/README.md).
