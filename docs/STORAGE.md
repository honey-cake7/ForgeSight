# Local inspection history — Stage 6

Each completed saved-image inspection can now be committed to a local SQLite
database, including failed images and their REVIEW decisions. The inspection
CLI stores by default; the real replay stores in its own output directory.
Python-level calls can pass `store=InspectionStore(path)` explicitly. Older
JSON-only outputs and manifests remain preserved as historical snapshots.

SQLite is already included in the existing Python environment; no package or
server was installed. This fits one local inspection application. Its
transaction commits and keys provide stronger restart/duplicate behavior than
counting JSON files alone. SQLite permits one writer at a time; this prototype
uses short transactions, a five-second lock timeout, rollback journaling and
FULL synchronous commits. This is not a factory-throughput benchmark.

## Schema and identity

Schema version 1 is established atomically for a new empty database using
`PRAGMA user_version`, with the ForgeSight application ID. Unrelated,
unversioned nonempty and unsupported/newer databases are refused, preserved
without automatic repair or destructive migrations. Future changes need an
explicit migration. Missing history paths fail rather than silently returning
a newly created empty database.

The inspection table stores the original full record as metadata JSON plus
indexed identifiers, UTC time, image path/hash, model name/checkpoint hash,
score-policy hash, grade-policy version/hash, status, grade and advice state.
Maps and images remain referenced files; image bytes do not enter SQLite.
The full snapshot preserves scores, regions, action text, errors, timings,
policy, input mode and artifact provenance.

- `inspection_id` is a unique key. Saving the exact canonical record again is
  an idempotent retry: it returns `False` and adds no row.
- Reusing an inspection ID with changed content is an explicit conflict.
- `part_id` is also unique **within one database**, including after restart.
  A new UUID cannot count that same declared part again. Use genuinely distinct
  part IDs, optionally including batch identity, across the application's life.
- Two distinct parts may use identical photo bytes. Image hashes establish
  provenance; they do not determine physical identity.
- `CaptureGate(already_recorded=store.has_part)` checks persistent history when
  re-arming. The CLI checks before model loading, and the inspection function
  checks before inference. The database unique constraint remains authoritative
  if two writers race. A racing caller can still perform extra inference before
  its final save is refused; no concurrent physical capture claim is made.

An accepted failed-image attempt is retained once, as ERROR/REVIEW. A database
retry means resaving its original JSON; it does not mean inspecting the part
again under another ID. A separate reinspection workflow is not implemented.

## Corrections and write failures

Correction notes have their own unique ID, actor, UTC time, reason, optional
proposed grade and foreign key to the original inspection. They remain
`UNVERIFIED_OPERATOR_CLAIM`. They do not overwrite the original decision, alter
its counts, become confirmed training labels or establish factory causes.
Storage supports append-only proposals now; a review interface and richer
verification workflow remain later work. Original records and correction notes
also have update/delete guards. Metadata checksums detect accidental changes;
they are not a cryptographically secured audit system.

The pipeline writes the original inspection JSON **before** saving to SQLite.
It returns success only after database commit. A database failure is reported
with the preserved JSON path; `import-record` can retry that same evidence
without another model forward. JSON and database writes are not one atomic
cross-file operation. Automatic pending-file recovery, process-kill/power-loss
tests and backup/restore are unverified. A JSON-write failure propagates.

The timer named `inspection_before_record_write_ms` excludes JSON and SQLite
writes, model/policy loading and the duplicate check. No storage or complete
Jetson inspection latency target is claimed.

## Observed proof

The actual Stage 6 saved-image replay ran three real forwards on already used
fit/development images and one deliberately missing input. Nine frames gave
four records, with five repeated frames suppressed. The model's scores and
pixel geometry exactly matched Stage 5 for those same three valid images.
No holdout inference, model/threshold changes or dependency installations took
place.

After the writer process exited, **new Python processes** read all four complete
records with identical contents:

| Stored outcome | Count |
| --- | ---: |
| Provisional PASS | 2 |
| REVIEW | 2 |
| REJECT | 0 |
| Analyzed images | 3 |
| Failed-image ERROR, included among REVIEW | 1 |

All four exact-record retries added zero rows. Changed-content and repeated-part
conflicts were refused. A restarted gate refused an existing ID; the actual
inspection CLI exited 2 before model loading and created no new output.
An explicitly synthetic correction-storage demonstration persisted separately
and left the original PASS/count unchanged. It is not a real operator finding.
REJECT storage is verified with a labelled synthetic unit fixture, not an actual
typed-model rejection. Integrity check returned `ok`, no foreign-key violations,
schema version 1, rollback journal `delete` and synchronous level 2 (FULL).

All **112 tests passed**, covering the previous 101 checks and 11 storage tests:
round-trip metadata, restart reads, identity, concurrent exact retries, all
grades/errors, pagination/filters, invalid metadata, immutable originals,
correction claims, refusal of foreign/newer databases and JSON retry after
database failure. The existing environment reports Python 3.12.14 and SQLite
3.53.1. SQLite compatibility/performance on the actual Jetson is unverified.

Local evidence:

- `artifacts/storage/inspection_replay_v1/inspections.sqlite3`
- `artifacts/storage/inspection_replay_v1/records/`
- `artifacts/storage/verification_v1/history_after_restart.json`
- `artifacts/storage/verification_v1/storage_report.json`
- `artifacts/storage_progress/` — execution logs

Committed metadata manifests record replay and verification hashes; the live
database and runtime outputs remain ignored. Its recorded SHA-256 is a snapshot
at verification time and will change if more records/claims are added.

## Commands and Python query API

Run from the project root in PowerShell. The saved evidence paths already
exist; use a new output path when repeating inference/replay. Original evidence
is never overwritten.

```powershell
# Read the actual Stage 6 database, without importing Torch or running a model.
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.storage --db artifacts/storage/inspection_replay_v1/inspections.sqlite3 summary
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.storage --db artifacts/storage/inspection_replay_v1/inspections.sqlite3 history --status ERROR
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.storage --db artifacts/storage/inspection_replay_v1/inspections.sqlite3 integrity

# Full suite uses the existing locked vision environment for geometry tests.
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m unittest discover -s tests -q

# First replay creates its own SQLite file; output must be new.
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.inspection_replay --output artifacts/storage/inspection_replay_v1
# Run after the replay writer exits. Adds one clearly marked demonstration note.
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.storage_proof --replay artifacts/storage/inspection_replay_v1/replay_report.json --output artifacts/storage/verification_v1

# Retry a saved JSON after an interrupted database save; no new inference.
wsl -d Ubuntu --exec .venv/bin/python -B -m forgesight.storage --db DATABASE.sqlite3 import-record RECORD.json
```

`InspectionStore` exposes `save_inspection`, `get_inspection`, `has_part`,
`history(limit, offset, grade, status, part_id)`, `summary`, `save_correction`,
`corrections` and `integrity`. Filters use bound SQL parameters. History orders
by normalized UTC timestamp and inspection ID; summaries count original
inspection outcomes and separate unverified notes. These are Python query
methods and local CLI commands, not HTTP endpoints. API/UI work is Stage 7.

References used for the implementation:
[Python sqlite3](https://docs.python.org/3.12/library/sqlite3.html),
[SQLite transactions](https://www.sqlite.org/lang_transaction.html),
[schema/version pragmas](https://www.sqlite.org/pragma.html), and
[foreign-key enforcement](https://www.sqlite.org/foreignkeys.html).

Stage 6 is verified for desktop saved-image persistence. Reliable defect names,
the two missed scratches, physical part tracking, industrial tolerances, Jetson
camera/offline operation and on-device Llama/ExecuTorch advice remain open.
**Next: Stage 7, the smallest local backend and operator screen.** Stop for review
before that stage.

## Stage 7 current command update

The full current suite includes FastAPI contract tests and uses
`/root/.cache/forgesight/app-v1/venv/bin/python`, with 119 tests passing. Earlier
stage-specific test counts/vision-environment commands above describe their
historical snapshots. See [the current operator workflow](OPERATOR_APP.md).
