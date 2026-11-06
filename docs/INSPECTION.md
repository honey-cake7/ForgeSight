# One declared part, one inspection — Stage 4

An image frame is not a part. The capture gate begins unarmed. An explicit
`arm(new_part_id)` accepts the next frame once; further frames are suppressed
until another distinct ID is armed. The gate latches before inference, including
failed attempts. Re-arming an already used ID or dropping an armed, uninspected
part is an error. Two distinct declared parts may have identical image bytes.

This is a saved-image replay with operator-declared identity, not verified camera
tracking. The gate's ID set is session-local. Durable identity, restart handling,
queries and SQLite come in Stage 6; no database is implemented here.

`InspectionRecord` includes a UUID, declared part ID, UTC time, image reference
and SHA-256, model/checkpoint/policy fingerprint, finite anomaly score,
normal/anomaly candidate, pixel regions, unknown type, processing/error state,
artifact references, timings and explicit pending decision/advice fields.
`inspect_one_image` decodes a bounded 700x700 RGB PNG/JPEG under this benchmark
profile, runs the real frozen calibrated model and saves a JSON, score-map NPZ
and response overlay. Invalid input saves an ERROR record; it never fabricates
a normal result. A record-write failure propagates to the caller: JSON alone
does not guarantee durable logging after disk/process failure.

Geometry counts 8-connected above-threshold response pixels with no filtering.
The box is `[x, y, width, height]`, origin at the upper-left, in source-image
pixels. Length is an explicitly named **box diagonal proxy**, not scratch
length. Pixel extent is a model response, not validated physical damage.
See [OpenCV's connected-component statistics](https://docs.opencv.org/4.6.0/d3/dc0/group__imgproc__shape.html).

Actual replay evidence is committed in `manifests/inspection_replay_v1.json`:
9 frames, 4 declared parts, 5 suppressed repeats, 3 actual model forwards and
1 explicit missing-file error. Parts A and C use identical normal image bytes
but distinct IDs; B uses the frozen development bent image. None of these are
new accuracy measurements and no holdout inference was repeated. Local outputs:
`artifacts/core/inspection_replay_v1/records/`.

```powershell
# New output directory; genuine saved-image replay.
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.inspection_replay --output artifacts/core/inspection_replay_v1
# One image, with an explicit ID. Uses the existing locked vision environment.
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.inspection data/mvtec_ad/metal_nut/train/good/000.png --part-id manual-001 --output artifacts/core/manual
# Full suite now includes NumPy/OpenCV geometry tests.
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m unittest discover -s tests -q
```

All 94 tests pass, including corner connectivity, area/box/length-proxy geometry,
threshold equality, duplicate suppression, explicit re-arm, same-image new parts,
invalid input, and callback failure. No dependencies were installed or upgraded.
Stage 5 adds versioned decisions; type classification, SQLite, camera, Llama
advice integration and Jetson verification remain separate steps.

## Stage 6 persistence update

The current inspection CLI commits JSON and SQLite by default and takes
`--database PATH`; replay creates a database inside its new output directory.
`inspect_one_image(..., store=store)` enables the same commit path in Python.
CaptureGate accepts `already_recorded=store.has_part` to refuse recorded IDs
after restart. Earlier Stage 4 JSON-only proof above remains a historical
snapshot. See [SQLite behavior, retries and proof](STORAGE.md).
