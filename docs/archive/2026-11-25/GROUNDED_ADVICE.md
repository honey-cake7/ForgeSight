> Historical snapshot archived on 25 November 2026. This page preserves development history; use the [current documentation](../../../README.md) for supported instructions.

# Standalone grounded Llama advice

This add-on reads saved inspection evidence and has no database write or
camera access. It uses a dedicated ExecuTorch worker socket. An idle worker
exits after the configured interval.

## Dashboard mode

Start `forgesight.jetson_live_ui` with `--grounded-auto-advice` and a dedicated
`--llama-worker-socket`. Keep the station's existing vision, camera and database
options. Each capture or upload runs vision first. Once a flagged inspection
is saved, the dashboard queues advice automatically and shows the result when
ready. Normal and failed inspections skip the advisory call. Live-camera
advice describes the record as diagnostic and uses a manual check.

The worker runs at lower CPU priority. The dashboard uses a bounded background
queue and presents the checklist when the worker is unavailable or returns
unsupported wording. If another advisory is still running, the vision result
is saved and the panel reports that Llama is busy. Completed reports go in
`grounded/` under the configured advice directory.

For rollback, restart the previous dashboard command or omit the opt-in flag.
Existing inspection records and legacy advice stay available to that version.

## Supported findings

The prototype rules cover possible scratches, bent nuts, orientation flips,
discoloration, unknown defects, uncertain localization, and scratch or bent
indications at the inner rim. Guidance is generic and subject to site QC
approval. Saved vision fields retain their original scope and grade. Severity
and response area describe vision evidence, with pixel measurements kept in
image units. PASS and failed or incomplete inspections do not invoke Llama.

Llama receives the finding, severity, approximate region, image-response
area, and applicable checklist actions. It returns one check in words. The
app validates the check and appends a reason from the saved vision evidence.
The report records the source of the check and the reason separately. Any
other reply or a worker error displays the first applicable instruction as
`CHECKLIST_FALLBACK`.

Plant-specific tolerances, lot recurrence, critical-surface priorities, and
release or quarantine decisions require a separate QC-approved procedure and
validated inputs. The add-on does not infer lighting faults or physical
defects from an anomaly score. A new image or a human review addresses gaps in
the existing record.

## Try on Jetson

Run from the standalone checkout using the installed ForgeSight Llama runtime.
Choose an unused private socket path. Start a worker in one terminal, then run
the advice command in another.

```sh
mkdir -p "$HOME/.cache/forgesight-grounded"
chmod 700 "$HOME/.cache/forgesight-grounded"
llama="$HOME/.local/share/forgesight/llama_cpu_v1"
nice -n 10 "$llama/venv/bin/python" -B -m forgesight.assist_edge_worker \
  --llama-root "$llama" \
  --socket "$HOME/.cache/forgesight-grounded/advice.sock" \
  --idle-timeout-seconds 60
```

```sh
python3 -B -m forgesight.grounded_advice \
  --record /path/to/saved/record.json \
  --socket "$HOME/.cache/forgesight-grounded/advice.sock"
```

Stop the standalone worker from its terminal to free the model memory. Use an
independent socket for standalone checks while the dashboard is running.
