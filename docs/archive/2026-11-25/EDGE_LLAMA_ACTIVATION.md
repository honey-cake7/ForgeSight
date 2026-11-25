> Historical snapshot archived on 25 November 2026. This page preserves development history; use the [current documentation](../../../README.md) for supported instructions.

# Llama hash-chain activation and rollback on the Jetson

`forgesight/edge_llama_activate.py` stages, activates and rolls back the
three files that change together when a new Llama export is adopted on the
device: the PTE, `edge_llama.json` and `llama3_2_1b_export.json` under the
private llama root's `model/` directory. The tokenizer and the assets
manifest must not change; the tool verifies the candidate against the live
assets/tokenizer, which is exactly what proves that.

It uses only the Python standard library, so it runs with the device system
Python before any venv is involved. It is not a device proof: the restarted
advisory worker's own `verify_inputs`, its bounded warmup and the first
accepted advisory remain the authoritative evidence.

## Why the trio moves together

The persistent worker verifies the five-file chain at startup
(`edge_jetson_smoke.verify_inputs`) and stat-checks the artifact on every
request; the controller re-derives the pinned artifact/tokenizer hashes from
the live manifests on every advisory request. A mixed set fails closed at
both layers. The operator app (`jetson_live_ui`) owns the advisory worker as
a child process, so the app must be stopped for the swap; the brief
mixed-set window during the three `os.replace` calls is then unobserved.

## Prerequisites

- The reviewed coherent commit (profile + new export manifest together) and
  the actual new PTE from a retained export attempt. A profile-only or
  manifest-only update makes the device refuse service by design.
- Operator app stopped (Ctrl+C on its terminal). Activation refuses while
  the app or worker process is detected, and also refuses when it cannot
  read every process cmdline: run the tool as the same user that runs the
  app (or via sudo), so the scan can prove the app is stopped. An
  unreadable process cannot be proven innocent, so the tool fails closed.

## Stage the candidate (PC to Jetson)

On the PC, assemble a fresh candidate directory with exactly these three
files under these exact names:

```sh
mkdir staging_candidate
cp artifacts/edge/export_attempt_02/exported/llama3_2_1b_xnnpack.pte staging_candidate/
cp artifacts/edge/export_attempt_02/llama3_2_1b_export.json staging_candidate/
cp configs/edge_llama.json staging_candidate/
sha256sum staging_candidate/* | tee staging_candidate.sha256
```

Copy them to a fresh staging directory on the Jetson and re-check the hashes
there before touching anything:

```sh
ssh nvidia@<JETSON_IP> 'mkdir -p /home/nvidia/.local/share/forgesight/llama_cpu_v1/staging/llama256_01/candidate'
scp staging_candidate/{llama3_2_1b_xnnpack.pte,llama3_2_1b_export.json,edge_llama.json} \
    nvidia@<JETSON_IP>:/home/nvidia/.local/share/forgesight/llama_cpu_v1/staging/llama256_01/candidate/
ssh nvidia@<JETSON_IP> 'cd /home/nvidia/.local/share/forgesight/llama_cpu_v1/staging/llama256_01/candidate && sha256sum *'
# Compare against staging_candidate.sha256; stop on any difference.
```

## Preflight (read-only)

```sh
cd /home/nvidia/.local/share/forgesight/llama_cpu_v1
python3 -B edge_llama_activate.py preflight \
  --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 \
  --candidate /home/nvidia/.local/share/forgesight/llama_cpu_v1/staging/llama256_01/candidate
```

`PREFLIGHT_OK` with an empty blocker list is required. It prints the live
and candidate chains, the planned `max_seq_length`/`max_context_length`
change, and the unchanged tokenizer/assets hashes. Any blocker means stop;
nothing was changed.

## Activate

```sh
python3 -B edge_llama_activate.py activate \
  --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 \
  --candidate /home/nvidia/.local/share/forgesight/llama_cpu_v1/staging/llama256_01/candidate \
  --backup-parent /home/nvidia/.local/share/forgesight/llama_cpu_v1/backups \
  --report /home/nvidia/.local/share/forgesight/llama_cpu_v1/runs/llama256_activate_01.json
```

The tool backs up the live trio with verified hashes into a fresh
`backups/llama-chain-<timestamp>-<candidate>` directory, replaces the trio,
re-verifies the live chain, and rolls back automatically if that check
fails. `ACTIVATED_VERIFIED_PENDING_APP_RESTART` is success at this layer.
The report path must be fresh; choose a new name on repeats.

Then start the operator app as usual and watch the worker startup: the
worker re-runs full hash verification and its bounded warmup. The recorded
`WORKER_READY` line must carry the new artifact SHA-256 from the new export
manifest. Only then run the device acceptance series (smoke regression,
prompt-boundary and measured cases); those results decide adoption, not this
script.

## Roll back

```sh
python3 -B edge_llama_activate.py rollback \
  --llama-root /home/nvidia/.local/share/forgesight/llama_cpu_v1 \
  --backup-dir /home/nvidia/.local/share/forgesight/llama_cpu_v1/backups/llama-chain-<recorded> \
  --report /home/nvidia/.local/share/forgesight/llama_cpu_v1/runs/llama256_rollback_01.json
```

Rollback verifies the backup before restoring and preserves whatever is
currently live under `pre_rollback_live/` inside the backup directory, so no
state is silently destroyed. `ROLLED_BACK_VERIFIED` means the old chain is
live and verified; restart the app afterwards.

## Failure outcomes

Every failure is reported with an explicit, truthful state; the tool never
claims "no changes" once a write may have happened:

- `BLOCKED` - a preflight or safety refusal. No live file was changed.
- `FAILED_LIVE_UNCHANGED` - the swap failed before the first file was
  replaced; the live trio still matches the verified backup of the original
  chain. Fix the cause and retry with a fresh report path.
- `FAILED_ROLLED_BACK_VERIFIED` - the swap failed after replacing files, so
  the live set was mixed; the verified backup was restored automatically and
  the restored chain passes verification.
- `FAILED_MANUAL_INTERVENTION` - the automatic restore did not complete.
  The live set may be mixed right now. Keep the app stopped and recover by
  hand: the report names the verified backup directory and (for rollback)
  the preserved previous live set; copy one coherent trio into `model/`,
  then run preflight. Do not start the app until the chain verifies.

If a crash interrupts activation, the backup directory's
`activation_journal.json` says where it stopped; run preflight, then
rollback from that backup directory to return to a coherent set.

## Scope

This proves a verified, reversible file swap only. Generation, quality,
offline operation and camera/physical-part behavior stay NOT VERIFIED until
their own device evidence exists. The dated
`configs/jetson_llama_runtime_v1.json` is a historical record of the old
chain and is intentionally not rewritten.
