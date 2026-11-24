# ForgeSight Jetson timing report

This page uses the user-supplied `artifacts/jetson_timings_report.md`, provided
on 24 November 2026. Its benchmark timestamp is **05:35:53 UTC / 11:05:53 IST**.
It describes an Orin Nano 8GB dual-recipe station with persistent CPU vision
workers and an ExecuTorch/XNNPACK Llama worker, with a reported four-thread
OpenMP configuration and 25 W power-mode setting.

These are supplied device observations, not measurements rerun during this
documentation change. They supplement the earlier [runtime report](MERGE_LATENCY_REPORT.md)
and [offline series](STAGE10_VERIFICATION.md). The source report does not pin
the deployed Git commit, so its results must not be assigned to the current
checkout automatically.

## Provenance

- Source: `artifacts/jetson_timings_report.md`, retained unchanged and ignored by Git.
- Source SHA-256: `00f74d8c3f8a1728d64433866c690b6a90551270e6f6f0c238b2fc6acc0fd4fb`.
- Structured transcription: [timing summary](../manifests/jetson_timings_summary_v1.json).
- Raw JSON cited by the report: `/home/nvidia/forgesight-feature-audits/plugin-recipe-test/timings_report.json`.
  No local copy was found or fetched during this update.
- Collector reviewed read-only: `scripts/record_timings.py`. It remains
  untracked and unchanged; this page is not a claim that its device execution
  was reproduced or that its raw output was independently checked.

## Warm vision: one inspection per recipe

| Measurement | Metal nut | Screw |
| --- | ---: | ---: |
| Inspection ID | INS-000032 | INS-000033 |
| Source | `test/bent/017.png` replay | 1024 x 1024 diagnostic `upload.png` |
| Model forward | 1,500.12 ms | 1,489.37 ms |
| Other work within the inspection timer, by subtraction | 697.45 ms | 685.60 ms |
| Reported inspection work | **2,197.57 ms** | **2,174.97 ms** |
| Image score / rounded threshold | 0.6828 / 0.2278 | 0.9572 / 0.2373 |
| Grade | REVIEW | REVIEW |
| Reported type | bent candidate | scratch_neck |
| Mode | PERSISTENT_WORKER | PERSISTENT_WORKER |

In current source, the station's `inspection_ms` is drawn from
`capture_file_to_record_ms` when available. That timer stops before record JSON
and SQLite writes, and excludes camera capture, browser rendering and advice.
The deployed revision is not pinned here; the collector reads that API field.
Therefore this page calls it **inspection work**, rather than claiming complete
camera-to-saved-and-displayed-result latency. The remainder is not a separately
profiled decode/classifier/database breakdown.

The screw upload's original dataset identity/hash is not supplied. The
collector can fall back to a synthetic image if no suitable saved upload is
found; without raw JSON, that fallback cannot be ruled out independently.
Neither timing is a new defect-accuracy or physical-part result.

## Warm Llama: direct worker microbenchmark

| Measurement | Supplied value |
| --- | ---: |
| Prompt / reported native generated tokens | 24 / 39 tokens |
| Worker generation call | **1,622.65 ms** |
| Socket request round-trip | **1,627.30 ms** |
| Round-trip minus generation, derived | 4.65 ms |
| Reported decode throughput | **28.18 tokens/s** |
| Reported prefill throughput | 101.27 tokens/s |
| Reported prefill / TTFT | 237 ms |

This is one direct request to an already-resident worker, not a distribution of
complete inspection-linked advisories. The collector's request uses a short
standalone prompt; it is not the full saved-record advisory flow. The worker's
generation timer wraps `runtime.generate(...)`. The preceding artifact-state
check, reset, prompt formatting and explicit token-count check are outside that
timer; native prompt processing within the generation call remains included.
The round-trip includes the other request handling and transport/serialization
work. The 4.65 ms difference is not an isolated measurement of kernel IPC cost.

The native generated-token statistic counts decode-loop tokens and excludes
the first token emitted during prefill in this runner's documented convention.
The reported 39 is therefore not asserted as the complete output-token length.

The source labels the 237 ms prefill observation as TTFT. With no raw token
timestamps available here, retain that attribution rather than interpreting it
as independently measured first-token display latency. The sample's checklist
wording is not evidence of factory SOP approval; bin names and release
instructions require separately authorized site procedures.

## Preloaded recipe switching

| Direction | n | Mean | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Metal nut to screw | 5 | **0.539 ms** | 0.281 ms | 0.700 ms |
| Screw to metal nut | 5 | **0.504 ms** | 0.434 ms | 0.556 ms |

Individual supplied runs are preserved in the structured summary. These times
are the server's `switch_ms` for changing references to preloaded recipes and
workers. They exclude the HTTP/browser path and initial model loading. They
demonstrate short routing changes in this sample, not zero downtime under every
load or sub-millisecond cold startup.

## Resident memory snapshot

| `free -m` field reported by the source | Value, interpreted as MiB |
| --- | ---: |
| Total | 7,450 |
| Used | 3,008 |
| Available | 4,186 |
| Buffer/cache | 1,833 |
| Swap used / total | 498 / 3,725 |

The source describes both vision models, Llama and the station as active during
this snapshot. `Available` includes reclaimable memory and is not the same as
unused/free RAM. These overlapping memory fields must not form an additive pie
chart. The snapshot is not simultaneous peak RSS, a swap-rate trace or proof of
no thrashing/OOM risk. The 25 W configuration is a power-mode setting, not a
measured electrical power draw.

## What improved latency

1. **Resident vision workers:** verify/load the frozen model at worker start
   and reuse it over a local socket instead of repeating Torch imports and
   model setup for every image.
2. **In-process inspection work:** the station can preprocess and execute the
   saved-image inspection path without another full controller process per image.
3. **Resident Llama worker:** reuse the ExecuTorch runner and loaded model;
   reset request state rather than repeating runtime/model initialization.
4. **Preloaded part recipes:** switching selects existing model/policy/worker
   references instead of loading new weights. This trades resident RAM for
   short switching latency.
5. **Separate advisory work:** grading/history do not wait for Llama's response.
   This improves the operator workflow; it does not reduce the neural network's
   forward time by itself.

The roughly 1.49-1.50 s vision forward remains visible inside the roughly
2.17-2.20 s inspection work. These are CPU results; no GPU speedup is implied.

## Comparisons and remaining measurements

The source's approximately **3.8 s cycle** adds inspection work to the separate
1.62265 s direct Llama call. The two possible sums are 3.82022 s and 3.79762 s.
This is an illustrative budget, not an observed complete camera-to-advice
cycle, and does not include every storage/UI/prompt-validation boundary.

Earlier cold-process measurements include broader startup/storage scopes and
different prompts/token counts. Keep them as historical baselines; do not
report a percentage speedup from unlike timings. A controlled comparison needs
matching inputs/prompts, pinned source/runtime/assets, repeated first/warm
requests, explicit timing boundaries and resource sampling. Full application
FPS, distinct physical parts/minute, camera-domain accuracy, measured power,
simultaneous peak memory and complete-cycle percentiles remain separate items.
