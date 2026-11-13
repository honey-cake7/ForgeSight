# Merged runtime changes and measured Jetson latency

This report covers the runtime PRs merged through `ca499bd` (PR #12), the
version deployed on the physical Jetson Orin Nano 8GB. Vision uses the frozen
EfficientAD-S FP32 CPU model. Advice uses Llama 3.2 1B Instruct through the
separate on-device ExecuTorch XNNPACK CPU runtime.

The main improvement is avoiding repeated process startup, package imports,
asset verification and model loading. These merges do not establish that the
neural network's own forward pass became faster.

## What each merge added

| PR / current-history merge | What it added | Effect on latency and evidence |
| --- | --- | --- |
| #1 / `9f5251e` | A persistent vision worker: hash-check and load EfficientAD once, reuse it over a local Unix socket, and preprocess in process when worker mode is selected. Retains a cold fallback and per-request tensor/evidence checks. | Avoids repeated Torch import, model loading and model-file hashing. Saved-image worker operation was measured on the Jetson. |
| #2 / `7c37303` | The GUI starts/manages the vision worker and calls the inspection function in its own process. Drains worker stderr and respawns a failed worker. | Avoids another Python process and repeated NumPy/Pillow imports per upload. Recovery returns later inspections to the persistent path. The measured benefit belongs to the combined path; there is no separate PR #1 versus #2 benchmark. |
| #3 / `45a3ad6` | State polling every 250 ms while busy and 1500 ms while idle, replacing fixed 700 ms polling. Adds optional JPEG overlays; numerical maps remain `.npy`. | Changes display/encoding work, rather than model computation. PNG remains the default and was used in the reported device runs. No Jetson JPEG-versus-PNG or browser display-latency comparison was measured. |
| #4 / `84615bc` | Bounds request reading, isolates cold fallback outputs under fresh filenames, and adds model/preprocessing/failure tests. The follow-up fix enforces a total request deadline, including clients sending bytes slowly. | Prevents a stalled client or late worker output from blocking/corrupting later inspections. This is reliability work; it has no measured normal-forward speed gain. The deadline and real fallback/recovery were checked on the Jetson. |
| #6 / `886d086` | Limits prepared tensor files to 1 MiB and replies to 16 MiB, enforces the total request deadline, and restores a fresh reply timeout after request reading. | Bounds failure-related resource use and waiting. Overlaps the deadline work in #4; it is not a second independent speed improvement. |
| #5 / `65caf59` | An optional persistent Llama worker verifies assets/imports the runtime once, reuses the runner, and resets runner state before each note. Adds controller evidence checks and GUI lifecycle handling. | Intended to remove repeated Llama runtime setup. Initial device execution exposed startup and evidence-validation failures; accepted persistent advice was obtained only after #9–#12. Keeping workers resident also retains RAM between requests; idle timeouts release it later. |
| #7 / `c1ca52d` | Displays advisory runtime mode and boot errors; compares persistent and cold notes for the same inspection. The merged gate checks actual runtime modes and matching model/tokenizer hashes. | Helps validate and diagnose the optimization. It does not accelerate inference. No completed warm/cold acceptance series is claimed here. |
| #9 / `e6e4799` | Makes the per-request prompt optional during worker asset verification; the one-shot smoke CLI still requires its prompt. Reports early worker crashes with stderr. | Removed the observed startup `KeyError: 'prompt'`. The worker then started on the Jetson, but advisory acceptance still failed at the asset-manifest gate. |
| #10 / `d98d184` | Gets the exported model hash from the export manifest, checks the pinned source-assets manifest, and validates tokenizer evidence against its asset pins. Malformed manifests fail closed. | Addresses the observed “Persistent worker evidence does not match the pinned asset manifest” error. This restores valid advisory acceptance, rather than changing generation speed. |
| #11 / `43a751b` | Gives the advisory reply a fresh timeout after reading the request. | Prevents a nearly expired read deadline from causing a valid generated reply to fail during sending. No isolated speed gain measured. |
| #12 / `ca499bd` | Classifies undecodable/malformed transport replies as transport failures so the controller can use its cold fallback. Explicit worker validation errors remain failures. | Improves recovery behavior. The deployed snapshot containing #9–#12 produced an accepted real advisory on the Jetson. The individual fixes were not benchmarked separately. |

PR #8, the draft for running the advisory controller in process, was closed
without merging and receives no performance credit. The GUI still launches an
advisory controller process per request; the large Llama runtime is reused by
the optional persistent worker. The earlier local synchronization merge
`6eb8055` combined existing work and added no separate runtime optimization.

## What was actually measured

| Observation | Measured time | Scope and source |
| --- | --- | --- |
| Earlier cold vision series, 20 replays | Model forward p50 **1.585 s**; complete cold process p50 **8.015 s** | Repetitions of two saved images. Full process includes work beyond the forward pass. [Cold vision evidence](../manifests/jetson_offline_vision_latency_v1.json). |
| Persistent vision uploads before the later fixes | **2.053–2.243 s** inspection work; **1.484–1.688 s** model forwards | Three worker requests on one saved image. One forced cold fallback took **8.219 s**. [Restored historical device evidence](../manifests/jetson_persistent_vision_worker_probe_v1.json). |
| Request-deadline follow-up device test | **2.258–2.286 s** with the worker; **8.211 s** on forced cold fallback | One saved image, then worker recovery; no benchmark percentile claim. [Deadline device evidence](../manifests/vision_worker_request_deadline_probe_v1.json). |
| Latest deployed version, `ca499bd` | **2.337 s** inspection work; **1.768 s** model forward | One original `test/bent/017.png` upload, score **0.682808876**, provisional REJECT. [Latest device evidence](../manifests/jetson_persistent_advice_success_v1.json). |
| Earlier cold Llama series, eight requests | Native process p50 **20.690 s**; controller process p50 **21.079 s** | 33 generated tokens per request from one saved REVIEW inspection. [Cold advisory evidence](../manifests/jetson_offline_advice_latency_v1.json). |
| Latest persistent Llama, first advisory request | Controller-to-worker time **5.311 s**; reported first model load **2.939 s**; native inference **2.367 s** | 89 prompt tokens and 40 generated tokens. The note passed host validation; SQLite and the inspection grade were unchanged. [Latest device evidence](../manifests/jetson_persistent_advice_success_v1.json). |

The observed vision path is shorter because repeated runtime setup is avoided.
The roughly 1.5–1.8 seconds spent in EfficientAD itself remains visible within
the roughly 2.1–2.3 second inspection work. Forced fallback takes the slower
fresh-process path and adds recovery work.

The successful Llama result is a **first advisory request**, not a steady warm
latency figure: native statistics report model loading during that request.
The first request's load plus inference roughly accounts for its 5.311 seconds.
A later request may avoid that load, but it has not been measured in this
record. The old and new advisory series used different prompts/token counts.
No percentage speedup, warm p50/p95, or per-PR improvement is established.

## Timing boundaries and current limits

`model_forward_ms` measures only the vision network call. The newer
`capture_file_to_record_ms` includes saved-image preprocessing, worker/model
work, maps, geometry, grading and overlay creation, and excludes camera capture,
UI polling, SQLite write and Llama. The older full-process vision figure has a
broader scope, so it should not be treated as an identical benchmark.

`controller_runtime_wall_ms` for persistent advice measures the request to the
worker through response receipt. It excludes vision and is not the complete
browser-click-to-displayed-note time. The current run has no new memory peak
or simultaneous-load benchmark.

The final deployment produced a checked operator note and a live C270 preview.
Its inspection test used an original saved MVTec image. It adds no physical-nut
accuracy, defect-type, conveyor-throughput or FPS evidence. The deployment was
connected through SSH; it does not extend the earlier offline evidence to the
new merged runtime. TensorRT engine building remains separate: these timings
come from CPU vision, and no GPU speedup is claimed.

The next performance measurement should use repeated, matching saved inputs
and prompts, record first-use versus later requests separately, and report
application time alongside forward/generation time. That measurement was not
performed as part of this documentation update.
