# Stage 10 verification ledger

**Status: INCOMPLETE.** The Jetson has run the listed checks with Wi-Fi
disconnected, but no physical nut was available in the C270 view. The live
camera frame showed a room. A physical-part camera-to-grade-to-history-to-
advice demonstration and distinct-parts throughput are therefore unverified.
Stage 11 submission acceptance must not inherit a completed Stage 10 claim.

## What ran on the physical Jetson

| Check | Observed result | Evidence |
| --- | --- | --- |
| Saved-image cases with no Wi-Fi route | Normal fitting reference PASS; two exposed development images REVIEW; all three exact SQLite readbacks | [case series](../manifests/jetson_offline_case_series_v1.json) |
| Loopback operator page service | HTML/CSS/JS served locally; real ExecuTorch advice, BUSY response, one-second timeout and UI process restart preserved the REVIEW grade | [UI pilot](../manifests/jetson_offline_loopback_ui_v1.json) |
| Missing model and corrupt image | Missing model after accepted input persisted ERROR/REVIEW; corrupt copy failed SHA-256 intake before inference; valid replay recovered | [input recovery](../manifests/jetson_offline_failure_recovery_v1.json) |
| Camera failure and recovery | Absent V4L2 index failed without frame/grade; connected C270 captured a live room frame and saved a frozen-model map | [camera recovery](../manifests/jetson_offline_camera_recovery_v1.json) |
| Repeated vision runtime | 20 cold replays of two previously used images; all 20 saved and read back | [vision latency](../manifests/jetson_offline_vision_latency_v1.json) |
| Repeated advisory runtime | Eight fresh ExecuTorch generations from one unchanged saved record; all eight passed the evidence contract | [advice latency](../manifests/jetson_offline_advice_latency_v1.json) |
| Original-PNG upload through live UI, connected | One previously exposed bent image accepted, scored 0.682808876 above 0.227765799, overlay and SQLite readback matched; altered bytes rejected | [upload smoke](../manifests/jetson_ui_original_png_upload_v1.json) |
| Jetson-copy replay through live UI, connected | Same permitted bent image scored 0.682808876 without browser file picker; overlay and SQLite readback matched; non-preview path rejected | [local replay](../manifests/jetson_ui_local_replay_v1.json) |

The separate reports sampled Wi-Fi, ethernet IPv4 addresses and default
routes during each disconnected interval, then verified Wi-Fi recovery and
stopped their independent reconnect timers. Sampling cannot rule out a brief
between-sample route or an unexamined interface. The private run directories
listed in each manifest retain full logs, maps and records; raw camera imagery
is not committed. [Jetson runtime commands](JETSON_RUNTIME.md) reproduce the
tested paths with fresh output directories and transient unit names.
The upload smoke ran while connected and provides no additional disconnected
or physical-part acceptance.

## Measured runtime table

These percentiles use linear interpolation at `(n-1)*p`. Each run loaded its
model in a fresh process. Values are descriptive for the exact saved-input
workflows tested here.

| Scope | n | p50 | p95 | Limit |
| --- | ---: | ---: | ---: | --- |
| EfficientAD first forward | 20 | 1,585 ms | 1,684 ms | Excludes decode, process startup, UI, storage and Llama |
| Saved-image replay through SQLite commit | 20 | 7,499 ms | 7,677 ms | Repeats two images; no physical parts |
| Cold replay process wall | 20 | 8,015 ms | 8,276 ms | Sequential saved-image processes |
| Vision child peak RSS | 20 | 436,336 KiB | 437,163 KiB | Per-process peak, not whole system peak |
| Native ExecuTorch advice process wall | 8 | 20,690 ms | 20,942 ms | Repeats one saved inspection; vision ran separately |
| Advice controller process wall | 8 | 21,079 ms | 21,304 ms | Includes controller startup and native child |
| Native ExecuTorch sampled peak RSS | 8 | 1,172,206 KiB | 1,189,735 KiB | Sampled child RSS; may miss short peaks |

During the 20-replay vision run, 162 network/resource samples showed CPU
sensor 47.468–49.812 °C, GPU sensor 47.687–49.437 °C and minimum sampled
MemAvailable 4,263,332 KiB. During the eight-advice run, 169 samples showed
CPU sensor 47.031–50.062 °C, GPU sensor 47.375–49.750 °C and minimum sampled
MemAvailable 3,591,692 KiB. These separate runs do not measure simultaneous
camera, vision and Llama pressure or establish absence of thermal throttling.

## Acceptance still required

| Roadmap gate | Current evidence gap |
| --- | --- |
| Physical nut camera inspection | The user clarified that they only have a dataset nut photo displayed on a screen. No physical nut has been presented to the C270 or inspected. The earlier C270 room frame is a diagnostic, not an inspection. |
| Disconnected complete application from cold start | Components ran locally in separate proven paths; one physical camera event through grade, SQLite, operator UI and ExecuTorch while disconnected has not run. |
| Normal, defective and uncertain real cases | Saved MVTec references exercised PASS and REVIEW. They are repeated known images; real camera-domain outcomes remain unknown. |
| Distinct parts per minute and physical false positives | Zero distinct physical parts were inspected; no rate or field false-positive estimate is valid. The PC benchmark result is a different domain and cannot fill this gate. |
| Rendered operator page while disconnected | Jetson loopback HTML/API worked and Firefox is now available on the Jetson, but disconnected local browser rendering has not been tested. The previous browser proof used a connected SSH tunnel. |
| Unplug/replug and power restart | Unavailable camera index and UI process restarts were tested; physical USB unplug/replug, OS reboot and power-loss recovery were not. |

The next physical acceptance run needs a declared nut in a stable, lit C270
view and a reproducible part identity. Until then, Stage 10 remains open even
though the saved-image and diagnostic checks above are verified.
