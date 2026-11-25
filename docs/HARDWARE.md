# Hardware record

| Item | Recorded fact |
| --- | --- |
| Target | NVIDIA Jetson Orin Nano 8GB, ARM64 Linux |
| Audited software | Ubuntu 20.04.6, L4T 35.6.1, system Python 3.8.10, CUDA 11.4 |
| Camera | Logitech C270 USB; 640 x 480 MJPG capture diagnostics recorded |
| Vision environment | Separate Python 3.12.14 / Torch 2.14.0+cpu |
| Advisory environment | Private ExecuTorch 1.5.1+cpu / XNNPACK environment; release layout in [SETUP](SETUP.md#launch-files-and-logs) |

On 25 November, INA3221 VDD_IN readings measured **8.52 W sampled peak** and
**6.60 W active-phase mean** during 12 saved-image attempts with live preview
(584 samples over 58.07 s; 100 ms target interval). Same-run preview-only mean
was 4.16 W; an earlier 30 s preview window averaged 5.46 W. These are module
input watts, not wall/whole-carrier power or absolute worst-case peaks. The
configured 25 W mode was unchanged. Energy per inspection and peak memory
remain unmeasured. [Power evidence](../manifests/jetson_power_measurement_v1.json).
CPU results do not establish GPU speed or physical-camera accuracy.

[Device audit](../manifests/jetson_device_audit_v1.json),
[runtime installation](../manifests/jetson_runtime_installation_v1.json),
[current measurements](TECHNICAL_REPORT.md#jetson-performance-and-resources),
[full historical hardware log](archive/2026-11-25/HARDWARE.md).
