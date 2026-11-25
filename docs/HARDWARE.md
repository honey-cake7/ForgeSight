# Hardware record

| Item | Recorded fact |
| --- | --- |
| Target | NVIDIA Jetson Orin Nano 8GB, ARM64 Linux |
| Audited software | Ubuntu 20.04.6, L4T 35.6.1, system Python 3.8.10, CUDA 11.4 |
| Camera | Logitech C270 USB; 640 x 480 MJPG capture diagnostics recorded |
| Vision environment | Separate Python 3.12.14 / Torch 2.14.0+cpu |
| Advisory environment | Separate ExecuTorch 1.5.1+cpu / XNNPACK runtime under `~/.local/share/forgesight/llama_cpu_v1` |

The supplied timing report names a 25 W power mode. Electrical consumption,
peak watts and energy per inspection have not been measured. Its resident
memory snapshot is not a peak-resource benchmark. CPU results do not establish
GPU inference speed or physical-camera accuracy.

[Device audit](../manifests/jetson_device_audit_v1.json),
[runtime installation](../manifests/jetson_runtime_installation_v1.json),
[current measurements](TECHNICAL_REPORT.md#jetson-performance-and-resources),
[full historical hardware log](archive/2026-11-25/HARDWARE.md).
