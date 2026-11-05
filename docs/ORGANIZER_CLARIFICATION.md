# PS5 organizer clarification

Source: organizer reply shared by the user on 2026-11-06 (Asia/Calcutta).
This records eligibility guidance, not a device test or completed compliance.

## Reply as supplied

> Yes, NVIDIA Jetson Orin can be considered an eligible hardware platform for
> Problem Statement 5, as it is based on a 64-bit ARM architecture and supports Linux.
>
> However, please ensure that your implementation satisfies the mandatory
> technology requirements specified for PS5:
>
> Use a model from the Llama family for the advisory or reasoning component.
>
> Deploy the Llama model on-device using ExecuTorch as the reference runtime.
>
> Ensure that the defect detection and quality-control pipeline operates
> on-device, without cloud dependency.
>
> You should also clearly document the hardware configuration, software stack,
> model deployment and inference pipeline in your technical report.

## Application to ForgeSight

Continue with the proposed Jetson Orin Nano route. The organizer's acceptance
of the Orin platform family does not identify our physical module, RAM, JetPack,
camera, or runtime feasibility. Those remain subject to the device audit.
No model/runtime substitution or architecture change is needed from this reply.

| Organizer requirement | Planned implementation | Evidence still required |
| --- | --- | --- |
| Llama-family advisory/reasoning | Pinned Llama 3.2 1B Instruct candidate | Actual model-generated advice, linked to inspection evidence |
| On-device ExecuTorch deployment | Quantized PTE and native Jetson ExecuTorch runtime | Real Jetson generation with artifact/runtime provenance and measurements |
| On-device detection and quality control without cloud dependency | Local vision, deterministic grading, local logs, and separate asynchronous Llama worker | Actual disconnected end-to-end device run, including local assets and cold start |
| Technical report | Record actual hardware, software, model deployment, and inference flow | Completed device-specific report backed by observed commands and results |

Stage 2.7 now has real desktop generation through ExecuTorch, following the
verified export. It does not close the on-device gate: Jetson execution and
offline operation remain unverified. Independent desktop vision work may
proceed under the roadmap's hardware-blocker rule. Downloading/exporting on the
PC is preparation for local
deployment; normal inspection operation must not depend on that PC or a cloud
service. Keep the roadmap's offline and device acceptance criteria unchanged.

The final technical report must distinguish desktop preparation from the
observed Jetson setup. Record the exact module/RAM, Linux/JetPack/L4T, camera,
runtime/backend/build versions, vision model and Llama revision/quantization,
artifact hashes, and deployment commands. Describe the on-device flow from
capture through detection, grading and logging to asynchronous advisory, then
include actual generation, resource/timing, and disconnected-run evidence.
