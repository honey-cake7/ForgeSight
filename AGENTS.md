# ForgeSight AI — Agent Instructions

## Scope
- Follow `FORGESIGHT_STEP_BY_STEP_ROADMAP.md` as the stage order and proof gate.
- Follow `FORGESIGHT_CODEX_MASTER_PROMPT_V3.md` for engineering and teaching standards.
- Treat the PDF architecture as reference, not as an unverified specification.

## Workflow
- Work on one meaningful substep at a time; explain purpose, choices, command, and proof.
- Preserve local work. Do not install or implement future stages early.
- Run focused tests and record evidence before committing.
- After verified work, commit honestly and push `origin main` when authorized.
- Keep responses concise; avoid repeating completed audits and logs.

## Product constraints
- Final system must run offline on NVIDIA Jetson Orin Nano 8GB.
- Llama-family advisory inference must run on-device through ExecuTorch.
- Defect detection must use real inspection evidence; do not claim unverified accuracy or FPS.
- Keep inference efficient and report latency, memory, and accuracy trade-offs.

## Jetson safety
- Do not change system Python or global packages without explicit approval.
- Prefer the isolated runtime under `/home/nvidia/.local/share/forgesight/llama_cpu_v1`.
- Never store passwords, tokens, or private keys in the repository.
- Verify device-side evidence separately from PC-only tests.

## Current status
- Llama ExecuTorch generation has been verified on the physical Jetson.
- Stage 8.1 verified a live C270 frame and audited missing vision dependencies.
- Next planned substep: prepare a separate compatible Jetson vision runtime and measure frozen EfficientAD inference.
