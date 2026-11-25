# ForgeSight — Agent Instructions

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
- Phase 1 source includes inspection, plugins, local ExecuTorch advice and history/report tools; offline saved-image operation has physical Jetson evidence.
- Use `docs/STATUS.md` for current evidence scope and next actions. Earlier stage-by-stage records are preserved under `docs/archive/`.
- Maintain the concise setup, user guide and technical report plus the required engineering summaries; do not append historical logs back into current guides.
- Documentation changes stay on local `main`; do not push until the user explicitly confirms.
