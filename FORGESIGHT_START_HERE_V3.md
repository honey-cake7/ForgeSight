# First message to paste into Codex — ForgeSight AI (step-by-step)

I am building **ForgeSight AI** and want to learn while making a real, offline industrial defect-inspection system on my NVIDIA Jetson Orin Nano.

**Read these files in the project folder first:**
- `FORGESIGHT_STEP_BY_STEP_ROADMAP.md` — authoritative stage order, proof criteria and exact next steps
- `FORGESIGHT_CODEX_MASTER_PROMPT_V3.md` — engineering standards and teaching rules
- `Edge AI Quality Control Model Pipeline.pdf` — background reference, if available

GitHub repository: `https://github.com/honey-cake7/Forge_Sight.git`.

**Important: I DO NOT want a date-wise plan or a complete project generated in one shot.** Teach me one task at a time: why it is necessary, how it works, what choices we have, which file/command to use and how we know it is correct. Implement and test one meaningful step, then show me the actual result, make an honest Git commit when verified and tell me exactly which next step comes after it. Stop at each stage gate for my review.

The challenge mandates real surface defect/anomaly detection, type and severity, useful operator recommendations, inspection logging, offline edge operation and **Llama-family inference on-device via ExecuTorch**. A model exported but not run, an LLM simulation or a PC-only prototype is not Jetson proof. Verify the exact hardware first. Our preferred benchmark is a single-component MVTec AD `metal_nut` prototype; final physical camera validation must use controlled real samples.

**Start only with Stage 0, Substep 0.1:** inspect my current workspace, files, `git status`, branch and remotes, relevant Python/GPU environment, and whether Jetson is physically connected. This is a read-only audit; preserve all local work, even if GitHub is empty. Explain your findings and the single next safe action. Do not install anything or code future stages yet.
