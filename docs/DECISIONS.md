# Engineering decisions

| Decision | Reason |
| --- | --- |
| Separate vision and Llama runtimes | Preserve platform compatibility and keep advisory work independent of inspection. |
| CPU baseline before GPU claims | Establish output parity and measurements before changing execution format/backend. |
| Deterministic ForgeGrade authority | Advice and operator claims cannot change the original decision. |
| Hash-bound part recipes | Keep each model, classifier, policy, guidance and dataset coherent. |
| Persistent workers | Avoid repeated runtime/model setup; accept the resident-memory trade-off. |
| SQLite plus referenced artifacts | Keep local queryable history and immutable original evidence without image blobs in the database. |
| Explicit source/uncertainty labels | Distinguish benchmark/replay, camera diagnostics, physical parts and unknown types. |
| Scope-specific benchmarks | Keep cold/warm, direct Llama, application, PC and Jetson measurements distinct. |

The [technical report](TECHNICAL_REPORT.md) explains current implementation and
trade-offs. The [historical decision log](archive/2026-11-25/DECISIONS.md)
preserves alternatives, experiments and earlier checkpoints.
