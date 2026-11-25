# Learning record

Current concepts: dataset leakage and frozen thresholds; anomaly response versus
confirmed damage; confidence versus severity; deterministic decisions versus
generated advice; persistent-worker startup versus forward latency; and PC
software checks versus device proof.

On 25 November 2026 the documentation was consolidated to reduce duplication.
The user guide explains operation; the technical report holds methods/results;
setup holds commands; status holds the current checkpoint. The original
stage-by-stage explanations remain in the [historical learning log](archive/2026-11-25/LEARNING_LOG.md).
Checks covered archive preservation, links, all 38 feature entries, rendered
Markdown tables and 14 existing recipe/advice tests. No new device result was generated.

Useful checks when reading the results:

- Can a timing include model forward while excluding storage and display?
- Why do two repeated frames not establish two distinct physical parts?
- Why is a 25 W mode setting not measured power consumption?

See [TECHNICAL_REPORT](TECHNICAL_REPORT.md) for the answers in context and
[STATUS](STATUS.md) for the next evidence/submission work.
