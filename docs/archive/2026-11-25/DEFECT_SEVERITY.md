> Historical snapshot archived on 25 November 2026. This page preserves development history; use the [current documentation](../../../README.md) for supported instructions.

# Provisional defect severity

Launch the Jetson operator app with
`--defect-type-profile configs/defect_type_mvtec_four_v1.json`. The current
result and inspection history show the provisional severity for flagged original
MVTec replays and declared MVTec uploads. The SQLite inspection record stores
the policy ID, level, reason and any measurements under `assessment.severity`.
ForgeGrade retains authority for PASS, REVIEW and REJECT.

## Rules

Policy `mvtec-metal-nut-severity-v1` uses the largest highlighted response
and the nut outline already found during type classification. The default
size band is Low below the Medium cutoff, Medium at or above that cutoff
and High at or above the High cutoff.

| Defect type | Measurement | Medium starts at | High starts at |
| --- | --- | ---: | ---: |
| Scratch | Response bounding-box diagonal divided by estimated nut diameter | 15% | 35% |
| Discoloration | Response area divided by estimated nut outline area | 2% | 8% |
| Bent | Response area divided by estimated nut outline area | 5% | 15% |

The diameter is the mean of the two sides of the rotated rectangle around
the nut outline. The outline area is the area enclosed by the outer contour,
including the opening. The response can extend outside the metal. Both
ratios describe highlighted image pixels, and the cutoffs are demonstration
rules set before running the severity evaluation. A scratch or bent response
that includes the approximate inner rim moves up one level, capped at High.
This gives rim responses priority for review because of the thread opening.

Flip is reported as an orientation issue with severity Unassessed. Unknown
types, missing geometry, uncertain locations and implausible response extents
also give Unassessed. Unflagged images have no severity assessment.

## Reproduce the checks

From the repository root, run the frozen model on the 24 subtype test images
and four development previews with the existing PC vision environment.

```sh
python -B -m scripts.evaluate_defect_severity --model artifacts/vision/edge_export_v1/efficientad_s_calibrated.ts --output artifacts/vision/defect_severity_v1/pc.json
```

For the isolated Jetson comparison, follow the
[defect-type device instructions](DEFECT_TYPE.md#reproduce-the-isolated-jetson-comparison).
Use the four-type candidate and its existing type `evaluation.json`, and add
`--severity-expectations pc.json` after staging the generated PC report.
The verifier exercises 24 benchmark uploads and four saved-image replays per
profile and checks the result API, history and SQLite.

## Results

The 24 benchmark uploads produced five Low, six Medium, one High, seven
Unassessed and five unflagged results. The four development replays produced
three Unassessed results and one unflagged result. All 28 Jetson outcomes
matched the PC. Detector scores, maps, overlays and ForgeGrade decisions
matched across paired runs. The severity rule ran in 0.045 ms median and
0.056 ms p95 over 22 flagged images, measured after subtype feature extraction.
See the [verification record](../../../manifests/mvtec_defect_severity_v1.json).

MVTec supplies defect-type labels. Severity accuracy against real physical
damage and factory acceptance thresholds requires separately labeled and
calibrated parts.
