# Frozen baseline evaluation — Stage 3.4

The trained, calibrated FP32 EfficientAD-S model was evaluated once on the 111
reserved metal-nut images. No model weights, normalization, image threshold,
precision or preprocessing were changed using those results. The historical
calibration reference remains an immutable pre-evaluation snapshot.

| Actual result | Count |
| --- | ---: |
| Defective images flagged | 87 / 89 |
| Good images not flagged | 20 / 22 |
| Missed defects | 2 |
| False alarms | 2 |

Recall and precision are both 97.75%; accuracy is 96.40%. Normal-image false
alarm rate is 9.09% (only 22 good images). Image AUROC is 0.975996 and pixel
AUROC is 0.969206 on the original 700x700 annotation grid, after bilinear map
resize. AUROC measures ranking, not physical damage size or thresholded mask
accuracy. No separate pixel-mask threshold has been validated.

Misses: `test/scratch/000.png`, `test/scratch/022.png`. False alarms:
`test/good/007.png`, `test/good/016.png`. These cases remain in the result;
the image threshold stays 0.22776579856872559. Predicted types remain `unknown`.
Grouping detections by ground-truth class is not type-classification evidence.

The predeclared geometry support rule uses source-grid responses strictly above
the frozen image threshold, 8-connected regions and no morphology. This is a
provisional model-response extent, not a separately calibrated defect mask.
The first member of each truth group and all four errors are shown in
`artifacts/vision/holdout_evaluation_v1/evaluation_review.png`; every prediction,
map and annotation is saved locally. Raw MVTec images and maps are not published.
Committed evidence: `manifests/efficientad_s_evaluation_v1.json` and the exposure
ledger. The ledger refuses repeated holdout inference; do not delete it to rerun
a test or select a better model. New model selection needs fresh evaluation data.

Reproduce metrics from saved predictions, without another model forward:

```powershell
wsl -d Ubuntu --exec /root/.cache/forgesight/vision-v1/venv/bin/python -B -m forgesight.vision_evaluate --output artifacts/vision/holdout_evaluation_v1 --summarize-only
```

The CLI saves `metrics_reproduced.json` exclusively; an existing reproduction is
preserved. `summarize(Path(...))` can also recompute in memory. Saved NPZ hashes
are checked before metrics. The first evaluation command was the same command
without `--summarize-only`, with `--allow-final-holdout` and a new output path.

This proves a PC saved-image anomaly baseline. The two misses rule out claiming
perfect screening. Production acceptance criteria, defect type, camera domain,
physical tolerances, Jetson latency and offline deployment remain unverified.
Next: inspection records and an explicit one-part capture gate, then pixel-only
grading with uncertainty routed to REVIEW.
