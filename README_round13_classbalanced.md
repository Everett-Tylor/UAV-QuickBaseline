# Round 13: class-specific pseudo-label confidence thresholds

Round 12 improved local overall mIoU from 78.3839% to 78.4366%, but barren-land IoU fell from 53.08% to 52.81%. Its fixed 0.95 confidence filter retained 32.99% of teacher-predicted barren pixels, versus 76.95% of vegetation pixels. Official round-12 feedback is still pending. These observations motivate a controlled filtering experiment; they do not prove the filter caused the class regression.

This experiment is inspired by [Class-Balanced Self-Training, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.html). It is a conservative adaptation, not an exact reproduction of that paper: teacher argmax labels are never reassigned, and no spatial prior is used.

## Filtering

The teacher remains the round-8 single DINOv3 checkpoint with user-reported official score 68.45. Its eight inference views are unchanged. Among pixels with at least 75% view agreement, confidence histograms are accumulated separately for classes 1–8. Each class aims to retain its top 60% confidence pixels. Thresholds are constrained to [0.92, 0.99]; the confidence floor, ceiling, histogram ties and agreement filter mean actual retention need not equal 60%. Ignore class 0 is always excluded.

Confidence is conservatively quantized to unsigned 16-bit values. Calibration uses 0.001-wide histogram bins. Filtering then compares each stored score against its own class threshold, without changing the semantic label. Empty classes receive the conservative maximum threshold. Cached teacher labels, confidence and agreement masks remain local for reproducibility.

The existing input-duplicate rejection, training/validation image exclusion, teacher/mask checksum checks and 2% per-image coverage filter remain active. Use of test_2 for pseudo-label training was explicitly confirmed as allowed by the user in round 12.

## Controlled training

Round 12 and round 13 use the same original teacher/student initialization, seed 20261005, original pyramid architecture, 3 epochs at 640 pixels, supervised batch 4, pseudo batch 2, accumulation 2, MixStyle, mild augmentations and learning rates. The pseudo CE weight still ramps to 0.25. Changing the eligible data changes batch contents; this is not a bit-for-bit paired experiment. Neither model averaging nor multi-model inference is used.

```powershell
python outputs/UAV-QuickBaseline/test_dino_pseudo.py
python outputs/UAV-QuickBaseline/round13_classbalanced_pipeline.py --workspace WORKSPACE_ROOT
```

The generator's default remains fixed-threshold filtering so previous commands retain their behaviour. Class filtering is selected with `--class-keep 0.6 --confidence-floor 0.92 --confidence-ceiling 0.99`. The round-13 wrapper uses the shared training/evaluation pipeline with distinct output paths.

Status and reports: `outputs/round13_classbalanced`. Training: `outputs/runs/round13_classbalanced640`. Candidate ZIP: `outputs/submission_round13_classbalanced_single.zip`. Evaluation records comparisons with both round 8 and round 12. The same predeclared stability gate from round 12 applies; an exported experimental package does not establish an official-score improvement. No new score is claimed before evaluation.

