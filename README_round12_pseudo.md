# Round 12: single-model pseudo-label self training

The user explicitly confirmed that competition rules allow pseudo-label training on the official test_2 images. This experiment uses those 1300 images as unlabeled target-domain data. The confirmation is user-provided; this repository does not independently establish the organizer's policy. The earlier prohibition on test-image training in previous rounds describes those rounds, not this newly authorized experiment.

Teacher and student initialization: round-8 DINOv3 MixStyle, user-reported official score 68.45. Round-10 robust640 scored 68.36. Round-11 decoder changes did not improve local validation and are not used here. The student has the original pyramid head.

## Pseudo-label generation

One fixed teacher predicts with the same scales 512/640/768/896, horizontal flip and brightness preprocessing as its existing submission. Softmax probabilities are averaged across eight views of the same model. A pixel is kept only when its maximum mean probability is at least 0.95, at least 75% of views agree with its assigned label, and its label is not zero. All other pixels become ignore label 0. These are model confidence thresholds, not measured correctness probabilities.

The generator rejects pixel duplicates and any official training/validation image found in the supplied input. Its completed manifest records teacher SHA256, image pixel hashes, mask hashes, per-image coverage, and retained pixels per class. Images with less than 2% retained pixels are excluded from training; fewer than 100 usable images stops the pipeline for inspection.

## Training

Three epochs at 640 pixels, supervised batch 4 plus pseudo batch 2, accumulation 2. Real labels retain weighted CE/Dice/Lovasz loss. Pseudo labels use weighted CE only, with coefficient ramping from zero to 0.25 during the first epoch. This limits the effect of incorrect pseudo labels. Both data sources use matched image/mask augmentation. Encoder LR 1e-6; decoder LR 1e-5; EMA and MixStyle retained.

The 700-image validation set remains excluded from all training. Before training, pseudo input pixels are checked against validation hashes, image and mask contents against the manifest, and student initialization against the teacher checksum. No new human labels or outside images are used. The teacher is not loaded during student training or final inference; the submission uses one student checkpoint.

```powershell
python outputs/UAV-QuickBaseline/test_dino_pseudo.py
python outputs/UAV-QuickBaseline/round12_pseudo_pipeline.py --workspace WORKSPACE_ROOT
```

The pipeline generates pseudo labels, performs a real supervised-plus-pseudo backward smoke test, trains, validates and conditionally exports. It refuses existing run directories to prevent silent overwrites. Test data is explicitly selected by this round's pipeline; the generic generator requires an input directory and has no default test path.

## Interpretation and export

This is a target-domain adaptation experiment. Before training, the export stability gate is set to allow at most 0.3 percentage point overall local mIoU regression and 0.5 point regression in each dark/low-contrast group. This is deliberately a different gate from rounds 10–11: passing it authorizes an experimental prediction package, **not a claim of validation improvement**. All actual scores and deltas are retained. An official-score gain, including the suggested 1–2 points, remains unverified until submission feedback.

Status and reports: `outputs/round12_pseudo`. Training run: `outputs/runs/round12_pseudo640`. Successful export: `outputs/submission_round12_pseudo_single.zip`, containing 1300 validated PNG masks. Generated pseudo masks are training artifacts and are never used as the final prediction ZIP.

