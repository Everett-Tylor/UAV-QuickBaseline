# Round 10: DINOv3 single-model generalization experiments

Round 9 SegFormer B5 reached 72.3544% local validation mIoU with TTA, below the previous DINOv3 result of 78.3839%. This round returns to the round-8 DINOv3 MixStyle checkpoint, whose user-reported official score was 68.45. The official score target of 70 has not been achieved or verified.

Two independent four-epoch candidates start from the same round-8 checkpoint:

* `robust640`: 640-pixel training with brightness, contrast, saturation, gamma, blur and bounded random crop augmentation. Batch 8, accumulation 1.
* `mixed768`: 768-pixel training with a 50/50 mixture of full-scene resize and random scale/crop, preserving some native-scale detail while retaining scene context. Batch 4, accumulation 2.

Both retain MixStyle, the same nine-class DINOv3 decoder, EMA, BF16, gradient checkpointing, fixed 6296/700 train/validation split, and the existing CE/Dice/Lovasz loss. Encoder LR is 2e-6, decoder LR 2e-5. No test images or extra data enter training. The 768-pixel experiment is independent, not a continuation of the 640-pixel experiment.

`round10_dino_pipeline.py` runs both candidates sequentially. Use `--smoke` first to verify each augmentation and memory setting; then run without the flag. Existing experiment directories cause a stop rather than silent overwriting.

```powershell
python outputs/UAV-QuickBaseline/round10_dino_pipeline.py --workspace WORKSPACE_ROOT --smoke
python outputs/UAV-QuickBaseline/round10_dino_pipeline.py --workspace WORKSPACE_ROOT
```

Each candidate is evaluated separately with exactly the round-8 inference settings: scales 512/640/768/896, horizontal flip, brightness floor 0.35 and maximum brightening factor 1.25. The same checkpoint supplies every prediction in each evaluation. No cross-model averaging or weight merging is used.

Selection criteria are set before evaluation: overall mIoU must be at least the baseline; dark and low-contrast group mIoUs may not decrease by more than 0.1 percentage point each. Among eligible candidates, maximize 0.5 times overall mIoU plus 0.25 times each group mIoU; this score must improve on the baseline. The groups overlap and this selection does not establish statistical significance or guarantee an official-score gain.

If neither candidate qualifies, the pipeline records `no_validated_improvement` and does not generate a new submission. If one qualifies, it generates 1300 masks using only that selected checkpoint, validates names, dimensions, IDs and ZIP CRC, and saves `outputs/submission_round10_dino_single.zip`. Status, per-candidate logs and reports are under `outputs/round10_dino`. Official score remains null until reported by the user.

Training source change: `dino_train.py --augmentation mild|robust|mixed`. The default remains `mild`, preserving previous commands. Model architecture and checkpoint state keys are unchanged.

