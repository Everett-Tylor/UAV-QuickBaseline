# Round 11: context and boundary refinement for DINOv3

The user reported official scores of 68.45 for round 8 and 68.36 for the round-10 robust640 submission. Round 11 therefore initializes from round 8, not round 10. It modifies the segmentation decoder while retaining the DINOv3 ViT-B/16 encoder and the fixed 6296/700 split. No official round-11 score exists yet.

## Head changes

* A residual context adapter operates on the stride-16 decoder feature. It pools at 1/2/4/8 bins, projects each branch to 32 channels, resizes and concatenates them, and projects back to 128 channels.
* A residual detail adapter refines the concatenated 128-channel decoder and 32-channel RGB-detail features before the existing semantic classifier.
* An auxiliary one-channel boundary head supervises these shared features during training. Boundaries are computed only between nonignored adjacent training labels, expanded with a 3x3 neighbourhood. Ignored pixels, their neighbours and the outer image border are excluded from boundary loss. BCE positive weighting is computed within each batch and capped at 10. Its loss coefficient is 0.1.

The context and detail adapters have zero-initialized output projections. This makes initial semantic logits identical to the old model, as verified by `test_dino_head.py`. The boundary head is not evaluated during inference. This is one encoder and one segmentation model, not an ensemble.

Checkpoint upgrade permits missing parameters only in the three new modules; any missing original parameter or unexpected key is rejected. Checkpoints store `head_variant=context_boundary`; prediction reads this field and strictly loads the correct architecture. Historical checkpoints without the field retain the original pyramid architecture.

## Training and selection

Six epochs at 640 pixels, batch 4 and accumulation 2. Epoch 1 freezes the backbone; epochs 2–6 fine-tune it at LR 1e-6. Existing decoder LR is 2e-5; added-module LR is 2e-4, with warmup and polynomial decay. Training retains EMA, MixStyle, mild augmentation and CE/Dice/Lovasz segmentation losses. Only official training images and labels are used. Test images are never training or adaptation inputs.

From the workspace root:

```powershell
python outputs/UAV-QuickBaseline/test_dino_head.py
python outputs/UAV-QuickBaseline/round11_head_pipeline.py --smoke
python outputs/UAV-QuickBaseline/round11_head_pipeline.py
```

The smoke test runs with an unfrozen backbone to check the more demanding gradient/memory case. The pipeline trains, evaluates the best EMA checkpoint with the baseline's exact scales 512/640/768/896, flip and brightness settings, and exports only if overall mIoU improves and neither dark nor low-contrast group falls by more than 0.1 percentage point. Otherwise it records `no_validated_improvement` without a new ZIP. This local gate does not guarantee an official-score improvement; round 10 demonstrated the limitation of local validation.

Logs/status: `outputs/round11_head`. Run: `outputs/runs/round11_context_boundary640`. On successful selection, the pipeline validates and exports 1300 masks to `outputs/submission_round11_head_single.zip`. Architecture, losses and training settings are experiments awaiting evaluation, not established gains.

