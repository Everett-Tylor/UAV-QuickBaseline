# UNetFormer continuation and tuning

This experiment continues the previous UNetFormer checkpoint (epoch 23,
validation mIoU 75.4886%). It keeps the same 6,296/700 labeled-image split.
Test2 is excluded from training and selection. Original masks are used for
full-resolution validation; class 0 is ignored and mIoU averages classes 1–8.

The prior barren-class precision/recall were 56.27%/76.81%, suggesting excessive
false positives. The experiments therefore reduce the spread of class weights
instead of increasing barren loss weight further.

## Predeclared training comparisons

All three trials initialize independently from the same prior checkpoint.
BatchNorm running statistics are frozen, affine parameters remain trainable,
and decoder dropout remains active. Effective batch is eight (batch four,
two-step accumulation). Cross entropy + 0.5 Dice and 0.4 auxiliary supervision
are retained. AdamW, warmup, polynomial learning-rate decay and clipping are used.

| Trial | Epochs | Encoder LR | Decoder LR | Class weights | Color range | Sizes |
|---|---:|---:|---:|---|---|---|
| low_lr_control | 6 | 5e-6 | 2.5e-5 | previous | 0.8–1.2 | 512 |
| gentle_weights | 8 | 8e-6 | 4e-5 | normalized square root of previous weights | 0.88–1.12 | 512 |
| scale_regularization | 6 | 5e-6 | 2.5e-5 | same gentler weights | 0.85–1.15 | 512,448,576 by epoch |

Each epoch evaluates both raw and exponential-moving-average weights (decay
0.995) at 512+horizontal flip. EMA updates after optimizer steps and copies
buffers. The best checkpoint across trials must beat the remeasured baseline;
otherwise that baseline remains selected. Recipes change several factors, so
the comparison cannot isolate the causal contribution of each factor.

## Inference comparisons

Only the selected checkpoint is compared across four predefined settings:
512 with horizontal flip; 512 with horizontal/vertical flip combinations;
448/512/576 with horizontal flip; and 512/640 with horizontal flip. Logits are
averaged, resized to original dimensions, then argmax is applied. The selected
recipe is used unchanged for test2. This is a single model, not an ensemble
of separately trained checkpoints. Repeated holdout selection can overestimate
generalization; no official competition improvement is claimed.

## Run

Use the prior environment (`requirements.txt`, Python 3.14, CUDA 13.0) and the
previously generated 768-pixel lossless cache and training-only class audit.

```powershell
python tune.py --images DATA/train/images --masks DATA/train/masks --cache CACHE768 --split split.json --init PRIOR_RUN/best.pth --audit PRIOR_RUN/data_audit.json --test-images TEST2/images --out run
python predict_tuned.py --run run --images TEST2/images --out export
python verify_export.py --zip export/submission_unetformer_tuned_test2.zip --report export/verification.json
```

Output directories must be fresh. Trial `last_training.pth` files contain
optimizer/scaler/EMA state for recovery work, but the CLI does not automatically
resume interrupted runs. `run/best.pth` and `run/selection.json` are sufficient
for prediction. Data and weights stay local. PNG filenames, count, dimensions,
mode, class IDs and ZIP CRCs are verified independently after export.

## Upstream and changes

UNetFormer is from [WangLibo1995/GeoSeg](https://github.com/WangLibo1995/GeoSeg),
revision `9453fe48209c4626b29e35e61bab93b61212c4b1`, under the included GPL-3.0
license. The model has one compatibility change: 4D reflection padding uses a
four-element tuple when padding width. This enables 448/576/640 inputs under
PyTorch 2.14 without changing the intended padding operation. The original
training helpers and fixed split are preserved from the preceding experiment.

`test_tuning.py` checks GPU training at all three training scales, frozen BN
statistics, EMA updates, and all inference settings. It expects the original
workspace's prior checkpoint and one test image. FP16/cuDNN benchmarking can
cause small pixel differences across repeated inference runs.
