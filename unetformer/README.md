# UNetFormer: UAV semantic segmentation

The model is the official UNetFormer implementation by Libo Wang et al., from
[GeoSeg](https://github.com/WangLibo1995/GeoSeg), pinned to commit
`9453fe48209c4626b29e35e61bab93b61212c4b1`. The upstream model file is unchanged.
It uses a ResNet18 encoder, global/local attention decoder, feature refinement
and auxiliary supervision. The training and export scripts adapt it to this
competition's nine class IDs. Model code is distributed under the upstream
GPL-3.0 license included in `LICENSE`; preserve upstream attribution.
ResNet18 SWSL weights come from Meta's semi-weakly supervised ImageNet models,
under CC-BY-NC-4.0, downloaded separately with a full SHA-256 check.

## Data and evaluation

- 6,996 labeled images, existing fixed split: 6,296 training / 700 validation.
- Class 0 ignored; report mean IoU across classes 1–8.
- Labels: background, building, road, water, barren, vegetation, agricultural, vehicle.
- Class weights are computed from training masks only.
- Test2 has 1,300 images and never enters training or model selection.
- Validation uses original full-resolution masks. Logits are resized before argmax.
- Compare with the prior B0 result of 74.2677% using the identical split.
  This image-level validation split may contain similar scenes; local gains are
  not official competition scores.

## Reproduce

Observed environment: Windows, Python 3.14, RTX 5060 Laptop 8GB, CUDA 13.0.

```powershell
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python download_pretrained.py --out pretrained
python unetformer_pipeline.py --images DATA/train/images --masks DATA/train/masks --split split.json --source pretrained/resnet18_swsl.pth --cache cache --test-images TEST2/images --out run --epochs 24 --finetune-epochs 4 --batch 4 --workers 4
```

The pipeline validates image/mask pairs and creates a lossless 768-pixel cache.
It trains 24 epochs at 512 pixels, then four at 768 pixels, initialized from the
best first-stage checkpoint. Effective batch size is eight with gradient
accumulation. AdamW, polynomial decay, warmup, FP16, clipping, flips, right-angle
rotations and color augmentation are used. Loss is training-class-weighted
cross entropy + 0.5 Dice, with auxiliary loss weight 0.4.

The best checkpoint is selected by validation mIoU. Its native size without
augmentation, 512+horizontal flip and 768+horizontal flip are compared before
test prediction. This is a single model; no B0 ensembling or fallback occurs.
`history.jsonl`, `validation.json` and `status.json` record measured results.

## Predict from a saved run

```powershell
python predict_unetformer.py --run run --images TEST2/images --out export
```

Requires exactly `test2_1.png` through `test2_1300.png`. Output is
`submission_unetformer_test2.zip`: same-name PNGs at the ZIP root, original
dimensions, mode L, class IDs 0–8. All files and ZIP CRCs are checked, and a
SHA-256 manifest is saved. The export directory must be fresh.

Training checkpoints support inference and stage initialization; optimizer state
is not saved for interrupted-training resume. FP16 and cuDNN benchmarking can
cause small differences at class boundaries on repeat inference. Weights,
data and predictions remain local; this GitHub branch contains source and reports.
Docker and technical PDF are deferred.

## Checks

`test_unetformer.py` performs real GPU forward/backward steps at 512 and 768,
checks training auxiliary outputs and evaluation outputs, and verifies the
all-ignored loss and confusion-matrix calculation. Run from the original
workspace with pretrained weights at `work/unetformer_pretrained/resnet18_swsl.pth`.
