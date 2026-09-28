# UNetFormer tuning results

Validation: **77.3524% mIoU**, up **1.8637 percentage points** from the remeasured 75.4887% baseline.

Same 6,296/700 split; test2 excluded from training and selection. These are local validation results, not official competition scores.

## Trial comparison (512 + horizontal flip)

| Trial | Best epoch | Weights | mIoU |
|---|---:|---|---:|
| low_lr_control | 3 | ema | 76.1836% |
| gentle_weights | 2 | ema | 76.7364% |
| scale_regularization | 5 | ema | 76.8613% |

Selected checkpoint: scale_regularization, epoch 5, ema.

## Inference comparison

| Setting | mIoU |
|---|---:|
| 512_hflip | 76.8610% |
| 512_hvflip | 77.3524% |
| 448_512_576_hflip | 76.8545% |
| 512_640_hflip | 76.8180% |

Selected inference: `512_hvflip`. One model is used; no checkpoint ensemble.

## Per-class changes

| Class | Previous IoU | New IoU | Change (pp) |
|---|---:|---:|---:|
| Background | 67.3430% | 69.2280% | +1.8850 |
| Building | 83.1013% | 84.0084% | +0.9071 |
| Road | 80.0541% | 81.6627% | +1.6086 |
| Water | 86.7996% | 88.5466% | +1.7470 |
| Barren | 48.0996% | 53.0163% | +4.9167 |
| Vegetation | 86.4508% | 86.8759% | +0.4251 |
| Agricultural | 75.1293% | 76.8571% | +1.7278 |
| Vehicle | 76.9315% | 78.6243% | +1.6928 |

## Deliverable

ZIP: `submission_unetformer_tuned_test2.zip`; 1,300 PNG files, 1024×1024, single-channel class IDs.
SHA-256: `5b3b9d27a94b8e9095497784edc8038ca0d8d415a252606fee770fcde37902ee`.

The raw/EMA checkpoints were selected on the same holdout repeatedly. This can make the reported gain optimistic; no independent unseen validation set or competition result is available.
All trials start from the same previous checkpoint. Each recipe changes multiple factors, so these results do not isolate individual effects.
Source includes the upstream GPL-3.0 license and a documented reflection-padding compatibility fix. Weights and predictions are retained locally.
