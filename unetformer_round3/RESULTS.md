# UNetFormer round-three results

Validation: **77.8714% mIoU**, up **0.5189 percentage points** from the remeasured 77.3525% baseline.

Same 6,296/700 split; test2 excluded from training and selection. These are local validation results, not official competition scores.

## Trial comparison (512 + horizontal/vertical flips)

| Trial | Best epoch | Weights | mIoU |
|---|---:|---|---:|
| continuation | 5 | ema | 77.5007% |
| rare_images | 8 | raw | 77.6974% |
| hard_pixels | 8 | ema | 77.6183% |

Selected checkpoint: rare_images, epoch 8, raw; prior-checkpoint parameter fraction 0.

## Parameter-space averages

| Previous checkpoint fraction | mIoU |
|---|---:|
| 0.25 | 77.6507% |
| 0.5 | 77.5814% |

## Inference comparison

| Setting | mIoU |
|---|---:|
| 512_hvflip | 77.6973% |
| 480_512_544_hvflip | 77.4553% |
| 512_dihedral8 | 77.8714% |

Selected inference: `512_dihedral8`. One model is used; no checkpoint ensemble.

## Per-class changes

| Class | Previous IoU | New IoU | Change (pp) |
|---|---:|---:|---:|
| Background | 69.2282% | 69.8618% | +0.6336 |
| Building | 84.0084% | 84.0308% | +0.0224 |
| Road | 81.6633% | 82.4363% | +0.7730 |
| Water | 88.5464% | 88.7929% | +0.2465 |
| Barren | 53.0173% | 54.3975% | +1.3802 |
| Vegetation | 86.8758% | 86.9145% | +0.0387 |
| Agricultural | 76.8564% | 77.4755% | +0.6191 |
| Vehicle | 78.6245% | 79.0620% | +0.4375 |

## Deliverable

ZIP: `submission_unetformer_round3_test2.zip`; 1,300 PNG files, 1024×1024, single-channel class IDs.
SHA-256: `c6679952cc9f575d87d23cb5328e9e8db4fdc056275a740a7c33503262259fee`.

The raw/EMA checkpoints were selected on the same holdout repeatedly. This can make the reported gain optimistic; no independent unseen validation set or competition result is available.
All trials start from the same previous checkpoint. Each recipe changes multiple factors, so these results do not isolate individual effects.
Source includes the upstream GPL-3.0 license and a documented reflection-padding compatibility fix. Weights and predictions are retained locally.

