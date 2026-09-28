# UNetFormer experiment results

Completed 28 epochs: 24 at 512 pixels, four at 768 pixels. Training and
per-epoch validation took 3,560.43 seconds after preprocessing. The best
checkpoint was epoch 23 at 512 pixels. Higher-resolution fine-tuning did not
improve validation, so the final ZIP uses that earlier checkpoint.

## Fixed 700-image validation

| Model / inference | mIoU classes 1–8 |
|---|---:|
| Prior SegFormer-B0, 512 + flip | 74.2677% |
| UNetFormer, 512 | 74.7347% |
| UNetFormer, 512 + flip (selected) | **75.4886%** |
| Same UNetFormer checkpoint, 768 + flip | 72.5061% |

The selected UNetFormer improves overall validation mIoU by **1.2209
percentage points** over B0 on the byte-identical split. This is a comparison
of final recipes, not an isolated architecture ablation: pretraining, training
duration and augmentation differ. No official competition score is available.

| Class | UNetFormer IoU | Change vs B0, percentage points |
|---|---:|---:|
| Background | 67.3432% | +1.5919 |
| Building | 83.1012% | +2.5013 |
| Road | 80.0535% | +1.5591 |
| Water | 86.8003% | -0.3242 |
| Barren | 48.0993% | -3.5354 |
| Vegetation | 86.4506% | +0.2977 |
| Agricultural | 75.1293% | +1.6631 |
| Vehicle | 76.9312% | +6.0133 |

The overall gain includes a decrease on barren land and a smaller decrease on
water. Test2 was used only for prediction. The random-image holdout can contain
similar scenes, so its gains do not establish generalization to the competition.

## Submission verification

- 1,300 same-name PNG files, 1024×1024, mode L, class IDs 0–8.
- All ZIP member CRCs, filenames, image dimensions and class IDs independently checked.
- ZIP size: 12,089,671 bytes.
- SHA-256: `24ea887b35876566b48fab5474d263efed1875ec9d75548b651cd7be8ebf8d16`.
- Single UNetFormer inference at 512 pixels with horizontal flip; no B0 ensemble.

Numerical evidence is in `reports/`: history, validation, comparison, data audit,
submission manifest and export verification. Model weights and prediction ZIP
are retained locally. Source attribution and license are included alongside code.
