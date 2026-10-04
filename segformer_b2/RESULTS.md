# SegFormer-B2 training and test2 predictions

The fixed split contains 6,296 training and 700 validation images. Class 0 is
ignored; mIoU averages classes 1-8. Test2 was excluded from training and model
selection. The previous best local validation score was 77.9326% for the
selected UNetFormer weight average.

| Model and inference | Validation mIoU |
|---|---:|
| Previous UNetFormer | 77.9326% |
| SegFormer-B2, ImageNet initialization, epoch 10, 512 single view | 76.1845% |
| SegFormer-B2, same checkpoint, 512 dihedral-8 views | 76.8727% |
| UNetFormer 65% + SegFormer-B2 35%, averaged probabilities | **78.5868%** |

The ensemble improved local validation mIoU by **0.6542 percentage points**
against the previous best. All eight per-class IoUs increased: background
+0.964, building +0.554, road +0.416, water +0.476, barren +1.591,
vegetation +0.426, agricultural +0.359 and vehicle +0.448 percentage points.
The four tested B2 fractions were 0.10, 0.20, 0.35 and 0.50; 0.35 was best.

An independent Cityscapes-initialized B2 trial was stopped after epoch 3.
It scored 66.3238% at that point, versus 72.7634% for the ImageNet-initialized
B2 at epoch 3. Its checkpoint and full training state remain local.

The selected submission is
`ensemble_run/submission_b2_unetformer_ensemble_test2.zip`: 1,300 PNGs,
1024 x 1024, mode L, class IDs 0-8. Independent verification passed file
names, sizes, class IDs and every ZIP member CRC. Size: **10,879,280 bytes**.
SHA-256: `7804419e995f10f2c540ea1e867491d40e3c12c6ab76bf456d45145fa2b9a35c`.

The pure-B2 submission is `run/submission_segformer_b2_test2.zip` (also
1,300 verified PNGs). Size: **11,323,536 bytes**. SHA-256:
`1c584285296c6795f625da9ac95cca9895ec966545fa3a59d02e37029da85e7c`.

These scores come from a validation split reused in prior tuning rounds, so
the measured gain may overstate the benefit on the official test set. No
official competition score has been measured. Weights, source images and
prediction ZIPs remain local; GitHub contains the source and this record.
