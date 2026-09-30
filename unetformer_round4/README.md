# UNetFormer continuation (round 4)

This run continues the selected round-3 UNetFormer checkpoint on the same
6,296-image training split. The fixed 700-image validation split is excluded
from training and sampling. Test2 is used only for final prediction.

The continuation uses eight epochs, AdamW, backbone learning rate 1.5e-6,
decoder learning rate 7.5e-6, effective batch size eight, FP16, gradient
clipping, EMA, frozen BatchNorm running statistics, rare-image sampling and
448/512-pixel alternating training. `continue_training.py` records the recipe,
every raw and EMA validation score, and the last optimizer state. It compares
the best checkpoint with the previous model using identical 512-pixel
dihedral-eight inference on the original 1024-pixel masks.

## Local validation

| Model | mIoU |
|---|---:|
| Previous selected checkpoint | 77.8714% |
| Round-4 checkpoint, epoch 7 raw | 77.9312% |

The gain is **0.0598 percentage points** on the reused holdout. Reusing this
holdout for model selection can overestimate the gain; no official test score
has been measured.

## Reproduce

Install the pinned `../unetformer_round3/requirements.txt`, then run from this
directory. The round-3 source is available alongside this directory:

```powershell
python continue_training.py --images DATA/train/images --masks DATA/train/masks --cache CACHE768 --split ../unetformer_round3/split.json --init PREVIOUS_RUN/best.pth --audit ORIGINAL_RUN/data_audit.json --test-images TEST2/images --prior-selection PREVIOUS_RUN/selection.json --out run
python predict_round4.py --run run --images TEST2/images --out reproduced
python ../unetformer_round3/verify_export.py --zip reproduced/submission_unetformer_round4_test2.zip --report reproduced/verification.json
```

The prior checkpoint, data, test images and generated weights are local assets;
they are not stored in GitHub. `../unetformer_round3/UNetFormer.py` is from
[WangLibo1995/GeoSeg](https://github.com/WangLibo1995/GeoSeg) at commit
`9453fe48209c4626b29e35e61bab93b61212c4b1`, under the included GPL-3.0
license. The 1,300 ZIP members are `test2_1.png` through `test2_1300.png`,
1024 �� 1024 single-channel PNGs with class values 0�C8.

