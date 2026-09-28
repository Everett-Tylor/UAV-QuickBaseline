# UNetFormer round 3

Continue from the previous selected UNetFormer EMA checkpoint. Its previous
700-image validation mIoU was 77.3524% with 512-pixel horizontal/vertical flips.
Keep the byte-identical 6,296/700 training/validation split. Test2 is excluded
from all training, sampling, checkpoint selection and inference selection.

## Training comparisons

Three independent trials start from the same checkpoint, with encoder LR 3e-6,
decoder LR 1.5e-5, frozen BatchNorm running statistics, trainable affine parameters,
AdamW, warmup/polynomial decay, FP16, gradient clipping, EMA decay 0.995 and
effective batch size eight. Sizes cycle through 512,448,576 pixels.
Training class weights use the normalized square root of the original class
audit's weights, matching the prior winning recipe. Color factors are 0.9–1.1.
Shuffle and augmentation seeds change deterministically each epoch.

| Trial | Epochs | Change |
|---|---:|---|
| continuation | 6 | Standard shuffled training |
| rare_images | 8 | Moderately oversample training images containing barren/vehicle pixels |
| hard_pixels | 8 | Blend ordinary CE+Dice with CE on the highest-loss half of valid pixels |

Rare-image sampling weights are `1 + 1.5*(barren_fraction>0.01) +
0.5*(vehicle_fraction>0.005)`, computed only from cached training masks.
Sampling is with replacement and preserves the number of samples per epoch.
Hard-pixel loss is 0.75 times the previous weighted CE+0.5 Dice loss plus 0.25
times weighted CE averaged over the hardest half of valid pixels. Both main
and auxiliary heads use it, with auxiliary weight 0.4. Ignore label 0 is excluded.

Raw and EMA weights are evaluated every epoch using 512+four flip combinations.
The previous checkpoint remains the baseline fallback. The best new checkpoint
is then compared with two parameter-space blends (25% and 50% previous weights).
Each blend produces one ordinary model; it does not run two models at inference.

## Inference

Three predefined settings are compared on the selected checkpoint:

- 512 with four horizontal/vertical flip combinations.
- 480,512,544 with the same four flips, averaged logits.
- 512 with eight dihedral transforms: four flips, with/without transpose.

Transforms are inverted on logits before averaging. Outputs are resized to
original dimensions before argmax. Original labels are used for evaluation.
This repeated validation selection is not an independent test; improvements
do not establish an official competition gain. All attempted trials are recorded.

## Run

Use the pinned `requirements.txt` and the previous 768-pixel cache, initial
checkpoint, and original training-only class audit.

```powershell
python round3.py --images DATA/train/images --masks DATA/train/masks --cache CACHE768 --split split.json --init PREVIOUS_RUN/best.pth --audit ORIGINAL_RUN/data_audit.json --test-images TEST2/images --out run
python predict_round3.py --run run --images TEST2/images --out export
python verify_export.py --zip export/submission_unetformer_round3_test2.zip --report export/verification.json
```

Fresh output directories are required. `last_training.pth` records training
state for recovery work; the CLI does not automatically resume. Model weights
and images stay local. The 1,300 PNG files in the final ZIP must match test2
filenames, original 1024×1024 dimensions, mode L and class IDs 0–8.

UNetFormer model code is from WangLibo1995/GeoSeg, revision
`9453fe48209c4626b29e35e61bab93b61212c4b1`, under the included GPL-3.0 license.
The prior four-element reflection-padding compatibility fix is retained.
`test_round3.py` checks real GPU training, ignored-pixel gradients, the all-ignore
case and all inference configurations in the original experiment workspace.
FP16/cuDNN benchmarking may change a small number of boundary pixels between runs.
