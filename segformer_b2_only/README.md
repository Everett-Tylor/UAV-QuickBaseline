# SegFormer-B2-only continuation for UAV test2

This run uses **only SegFormer-B2** for training, validation and test2
prediction. It starts from the previous pure B2 epoch-10 checkpoint, which
scored 0.76872731 mIoU with eight geometric views on the fixed 700-image
validation split. No other model weights or predictions enter this run.

The fixed split has 6,296 train and 700 validation images. Class ID 0 is
ignored; IDs 1�C8 are scored. Validation masks are read at their original
resolution. Test2 is used only to generate predictions.

## Reproduce

Use CUDA PyTorch and the versions in `requirements.txt`. From this directory:

```powershell
python download_pretrained.py --out pretrained/mit-b2
python prepare_cache.py --images DATA/train/images --masks DATA/train/masks --split split.json --cache CACHE512 --out prep
python train_b2.py --images DATA/train/images --masks DATA/train/masks --split split.json --source pretrained/mit-b2 --cache CACHE512 --audit prep/data_audit.json --out previous_run --epochs 10 --batch 4 --accum 2 --workers 4
python train_b2.py --images DATA/train/images --masks DATA/train/masks --split split.json --source pretrained/mit-b2 --cache CACHE512 --audit prep/data_audit.json --out continuation_run --init-checkpoint previous_run/best.pth --epochs 5 --batch 4 --accum 2 --workers 4 --encoder-lr 1e-5 --decoder-lr 1e-4
python select_predict_b2.py --run continuation_run --images DATA/train/images --masks DATA/train/masks --split split.json --test-images TEST2/images
python verify_export.py --zip continuation_run/submission_segformer_b2_test2.zip --report continuation_run/verification.json
```

`train_b2.py` saves a full optimizer checkpoint after each epoch. Add
`--resume` to the same continuation command to restart after a completed
epoch. The validation script compares single-view, flip, eight-view and
640-pixel inference, then exports 1,300 original-size indexed PNGs in one
ZIP. It reports validation mIoU separately from any official test score.

The encoder initialization is [NVIDIA MiT-B2](https://huggingface.co/nvidia/mit-b2)
at revision `3bb39e8739149c3777d0325349b2a6c32c6413db`. Trained weights
and image data remain local.

## Result

The best continuation checkpoint was epoch 4 of 5. On the fixed 700-image
validation set, single-view mIoU increased from 0.76184455 to 0.76957040.
Eight-view mIoU increased from **0.76872731 to 0.77630380**, a gain of
0.00757649 (0.7576 percentage points). These are local validation results,
not an official competition score.

The test2 ZIP contains exactly 1,300 grayscale 1024x1024 PNG masks, with
class IDs 1�C8. Independent verification passed all filename, PNG and ZIP
checks. ZIP SHA-256:
`b9786b45fff5a82f7cceebc931507bf1660ebf28f044ac5af6a775e1a3e43974`.
