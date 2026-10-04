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
```

`train_b2.py` saves a full optimizer checkpoint after each epoch. Add
`--resume` to the same continuation command to restart after a completed
epoch. The validation script compares single-view, flip, eight-view and
640-pixel inference, then exports 1,300 original-size indexed PNGs in one
ZIP. It reports validation mIoU separately from any official test score.

The encoder initialization is [NVIDIA MiT-B2](https://huggingface.co/nvidia/mit-b2)
at revision `3bb39e8739149c3777d0325349b2a6c32c6413db`. Trained weights
and image data remain local.
