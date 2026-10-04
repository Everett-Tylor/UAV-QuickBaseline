# SegFormer-B2 UAV semantic segmentation

This experiment trains a SegFormer-B2 segmentation model initialized from
NVIDIA's ImageNet-pretrained MiT-B2 encoder. The segmentation decoder starts
with new weights. Class 0 is ignored during training and validation; the eight
scored classes are background, building, road, water, barren, vegetation,
agricultural and vehicle.

The fixed split contains 6,296 training and 700 validation images. It is
byte-identical to the SegFormer-B0 split, and `data_audit.json` contains class
weights computed only from training masks. Test2 is used only for prediction.

## Reproduce

Use Python 3.14, CUDA PyTorch and the pinned `requirements.txt`. The local
machine used an RTX 5060 Laptop GPU with 8 GB of VRAM. Run from this directory:

```powershell
python download_pretrained.py --out pretrained/mit-b2
python prepare_cache.py --images DATA/train/images --masks DATA/train/masks --split split.json --cache CACHE512 --out prep
python train_b2.py --images DATA/train/images --masks DATA/train/masks --split split.json --source pretrained/mit-b2 --cache CACHE512 --audit prep/data_audit.json --out run --epochs 10 --batch 4 --accum 2 --workers 4
python select_predict_b2.py --run run --images DATA/train/images --masks DATA/train/masks --split split.json --test-images TEST2/images
python ../unetformer_round3/verify_export.py --zip run/submission_segformer_b2_test2.zip --report run/verification.json
```

The best pure B2 checkpoint came from epoch 10. Its original-size validation
mIoU was 76.1845% with one 512-pixel view and 76.8727% with eight geometric
views. The independently verified pure-B2 ZIP is
`run/submission_segformer_b2_test2.zip`.

An independent initialization trial uses the NVIDIA Cityscapes-pretrained
SegFormer-B2 (segmentation encoder and decoder features; final classifier
reinitialized for nine classes):

```powershell
python download_cityscapes.py --out pretrained/b2-cityscapes
python train_b2.py --images DATA/train/images --masks DATA/train/masks --split split.json --source pretrained/b2-cityscapes --cache CACHE512 --audit data_audit.json --out cityscapes_run --epochs 8 --batch 4 --accum 2 --encoder-lr 1e-5 --decoder-lr 1e-4 --workers 4
```

This trial was stopped after epoch 3 because it reached 66.3238% mIoU,
compared with 72.7634% for the ImageNet initialization at the same epoch.

The strongest validated result combines 35% B2 and 65% of the selected
UNetFormer checkpoint as an average of per-class probabilities:

```powershell
python ensemble_with_unetformer.py --b2-run run --unet-checkpoint UNETFORMER/best_blend.pth --images DATA/train/images --masks DATA/train/masks --split split.json --test-images TEST2/images --out ensemble_run --baseline 0.7793262914611663
python ../unetformer_round3/verify_export.py --zip ensemble_run/submission_b2_unetformer_ensemble_test2.zip --report ensemble_run/verification.json
```

This ensemble scored 78.5868% mIoU on the reused 700-image validation split,
up 0.6542 percentage points over the previous best. No official test score
has been measured. Repeated selection on this holdout may overstate the gain.
Exact per-class scores, training histories, download hashes and ZIP checks are
stored in `reports/`; `RESULTS.md` summarizes the comparison.

`CACHE512/images` and `CACHE512/masks` are 512-pixel PNG caches of the 6,996
labeled images. The validation script reads original masks, and the ZIP stores
1,300 original-size single-channel PNGs. The training command saves `best.pth`
and `last_training.pth` after every epoch. Add `--resume` with the same options
to continue from the last complete epoch after an interruption.

The MiT-B2 source is [nvidia/mit-b2](https://huggingface.co/nvidia/mit-b2),
pinned to revision `3bb39e8739149c3777d0325349b2a6c32c6413db`.
The Cityscapes source is
[nvidia/segformer-b2-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024),
pinned to revision `d633b2072669ca68d8f8e309de9b52bfdbf6bf72`.
`download_pretrained.py` records file hashes. The trained weights and image
data stay local; this branch contains source and validation records.
