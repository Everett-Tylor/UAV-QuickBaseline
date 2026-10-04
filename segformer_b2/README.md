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
python train_b2.py --images DATA/train/images --masks DATA/train/masks --split split.json --source pretrained/mit-b2 --cache CACHE512 --audit data_audit.json --out run --epochs 10 --batch 4 --accum 2 --workers 4
python select_predict_b2.py --run run --images DATA/train/images --masks DATA/train/masks --split split.json --test-images TEST2/images
python ../unetformer_round3/verify_export.py --zip run/submission_segformer_b2_test2.zip --report run/verification.json
```

`CACHE512/images` and `CACHE512/masks` are 512-pixel PNG caches of the 6,996
labeled images. The validation script reads original masks, and the ZIP stores
1,300 original-size single-channel PNGs. The training command saves `best.pth`
and `last_training.pth` after every epoch. Add `--resume` with the same options
to continue from the last complete epoch after an interruption.

The MiT-B2 source is [nvidia/mit-b2](https://huggingface.co/nvidia/mit-b2),
pinned to revision `3bb39e8739149c3777d0325349b2a6c32c6413db`.
`download_pretrained.py` records file hashes. The trained weights and image
data stay local; this branch contains source and validation records.
