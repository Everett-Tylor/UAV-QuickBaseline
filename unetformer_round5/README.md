# UNetFormer round 5

This run tested whether continuing the round-4 checkpoint or combining the
round-3 and round-4 checkpoints would improve the fixed 700-image validation
split. The split was not changed. Test2 was used only for final prediction.

## Results

| Method | 512-pixel dihedral-8 validation mIoU |
|---|---:|
| Round-4 checkpoint | 77.9312249% |
| Weight average: 25% round 3, 75% round 4 | 77.9326291% |
| Logit ensemble: 25% round 3, 75% round 4 | 77.9313686% |

The weight average is the selected local submission. Its gain over round 4 is
only **0.001404 percentage points**. This is too small to establish a reliable
improvement on the official test set. No official score is available. The
same validation split has been reused for several rounds of selection.

Eight additional training epochs used alternating 512/576-pixel images, a
1e-6 backbone learning rate, moderate rare-image sampling, EMA and the
previous loss. Training was interrupted after epoch 5 and resumed from the
saved model, optimizer and scaler states with `resume_round5.py`. No continued
checkpoint exceeded the round-4 four-flip baseline of 77.7361%; the highest
continued result was 77.6617% at epoch 3 EMA. This trial was rejected.

## Reproduce

Install `../unetformer_round3/requirements.txt`. This branch includes the
UNetFormer model code, split and validation helpers. The
training images, original data audit, earlier checkpoints and new predictions
remain local.

```powershell
python round5.py --images DATA/train/images --masks DATA/train/masks --cache CACHE768 --split ../unetformer_round3/split.json --init ROUND4/best.pth --audit ORIGINAL/data_audit.json --test-images TEST2/images --out run --prior-selection ROUND4/selection.json
python blend_checkpoints.py --older ROUND3/best.pth --newer ROUND4/best.pth --images DATA/train/images --masks DATA/train/masks --split ../unetformer_round3/split.json --out blends34
python predict_blend.py --checkpoint blends34/best_blend.pth --images TEST2/images --out blend_submission
python ../unetformer_round3/verify_export.py --zip blend_submission/submission_unetformer_round5_test2.zip --report blend_submission/verification.json
```

`resume_round5.py --out run` resumes only when `run` has a saved epoch-boundary
`last_training.pth`. The prediction ZIP contains 1,300 original-size,
single-channel PNGs named `test2_1.png` through `test2_1300.png`, with class
values 0-8. The round-3 `UNetFormer.py` retains its upstream GPL-3.0 license.
