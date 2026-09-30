# Round 4 result

Starting checkpoint: round 3 selected UNetFormer raw weights. Fixed split:
6,296 train images and 700 validation images. Training: eight epochs, 448/512
alternating sizes, rare-image sampling, lower learning rate, EMA monitored.

The best checkpoint is epoch 7 raw. Four-flip validation mIoU rose from
77.6973% to 77.7361%. The final comparison uses the same full-resolution
512-pixel dihedral-eight inference for both models:

| Checkpoint | Validation mIoU |
|---|---:|
| Round 3 | 77.8714% |
| Round 4 | 77.9312% |

Delta: +0.0598 percentage points. The official competition test score is
unknown. The validation set was reused from earlier rounds, so this small
observed gain might not carry over to the official test.

The local submission ZIP is `run/submission_unetformer_round4_test2.zip`
(1,300 PNGs; 11,163,138 bytes; SHA-256
`5d0a704f66c09c2403e149a075d42238b2cd19891cb587059f356b4ab60b7298`).
The model checkpoint, images and ZIP are kept local; GitHub contains the
reproducible source and this result summary.

