# Round 5 validation and submission

The fixed 700-image validation set gives these 512-pixel dihedral-eight scores:

| Candidate | mIoU |
|---|---:|
| Round-4 selected checkpoint | 77.9312249% |
| Round-3/4 parameter average (25%/75%) | 77.9326291% |
| Round-3/4 logit ensemble (25%/75%) | 77.9313686% |

The parameter average is the highest local result, a nominal gain of
**0.001404 percentage points** over round 4. The other parameter fractions
(50% and 75% round 3) scored 77.9127558% and 77.8907641%, respectively.
The eight-epoch continuation from round 4 did not exceed its 77.7360631%
four-flip baseline; its best four-flip score was 77.6617242%.

The selected ZIP is `blend_submission/submission_unetformer_round5_test2.zip`:
1,300 PNGs, each 1024 x 1024, mode L, class values 0-8. Independent ZIP
verification passed all member CRCs. Size: **11,169,157 bytes**. SHA-256:
`993690c5d0bbc64d73d04f3a668c225565a7988f0710c89870372ef1b5931e5d`.

This is a very small holdout difference after repeated selection on the same
validation images. It does not establish that the official test score will
increase. No official score has been measured. Model weights, data and the
prediction ZIP remain local; this branch stores source and result records.
