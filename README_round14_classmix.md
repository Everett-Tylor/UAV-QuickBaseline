# Round 14: ClassMix with true barren emphasis

Offline DACS-inspired experiment, not a reproduction of online EMA-teacher DACS.
One DINOv3 checkpoint at inference. Official labeled data and explicitly permitted
test_2 pseudo labels only; fixed 6296/700 split remains unchanged.

Initialize from round8; reuse hash-checked round12 confidence .95/agreement .75
pseudo labels. Mix random half of nonignored source classes into target images,
always including true barren (ID 5) when present. Ignore ID 0 is never pasted.
Apply brightness/contrast and optional blur to mixed RGB, preserve class IDs.
Training images with >=1% true barren pixels receive 2x sampling weight, with
replacement for the same number of samples per epoch. Validation is not sampled.
Mixed CE weight ramps to .5 over one epoch. Supervised CE/Dice/Lovasz retained.
Three epochs,640px,source batch4,target2,accum2,encoder1e-6,head1e-5.

This combines mixing and barren emphasis per user request, so any gain cannot
be attributed to either separately. Track all class IoUs, overall and dark/low
contrast groups against round12; unchanged TTA512/640/768/896 + flip. Automatic
export gate is only stability (overall -0.3pp, groups -0.5pp vs round8), not proof
of improvement. Official score is pending and no 70+ claim is made.

Run from workspace root with the existing environment:
`python outputs/UAV-QuickBaseline/round14_classmix_pipeline.py --workspace .`
Logs/status: outputs/round14_classmix_barren. Existing run directories are rejected.
Reference: https://openaccess.thecvf.com/content/WACV2021/html/Tranheden_DACS_Domain_Adaptation_via_Cross-Domain_Mixed_Sampling_WACV_2021_paper.html
