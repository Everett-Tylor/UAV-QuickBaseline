# Round15: true-label barren error mining

Initialize one DINOv3 from round12 best. Mine only the fixed 6296 training images,
using 640 single-scale predictions with brightness floor .35 and cap1.25.
Compare predictions to official labels, excluding ignore0. For each image and
error type (barren false positive / false negative), choose the highest-error
256px grid cell and save a surrounding 768px context crop with original labels.
Require >=1024 total error pixels and >=512 in the selected cell.

Training dataset verifies every crop source belongs to its training split.
80% full images /20% random difficult crops in expectation. Difficult crops:
2/3 false positives,1/3 false negatives. No class-weight increase, no ClassMix,
no further test pseudo training. Existing pseudo-trained initialization retained.
3 epochs,640px,batch4,accum2,encoder5e-7,decoder5e-6,MixStyle,EMA.

Validation remains700 independent images. Compare identical4scale+flip settings
to round12. Export only if overall and barren IoU do not decline and no group
drops more than.2pp. Passing is local evidence only, not an official score.
All8 per-class metrics and confusion matrices are preserved in validation.json.

From workspace root: python outputs/UAV-QuickBaseline/round15_hard_pipeline.py
Status/logs: outputs/round15_hard. Fresh output directories required.
