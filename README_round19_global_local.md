# Round19 global-context/native-detail single model

One shared DINOv3 backbone/decoder processes global640 and four native512 tiles
from1024 input. Stitch local160-channel features spatially; concatenate aligned
global features; learned320->64->9 residual head corrects global logits. Last
layer zero initialized, so initial output exactly equals resized global logits.
Not a reproduction of HRDA; no independent models or checkpoint ensemble.

Initialize round17 control best; fixed6296/700 split. Train3epochs batch1accum8,
input1024,encoder5e-7,oldhead5e-6,newfusion5e-5. Full images, no cached crops,
no MixStyle (batch1); these recipe changes mean this is an end-to-end candidate
comparison, not an isolated architecture ablation. GPU smoke required first.

Validation candidate uses native1024 global/local withflip and established
brightness correction. Compare against round17 four-scale flip reference:
inference strategies differ deliberately. Export only if overall improves,
barren does not decline and hard groups drop no more than.2pp. Record all-class
metrics, no official-gain claim. Test-set prediction uses identical candidate mode.

Run: python outputs/UAV-QuickBaseline/round19_global_local_pipeline.py
Logs/status: outputs/round19_global_local. Fresh run directories required.
