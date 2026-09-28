# Round16 full labeled training

User reports round15 official68.75 and authorizes full labeled training.
Initialize round15 best, combine original6296+700 into6996 unique labeled images.
Reuse round15 hard crops and80%full/20%hard sampling policy; no new mining on
former validation and no new test pseudo labels. Three fixed epochs,640px,
batch4 accumulation2,encoder5e-7/head5e-6,MixStyle,EMA; same as round15 except
initialization/full data/seed. Final epoch EMA checkpoint is exported as final.pth.
No validation metrics, early stopping, or best-checkpoint selection in full mode.
Original700 are now training data and cannot support an independent gain claim.
Predict single checkpoint with512/640/768/896 plusflip and established brightness
settings. Validate1300 PNG names, dimensions, class IDs andZIP integrity.

Run: python outputs/UAV-QuickBaseline/round16_full_pipeline.py
Status: outputs/round16_full/status.json. Fresh run directories required.
