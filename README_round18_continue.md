# Round18 ordinary continuation

User requests continuing ordinary training. Initialize round17 control best,
not full-data round16. Same6296/700 split,640px,batch4accum2,MixStyle,EMA,
existing20% hard-region sampling,encoder5e-7/head5e-6;3epochs seed20261009.
Contrast loss disabled. Restart optimizer/schedule for this fine-tuning stage.
Compare identical4scale+flip+brightness validation to round17 control.
Export one checkpoint only if overall improves, barren does not decline, and
neither difficult group falls more than.2pp. No official-score guarantee.
Repeated holdout selection may overfit validation; official feedback remains needed.

Run: python outputs/UAV-QuickBaseline/round18_continue_pipeline.py
Status/logs: outputs/round18_continue. Existing run directories are rejected.
