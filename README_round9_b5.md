# Round 9: single SegFormer B5

The user reported an official round-8 score of **68.45**. The historical dual-model score of 69.88 is not an eligible submission under the single-model rule. This experiment switches to the user's local public SegFormer B5 ADE20K checkpoint. No round-9 score is available yet.

`b5_train.py` retains the pretrained encoder and decoder and initializes a new nine-class classifier. Labels remain 0=ignore and 1–8=evaluated classes; ADE20K's label-reduction preprocessing is not applied. Training uses the unchanged 6296/700 split, mild geometric/photometric augmentation, weighted cross entropy, Dice and Lovasz losses, BF16, gradient checkpointing, gradient accumulation and EMA. Test-2 images are used only for final inference, never training or adaptation. Final inference must use one selected checkpoint only.

Initial schedule: 8 epochs at 512 pixels, batch 2, accumulation 4, encoder learning rate 1e-5 and decoder learning rate 1e-4. Validation uses original-resolution masks. The best EMA checkpoint is selected by validation mIoU, with all epoch results retained. These settings are an experiment, not evidence of an official score improvement.

Run from the workspace root (replace local paths as needed):

```powershell
python outputs/UAV-QuickBaseline/b5_train.py --images DATA/train_images --masks DATA/train_masks --split outputs/runs/balanced_finetune/split.json --class-balance outputs/runs/balanced_finetune/class_balance.json --source D:/deepseek-harness/ds/models/segformer/segformer-b5-finetuned-ade-640-640 --out outputs/runs/round9_b5_512
```

The run writes the nine-class architecture configuration under `model_config`, `best.pth`, `last.pth`, the split, arguments and per-epoch validation history. Use this architecture configuration with `Segmenter(config_only=True)` and strictly load the selected checkpoint for inference. Weights and competition datasets are not uploaded to GitHub.

