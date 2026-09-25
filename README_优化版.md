# UAV 分割优化版

`improvedseg.py` 在保留 `quickseg.py` 原版的基础上增加裁剪训练、CE + Dice 损失、学习率预热与余弦衰减、原分辨率滑窗验证/预测，以及可选的多尺度空洞卷积上下文模块。不使用外部数据、外部预训练权重或模型集成。

这些改动需要用固定验证集实测。损失下降不等于 mIoU 提高，新的模型结构也不保证比微调旧模型更好。

## 安装与测试

先按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装适配显卡的 PyTorch，再执行：

```powershell
python -m pip install -r requirements.txt
python -m unittest test_improvedseg -v
```

## 优先复用已有训练结果

如果已有 `runs/quick/best.pth` 和 `runs/quick/split.json`，可先进行短周期微调。必须复用旧模型的训练/验证划分，避免把已训练过的图像放进验证集。

```powershell
python improvedseg.py train --images data/train_images --masks data/train_masks --init-weights runs/quick/best.pth --split runs/quick/split.json --architecture unet --sampling resize --class-balance sqrt --epochs 2 --warmup-epochs 0 --crops-per-image 1 --size 384 --batch 4 --lr 0.0003 --output runs/balanced_finetune
```

`--init-weights` 仅用于加载自己用官方训练集得到的权重，不是完整断点续训：优化器与学习率计划会重新开始。新版权重记录划分，可检查训练集一致性；旧版权重没有划分元数据，调用者必须提供其原始 `split.json`。不同运行请用不同输出目录，防止覆盖已有结果。

`--sampling resize` 保持旧模型的整图缩放尺度，避免短周期微调时突然改变目标大小。`--class-balance sqrt` 仅统计训练划分内的像素频率，对 CE 使用逆平方根权重并裁剪至 0.25–4；0 类权重为零。统计会扫描训练标签，结果写入 `class_balance.json`。权重不使用验证或测试标签。

## 从零训练上下文模型

```powershell
python improvedseg.py train --images data/train_images --masks data/train_masks --split runs/quick/split.json --architecture context --epochs 60 --size 384 --batch 4 --output runs/context
```

- 原图按 0.75–1.5 倍随机缩放再裁剪，保留局部细节；不把整张图固定压缩成 384×384。
- 图像和标签同步水平/垂直翻转及 90° 旋转，RGB 图像额外做轻度亮度、对比度和色彩变化。
- CE + 0.5×Dice；Dice 仅计算当前批次有真值的实际类别，0 类像素对两个损失均无贡献。全忽略批次跳过更新。
- 可选 `context` 在 U-Net 的 1/8 分辨率瓶颈加入膨胀率 1/2/4 的深度卷积分支。`unet` 与旧版权重结构一致。
- 训练使用 CUDA 混合精度与梯度裁剪；验证和推理使用 FP32。显存不足先降 `--batch 2`。未指定 GPU 时自动回退 CPU，正式训练优先使用 GPU。
- 每张图每轮默认采样两次；短周期微调示例为一次。60 轮为可调整的训练配置，不是已证明的最优轮数。

## 公平比较 mIoU

原版训练日志在缩小后的标签上评价。优化版在原尺寸标签上评价，二者不能直接相减。用下面两条命令在**同一 700 张或你自己固定的验证集**上重新测量：

```powershell
python improvedseg.py eval --images data/train_images --masks data/train_masks --weights runs/quick/best.pth --split runs/quick/split.json --report runs/baseline_native.json
python improvedseg.py eval --images data/train_images --masks data/train_masks --weights runs/balanced_finetune/best.pth --split runs/quick/split.json --report runs/balanced_native.json
```

旧权重使用与旧版推理一致的 PIL 缩图、logits 上采样；新版权重自动使用训练时记录的推理方式：整图训练对应整图预测，裁剪训练对应滑窗预测。滑窗重叠区域按有正边缘权重的 Hann 窗融合 logits；不进行 TTA 或模型集成。`eval` 和 `infer` 可用 `--mode resize` 或 `--mode tile` 做对照；若指定了覆盖参数，生成正式预测时也应使用验证选定的同一设置。

报告包含 1–8 类 IoU、混淆矩阵、非零并集类别的 mIoU，以及缺失类按零处理的固定八类均值。官方如何处理缺失类应以正式规则为准。选择 `best.pth` 的指标为 `mIoU_present_nonignored`。对验证集无真值、但模型产生误报的类别，IoU 按零计入。

每个运行保存 `config.json`、`split.json`、`history.jsonl`、`last.pth` 和 `best.pth`。如果图像来自同一场景的相邻切片，应事先提供按场景分开的 JSON：`{"train": ["0000.png"], "val": ["0001.png"]}`，包含全部图像且两部分互斥。随机划分只能用于当前基线的可比实验，不能消除同源场景泄漏。

可做消融：`--architecture unet` 关闭上下文模块，`--dice-weight 0` 只使用 CE；必须固定划分、训练预算和随机种子再比较。

## 预测和打包

```powershell
python improvedseg.py infer --images data/test1_images --weights runs/balanced_finetune/best.pth --output predictions_detail
python improvedseg.py pack --images data/test1_images --predictions predictions_detail --zip submission_detail.zip
```

正式结果为 1024×1024、单通道 PNG、类别 ID 0–8，直接复用原版打包校验。使用新的预测目录，避免混入旧文件。官方图像和标签不包含在本仓库；本次按仓库所有者要求发布的权重与预测 ZIP 位于 artifacts/。

## 参考

Dice 目标参考 [V-Net 原始论文](https://arxiv.org/abs/1606.04797)。这里采用多类二维实现，并非复现其网络或宣称其医学数据集结果可以直接迁移到本赛题。

## 已训练产物

固定 700 张验证图上 mIoU：42.47% → 50.99%（+8.52 个百分点）。这不是竞赛平台分数。

- [最佳模型](artifacts/best_optimized.pth)
- [500 张预测的提交 ZIP](artifacts/submission_optimized.zip)
- [优化源码快照](artifacts/UAV-QuickBaseline_optimized_source.zip)
- [完整实验报告](reports/optimization_results.md)
- [文件 SHA-256 校验清单](artifacts/manifest.json)

下载最佳模型后，可用 `python improvedseg.py infer --images data/test1_images --weights artifacts/best_optimized.pth --output predictions_detail` 预测。
