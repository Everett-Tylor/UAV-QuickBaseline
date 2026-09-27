# B2 第二轮优化

第一轮已完成：单模型 B2，多尺度 512/640/768 与水平翻转，固定 700 张验证集 mIoU 为
74.9372%。1300 张 test2 预测已生成。该结果低于历史单模型本地最高值 78.4116%，
没有取得官方评测成绩。

## 误差依据

首次类别校正已完成：强度 0.5 在调参子集从 74.8500% 提升到 75.4192%，
复核子集从 74.9456% 提升到 75.5139%，整体从 74.9372% 提升到 75.5002%，
增加 0.5630 个百分点，仍未超过历史最好值。后续微调结果尚待完成。
`submission_b2_calibrated.zip` 是这个已验证方案的预测包；只有文件完成校验后才会出现。

原 B2 裸地的精确率/召回率为 59.78%/77.06%，农业用地为 78.50%/91.44%，
车辆为 77.56%/93.12%。稀有类交叉熵加权可能导致偏多的假阳性，这是待验证的诊断。

1. 固定原 B2 权重，测试概率乘以 `class_weight ** (-strength)`，只测试事先设定的
   `strength = 0, 0.1, 0.2, 0.3, 0.5`。类别 0 系数始终为 1。
2. 原 700 张验证集以固定种子分成 350 张调参和 350 张复核。只在调参部分选强度；
   若该候选在复核部分不提升，则回退到不校正。不会用复核结果重新搜索替代强度。
   两部分都来自此前用于模型选择的验证集，不能视为全新测试集。
3. 从第一轮最佳 B2 开始训练 4 轮，640 分辨率，batch 2、累积 4，编码器学习率
   8e-6、解码器 8e-5，使用训练类别权重的平方根、随机比例裁剪和更强颜色/gamma 增强。
4. 对新权重应用相同调参/复核流程。只有调参、复核和整体 mIoU 均超过原始 B2 的
   候选才可用于新预测包；多个合格候选选整体 mIoU 较高者。

全程仍为单模型推理；test2 不参与训练、调参或验证。没有合格候选时保留原预测，不声称提分。
即使验证提升，也必须通过官方平台评测才能确认官方分数提升。

## 运行

先按 README_segformer_b2.md 完成第一轮，保持原来的工作区目录结构：

```powershell
python work/segformer-b2/b2_calibrate.py --checkpoint outputs/segformer_b2/train512/best.pth --source outputs/segformer_b2/train512/model_config --images work/data/train/images --masks work/data/train/masks --split work/segformer-b2/reports/round2/split.json --class-balance outputs/segformer_b2/class_balance.json --out outputs/segformer_b2_optimized/calibration_initial.json
python work/segformer-b2/b2_optimize_pipeline.py --workspace .
```

若源码位于其他目录，替换命令中的脚本目录；优化脚本相对自身定位其依赖。
状态和所有候选指标保存在 `outputs/segformer_b2_optimized`。
成功完成后生成 `submission_b2_optimized.zip`，并校验 1300 张 1024×1024 单通道 PNG。
当前实验进度以 `status.json` 为准，源码发布不表示优化实验已经完成。

测试：`python -m unittest test_b2 test_b2_calibrate -v`。
