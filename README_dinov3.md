# 第六轮主实验：DINOv3 ViT-B/16 分割

第五轮官方成绩为 67.8 分（用户回报），本轮以 70 分为目标；官方是否达到目标必须由竞赛评测确认。

## 本地预训练权重

用户提供 `facebook/dinov3-vitb16-pretrain-lvd1689m` 的本地模型目录。权重 SHA256：`9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`，342662192 字节。代码只从本地加载，不执行远程仓库代码。原模型遵循 DINOv3 许可，使用者需自行取得合法权重并遵守原许可。

## 模型与训练

DINOv3 编码器含 12 个 Transformer 块、768 隐藏维度、16 像素 patch 和 4 个 register tokens。提取第 3/6/9/12 个块的特征并应用编码器 LayerNorm，移除 CLS 和 register tokens，构建 128 通道多尺度特征金字塔。解码器使用 GroupNorm、逐级融合，并融合一个从原图提取的 1/4 分辨率细节分支，预测 9 类。

先冻结编码器训练新解码器 1 轮，再联合微调。第一阶段共 8 轮、512 输入、batch 8、有效 batch 8、编码器初始学习率 1e-5、解码器 3e-4、AdamW weight decay 0.01、warmup + polynomial decay、EMA。BF16 和梯度检查点降低显存使用。固定 seed=20260927。

训练/验证仍为原 6296/700 划分。损失为加权 CE + 0.5 Dice + 0.3 Lovasz-Softmax，忽略标签 0，验证计算类别 1-8。使用温和颜色增强和几何翻转/旋转，测试集未用于训练或伪标签。

## 复现

使用 requirements_round2.txt 对应环境和用户获准使用的外部预训练权重。

```powershell
python dino_train.py --images data/train_images --masks data/train_masks --split reports/round2/split.json --class-balance reports/round2/class_balance.json --source pretrained/dinov3-vitb16-pretrain-lvd1689m --out runs/round6_dino512 --epochs 8 --freeze-epochs 1 --size 512 --batch 8 --accum 1 --workers 4
```

`--freeze-epochs 0 --smoke` 可验证全模型两步训练；本机 batch 8 检查已通过，日志显示峰值分配约 3.18GB。实际长期训练峰值以运行日志为准。

推理使用 dino_predict.py。`--source` 只需要包含配置的目录，模型权重由 `--checkpoint` 指定，最终尺度和是否融合以 reports/round6/selection.json 为准。若与 SegFormer 融合，需显式设置 `--ensemble-architecture segformer --ensemble-source round2_model --ensemble-checkpoint ...`。

## 未采用的难样本采样候选

在用户提供 DINOv3 权重前，已完成 4 轮 SegFormer 难组采样训练。新模型多尺度 mIoU 为 75.3419%，与旧模型等权融合为 75.5288%，均未超过第五轮的 75.5864%，因此没有直接用于提交。具体代码和方法见 README_round6.md，报告保留作对照。

本地固定验证集经过多轮模型选择，可能偏乐观；不得把本地结果当作官方分数。本分支提供源码和实验记录，新权重保存在本机，预测 ZIP 单独交付。

## 640 分辨率精修

512 阶段 8 轮验证仍持续提高，因此追加 640 分辨率监督精修：从其最佳 EMA 权重初始化，4 轮、batch 8、累积 1、编码器学习率 3e-6、解码器 5e-5，不再冻结编码器，seed=20260928。其他损失与增强不变。使用同一固定验证集选择最佳权重，不能预先视为改进。

```powershell
python dino_train.py --images data/train_images --masks data/train_masks --split reports/round2/split.json --class-balance reports/round2/class_balance.json --source model_configs/dinov3 --init runs/round6_dino512/best.pth --out runs/round6_dino640 --epochs 4 --freeze-epochs 0 --size 640 --batch 8 --accum 1 --workers 4 --lr 0.000003 --head-lr 0.00005 --seed 20260928
```

## 最终验证与提交

固定 700 张验证集，原始分辨率统计类别 1-8 的 IoU，忽略真实标签 0。所有下表推理均使用 512/768/896 三尺度与水平翻转。亮度、对比度分组阈值来自训练集第 25 百分位；分别含 164 和 156 张验证图，可重叠。

|模型|mIoU %|暗光 %|低对比度 %|
|---|---:|---:|---:|
|第五轮 SegFormer B1|75.5864|68.8724|71.9203|
|DINOv3 512|77.1158|70.7637|73.4964|
|DINOv3 512 + B1|78.0524|71.5013|74.1758|
|DINOv3 640 精修|77.9913|71.5729|74.5861|
|DINOv3 640 精修 + B1|78.3380|71.8316|74.7114|

最终选择 `dino640_b1_ensemble.json`，融合时对各模型、尺度与翻转视图的原尺寸 softmax 概率等权求和，再取 argmax。没有使用 MixStyle。640 精修的最佳单尺度权重来自第 2 轮（77.7200%）。

最终预测命令：

```powershell
python dino_predict.py --checkpoint runs/round6_dino640/best.pth --source model_configs/dinov3 --images data/test2_images --out submission_round6 --sizes 512 768 896 --hflip --ensemble-checkpoint runs/round5_mild768/best.pth --ensemble-source round2_model --ensemble-architecture segformer
```

ZIP 含 1300 张原尺寸单通道 PNG，根目录直接存放预测图，文件名与测试输入对应，类别值 0-8。已校验尺寸、文件名、像素类别和 ZIP CRC。最终模型与 ZIP SHA256 见 `reports/round6/selection.json`。

本轮官方评分尚未取得。上一轮官方 67.8 分来自用户反馈，本地验证提升不能保证官方达到 70。
