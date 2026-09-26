# 复赛第八轮：单模型 MixStyle

用户反馈第七轮 DINOv3 单模型官方成绩为 67.70 分，低于早前不符合规则的双模型融合 69.88 分。官方测试图中亮度低于 0.35 的比例为 52.0%，训练/验证集分别为 42.8%/42.4%；这表明图像风格分布有差异，但不能单凭统计证明退分的具体原因。

本轮从第七轮 DINOv3 单模型最佳权重继续训练，在第 3、6 个编码器块的投影特征上加入 MixStyle。训练时以概率 0.5 混合同一批图像的通道均值和标准差，Beta 分布参数为 0.1。验证和推理阶段关闭 MixStyle，因此最终仍只有一个 DINOv3 模型权重。该方法参考 [Zhou 等人的 MixStyle 论文](https://arxiv.org/abs/2104.02008)。

固定划分为 6296 张官方训练图、700 张验证图。训练 3 轮，640 输入、batch 8、编码器学习率 2e-6、解码器学习率 3e-5、EMA，最佳权重来自第 3 轮。没有用复赛测试图训练、制作伪标签或更新归一化参数。

```powershell
python dino_train.py --images data/train_images --masks data/train_masks --split reports/round2/split.json --class-balance reports/round2/class_balance.json --source model_configs/dinov3 --init runs/round7_dino768/best.pth --out runs/round8_mixstyle640 --epochs 3 --freeze-epochs 0 --size 640 --batch 8 --accum 1 --workers 4 --lr 0.000002 --head-lr 0.00003 --seed 20260930 --mixstyle
```

验证结果均为同一权重的多尺度、水平翻转推理，类别 1–8 的 mIoU：

| 方案 | 全部 | 暗光 | 低对比度 |
| --- | ---: | ---: | ---: |
| 第七轮单模型，768/896/1024/1152 | 78.2101% | 71.9170% | 74.4135% |
| 本轮单模型，512/640/768/896 | 78.3619% | 71.8271% | 74.4475% |
| 本轮再对均值低于 0.35 的输入做温和亮度校正 | **78.3839%** | 71.8286% | **74.5959%** |

输入亮度校正上限为原亮度的 1.25 倍；将目标提高到 0.40 反而使总分和暗光组下降，所以未采用。亮度校正只作用于推理输入，不改变权重或使用其他模型。

```powershell
python dino_predict.py --checkpoint runs/round8_mixstyle640/best.pth --source model_configs/dinov3 --images data/test2_images --sizes 512 640 768 896 --hflip --brightness-floor 0.35 --max-brightening 1.25 --out submission_round8_mixstyle_single
```

本地验证仅比旧双模型融合的 78.3380% 高 0.0459 个百分点，且暗光组比第七轮稍低。官方是否达到 70 分仍须提交后确认。本分支不包含模型权重或预测 ZIP；记录见 `reports/round8/`。
