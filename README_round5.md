# 第五轮：更高分辨率监督微调

第四轮提交官方 mIoU 为 67.6 分（用户于 2026-09-25 回报），此前第三轮为 67 分。本轮官方成绩待提交确认。

## 训练方法

从第四轮最佳精修权重继续训练，输入从 640 提高到 768。沿用 SegFormer B1、加权 CE + 0.5 Dice + 0.3 Lovasz-Softmax、EMA、温和颜色增强及几何翻转。4 轮，batch 4，梯度累积 2，有效 batch 8；编码器学习率 5e-6，解码头 10 倍，AdamW 权重衰减 0.02，seed=20260925。

保持固定 6296 张训练 / 700 张验证划分，标签 0 忽略、计算类别 1-8 的 mIoU。复赛测试集只用于最终预测，没有参与训练或参数选择。提高分辨率的目的在于保留细小结构，实际收益依据验证比较决定。

初始化 SHA256：`dbc968a8d3528d2bc02af843c64468a361d751ea717b659458429215b38984fa`。它继承前几轮已获用户允许的外部 B1 预训练编码器。

## 复现

使用 requirements_round2.txt 环境，先按第四轮说明得到初始化权重。训练器仍为 round4_train.py，新增 round5_train.py 固定本轮默认设置；二者调用相同训练逻辑。

```powershell
python round5_train.py --data-root data --source round2_model --init runs/round4_mild640/best.pth --split reports/round2/split.json --class-balance reports/round2/class_balance.json --out runs/round5_mild768
```

可加 `--dry-run` 检查配置，或 `--smoke` 进行两批训练检查。实际参数见 reports/round5/train_config.json，逐轮结果见 train_history.jsonl。

验证沿用 round4_predict.py，比较旧提交的 512/768 水平翻转设置与 640/896 水平翻转设置，同时用旧模型运行高分辨率推理作对照。暗图及低对比度分组仍使用训练图像统计的第 25 百分位阈值。最终选定参数以 selection.json 为准；本地验证成绩不能代替官方成绩，多轮选择可能对固定验证集产生适配。

本分支上传源码、配置和报告。继承的旧版 artifacts 不代表本轮权重；新权重保存在本机，预测 ZIP 单独交付。

由于 512/768 与 640/896 在整体和低对比度分组上呈现取舍，追加一次 512/768/896 三尺度验证。候选数量及结果完整记录，不按测试标签选择。

## 验证对照

| 模型及推理 | 整体 | 暗图 | 低对比度 |
|---|---:|---:|---:|
| 第四轮原提交 512/768 | 75.3233% | 68.5649% | 71.5543% |
| 第五轮 512/768 | 75.4906% | 68.7906% | 71.5459% |
| 第五轮 640/896 | 75.4699% | 68.7255% | 71.8545% |
| 第四轮 640/896 | 75.0564% | 68.3183% | 71.6567% |
| 第五轮 512/768/896 | 75.5864% | 68.8724% | 71.9203% |

所有设置均包含水平翻转和概率平均。最终选择见 selection.json；以上为固定本地验证集成绩，官方提升尚未确认。

最终预测命令：
```powershell
python round4_predict.py --checkpoint runs/round5_mild768/best.pth --source round2_model --images data/test2_images --sizes 512 768 896 --hflip --out submission_round5
```
