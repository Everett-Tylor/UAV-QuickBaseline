# 第三轮优化：以官方 65 分为旧版基准

用户回报此前版本官方 mIoU 为 65 分。本轮没有官方成绩，不把本地验证分数当作竞赛分数。

## 方法

从上一轮 SegFormer-B1 最佳权重继续监督微调 6 轮：640 输入、batch 4、累积 2（有效 batch 8）、编码器学习率 1.5e-5、解码头 10 倍、AdamW weight decay 0.02、加权 CE + 0.5 Dice。固定 seed=20260923。保持原训练 6296 / 验证 700 划分。

随机 0.75-1.0 尺度裁剪（50% 概率）、翻转/90 度旋转、更强亮度/对比度/色彩增强、随机 gamma 和轻度模糊。加入 EMA；仅在优化器有效更新后更新平均权重。通过固定验证集比较单尺度与多尺度水平翻转，另测试仅用干净训练图像重新估计 BatchNorm 统计。

复赛测试图像没有标签，没有用于监督训练、伪标签训练或 BN 校准。数据检查未发现跨组完全相同的像素图像，但没有证明场景隔离。复赛图像亮度中位数约 0.340，训练集约 0.381；这只是分布差异的证据，不能解释全部官方分差。

## 验证

| 验证分组 | 旧模型 | 本次选定设置 |
|---|---:|---:|
| 全部 700 张 | 74.22% | 75.08% |
| 暗图 164 张 | 66.88% | 68.27% |
| 低对比度 156 张 | 71.47% | 71.59% |

最终配置见 [selection.json](reports/round3/selection.json)。各组仍来自同一验证集，不是独立外部测试；暗图和低对比度阈值由训练图像统计的第 25 百分位确定。沿用随机图像划分可能偏乐观，多次模型选择也可能导致验证集适配。官方提升需提交后确认。

## 复现

沿用 requirements_round2.txt 中的环境。先还原上一轮模型包为 round2_model/，固定划分与类别权重位于 reports/round2/。

```powershell
python round3_audit.py --images data/train_images --test-images data/test2_images --split reports/round2/split.json --out runs/round3_audit
python round3_train.py --images data/train_images --masks data/train_masks --split reports/round2/split.json --class-balance reports/round2/class_balance.json --source round2_model --init round2_model/best.pth --out runs/round3_robust640 --epochs 6
python round3_predict.py --checkpoint runs/round3_robust640/best.pth --source round2_model --images data/train_images --masks data/train_masks --split reports/round2/split.json --profiles runs/round3_audit/image_profiles.json --sizes 640 --out runs/eval_single.json
python round3_predict.py --checkpoint runs/round3_robust640/best.pth --source round2_model --images data/train_images --masks data/train_masks --split reports/round2/split.json --profiles runs/round3_audit/image_profiles.json --sizes 512 768 --hflip --out runs/eval_tta.json
```

BN 校准是一个待评估候选，不能默认认为一定提高：

```powershell
python round3_calibrate_bn.py --checkpoint runs/round3_robust640/best.pth --source round2_model --images data/train_images --split reports/round2/split.json --out runs/round3_calibrated.pth
```

使用 selection.json 记录的权重和 sizes/hflip 参数预测：把上述评估命令的 images 换为 data/test2_images，移除 masks/split/profiles 参数，out 指向全新的预测目录。脚本生成 PNG 和 ZIP，并执行格式及 CRC 校验。

本次按用户要求上传源码和实验记录；新权重保存在本机，预测 ZIP 单独交付。之前分支继承的 artifacts 属于上一轮，不能当成本轮权重。初始化和训练过程均依赖用户此前允许的外部预训练编码器。

## 两轮温和精修

为改善强增强后的分组取舍，另外从 robust640 最佳模型进行两轮温和精修，使用原 round2seg.py 的全图缩放和轻度颜色增强，关闭 EMA 与 MixStyle。相同划分，640 输入，batch 8，编码器学习率 5e-6，头部 10 倍，seed=20260924。

```powershell
python round2seg.py --images data/train_images --masks data/train_masks --split reports/round2/split.json --class-balance reports/round2/class_balance.json --source round2_model --init runs/round3_robust640/best.pth --out runs/round3_refine640 --epochs 2 --batch 8 --accum 1 --size 640 --lr 0.000005 --seed 20260924
```

旧模型的同一多尺度翻转对照记录于 baseline_tta.json，避免把全部推理收益归因于新训练。最终是否采用精修或校准，以 selection.json 为准。
