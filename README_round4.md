# 第四轮：以官方 67 分为基准优化 IoU

用户反馈第三轮提交的官方 mIoU 为 67 分，第二轮为 65 分。第四轮的新官方分数尚未取得。本地固定验证集与官方测试集不同，不能将本地成绩直接视为官方成绩。

## 方法

从第三轮 `round3_refine640/best.pth` 继续监督训练。沿用 SegFormer B1、固定 6296/700 划分、640 输入、batch 4、累积 2、4 轮、编码器学习率 5e-6（解码头 10 倍）、AdamW 权重衰减 0.02、EMA，以及第三轮的随机尺度裁剪与光度增强。

目标函数为加权交叉熵 + 0.5 Dice + 0.3 Lovasz-Softmax。Lovasz 项按当前 batch 出现的非忽略类别平均，标签 0 的像素不参与损失。方法参考 Berman 等人的 [Lovasz-Softmax 论文和项目](https://github.com/bermanmaxim/LovaszSoftmax)。本次实现使用 FP32 概率与排序，并检查全忽略标签和梯度。

原定更大 B2 编码器实验因当前环境无法下载权重而未开展；本轮结果不包含 B2。继承的 B1 预训练来源见历史报告。测试集未参与训练、伪标签或 BN 校准。

## 复现

使用 requirements_round2.txt 环境，并先按第三轮文档训练得到初始化权重。新权重保存在本机，未上传至源码分支。第三轮初始化 SHA256：`bead386bab293b40e8c64ec1daba6aab26ccca21a5168305fb97d014012f65b7`。

```powershell
python round4_train.py --images data/train_images --masks data/train_masks --split reports/round2/split.json --class-balance reports/round2/class_balance.json --source round2_model --init runs/round3_refine640/best.pth --out runs/round4_lovasz640 --epochs 4 --size 640 --batch 4 --accum 2 --workers 4 --lr 0.000005 --seed 20260924
python round4_predict.py --checkpoint runs/round4_lovasz640/best.pth --source round2_model --images data/train_images --masks data/train_masks --split reports/round2/split.json --profiles runs/round3_audit/image_profiles.json --sizes 512 768 --hflip --out runs/round4_eval.json
```

`--ensemble-checkpoint runs/round3_refine640/best.pth` 可进行等权概率融合。仅在固定验证集上比较新模型和融合，不按测试标签调参。最终采用设置和全部比较见 reports/round4/selection.json 及评估报告。

预测时使用选定参数，把 images 换为 data/test2_images，移除 masks/split/profiles 参数，out 指向新的预测目录。脚本自动生成 ZIP 并验证文件名、尺寸、像素类别及 CRC。

## 温和精修候选

第一阶段结束后，另从其最佳模型训练 2 轮：使用 `--mild` 全图缩放和较轻颜色增强，编码器学习率 2e-6，seed=20260925，其余损失和 EMA 不变。此阶段只有验证结果更好时才选用。

```powershell
python round4_train.py --images data/train_images --masks data/train_masks --split reports/round2/split.json --class-balance reports/round2/class_balance.json --source round2_model --init runs/round4_lovasz640/best.pth --out runs/round4_mild640 --epochs 2 --size 640 --batch 4 --accum 2 --workers 4 --lr 0.000002 --seed 20260925 --mild
python check_round4_loss.py
```

## 完整验证结果

| 候选 | 整体 mIoU | 暗图 | 低对比度 |
|---|---:|---:|---:|
| 上次官方67分版本 | 75.0788% | 68.2742% | 71.5872% |
| Lovasz强增强 | 75.3115% | 68.4950% | 71.5505% |
| 新旧等权融合 | 75.2362% | 68.4469% | 71.6201% |
| 温和精修（选定） | 75.3233% | 68.5649% | 71.5543% |

所有候选使用相同 512/768 多尺度与水平翻转；分组阈值来自训练集统计，沿用第三轮定义。精修版整体和暗图较旧版改善，低对比度组略降约 0.033 个百分点。固定验证集多次选择存在适配风险，尚无本轮官方成绩。

最终预测命令：
```powershell
python round4_predict.py --checkpoint runs/round4_mild640/best.pth --source round2_model --images data/test2_images --sizes 512 768 --hflip --out submission_round4
```
