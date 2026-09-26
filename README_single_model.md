# 复赛单模型方案

**更新：第七轮单模型经官方评分为 67.70 分。新的 MixStyle 单模型实验见 [第八轮说明](README_round8_mixstyle.md)，官方成绩尚未取得。**

比赛规则要求最终结果只包含一个模型。先前的 DINOv3 + SegFormer B1 融合预测虽然得到用户反馈的官方 69.88 分，但不符合这项要求，不应继续提交。本方案只加载一个 DINOv3 ViT-B/16 分割模型权重；不同输入尺度和水平翻转都是同一个权重的推理视图，没有第二个模型。

本模型沿用官方 6296 张训练图与固定 700 张验证图，未用复赛测试图训练或制作伪标签。初始化使用用户提供的公开 DINOv3 ViT-B/16 预训练权重，之后只用官方训练集监督微调。DINOv3 权重不包含在本分支中。

从第六轮 640 输入的最佳单模型 EMA 权重继续训练：768 输入、batch 4、累积 2、编码器学习率 2e-6、解码器学习率 3e-5，训练 4 轮。最佳权重来自第 4 轮。训练源码仍为 `dino_train.py`，结构为 `dinoseg.py`。

```powershell
python dino_train.py --images data/train_images --masks data/train_masks --split reports/round2/split.json --class-balance reports/round2/class_balance.json --source model_configs/dinov3 --init runs/round6_dino640/best.pth --out runs/round7_dino768 --epochs 4 --freeze-epochs 0 --size 768 --batch 4 --accum 2 --workers 4 --lr 0.000002 --head-lr 0.00003 --seed 20260929
```

固定验证集的类别 1–8 mIoU：

| 推理配置 | mIoU |
| --- | ---: |
| 第六轮双模型融合，现已停用 | 78.3380% |
| 第六轮 DINOv3 单模型，640 单尺度 | 77.7201% |
| 第七轮 DINOv3 单模型，768 单尺度 | 77.9898% |
| 第七轮 DINOv3 单模型，512/768/896 + 水平翻转 | 78.0843% |
| 第七轮 DINOv3 单模型，640/768/896/1024 + 水平翻转 | 78.2027% |
| 第七轮 DINOv3 单模型，768/896/1024/1152 + 水平翻转 | **78.2101%** |

最终配置仍比旧双模型融合低 0.1279 个百分点。不能根据本地验证推断官方是否超过旧结果 69.88 分，需提交后确认。裸地类阈值校准在调参子集上增分，但在留出子集上下降，因此未采用。

```powershell
python dino_predict.py --checkpoint runs/round7_dino768/best.pth --source model_configs/dinov3 --images data/test2_images --sizes 768 896 1024 1152 --hflip --out submission_round7_single_model
```

输出 ZIP 根目录包含与复赛测试图同名的 1300 张原始尺寸单通道 PNG，类别 ID 为 0–8。验证与训练记录在 `reports/round7/`。比赛如果进一步禁止单模型的多尺度或翻转推理，可改用单尺度方案，并重新提交对应预测。
