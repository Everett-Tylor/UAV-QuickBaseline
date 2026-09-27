# 复赛训练结果

最终本地验证 mIoU：**74.22%**；此前优化版为 50.99%，提升 **23.23 个百分点**。

这不是竞赛平台成绩。使用同一份 700 张验证图像、原始 1024×1024 标签评估，忽略类别 0，平均类别 1–8。验证集不参与训练；原始划分为随机图像划分，场景相近可能使本地成绩偏乐观。

## 实验对照

| 方案 | 最佳 mIoU |
|---|---:|
| 原 U-Net 优化版 | 50.99% |
| SegFormer-B1，6 轮，384 | 73.36% |
| 相同起点普通微调 2 轮，384 | 74.11% |
| 相同起点 MixStyle(p=0.2) 微调 2 轮，384 | 73.68% |
| 选定普通微调权重，512 推理 | 74.22% |

MixStyle 在本次设置下没有超过普通微调，因此最终未采用。两组均从基础训练第 5 轮的同一权重开始，使用相同数据划分、训练顺序种子、学习率和轮数；这是一组单种子对照，不代表 MixStyle 在所有设置下均无效。

## 每类 IoU

| 类别 | 旧模型 | 本次最终 |
|---|---:|---:|
| 1 背景 | 41.28% | 65.97% |
| 2 建筑 | 56.40% | 80.16% |
| 3 道路 | 49.77% | 77.67% |
| 4 水体 | 64.18% | 86.68% |
| 5 裸地 | 24.10% | 50.70% |
| 6 植被 | 74.46% | 86.13% |
| 7 农业用地 | 54.79% | 73.79% |
| 8 车辆 | 42.92% | 72.64% |

## 训练设置

- NVIDIA MiT-B1 ImageNet-1k 预训练编码器，9 类随机分割头。已获得用户允许使用外部预训练的确认。
- 训练 6296 张，验证 700 张；训练输入 384；batch 8，AMP；AdamW，weight decay 0.01，梯度裁剪 1。
- 基础训练 6 轮：编码器初始学习率 6e-5，分割头为其 10 倍。微调 2 轮：编码器 2e-5，头部 10 倍；预热 100 个 batch，polynomial 衰减。
- 加权交叉熵 + 0.5 Dice，权重仅由训练集统计；水平/垂直翻转、90°旋转、轻度亮度/对比度/色彩增强。
- 最终模型：基础训练第 5 轮 + 普通微调第 2 轮；单模型 512 输入，无 TTA、无模型集成。
- 硬件：RTX 5070 Laptop 8GB；训练张量峰值约 2.10 GB，不包含全部驱动/桌面显存。

## 复赛数据与文件

test_2.zip 包含 1300 张 RGB 图片，全部 1024×1024，没有标签。仅用于推理，没有用于伪标签训练或验证。
提交文件：submission_round2_1300.zip。PNG 位于 ZIP 根目录，文件名匹配原图，模式 L，1024×1024，像素为类别 ID 0–8。
模型包：round2_model.zip（best.pth、config.json、inference.json 和预训练来源信息）。不需要重新下载初始化权重即可推理。
源码包：UAV_复赛源码与报告.zip；包含固定划分、类别统计、实验日志与指标。

## 推理示例

安装 requirements_round2.txt，并根据显卡安装 CUDA 版 PyTorch。解压模型包后运行：

```powershell
python round2_predict.py --checkpoint round2_model/best.pth --source round2_model --images "复赛图片目录" --size 512 --out submission_round2
```

省略 --size 时脚本沿用训练输入 384；复现最终提交必须显式指定 --size 512。

## 来源

- https://huggingface.co/nvidia/mit-b1
- https://github.com/NVlabs/SegFormer
- https://github.com/KaiyangZhou/mixstyle-release

权重通过 hf-mirror.com 下载，原始文件 SHA-256 与镜像 API 的 LFS 标识一致，完整记录见 round2_pretrained_provenance.json。群聊中的其他分数未视为已验证结果。

## GitHub 模型下载

模型包受上传接口限制拆为 7 片。将 artifacts/round2 中的 round2_model.zip.part01 至 part07 和 restore_round2_model.ps1 放在同一目录，在 PowerShell 中运行 ./restore_round2_model.ps1。脚本会还原 ZIP、检查 SHA-256 并解压。还原后的模型 ZIP 与本地交付文件完全一致。
