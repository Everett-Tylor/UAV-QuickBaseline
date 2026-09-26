# 第六轮：针对薄弱类别和成像条件的训练采样

第五轮官方成绩为 67.8 分（用户回报）。本轮目标是继续向 70 分推进，但本地验证不能保证官方达到目标。

## 动机与方法

第五轮本地整体 mIoU 75.5864%，暗图 68.8724%，低对比度 71.9203%；裸地类别 IoU 51.5540%。本轮使用已有最佳 SegFormer B1 权重，改变训练样本出现频率并恢复较强光度增强。当前环境网络限制导致新 B2 权重下载失败，本难组采样候选没有使用 B2 或 DINOv3；之后用户提供了本地 DINOv3 权重，主实验见 README_dinov3.md。

仅在固定 6296 张训练图像上统计 mask 类别像素。每张图初始采样权重 1，裸地占比至少 1% 时加 0.75，车辆占比至少 0.1% 时加 0.35；亮度或对比度低于训练集各自 25% 分位时分别加 0.35。权重有界，以放回方式每轮抽取 6296 张。采样统计必须与训练名单完全一致，禁止包含验证掩码。

这是自定义的有界难组采样，借鉴稀有类别增加采样机会的通用思路，不是完整复现 DAFormer 或其域适应流程。参考：[DAFormer, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.pdf)。本实验未使用无标签测试数据进行域适应。

4 轮、768 输入、batch 4、梯度累积 2、seed=20260926、编码器学习率 6e-6（解码头 10 倍）、AdamW weight decay 0.02、EMA。损失仍为加权 CE + 0.5 Dice + 0.3 Lovasz-Softmax。沿用第三/四轮 RobustDataset 的随机裁剪、翻转旋转、亮度/对比度/色彩、gamma 和轻度模糊增强。验证保持原固定 700 张，不参与采样权重估计。

## 复现

使用 requirements_round2.txt，先按第五轮说明得到初始化模型。其 SHA256 为 `8c5bf6bc394ab26254c1cbc39f4466bf782a475770e32a369b515b493bb6217f`。

```powershell
python round3_audit.py --images data/train_images --test-images data/test2_images --split reports/round2/split.json --out runs/round3_audit
python round6_mask_counts.py --masks data/train_masks --split reports/round2/split.json --out runs/round6_train_counts.json
python check_round6_sampling.py
python round6_train.py --images data/train_images --masks data/train_masks --split reports/round2/split.json --class-balance reports/round2/class_balance.json --source round2_model --init runs/round5_mild768/best.pth --mask-counts runs/round6_train_counts.json --profiles runs/round3_audit/image_profiles.json --out runs/round6_balanced768 --epochs 4 --size 768 --batch 4 --accum 2 --workers 4 --lr 0.000006 --seed 20260926
```

使用 round4_predict.py 在 512/768/896 三尺度及水平翻转下统一比较新模型与第五轮。必要时比较新旧模型等权概率融合。全部候选报告会保留，最终选择见 reports/round6/selection.json。多次模型选择可能对固定验证集产生适配，官方结果需要单独提交确认。

本分支上传源码与实验记录，新权重保存在本机，预测 ZIP 单独交付。
