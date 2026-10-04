# DINOv3 ViT-B/16：裸地优先的伪标签训练

## 输入与来源

使用 6996 张官方带标签图像的固定划分（6296 训练、700 验证）及 1300 张 test2 未标注图像。test2 不使用真值标签。`round3_audit.py` 检查跨组像素完全重复，`dino_pseudo.py` 保存并验证图像和伪标签哈希。

主干模型为 Meta 的 `facebook/dinov3-vitb16-pretrain-lvd1689m`。运行前需取得其原始 ViT-B/16 预训练权重，并遵守 [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)。`convert_dinov3_vitb.py` 仅接受 SHA-256 为 `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c` 的原始 `.pth`，严格匹配所有 211 个 Transformers 张量，并校验保存后的权重。模型权重、训练数据和生成的预测文件不存入 GitHub 源码分支。

## 实验流程

1. 以预训练 ViT-B 初始化九类分割网络；第一轮冻结主干，之后联合训练 8 轮，使用 512 像素、MixStyle、EMA、真实标签和裸地重点损失。该损失兼顾裸地召回及背景误判为裸地的问题。
2. 单一教师在固定验证集记录整体 mIoU 与裸地 IoU，再以 512/640/768/896 四尺度和水平翻转为 test2 生成伪标签。普通类别要求至少 0.95 置信度、6/8 视角一致；裸地要求至少 0.98 置信度、7/8 视角一致。低于 2% 有效像素的图像不参与伪标签训练。
3. 学生从教师权重初始化，训练 3 轮；全部真实标签保留，伪标签损失系数首轮从 0 升到 0.2。含较多裸地伪标签的图像采样概率增加 1.5 倍。
4. 在同一 700 张验证集比较教师和学生。记录是否同时提高整体与裸地 IoU，以及是否超过历史 78.4116% mIoU。生成候选 test2 预测 ZIP，并校验 1300 个 PNG 的文件名、尺寸和类别 ID。只有验证指标支持时才建议提交；官方分数需另行评测。

训练分别保留整体 mIoU 最佳和裸地 IoU 最佳检查点。如果二者来自不同轮次，程序对两者都进行完整验证；只有裸地最佳的整体 mIoU 不比整体最佳低超过 0.3 个百分点、且裸地 IoU 更高时，才选它导出预测。

## 命令

先转换与 SHA-256 匹配的原始权重：

```powershell
python convert_dinov3_vitb.py --checkpoint D:/models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --config model_configs/dinov3 --out D:/models/dinov3-vitb-hf
```

然后从本目录运行，替换实际路径：

```powershell
python dino_vitb_bare_pipeline.py --source D:/models/dinov3-vitb-hf --images D:/data/train/images --masks D:/data/train/masks --test-images D:/data/test2/images --split reports/round2/split.json --class-balance reports/round2/class_balance.json --out D:/results/dino_vitb_bare
```

程序自动生成图像审计，也可用 `--profiles` 指向已有 `image_profiles.json`。输出目录必须为空；`status.json`、各阶段日志、`teacher_validation.json`、`student_validation.json` 和 `selection.json` 用于审阅。成功后候选预测包是 `submission_dino_vitb_bare.zip`。

已有训练好的九类 DINOv3 ViT-B 教师时，可加 `--teacher D:/models/teacher-best.pth` 跳过监督训练，继续伪标签和学生训练。此时 `--source` 只需模型配置，无需再次提供预训练权重。

若初始教师的验证分数仍偏低，可先做 640 像素精修，再将精修结果作为 `--teacher`。本次实验从 512 像素第 8 轮最佳检查点继续，精修参数为 4 轮、batch 8、编码器学习率 `3e-6`、解码器学习率 `5e-5`、温和增强、MixStyle、EMA 与裸地重点损失。示例：

```powershell
python dino_train.py --images D:/data/train/images --masks D:/data/train/masks --split reports/round2/split.json --class-balance reports/round2/class_balance.json --source model_configs/dinov3 --init D:/results/teacher/best.pth --out D:/results/refine640 --size 640 --batch 8 --accum 1 --workers 2 --epochs 4 --freeze-epochs 0 --lr 3e-6 --head-lr 5e-5 --augmentation mild --focus-bare --mixstyle --seed 20261004
```

待 `refine640/best.pth` 的固定验证指标写出后，再传入主流程的 `--teacher`。所有阶段均从新输出目录运行，避免覆盖检查点。
