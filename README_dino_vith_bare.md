# DINOv3 ViT-H+/16 裸地伪标签实验

## 状态

这是可运行的实验源码，**尚未训练、尚未生成预测文件，也没有验证提分**。本机有 6996 张带标签图片（固定划分 6296 训练、700 验证）、1300 张 test2 图片和 CUDA 环境，但没有找到 DINOv3 ViT-H+ 预训练权重。仓库只保存源码、配置和报告，不上传数据或权重。

模型使用 Meta 的 `facebook/dinov3-vith16plus-pretrain-lvd1689m` Hugging Face 格式权重。`--source` 必须是本机目录，包含该模型的 `config.json` 和 `model.safetensors`（或其分片索引与分片文件）。`model_configs/dinov3-vith16plus/config.json` 仅用于架构核对，不是权重。ViT-B 分割权重不能代替 ViT-H+ 预训练权重。

## 方法

1. 以预训练 ViT-H+ 初始化九类分割模型，冻结主干，使用全部真实标签训练分割头；在真实标签上提高裸地 Dice 约束，并惩罚背景误判为裸地。
2. 用训练后的单一 ViT-H+ 教师对 test2 未标注图像生成伪标签。普通类别需至少 0.95 置信度和 3/4 视角一致；裸地需至少 0.98 置信度和 4/4 视角一致。忽略类 0 不作为训练标签。逐图核对与训练、验证图像的像素哈希，防止泄漏。
3. 从教师权重初始化学生，冻结主干，混合真实标签和权重逐渐升至 0.2 的伪标签训练。含较多裸地伪标签的图像被温和地提高采样概率（1.5 倍）。
4. 在同一固定验证集比较教师与学生的整体 mIoU 和裸地 IoU，记录是否优于历史 78.4116% mIoU。无论是否提分，输出学生的候选预测包；只有验证报告显示改进时才建议提交。官方成绩仍需平台评测。

默认输入 448 像素，验证和伪标签采用 448/512 双尺度与水平翻转；主干 BF16、无 EMA，以控制显存。8 GB 显卡是否足以完成整个实验仍需权重到位后做 GPU 冒烟测试。若显存不足，应改用更大显存的训练环境，不应声称实验成功。

## 运行

从本目录执行，替换实际路径：

```powershell
python dino_vith_bare_pipeline.py --source D:/models/dinov3-vith16plus --images D:/data/train/images --masks D:/data/train/masks --test-images D:/data/test2/images --split reports/round2/split.json --class-balance reports/round2/class_balance.json --profiles audit/image_profiles.json --out D:/results/dino_vith_bare
```

使用本机已有的 Python 环境时，把 `python` 换成完整的 `.../b2-env/Scripts/python.exe` 路径。输出目录必须为空；`status.json` 和各阶段日志记录进度。成功后检查 `selection.json`、`student_validation.json` 与 `submission_dino_vith_bare.zip`。

测试：

```powershell
python -m unittest discover -s . -p "test_*.py" -v
```
