# DINOv3 伪标签训练：裸地优先

## 当前状态与前提

这是一个待运行的单模型实验。当前电脑已找到官方训练集和 test2，但**没有找到第 8 轮 DINOv3
分割模型 `best.pth`**，因此还没有生成伪标签、训练新模型或预测 test2。
GitHub 保存了第 8 轮的源码、模型配置和报告，没有保存该权重文件。预训练 DINOv3 原始权重不能替代已训练的九类分割教师模型。
Meta 的 [DINOv3 ViT-B/16 模型](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
需要登录并接受访问条件；本机目前没有 Hugging Face 授权。

历史第 8 轮固定验证 mIoU 为 78.3839%，裸地 IoU 为 53.0777%；
第 10 轮最高本地 mIoU 为 78.4116%，但官方成绩没有提高。
原第 8 轮裸地真值中约 76.54% 被找回，预测为裸地的像素中仅约 63.39% 正确；
最大的误检来源是背景（类别 1），因此本轮重点是减少背景→裸地误判。
这些都是历史验证结果，不是本轮成绩。

## 方法

固定教师推理 test2 的 1300 张未标注图像。普通非忽略类别要求平均置信度至少 0.95、
多视角一致率至少 0.75；裸地伪标签要求更严格的 0.98 和 0.875（八视角至少七个一致）。
测试图片不使用真值标签。严格检查测试图与训练/验证图没有像素重复、伪标签的哈希未变化、
学生初始化与教师权重一致。

训练时仍使用全部真实训练标签。仅对保留裸地伪标签的图片温和提高抽样概率（1.5 倍）；
真实标签损失额外惩罚背景像素被预测为裸地，并给裸地 Dice 小幅权重。
因此裸地重点同时覆盖正样本和主要误检来源，避免仅提高裸地召回率、让误检更严重。
伪标签损失系数从 0 增至 0.2；单模型、无权重或预测集成。

新模型只有在相同验证集上同时满足以下条件才会生成新的 test2 提交包：
整体 mIoU 超过教师和历史 78.4116%，裸地 IoU 超过教师，暗图和低对比图 mIoU
没有超过 0.5 个百分点的回退。满足这些条件仍不代表官方成绩必然提升。

## 数据与运行

已完成的本地数据检查：6296 张训练、700 张固定验证、1300 张 test2，跨组像素完全重复为 0。
请提供第 8 轮 `best.pth` 的**本机完整路径**，且该权重应匹配
`model_configs/dinov3/config.json` 的 DINOv3 ViT-B/16 金字塔分割头。
此实验不会使用 B2 的权重。

在工作区根目录：

```powershell
python work/dinov3-pseudo/dino_bare_pipeline.py --teacher "D:/path/to/round8/best.pth" --source work/dinov3-pseudo/model_configs/dinov3 --images work/data/train/images --masks work/data/train/masks --test-images work/data/test2/images --split work/dinov3-pseudo/reports/round2/split.json --class-balance work/dinov3-pseudo/reports/round2/class_balance.json --profiles work/dinov3-pseudo/audit/image_profiles.json --out outputs/dinov3_bare_pseudo
```

工作区虚拟环境为 `work/b2-env`，已装 CUDA PyTorch 2.8.0 与 Transformers 4.57.1。
先运行测试：

```powershell
work/b2-env/Scripts/python.exe -m unittest discover -s work/dinov3-pseudo -p "test_*.py" -v
```

程序先验证教师、生成伪标签、进行 640 分辨率 GPU 反向传播测试，之后训练 3 轮、
验证并按条件生成 `submission_dino_bare.zip`。状态和日志写入输出目录。
若权重缺失、伪标签不足、GPU 显存不够或验证未改善，会明确记录原因并保留已有结果。
