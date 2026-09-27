# 复赛：SegFormer-B1，验证 mIoU 74.22%

同一份 700 张验证集，mIoU 从 50.99% 提高到 74.22%（+23.23 个百分点）。这是本地验证结果，尚无竞赛平台评分。

- [1300 张复赛预测 ZIP](artifacts/round2/submission_round2_1300.zip)
- [模型权重与配置](artifacts/round2)
- [源码与报告快照](artifacts/round2/round2_source_and_reports.zip)
- [完整实验报告](reports/round2/results.md)
- [SHA-256 校验清单](artifacts/round2/manifest.json)

解压模型包后，使用以下命令复现最终推理设置：

```powershell
python -m pip install -r requirements_round2.txt
python round2_predict.py --checkpoint round2_model/best.pth --source round2_model --images "复赛图片目录" --size 512 --out submission_round2
```

# 实验说明

用户已确认可以使用外部预训练权重。该许可适用于本次复赛实验；旧版 README 的“不使用预训练”描述仅对应旧版 U-Net。

## 数据与评估

- 官方有标注数据 6996 张，沿用此前固定划分：训练 6296 / 验证 700；类别 0 忽略，报告类别 1–8 的 mIoU。
- test_2.zip：1300 张 RGB、1024×1024 图片，无标注。只用于预测，不进入有监督训练或验证。
- 验证输入按双线性缩放到训练尺寸，输出 logits 双线性还原至原始尺寸，再 argmax；原始标签不缩放。
- 此前随机图像划分可能包含相近场景，因此本地验证不等同于复赛平台分数。

## 模型

使用 NVIDIA MiT-B1 ImageNet-1k 预训练编码器和随机初始化的 9 类 SegFormer 分割头。
模型源：https://huggingface.co/nvidia/mit-b1
上游版本：13ddceec4e8bdf401e7cd7acf5aebc526222518c
权重下载通过 hf-mirror.com；SHA-256 与镜像返回的 LFS 标识一致，详见 round2_pretrained_provenance.json。未下载 DINOv3 权重。

RGB 使用 ImageNet 均值/标准差；全图缩放、翻转、90°旋转、轻度颜色增强；训练集像素统计得到的加权交叉熵 + 0.5 Dice；FP16 AMP、AdamW、梯度裁剪、预热及 polynomial 学习率衰减。

MixStyle 在前两个编码器阶段输出上混合 batch 内特征均值/标准差，仅训练开启。`--mixstyle 0` 关闭，非零为应用概率；实现参考 https://github.com/KaiyangZhou/mixstyle-release 。是否保留由固定验证集的对照结果决定。

## 环境

本机已验证 Python 3.14 / PyTorch 2.14.0+cu130 / Transformers 5.17.0。使用独立虚拟环境，继承已有 CUDA PyTorch。代码使用 Transformers 5.17 的 SegFormer stages 接口，请固定该版本。

## 使用

`round2seg.py --help` 查看训练参数；`round2_predict.py --help` 查看验证/预测参数。
`--source` 指向已保存的本地 SegFormer 初始化目录（含 config.json、model.safetensors）；后续加载不会联网。
`--init` 从训练检查点加载模型参数开展新一段微调，优化器和学习率计划重新初始化，不代表断点续训。
`--class-balance` 必须对应相同训练划分的统计文件；本次使用旧实验保存的 class_balance.json。
推理 ZIP 的 PNG 位于压缩包根目录，模式 L，尺寸与输入一致，像素为 0–8 类别 ID。

实验结果需以保存的 history.jsonl 与最终报告为准，不将群聊中的分数视为已复现结果。

## GitHub 模型下载

模型包受上传接口限制拆为 7 片。将 artifacts/round2 中的 round2_model.zip.part01 至 part07 和 restore_round2_model.ps1 放在同一目录，在 PowerShell 中运行 ./restore_round2_model.ps1。脚本会还原 ZIP、检查 SHA-256 并解压。还原后的模型 ZIP 与本地交付文件完全一致。

## 复赛提交材料

- [1300 张预测结果 ZIP](artifacts/round2/submission_round2_1300.zip)
- [技术方案 PDF](docs/round2_technical_proposal.pdf)：设计动机、方法、例代码和实验结果。
- [Docker 复现目录](docker_repro/README.md)：完整训练/验证/预测代码、环境及脚本。
- [验证状态](reports/round2/verification_status.json)

从本分支完整下载后，先运行 `python docker_repro/prepare_context.py` 还原最终与初始化权重，再进入 docker_repro 构建镜像。全部模型分片在 artifacts/round2，脚本自动核对 SHA-256。

当前主机没有 Docker，容器构建和 Linux GPU 验证尚未执行，未生成镜像 TAR。Windows 下统一入口已通过 GPU 前向、两步训练试跑和完整 700 张验证（74.2165%，与原结果完全一致）。若赛方要求镜像 TAR，请在可用 GPU Docker 环境执行构建目录中的 build_and_export 脚本。
