# SegFormer-B2 单模型训练与预测

## 状态

本分支提供 B2 训练、验证选优和预测打包源码。正式训练数据和 test2 已找到并解压，
共 6996 张有标签图像、1300 张 test2，使用固定的 6296/700 划分。
GPU 训练、保存权重、验证和预测打包的三个程序测试已通过。
正式实验的完成状态以运行目录的 status.json 为准，尚未取得本次官方成绩。
第一轮现已完成，最佳验证 mIoU 为 74.9372%，1300 张 test2 预测已通过校验。
第二轮已验证类别校正达到 75.5002%，进一步微调流程见 [优化说明](README_b2_optimization.md)。
测试使用的人造图片只用于程序验证，不用于正式训练或提交。

以 `codex/round9-segformer-b5-20260927` 的提交
`a53cad3585b757baa0b4a42b51a635a73abecd78` 为基础，保留既有训练/验证划分。
旧分支中的权重、预测文件和历史成绩不属于本次 B2 实验。

## 方法

- NVIDIA SegFormer-B2 ADE20K 预训练模型；结构检查要求 B2 的四阶段深度与宽度。
- 保留预训练编码器和解码器，仅替换为九分类输出层；0 忽略，1–8 为有效类别，不做标签减一。
- 固定原有 6296/700 划分；类别权重只由训练标签计算。若数据不同，显式传入新的无重叠 split JSON。
- 加权交叉熵、Dice、Lovasz、轻度几何/颜色增强、梯度累积、BF16 和 EMA。
- 512 训练 12 轮，640 小学习率微调 3 轮。默认 batch=1、累积=8；本机 512 的 batch=2 测试通过，训练张量峰值约 3.16GB（不包含驱动和桌面显存）。
- 原始分辨率验证，分别比较两个阶段的最佳权重，以及 512 单尺度和 512/640/768 水平翻转推理。
- 仅由验证集选取一个权重及推理设置；测试图片仅用于最终预测，不做训练、伪标签或模型集成。
- 报告同时记录出现类 mIoU、全部八类 mIoU 和逐类 IoU。验证结果不能等同官方分数。

已读取的历史单模型记录中，本地验证参考最高值为 78.4116%（round10），
用户报告的官方参考成绩为 68.45（round8）。round10 官方成绩为 68.36，
说明本地验证变好并不保证官方成绩提高。selection.json 会明确记录与本地参考值的差值；
即使未超过参考值仍可导出实验预测文件，但不会声称已提分。

## 安装

Python 3.12；安装到单独虚拟环境。下面是 CUDA 12.8 版本的安装示例：

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_b2.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
python -m unittest test_b2 -v
```

训练需要支持 BF16 的 CUDA GPU。原训练分支依赖 transformers 5.x；本 B2 路径使用
独立模型封装并固定 transformers 4.57.1，不依赖旧代码中的内部 `stages` 接口。

## 一键正式训练、验证和预测

```powershell
python b2_pipeline.py --images DATA/train_images --masks DATA/train_masks --test-images DATA/test_images --out outputs/segformer_b2
```

默认首次运行从 `nvidia/segformer-b2-finetuned-ade-512-512` 下载公开预训练权重。
也可传入 `--source LOCAL_MODEL_DIRECTORY` 使用本地 Hugging Face 格式的配置和权重。
默认 `--split reports/round2/split.json`。预训练来源见
[NVIDIA 模型页](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)。
安装依据见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

结果目录包含：

- `status.json`、训练/验证日志、训练配置、固定划分和训练类别统计。
- `train512/best.pth` 与 `refine640/best.pth`、各自模型配置和训练历史。
- `selection.json`：全部候选的验证结果和最终选择。
- `submission_b2/`：逐图预测；`submission_b2.zip`：可提交预测包。
- `submission_b2.manifest.json`：图片数量、SHA256、使用权重、尺度与翻转配置。

每个输出 PNG 必须是 1024×1024、L 模式、值域 0–8；文件名与测试图片主文件名一一对应。
ZIP 在完整校验后才重命名为最终文件。重复主文件名、空输入或非空输出目录会报错。
出现失败时检查 `status.json` 和相应日志；不会将旧目录的预测冒充本次结果。
当前训练入口不支持中断续训；`last.pth` 保存学生模型与优化器供诊断，正式重跑使用新目录。

数据、缓存、模型权重及预测包只保存在运行机器，不作为本次源码提交内容。
