# 复赛 Docker 复现材料

## 状态

这是包含全部代码与权重的 Docker 构建目录，不是已构建的镜像 TAR。当前主机无 Docker，镜像构建和 Linux GPU 运行尚未验证。Windows CUDA 下训练、700 张验证、1300 张预测已实测。容器版本与本机观测版本固定一致，但 Linux wheel/基础镜像可用性和平台差异仍需实际构建验证。

## 文件

- src/：完整训练、验证、预测和基础工具代码。
- weights/initial/：本次训练使用的预训练编码器 + 随机分割头初始化快照。
- weights/final/：最终权重及配置；无需联网获取模型即可运行。
- metadata/：固定训练/验证划分和训练集类别统计。
- evidence/：原始指标、逐轮日志和实验配置；配置中的旧本地路径仅为历史记录。
- reproduce.py：统一入口。MANIFEST.json：运行文件 SHA-256。
- environment_observed.txt：本机实测完整 Python 依赖；requirements.txt 是容器直接依赖。

## 数据目录

自行提供官方数据，不在材料包内重新分发图片和标签：

```
data/
  train_images/   # 6996 PNG
  train_masks/    # 6996 class-ID PNG
  test2_images/   # 1300 PNG，解压 test_2.zip 后把 images 内容放这里
```

## 构建、检查和导出

需要 Linux/amd64 Docker 环境、NVIDIA GPU 容器支持及与 CUDA 13.0 兼容的驱动。Windows 可用配置了 GPU 支持的 Docker Desktop Linux 容器。构建需联网获取基础镜像和依赖；运行阶段设置为模型离线模式。

```
docker build -t uav-round2:74.22 .
docker run --rm --gpus all uav-round2:74.22 check
docker save --output uav-round2-image.tar uav-round2:74.22
```

PowerShell 可运行 ./build_and_export.ps1；Linux 可运行 sh build_and_export.sh。脚本只有在构建和 GPU 前向检查成功后才导出镜像。若赛方明确要求 docker load 镜像文件，应提交导出的 TAR，不能将本构建 ZIP 改名为 TAR。

## 复现命令（Linux Shell）

下面 /official/data 和 /absolute/output 均替换为实际绝对路径。输出目录可写，数据只读挂载。

```sh
docker run --rm --gpus all --shm-size=4g -v /official/data:/data:ro -v /absolute/output:/out uav-round2:74.22 eval
docker run --rm --gpus all --shm-size=4g -v /official/data:/data:ro -v /absolute/output:/out uav-round2:74.22 predict
docker run --rm --gpus all --shm-size=4g -v /official/data:/data:ro -v /absolute/new_training:/out uav-round2:74.22 train
docker run --rm --gpus all --shm-size=4g -v /official/data:/data:ro -v /absolute/new_training:/out uav-round2:74.22 ablation
```

eval 写出 validation_512.json。predict 写出 submission_round2_1300/ 和 submission_round2_1300.zip，并检查输出格式。train 执行基础 6 轮后从最佳模型微调 2 轮；ablation 从相同基础最佳模型做 MixStyle 两轮对照。测试训练流程可给 train 追加 --smoke。

eval 默认使用随包最终权重。验证重新训练结果需传 --checkpoint /out/control/best.pth；使用相同 /out 挂载目录。

无需 Docker 的本机验证方式：使用已安装本项目 CUDA 依赖的 Python 执行 python reproduce.py check；其余命令传 --data 官方数据目录 --out 新输出目录。不要覆盖历史实验目录。

## 已观测成绩与限制

固定 700 张验证图、原始标签、忽略 0 类，最终 mIoU 74.22%；普通微调优于本次 MixStyle 对照。详见技术方案 PDF 和 evidence/。没有复赛平台成绩；现有划分未做到已验证的场景级隔离。跨系统、GPU 和非确定性算子可能带来数值差异，不保证重新训练逐位一致。

## 第三方来源

SegFormer / NVIDIA MiT-B1: https://github.com/NVlabs/SegFormer 与 https://huggingface.co/nvidia/mit-b1 。MixStyle: https://github.com/KaiyangZhou/mixstyle-release 。外部预训练使用已获参赛者确认；保留上游模型与代码的来源、许可要求。本项目不把已有网络或 MixStyle 声称为原创。

## 从 GitHub 分支恢复构建目录

本地 Docker 复现材料 ZIP 已含完整权重，可直接构建。GitHub 中权重以分片存放，先在仓库根目录运行 python docker_repro/prepare_context.py，再执行 docker build -t uav-round2:74.22 docker_repro。脚本核对两个 ZIP 和全部权重的 SHA-256。

