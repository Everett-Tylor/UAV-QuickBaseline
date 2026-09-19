# 无人机航拍语义分割：赶截止时间的最小可运行版本

**这不是已训练好的参赛作品。** 本包不含官方数据、训练权重或真实预测结果。必须用官方训练集训练，再使用官方初赛测试集生成并校验 `submission.zip`；未经训练、直接上传本代码压缩包不能替代预测提交。不能保证得分或超过官方基线。

## 这个简版有什么
- 一个从零训练的轻量级 U-Net（不是 SegFormer，也不使用外部数据或预训练权重），便于紧急跑通。
- 读取官方 RGB 图像和单通道类别 ID 标签；处理 0=忽略、1–8=实际类别。
- 训练过程中输出验证集 mIoU（**仅按验证集中实际出现的非忽略类求均值**，若官方计算缺失类的方式不同，数值可能不同）。
- 输出完整 1024×1024 单通道 `L` 模式 PNG，像素值 0–8；自动检查文件名、数量、尺寸、模式和类别值，最终打包 ZIP。

## 一、先准备数据
报名后获取赛事**正式训练集及测试集**，不要公开传播官方数据。调整成下面的路径；图片和标签文件的主文件名必须一致（例如 `0001.jpg` 对应 `0001.png`）：

```
UAV_QuickBaseline/
  quickseg.py
  requirements.txt
  data/
    train_images/    # 官方训练 RGB 图片，png/jpg均可
    train_masks/     # 官方单通道标签图，0-8
    test1_images/    # 初赛测试 RGB 图片
```

如果官方数据本身有不同目录名字，不需要重命名全部文件，只改下面命令的目录参数即可。先抽查标签必须为**像素值 ID**，不能把彩色可视化标签当作真实标签。若训练数据存在同源相邻裁剪图片，本脚本随机验证划分仅用于紧急自检，不能替代严格的按场景划分。

## 二、安装（3090/4090/5090机器）
Python 3.10/3.11/3.12均可先尝试。先到 https://pytorch.org/get-started/locally/ 按显卡、驱动、CUDA选择官方 PyTorch 安装命令。之后在项目目录执行：

```
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

必须看到 `True`，否则全量训练可能很慢。特别是5090应使用与其架构兼容的PyTorch/CUDA构建，不要强装旧版组合。

## 三、训练：直接复制（先5轮，视截止时间加轮次）

```
python quickseg.py train --images data/train_images --masks data/train_masks --epochs 5 --size 384 --batch 8 --output runs/quick
```

若显存不足 `--batch 4`；若时间允许，可以试 `--epochs 10 --size 512 --batch 4` 重新训练对比。训练产物在 `runs/quick/best.pth`。这是随机初始化 U-Net；训练不充分可能只有很低的得分，需查看loss、各类别IoU，并保证模型不是只预测背景。**不要根据测试集标签反复调参，也不要手工修改预测结果。**

## 四、用官方初赛测试集预测

```
python quickseg.py infer --images data/test1_images --weights runs/quick/best.pth --output predictions
```

自动保存与测试图片主文件名相同的PNG。如官方测试集包含两个文件同名但扩展名不同，脚本会报错，请先核对官方命名规则。

## 五、生成真正要上传的结果ZIP

```
python quickseg.py pack --images data/test1_images --predictions predictions --zip submission.zip
```

打印 `VALIDATED ...` 才意味着格式检查通过。**初赛要提交的是 `submission.zip`，而不是本项目源代码 ZIP。** 先确认官方平台当前轮次的具体提交要求。本工具只实现赛题文件明确给出的初赛预测格式；后续复赛、半决赛还需补齐 Docker、技术方案PDF等材料。

## 注意事项
- 只允许官方数据；本模型从零训练，不引入外部数据或预训练权重，不做模型集成。
- 输出PNG必须完整 1024×1024、单通道且像素值在0–8、文件名一一对应。0是忽略类别；模型仍可能预测0，如果该行为影响官方得分，可在仅依据训练集与公开规则的前提下设计后处理并通过验证集测试。
- 提交前一定要有真正训练完成的 `best.pth` 和 `submission.zip`，程序只能帮你生成，无法在缺少官方训练/测试图片的情况下替你凭空产出有效成绩。
- 本包的本地测试只使用人为生成的测试样例检查程序可运行，不会将模拟数据用于实际训练或正式提交。
