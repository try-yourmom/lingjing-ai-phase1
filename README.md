# 灵境竞赛组后端 AI 方向第一阶段项目

## 项目方向

基于手写 ResNet 的植物叶片病害识别与可解释诊断 API。

项目使用 PlantVillage 数据集中的苹果和番茄共 8 个类别，完成数据分析、图像分类训练、迁移学习对比和 FastAPI 推理接口。

## 当前进度

- [x] 建立独立的 Miniforge Python 3.11 环境
- [x] 安装 NumPy、Pandas、Matplotlib 和 JupyterLab
- [x] 初始化 Git 仓库并推送 GitHub
- [x] 数据审计、去重与分层划分
- [x] PyTorch Dataset、DataLoader 和 WeightedRandomSampler
- [x] 手写 BasicBlock 和 ResNet18
- [ ] 模型训练与验证
- [ ] 混淆矩阵和错误分析
- [ ] 迁移学习对比
- [ ] FastAPI 推理接口
- [ ] 项目文档与运行视频

## 数据集

原始数据保存在：

```text
data/raw/plantvillage/plantvillage dataset/color
```

项目使用的 8 个类别：

```text
Apple___Apple_scab
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___healthy
```

数据清单：

```text
data/processed/plantvillage8_manifest.csv
```

## 开发环境

- Python 3.11.16
- PyTorch 2.11.0 + CUDA 12.8
- torchvision 0.26.0
- NumPy 2.4.6
- Pandas 3.0.6
- Matplotlib 3.11.2
- NVIDIA GeForce RTX 5060 Laptop GPU

创建 Conda 环境：

```powershell
conda env create -f environment.yml
conda activate lingjing-ai
```

安装 CUDA 版 PyTorch：

```powershell
python -m pip install -r requirements-torch.txt
```

如果当前机器无法正常执行 `conda activate`，可以直接使用环境中的 Python 绝对路径：

```powershell
F:\Miniforge3\envs\lingjing-ai\python.exe
```

## 数据审计

```powershell
python src/prepare_dataset.py
python src/analyze_data.py
```

生成的数据审计和图表位于：

```text
reports/data_audit
reports/data_analysis
```

## 数据加载验证

```powershell
python -m src.check_dataloader
```

预期输出包含：

```text
一个批次的图片形状： torch.Size([32, 3, 224, 224])
标签形状： torch.Size([32])
```

## 手写 ResNet 验证

```powershell
python -m src.check_model
python -m pytest tests -q
```

模型输入形状：

```text
[B, 3, 224, 224]
```

模型输出形状：

```text
[B, 8]
```

## 目录结构

```text
.
├─ data/                 数据目录
├─ docs/                 项目说明与学习日志
├─ models/               模型权重，不提交大文件
├─ notebooks/            数据分析 Notebook
├─ reports/              图表与实验报告
├─ src/                  项目源代码
└─ tests/                自动化测试
```