# 灵境竞赛组后端 AI 方向第一阶段项目

## 项目方向

基于手写 ResNet 的植物叶片病害识别与可解释诊断 API。

项目使用 PlantVillage 数据集中苹果和番茄共 8 个类别，完成数据分析、手写 ResNet、迁移学习对比、错误分析、Grad-CAM 和 FastAPI 推理接口。

## 当前进度

- [x] 建立独立的 Miniforge Python 3.11 环境
- [x] 安装 PyTorch 2.11.0 + CUDA 12.8
- [x] 初始化 Git 仓库并推送 GitHub
- [x] 数据审计、去重、近重复检测与分层划分
- [x] PyTorch Dataset、DataLoader 和 WeightedRandomSampler
- [x] 手写 BasicBlock 和 ResNet18
- [x] 手写 ResNet18 完整训练与测试集评估
- [x] 错误样本分析
- [x] 迁移学习 ResNet18 对比实验
- [x] Grad-CAM 可解释性可视化
- [x] FastAPI 推理接口
- [ ] 运行视频
- [ ] 个人总结与答辩稿

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

数据划分：

```text
训练集：5996
验证集：1284
测试集：1291
```

## 开发环境

- Python 3.11.16
- PyTorch 2.11.0 + CUDA 12.8
- torchvision 0.26.0
- NumPy 2.4.6
- Pandas 3.0.6
- Matplotlib 3.11.2
- FastAPI 0.143.0
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

如果当前机器无法正常执行 `conda activate`，可以直接使用：

```powershell
F:\Miniforge3\envs\lingjing-ai\python.exe
```

## 数据准备

```powershell
python src/prepare_dataset.py
python src/analyze_data.py
```

数据审计和图表：

```text
reports/data_audit
reports/data_analysis
```

## 数据加载验证

```powershell
python -m src.check_dataloader
```

## 手写 ResNet

模型结构：

```text
BasicBlock + downsample + ResNet18
输入：[B, 3, 224, 224]
输出：[B, 8]
参数量：11,180,616
```

验证：

```powershell
python -m src.check_model
python -m pytest tests -q
```

训练：

```powershell
python -m src.train --run-name resnet18 --epochs 25
```

手写 ResNet 测试结果：

```text
Accuracy：0.9806
Macro-F1：0.9837
最佳轮次：23
```

## 迁移学习

```powershell
python -m src.transfer_learning --run-name transfer_resnet18 --epochs 12
```

迁移学习测试结果：

```text
Accuracy：0.9930
Macro-F1：0.9936
最佳轮次：2
预训练权重：IMAGENET1K_V1
```

## 错误分析

```powershell
python -m src.analyze_errors
```

测试集共有 1291 张图片，其中 25 张误判，主要错误为：

```text
Tomato___Late_blight -> Tomato___Early_blight：13 张
```

错误分析结果：

```text
reports/error_analysis/resnet18
```

## 模型对比

```powershell
python -m src.compare_models
```

对比报告：

```text
reports/comparison/model_comparison.md
```

## Grad-CAM

```powershell
python -m src.gradcam
```

输出：

```text
reports/gradcam/transfer_resnet18
```

## FastAPI 推理接口

启动服务：

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

接口：

```text
GET  /health
GET  /model-info
POST /predict
```

`/predict` 接收图片文件和 `top_k` 参数，返回中文类别名称和 Top-K 概率。

接口自动测试：

```powershell
python -m src.check_api
```

## 单张图片预测

```powershell
python -m src.predict --image "图片路径" --top-k 3
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