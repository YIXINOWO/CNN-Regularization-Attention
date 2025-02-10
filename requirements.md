# 项目依赖库

本项目使用 Python 3.10.16 进行开发，以下是主要的依赖库及其版本：

## 深度学习框架
- PyTorch == 2.0.1
- torchvision == 0.15.2
- torchaudio == 2.0.2
- pytorch-cuda == 11.8

## 数据处理和科学计算
- NumPy == 1.24.3
- Pandas == 2.0.3
- scikit-learn == 1.3.0  # 用于数据集分割

## 数据可视化
- Matplotlib == 3.7.2

## Jupyter相关
- ipykernel
- jupyter

## 工具库
- tqdm == 4.66.1
- kaggle (待安装)

## 安装命令

### PyTorch安装
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Jupyter相关安装
```bash
conda install -n pytorch ipykernel --update-deps --force-reinstall
# 或者
pip install ipykernel
python -m ipykernel install --user --name pytorch --display-name "Python (pytorch)"
```

### Kaggle安装
```bash
pip install kaggle
```

## 环境验证代码
```python
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# 打印各个库的版本
print(f"Python版本: {platform.python_version()}")
print(f"PyTorch版本: {torch.__version__}")
print(f"NumPy版本: {np.__version__}")
print(f"Pandas版本: {pd.__version__}")
print(f"Matplotlib版本: {plt.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
```

## 注意事项
1. 安装完 kaggle 后，需要配置 API 凭证
2. 将 kaggle.json 文件放在 `C:\Users\<Windows-username>\.kaggle\` 目录下
3. 确保使用兼容的 CUDA 版本（当前使用 CUDA 11.8）
4. 确保已安装 ipykernel 以运行 Jupyter notebook

## 数据集获取

### 方式1：Kaggle Dogs vs Cats 数据集
1. 访问 https://www.kaggle.com/c/dogs-vs-cats
2. 接受比赛规则
3. 运行下载命令：
```bash
kaggle competitions download -c dogs-vs-cats
```

### 方式2：Microsoft Cats and Dogs Dataset
```bash
wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
```

### 方式3：较小的测试数据集
```bash
kaggle datasets download -d tongpython/cat-and-dog
```

## 数据预处理
1. 解压数据：
```bash
unzip dogs-vs-cats.zip
unzip train.zip -d data/
```

2. 运行数据组织脚本：
```bash
python organize_data.py
```

这将创建训练集（80%）和验证集（20%）的标准目录结构