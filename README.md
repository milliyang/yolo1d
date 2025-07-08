# YOLO1D - 高性能时域数据检测模型

## 🚀 项目简介

YOLO1D 是一个基于 YOLOv8 架构思想设计的1D卷积神经网络模型，专门用于时域数据的分类与检测任务。它将YOLO强大的检测能力扩展到一维时间序列数据，能够在复杂的信号中准确定位和分类感兴趣的事件或模式。

本项目经过了大规模重构，引入了统一的训练器、配置管理、混合精度训练和高级数据增强等功能，旨在提供一个高性能、易于使用和扩展的时域检测框架。

## 📁 项目结构

```
yolo1d/
├── trainer_base.py          # 训练器基类
├── trainer_amp.py           # 混合精度训练器
├── train.py                 # 统一训练脚本
├── data_augmentation.py     # 数据增强模块
├── test_improvements.py     # 测试脚本
├── yolo1d_model.py          # 模型定义
├── yolo1d_loss.py           # 损失函数
├── dataset_generator.py     # 数据集生成
├── inference_yolo1d.py      # 推理脚本
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖包
└── README.md                # 项目说明文档
```

## 🛠️ 快速开始

### 1. 环境设置

```bash
# 创建并激活虚拟环境
conda create --name yolo1d python=3.9 -y
conda activate yolo1d

# 安装依赖
pip install -r requirements.txt
```

### 2. 生成数据集

```bash
python dataset_generator.py
```

### 3. 训练模型

我们提供了一个统一的训练脚本，支持从配置文件启动、恢复训练和自定义实验名称。

```bash
# 使用默认配置进行训练
python train.py

# 指定配置文件
python train.py --config config.yaml

# 从断点恢复训练
python train.py --resume best_model.pth

# 指定实验运行名称
python train.py --run-name my_experiment
```

### 4. 推理预测

```bash
python inference_yolo1d.py --model best_model.pth
```

## 🧠 核心特性与改进

### 1. 统一训练器架构
通过模块化的基类 (`BaseTrainer`) 和子类 (`YOLO1DTrainer`, `YOLO1DAMPTrainer`) 设计，实现了代码的高度复用和功能的轻松扩展。

### 2. 配置文件系统
所有模型、训练和数据相关的参数都通过 `config.yaml` 集中管理，使实验配置和复现变得简单。

### 3. 增强的数据增强
内置了专为时域数据设计的丰富数据增强方法，如时间扭曲、频域掩码、Mixup等，可通过配置 (`light`, `medium`, `heavy`) 或自定义参数调用。

### 4. 混合精度训练 (AMP)
集成了自动混合精度训练，可将训练速度提升1.5-2倍，显存使用减少约50%，而几乎不损失精度。

### 5. 早停机制
自动监控验证集性能，在模型停止改进时提前终止训练，防止过拟合，节约计算资源。

### 6. 详细的监控与日志
使用 TensorBoard 记录训练过程中的损失、mAP、学习率等关键指标，并提供结构化的日志输出。

```bash
tensorboard --logdir runs
```

## 📝 配置说明

核心配置位于 `config.yaml` 文件中。

```yaml
# 模型配置
model_size: 'n'          # 模型尺寸 (n, s, m, l, x)
num_classes: 2           # 类别数
input_channels: 1        # 输入通道数
input_length: 1024       # 输入序列长度

# 数据配置
dataset_path: 'sin_wave_dataset'
num_workers: 4

# 训练配置
epochs: 100
batch_size: 16
learning_rate: 0.001
scheduler: 'cosine'      # 学习率调度器 ('cosine', 'onecycle', 'step')
patience: 10             # 早停耐心值

# 损失函数超参数
hyp:
  box: 1.0  # 边界框损失权重
  cls: 3.0  # 分类损失权重
  dfl: 0.8  # DFL损失权重
```

## 🎯 模型性能

| 模型尺寸 | 参数量 | 推理速度 | 相对mAP |
|----------|--------|----------|---------|
| nano (n) | 0.5M   | 最快     | 基准    |
| small (s)| 2M     | 快       | +5%     |
| medium(m)| 5M     | 中等     | +10%    |
| large (l)| 10M    | 慢       | +15%    |
| xlarge(x)| 15M    | 最慢     | +20%    |

## 💡 应用场景

- **异常检测**: 在工业传感器、金融交易等时间序列中检测异常事件。
- **信号分析**: 从连续信号中识别和定位特定的模式（如通信信号）。
- **生物医学**: 分析ECG、EEG等生理信号，检测心律失常或癫痫发作。
- **语音处理**: 实现语音活动检测（VAD）或关键字识别。
- **工业监控**: 根据振动或声音信号监测设备状态。

## 🔮 未来计划

我们为项目制定了详细的未来开发路线图，包括添加注意力机制、模型量化、TensorRT支持和多GPU训练等。详情请参阅 `PROJECT_IMPROVEMENTS.md`。

## 🤝 贡献指南

我们欢迎任何形式的贡献！请遵循代码和提交规范。

### 代码规范
- 使用类型注解
- 添加详细的文档字符串
- 遵循PEP 8规范

### 提交规范
```
feat: 添加新功能
fix: 修复bug
docs: 更新文档
style: 代码格式调整
refactor: 代码重构
test: 添加测试
```

## �� 许可证

No License 