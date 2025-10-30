# YOLO1D - High-Performance Time-Series Detection

## 🚀 Overview

YOLO1D is a 1D convolutional neural network inspired by the YOLOv8 architecture, designed specifically for classification and detection on time-series data. It extends YOLO's strong detection capability to one-dimensional signals, accurately localizing and classifying events or patterns in complex sequences.

This project has undergone a major refactor, introducing a unified trainer, configuration management, automatic mixed precision (AMP) training, and advanced data augmentation to provide a high-performance, easy-to-use, and extensible time-series detection framework.

## 📁 Project Structure

```
yolo1d/
├── trainer_base.py          # Base trainer
├── trainer_amp.py           # AMP trainer
├── train.py                 # Unified training script
├── data_augmentation.py     # Data augmentation module
├── test_improvements.py     # Test scripts
├── yolo1d_model.py          # Model definition
├── yolo1d_loss.py           # Loss functions
├── dataset_generator.py     # Dataset generation
├── inference_yolo1d.py      # Inference script
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## 🛠️ Quick Start

### 1. Environment Setup

```bash
# Create and activate a virtual environment
conda create --name yolo1d python=3.9 -y
conda activate yolo1d

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python dataset_generator.py
```

### 3. Train the Model

We provide a unified training script that supports starting from a config file, resuming from checkpoints, and setting a custom run name.

```bash
# Train with the default config
python train.py

# Specify a config file
python train.py --config config.yaml

# Resume from a checkpoint
python train.py --resume best_model.pth

# Set a custom run name
python train.py --run-name my_experiment
```

### 4. Inference

```bash
python inference_yolo1d.py --model best_model.pth
```

## 🧠 Key Features and Improvements

### 1. Unified Trainer Architecture
Modular design with a base class (`BaseTrainer`) and derived classes (`YOLO1DTrainer`, `YOLO1DAMPTrainer`) for high code reuse and easy extensibility.

### 2. Config-Driven Experiments
All model, training, and data parameters are centrally managed via `config.yaml`, making experiments reproducible and configurable.

### 3. Enhanced Augmentation for Time-Series
Rich augmentations tailored for time-domain data, such as time warping, frequency masking, and Mixup. Use presets (`light`, `medium`, `heavy`) or customize parameters.

### 4. Automatic Mixed Precision (AMP)
Integrated AMP can speed up training by 1.5–2x and reduce memory usage by ~50% with minimal accuracy loss.

### 5. Early Stopping
Automatically monitors validation performance to stop training when improvements plateau, preventing overfitting and saving compute.

### 6. Detailed Monitoring and Logging
Use TensorBoard to track losses, mAP, learning rate, and more, with structured log outputs.

```bash
tensorboard --logdir runs
```

## 📝 Configuration

Core settings live in `config.yaml`.

```yaml
# Model
model_size: 'n'          # model size (n, s, m, l, x)
num_classes: 2           # number of classes
input_channels: 1        # input channels
input_length: 1024       # input sequence length

# Data
dataset_path: 'sin_wave_dataset'
num_workers: 4

# Training
epochs: 100
batch_size: 16
learning_rate: 0.001
scheduler: 'cosine'      # LR scheduler ('cosine', 'onecycle', 'step')
patience: 10             # early stopping patience

# Loss hyperparameters
hyp:
  box: 1.0  # bbox loss weight
  cls: 3.0  # classification loss weight
  dfl: 0.8  # DFL loss weight
```

## 🎯 Model Performance

| Model Size | Params | Inference Speed | Relative mAP |
|------------|--------|-----------------|--------------|
| nano (n)   | 0.5M   | Fastest         | Baseline     |
| small (s)  | 2M     | Fast            | +5%          |
| medium (m) | 5M     | Medium          | +10%         |
| large (l)  | 10M    | Slow            | +15%         |
| xlarge (x) | 15M    | Slowest         | +20%         |

## 💡 Use Cases

- **Anomaly Detection**: Detect anomalies in industrial sensors, financial time series, etc.
- **Signal Analysis**: Identify and localize patterns in continuous signals (e.g., communications).
- **Biomedical**: Analyze ECG/EEG to detect arrhythmia or seizures.
- **Speech Processing**: Voice activity detection (VAD) or keyword spotting.
- **Industrial Monitoring**: Monitor equipment via vibration or acoustic signals.

## 🤝 Contributing

We welcome contributions of all kinds. Please follow the coding and commit guidelines.

### Coding Style
- Use type annotations
- Add clear and complete docstrings
- Follow PEP 8

### Commit Messages
```
feat: add new feature
fix: fix bug
docs: update documentation
style: code style adjustments
refactor: code refactor
test: add tests
```

## 📄 License

No License
