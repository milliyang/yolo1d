#!/usr/bin/env python3
"""
优化的YOLO1D训练脚本
结合train_simple.py的成功经验，提升训练效果
"""

import torch
import torch.utils.data as data
import argparse
import sys
import os
from pathlib import Path
import numpy as np

from trainer_base import YOLO1DTrainer, ConfigManager, YOLO1DError
from dataset_generator import SinWaveDataset, collate_fn


class DataAugmentation:
    """数据增强类 - 从train_simple.py移植"""
    def __init__(self, noise_std=0.02, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.scale_range = scale_range
    
    def __call__(self, signal):
        # 添加噪声
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, signal.shape)
            signal = signal + noise
        
        # 缩放
        if self.scale_range:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            signal = signal * scale
        
        return signal


class AugmentedSinWaveDataset(SinWaveDataset):
    """支持数据增强的SinWaveDataset"""
    def __init__(self, dataset_path, split='train', input_length=1024, transform=None):
        super().__init__(dataset_path, split, input_length, transform)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        
        if self.transform:
            signal = self.transform(signal)
        
        # 转换为张量
        signal = torch.FloatTensor(signal).unsqueeze(0)  # 添加通道维度 [1, sequence_length]
        
        # 处理标签 - 转换为YOLO格式 [image_idx, class, x_center, width]
        sample_labels = self.labels[idx]
        if sample_labels:
            targets = []
            for label in sample_labels:
                class_id, x_center, width = label
                targets.append([idx, class_id, x_center, width])
            targets = torch.FloatTensor(targets)
        else:
            targets = torch.zeros(0, 4)  # 空标签
        
        return signal, targets


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("💻 使用CPU")
    
    return device


def create_data_loaders(config):
    """创建数据加载器 - 支持数据增强"""
    print("📊 创建数据加载器...")
    
    # 数据集路径
    dataset_path = config['dataset_path']
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集不存在: {dataset_path}")
        print("请先运行 dataset_generator.py 生成数据集")
        sys.exit(1)
    
    # 数据增强配置
    data_aug_config = config.get('data_augmentation', {})
    use_augmentation = data_aug_config.get('enabled', False)
    
    if use_augmentation:
        print("🔧 启用数据增强")
        train_transform = DataAugmentation(
            noise_std=data_aug_config.get('noise_std', 0.02),
            scale_range=tuple(data_aug_config.get('scale_range', [0.9, 1.1]))
        )
        val_transform = None
    else:
        train_transform = None
        val_transform = None
    
    # 创建训练集
    train_dataset = AugmentedSinWaveDataset(
        dataset_path=dataset_path,
        split='train',
        input_length=config['input_length'],
        transform=train_transform
    )
    
    # 创建验证集
    val_dataset = AugmentedSinWaveDataset(
        dataset_path=dataset_path,
        split='val',
        input_length=config['input_length'],
        transform=val_transform
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器 - 使用自定义collate_fn
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn  # 使用自定义collate函数
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn  # 使用自定义collate函数
    )
    
    return train_loader, val_loader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化的YOLO1D训练脚本')
    parser.add_argument('--config', type=str, default='config_optimized.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--run-name', type=str, default=None,
                       help='实验运行名称')
    
    args = parser.parse_args()
    
    try:
        print("🎯 优化的YOLO1D训练脚本")
        print("=" * 50)
        
        # 1. 加载配置
        print(f"📋 加载配置: {args.config}")
        config_manager = ConfigManager(args.config)
        config = config_manager.get_training_config()
        
        # 更新运行名称
        if args.run_name:
            config['run_name'] = args.run_name
        
        print(f"模型尺寸: {config['model_size']}")
        print(f"类别数: {config['num_classes']}")
        print(f"输入长度: {config['input_length']}")
        print(f"训练轮数: {config['epochs']}")
        print(f"批大小: {config['batch_size']}")
        print(f"学习率: {config['learning_rate']}")
        
        # 显示损失权重
        hyp = config.get('hyp', {})
        print(f"损失权重: box={hyp.get('box', 7.5)}, cls={hyp.get('cls', 0.5)}, dfl={hyp.get('dfl', 1.5)}")
        
        # 显示数据增强状态
        data_aug_config = config.get('data_augmentation', {})
        if data_aug_config.get('enabled', False):
            print(f"数据增强: 启用 (噪声={data_aug_config.get('noise_std', 0.02)}, 缩放={data_aug_config.get('scale_range', [0.9, 1.1])})")
        else:
            print("数据增强: 禁用")
        
        # 2. 设置设备
        device = setup_device()
        
        # 3. 创建数据加载器
        train_loader, val_loader = create_data_loaders(config)
        
        # 4. 创建训练器
        print("🏗️ 创建训练器...")
        trainer = YOLO1DTrainer(
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            resume_from=args.resume
        )
        
        # 5. 开始训练
        print("🚀 开始训练...")
        trainer.train()
        
        print("✅ 训练完成！")
        
    except YOLO1DError as e:
        print(f"❌ YOLO1D错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 