#!/usr/bin/env python3
"""
测试优化效果
对比原始train.py和优化版本的性能差异
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import yaml
import os
import time

from trainer_base import YOLO1DTrainer
from yolo1d_model import create_yolo1d_model
from dataset_generator import SinWaveDataset, SinWaveDatasetGenerator, collate_fn


def create_test_config():
    """创建测试配置"""
    config = {
        'model_size': 'n',
        'num_classes': 2,
        'input_channels': 1,
        'input_length': 1024,
        'epochs': 5,  # 减少epoch数用于快速测试
        'batch_size': 8,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'dataset_path': 'test_dataset',
        'num_workers': 2,
        'run_name': 'test_optimization',
        'patience': 5,
        'min_delta': 0.001,
        'grad_clip': 1.0,
        'reg_max': 16
    }
    return config


def create_test_dataset():
    """创建测试数据集"""
    print("📊 创建测试数据集...")
    
    # 创建数据集生成器
    generator = SinWaveDatasetGenerator(
        num_samples=50,
        output_dir='test_dataset'
    )
    
    # 生成数据
    generator.generate_and_label_data()
    generator.split_and_save(train_split=0.7)
    generator.save_dataset()
    
    return 'test_dataset'


def test_original_config():
    """测试原始配置"""
    print("\n🔧 测试原始配置 (config.yaml)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = create_test_config()
    
    # 原始损失权重
    config['hyp'] = {
        'box': 1.0,
        'cls': 3.0,
        'dfl': 0.8
    }
    
    # 原始调度器 - 修复为字典格式
    config['scheduler'] = {
        'type': 'cosine',
        'eta_min': 0.00001
    }
    
    # 无数据增强
    config['data_augmentation'] = {'enabled': False}
    
    # 创建数据集
    dataset_path = create_test_dataset()
    
    # 创建数据集实例
    train_dataset = SinWaveDataset(
        dataset_path=dataset_path,
        split='train',
        input_length=config['input_length']
    )
    
    val_dataset = SinWaveDataset(
        dataset_path=dataset_path,
        split='val',
        input_length=config['input_length']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 测试训练器
    try:
        trainer = YOLO1DTrainer(
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        # 运行一个epoch的验证
        start_time = time.time()
        val_loss, mAP = trainer.validate(0)
        end_time = time.time()
        
        print(f"原始配置结果:")
        print(f"  - Val Loss: {val_loss:.4f}")
        print(f"  - mAP: {mAP:.4f}")
        print(f"  - 验证时间: {end_time - start_time:.2f}秒")
        
        return val_loss, mAP
        
    except Exception as e:
        print(f"❌ 原始配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_optimized_config():
    """测试优化配置"""
    print("\n⚡ 测试优化配置 (config_optimized.yaml)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = create_test_config()
    
    # 优化损失权重 (train_simple.py的成功配置)
    config['hyp'] = {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5
    }
    
    # 优化调度器
    config['scheduler'] = {
        'type': 'onecycle',
        'max_lr': 0.001
    }
    
    # 启用数据增强
    config['data_augmentation'] = {
        'enabled': True,
        'noise_std': 0.02,
        'scale_range': [0.9, 1.1]
    }
    
    # 创建数据集
    dataset_path = 'test_dataset'  # 使用已创建的数据集
    
    # 创建数据集实例
    train_dataset = SinWaveDataset(
        dataset_path=dataset_path,
        split='train',
        input_length=config['input_length']
    )
    
    val_dataset = SinWaveDataset(
        dataset_path=dataset_path,
        split='val',
        input_length=config['input_length']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 测试训练器
    try:
        trainer = YOLO1DTrainer(
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        # 运行一个epoch的验证
        start_time = time.time()
        val_loss, mAP = trainer.validate(0)
        end_time = time.time()
        
        print(f"优化配置结果:")
        print(f"  - Val Loss: {val_loss:.4f}")
        print(f"  - mAP: {mAP:.4f}")
        print(f"  - 验证时间: {end_time - start_time:.2f}秒")
        
        return val_loss, mAP
        
    except Exception as e:
        print(f"❌ 优化配置测试失败: {e}")
        return None, None


def cleanup():
    """清理测试文件"""
    import shutil
    if os.path.exists('test_dataset'):
        shutil.rmtree('test_dataset')
    print("🧹 清理测试文件完成")


if __name__ == "__main__":
    print("🚀 开始优化效果测试")
    
    try:
        # 测试原始配置
        orig_loss, orig_map = test_original_config()
        
        # 测试优化配置
        opt_loss, opt_map = test_optimized_config()
        
        # 对比结果
        if orig_loss is not None and opt_loss is not None:
            print("\n📊 对比结果:")
            print(f"原始配置: Val Loss={orig_loss:.4f}, mAP={orig_map:.4f}")
            print(f"优化配置: Val Loss={opt_loss:.4f}, mAP={opt_map:.4f}")
            
            loss_improvement = (orig_loss - opt_loss) / orig_loss * 100
            map_improvement = (opt_map - orig_map) / max(orig_map, 0.001) * 100
            
            print(f"\n🎯 改进效果:")
            print(f"  损失改进: {loss_improvement:+.2f}%")
            print(f"  mAP改进: {map_improvement:+.2f}%")
            
            if opt_map > orig_map:
                print("✅ 优化配置表现更好！")
            else:
                print("⚠️ 优化配置需要进一步调整")
        else:
            print("❌ 测试失败，无法对比结果")
    
    finally:
        # 清理测试文件
        cleanup() 