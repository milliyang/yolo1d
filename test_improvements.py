#!/usr/bin/env python3
"""
测试项目改进效果
验证新的训练器、配置管理和数据增强功能
"""

import torch
import torch.utils.data as data
import numpy as np
import time
import sys
import os
from pathlib import Path

# 导入改进的模块
from trainer_base import YOLO1DTrainer, ConfigManager, YOLO1DError
from trainer_amp import YOLO1DAMPTrainer
from data_augmentation import create_augmentation, AdvancedDataAugmentation
from dataset_generator import SinWaveDataset, collate_fn


def test_config_manager():
    """测试配置管理器"""
    print("🧪 测试配置管理器...")
    
    try:
        # 测试正常配置
        config_manager = ConfigManager('config.yaml')
        config = config_manager.get_training_config()
        print(f"✅ 配置加载成功: {len(config)} 个参数")
        
        # 测试配置验证
        config_manager.validate_config()
        print("✅ 配置验证通过")
        
        # 测试配置更新
        config_manager.update_config({'learning_rate': 0.002})
        print("✅ 配置更新成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置管理器测试失败: {e}")
        return False


def test_data_augmentation():
    """测试数据增强"""
    print("\n🧪 测试数据增强...")
    
    try:
        # 创建测试数据
        test_data = torch.randn(4, 1, 1024)  # [batch, channels, length]
        
        # 测试不同强度的增强
        for strength in ['light', 'medium', 'heavy']:
            print(f"  测试 {strength} 增强...")
            augmentation = create_augmentation(strength)
            
            # 应用增强
            augmented_data = augmentation(test_data)
            
            # 检查数据形状
            assert augmented_data.shape == test_data.shape, f"数据形状不匹配: {augmented_data.shape} vs {test_data.shape}"
            
            # 检查数据范围（应该合理）
            assert torch.isfinite(augmented_data).all(), "增强后数据包含无效值"
            
            print(f"  ✅ {strength} 增强测试通过")
        
        # 测试自定义配置
        custom_config = {
            'noise_std_range': (0.01, 0.05),
            'noise_prob': 1.0,  # 100%应用噪声
            'scale_prob': 0.0   # 不应用缩放
        }
        custom_aug = create_augmentation(custom_config=custom_config)
        custom_result = custom_aug(test_data)
        
        # 验证噪声确实被添加了
        noise_diff = torch.abs(custom_result - test_data).mean()
        assert noise_diff > 0, "噪声增强未生效"
        print("  ✅ 自定义增强测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据增强测试失败: {e}")
        return False


def create_data_loaders(config):
    """创建数据加载器"""
    print("📊 创建数据加载器...")
    
    # 数据集路径
    dataset_path = config['dataset_path']
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集不存在: {dataset_path}")
        print("请先运行 dataset_generator.py 生成数据集")
        sys.exit(1)
    
    # 创建训练集
    train_dataset = SinWaveDataset(
        dataset_path=dataset_path,
        split='train',
        input_length=config['input_length']
    )
    
    # 创建验证集
    val_dataset = SinWaveDataset(
        dataset_path=dataset_path,
        split='val',
        input_length=config['input_length']
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


def test_trainer_creation():
    """测试训练器创建"""
    print("\n🧪 测试训练器创建...")
    
    try:
        # 加载配置
        config_manager = ConfigManager('config.yaml')
        config = config_manager.get_training_config()
        
        # 创建小规模数据集用于测试
        test_dataset = SinWaveDataset(
            dataset_path='sin_wave_dataset',
            split='train',
            input_length=config['input_length']
        )
        
        # 限制数据集大小以加快测试
        test_dataset.signals = test_dataset.signals[:10]
        test_dataset.labels = test_dataset.labels[:10]
        
        # 创建数据加载器 - 使用自定义collate_fn
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn  # 使用自定义collate函数
        )
        
        # 测试普通训练器
        print("  测试普通训练器...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        trainer = YOLO1DTrainer(
            train_loader=test_loader,
            val_loader=test_loader,
            config=config,
            device=device
        )
        print("  ✅ 普通训练器创建成功")
        
        # 测试AMP训练器
        if torch.cuda.is_available():
            print("  测试AMP训练器...")
            amp_trainer = YOLO1DAMPTrainer(
                train_loader=test_loader,
                val_loader=test_loader,
                config=config,
                device=device
            )
            print("  ✅ AMP训练器创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练器创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试训练步骤"""
    print("\n🧪 测试训练步骤...")
    
    try:
        # 加载配置
        config_manager = ConfigManager('config.yaml')
        config = config_manager.get_training_config()
        
        # 修改配置以加快测试
        config['epochs'] = 1
        config['batch_size'] = 2
        
        # 创建小规模数据集
        test_dataset = SinWaveDataset(
            dataset_path='sin_wave_dataset',
            split='train',
            input_length=config['input_length']
        )
        
        # 限制数据集大小
        test_dataset.signals = test_dataset.signals[:4]
        test_dataset.labels = test_dataset.labels[:4]
        
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn  # 使用自定义collate函数
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试普通训练器的一个epoch
        print("  测试普通训练器训练步骤...")
        trainer = YOLO1DTrainer(
            train_loader=test_loader,
            val_loader=test_loader,
            config=config,
            device=device
        )
        
        # 测试一个训练步骤
        start_time = time.time()
        train_loss = trainer.train_epoch(0)
        train_time = time.time() - start_time
        
        print(f"  ✅ 训练步骤完成，损失: {train_loss:.4f}, 时间: {train_time:.2f}s")
        
        # 测试验证步骤
        start_time = time.time()
        val_loss, mAP = trainer.validate(0)
        val_time = time.time() - start_time
        
        print(f"  ✅ 验证步骤完成，损失: {val_loss:.4f}, mAP: {mAP:.4f}, 时间: {val_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_save_load():
    """测试检查点保存和加载"""
    print("\n🧪 测试检查点保存和加载...")
    
    try:
        # 加载配置
        config_manager = ConfigManager('config.yaml')
        config = config_manager.get_training_config()
        
        # 修改配置
        config['epochs'] = 1
        config['batch_size'] = 2
        
        # 创建小规模数据集
        test_dataset = SinWaveDataset(
            dataset_path='sin_wave_dataset',
            split='train',
            input_length=config['input_length']
        )
        test_dataset.signals = test_dataset.signals[:4]
        test_dataset.labels = test_dataset.labels[:4]
        
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn  # 使用自定义collate函数
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建训练器
        trainer = YOLO1DTrainer(
            train_loader=test_loader,
            val_loader=test_loader,
            config=config,
            device=device
        )
        
        # 保存检查点
        print("  保存检查点...")
        trainer.save_checkpoint(0, is_best=True)
        
        # 验证文件存在
        assert os.path.exists("best_model.pth"), "最佳模型文件未创建"
        assert os.path.exists("checkpoint_epoch_0.pth"), "检查点文件未创建"
        print("  ✅ 检查点保存成功")
        
        # 创建新的训练器并加载检查点
        print("  加载检查点...")
        new_trainer = YOLO1DTrainer(
            train_loader=test_loader,
            val_loader=test_loader,
            config=config,
            device=device,
            resume_from="best_model.pth"
        )
        
        print("  ✅ 检查点加载成功")
        
        # 清理测试文件
        if os.path.exists("best_model.pth"):
            os.remove("best_model.pth")
        if os.path.exists("checkpoint_epoch_0.pth"):
            os.remove("checkpoint_epoch_0.pth")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查点测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n🧪 测试错误处理...")
    
    try:
        # 测试无效配置文件
        print("  测试无效配置文件...")
        try:
            ConfigManager('nonexistent_config.yaml')
            print("  ❌ 应该抛出异常")
            return False
        except YOLO1DError:
            print("  ✅ 正确处理了无效配置文件")
        
        # 测试无效配置参数
        print("  测试无效配置参数...")
        try:
            config_manager = ConfigManager('config.yaml')
            config_manager.update_config({'learning_rate': -1})
            print("  ❌ 应该抛出异常")
            return False
        except YOLO1DError:
            print("  ✅ 正确处理了无效配置参数")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False


def benchmark_performance():
    """性能基准测试"""
    print("\n🧪 性能基准测试...")
    
    try:
        # 加载配置
        config_manager = ConfigManager('config.yaml')
        config = config_manager.get_training_config()
        
        # 修改配置
        config['epochs'] = 1
        config['batch_size'] = 4
        
        # 创建数据集
        test_dataset = SinWaveDataset(
            dataset_path='sin_wave_dataset',
            split='train',
            input_length=config['input_length']
        )
        test_dataset.signals = test_dataset.signals[:8]
        test_dataset.labels = test_dataset.labels[:8]
        
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn  # 使用自定义collate函数
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试普通训练器性能
        print("  测试普通训练器性能...")
        trainer = YOLO1DTrainer(
            train_loader=test_loader,
            val_loader=test_loader,
            config=config,
            device=device
        )
        
        start_time = time.time()
        train_loss = trainer.train_epoch(0)
        normal_time = time.time() - start_time
        
        print(f"  ✅ 普通训练器: {normal_time:.2f}s")
        
        # 测试AMP训练器性能
        if torch.cuda.is_available():
            print("  测试AMP训练器性能...")
            amp_trainer = YOLO1DAMPTrainer(
                train_loader=test_loader,
                val_loader=test_loader,
                config=config,
                device=device
            )
            
            start_time = time.time()
            amp_train_loss = amp_trainer.train_epoch(0)
            amp_time = time.time() - start_time
            
            print(f"  ✅ AMP训练器: {amp_time:.2f}s")
            
            # 计算加速比
            if normal_time > 0:
                speedup = normal_time / amp_time
                print(f"  📊 AMP加速比: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能基准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🎯 YOLO1D 项目改进测试")
    print("=" * 50)
    
    # 检查数据集是否存在
    if not os.path.exists('sin_wave_dataset'):
        print("❌ 数据集不存在，请先运行 dataset_generator.py")
        return False
    
    tests = [
        ("配置管理器", test_config_manager),
        ("数据增强", test_data_augmentation),
        ("训练器创建", test_trainer_creation),
        ("训练步骤", test_training_step),
        ("检查点保存加载", test_checkpoint_save_load),
        ("错误处理", test_error_handling),
        ("性能基准", benchmark_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目改进成功！")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步检查")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 