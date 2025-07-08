import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from yolo1d_model import create_yolo1d_model
from yolo1d_loss import YOLO1DLoss, compute_loss
from dataset_generator import load_generated_dataset
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision
from utils import plot_training_curves, plot_validation_predictions, postprocess_for_metrics
import copy
from pathlib import Path


class SinWaveDataset(Dataset):
    """Sin波峰检测数据集类"""
    def __init__(self, signals, labels, transform=None):
        """
        Args:
            signals: 信号数据数组 [N, sequence_length]
            labels: 标签列表 [N, [class_id, x_center, width]]
            transform: 数据变换
        """
        self.signals = signals
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        
        if self.transform:
            signal = self.transform(signal)
        
        # 转换为张量
        signal = torch.FloatTensor(signal).unsqueeze(0)  # 添加通道维度 [1, sequence_length]
        
        # 处理标签 - 转换为YOLO格式 [image_idx, class, x_center, width]
        sample_labels = self.labels[idx]
        if sample_labels:
            # 添加image_idx(在collate_fn中会被重新设置)
            targets = []
            for label in sample_labels:
                class_id, x_center, width = label
                targets.append([idx, class_id, x_center, width])
            targets = torch.FloatTensor(targets)
        else:
            targets = torch.zeros(0, 4)  # 空标签
        
        return signal, targets


def collate_fn(batch):
    """自定义批处理函数"""
    signals, targets_list = zip(*batch)
    
    # 堆叠信号
    signals = torch.stack(signals)
    
    # 处理标签 - 添加batch索引
    targets = []
    for i, target in enumerate(targets_list):
        if target.shape[0] > 0:
            # 更新batch索引
            target[:, 0] = i
            targets.append(target)
    
    if targets:
        targets = torch.cat(targets, 0)
    else:
        targets = torch.zeros(0, 4)
    
    return signals, targets


class DataAugmentation:
    """数据增强类"""
    def __init__(self, noise_std=0.05, scale_range=(0.8, 1.2)):
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


class Trainer:
    """训练器类"""
    def __init__(self, model, train_loader, val_loader, device, config, resume_from=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.start_epoch = 0  # 添加起始epoch
        
        # 修复步长问题
        self.fix_stride_issue()
        
        # 定义损失超参数
        hyp = config.get('hyp', {})
        print(f"Using loss hyperparameters: {hyp}")

        # 损失函数
        self.criterion = YOLO1DLoss(
            num_classes=config['num_classes'],
            reg_max=config.get('reg_max', 16),
            use_dfl=True,
            hyp=hyp
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader)
        )
        
        # 日志和指标
        self.log_dir = Path(f"runs/{config.get('run_name', 'experiment')}")
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.weights_dir = self.log_dir / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        self.map_metric = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox'
        )
        
        self.train_losses = []
        self.val_losses = []
        self.mAPs = []
        self.best_map = 0.0
        
        # 如果指定了恢复检查点，则加载
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def fix_stride_issue(self):
        """修复步长问题"""
        print("🔧 修复步长问题")
        
        # 根据输入长度计算正确的步长
        input_length = self.config['input_length']
        
        # 典型的YOLO步长配置
        if input_length == 1024:
            correct_strides = [8, 16, 32]  # 常见的步长配置
        else:
            # 根据输入长度动态计算
            correct_strides = [input_length // 128, input_length // 64, input_length // 32]
        
        print(f"  输入长度: {input_length}")
        print(f"  计算出的正确步长: {correct_strides}")
        
        # 修复步长
        with torch.no_grad():
            for i, stride in enumerate(correct_strides):
                self.model.detect.stride[i] = float(stride)
        
        print(f"  修复后的步长: {self.model.detect.stride}")
    
    def load_checkpoint(self, checkpoint_path):
        """从检查点恢复训练状态，支持绝对路径和相对路径。"""
        resume_path = Path(checkpoint_path)
        if not resume_path.is_file():
            print(f"❌ 错误: 检查点文件不存在或不是一个文件 -> {resume_path}")
            return False
            
        try:
            print(f"📂 正在从检查点恢复训练: {resume_path.resolve()}")
            checkpoint = torch.load(resume_path, map_location=self.device)
            
            # 加载模型状态
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ 模型状态已加载")
            
            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ 优化器状态已加载")
            
            # 加载学习率调度器状态
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ 学习率调度器状态已加载")
            
            # 加载训练历史
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
                print(f"✓ 训练损失历史已加载 (共{len(self.train_losses)}个epoch)")
            
            if 'val_losses' in checkpoint:
                self.val_losses = checkpoint['val_losses']
                print(f"✓ 验证损失历史已加载 (共{len(self.val_losses)}个epoch)")
            
            if 'mAPs' in checkpoint:
                self.mAPs = checkpoint['mAPs']
                print(f"✓ mAP历史已加载 (共{len(self.mAPs)}个epoch)")
            
            # 加载最佳mAP
            if 'best_map' in checkpoint:
                self.best_map = checkpoint['best_map']
                print(f"✓ 最佳mAP已加载: {self.best_map:.4f}")
            
            # 设置起始epoch
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"✓ 将从第 {self.start_epoch} 个epoch开始训练")
            
            # 验证配置一致性
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                current_config = self.config
                
                # 检查关键配置是否一致
                key_configs = ['model_size', 'num_classes', 'input_channels', 'input_length']
                config_mismatch = []
                for key in key_configs:
                    if key in saved_config and key in current_config:
                        if saved_config[key] != current_config[key]:
                            config_mismatch.append(f"{key}: {saved_config[key]} -> {current_config[key]}")
                
                if config_mismatch:
                    print("⚠️  配置不匹配:")
                    for mismatch in config_mismatch:
                        print(f"    {mismatch}")
                    print("建议使用与检查点相同的配置进行恢复训练")
                else:
                    print("✓ 配置一致性检查通过")
            
            print(f"✓ 检查点恢复完成！将从第 {self.start_epoch} 个epoch继续训练")
            return True
            
        except Exception as e:
            print(f"❌ 加载检查点失败: {str(e)}")
            return False
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} - Training')
        for batch_idx, (signals, targets) in enumerate(pbar):
            signals = signals.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            preds = self.model(signals)
            
            # 计算损失
            loss, loss_items = compute_loss(preds, targets, self.model, self.criterion)
            
            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}: {loss}")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录损失
            epoch_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{loss_items[0]:.4f}',
                'cls': f'{loss_items[1]:.4f}',
                'dfl': f'{loss_items[2]:.4f}',
            })
            
            # 记录到TensorBoard (每个step)
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/train_step', loss.item(), global_step)
            self.writer.add_scalar('LR/step', self.optimizer.param_groups[0]['lr'], global_step)

        return epoch_loss / len(self.train_loader)
    
    def validate(self, epoch):
        """验证, 计算损失和mAP"""
        self.model.eval()
        val_loss = 0.0
        
        preds_for_map = []
        targets_for_map = []
        
        # 存储第一个batch用于可视化
        first_batch_data_for_vis = None

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} - Validation')
            for signals, targets in pbar:
                signals = signals.to(self.device)
                targets = targets.to(self.device)
                
                # 推理
                preds_raw = self.model(signals)
                
                # 计算损失
                loss, _ = compute_loss(preds_raw, targets, self.model, self.criterion)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                
                # --- 为mAP指标格式化输出 ---
                preds_processed = postprocess_for_metrics(preds_raw, self.model, conf_thresh=0.1)  # 降低置信度阈值
                preds_for_map.extend(preds_processed)
                
                # 格式化真实标签
                for b in range(signals.shape[0]):
                    gt_targets = targets[targets[:, 0] == b]
                    # cx_norm, w_norm -> x1, y1, x2, y2
                    cx_norm = gt_targets[:, 2]
                    w_norm = gt_targets[:, 3]
                    x1 = (cx_norm - w_norm / 2) * self.config['input_length']
                    x2 = (cx_norm + w_norm / 2) * self.config['input_length']
                    
                    boxes = torch.stack([
                        x1,
                        torch.zeros_like(x1),
                        x2,
                        torch.ones_like(x2)
                    ], dim=1)
                    targets_for_map.append({'boxes': boxes, 'labels': gt_targets[:, 1].long()})

                # 保存第一个batch用于可视化
                if first_batch_data_for_vis is None:
                    first_batch_data_for_vis = {
                        "signals": signals.cpu(),
                        "preds": copy.deepcopy(preds_processed),
                        "gts": copy.deepcopy(targets_for_map[:signals.shape[0]])
                    }

        # --- 计算指标 ---
        self.map_metric.update(preds_for_map, targets_for_map)
        map_stats = self.map_metric.compute()
        self.map_metric.reset()
        
        final_val_loss = val_loss / len(self.val_loader)

        # --- 日志记录 ---
        self.writer.add_scalar('Loss/validation', final_val_loss, epoch)
        self.writer.add_scalar('mAP/0.5', map_stats['map_50'], epoch)
        self.writer.add_scalar('mAP/0.5:0.95', map_stats['map'], epoch)

        return final_val_loss, map_stats['map_50'].item()
    
    def train(self):
        """完整训练流程"""
        print(f"开始训练，设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"日志保存在: {self.writer.log_dir}")
        print(f"训练将从第 {self.start_epoch + 1} 个epoch开始，总共 {self.config['epochs']} 个epoch")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            
            # 验证
            val_loss, current_map = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.mAPs.append(current_map)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mAP@0.5: {current_map:.4f}")
            
            # 可视化一个验证样本到TensorBoard (每个epoch)
            self.visualize_to_tensorboard(epoch)

            # 保存最佳模型 (基于mAP)
            if current_map > self.best_map:
                self.best_map = current_map
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"新纪录! 保存最佳模型，mAP@0.5: {current_map:.4f}")
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
        
        # 绘制最终训练曲线
        plot_training_curves(self.train_losses, self.val_losses, self.mAPs)
        self.writer.close()
        print("训练完成！")
    
    def visualize_to_tensorboard(self, epoch):
        """将一个batch的预测结果可视化到TensorBoard"""
        self.model.eval()
        # 使用一个固定的验证batch进行可视化
        signals, targets = next(iter(self.val_loader))
        signals = signals.to(self.device)

        with torch.no_grad():
            preds_raw = self.model(signals)
            preds_processed = postprocess_for_metrics(preds_raw, self.model)

        # 格式化GT
        gt_targets = targets[targets[:, 0] == 0] # 只取第一个样本
        # cx_norm, w_norm -> x1, y1, x2, y2
        cx_norm = gt_targets[:, 2]
        w_norm = gt_targets[:, 3]
        x1 = (cx_norm - w_norm / 2) * self.config['input_length']
        x2 = (cx_norm + w_norm / 2) * self.config['input_length']
        gt_boxes = torch.stack([x1, torch.zeros_like(x1), x2, torch.ones_like(x2)], dim=1)

        gt_for_vis = {'boxes': gt_boxes, 'labels': gt_targets[:, 1].long()}
        
        # 使用第一个样本进行绘图
        fig = plot_validation_predictions(
            signals[0].cpu().numpy().flatten(),
            gt_for_vis,
            preds_processed[0],
            self.config['class_names'],
            title=f"Validation Prediction at Epoch {epoch+1}"
        )
        self.writer.add_figure('Validation/prediction_sample', fig, global_step=epoch)

    def save_checkpoint(self, epoch, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'mAPs': self.mAPs,
            'best_map': self.best_map,
            'config': self.config
        }
        save_path = self.weights_dir / filename
        torch.save(checkpoint, save_path)
        print(f"💾 Checkpoint saved to {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train YOLOv1D model on a custom dataset.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from.')
    args = parser.parse_args()

    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print("YAML configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    if not os.path.exists(config['dataset_path']):
        print(f"数据集不存在: {config['dataset_path']}")
        print("请先运行 dataset_generator.py 生成数据集")
        return
    
    try:
        (train_signals, train_labels), (val_signals, val_labels), dataset_info = load_generated_dataset(
            config['dataset_path']
        )
        print(f"数据集加载成功:")
        print(f"  训练样本: {len(train_signals)}")
        print(f"  验证样本: {len(val_signals)}")
        print(f"  类别数: {dataset_info['num_classes']}")
        print(f"  类别名称: {dataset_info['class_names']}")
        
        # 更新配置
        config['num_classes'] = dataset_info['num_classes']
        config['input_length'] = dataset_info['sequence_length']
        config['class_names'] = dataset_info['class_names']
        # 添加一个运行名称
        config['run_name'] = f"run_{config['model_size']}_bs{config['batch_size']}_{config['epochs']}e"
        
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        return
    
    # 创建模型
    print("🏗️  创建模型...")
    model = create_yolo1d_model(
        model_size=config['model_size'],
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        input_length=config['input_length'],
        reg_max=config.get('reg_max', 16)
    )
    
    # 数据增强
    train_transform = DataAugmentation(noise_std=0.02, scale_range=(0.9, 1.1))
    
    # 数据集
    train_dataset = SinWaveDataset(train_signals, train_labels, transform=train_transform)
    val_dataset = SinWaveDataset(val_signals, val_labels, transform=None)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 显示数据统计信息
    print(f"\n数据统计:")
    print(f"  批大小: {config['batch_size']}")
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    
    # 创建训练器
    print("🚂 创建训练器...")
    trainer = Trainer(model, train_loader, val_loader, device, config, resume_from=args.resume)
    
    # 开始训练
    trainer.train()
    
    print("\n训练完成！")
    print("可以使用以下命令进行推理:")
    print("python inference_yolo1d.py --model best_model.pth --signal-type anomaly")


if __name__ == "__main__":
    main() 