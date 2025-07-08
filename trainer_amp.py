#!/usr/bin/env python3
"""
支持混合精度训练的YOLO1D训练器
提升训练效率，减少内存使用
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from typing import Dict, Any, Optional, Tuple
import logging

from trainer_base import BaseTrainer, YOLO1DTrainer
from utils import postprocess_for_metrics


class AMPTrainer(BaseTrainer):
    """自动混合精度训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 config: Dict[str, Any],
                 device: torch.device,
                 resume_from: Optional[str] = None):
        """
        初始化AMP训练器
        
        Args:
            model: YOLO1D模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
            device: 计算设备
            resume_from: 恢复训练的检查点路径
        """
        super().__init__(model, train_loader, val_loader, config, device, resume_from)
        
        # 设置混合精度训练
        self.setup_amp()
    
    def setup_amp(self):
        """设置自动混合精度训练"""
        if self.device.type == 'cuda':
            self.scaler = GradScaler('cuda')
            self.use_amp = True
            self.logger.info("✅ 启用自动混合精度训练")
        else:
            self.use_amp = False
            self.logger.info("⚠️ CPU设备，禁用混合精度训练")
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch（混合精度版本）"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = self._create_progress_bar(epoch, "Training")
        
        for batch_idx, (signals, targets) in enumerate(pbar):
            try:
                signals = signals.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播（混合精度）
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        preds = self.model(signals)
                        loss, loss_items = self.criterion(preds, targets, self.model)
                else:
                    preds = self.model(signals)
                    loss, loss_items = self.criterion(preds, targets, self.model)
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Warning: Invalid loss at batch {batch_idx}: {loss}")
                    continue
                
                # 反向传播（混合精度）
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.get('grad_clip', 1.0)
                    )
                    
                    # 优化器步骤
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.get('grad_clip', 1.0)
                    )
                    
                    self.optimizer.step()
                
                # 更新学习率（OneCycleLR）
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                
                # 记录损失
                epoch_loss += loss.item()
                
                # 更新进度条
                self._update_progress_bar(pbar, loss, loss_items, batch_idx, epoch)
                
                # 记录到TensorBoard
                self._log_training_step(loss, loss_items, batch_idx, epoch)
                
            except Exception as e:
                self.logger.error(f"训练步骤出错 (batch {batch_idx}): {e}")
                continue
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """验证（混合精度版本）"""
        self.model.eval()
        val_loss = 0.0
        
        # 重置mAP指标
        self.map_metric.reset()
        
        with torch.no_grad():
            pbar = self._create_progress_bar(epoch, "Validation")
            
            for batch_idx, (signals, targets) in enumerate(pbar):
                try:
                    signals = signals.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向传播（混合精度）
                    if self.use_amp:
                        with autocast(device_type='cuda', dtype=torch.float16):
                            preds = self.model(signals)
                            loss, loss_items = self.criterion(preds, targets, self.model)
                    else:
                        preds = self.model(signals)
                        loss, loss_items = self.criterion(preds, targets, self.model)
                    
                    val_loss += loss.item()
                    
                    # 计算mAP
                    self._update_map_metric(preds, targets, signals)
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'val_loss': f'{loss.item():.4f}',
                        'box': f'{loss_items[0]:.4f}',
                        'cls': f'{loss_items[1]:.4f}',
                        'dfl': f'{loss_items[2]:.4f}',
                    })
                    
                except Exception as e:
                    self.logger.error(f"验证步骤出错 (batch {batch_idx}): {e}")
                    continue
        
        # 计算平均损失和mAP
        avg_val_loss = val_loss / len(self.val_loader)
        
        # 检查mAP指标是否有数据
        try:
            mAP_result = self.map_metric.compute()
            mAP = mAP_result['map'].item()
                
        except Exception as e:
            self.logger.warning(f"mAP计算失败: {e}，使用默认值0.0")
            mAP = 0.0
        
        return avg_val_loss, mAP
    
    def _create_progress_bar(self, epoch: int, mode: str):
        """创建进度条"""
        from tqdm import tqdm
        return tqdm(
            self.train_loader if mode == "Training" else self.val_loader,
            desc=f'Epoch {epoch+1}/{self.config["epochs"]} - {mode}'
        )
    
    def _update_progress_bar(self, pbar, loss, loss_items, batch_idx, epoch):
        """更新进度条"""
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'box': f'{loss_items[0]:.4f}',
            'cls': f'{loss_items[1]:.4f}',
            'dfl': f'{loss_items[2]:.4f}',
            'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
        })
    
    def _log_training_step(self, loss, loss_items, batch_idx, epoch):
        """记录训练步骤到TensorBoard"""
        global_step = epoch * len(self.train_loader) + batch_idx
        self.writer.add_scalar('Loss/train_step', loss.item(), global_step)
        self.writer.add_scalar('LR/step', self.optimizer.param_groups[0]['lr'], global_step)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点（包含AMP状态）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'mAPs': self.mAPs,
            'best_map': self.best_map
        }
        
        # 保存AMP状态
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        latest_path = f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, latest_path)
        self.logger.info(f"💾 保存检查点: {latest_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏆 保存最佳模型: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """从检查点恢复训练状态（包含AMP状态）"""
        try:
            self.logger.info(f"📂 加载检查点: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("✓ 模型权重加载成功")
            
            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("✓ 优化器状态加载成功")
            
            # 加载调度器状态
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("✓ 调度器状态加载成功")
            
            # 加载AMP状态
            if self.use_amp and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.logger.info("✓ AMP状态加载成功")
            
            # 加载训练历史
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                self.val_losses = checkpoint['val_losses']
            if 'mAPs' in checkpoint:
                self.mAPs = checkpoint['mAPs']
            if 'best_map' in checkpoint:
                self.best_map = checkpoint['best_map']
            
            # 设置起始epoch
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                self.logger.info(f"✓ 将从第 {self.start_epoch} 个epoch开始训练")
            
            # 验证配置一致性
            if 'config' in checkpoint:
                self._validate_config_consistency(checkpoint['config'])
            
            self.logger.info(f"✓ 检查点恢复完成！将从第 {self.start_epoch} 个epoch继续训练")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 加载检查点失败: {str(e)}")
            return False

    def _update_map_metric(self, preds, targets, signals):
        """更新mAP指标 - 使用与train_yolo1d_with_dataset.py一致的逻辑"""
        try:
            # 使用postprocess_for_metrics进行后处理
            preds_processed = postprocess_for_metrics(preds, self.model, conf_thresh=0.1)
            
            # 格式化真实标签
            for b in range(signals.shape[0]):
                gt_targets = targets[targets[:, 0] == b]
                if gt_targets.shape[0] > 0:
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
                    
                    # 更新mAP指标
                    self.map_metric.update(
                        preds=[preds_processed[b]],
                        target=[{
                            'boxes': boxes,
                            'labels': gt_targets[:, 1].long()
                        }]
                    )
                    
        except Exception as e:
            # 如果mAP更新失败，记录错误但不中断训练
            self.logger.debug(f"mAP更新失败: {e}")


class YOLO1DAMPTrainer(AMPTrainer):
    """YOLO1D专用AMP训练器"""
    
    def __init__(self, 
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 config: Dict[str, Any],
                 device: torch.device,
                 resume_from: Optional[str] = None):
        """
        初始化YOLO1D AMP训练器
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
            device: 计算设备
            resume_from: 恢复训练的检查点路径
        """
        from yolo1d_model import create_yolo1d_model
        
        # 创建模型
        try:
            model = create_yolo1d_model(
                model_size=config['model_size'],
                num_classes=config['num_classes'],
                input_channels=config['input_channels'],
                input_length=config['input_length']
            )
        except Exception as e:
            raise Exception(f"模型初始化失败: {e}")
        
        # 调用父类初始化
        super().__init__(model, train_loader, val_loader, config, device, resume_from)
    
    def _update_map_metric(self, preds, targets, signals):
        """更新mAP指标 - 使用与train_yolo1d_with_dataset.py一致的逻辑"""
        try:
            # 使用postprocess_for_metrics进行后处理
            preds_processed = postprocess_for_metrics(preds, self.model, conf_thresh=0.1)
            
            # 格式化真实标签
            for b in range(signals.shape[0]):
                gt_targets = targets[targets[:, 0] == b]
                if gt_targets.shape[0] > 0:
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
                    
                    # 更新mAP指标
                    self.map_metric.update(
                        preds=[preds_processed[b]],
                        target=[{
                            'boxes': boxes,
                            'labels': gt_targets[:, 1].long()
                        }]
                    )
                    
        except Exception as e:
            # 如果mAP更新失败，记录错误但不中断训练
            self.logger.debug(f"mAP更新失败: {e}") 