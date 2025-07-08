#!/usr/bin/env python3
"""
YOLO1D 训练器基类
统一训练逻辑，减少重复代码
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import yaml
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from yolo1d_model import create_yolo1d_model
from yolo1d_loss import YOLO1DLoss, compute_loss
from utils import plot_validation_predictions, postprocess_for_metrics


class YOLO1DError(Exception):
    """YOLO1D专用异常类"""
    pass


class ModelInitializationError(YOLO1DError):
    """模型初始化错误"""
    pass


class TrainingError(YOLO1DError):
    """训练错误"""
    pass


class ConfigError(YOLO1DError):
    """配置错误"""
    pass


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.validate_config()
    
    def load_config(self, path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ConfigError(f"加载配置文件失败 {path}: {e}")
    
    def validate_config(self):
        """验证配置有效性"""
        required_keys = [
            'model_size', 'num_classes', 'input_channels', 
            'input_length', 'epochs', 'batch_size', 'learning_rate'
        ]
        
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ConfigError(f"配置文件缺少必需参数: {missing_keys}")
        
        # 验证数值范围
        if self.config['learning_rate'] <= 0:
            raise ConfigError("学习率必须大于0")
        
        if self.config['batch_size'] <= 0:
            raise ConfigError("批大小必须大于0")
        
        if self.config['epochs'] <= 0:
            raise ConfigError("训练轮数必须大于0")
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        self.config.update(updates)
        self.validate_config()


class BaseTrainer:
    """训练器基类，包含通用功能"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 config: Dict[str, Any],
                 device: torch.device,
                 resume_from: Optional[str] = None):
        """
        初始化训练器
        
        Args:
            model: YOLO1D模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
            device: 计算设备
            resume_from: 恢复训练的检查点路径
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.start_epoch = 0
        
        # 设置日志
        self.setup_logging()
        
        # 修复步长问题
        self.fix_stride_issue()
        
        # 设置训练组件
        self.setup_training_components()
        
        # 设置监控
        self.setup_monitoring()
        
        # 如果指定了恢复检查点，则加载
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def fix_stride_issue(self):
        """修复步长问题"""
        self.logger.info("🔧 修复步长问题")
        
        input_length = self.config['input_length']
        
        if input_length == 1024:
            correct_strides = [8, 16, 32]
        else:
            correct_strides = [input_length // 128, input_length // 64, input_length // 32]
        
        self.logger.info(f"  输入长度: {input_length}")
        self.logger.info(f"  计算出的正确步长: {correct_strides}")
        
        with torch.no_grad():
            for i, stride in enumerate(correct_strides):
                self.model.detect.stride[i] = float(stride)
        
        self.logger.info(f"  修复后的步长: {self.model.detect.stride}")
    
    def setup_training_components(self):
        """设置训练组件"""
        # 定义损失超参数
        hyp = self.config.get('hyp', {})
        self.logger.info(f"使用损失超参数: {hyp}")

        # 损失函数
        self.criterion = YOLO1DLoss(
            num_classes=self.config['num_classes'],
            reg_max=self.config.get('reg_max', 16),
            use_dfl=True,
            hyp=hyp
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.0005)
        )
        
        # 学习率调度器
        self.setup_scheduler()
    
    def setup_scheduler(self):
        """设置学习率调度器"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'onecycle':
            # 使用OneCycleLR (train_simple.py的成功策略)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=scheduler_config.get('max_lr', self.config['learning_rate']),
                epochs=self.config['epochs'],
                steps_per_epoch=len(self.train_loader)
            )
            self.logger.info("✅ 使用OneCycleLR调度器")
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=scheduler_config.get('eta_min', self.config['learning_rate'] * 0.01)
            )
            self.logger.info("✅ 使用CosineAnnealingLR调度器")
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 30),
                gamma=self.config.get('lr_gamma', 0.1)
            )
            self.logger.info("✅ 使用StepLR调度器")
    
    def setup_monitoring(self):
        """设置监控"""
        # TensorBoard日志
        self.log_dir = Path(f"runs/{self.config.get('run_name', 'experiment')}")
        self.writer = SummaryWriter(str(self.log_dir))

        # 权重保存目录
        self.weights_dir = self.log_dir / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # mAP指标
        self.map_metric = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox'
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.mAPs = []
        self.best_map = 0.0
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=self.config.get('patience', 10),
            min_delta=self.config.get('min_delta', 0.001)
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """从检查点恢复训练状态，支持绝对路径和相对路径。"""
        try:
            resume_path = Path(checkpoint_path)
            if not resume_path.is_file():
                self.logger.error(f"❌ 恢复训练失败：检查点文件不存在或不是一个文件 -> {resume_path}")
                return False

            self.logger.info(f"📂 正在加载检查点: {resume_path.resolve()}")
            
            checkpoint = torch.load(resume_path, map_location=self.device)
            
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
            self.logger.error(f"❌ 加载检查点失败: {str(e)}", exc_info=True)
            return False
    
    def _validate_config_consistency(self, saved_config: Dict[str, Any]):
        """验证配置一致性"""
        key_configs = ['model_size', 'num_classes', 'input_channels', 'input_length']
        config_mismatch = []
        
        for key in key_configs:
            if key in saved_config and key in self.config:
                if saved_config[key] != self.config[key]:
                    config_mismatch.append(f"{key}: {saved_config[key]} -> {self.config[key]}")
        
        if config_mismatch:
            self.logger.warning("⚠️  配置不匹配:")
            for mismatch in config_mismatch:
                self.logger.warning(f"    {mismatch}")
            self.logger.warning("建议使用与检查点相同的配置进行恢复训练")
        else:
            self.logger.info("✓ 配置一致性检查通过")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
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
        
        # 保存最新检查点
        latest_path = self.weights_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, latest_path)
        self.logger.info(f"💾 保存检查点: {latest_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = self.weights_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏆 保存最佳模型: {best_path}")
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} - Training')
        
        for batch_idx, (signals, targets) in enumerate(pbar):
            try:
                signals = signals.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                preds = self.model(signals)
                
                # 计算损失
                loss, loss_items = self.criterion(preds, targets, self.model)
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Warning: Invalid loss at batch {batch_idx}: {loss}")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.get('grad_clip', 1.0)
                )
                
                self.optimizer.step()
                
                # 更新学习率（OneCycleLR）
                if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                
                # 记录损失
                epoch_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'box': f'{loss_items[0]:.4f}',
                    'cls': f'{loss_items[1]:.4f}',
                    'dfl': f'{loss_items[2]:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                })
                
                # 记录到TensorBoard
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/train_step', loss.item(), global_step)
                self.writer.add_scalar('LR/step', self.optimizer.param_groups[0]['lr'], global_step)
                
            except Exception as e:
                self.logger.error(f"训练步骤出错 (batch {batch_idx}): {e}")
                continue
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        val_loss = 0.0
        
        # 重置mAP指标
        self.map_metric.reset()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} - Validation')
            
            for batch_idx, (signals, targets) in enumerate(pbar):
                try:
                    signals = signals.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向传播
                    preds = self.model(signals)
                    
                    # 计算损失
                    loss, loss_items = self.criterion(preds, targets, self.model)
                    val_loss += loss.item()
                    
                    # 计算mAP
                    self._update_map_metric(preds, targets)
                    
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
    
    def _update_map_metric(self, preds: List[Dict[str, torch.Tensor]], targets: torch.Tensor):
        """
        使用来自postprocess_for_metrics的预测更新mAP指标
        preds: 经过后处理的预测列表，其坐标为绝对坐标 (e.g., 0-1024)
        targets: 原始批次目标张量，其坐标为归一化坐标 [0, 1]
        """
        # 转换目标以匹配指标格式
        processed_targets = []
        num_samples_in_batch = len(preds) # 修正: 使用实际批次大小

        for i in range(num_samples_in_batch): # 修正: 遍历实际样本数
            mask = targets[:, 0] == i
            batch_targets = targets[mask]
            
            if batch_targets.shape[0] > 0:
                # [class, x_center, width] -> [x1, y1, x2, y2]
                #
                # --- 坐标系对齐 (关键) ---
                # postprocess_for_metrics 输出的预测框(preds)是绝对坐标 (0 to input_length)。
                # 而从DataLoader加载的标签(targets)是归一化坐标 (0 to 1)。
                # 因此，在计算mAP之前，必须将真实标签(GT)的坐标乘以input_length，
                # 将其从"归一化"转换为"绝对坐标"，以匹配预测框的坐标系。
                #
                input_len = self.config['input_length']
                boxes_x1 = (batch_targets[:, 2] - batch_targets[:, 3] / 2) * input_len
                boxes_x2 = (batch_targets[:, 2] + batch_targets[:, 3] / 2) * input_len
                boxes_xyxy = torch.stack([boxes_x1, torch.zeros_like(boxes_x1), boxes_x2, torch.ones_like(boxes_x1)], dim=1)
                
                processed_targets.append({
                    'boxes': boxes_xyxy,
                    'labels': batch_targets[:, 1].long()
                })
            else:
                 processed_targets.append({
                    'boxes': torch.empty(0, 4, device=self.device),
                    'labels': torch.empty(0, dtype=torch.long, device=self.device)
                })

        try:
            # 现在 preds 和 processed_targets 的长度保证一致
            self.map_metric.update(preds, processed_targets)
        except Exception as e:
            self.logger.warning(f"更新mAP指标时出错: {e}")

    def train(self):
        """完整训练流程"""
        self.logger.info("🚀 开始训练")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"训练轮数: {self.config['epochs']}")
        self.logger.info(f"批大小: {self.config['batch_size']}")
        self.logger.info(f"学习率: {self.config['learning_rate']}")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            try:
                # 训练
                train_loss = self.train_epoch(epoch)
                self.train_losses.append(train_loss)
                
                # 验证
                val_loss, mAP = self.validate(epoch)
                self.val_losses.append(val_loss)
                self.mAPs.append(mAP)
                
                # 更新学习率（非OneCycleLR）
                if not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                
                # 记录到TensorBoard
                self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
                self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)
                self.writer.add_scalar('Metrics/mAP', mAP, epoch)
                self.writer.add_scalar('LR/epoch', self.optimizer.param_groups[0]['lr'], epoch)
                
                # 保存检查点
                is_best = mAP > self.best_map
                if is_best:
                    self.best_map = mAP
                
                self.save_checkpoint(epoch, is_best)
                
                # 早停检查
                if self.early_stopping(val_loss):
                    self.logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config['epochs']} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"mAP: {mAP:.4f}, "
                    f"Best mAP: {self.best_map:.4f}"
                )
                
            except Exception as e:
                self.logger.error(f"训练epoch {epoch+1} 出错: {e}")
                continue
        
        self.logger.info("✅ 训练完成")
        self.writer.close()


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class YOLO1DTrainer(BaseTrainer):
    """YOLO1D训练器，覆盖特定方法"""
    
    def __init__(self, 
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 config: Dict[str, Any],
                 device: torch.device,
                 resume_from: Optional[str] = None):
        
        # 1. 创建模型
        model = create_yolo1d_model(
            model_size=config['model_size'],
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            input_length=config['input_length'],
            reg_max=config.get('reg_max', 16)
        )
        
        super().__init__(model, train_loader, val_loader, config, device, resume_from)

    def _update_map_metric(self, preds: List[Dict[str, torch.Tensor]], targets: torch.Tensor):
        """
        使用来自postprocess_for_metrics的预测更新mAP指标
        preds: 经过后处理的预测列表，其坐标为绝对坐标 (e.g., 0-1024)
        targets: 原始批次目标张量，其坐标为归一化坐标 [0, 1]
        """
        # 转换目标以匹配指标格式
        processed_targets = []
        num_samples_in_batch = len(preds) # 修正: 使用实际批次大小

        for i in range(num_samples_in_batch): # 修正: 遍历实际样本数
            mask = targets[:, 0] == i
            batch_targets = targets[mask]
            
            if batch_targets.shape[0] > 0:
                # [class, x_center, width] -> [x1, y1, x2, y2]
                #
                # --- 坐标系对齐 (关键) ---
                # postprocess_for_metrics 输出的预测框(preds)是绝对坐标 (0 to input_length)。
                # 而从DataLoader加载的标签(targets)是归一化坐标 (0 to 1)。
                # 因此，在计算mAP之前，必须将真实标签(GT)的坐标乘以input_length，
                # 将其从"归一化"转换为"绝对坐标"，以匹配预测框的坐标系。
                #
                input_len = self.config['input_length']
                boxes_x1 = (batch_targets[:, 2] - batch_targets[:, 3] / 2) * input_len
                boxes_x2 = (batch_targets[:, 2] + batch_targets[:, 3] / 2) * input_len
                boxes_xyxy = torch.stack([boxes_x1, torch.zeros_like(boxes_x1), boxes_x2, torch.ones_like(boxes_x1)], dim=1)
                
                processed_targets.append({
                    'boxes': boxes_xyxy,
                    'labels': batch_targets[:, 1].long()
                })
            else:
                 processed_targets.append({
                    'boxes': torch.empty(0, 4, device=self.device),
                    'labels': torch.empty(0, dtype=torch.long, device=self.device)
                })

        try:
            # 现在 preds 和 processed_targets 的长度保证一致
            self.map_metric.update(preds, processed_targets)
        except Exception as e:
            self.logger.warning(f"更新mAP指标时出错: {e}")

    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        验证模型 - 与train_simple.py对齐的简化版本
        """
        self.model.train()  # 关键: 保持训练模式以获得原始输出
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for signals, targets in pbar:
                signals = signals.to(self.device)
                targets = targets.to(self.device)

                # 1. 前向传播
                preds_raw = self.model(signals)
                
                # 2. 计算损失 (修正: 直接调用criterion并传入model)
                loss, _ = self.criterion(preds_raw, targets, self.model)
                total_loss += loss.item()

                # 3. 后处理以计算mAP
                # 使用conf_thresh=0.001, iou_thresh=0.6作为标准验证设置
                processed_preds = postprocess_for_metrics(preds_raw, self.model, conf_thresh=0.001, iou_thresh=0.6)

                # 4. 更新mAP指标
                self._update_map_metric(processed_preds, targets)
        
        # 计算平均损失和mAP
        avg_loss = total_loss / len(self.val_loader)
        
        try:
            map_stats = self.map_metric.compute()
            mAP = map_stats['map_50'].item()
            self.map_metric.reset()
        except Exception as e:
            self.logger.error(f"计算mAP时出错: {e}")
            mAP = 0.0

        # 验证后无需切换模式，因为我们一直处于train()模式
        return avg_loss, mAP 