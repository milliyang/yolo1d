#!/usr/bin/env python3
"""
增强的数据增强模块
专门针对时域数据的增强方法
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import random
import scipy.signal as signal
from scipy.interpolate import interp1d


class BaseAugmentation:
    """数据增强基类"""
    
    def __init__(self, probability: float = 0.5):
        """
        初始化增强器
        
        Args:
            probability: 应用增强的概率
        """
        self.probability = probability
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        应用增强
        
        Args:
            data: 输入数据 [batch, channels, length]
            
        Returns:
            增强后的数据
        """
        if random.random() < self.probability:
            return self.apply(data)
        return data
    
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        """具体的增强实现，子类需要重写"""
        raise NotImplementedError


class GaussianNoise(BaseAugmentation):
    """高斯噪声增强"""
    
    def __init__(self, std_range: Tuple[float, float] = (0.01, 0.1), probability: float = 0.5):
        super().__init__(probability)
        self.std_range = std_range
    
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(data) * std
        return data + noise


class AmplitudeScaling(BaseAugmentation):
    """振幅缩放增强"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), probability: float = 0.5):
        super().__init__(probability)
        self.scale_range = scale_range
    
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        scale = random.uniform(*self.scale_range)
        return data * scale


class TimeShifting(BaseAugmentation):
    """时间偏移增强"""
    
    def __init__(self, shift_range: Tuple[int, int] = (-50, 50), probability: float = 0.5):
        super().__init__(probability)
        self.shift_range = shift_range
    
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        shift = random.randint(*self.shift_range)
        if shift > 0:
            # 向右偏移
            return F.pad(data, (0, shift), mode='constant', value=0)[:, :, shift:]
        else:
            # 向左偏移
            return F.pad(data, (-shift, 0), mode='constant', value=0)[:, :, :shift]


class TimeWarping(BaseAugmentation):
    """时间扭曲增强"""
    
    def __init__(self, warp_factor: float = 0.1, probability: float = 0.3):
        super().__init__(probability)
        self.warp_factor = warp_factor
    
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = data.shape
        device = data.device
        
        # 创建扭曲函数
        x_original = torch.linspace(0, 1, length, device=device)
        warp = torch.sin(2 * np.pi * random.uniform(0.5, 2.0) * x_original) * self.warp_factor
        x_warped = x_original + warp
        x_warped = torch.clamp(x_warped, 0, 1)
        
        # 应用扭曲
        warped_data = torch.zeros_like(data)
        for b in range(batch_size):
            for c in range(channels):
                # 使用线性插值
                f = interp1d(x_original.cpu().numpy(), 
                           data[b, c].cpu().numpy(), 
                           kind='linear', 
                           bounds_error=False, 
                           fill_value='extrapolate')
                warped_data[b, c] = torch.from_numpy(f(x_warped.cpu().numpy())).to(device)
        
        return warped_data


class FrequencyMasking(BaseAugmentation):
    """频域掩码增强"""
    
    def __init__(self, mask_ratio: float = 0.1, probability: float = 0.3):
        super().__init__(probability)
        self.mask_ratio = mask_ratio
    
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = data.shape
        
        # 计算FFT
        fft_data = torch.fft.fft(data, dim=-1)
        
        # 创建掩码
        mask_size = int(length * self.mask_ratio)
        start_idx = random.randint(0, length - mask_size)
        
        # 应用掩码（对称掩码）
        fft_data[:, :, start_idx:start_idx + mask_size] = 0
        if start_idx + mask_size < length // 2:
            # 对称部分
            end_idx = length - start_idx - mask_size
            fft_data[:, :, end_idx:end_idx + mask_size] = 0
        
        # 逆FFT
        return torch.fft.ifft(fft_data, dim=-1).real


class SpecAugment(BaseAugmentation):
    """SpecAugment增强（适用于时域信号）"""
    
    def __init__(self, 
                 time_mask_param: int = 20,
                 freq_mask_param: int = 10,
                 probability: float = 0.3):
        super().__init__(probability)
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
    
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = data.shape
        
        # 时间掩码
        time_mask_size = random.randint(0, self.time_mask_param)
        if time_mask_size > 0:
            time_start = random.randint(0, length - time_mask_size)
            data[:, :, time_start:time_start + time_mask_size] = 0
        
        # 频率掩码（通过FFT实现）
        if self.freq_mask_param > 0:
            freq_mask_size = random.randint(0, self.freq_mask_param)
            if freq_mask_size > 0:
                fft_data = torch.fft.fft(data, dim=-1)
                freq_start = random.randint(0, length // 2 - freq_mask_size)
                fft_data[:, :, freq_start:freq_start + freq_mask_size] = 0
                fft_data[:, :, length - freq_start - freq_mask_size:length - freq_start] = 0
                data = torch.fft.ifft(fft_data, dim=-1).real
        
        return data


class Mixup(BaseAugmentation):
    """Mixup增强"""
    
    def __init__(self, alpha: float = 0.2, probability: float = 0.3):
        super().__init__(probability)
        self.alpha = alpha
    
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        batch_size = data.shape[0]
        if batch_size < 2:
            return data
        
        # 随机选择另一个样本
        idx = random.randint(0, batch_size - 1)
        
        # 生成混合权重
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 混合
        mixed_data = lam * data + (1 - lam) * data[idx:idx+1]
        
        return mixed_data


class CutMix(BaseAugmentation):
    """CutMix增强"""
    
    def __init__(self, cut_ratio: float = 0.3, probability: float = 0.3):
        super().__init__(probability)
        self.cut_ratio = cut_ratio
    
    def apply(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = data.shape
        if batch_size < 2:
            return data
        
        # 随机选择另一个样本
        idx = random.randint(0, batch_size - 1)
        
        # 计算切割区域
        cut_length = int(length * self.cut_ratio)
        cut_start = random.randint(0, length - cut_length)
        
        # 创建混合数据
        mixed_data = data.clone()
        mixed_data[:, :, cut_start:cut_start + cut_length] = data[idx:idx+1, :, cut_start:cut_start + cut_length]
        
        return mixed_data


class AdvancedDataAugmentation:
    """高级数据增强组合"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化高级数据增强
        
        Args:
            config: 增强配置
        """
        if config is None:
            config = {}
        
        self.augmentations = [
            GaussianNoise(
                std_range=config.get('noise_std_range', (0.01, 0.1)),
                probability=config.get('noise_prob', 0.5)
            ),
            AmplitudeScaling(
                scale_range=config.get('scale_range', (0.8, 1.2)),
                probability=config.get('scale_prob', 0.5)
            ),
            TimeShifting(
                shift_range=config.get('shift_range', (-50, 50)),
                probability=config.get('shift_prob', 0.3)
            ),
            TimeWarping(
                warp_factor=config.get('warp_factor', 0.1),
                probability=config.get('warp_prob', 0.2)
            ),
            FrequencyMasking(
                mask_ratio=config.get('freq_mask_ratio', 0.1),
                probability=config.get('freq_mask_prob', 0.2)
            ),
            SpecAugment(
                time_mask_param=config.get('time_mask_param', 20),
                freq_mask_param=config.get('freq_mask_param', 10),
                probability=config.get('spec_augment_prob', 0.2)
            ),
            Mixup(
                alpha=config.get('mixup_alpha', 0.2),
                probability=config.get('mixup_prob', 0.1)
            ),
            CutMix(
                cut_ratio=config.get('cutmix_ratio', 0.3),
                probability=config.get('cutmix_prob', 0.1)
            )
        ]
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        应用所有增强
        
        Args:
            data: 输入数据 [batch, channels, length]
            
        Returns:
            增强后的数据
        """
        augmented_data = data.clone()
        
        for augmentation in self.augmentations:
            augmented_data = augmentation(augmented_data)
        
        return augmented_data
    
    def add_augmentation(self, augmentation: BaseAugmentation):
        """添加新的增强方法"""
        self.augmentations.append(augmentation)
    
    def remove_augmentation(self, augmentation_type: type):
        """移除指定类型的增强方法"""
        self.augmentations = [aug for aug in self.augmentations 
                            if not isinstance(aug, augmentation_type)]
    
    def get_augmentation_config(self) -> Dict:
        """获取当前增强配置"""
        config = {}
        for aug in self.augmentations:
            aug_name = aug.__class__.__name__
            config[f'{aug_name.lower()}_prob'] = aug.probability
            if hasattr(aug, 'std_range'):
                config[f'{aug_name.lower()}_std_range'] = aug.std_range
            if hasattr(aug, 'scale_range'):
                config[f'{aug_name.lower()}_scale_range'] = aug.scale_range
            # 添加其他属性...
        return config


class AutoAugment:
    """自动数据增强策略搜索"""
    
    def __init__(self, search_space: Dict):
        """
        初始化自动增强
        
        Args:
            search_space: 搜索空间配置
        """
        self.search_space = search_space
        self.best_policy = None
        self.best_score = 0.0
    
    def search_best_policy(self, model, dataset, num_trials: int = 10):
        """
        搜索最佳增强策略
        
        Args:
            model: 模型
            dataset: 数据集
            num_trials: 搜索次数
            
        Returns:
            最佳增强策略
        """
        for trial in range(num_trials):
            # 随机生成策略
            policy = self._generate_random_policy()
            
            # 评估策略
            score = self._evaluate_policy(policy, model, dataset)
            
            # 更新最佳策略
            if score > self.best_score:
                self.best_score = score
                self.best_policy = policy
        
        return self.best_policy
    
    def _generate_random_policy(self) -> Dict:
        """生成随机策略"""
        policy = {}
        for key, value_range in self.search_space.items():
            if isinstance(value_range, tuple):
                if isinstance(value_range[0], int):
                    policy[key] = random.randint(*value_range)
                else:
                    policy[key] = random.uniform(*value_range)
            elif isinstance(value_range, list):
                policy[key] = random.choice(value_range)
        return policy
    
    def _evaluate_policy(self, policy: Dict, model, dataset) -> float:
        """评估策略性能"""
        # 这里需要实现具体的评估逻辑
        # 可以使用验证集上的性能作为评估指标
        return random.random()  # 临时返回随机值


# 预定义的增强配置
PREDEFINED_CONFIGS = {
    'light': {
        'noise_std_range': (0.01, 0.05),
        'noise_prob': 0.3,
        'scale_range': (0.9, 1.1),
        'scale_prob': 0.3,
        'shift_range': (-20, 20),
        'shift_prob': 0.2,
        'warp_prob': 0.0,
        'freq_mask_prob': 0.0,
        'spec_augment_prob': 0.0,
        'mixup_prob': 0.0,
        'cutmix_prob': 0.0
    },
    'medium': {
        'noise_std_range': (0.01, 0.1),
        'noise_prob': 0.5,
        'scale_range': (0.8, 1.2),
        'scale_prob': 0.5,
        'shift_range': (-50, 50),
        'shift_prob': 0.3,
        'warp_factor': 0.1,
        'warp_prob': 0.2,
        'freq_mask_ratio': 0.1,
        'freq_mask_prob': 0.2,
        'time_mask_param': 20,
        'freq_mask_param': 10,
        'spec_augment_prob': 0.2,
        'mixup_alpha': 0.2,
        'mixup_prob': 0.1,
        'cutmix_ratio': 0.3,
        'cutmix_prob': 0.1
    },
    'heavy': {
        'noise_std_range': (0.02, 0.15),
        'noise_prob': 0.7,
        'scale_range': (0.7, 1.3),
        'scale_prob': 0.7,
        'shift_range': (-100, 100),
        'shift_prob': 0.5,
        'warp_factor': 0.2,
        'warp_prob': 0.4,
        'freq_mask_ratio': 0.2,
        'freq_mask_prob': 0.4,
        'time_mask_param': 40,
        'freq_mask_param': 20,
        'spec_augment_prob': 0.4,
        'mixup_alpha': 0.3,
        'mixup_prob': 0.2,
        'cutmix_ratio': 0.5,
        'cutmix_prob': 0.2
    }
}


def create_augmentation(config_name: str = 'medium', custom_config: Optional[Dict] = None) -> AdvancedDataAugmentation:
    """
    创建数据增强实例
    
    Args:
        config_name: 预定义配置名称 ('light', 'medium', 'heavy')
        custom_config: 自定义配置
        
    Returns:
        数据增强实例
    """
    if custom_config is not None:
        config = custom_config
    elif config_name in PREDEFINED_CONFIGS:
        config = PREDEFINED_CONFIGS[config_name]
    else:
        config = PREDEFINED_CONFIGS['medium']
    
    return AdvancedDataAugmentation(config) 