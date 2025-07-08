import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import random
from scipy.signal import find_peaks
import shutil
import torch
from torch.utils.data import Dataset


# 全局配置
SEQUENCE_LENGTH = 1024
CLASS_NAMES = ['peak', 'trough'] # 0: 波峰, 1: 波谷
NUM_CLASSES = len(CLASS_NAMES)


def find_events_in_signal(signal, min_width=20, min_prominence=0.3):
    """
    在信号中同时检测波峰和波谷
    
    Args:
        signal (np.ndarray): 输入信号
        min_width (int): 事件的最小宽度
        min_prominence (float): 事件的最小突起程度
        
    Returns:
        list: 标签列表 [[class_id, x_center, width], ...]
    """
    labels = []
    
    # 1. 检测波峰 (class_id = 0)
    # 使用宽度和突起参数来过滤噪声
    peaks, properties = find_peaks(
        signal, 
        prominence=min_prominence,
        width=min_width
    )
    
    for i, peak_pos in enumerate(peaks):
        # scipy返回的宽度是在半突起处的宽度，我们直接使用它
        width = properties['widths'][i]
        
        # 归一化
        x_center_norm = peak_pos / SEQUENCE_LENGTH
        width_norm = width / SEQUENCE_LENGTH
        
        labels.append([0, x_center_norm, width_norm])
        
    # 2. 检测波谷 (class_id = 1) by inverting the signal
    troughs, properties = find_peaks(
        -signal, 
        prominence=min_prominence,
        width=min_width
    )

    for i, trough_pos in enumerate(troughs):
        width = properties['widths'][i]
        
        # 归一化
        x_center_norm = trough_pos / SEQUENCE_LENGTH
        width_norm = width / SEQUENCE_LENGTH
        
        labels.append([1, x_center_norm, width_norm])
        
    return labels


class SinWaveDatasetGenerator:
    """
    生成包含多种波形和带标签峰值的数据集
    - 支持多种信号类型
    - 自动使用find_peaks进行标注
    - 保存为训练和验证集
    """
    def __init__(self, num_samples, output_dir='sin_wave_dataset'):
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.sequence_length = SEQUENCE_LENGTH
        self.class_names = CLASS_NAMES
        self.num_classes = NUM_CLASSES
        
        # 确保输出目录存在且清空
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        
        # 信号生成器字典
        self.signal_generators = {
            'single_freq': self._generate_single_frequency_signal,
            'multi_freq': self._generate_multi_frequency_signal,
            'high_noise': self._generate_high_noise_signal,
            'low_freq_large_amp': self._generate_low_freq_large_amp_signal,
            'mixed': self._generate_mixed_signal
        }
        
        # 初始化数据容器
        self.signals = []
        self.labels = []

    def _generate_single_frequency_signal(self):
        """生成单频信号"""
        t = np.linspace(0, 100, self.sequence_length)
        freq = np.random.uniform(1, 5)
        amplitude = np.random.uniform(0.5, 2)
        signal = amplitude * np.sin(2 * np.pi * freq * t / 100)
        signal += 0.1 * np.random.randn(self.sequence_length)
        
        # 随机添加一个显著的波峰或波谷
        event_type = np.random.choice(['peak', 'trough'])
        start = np.random.randint(self.sequence_length // 4, 3 * self.sequence_length // 4)
        width = np.random.randint(30, 80)
        end = min(start + width, self.sequence_length)
        event_amplitude = np.random.uniform(1.5, 3.0)
        
        if event_type == 'peak':
            signal[start:end] += event_amplitude
        else: # trough
            signal[start:end] -= event_amplitude
            
        return signal

    def _generate_multi_frequency_signal(self):
        """生成多频信号"""
        t = np.linspace(0, 100, self.sequence_length)
        signal = np.zeros(self.sequence_length)
        num_freqs = np.random.randint(2, 4)
        
        for _ in range(num_freqs):
            freq = np.random.uniform(1, 10)
            amplitude = np.random.uniform(0.5, 1.5)
            signal += amplitude * np.sin(2 * np.pi * freq * t / 100)
        
        signal += 0.15 * np.random.randn(self.sequence_length)

        # 添加多个事件
        num_events = np.random.randint(1, 4)
        for _ in range(num_events):
            event_type = np.random.choice(['peak', 'trough'])
            start = np.random.randint(0, self.sequence_length - 50)
            width = np.random.randint(25, 60)
            end = min(start + width, self.sequence_length)
            event_amplitude = np.random.uniform(1.0, 2.5)
            
            if event_type == 'peak':
                signal[start:end] += event_amplitude
            else: # trough
                signal[start:end] -= event_amplitude
        
        return signal

    def _generate_high_noise_signal(self):
        """生成高噪声信号"""
        t = np.linspace(0, 100, self.sequence_length)
        signal = np.random.randn(self.sequence_length) * np.random.uniform(0.5, 1.0)
        
        # 添加一个被噪声掩盖的事件
        event_type = np.random.choice(['peak', 'trough'])
        start = np.random.randint(self.sequence_length // 4, 3 * self.sequence_length // 4)
        width = np.random.randint(40, 100)
        end = min(start + width, self.sequence_length)
        amplitude = np.random.uniform(2.0, 4.0)
        
        if event_type == 'peak':
            signal[start:end] += amplitude
        else:
            signal[start:end] -= amplitude
            
        return signal

    def _generate_low_freq_large_amp_signal(self):
        """生成低频、大振幅变化的信号"""
        t = np.linspace(0, 100, self.sequence_length)
        base_freq = np.random.uniform(0.2, 0.8)
        base_amp = np.random.uniform(2, 4)
        
        # 振幅调制
        amp_mod_freq = np.random.uniform(0.05, 0.2)
        amp_mod = base_amp + np.sin(2 * np.pi * amp_mod_freq * t / 100)
        
        signal = amp_mod * np.sin(2 * np.pi * base_freq * t / 100)
        signal += 0.2 * np.random.randn(self.sequence_length)
        
        return signal

    def _generate_mixed_signal(self):
        """混合多种信号"""
        signal1 = self._generate_multi_frequency_signal()
        signal2 = self._generate_low_freq_large_amp_signal()
        
        mix_ratio = np.random.uniform(0.4, 0.6)
        signal = signal1 * mix_ratio + signal2 * (1 - mix_ratio)
        
        # 可能添加一个额外的大事件
        if np.random.rand() > 0.5:
            event_type = np.random.choice(['peak', 'trough'])
            start = np.random.randint(0, self.sequence_length - 80)
            width = np.random.randint(50, 100)
            end = min(start + width, self.sequence_length)
            amplitude = np.random.uniform(2.0, 3.5)
            
            if event_type == 'peak':
                signal[start:end] += amplitude
            else:
                signal[start:end] -= amplitude

        return signal

    def generate_and_label_data(self):
        """生成并标注所有数据"""
        print(f"开始生成 {self.num_samples} 个样本...")
        
        for i in tqdm(range(self.num_samples), desc="Generating Data"):
            # 随机选择信号类型
            signal_type = np.random.choice(
                ['single_freq', 'multi_freq', 'high_noise', 'low_freq_large_amp', 'mixed'],
                p=[0.00, 0.00, 0.00, 0.00, 1.0]
            )
                        
            signal = self.signal_generators[signal_type]()
            
            # 使用find_events_in_signal自动标注
            labels = find_events_in_signal(signal, min_width=20, min_prominence=0.5)
            
            # 只有当找到标签时才保存样本
            if labels:
                self.signals.append(signal)
                self.labels.append(labels)
    
    def split_and_save(self, train_split=0.8):
        """分割并保存数据集"""
        num_generated = len(self.signals)
        print(f"\n成功生成 {num_generated} 个带标签的样本。")
        
        if num_generated == 0:
            print("未能生成任何有效样本，请检查生成和标注逻辑。")
            return
            
        indices = np.random.permutation(num_generated)
        split_point = int(num_generated * train_split)
        
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
        
        # 准备训练集
        train_signals = [self.signals[i] for i in train_indices]
        train_labels = [self.labels[i] for i in train_indices]
        self.train_data = (train_signals, train_labels)

        # 准备验证集
        val_signals = [self.signals[i] for i in val_indices]
        val_labels = [self.labels[i] for i in val_indices]
        self.val_data = (val_signals, val_labels)

        # 保存为npy和json格式
        np.save(os.path.join(self.output_dir, 'train_signals.npy'), np.array(train_signals))
        with open(os.path.join(self.output_dir, 'train_labels.json'), 'w') as f:
            json.dump(train_labels, f)
            
        np.save(os.path.join(self.output_dir, 'val_signals.npy'), np.array(val_signals))
        with open(os.path.join(self.output_dir, 'val_labels.json'), 'w') as f:
            json.dump(val_labels, f)
            
        # 保存数据集信息
        dataset_info = {
            'num_train': len(train_signals),
            'num_val': len(val_signals),
            'sequence_length': self.sequence_length,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        with open(os.path.join(self.output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        print("\n数据集已保存:")
        print(f"  - 训练集: {len(train_signals)} 样本")
        print(f"  - 验证集: {len(val_signals)} 样本")
        print(f"  - 路径: '{self.output_dir}'")

    def save_dataset(self):
        """废弃 - 功能已合并到 split_and_save"""
        pass

    def visualize_sample(self, index, dataset_type='train'):
        """可视化单个样本及其标签"""
        if dataset_type == 'train':
            signals, labels_list = self.train_data
        else:
            signals, labels_list = self.val_data
            
        signal = signals[index]
        labels = labels_list[index]
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 6))
        
        plt.plot(signal, label='Signal', color='blue', alpha=0.8)
        
        colors = ['red', 'green'] # red for peak, green for trough
        
        for i, label in enumerate(labels):
            class_id, x_center_norm, width_norm = label
            
            # 反归一化
            x_center = x_center_norm * self.sequence_length
            width = width_norm * self.sequence_length
            x1 = x_center - width / 2
            x2 = x_center + width / 2
            
            color = colors[int(class_id)]
            event_name = self.class_names[int(class_id)]
            
            plt.axvspan(x1, x2, color=color, alpha=0.3, label=f'Event {i+1}: {event_name}')
            plt.axvline(x_center, color=color, linestyle='--', alpha=0.8)
        
        plt.title(f'Sample {index} - Type: {dataset_type}')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        
        # 处理图例以避免重复
        handles, labels_text = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels_text, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.show()


def load_generated_dataset(data_dir):
    """加载生成的数据集"""
    # 加载训练数据
    train_signals = np.load(os.path.join(data_dir, 'train_signals.npy'))
    with open(os.path.join(data_dir, 'train_labels.json'), 'r') as f:
        train_labels = json.load(f)
    
    # 加载验证数据
    val_signals = np.load(os.path.join(data_dir, 'val_signals.npy'))
    with open(os.path.join(data_dir, 'val_labels.json'), 'r') as f:
        val_labels = json.load(f)
        
    # 加载数据集信息
    with open(os.path.join(data_dir, 'dataset_info.json'), 'r') as f:
        dataset_info = json.load(f)
        
    return (train_signals, train_labels), (val_signals, val_labels), dataset_info


def main():
    """主函数，生成并保存数据集"""
    NUM_SAMPLES = 2000
    OUTPUT_DIR = 'sin_wave_dataset'
    
    print("开始生成数据集...")
    generator = SinWaveDatasetGenerator(num_samples=NUM_SAMPLES, output_dir=OUTPUT_DIR)
    
    # 生成数据
    generator.generate_and_label_data()
    
    # 分割并保存
    generator.split_and_save(train_split=0.8)
    
    print("\n数据集生成完成！")
    
    # 可视化一个训练样本和一个验证样本
    print("显示示例样本...")
    try:
        if generator.train_data and len(generator.train_data[0]) > 0:
            generator.visualize_sample(index=0, dataset_type='train')
        if generator.val_data and len(generator.val_data[0]) > 0:
            generator.visualize_sample(index=0, dataset_type='val')
    except IndexError:
        print("无法可视化样本，可能是因为未能成功生成任何带标签的样本。")


class SinWaveDataset(Dataset):
    """Sin波峰检测数据集类"""
    def __init__(self, dataset_path, split='train', input_length=1024, transform=None):
        """
        初始化数据集
        
        Args:
            dataset_path: 数据集路径
            split: 数据集分割 ('train' 或 'val')
            input_length: 输入序列长度
            transform: 数据变换
        """
        self.dataset_path = dataset_path
        self.split = split
        self.input_length = input_length
        self.transform = transform
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据集"""
        if self.split == 'train':
            signals_file = os.path.join(self.dataset_path, 'train_signals.npy')
            labels_file = os.path.join(self.dataset_path, 'train_labels.json')
        else:
            signals_file = os.path.join(self.dataset_path, 'val_signals.npy')
            labels_file = os.path.join(self.dataset_path, 'val_labels.json')
        
        # 检查文件是否存在
        if not os.path.exists(signals_file):
            raise FileNotFoundError(f"信号文件不存在: {signals_file}")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"标签文件不存在: {labels_file}")
        
        # 加载数据
        self.signals = np.load(signals_file)
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        print(f"加载 {self.split} 数据集: {len(self.signals)} 样本")
    
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


if __name__ == "__main__":
    main() 