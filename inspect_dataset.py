import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from dataset_generator import load_generated_dataset

def visualize_sample(signal, labels, index, dataset_type, class_names):
    """
    可视化单个样本及其标签。

    Args:
        signal (np.ndarray): 信号数据。
        labels (list): 该信号对应的标签列表。
        index (int): 样本的索引。
        dataset_type (str): 数据集类型 ('train' or 'val')。
        class_names (list): 类别名称列表。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 7))

    # 绘制原始信号
    ax.plot(signal, label='Signal', color='blue', alpha=0.8)

    # 定义颜色 (0: 波峰-红色, 1: 波谷-绿色)
    colors = ['red', 'green']
    
    # 用于图例的代理艺术家
    legend_handles = {}

    for i, label in enumerate(labels):
        class_id_float, x_center_norm, width_norm = label
        class_id = int(class_id_float)
        
        # 反归一化
        sequence_length = len(signal)
        x_center = x_center_norm * sequence_length
        width = width_norm * sequence_length
        x1 = x_center - width / 2
        x2 = x_center + width / 2
        
        # 确保class_id有效
        if class_id < 0 or class_id >= len(colors):
            print(f"  - Warning: Invalid class_id {class_id} found in sample {index}. Skipping label.")
            continue
            
        color = colors[class_id]
        event_name = class_names[class_id]
        
        # 绘制标注区域
        ax.axvspan(x1, x2, color=color, alpha=0.3)
        ax.axvline(x_center, color=color, linestyle='--', alpha=0.7)
        
        # 创建图例句柄
        if event_name not in legend_handles:
            legend_handles[event_name] = plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.4)

    # 设置图表标题和标签
    ax.set_title(f'Inspecting Sample #{index} from {dataset_type.upper()} set', fontsize=16)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)

    # 创建并显示图例
    if legend_handles:
        # 获取原始信号的图例
        signal_handle, signal_label = ax.get_legend_handles_labels()

        # 合并图例
        handles = signal_handle + [legend_handles[name] for name in class_names if name in legend_handles]
        labels = signal_label + [name for name in class_names if name in legend_handles]
        
        ax.legend(handles=handles, labels=labels, loc='upper right', title="Legend")
    else:
        ax.legend(loc='upper right')

    plt.show()


def main():
    """主函数，用于加载并逐个可视化数据集样本"""
    parser = argparse.ArgumentParser(description="Tool to visually inspect dataset labels.")
    parser.add_argument(
        '--path', 
        type=str, 
        default='sin_wave_dataset', 
        help='Path to the generated dataset directory.'
    )
    parser.add_argument(
        '--set', 
        type=str, 
        default='train', 
        choices=['train', 'val'],
        help="Which dataset set to inspect ('train' or 'val')."
    )
    
    args = parser.parse_args()

    # 检查数据集是否存在
    if not os.path.isdir(args.path):
        print(f"Error: Dataset directory not found at '{args.path}'")
        print("Please run dataset_generator.py first.")
        return

    try:
        # 加载数据集
        (train_signals, train_labels), (val_signals, val_labels), info = load_generated_dataset(args.path)
        print("Dataset loaded successfully.")
        print(f"Class names: {info['class_names']}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.set == 'train':
        signals_to_inspect = train_signals
        labels_to_inspect = train_labels
    else:
        signals_to_inspect = val_signals
        labels_to_inspect = val_labels
        
    num_samples = len(signals_to_inspect)
    print(f"\nNow inspecting {num_samples} samples from the '{args.set}' set.")
    print("Press 'Enter' to see the next sample, or type 'q' and 'Enter' to quit.")

    # 逐个显示样本
    for i in range(num_samples):
        signal = signals_to_inspect[i]
        labels = labels_to_inspect[i]
        
        print("-" * 50)
        print(f"Displaying sample {i+1}/{num_samples}...")
        
        # 可视化
        visualize_sample(signal, labels, i + 1, args.set, info['class_names'])
        
        # 交互式提示
        user_input = input("Press Enter for next, 'q' to quit: ").strip().lower()
        if user_input == 'q':
            print("Exiting inspection tool.")
            break
            
    print("\nInspection finished.")

if __name__ == "__main__":
    main() 