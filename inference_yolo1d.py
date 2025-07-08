import torch
import numpy as np
import matplotlib.pyplot as plt
from yolo1d_model import create_yolo1d_model
import json
import argparse
import torch.nn.functional as F


class YOLO1DInference:
    """YOLO1D推理类"""
    
    def __init__(self, model_path, config_path=None, device='auto'):
        """
        Args:
            model_path: 模型权重路径
            config_path: 配置文件路径
            device: 设备 ('auto', 'cpu', 'cuda')
        """
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载模型
        self.model, self.config = self._load_model(model_path, config_path)
        self.model.eval()
        
        print(f"模型加载完成，设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self, model_path, config_path):
        """加载模型和配置"""
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'config' in checkpoint:
            config = checkpoint['config']
        elif config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError("无法找到模型配置，请提供config_path或使用包含配置的检查点")

        # 创建模型
        model = create_yolo1d_model(
            model_size=config.get('model_size', 'n'),
            num_classes=config.get('num_classes', 1),
            input_channels=config.get('input_channels', 1),
            input_length=config.get('input_length', 1024)
        )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容旧格式
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        
        # 为模型的检测头设置stride
        if hasattr(model, 'detect') and hasattr(model.detect, 'stride'):
            s = config['input_length']
            # 必须临时切换到训练模式，以获取原始特征图用于stride计算
            model.train()
            with torch.no_grad():
                dummy_input = torch.zeros(1, config['input_channels'], s, device=self.device)
                features = model(dummy_input)
                model.detect.stride = torch.tensor([s / x.shape[-1] for x in features]).to(self.device)
            # 在__init__方法中，此函数返回后会立即调用model.eval()，恢复评估模式

        return model, config
    
    def preprocess(self, signal):
        """预处理输入信号"""
        if isinstance(signal, np.ndarray):
            signal = torch.FloatTensor(signal)
        
        # 确保正确的形状 [batch, channels, length]
        if signal.dim() == 1:
            signal = signal.unsqueeze(0).unsqueeze(0)  # [length] -> [1, 1, length]
        elif signal.dim() == 2:
            if signal.shape[0] == 1:
                signal = signal.unsqueeze(0)  # [1, length] -> [1, 1, length]
            else:
                signal = signal.unsqueeze(1)  # [batch, length] -> [batch, 1, length]
        
        # 调整长度到模型期望的输入长度
        target_length = self.config['input_length']
        current_length = signal.shape[-1]
        
        if current_length != target_length:
            if current_length > target_length:
                # 截断
                signal = signal[..., :target_length]
            else:
                # 填充
                pad_size = target_length - current_length
                signal = torch.nn.functional.pad(signal, (0, pad_size), mode='constant', value=0)
        
        return signal.to(self.device)
    
    def postprocess(self, outputs, conf_thresh=0.25, iou_thresh=0.45, min_width_pixels=10):
        """
        对模型输出进行后处理，包括解码、过滤和NMS。
        """
        all_detections = [torch.empty((0, 6), device=self.device) for _ in range(outputs[0].shape[0])] # 6: x1, x2, score, class, cx, w
        
        for i, pred_layer in enumerate(outputs):
            stride = self.model.detect.stride[i]
            batch_size, num_channels, seq_len = pred_layer.shape
            
            # [B, C, L] -> [B, L, C]
            pred = pred_layer.view(batch_size, num_channels, seq_len).permute(0, 2, 1).contiguous()
            
            # 分离预测: [B, L, reg_max*2] 和 [B, L, num_classes]
            pred_box, pred_cls = pred.split((self.model.detect.reg_max * 2, self.model.detect.nc), dim=-1)
            
            # 解码边界框 (DFL)
            # [B, L, reg_max*2] -> [B, L, 2, reg_max]
            pred_box = pred_box.view(batch_size, seq_len, 2, self.model.detect.reg_max)
            # 使用softmax和卷积解码
            dist_pts = torch.arange(self.model.detect.reg_max, device=self.device).float()
            decoded_box = F.softmax(pred_box, dim=-1) @ dist_pts
            # [B, L, 2] -> (dist_left, dist_right)

            # 获取锚点/网格点
            grid = torch.arange(seq_len, device=self.device).float().view(1, -1, 1)
            
            # 解码为 x1, x2 坐标
            x1 = (grid - decoded_box[..., 0:1]) * stride
            x2 = (grid + decoded_box[..., 1:2]) * stride
            
            # 转换为中心点和宽度
            cx = (x1 + x2) / 2
            w = x2 - x1
            
            # [B, L, 1, 2]
            boxes = torch.cat((x1, x2), dim=-1)
            
            # 获取分数和类别ID
            scores, class_ids = torch.max(torch.sigmoid(pred_cls), dim=-1, keepdim=True)
            
            # 过滤低置信度
            detections = torch.cat((boxes, scores, class_ids.float(), cx, w), dim=-1) # [B, L, 6]
            
            for b in range(batch_size):
                batch_dets = detections[b] # [L, 6]
                conf_mask = batch_dets[:, 2] >= conf_thresh
                batch_dets = batch_dets[conf_mask]
                
                # 过滤掉宽度过小的检测
                width_mask = batch_dets[:, 5] >= min_width_pixels
                batch_dets = batch_dets[width_mask]
                
                if batch_dets.shape[0] == 0:
                    continue
                
                all_detections[b] = torch.cat((all_detections[b], batch_dets))

        # 对每个批次的所有检测结果进行NMS
        final_detections = []
        for b in range(len(all_detections)):
            dets = all_detections[b]
            if dets.shape[0] == 0:
                final_detections.append([])
                continue

            # 按类别进行NMS
            unique_classes = dets[:, 3].unique()
            
            batch_final_dets = []
            for c in unique_classes:
                class_mask = dets[:, 3] == c
                class_dets = dets[class_mask]

                # 应用NMS
                keep_indices = self._nms_1d(class_dets[:, :2], class_dets[:, 2], iou_thresh)
                
                kept_dets = class_dets[keep_indices]

                for det in kept_dets:
                    batch_final_dets.append({
                        'x1': det[0].item(),
                        'x2': det[1].item(),
                        'confidence': det[2].item(),
                        'class': int(det[3].item()),
                        'x_center': det[4].item() / self.config['input_length'], # 归一化
                        'width': det[5].item() / self.config['input_length'],   # 归一化
                    })
            
            # 按置信度排序
            batch_final_dets = sorted(batch_final_dets, key=lambda x: x['confidence'], reverse=True)
            final_detections.append(batch_final_dets)
            
        return final_detections
    
    def _nms_1d(self, boxes, scores, iou_thresh):
        """1D非极大值抑制 (boxes are [x1, x2])"""
        if boxes.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # 按分数排序
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep = []
        while len(sorted_indices) > 0:
            # 保留当前最高分数的框
            current_idx = sorted_indices[0]
            keep.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # 计算IoU
            current_box = boxes[current_idx]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            # 计算交集
            inter_x1 = torch.max(current_box[0], remaining_boxes[:, 0])
            inter_x2 = torch.min(current_box[1], remaining_boxes[:, 1])
            inter = (inter_x2 - inter_x1).clamp(0)
            
            # 计算并集
            area_current = current_box[1] - current_box[0]
            area_remaining = remaining_boxes[:, 1] - remaining_boxes[:, 0]
            union = area_current + area_remaining - inter
            
            # IoU
            iou = inter / (union + 1e-7)
            
            # 保留IoU小于阈值的框
            valid_mask = iou <= iou_thresh
            sorted_indices = sorted_indices[1:][valid_mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def predict(self, signal, conf_thresh=0.25, iou_thresh=0.45):
        """预测单个信号"""
        # 预处理
        input_tensor = self.preprocess(signal)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # 后处理
        return self.postprocess(outputs, conf_thresh, iou_thresh)
    
    def predict_batch(self, signals, conf_thresh=0.25, iou_thresh=0.45):
        """预测一批信号"""
        # 预处理
        input_tensor = self.preprocess(signals)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # 后处理
        return self.postprocess(outputs, conf_thresh, iou_thresh)
    
    def visualize_prediction(self, signal, detections, title="YOLO1D Prediction", save_path=None):
        """可视化预测结果"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18, 7))
        
        # 确保signal是numpy数组
        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()
        
        # 如果是多维，取第一个样本
        if signal.ndim > 1:
            signal = signal[0] if signal.shape[0] == 1 else signal.flatten()
        
        # 绘制原始信号
        time_axis = np.arange(len(signal))
        ax.plot(time_axis, signal, 'b-', linewidth=1, alpha=0.7, label='Signal')
        
        # 获取类别名称
        class_names = self.config.get('class_names', [f'class_{i}' for i in range(self.config['num_classes'])])
        
        # 检查detections是否是列表的列表
        if detections and isinstance(detections[0], list):
            # 如果是批处理结果，只取第一个样本进行可视化
            if not detections[0]:
                 print("No detections for the first sample in the batch.")
                 det_list = []
            else:
                 det_list = detections[0]
        else:
            # 兼容旧格式（直接是检测字典的列表）
            det_list = detections

        # 定义颜色映射
        colors = plt.colormaps['tab10'].resampled(len(class_names))
        
        # 用于图例的代理艺术家
        legend_handles = {
            'Signal': plt.Line2D([0], [0], color='royalblue', lw=2)
        }

        for det in det_list:
            class_id = det['class']
            conf = det['confidence']
            
            # 反归一化坐标
            x_center_abs = det['x_center'] * len(signal)
            width_abs = det['width'] * len(signal)
            x1 = x_center_abs - width_abs / 2
            x2 = x_center_abs + width_abs / 2
            
            color = colors(class_id)
            class_name = class_names[class_id]
            
            # 绘制检测框
            ax.axvspan(x1, x2, color=color, alpha=0.4)
            
            # 添加文本标签
            label_text = f"cls {class_id} C:{conf:.2f} W:{width_abs:.1f}"
            ax.text(x_center_abs, ax.get_ylim()[1] * 0.95, label_text, 
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc=color, ec='none', alpha=0.8))

            # 创建图例句柄
            if class_name not in legend_handles:
                 legend_handles[class_name] = plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.5)

        # 设置图表标题和标签
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        
        # 创建并显示图例
        ax.legend(handles=legend_handles.values(), labels=legend_handles.keys(), loc='upper right', title="Legend")

        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # 打印检测结果
        print(f"\n检测到 {len(det_list)} 个目标:")
        for i, det in enumerate(det_list):
            class_id = det['class']
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            print(f"目标 {i+1}:")
            print(f"  类别: {class_name}")
            print(f"  置信度: {det['confidence']:.4f}")
            print(f"  中心位置: {det['x_center'] * self.config['input_length']:.1f} steps")
            print(f"  宽度: {det['width'] * self.config['input_length']:.1f} steps")
            print(f"  边界: [{det['x1']:.4f}, {det['x2']:.4f}]")


def generate_test_signal(length=1024, signal_type='anomaly'):
    """生成测试信号"""
    t = np.linspace(0, 10, length)
    
    if signal_type == 'anomaly':
        # 正常信号 + 异常
        signal = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 2 * t)
        
        # 添加异常
        start = length // 3
        end = start + length // 8
        signal[start:end] += 2.0
        
        # 添加噪声
        signal += 0.1 * np.random.randn(length)
        
    elif signal_type == 'multiple':
        # 多个异常
        signal = np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.random.randn(length)
        
        # 异常1
        start1 = length // 4
        end1 = start1 + length // 10
        signal[start1:end1] += 1.5
        
        # 异常2
        start2 = 3 * length // 4
        end2 = start2 + length // 12
        signal[start2:end2] -= 1.8
        
    else:
        # 纯正常信号
        signal = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 2 * t)
        signal += 0.1 * np.random.randn(length)
    
    return signal


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO1D Inference")
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--config', type=str, default=None, help='Path to the model config file (optional).')
    parser.add_argument('--signal-length', type=int, default=1024, help='Length of the test signal.')
    parser.add_argument('--signal-type', type=str, default='anomaly', 
                        choices=['simple', 'multi_freq', 'noise', 'anomaly', 'flat'],
                        help='Type of test signal to generate.')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='Confidence threshold for detection.')
    parser.add_argument('--iou-thresh', type=float, default=0.45, help='IoU threshold for NMS.')
    parser.add_argument('--save-path', type=str, default='prediction.png', help='Path to save the visualization.')
    
    args = parser.parse_args()
    
    try:
        # 创建推理器
        inference_engine = YOLO1DInference(model_path=args.model, config_path=args.config)
        
        # 生成测试信号
        test_signal = generate_test_signal(length=args.signal_length, signal_type=args.signal_type)
        
        # 执行预测
        detections = inference_engine.predict(
            test_signal,
            conf_thresh=args.conf_thresh,
            iou_thresh=args.iou_thresh
        )
        
        # 可视化结果
        inference_engine.visualize_prediction(
            test_signal,
            detections,
            title=f"Prediction on '{args.signal_type}' signal",
            save_path=args.save_path
        )
        
        # 打印检测结果
        print("\nDetected objects:")
        if detections and detections[0]:
            class_names = inference_engine.config.get('class_names', ['N/A'])
            for det in detections[0]:
                class_id = det['class']
                class_name = class_names[class_id] if class_id < len(class_names) else 'N/A'
                print(f"  - Class: {class_name} (ID: {class_id}), "
                      f"Confidence: {det['confidence']:.4f}, "
                      f"Position: {det['x_center'] * inference_engine.config['input_length']:.1f} steps")
        else:
            print("  No objects detected.")

    except Exception as e:
        print(f"推理失败: {e}")
        print("请确保模型文件存在且格式正确")


if __name__ == "__main__":
    main() 