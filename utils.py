import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
import torch.nn.functional as F

def plot_training_curves(train_losses, val_losses, mAP, output_path='training_curves.png'):
    """
    绘制训练和验证损失以及mAP曲线。

    Args:
        train_losses (list): 训练损失列表。
        val_losses (list): 验证损失列表。
        mAP (list): mAP分数列表。
        output_path (str): 图像保存路径。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制mAP曲线
    ax2.plot(mAP, label='mAP@0.5', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.set_title('Validation mAP@0.5')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # 关闭图像，防止显示

def plot_validation_predictions(signal, gt_labels, pred_labels, class_names, title="Validation Prediction"):
    """
    为TensorBoard生成验证集预测的可视化图像。

    Args:
        signal (np.ndarray): 原始信号。
        gt_labels (dict): 真实标签。
        pred_labels (dict): 预测标签。
        class_names (list): 类别名称列表。
        title (str): 图像标题。

    Returns:
        matplotlib.figure.Figure: 生成的图像对象。
    """
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(signal, color='royalblue', label='Signal', alpha=0.8)

    colors = plt.cm.get_cmap('tab10', len(class_names))

    # 绘制真实标签 (Ground Truth)
    # xyxy -> x1, x2
    gt_x1 = gt_labels['boxes'][:, 0]
    gt_x2 = gt_labels['boxes'][:, 2]

    for i, box in enumerate(zip(gt_x1, gt_x2)):
        class_id = int(gt_labels['labels'][i])
        x1, x2 = box
        # Create a proxy artist for the legend if it doesn't exist
        proxy_label = f'GT: {class_names[class_id]}'
        if proxy_label not in [h.get_label() for h in ax.get_legend_handles_labels()[0]]:
            ax.axvspan(x1, x2, color=colors(class_id), alpha=0.2, label=proxy_label)
        else:
            ax.axvspan(x1, x2, color=colors(class_id), alpha=0.2)

    # 绘制预测标签 (Predictions)
    if pred_labels['boxes'].shape[0] > 0:
        # xyxy -> x1, x2
        pred_x1 = pred_labels['boxes'][:, 0]
        pred_x2 = pred_labels['boxes'][:, 2]

        for i, box in enumerate(zip(pred_x1, pred_x2)):
            class_id = int(pred_labels['labels'][i])
            score = pred_labels['scores'][i]
            x1, x2 = box

            proxy_label = f'Pred: {class_names[class_id]}'
            if proxy_label not in [h.get_label() for h in ax.get_legend_handles_labels()[0]]:
                 ax.axvspan(x1.item(), x2.item(), color=colors(class_id), alpha=0.6, hatch='//', label=proxy_label)
            else:
                 ax.axvspan(x1.item(), x2.item(), color=colors(class_id), alpha=0.6, hatch='//')

            label_text = f"{class_names[class_id]}: {score:.2f}"
            ax.text(((x1 + x2) / 2).item(), ax.get_ylim()[1] * 0.9, label_text, ha='center', va='top', fontsize=9, color='white', 
                    bbox=dict(boxstyle='round,pad=0.2', fc=colors(class_id), ec='none'))

    ax.set_title(title)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Amplitude")
    
    # 创建一个干净的图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    return fig

def postprocess_for_metrics(outputs, model, conf_thresh=0.25, iou_thresh=0.45, max_pre_nms=1000, max_post_nms=100):
    """
    对模型输出进行后处理，专门为torchmetrics格式化，并包含高级NMS。
    
    重要: 此函数输出的边界框坐标是绝对坐标 (相对于input_length)，而不是归一化坐标。
    例如，坐标范围为 [0, 1024]，而不是 [0, 1]。

    Args:
        outputs (list): 来自模型原始输出的特征图列表。
        model (nn.Module): 模型实例，用于访问stride等属性。
        conf_thresh (float): 置信度阈值。
        iou_thresh (float): NMS的IoU阈值。
        max_pre_nms (int): NMS前保留的最大框数。
        max_post_nms (int): NMS后保留的最大框数。
        
    Returns:
        list[dict]: 每个样本一个字典，包含'boxes', 'scores', 'labels'。
                    'boxes' 的坐标是绝对坐标 [x1, 0, x2, 1]。
    """
    all_preds_by_batch = []
    num_batches = outputs[0].shape[0]

    for b in range(num_batches):
        # 收集该batch item在所有特征层上的预测
        batch_preds = []
        for i, pred_layer in enumerate(outputs):
            stride = model.detect.stride[i].item()
            nc = model.detect.nc
            reg_max = model.detect.reg_max
            
            # [B, C, L] -> [1, C, L]
            pred = pred_layer[b:b+1].view(1, nc + reg_max * 2, -1).permute(0, 2, 1).contiguous()
            pred_box, pred_cls = pred.split((reg_max * 2, nc), dim=-1)
            
            dist_pts = torch.arange(reg_max, device=pred.device).float()
            decoded_box = (F.softmax(pred_box.view(1, -1, 2, reg_max), dim=-1) @ dist_pts).squeeze(0)
            
            # 将解码后的距离转换为绝对坐标 (相对于input_length)
            grid = torch.arange(pred.shape[1], device=pred.device).float()
            x1 = (grid - decoded_box[:, 0]) * stride
            x2 = (grid + decoded_box[:, 1]) * stride
            
            # 修复：正确计算分类分数
            cls_scores = torch.sigmoid(pred_cls.squeeze(0))  # [L, nc]
            scores, class_ids = torch.max(cls_scores, dim=-1)  # [L], [L]
            
            # [N, 4] -> [x1, x2, score, class_id]
            batch_preds.append(torch.stack([x1, x2, scores, class_ids.float()], dim=-1))

        # --- 高级NMS流程 ---
        # 1. 合并所有层的预测
        if batch_preds:
            batch_preds = torch.cat(batch_preds, 0)
        else:
            all_preds_by_batch.append({'boxes': torch.empty(0, 4, device=outputs[0].device), 'scores': torch.empty(0, device=outputs[0].device), 'labels': torch.empty(0, dtype=torch.long, device=outputs[0].device)})
            continue
        
        # 2. 过滤低置信度
        conf_mask = batch_preds[:, 2] >= conf_thresh
        batch_preds = batch_preds[conf_mask]
        
        # 3. 限制进入NMS前的框数量
        if batch_preds.shape[0] > max_pre_nms:
            batch_preds = batch_preds[batch_preds[:, 2].argsort(descending=True)[:max_pre_nms]]
            
        if batch_preds.shape[0] == 0:
            all_preds_by_batch.append({'boxes': torch.empty(0, 4, device=outputs[0].device), 'scores': torch.empty(0, device=outputs[0].device), 'labels': torch.empty(0, dtype=torch.long, device=outputs[0].device)})
            continue

        # 4. 按类别进行NMS
        unique_classes = batch_preds[:, 3].unique()
        final_dets = []
        for c in unique_classes:
            class_mask = batch_preds[:, 3] == c
            class_dets = batch_preds[class_mask]
            
            nms_boxes = torch.stack([class_dets[:, 0], torch.zeros_like(class_dets[:, 0]), class_dets[:, 1], torch.ones_like(class_dets[:, 0])], dim=1)
            keep_indices = torchvision.ops.nms(nms_boxes, class_dets[:, 2], iou_thresh)
            final_dets.append(class_dets[keep_indices])
            
        if not final_dets:
            all_preds_by_batch.append({'boxes': torch.empty(0, 4, device=outputs[0].device), 'scores': torch.empty(0, device=outputs[0].device), 'labels': torch.empty(0, dtype=torch.long, device=outputs[0].device)})
            continue
            
        final_dets = torch.cat(final_dets)

        # 5. 限制最终检测数量
        if final_dets.shape[0] > max_post_nms:
            final_dets = final_dets[final_dets[:, 2].argsort(descending=True)[:max_post_nms]]
        
        # 6. 格式化为torchmetrics所需格式 [x1, y1, x2, y2]
        # 注意: 这里的坐标已经是绝对坐标
        final_boxes_xyxy = torch.stack([
            final_dets[:, 0], 
            torch.zeros_like(final_dets[:, 0]), 
            final_dets[:, 1], 
            torch.ones_like(final_dets[:, 1])
        ], dim=1)

        all_preds_by_batch.append({
            'boxes': final_boxes_xyxy,
            'scores': final_dets[:, 2],
            'labels': final_dets[:, 3].long()
        })
            
    return all_preds_by_batch