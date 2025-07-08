import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YOLO1DLoss(nn.Module):
    """YOLO 1D损失函数"""
    def __init__(self, num_classes=80, reg_max=16, use_dfl=True, hyp=None):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        
        # 损失权重
        self.hyp = {
            'box': 7.5,      # 边界框损失权重 (原yolov8: 7.5)
            'cls': 0.5,       # 分类损失权重 (原yolov8: 0.5)
            'dfl': 1.5,       # DFL损失权重 (原yolov8: 1.5)
        }
        if hyp:
            self.hyp.update(hyp)
        
        # 损失函数
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, preds, targets, model):
        """
        Args:
            preds: 模型预测结果 [P3, P4, P5]
            targets: 真实标签 [image_idx, class, x_center, width]
            model: 模型实例
        """
        device = preds[0].device
        loss_box = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        
        # 如果没有目标，返回零损失
        if targets.shape[0] == 0:
            return loss_box + loss_cls + loss_dfl, torch.stack([loss_box, loss_cls, loss_dfl, loss_box + loss_cls + loss_dfl])
        
        # 构建目标张量
        targets_out = self.build_targets(preds, targets, model)
        
        # 遍历每个检测层
        for i, pred in enumerate(preds):
            b, c, l = pred.shape  # batch, channels, length
            
            # 重塑预测
            pred = pred.view(b, self.reg_max * 2 + self.num_classes, l).permute(0, 2, 1).contiguous()
            pred_box = pred[..., :self.reg_max * 2]
            pred_cls = pred[..., self.reg_max * 2:]
            
            # 获取该层的目标
            targets_i = targets_out[i]
            if targets_i.shape[0] == 0:
                # 如果没有目标，添加背景类损失
                if self.num_classes > 1:
                    # 所有位置都是背景
                    cls_targets = torch.zeros_like(pred_cls[..., 0], device=device, dtype=torch.long)
                    loss_cls += F.cross_entropy(pred_cls.view(-1, self.num_classes), 
                                              cls_targets.view(-1), reduction='mean')
                continue
                
            # 分类损失、边界框损失、DFL损失
            if self.num_classes > 1:
                targets_i_filtered = targets_i[targets_i[:, 0] >= 0] # Filter out placeholder targets
                if targets_i_filtered.shape[0] == 0:
                     if self.num_classes > 0:
                         cls_targets = torch.zeros((b, l), device=device, dtype=torch.long)
                         loss_cls += F.cross_entropy(pred_cls.view(-1, self.num_classes), cls_targets.view(-1), reduction='mean')
                     continue

                # --- 正确的损失计算逻辑 ---
                # 1. 提取所需张量
                gt_cls = targets_i_filtered[:, 2].long()
                gt_bboxes_norm = targets_i_filtered[:, 3:5] # [cx_norm, w_norm] 归一化坐标 (0-1)
                
                b_idx = targets_i_filtered[:, 0].long()
                g_idx = targets_i_filtered[:, 1].long()
                
                # 2. 获取对应位置的预测
                pred_box_logits = pred_box[b_idx, g_idx] # [N, reg_max*2]
                pred_cls_logits = pred_cls[b_idx, g_idx] # [N, num_classes]

                # 3. 分类损失 (只对正样本)
                # 使用标签平滑
                gt_cls_one_hot = F.one_hot(gt_cls, self.num_classes).float()
                gt_cls_one_hot = gt_cls_one_hot * (1.0 - 0.1) + 0.1 / self.num_classes
                loss_cls += self.bce(pred_cls_logits, gt_cls_one_hot).mean()
                
                # 4. 边界框和DFL损失
                # --- 坐标系对齐: 特征图尺度 ---
                # 为了计算IoU和DFL损失, 预测框和真实框都必须转换到相同的"特征图尺度"上。
                # a. 将归一化的GT框 [cx, w] 转换为当前特征图尺度的 [x1, x2]
                gt_cx_norm, gt_w_norm = gt_bboxes_norm.chunk(2, 1)
                gt_cx_feat = gt_cx_norm.squeeze(1) * l # l 是当前特征图的长度
                gt_w_feat = gt_w_norm.squeeze(1) * l
                gt_bboxes_x1x2_feat = torch.stack((gt_cx_feat - gt_w_feat / 2, 
                                                   gt_cx_feat + gt_w_feat / 2), dim=1)

                # b. 将预测的 logits 解码为特征图尺度的 [x1, x2]
                dfl_pts = torch.arange(self.reg_max, device=pred.device).float()
                pred_dists = (F.softmax(pred_box_logits.view(-1, self.reg_max), dim=-1) @ dfl_pts).view(-1, 2)
                grid_x = g_idx.float() # 网格点索引本身就是特征图尺度上的坐标
                pred_bboxes_x1x2_feat = torch.stack((grid_x - pred_dists[:, 0], 
                                                     grid_x + pred_dists[:, 1]), dim=1)

                # IoU损失 (输入都必须在相同的特征图尺度上)
                loss_box += self.bbox_loss(pred_bboxes_x1x2_feat, gt_bboxes_x1x2_feat)

                # DFL损失
                if self.use_dfl:
                    # 计算GT在特征图尺度上的左右距离 [d_l, d_r]
                    target_d_left = grid_x - gt_bboxes_x1x2_feat[:, 0]
                    target_d_right = gt_bboxes_x1x2_feat[:, 1] - grid_x
                    target_dists = torch.stack((target_d_left, target_d_right), dim=1).clamp(0, self.reg_max - 1.01)

                    # 编码GT距离
                    target_dist_encoded = self.encode_bbox(target_dists)
                    loss_dfl += self.df_loss(pred_box_logits, target_dist_encoded.view(-1, 2 * self.reg_max))
        
        # 总损失
        total_loss = (loss_box * self.hyp['box'] + 
                     loss_cls * self.hyp['cls'] + 
                     loss_dfl * self.hyp['dfl'])
        
        return total_loss, torch.stack([loss_box, loss_cls, loss_dfl, total_loss]).detach()
    
    def build_targets(self, preds, targets, model):
        """
        构建训练目标, 将归一化的输入标签匹配到对应的多层特征图上。
        
        Args:
            preds (list): 模型输出的各层特征图。
            targets (torch.Tensor): 归一化的真实标签 [b_idx, cls, cx_norm, w_norm]。
            model (nn.Module): 模型实例。

        Returns:
            list: 每一层特征图对应的目标列表。
                  每个目标格式: [b_idx, grid_idx, cls, cx_norm, w_norm]
                  其中 grid_idx 是在特征图尺度上的索引。
        """
        targets_out = []
        
        # 获取每层的stride（简化处理）
        strides = [8, 16, 32]  # 默认stride
        
        for i, pred in enumerate(preds):
            b, c, l = pred.shape
            stride = strides[i] if i < len(strides) else 32
            
            # 创建一个占位符，以确保targets_out总是有正确的长度
            targets_per_layer = torch.full((1, 5), -1.0, device=targets.device)

            # 将targets缩放到当前特征图尺度并添加位置索引
            if targets.shape[0] > 0:
                targets_scaled = []
                for target in targets:
                    batch_idx = target[0]
                    class_id = target[1]
                    x_center_norm = target[2] # 归一化中心点 (0-1)
                    width_norm = target[3]    # 归一化宽度 (0-1)
                    
                    # 将归一化坐标转换为当前特征图尺度上的坐标
                    x_center_feat = x_center_norm * l # l 是当前特征图的长度
                    
                    # 找到最近的网格点 (即在特征图尺度上的索引)
                    grid_x = int(x_center_feat)
                    
                    # --- 多点匹配策略 ---
                    # 一个目标可以匹配到中心点及其左右邻近的网格
                    for offset in [-1, 0, 1]:
                        g_idx_offset = grid_x + offset
                        if 0 <= g_idx_offset < l and width_norm > 0:
                            # 注意: 此处保存的仍是归一化的中心点和宽度，
                            # 因为最终的损失计算需要它们。
                            targets_scaled.append([batch_idx, g_idx_offset, class_id, x_center_norm, width_norm])
                
                if targets_scaled:
                    # 去重，以防边缘情况导致重复添加
                    targets_per_layer = torch.tensor(list(set(map(tuple, targets_scaled))), device=targets.device)
            
            targets_out.append(targets_per_layer)
        
        return targets_out
    
    def decode_bbox(self, pred_box):
        """解码边界框预测"""
        if self.use_dfl and pred_box.shape[-1] == self.reg_max * 2:
            # DFL解码
            pred_box = pred_box.view(-1, 2, self.reg_max)
            pred_box = F.softmax(pred_box, dim=2)
            conv_weight = torch.arange(self.reg_max, device=pred_box.device, dtype=pred_box.dtype).view(1, 1, -1)
            pred_box = (pred_box * conv_weight).sum(dim=2)
        elif pred_box.shape[-1] == 2:
            # 直接预测
            pass
        else:
            # 如果维度不匹配，取前两个维度
            pred_box = pred_box[..., :2]
        
        return pred_box
    
    def encode_bbox(self, target_dists):
        """编码边界框目标为DFL分布"""
        if self.use_dfl:
            # 转换为DFL目标分布
            device = target_dists.device
            target_encoded = torch.zeros(target_dists.shape[0], 2, self.reg_max, device=device)
            
            for i in range(2):  # d_left, d_right
                coord = target_dists[:, i].clamp(0, self.reg_max - 1.01) # 增加稳定性
                coord_floor = coord.floor().long()
                coord_ceil = coord_floor + 1
                
                # 双线性分布
                weight_floor = coord_ceil.float() - coord
                weight_ceil = coord - coord_floor.float()
                
                # 设置权重
                for j in range(target_dists.shape[0]):
                    if coord_floor[j] < self.reg_max:
                        target_encoded[j, i, coord_floor[j]] = weight_floor[j]
                    if coord_ceil[j] < self.reg_max:
                        target_encoded[j, i, coord_ceil[j]] = weight_ceil[j]
            
            return target_encoded
        else:
            # 如果不使用DFL，则此函数不应被调用
            return target_dists
    
    def bbox_loss(self, pred_bboxes, target_bboxes):
        """边界框IoU损失 (输入为 x1, x2 格式)"""
        # 计算1D IoU
        pred_x1 = pred_bboxes[:, 0]
        pred_x2 = pred_bboxes[:, 1]
        
        target_x1 = target_bboxes[:, 0]
        target_x2 = target_bboxes[:, 1]
        
        # 计算交集
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter = torch.clamp(inter_x2 - inter_x1, min=0)
        
        # 计算并集
        union = (pred_x2 - pred_x1) + (target_x2 - target_x1) - inter
        
        # IoU
        iou = inter / (union + 1e-7)
        
        # 返回1 - IoU作为损失
        return (1.0 - iou).mean()
    
    def df_loss(self, pred_dist, target_dist):
        """分布式焦点损失 (DFL).
        Args:
            pred_dist (torch.Tensor): 预测分布的logits, shape [N, 2 * reg_max].
            target_dist (torch.Tensor): 目标分布的概率, shape [N, 2 * reg_max].
        """
        if target_dist.shape[0] == 0:
            return torch.tensor(0.0, device=pred_dist.device)
            
        # 将输入变形以分别处理左右分布
        # pred_dist_reshaped -> [N * 2, reg_max]
        pred_dist_reshaped = pred_dist.view(-1, self.reg_max)
        # target_dist_reshaped -> [N * 2, reg_max]
        target_dist_reshaped = target_dist.view(-1, self.reg_max)
        
        # 计算交叉熵损失。对于软目标(概率分布)，这等价于 -(target * log_softmax(pred)).sum()
        # reduction='none' 会为 N*2 个分布中的每一个都计算损失
        loss_per_dist = F.cross_entropy(pred_dist_reshaped, target_dist_reshaped, reduction='none')
        
        # 将损失变形回 [N, 2] 以便对每个样本的左右损失求和
        loss_summed = loss_per_dist.view(-1, 2).sum(dim=1)
        
        # 对所有正样本求平均
        return loss_summed.mean()


class FocalLoss(nn.Module):
    """Focal Loss实现"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_loss(preds, targets, model, loss_fn):
    """计算损失的便捷函数"""
    return loss_fn(preds, targets, model) 