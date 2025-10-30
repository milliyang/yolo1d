# YOLO1D 项目坐标系详解 (coor.md)

本文档详细解释了`yolo1d`项目中使用的三种核心坐标系，以及它们在数据处理、模型训练和评估中的转换关系。理解这些坐标系是维护和迭代本项目的关键。

---

## 1. 三大核心坐标系

项目主要涉及以下三种坐标系：

### a. 绝对坐标 (Absolute Coordinates)

- **定义**: 相对于原始输入信号的位置索引。
- **范围**: `[0, input_length - 1]`，例如 `[0, 1023]`。
- **单位**: 时间步 (Time Steps)。
- **应用场景**:
    - `dataset_generator.py`: `scipy.signal.find_peaks` 返回的初始事件位置和宽度。
    - `utils.py`: `postprocess_for_metrics` 函数输出的最终预测框，用于mAP计算和可视化。
    - `inference_yolo1d.py`: 可视化时展示的最终结果。

### b. 归一化坐标 (Normalized Coordinates)

- **定义**: 相对于输入信号总长度的比例。
- **范围**: `[0.0, 1.0]`。
- **单位**: 无。
- **转换关系**:
    - `coord_norm = coord_abs / input_length`
    - `coord_abs = coord_norm * input_length`
- **应用场景**:
    - **YOLO标准格式**: 这是YOLO系列模型存储和处理标签的标准方式。
    - `dataset_generator.py`: 在将标签保存到文件前，会将绝对坐标转换为归一化坐标。
    - `DataLoader`: 从数据集中加载并传递给训练器的标签（targets）使用归一化坐标。

### c. 特征图坐标 (Feature Map Coordinates)

- **定义**: 相对于模型中特定特征图（Feature Map）的位置索引。
- **范围**: `[0, feature_map_length - 1]`。例如，如果输入长度为1024，步长（stride）为8，则该特征图长度为128，坐标范围为`[0, 127]`。
- **单位**: 网格单元 (Grid Cells)。
- **转换关系**:
    - `coord_feat = coord_abs / stride`
    - `coord_feat = coord_norm * feature_map_length`
- **应用场景**:
    - `yolo1d_loss.py`: 在计算损失函数（IoU Loss, DFL Loss）时，预测和真实标签必须被对齐到同一个特征图尺度上进行比较。

---

## 2. 坐标系的生命周期与转换流程

一个标签（bounding box）在项目中的生命周期如下：

**① 数据生成 (`dataset_generator.py`)**
   - `find_peaks` 产生事件的中心点和宽度，此时为 **绝对坐标**。
   - 在保存到文件前，通过除以 `sequence_length` 将其转换为 **归一化坐标**。

**② 数据加载 (`SinWaveDataset` -> `DataLoader`)**
   - 从文件中读取标签，此时为 **归一化坐标**。
   - `DataLoader` 将包含归一化坐标的 `targets` 张量传递给训练器。

**③ 损失计算 (`yolo1d_loss.py`)**
   - 损失函数接收到模型的原始预测（在 **特征图坐标** 系）和来自数据加载器的真实标签（在 **归一化坐标** 系）。
   - **关键转换**: 在 `build_targets` 和 `forward` 方法中，将归一化的真实标签乘以特征图的长度，转换为 **特征图坐标**。
   - `IoU Loss` 和 `DFL Loss` 的计算在 **特征图坐标** 系下完成。

**④ 指标计算 (`utils.py` -> `trainer_base.py`)**
   - `postprocess_for_metrics` 函数接收多层级的 **特征图坐标** 预测，并通过乘以 `stride` 将它们统一解码为 **绝对坐标**。
   - `_update_map_metric` 方法接收到：
     - 预测框 (来自 `postprocess_for_metrics`): **绝对坐标**
     - 真实框 (来自 `DataLoader`): **归一化坐标**
   - **关键转换**: `_update_map_metric` 必须将真实框的 **归一化坐标** 乘以 `input_length`，转换为 **绝对坐标**。
   - `torchmetrics.detection.MeanAveragePrecision` 在 **绝对坐标** 系下完成mAP的计算。

---

## 3. 关键代码片段

### a. `trainer_base.py`: 对齐mAP计算的坐标系

```python
# 在 _update_map_metric 方法中:
# --- 坐标系对齐 (关键) ---
# postprocess_for_metrics 输出的预测框(preds)是绝对坐标 (0 to input_length)。
# 而从DataLoader加载的标签(targets)是归一化坐标 (0 to 1)。
# 因此，在计算mAP之前，必须将真实标签(GT)的坐标乘以input_length，
# 将其从"归一化"转换为"绝对坐标"，以匹配预测框的坐标系。
input_len = self.config['input_length']
boxes_x1 = (batch_targets[:, 2] - batch_targets[:, 3] / 2) * input_len
boxes_x2 = (batch_targets[:, 2] + batch_targets[:, 3] / 2) * input_len
```

### b. `utils.py`: 从特征图解码到绝对坐标

```python
# 在 postprocess_for_metrics 方法中:
# 将解码后的距离转换为绝对坐标 (相对于input_length)
grid = torch.arange(pred.shape[1], device=pred.device).float() # 特征图坐标
stride = model.detect.stride[i].item() # 步长
x1 = (grid - decoded_box[:, 0]) * stride
x2 = (grid + decoded_box[:, 1]) * stride
```

### c. `yolo1d_loss.py`: 对齐损失计算的坐标系

```python
# 在 forward 方法中:
# --- 坐标系对齐: 特征图尺度 ---
# a. 将归一化的GT框 [cx, w] 转换为当前特征图尺度的 [x1, x2]
gt_cx_norm, gt_w_norm = gt_bboxes_norm.chunk(2, 1)
gt_cx_feat = gt_cx_norm.squeeze(1) * l # l 是当前特征图的长度
gt_w_feat = gt_w_norm.squeeze(1) * l
gt_bboxes_x1x2_feat = torch.stack((gt_cx_feat - gt_w_feat / 2, gt_cx_feat + gt_w_feat / 2), dim=1)

# b. 将预测的 logits 解码为特征图尺度的 [x1, x2]
grid_x = g_idx.float() # 网格点索引本身就是特征图尺度上的坐标
pred_bboxes_x1x2_feat = torch.stack((grid_x - pred_dists[:, 0], grid_x + pred_dists[:, 1]), dim=1)

# IoU损失 (输入都必须在相同的特征图尺度上)
loss_box += self.bbox_loss(pred_bboxes_x1x2_feat, gt_bboxes_x1x2_feat)
``` 