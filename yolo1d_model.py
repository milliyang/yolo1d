import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class Conv1d(nn.Module):
    """1D卷积层，包含BN和激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck1d(nn.Module):
    """1D Bottleneck模块，类似于YOLO v8的C2f模块"""
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv1d(in_channels, hidden_channels, 1)
        self.cv2 = Conv1d(hidden_channels, out_channels, 3, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f1d(nn.Module):
    """1D C2f模块，YOLO v8的核心组件"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, groups=1, expansion=0.5):
        super().__init__()
        self.c = int(out_channels * expansion)
        self.cv1 = Conv1d(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv1d((2 + n) * self.c, out_channels, 1)
        self.m = nn.ModuleList(Bottleneck1d(self.c, self.c, shortcut, groups, expansion=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF1d(nn.Module):
    """1D空间金字塔池化快速版本"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv1d(in_channels, hidden_channels, 1)
        self.cv2 = Conv1d(hidden_channels * 4, out_channels, 1)
        self.m = nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Detect1d(nn.Module):
    """1D检测头，用于分类和边界框回归"""
    def __init__(self, nc=80, ch=(), reg_max=16):
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = reg_max  # DFL通道数
        self.no = nc + self.reg_max * 2  # 每个anchor的输出数
        self.stride = torch.zeros(self.nl)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        
        # 分类和回归分支
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv1d(x, c2, 3), Conv1d(c2, c2, 3), nn.Conv1d(c2, 2 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv1d(x, c3, 3), Conv1d(c3, c3, 3), nn.Conv1d(c3, self.nc, 1)) for x in ch)
        
        # DFL卷积
        self.dfl = DFL1d(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """
        统一前向传播，为训练和推理返回相同格式的原始输出。
        返回一个列表，其中每个张量的形状为 [batch, a, length]，
        a = nc + reg_max * 2 (例如 2 + 16*2 = 34)。
        """
        outputs = []
        for i in range(self.nl):
            outputs.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        return outputs
    
    def inference(self, x):
        """推理时的后处理"""
        dtype = x[0].dtype
        for i in range(self.nl):
            b, _, l = x[i].shape
            x[i] = x[i].view(b, self.no, l).permute(0, 2, 1).contiguous()
            
            if not self.training:
                # 分离边界框和分类预测
                box, cls = x[i].split((2 * self.reg_max, self.nc), 2)
                
                # DFL处理
                if self.reg_max > 1:
                    # 重塑box张量以适配DFL
                    b_box, l_box, c_box = box.shape
                    if c_box == 2 * self.reg_max:
                        # box shape: [batch, length, 2*reg_max] -> [batch, 2*reg_max, length] 
                        box_reshaped = box.permute(0, 2, 1).contiguous()
                        dbox = self.dfl(box_reshaped)
                        # dbox shape: [batch, 2, length] -> [batch, length, 2]
                        dbox = dbox.permute(0, 2, 1).contiguous()
                    else:
                        dbox = box
                else:
                    dbox = box
                
                # 激活分类得分
                cls = cls.sigmoid()
                x[i] = torch.cat((dbox, cls), 2)
        
        return x


class DFL1d(nn.Module):
    """分布式焦点损失，用于1D边界框回归"""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv1d(c1, 1, 1, bias=False)
        # 初始化权重但保持可训练
        with torch.no_grad():
            self.conv.weight.data = torch.arange(c1, dtype=torch.float).view(1, c1, 1)
        self.c1 = c1

    def forward(self, x):
        # x shape: [batch, 2*reg_max, length]
        b, c, l = x.shape
        
        # 确保通道数是2*reg_max的倍数
        if c != 2 * self.c1:
            raise ValueError(f"Expected {2 * self.c1} channels, got {c}")
        
        # 重塑为 [batch, 2, reg_max, length]
        x = x.view(b, 2, self.c1, l)
        
        # 对每个坐标分量应用softmax和加权求和
        # x shape: [batch, 2, reg_max, length]
        # 转置为 [batch, length, 2, reg_max] 然后应用softmax
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, length, 2, reg_max]
        x = F.softmax(x, dim=3)  # 在reg_max维度上softmax
        
        # 加权求和
        weights = torch.arange(self.c1, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        x = (x * weights).sum(dim=3)  # [batch, length, 2]
        
        # 转回 [batch, 2, length]
        x = x.permute(0, 2, 1).contiguous()
        
        return x


class YOLO1D(nn.Module):
    """YOLO 1D模型，用于时域数据分类检测"""
    def __init__(self, 
                 input_channels=1, 
                 num_classes=80, 
                 depth_multiple=1.0, 
                 width_multiple=1.0,
                 input_length=1024,
                 reg_max=16):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 定义通道数（基于width_multiple调整）
        def make_divisible(x, divisor=8):
            return math.ceil(x / divisor) * divisor
        
        channels = [int(make_divisible(x * width_multiple)) for x in [64, 128, 256, 512, 1024]]
        
        # 定义重复次数（基于depth_multiple调整）
        depths = [int(x * depth_multiple) for x in [3, 6, 6, 3]]
        
        # Backbone
        self.backbone = nn.ModuleList([
            # Stage 0: Stem
            Conv1d(input_channels, channels[0], 6, 2, 2),  # 640 -> 320
            
            # Stage 1
            Conv1d(channels[0], channels[1], 3, 2),  # 320 -> 160
            C2f1d(channels[1], channels[1], depths[0]),
            
            # Stage 2
            Conv1d(channels[1], channels[2], 3, 2),  # 160 -> 80
            C2f1d(channels[2], channels[2], depths[1]),
            
            # Stage 3
            Conv1d(channels[2], channels[3], 3, 2),  # 80 -> 40
            C2f1d(channels[3], channels[3], depths[2]),
            
            # Stage 4
            Conv1d(channels[3], channels[4], 3, 2),  # 40 -> 20
            C2f1d(channels[4], channels[4], depths[3]),
            SPPF1d(channels[4], channels[4]),
        ])
        
        # Neck (FPN-like structure)
        self.neck = nn.ModuleList([
            # Upsample path
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2f1d(channels[4] + channels[3], channels[3], depths[3]),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            C2f1d(channels[3] + channels[2], channels[2], depths[2]),
            
            # Downsample path
            Conv1d(channels[2], channels[2], 3, 2),
            C2f1d(channels[2] + channels[3], channels[3], depths[2]),
            
            Conv1d(channels[3], channels[3], 3, 2),
            C2f1d(channels[3] + channels[4], channels[4], depths[1]),
        ])
        
        # Detection head
        self.detect = Detect1d(num_classes, (channels[2], channels[3], channels[4]), reg_max)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone前向传播
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 9]:  # 保存特征层
                features.append(x)
        
        # Neck前向传播
        # P5 (最深层特征)
        p5 = features[2]
        
        # P4 (上采样并融合)
        p4_up = self.neck[0](p5)  # Upsample
        p4 = torch.cat([p4_up, features[1]], 1)
        p4 = self.neck[1](p4)  # C2f
        
        # P3 (继续上采样并融合)
        p3_up = self.neck[2](p4)  # Upsample
        p3 = torch.cat([p3_up, features[0]], 1)
        p3 = self.neck[3](p3)  # C2f
        
        # P4 (下采样路径)
        p4_down = self.neck[4](p3)  # Downsample
        p4 = torch.cat([p4_down, p4], 1)
        p4 = self.neck[5](p4)  # C2f
        
        # P5 (继续下采样)
        p5_down = self.neck[6](p4)  # Downsample
        p5 = torch.cat([p5_down, p5], 1)
        p5 = self.neck[7](p5)  # C2f
        
        # 检测头
        return self.detect([p3, p4, p5])


def create_yolo1d_model(model_size='n', num_classes=80, input_channels=1, input_length=1024, reg_max=16):
    """
    创建YOLO1D模型

    Args:
        model_size (str): 模型尺寸 ('n', 's', 'm', 'l', 'x')
        num_classes (int): 类别数
        input_channels (int): 输入通道数
        input_length (int): 输入序列长度
        reg_max (int): DFL回归最大值

    Returns:
        YOLO1D: 创建的模型
    """
    model_configs = {
        'n': {'depth': 0.33, 'width': 0.25},  # nano
        's': {'depth': 0.33, 'width': 0.50},  # small
        'm': {'depth': 0.67, 'width': 0.75},  # medium
        'l': {'depth': 1.00, 'width': 1.00},  # large
        'x': {'depth': 1.33, 'width': 1.25},  # xlarge
    }
    
    config = model_configs.get(model_size, model_configs['n'])
    
    model = YOLO1D(
        input_channels=input_channels,
        num_classes=num_classes,
        depth_multiple=config['depth'],
        width_multiple=config['width'],
        input_length=input_length,
        reg_max=reg_max
    )
    
    # 修复模型步长
    # dummy_input = torch.randn(1, input_channels, input_length)
    # strides = model.get_strides(dummy_input)
    # model.set_strides(strides)

    return model


if __name__ == "__main__":
    # 测试模型
    model = create_yolo1d_model('n', num_classes=10, input_channels=1, input_length=1024)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(2, 1, 1024)  # batch_size=2, channels=1, length=1024
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {[o.shape for o in output]}") 