import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d


# 新增模块定义 --------------------------------------------------
class MultiScaleTemporalConv(nn.Module):
    """多尺度时序特征提取"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, (1, 16), padding=(0, 8), bias=False)
        self.branch2 = nn.Conv2d(in_channels, out_channels // 4, (1, 32), padding=(0, 16), bias=False)
        self.branch3 = nn.Conv2d(in_channels, out_channels // 4, (1, 64), padding=(0, 32), bias=False)
        self.branch4 = nn.Conv2d(in_channels, out_channels // 4, (1, 8), padding=(0, 4), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Mish()  # 更平滑的激活函数

    def forward(self, x):
        b1 = F.elu(self.branch1(x))
        b2 = F.elu(self.branch2(x))
        b3 = F.elu(self.branch3(x))
        b4 = F.elu(self.branch4(x))
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.activation(self.bn(out))


class SpatialEnhancementModule(nn.Module):
    """空间注意力增强"""

    def __init__(self, in_channels):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),  # 通道压缩
            LayerNorm2d(1),  # 电极敏感归一化
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_channels, in_channels, (3, 3), padding=1, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # 空间注意力加权
        att = self.spatial_att(x.mean(dim=1, keepdim=True))
        # 局部特征增强
        residual = x
        x = self.bn(self.conv(x))
        return x * att + residual  # 残差连接


class ProjectionHead(nn.Module):
    """自监督对比学习投影头"""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1)
        )

    def forward(self, x):
        return F.normalize(self.head(x), dim=1)  # 归一化用于对比损失