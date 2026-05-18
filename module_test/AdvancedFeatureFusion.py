import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块
    
    通过让两个特征互相关注，捕获它们之间的相互关系，实现更有效的特征融合
    """
    def __init__(self, in_channels):
        super(CrossAttentionFusion, self).__init__()
        self.query_conv_1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv_1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.query_conv_2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv_2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
        # 最终融合层
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
    def forward(self, x1, x2):
        # x1关注x2
        batch_size, C, height, width = x1.size()
        
        # 特征1关注特征2
        proj_query1 = self.query_conv_1(x1).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key2 = self.key_conv_1(x2).view(batch_size, -1, height * width)
        energy1 = torch.bmm(proj_query1, proj_key2)
        attention1 = self.softmax(energy1)
        proj_value2 = self.value_conv_1(x2).view(batch_size, -1, height * width)
        out1 = torch.bmm(proj_value2, attention1.permute(0, 2, 1))
        out1 = out1.view(batch_size, C, height, width)
        out1 = self.gamma1 * out1 + x1
        
        # 特征2关注特征1
        proj_query2 = self.query_conv_2(x2).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key1 = self.key_conv_2(x1).view(batch_size, -1, height * width)
        energy2 = torch.bmm(proj_query2, proj_key1)
        attention2 = self.softmax(energy2)
        proj_value1 = self.value_conv_2(x1).view(batch_size, -1, height * width)
        out2 = torch.bmm(proj_value1, attention2.permute(0, 2, 1))
        out2 = out2.view(batch_size, C, height, width)
        out2 = self.gamma2 * out2 + x2
        
        # 融合两个增强的特征
        fusion = torch.cat([out1, out2], dim=1)
        output = self.fusion_conv(fusion)
        
        return output

class GatedFusion(nn.Module):
    """门控融合模块
    
    使用门控机制动态控制两个特征的融合比例，类似于LSTM/GRU的门控机制
    """
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
    def forward(self, x1, x2):
        # 计算门控权重
        combined = torch.cat([x1, x2], dim=1)
        gate = self.gate_conv(combined)
        
        # 应用门控
        gated_x1 = x1 * gate
        gated_x2 = x2 * (1 - gate)
        
        # 融合
        fusion = torch.cat([gated_x1, gated_x2], dim=1)
        output = self.fusion_conv(fusion)
        
        return output

class BilinearFusion(nn.Module):
    """双线性融合模块
    
    通过双线性池化捕获两个特征之间的二阶交互，比简单相加能捕获更复杂的特征关系
    """
    def __init__(self, in_channels, reduction=8):
        super(BilinearFusion, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.reduced_channels = self.in_channels // self.reduction
        
        # 降维
        self.conv1_1 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        
        # 融合后的处理
        self.conv_fusion = nn.Conv2d(self.reduced_channels * self.reduced_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        h1, w1 = x1.size(2), x1.size(3)
        h2, w2 = x2.size(2), x2.size(3)
        # 降维并重塑特征
        x1_reduced = self.conv1_1(x1).view(batch_size, self.reduced_channels, -1)
        x2_reduced = self.conv1_2(x2).view(batch_size, self.reduced_channels, -1)
        
        # 计算双线性交互
        bilinear = torch.bmm(x1_reduced, x2_reduced.permute(0, 2, 1))
        
        # 重塑为期望的输出形状
        bilinear = bilinear.view(batch_size, self.reduced_channels * self.reduced_channels, h1, w1)
        
        # 恢复通道数
        output = self.conv_fusion(bilinear)
        output = self.relu(self.bn(output))
        
        # 添加残差连接
        output = output + x1 + x2
        
        return output

class DynamicWeightedFusion(nn.Module):
    """动态加权融合模块
    
    根据输入特征的重要性动态分配融合权重，比固定权重更灵活
    """
    def __init__(self, in_channels):
        super(DynamicWeightedFusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        # 提取全局特征
        x1_pool = self.avg_pool(x1).view(batch_size, -1)
        x2_pool = self.avg_pool(x2).view(batch_size, -1)
        
        # 计算动态权重
        weights = self.fc(torch.cat([x1_pool, x2_pool], dim=1))
        
        # 应用权重
        weight1 = weights[:, 0].view(batch_size, 1, 1, 1)
        weight2 = weights[:, 1].view(batch_size, 1, 1, 1)
        
        output = weight1 * x1 + weight2 * x2
        
        return output

class AdvancedFeatureFusion(nn.Module):
    """高级特征融合模块
    
    集成多种融合策略，可以根据需要选择不同的融合方法
    """
    def __init__(self, in_channels, fusion_type='cross_attention'):
        super(AdvancedFeatureFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'cross_attention':
            self.fusion_module = CrossAttentionFusion(in_channels)
        elif fusion_type == 'gated':
            self.fusion_module = GatedFusion(in_channels)
        elif fusion_type == 'bilinear':
            self.fusion_module = BilinearFusion(in_channels)
        elif fusion_type == 'dynamic_weighted':
            self.fusion_module = DynamicWeightedFusion(in_channels)
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
    
    def forward(self, x1, x2):
        return self.fusion_module(x1, x2)