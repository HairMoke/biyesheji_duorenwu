'''
Author: Tammie li
Description: Define model
FilePath: \model.py

MTCN模型添加了图卷积的模型代码 0.93
图卷积 + 共享特征提取器使用时间差分卷积  + 修改为使用P300_TDC + 主任务特征提取器也是用时间差分卷积 可以跑到0.9317
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from module_test.Conv_Blocks import EEGInception
from module_test.ChannelAttention import CBAM
from module_test.ECAAttention import ECAAttention
from scipy.spatial.distance import cdist


#######################################
# 测试一下新加的DCT

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------- 1. 适配脑电的TDN移位模块 --------------------------
# 这个模块不进行乘法运算（零参数量），而是通过“搬运”数据，让当前时刻能看到“过去”和“未来”的信息。通常用于替代计算量大的 3D 卷积。
class EEGShiftModule(nn.Module):
    def __init__(self, in_channels, n_div=8, mode='shift'):
        super().__init__()
        self.in_channels = in_channels  # 输入通道数8通道
        self.fold_div = n_div # 8，表示要把通道分成8份
        self.fold = self.in_channels // self.fold_div # 每份有多少个通道，这里是 8//8=1

        # -----------------------------------------------------------
        # 难点解读 1: 为什么用 Conv1d 做移位？
        # 通常 PyTorch 用 torch.roll 来移位，但这里作者为了工程部署或特定实现，
        # 强行用一个固定权重的 Conv1d 来实现“搬运”像素的功能。
        # group=in_channels 意味着这是一个 Depthwise 卷积，每个通道独立处理。
        # -----------------------------------------------------------
        # 适配脑电的2D维度（导联C=64，时间步T），仅对时间步维度做移位
        self.conv_shift = nn.Conv1d(
            in_channels=self.fold_div * self.fold,
            out_channels=self.fold_div * self.fold,
            kernel_size=3,
            padding=1,
            groups=self.fold_div * self.fold,   # 当 groups == in_channels 时，这就是一个标准的 深度可分离卷积（Depthwise Convolution）。第 i个通道的数据，只跟第i个卷积核运算。
            bias=False
        )
        # 初始化移位核（TDN核心：左移/右移/固定）
        # -----------------------------------------------------------
        # 难点解读 2: 初始化“硬编码”权重
        # 这里的权重不是训练出来的，是人为写死的 0 和 1。
        # 卷积操作本来是：y = w1*x1 + w2*x2 + w3*x3
        # 如果设 w=[0,0,1]，那 y = x3 (实现了左移)
        # 如果设 w=[1,0,0]，那 y = x1 (实现了右移)
        # [[[0, 0, 1]]
        #  [[1, 0, 0]]
        #  [[0, 1, 0]]
        #  [[0, 1, 0]]
        #  [[0, 1, 0]]
        #  [[0, 1, 0]]
        #  [[0, 1, 0]]
        #  [[0, 1, 0]]]
        # -----------------------------------------------------------
        if mode == 'shift':
            self.conv_shift.weight.data.zero_() # 先全设为0
            self.conv_shift.weight.data[:self.fold, 0, 2] = 1  # 时间步左移 前1个通道：把卷积核第2位设为1 -> 捕捉 t+1 (未来信息，相当于左移)
            self.conv_shift.weight.data[self.fold:2 * self.fold, 0, 0] = 1  # 时间步右移 中间1个通道：把卷积核第0位设为1 -> 捕捉 t-1 (过去信息，相当于右移)
            if 2 * self.fold < self.in_channels: # 剩余通道：把卷积核第1位设为1 -> 保持 t (当前信息，不动)
                self.conv_shift.weight.data[2 * self.fold:, 0, 1] = 1  # 固定通道
        elif mode == 'fixed':
            self.conv_shift.weight.data.zero_()
            self.conv_shift.weight.data[:, 0, 1] = 1  # 无移位（基准）
        # 在初始化完权重后，加上这一句
        # self.conv_shift.weight.requires_grad = False # 这样，无论训练多久，它永远保持 0 和 1，只做搬运工，不参与学习
        # 如果权重可更新，反向传播后（Epoch 1+）：优化器（Optimizer）会发现这些权重对 Loss 有贡献，计算出梯度，并修改这些权重。
        # 最终状态：原本的 0 可能会变成 0.001，1 可能会变成 0.99。它会从一个“硬性的移位操作”慢慢变成一个“可学习的时序混合卷积”。

    def forward(self, x):
        # x: [B, ch, C, T] → 仅对时间步T做移位（维度严格不变）
        B, ch, C, T = x.shape #[32,8,64,286]

        # -----------------------------------------------------------
        # 难点解读 3: 维度重塑 (Reshape)
        # Conv1d 只能处理 3D 数据 [N, C, L]。
        # 我们的脑电是 4D 的，为了对 Time 维度做卷积，
        # 必须把 Batch 和 Electrodes 合并，看作很多个独立的“样本”。
        # permute(0, 2, 1, 3) -> [B, C, ch, T]
        # reshape -> [B*C, ch, T]  (相当于把64个电极拆成单独的行来处理)
        # -----------------------------------------------------------
        # 重塑为Conv1d输入格式：[B*C, ch, T]
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * C, ch, T)
        x_shifted = self.conv_shift(x_reshaped) # 执行移位操作
        # 还原维度：[B, ch, C, T]（确保无维度变化）
        x_out = x_shifted.reshape(B, C, ch, T).permute(0, 2, 1, 3)
        return x_out


# -------------------------- 2. TDN时序差分模块（适配脑电，维度完全对齐） --------------------------
# P300 信号是一个波形的突变。这个模块通过计算 $Time_{t+1} - Time_{t}$ 来提取变化率（即导数），并用多尺度（看局部 vs 看全局）来增强这种变化特征。
class EEGTemporalDifferenceModule(nn.Module):
    def __init__(self, in_channels, target_C=64, target_T=287, reduction=4):
        super().__init__()
        self.in_channels = in_channels  # 8通道
        self.reduction = reduction
        self.mid_channels = self.in_channels // self.reduction  # 8//4=2（有效通道）
        self.target_C = target_C  # 固定导联数
        self.target_T = target_T  # 固定目标时间步

        # 1. 特征降维（TDN mSEModule的conv1/bn1）
        self.conv_reduce = nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(self.mid_channels)

        # 2. 多尺度池化（TDN的avg_pool_forward2/4，仅时间步下采样）
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.avg_pool4 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # 3. 多尺度卷积（TDN的conv3_smallscale2/4）
        self.conv_scale2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=(1, 3), padding=(0, 1),
                                     bias=False)
        self.bn_scale2 = nn.BatchNorm2d(self.mid_channels)
        self.conv_scale4 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=(1, 3), padding=(0, 1),
                                     bias=False)
        self.bn_scale4 = nn.BatchNorm2d(self.mid_channels)

        # 4. 特征升维+注意力激活（TDN的conv3/bn3/sigmoid）
        self.conv_restore = nn.Conv2d(self.mid_channels, self.in_channels, kernel_size=1, bias=False)
        self.bn_restore = nn.BatchNorm2d(self.in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 强制对齐输入维度到目标维度（防止前端卷积维度偏差）
        # -----------------------------------------------------------
        # 步骤 0: 强制对齐
        # 为了防止上一层卷积改变了时间步长，这里用插值强制把尺寸
        # 变回 [64, 287]。RSVP任务对时间对齐要求很高。
        # -----------------------------------------------------------
        if x.shape[2:] != (self.target_C, self.target_T):
            x = F.interpolate(x, size=(self.target_C, self.target_T), mode='bilinear', align_corners=False)

        B, ch, C, T = x.shape  # 此时T=287，C=64

        # Step1: 计算前向/后向时序差分（TDN核心，补零后保持287）
        # -----------------------------------------------------------
        # 步骤 1: 核心！计算时序差分 (Gradient)
        # 这里的数学含义是离散差分。
        # x_forward:  x[t+1] - x[t] (向后差分)
        # x_backward: x[t] - x[t-1] (向前差分)
        # -----------------------------------------------------------
        x_forward = x[:, :, :, 1:] - x[:, :, :, :-1]  # [B,8,64,286]
        x_backward = x[:, :, :, :-1] - x[:, :, :, 1:]  # [B,8,64,286]
        # 补零还原维度，补 0 是为了让时间长度变回 287，否则做减法后长度会变成 286
        x_forward = F.pad(x_forward, (0, 1, 0, 0), mode='constant', value=0)  # [B,8,64,287]
        x_backward = F.pad(x_backward, (1, 0, 0, 0), mode='constant', value=0)  # [B,8,64,287]
        x_diff = 0.5 * x_forward + 0.5 * x_backward # 融合前后差分，得到平滑后的变化特征

        # Step2: 差分特征降维（维度不变，[B,2,64,287]）(1x1 Conv)，输入是 8 通道，这里通过 1x1 卷积降到了 2 通道（reduction=4）。
        # 作用： 提取“精华”。差分信号里有很多噪声，我们不需要保留那么多通道。通过学习，模型会自动把最有用的“变化模式”压缩到这 2 个通道里。这也为后面多尺度处理减少了计算量。
        x_diff_reduce = self.bn_reduce(self.conv_reduce(x_diff))

        # Step3: 多尺度特征提取（所有特征最终对齐到287，确保可相加）:这是为了解决 P300 “千人千面” 的问题。
        # -----------------------------------------------------------
        # 步骤 3: 多尺度特征提取 (Multi-scale)
        # 为什么要做这个？
        # feat1: 原始的细微变化 (高频)
        # feat2: avg_pool2 后，看的是每2个时间点的平均变化
        # feat3: avg_pool4 后，看的是每4个时间点的平均变化 (低频趋势)
        # 插值(interpolate)是为了把缩小后的特征再拉回 287 长度，以便相加。
        # -----------------------------------------------------------
        # 尺度1：原始差分（直接使用，后续无维度变化） (高频细节) 直接看原始差分。它对尖锐的、持续时间短的波峰最敏感。
        feat1 = x_diff_reduce  # [B,2,64,287]
        # 尺度2：2倍下采样+卷积+上采样还原（严格对齐目标维度）(中频特征) 先做 AvgPool2（2倍平均池化）。这意味着它把每 2 个时间点看作一个整体。它对稍宽一点的波峰敏感。
        feat2 = self.bn_scale2(self.conv_scale2(self.avg_pool2(x_diff_reduce)))
        feat2 = F.interpolate(feat2, size=(self.target_C, self.target_T), mode='bilinear',
                              align_corners=False)  # [B,2,64,287]
        # 尺度3：4倍下采样+卷积+上采样还原（严格对齐目标维度）(低频轮廓) 先做 AvgPool4。它忽略细节，只看大趋势。它能捕捉到持续时间很长、很平缓的 P300 凸起。
        feat3 = self.bn_scale4(self.conv_scale4(self.avg_pool4(x_diff_reduce)))
        feat3 = F.interpolate(feat3, size=(self.target_C, self.target_T), mode='bilinear',
                              align_corners=False)  # [B,2,64,287]

        # Step4: 多尺度融合+升维+注意力加权（维度保持287）融合生成注意力权重
        feat_fusion = (feat1 + feat2 + feat3) / 3  # 此时三者维度完全一致，可正常相加
        feat_restore = self.bn_restore(self.conv_restore(feat_fusion))  # [B,8,64,287] 升维还原
        attn = self.sigmoid(feat_restore) - 0.5 # Sigmoid 归一化到 (0,1)，减 0.5 是为了让权重有正有负（抑制或增强）

        # Step5: 差分特征加权融合（维度不变）
        # -----------------------------------------------------------
        # 步骤 5: 残差增强 (Residual Attention)
        # 原始特征 x 加上 (原始特征 x * 差分注意力 attn)
        # 含义：如果某个时刻信号变化很剧烈（差分大），就放大该时刻的特征值。
        # -----------------------------------------------------------
        x_out = x + x * attn
        return x_out


# -------------------------- 3. 主模块：融合TDN创新的脑电时序卷积（核心修正padding） --------------------------
class TDNEnhancedEEGConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, kernel_size=32, target_time_steps=None):
        super().__init__()
        self.target_C = 64  # 固定导联数
        self.target_T = target_time_steps if target_time_steps else 287  # 可配置目标时间步
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # 计算padding以保持时间维度
        if target_time_steps:
            # 根据目标时间步计算padding
            self.padding = (0, (target_time_steps - 256 + kernel_size - 1) // 2)
        else:
            # 使用固定padding，输出287时间步
            self.padding = (0, 31)

        # 基础时序卷积，适配temporal_stream的输入输出要求
        self.base_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=self.padding,
            bias=False
        )

        # 3. TDN创新模块1：时序移位增强（维度不变）
        self.shift_module = EEGShiftModule(in_channels=8, n_div=8, mode='shift')

        # 4. TDN创新模块2：时序差分+多尺度融合（传入固定目标维度）
        self.td_module = EEGTemporalDifferenceModule(
            in_channels=8,
            target_C=self.target_C,
            target_T=self.target_T,
            reduction=4
        )

        # 5. 特征融合层（保持通道和维度不变）
        self.fusion_conv = nn.Conv2d(8, 8, kernel_size=(1, 1), bias=False)
        self.bn_fusion = nn.BatchNorm2d(8)

    def forward(self, x):
        # 输入：[B, C, H, T]，适配temporal_stream的输入
        B, C, H, T = x.shape

        # Step1: 基础时序卷积 → [B,out_channels,H,T']（直接使用输入，不再内部padding）
        x_conv = self.base_conv(x)

        # Step3: TDN移位增强 → [B,8,64,287]（维度不变）
        x_shift = self.shift_module(x_conv)

        # Step4: TDN时序差分+多尺度融合 → [B,8,64,287]（维度完全对齐）
        x_td = self.td_module(x_shift)

        # Step5: 特征融合+BN → 最终输出[B,8,64,287]（严格匹配期望）
        x_out = self.bn_fusion(self.fusion_conv(x_td))

        return x_out


# -------------------------- 测试验证 --------------------------
# if __name__ == "__main__":
#     # 初始化模型
#     model = TDNEnhancedEEGConv()
#     # 模拟脑电输入：B=32, 1通道, 64导联, 256时间步
#     input_eeg = torch.randn(32, 1, 64, 256)
#     # 前向传播
#     output = model(input_eeg)
#     # 验证维度
#     print(f"输入形状: {input_eeg.shape}")  # torch.Size([32, 1, 64, 256])
#     print(f"输出形状: {output.shape}")  # torch.Size([32, 8, 64, 287])（完全匹配期望）





##########################################

class DualStreamSpatioTemporalExtractor(nn.Module):
    """双流时空协同特征提取器：时间流和空间流的协同学习"""
    def __init__(self, in_channels=1, out_channels=16, num_electrodes=64, kernel_size=32, dropout=0.5):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.out_channels = out_channels

        # ============ 时间流（Temporal Stream）============
        self.temporal_stream = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            # P300_TDC(in_channels, out_channels//2, kernel_size),
            TDNEnhancedEEGConv(in_channels=in_channels, out_channels=out_channels//2, kernel_size=kernel_size, target_time_steps=256),
            nn.BatchNorm2d(out_channels//2),
            nn.ELU()
        )

        # 修复：确保多尺度特征融合后通道数正确
        # self.temporal_multiscale = nn.ModuleList([
        #     nn.Conv2d(out_channels//2, out_channels//8, (1, 3), padding=(0, 1)),  # 2通道
        #     nn.Conv2d(out_channels//2, out_channels//8, (1, 5), padding=(0, 2)),  # 2通道
        #     nn.Conv2d(out_channels//2, out_channels//8, (1, 7), padding=(0, 3)),  # 2通道
        #     nn.Conv2d(out_channels//2, out_channels//8, (1, 1))                   # 2通道
        # ])  # 总共8通道 = out_channels//2 = 8
        # ============ 改进后的多尺度模块 ============
        # 输入通道数：out_channels // 2 (例如 8)
        # 我们希望总输出通道数仍为 8，所以分给 4 个分支，每个分支 2 个通道
        branch_channels = (out_channels // 2) // 4
        self.temporal_multiscale = nn.ModuleList([
            # 分支 1: 1x3 卷积 (捕捉尖峰细节)
            nn.Sequential(
                nn.Conv2d(out_channels // 2, branch_channels, (1, 3), padding=(0, 1)),
                nn.BatchNorm2d(branch_channels),
                nn.ELU()
            ),
            # 分支 2: 1x5 卷积 (捕捉中等波形)
            nn.Sequential(
                nn.Conv2d(out_channels // 2, branch_channels, (1, 5), padding=(0, 2)),
                nn.BatchNorm2d(branch_channels),
                nn.ELU()
            ),
            # 分支 3: 1x7 卷积 (捕捉宽缓波形)
            nn.Sequential(
                nn.Conv2d(out_channels // 2, branch_channels, (1, 7), padding=(0, 3)),
                nn.BatchNorm2d(branch_channels),
                nn.ELU()
            ),
            # ===【新增改进点】分支 4: Max Pooling 分支 ===
            # 作用：捕捉最显著的特征，并提供平移不变性
            nn.Sequential(
                nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1)),
                # 池化后接一个 1x1 卷积，用于调整通道数和特征变换
                nn.Conv2d(out_channels // 2, branch_channels, 1),
                nn.BatchNorm2d(branch_channels),
                nn.ELU()
            )
        ])

        # ============ 空间流（Spatial Stream）============
        self.spatial_stream_conv = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(in_channels, out_channels//2, (1, kernel_size), bias=False),
            nn.BatchNorm2d(out_channels//2)
        )

        # 动态图卷积模块
        # adj = build_eeg_graph(num_electrodes, adj_type='conduction')
        adj = build_eeg_graph(num_electrodes, adj_type='knn')
        self.register_buffer('base_adj', adj)
        self.graph_conv = GraphConv(out_channels//2, out_channels//2, dropout)

        # 空间流的注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_channels//2, out_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels//8, 1, 1),
            nn.Sigmoid()
        )

        # ============ 双流交互模块（Cross-Stream Interaction）============
        self.cross_stream_fusion = CrossStreamFusion(out_channels//2)

        # ============ 特征重构与输出============
        self.feature_reconstructor = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (num_electrodes, 1), groups=out_channels//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # ============ 时间流处理 ============
        temporal_feat = self.temporal_stream(x)  # [B, 8, 64, T]

        # 多尺度时间特征融合 - 修复通道数
        multiscale_feats = []
        for conv in self.temporal_multiscale:
            multiscale_feats.append(conv(temporal_feat))
        temporal_enhanced = torch.cat(multiscale_feats, dim=1)  # [B, C/2, 64, T] = [B, 8, 64, T]

        # ============ 空间流处理 ============
        spatial_feat = self.spatial_stream_conv(x)  # [B, C/2, 64, T] = [B, 8, 64, T]

        # 图卷积处理：[B, C/2, 64, T] -> [B, 64, C/2] -> [B, C/2, 64, T]
        B, C, H, W = spatial_feat.shape
        node_feat = spatial_feat.mean(dim=3).transpose(1, 2)  # [B, 64, C/2]
        graph_feat = self.graph_conv(node_feat, self.base_adj)  # [B, 64, C/2]
        graph_feat = graph_feat.transpose(1, 2).unsqueeze(3).expand(-1, -1, -1, W)  # [B, C/2, 64, T]

        # 空间注意力增强
        spatial_att = self.spatial_attention(spatial_feat)
        spatial_enhanced = spatial_feat * spatial_att + graph_feat * (1 - spatial_att)

        # ============ 双流交互融合 ============
        # 现在两个特征都是 [B, C/2, 64, T] = [B, 8, 64, T]
        fused_features = self.cross_stream_fusion(temporal_enhanced, spatial_enhanced) # [8,16,64,256]

        # ============ 特征重构 ============
        output = self.feature_reconstructor(fused_features) #[8,16,1,64]

        return output



# class CrossStreamFusion(nn.Module):
#     """双流交互融合模块：实现时间流和空间流的深度交互"""
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels

#         # 交叉注意力机制
#         self.temp_to_spat_attention = nn.MultiheadAttention(
#             embed_dim=channels, num_heads=4, batch_first=True
#         )
#         self.spat_to_temp_attention = nn.MultiheadAttention(
#             embed_dim=channels, num_heads=4, batch_first=True
#         )

#         # 特征门控机制
#         self.gate = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 1),
#             nn.Sigmoid()
#         )

#         # 残差连接
#         self.residual_proj = nn.Conv2d(channels * 2, channels * 2, 1)

#     def forward(self, temporal_feat, spatial_feat):
#         B, C, H, W = temporal_feat.shape

#         # 展平为序列格式进行注意力计算
#         temp_seq = temporal_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
#         spat_seq = spatial_feat.flatten(2).transpose(1, 2)   # [B, H*W, C]

#         # 交叉注意力：时间特征查询空间特征
#         temp_enhanced, _ = self.temp_to_spat_attention(temp_seq, spat_seq, spat_seq)
#         # 交叉注意力：空间特征查询时间特征
#         spat_enhanced, _ = self.spat_to_temp_attention(spat_seq, temp_seq, temp_seq)

#         # 重构为卷积格式
#         temp_enhanced = temp_enhanced.transpose(1, 2).view(B, C, H, W)
#         spat_enhanced = spat_enhanced.transpose(1, 2).view(B, C, H, W)

#         # 门控融合
#         concat_feat = torch.cat([temp_enhanced, spat_enhanced], dim=1)
#         gate_weight = self.gate(concat_feat)

#         # 加权融合 + 残差连接
#         fused = gate_weight * temp_enhanced + (1 - gate_weight) * spat_enhanced
#         final_feat = torch.cat([fused, concat_feat], dim=1)  # [B, 2C, H, W]

#         return self.residual_proj(final_feat)

class CrossStreamFusion(nn.Module):
    """内存优化的双流交互融合模块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 使用更轻量的交互方式替代多头注意力
        self.temp_proj = nn.Conv2d(channels, channels//2, 1)
        self.spat_proj = nn.Conv2d(channels, channels//2, 1)

        # 轻量级交互模块
        self.interaction = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels//2),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )

        # 特征门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

        # 修复：残差连接的输入输出通道数
        self.residual_proj = nn.Conv2d(channels * 3, channels * 2, 1)  # 24 -> 16

    def forward(self, temporal_feat, spatial_feat):
        B, C, H, W = temporal_feat.shape  # C = 8

        # 轻量级特征投影
        temp_proj = self.temp_proj(temporal_feat)  # [B, 4, H, W]
        spat_proj = self.spat_proj(spatial_feat)   # [B, 4, H, W]

        # 交互特征融合
        cross_feat = torch.cat([temp_proj, spat_proj], dim=1)  # [B, 8, H, W]
        interaction_feat = self.interaction(cross_feat)  # [B, 8, H, W]

        # 增强原始特征
        temp_enhanced = temporal_feat + interaction_feat * 0.1  # [B, 8, H, W]
        spat_enhanced = spatial_feat + interaction_feat * 0.1   # [B, 8, H, W]

        # 门控融合
        concat_feat = torch.cat([temp_enhanced, spat_enhanced], dim=1)  # [B, 16, H, W]
        gate_weight = self.gate(concat_feat)  # [B, 8, H, W]

        # 加权融合 + 残差连接
        fused = gate_weight * temp_enhanced + (1 - gate_weight) * spat_enhanced  # [B, 8, H, W]
        final_feat = torch.cat([fused, concat_feat], dim=1)  # [B, 24, H, W] = 8 + 16

        return self.residual_proj(final_feat)  # [B, 16, H, W]



# ---------------------- 2. 跨任务对比损失 ----------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, task_features):
        if len(task_features) < 2:
            # 处理列表和字典两种情况
            if isinstance(task_features, dict):
                return torch.tensor(0.0, device=next(iter(task_features.values())).device)
            else:
                return torch.tensor(0.0, device=task_features[0].device)

        # 处理列表和字典两种输入格式
        if isinstance(task_features, dict):
            norm_feats = [F.normalize(feat, p=2, dim=1) for feat in task_features.values()]
        else:
            norm_feats = [F.normalize(feat, p=2, dim=1) for feat in task_features]

        # 找到最小批次大小，统一所有特征的批次大小
        min_batch_size = min(feat.shape[0] for feat in norm_feats)
        norm_feats = [feat[:min_batch_size] for feat in norm_feats]

        num_tasks = len(norm_feats)
        total_loss = 0.0

        for i in range(num_tasks):
            for j in range(i+1, num_tasks):
                sim_matrix = torch.matmul(norm_feats[i], norm_feats[j].T) / self.temperature
                positive_mask = torch.eye(min_batch_size, device=sim_matrix.device)
                negative_mask = 1 - positive_mask

                exp_sim = torch.exp(sim_matrix)
                pos_sim = (exp_sim * positive_mask).sum(dim=1)
                neg_sim = (exp_sim * negative_mask).sum(dim=1)
                task_pair_loss = -torch.log(pos_sim / (neg_sim + 1e-8)).mean()
                total_loss += task_pair_loss

        return total_loss / (num_tasks * (num_tasks - 1) / 2)



class Multi_Stage_P300_TDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=32):
        super().__init__()
        self.original_conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=False)

        # 确保三个分支通道数加起来等于out_channels
        branch_channels = out_channels // 3
        remaining_channels = out_channels - 2 * branch_channels

        self.early_conv = nn.Conv2d(in_channels, branch_channels, (1, kernel_size), bias=False)
        self.peak_conv = nn.Conv2d(in_channels, branch_channels, (1, kernel_size), bias=False)
        self.late_conv = nn.Conv2d(in_channels, remaining_channels, (1, kernel_size), bias=False)  # 处理余数

        self.stage_weights = nn.Parameter(torch.tensor([0.05, 0.15, 0.05]))

    def get_stage_weights(self, stage):
        weight = getattr(self, f'{stage}_conv').weight
        kT = weight.shape[3]
        grad_weight = torch.zeros_like(weight) # 全零初始化，与weight完全相同形状

        if stage == 'early':
            grad_weight[:, :, :, 1:] = weight[:, :, :, 1:] - weight[:, :, :, :-1] # 相邻两个 后一个-前一个
            grad_weight[:, :, :, 0] = weight[:, :, :, 0]
        elif stage == 'peak':
            mid = kT // 2
            grad_weight[:, :, :, :mid] = -weight[:, :, :, :mid] # 前半部分取反
            grad_weight[:, :, :, mid:] = weight[:, :, :, mid:] # 后半部分保持不变
        elif stage == 'late':
            grad_weight[:, :, :, :-1] = weight[:, :, :, :-1] - weight[:, :, :, 1:] # 相邻两个 前一个-后一个
            grad_weight[:, :, :, -1] = -weight[:, :, :, -1]

        return grad_weight

    def forward(self, x):
        # [16, 1, 64, 256]
        original_feat = self.original_conv(x) # [16, 8, 64, 225]

        early_weight = self.get_stage_weights('early') # [8, 1, 1, 32]
        peak_weight = self.get_stage_weights('peak') # [8, 1, 1, 32]
        late_weight = self.get_stage_weights('late') # [8, 1, 1, 32]

        early_feat = F.conv2d(x, early_weight, padding=self.original_conv.padding) # [16, 8, 64, 225]
        peak_feat = F.conv2d(x, peak_weight, padding=self.original_conv.padding) # [16, 8, 64, 225]
        late_feat = F.conv2d(x, late_weight, padding=self.original_conv.padding) # [16, 8, 64, 225]

        p300_feat = torch.cat([early_feat, peak_feat, late_feat], dim=1)  # [16, 24, 64, 256]

        alpha = torch.sigmoid(self.stage_weights).sum() * 0.2 # 计算自适应权重
        return original_feat + alpha * p300_feat  # 8通道 + 8通道 = 正确！


class P300_TDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=32):
        super().__init__()
        # 原始卷积分支
        self.original_conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=False)

        # P300差分卷积分支
        self.p300_conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=False)

        # 自适应融合权重
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def get_p300_diff_weight(self):
        weight = self.p300_conv.weight  # [out_ch, in_ch, 1, kernel_size]
        kT = weight.shape[3]
        diff_weight = torch.zeros_like(weight)

        # 基于P300时间特性的差分权重设计
        # 将32个时间点映射到P300的不同阶段
        early_end = kT // 4      # 前25%: 基线期
        n200_end = kT // 2       # 25%-50%: N200期
        p300_end = 3 * kT // 4   # 50%-75%: P300期
        # 75%-100%: 晚期成分

        for i in range(kT):
            if i < early_end:
                # 基线期：轻微抑制
                diff_weight[:, :, :, i] = 0.3 * weight[:, :, :, i]
            elif i < n200_end:
                # N200期：负向差分
                if i > 0:
                    diff_weight[:, :, :, i] = weight[:, :, :, i] - weight[:, :, :, i-1]
                else:
                    diff_weight[:, :, :, i] = weight[:, :, :, i]
            elif i < p300_end:
                # P300期：正向跃迁差分（关键）
                if i >= 2:
                    diff_weight[:, :, :, i] = weight[:, :, :, i] - weight[:, :, :, i-2]
                else:
                    diff_weight[:, :, :, i] = weight[:, :, :, i]
            else:
                # 晚期：衰减差分
                if i > 0:
                    diff_weight[:, :, :, i] = 0.5 * (weight[:, :, :, i] - weight[:, :, :, i-1])
                else:
                    diff_weight[:, :, :, i] = 0.5 * weight[:, :, :, i]

        return diff_weight

    def forward(self, x):
        # 原始时间特征
        # original_feat = self.original_conv(x)
        original_feat = self.p300_conv(x)

        # P300差分特征
        diff_weight = self.get_p300_diff_weight()
        p300_feat = F.conv2d(x, diff_weight, padding=self.original_conv.padding)

        # 自适应融合
        alpha = torch.sigmoid(self.alpha) * 0.5  # 限制在[0, 0.5]
        return original_feat + alpha * p300_feat




class EEG_TDC_1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=32, step=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=False)
        self.step = step

    def get_temporal_gradient_weight(self):
        weight = self.conv.weight  # [out_ch, in_ch, 1, kernel_size]
        kT = weight.shape[3]  # 时间维度
        grad_weight = torch.zeros_like(weight)

        if kT >= 5 and self.step == 1:
            # EEG时间差分权重重构（适配32长度卷积核）
            grad_weight[:, :, :, -1] = weight[:, :, :, -1]
            for i in range(kT-2, -1, -1):
                if i == kT-2:
                    grad_weight[:, :, :, i] = weight[:, :, :, i] - weight[:, :, :, i+1]
                elif i == 0:
                    grad_weight[:, :, :, i] = -weight[:, :, :, i+1]
                else:
                    grad_weight[:, :, :, i] = weight[:, :, :, i] - weight[:, :, :, i+1]
        else:
            grad_weight = weight

        return grad_weight, self.conv.bias

    def forward(self, x):
        weight, bias = self.get_temporal_gradient_weight()
        return F.conv2d(x, weight, bias, padding=self.conv.padding)



# ---------------------- 1. 图卷积层（来自GPFMTL）----------------------
class GraphConv(nn.Module):
    """脑区图卷积层：建模电极（节点）间的空间/功能关联"""
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GraphConv, self).__init__()
        self.weight = nn.Linear(in_features, out_features)  # 特征投影
        self.bn = nn.BatchNorm1d(out_features)  # 批量归一化
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, x, adj):
        """
        x: 输入特征 [batch_size, num_electrodes, in_features]（电极数×特征数）
        adj: 邻接矩阵 [num_electrodes, num_electrodes]（脑区连接图）
        """
        # 特征线性变换：[batch, num_electrodes, in_feat] → [batch, num_electrodes, out_feat]
        x = self.weight(x)
        # 确保邻接矩阵与输入在同一设备
        adj = adj.to(x.device)
        # 图卷积操作：特征 × 邻接矩阵（聚合邻居节点信息）
        x = torch.matmul(adj, x)  # [batch, num_electrodes, out_feat]
        # 激活与归一化
        x = self.elu(self.bn(x.transpose(1, 2))).transpose(1, 2)  # BN需调整通道维度
        x = self.dropout(x)
        return x

# ---------------------- 2. 图构建函数（已移除对比损失）----------------------

def build_eeg_graph(num_electrodes=64, adj_type='distance'):
    """构建基于真实10-20系统的EEG电极邻接矩阵"""
    # 完整的64个电极坐标 (基于标准10-20系统扩展)
    electrode_coords = np.array([
        # 前额区 (8个)
        [-0.309, 0.951, 0.000], [0.309, 0.951, 0.000],   # Fp1, Fp2
        [-0.588, 0.809, 0.000], [0.588, 0.809, 0.000],   # F3, F4
        [-0.809, 0.588, 0.000], [0.809, 0.588, 0.000],   # F7, F8
        [0.000, 0.809, 0.588], [-0.454, 0.891, 0.000],   # Fz, AF3

        # 中央区 (12个)
        [-0.707, 0.000, 0.707], [0.707, 0.000, 0.707],   # C3, C4
        [-1.000, 0.000, 0.000], [1.000, 0.000, 0.000],   # T7, T8
        [0.000, 0.000, 1.000], [-0.454, 0.000, 0.891],   # Cz, C1
        [0.454, 0.000, 0.891], [-0.891, 0.000, 0.454],   # C2, C5
        [0.891, 0.000, 0.454], [-0.588, 0.000, 0.809],   # C6, FC3
        [0.588, 0.000, 0.809], [0.000, 0.309, 0.951],    # FC4, FCz

        # 顶区 (12个)
        [-0.588, -0.809, 0.000], [0.588, -0.809, 0.000], # P3, P4
        [-0.809, -0.588, 0.000], [0.809, -0.588, 0.000], # P7, P8
        [0.000, -1.000, 0.000], [-0.454, -0.891, 0.000], # Pz, P1
        [0.454, -0.891, 0.000], [-0.707, -0.707, 0.000], # P2, P5
        [0.707, -0.707, 0.000], [-0.309, -0.809, 0.588], # P6, CP3
        [0.309, -0.809, 0.588], [0.000, -0.809, 0.588],  # CP4, CPz

        # 枕区 (8个)
        [-0.309, -0.951, 0.000], [0.309, -0.951, 0.000], # O1, O2
        [0.000, -0.951, 0.309], [-0.588, -0.809, 0.000], # Oz, PO3
        [0.588, -0.809, 0.000], [-0.454, -0.891, 0.000], # PO4, PO7
        [0.454, -0.891, 0.000], [0.000, -0.891, 0.454],  # PO8, POz

        # 颞区 (12个)
        [-0.951, 0.309, 0.000], [0.951, 0.309, 0.000],   # FT7, FT8
        [-0.951, -0.309, 0.000], [0.951, -0.309, 0.000], # TP7, TP8
        [-0.809, 0.309, 0.500], [0.809, 0.309, 0.500],   # T3, T4
        [-0.809, -0.309, 0.500], [0.809, -0.309, 0.500], # T5, T6
        [-0.707, 0.500, 0.500], [0.707, 0.500, 0.500],   # FC5, FC6
        [-0.707, -0.500, 0.500], [0.707, -0.500, 0.500], # CP5, CP6

        # 额外电极 (12个)
        [-0.156, 0.988, 0.000], [0.156, 0.988, 0.000],   # AF7, AF8
        [-0.156, 0.500, 0.851], [0.156, 0.500, 0.851],   # F1, F2
        [-0.156, -0.500, 0.851], [0.156, -0.500, 0.851], # P1, P2
        [-0.156, -0.988, 0.000], [0.156, -0.988, 0.000], # PO1, PO2
        [0.000, 0.500, 0.866], [0.000, -0.500, 0.866],   # FC1, CP1
        [0.000, 0.707, 0.707], [0.000, -0.707, 0.707]    # FC2, CP2
    ])   # [64, 3]

    # 确保正好64个电极
    if len(electrode_coords) != num_electrodes:
        # 如果不足64个，用最后一个坐标填充
        if len(electrode_coords) < num_electrodes:
            last_coord = electrode_coords[-1:].repeat(num_electrodes - len(electrode_coords), axis=0)
            electrode_coords = np.vstack([electrode_coords, last_coord])
        else:
            electrode_coords = electrode_coords[:num_electrodes]

    # 计算欧几里得距离矩阵
    dist_matrix = cdist(electrode_coords, electrode_coords, metric='euclidean') # [64, 64]
    # - 对称距离矩阵 ： dist_matrix[i,j] 表示第i个电极与第j个电极之间的欧氏距离
    # - 对角线元素 ： dist_matrix[i,i] = 0 (电极到自身的距离为0)
    # - 对称性 ： dist_matrix[i,j] = dist_matrix[j,i]

    if adj_type == 'distance':
        # 使用高斯核：距离越近权重越大
        sigma = np.std(dist_matrix) * 0.5 # 基于距离标准差确定衰减系数
        adj = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2)) # 形状仍为 [64, 64]
    elif adj_type == 'knn':
        # K近邻连接：每个电极只连接最近的k个邻居
        k = 6
        adj = np.zeros_like(dist_matrix)  # 先创建零矩阵 [64, 64]
        for i in range(num_electrodes):  # 然后填充每个电极的6个最近邻连接
            nearest_k = np.argsort(dist_matrix[i])[1:k+1]
            adj[i, nearest_k] = 1
            adj[nearest_k, i] = 1
    elif adj_type == 'conduction':
        # 传导速度-有向图卷积（CV-GCN）
        v = 5.0  # 白质传导速度 5 m/s
        time = dist_matrix / v  # 传导时间
        sigma = np.std(dist_matrix) * 0.5
        # 结合物理距离和传导时间的权重
        adj = np.exp(-dist_matrix**2 / (2 * sigma**2)) * np.exp(-np.abs(time) / 0.025)
        # 对称化处理，保证双向信息传播
        adj = adj + adj.T

    # 添加自环并归一化
    adj = adj + np.eye(num_electrodes) # 加上一个单位阵 [64, 64]
    # - 信息保留 ：节点在特征聚合时保留自身信息
    # - 数值稳定性 ：防止除以零错误（后续归一化需要）
    # - 生物学合理性 ：神经元具有自反馈机制
    degree = np.diag(np.sum(adj, axis=1) ** -0.5)  # 计算度矩阵的逆平方根
    adj = degree @ adj @ degree # 对称归一化

    return torch.tensor(adj, dtype=torch.float32) # [64, 64]


class HilbertGroupConv(nn.Module):
    """
    Improved Hilbert-Group Spatial Convolution

    改进点：
    1. 正确的Hilbert变换在时间维度上进行
    2. 多尺度相位特征提取
    3. 增加非线性激活和归一化
    4. 简化的复数卷积实现
    5. 残差连接和注意力机制

    Args:
        c_in: 输入通道数
        c_out: 输出通道数
        k: 每组通道数，默认为8
    """
    def __init__(self, c_in, c_out, k=8):
        super(HilbertGroupConv, self).__init__()
        self.k = k
        self.g = c_in // k  # 组数
        self.c_in = c_in
        self.c_out = c_out

        # 确保输入通道数能被k整除
        assert c_in % k == 0, f"Input channels {c_in} must be divisible by group size {k}"

        # 复数卷积核权重 [c_out, c_in, 1, kernel_size]
        self.kernel_size = 3
        self.complex_weight = nn.Parameter(torch.randn(c_out, c_in, 1, self.kernel_size) / math.sqrt(c_in))

        # 相位特征提取网络
        self.phase_feature_extractor = nn.Sequential(
            nn.Conv1d(c_in, c_in // 2, 1),
            nn.BatchNorm1d(c_in // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_in // 2, c_in, 1),
            nn.BatchNorm1d(c_in),
            nn.Tanh()
        )

        # 注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(c_out, c_out // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out // 16, c_out, 1),
            nn.Sigmoid()
        )

        # 残差连接
        if c_in == c_out:
            self.residual_conv = None
        else:
            self.residual_conv = nn.Conv1d(c_in, c_out, 1)

        # 输出归一化
        self.output_norm = nn.BatchNorm1d(c_out)

    def hilbert_transform(self, x):
        """
        在时间维度上进行Hilbert变换
        Args:
            x: 输入张量 [B, C, T]
        Returns:
            解析信号 [B, C, T] (复数)
        """
        # FFT在时间维度上
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')

        # 创建Hilbert变换滤波器
        h = torch.zeros_like(x_fft)
        h[..., 1:x_fft.shape[-1]//2+1] = 2  # 正频率分量乘以2
        h[..., 0] = 1  # 直流分量保持不变

        # 应用Hilbert变换
        x_hilbert = x_fft * h

        # 逆FFT得到解析信号
        x_analytic = torch.fft.irfft(x_hilbert, n=x.shape[-1], dim=-1, norm='ortho')

        # 构造复数解析信号
        x_complex = x + 1j * x_analytic

        return x_complex

    def extract_multiscale_phase(self, x_complex):
        """
        多尺度相位特征提取
        Args:
            x_complex: 复数解析信号 [B, C, T]
        Returns:
            增强的相位特征 [B, C, T]
        """
        # 提取瞬时相位
        phase = torch.angle(x_complex)

        # 多尺度相位处理
        phase_features = []
        kernel_sizes = [3, 5, 7]

        for ks in kernel_sizes:
            padding = ks // 2
            phase_conv = F.conv1d(
                phase,
                weight=torch.ones(1, 1, ks).to(phase.device) / ks,
                padding=padding,
                groups=1
            )
            phase_features.append(phase_conv)

        # 合并多尺度特征
        phase_enhanced = torch.stack(phase_features, dim=0).mean(dim=0)

        # 应用相位特征提取网络
        phase_final = self.phase_feature_extractor(phase_enhanced)

        return phase_final

    def complex_convolution(self, x_complex, phase_features):
        """
        复数卷积操作
        Args:
            x_complex: 复数解析信号 [B, C, T]
            phase_features: 相位特征 [B, C, T]
        Returns:
            卷积结果 [B, c_out, T]
        """
        B, C, T = x_complex.shape

        # 将复数信号转换为极坐标形式
        magnitude = torch.abs(x_complex)
        phase = torch.angle(x_complex)

        # 结合提取的相位特征
        enhanced_phase = phase + 0.1 * phase_features  # 残差式相位增强

        # 重构复数信号
        x_enhanced = magnitude * torch.exp(1j * enhanced_phase)

        # 实部和虚部分别卷积
        real_part = x_enhanced.real
        imag_part = x_enhanced.imag

        # 实部卷积
        out_real = F.conv1d(
            real_part,
            self.complex_weight.real,
            padding=self.kernel_size // 2
        )

        # 虚部卷积
        out_imag = F.conv1d(
            imag_part,
            self.complex_weight.imag,
            padding=self.kernel_size // 2
        )

        # 合并实部和虚部结果
        output = out_real - out_imag  # 复数乘法的实部

        return output

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，可以是 [B, C, T] 或 [B, C, H, T] 或 [B, C, H, W, T]

        Returns:
            输出张量 [B, c_out, T]
        """
        # 处理不同维度的输入
        if x.dim() == 4:  # [B, C, H, T]
            B, C, H, T = x.shape
            # 去掉空间维度
            x = x.squeeze(2)  # [B, C, T]
        elif x.dim() == 3:  # [B, C, T]
            B, C, T = x.shape
        elif x.dim() == 5:  # [B, C, H, W, T]
            B, C, H, W, T = x.shape
            # 去掉空间维度
            x = x.squeeze(2).squeeze(2)  # [B, C, T]
        else:
            raise ValueError(f"Expected input shape [B, C, T], [B, C, H, T] or [B, C, H, W, T], got {x.shape}")

        # 确保输入通道数正确
        assert C == self.g * self.k, f"Input channels {C} don't match expected {self.g * self.k}"

        # Hilbert变换获取解析信号
        x_complex = self.hilbert_transform(x)

        # 多尺度相位特征提取
        phase_features = self.extract_multiscale_phase(x_complex)

        # 复数卷积
        conv_output = self.complex_convolution(x_complex, phase_features)

        # 残差连接
        if self.residual_conv is None:
            residual = x
        else:
            residual = self.residual_conv(x)

        # 确保维度匹配
        if conv_output.shape[-1] != residual.shape[-1]:
            # 调整残差维度
            diff = residual.shape[-1] - conv_output.shape[-1]
            if diff > 0:
                residual = residual[..., :conv_output.shape[-1]]
            else:
                conv_output = conv_output[..., :residual.shape[-1]]

        output = conv_output + residual

        # 应用注意力机制
        attention_weights = self.channel_attention(output)
        output = output * attention_weights

        # 输出归一化
        output = self.output_norm(output)

        return output


class MTCN(nn.Module):
    def __init__(self, n_class_primary, T = 256, channels=64, n_kernel_t=8,
    n_kernel_s=16, dropout=0.5, kernel_length=32, use_graph_conv=True, **kwargs):
        super(MTCN, self).__init__()

        self.n_class_primary = n_class_primary
        self.channels = channels
        self.n_kernel_t = n_kernel_t
        self.n_kernel_s = n_kernel_s
        self.dropout = dropout
        self.kernel_length = kernel_length
        self.T = T


        self.use_graph_conv = use_graph_conv  # 图卷积开关

        # ---------------------- 图卷积相关初始化（基于共享特征做图） ----------------------
        self.num_electrodes = channels  # 64个电极作为图节点
        self.graph_feat_dim = 16        # 与 block_shared_feature_extractor 输出通道一致

        # if self.use_graph_conv:
        #     # 使用改进的邻接矩阵构建，并注册为 buffer，自动跟随设备迁移
        #     adj = build_eeg_graph(self.num_electrodes, adj_type='knn')  # [64, 64]
        #     self.register_buffer('adj', adj) # 注册为buffer，数据自动跟随模型加载设备
        #     # 图卷积直接作用于共享特征：[B,64,16] -> [B,64,16]
        #     self.graph_conv = GraphConv(
        #         in_features=self.graph_feat_dim,
        #         out_features=self.graph_feat_dim,
        #         dropout=dropout
        #     )
        #     # 图特征融合强度，初始化为更保守的值，训练中自适应调整
        #     self.graph_alpha = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        # else:
        #     self.adj = None
        #     self.graph_conv = None
        #     self.graph_alpha = None



        # 新增图卷积层（插入共享特征提取后）
        # self.graph_conv = GraphConv(
        #     in_features=self.graph_feat_dim,
        #     out_features=self.graph_feat_dim,
        #     dropout=dropout
        # )
        # 移除对比损失（样本对应关系错误且与自监督任务目标冲突）
        self.contrastive_loss = ContrastiveLoss(temperature=0.1)


        # 修改后的共享特征提取器
        # self.block_shared_feature_extractor = nn.Sequential(
        #     nn.ZeroPad2d((15, 16, 0, 0)),
        #     # nn.Conv2d(1, 8, (1, 32), bias=False),  # 保持原有卷积
        #     # EEG_TDC_1D(1, 8, kernel_size=32, step=1),  # 替换原始时间卷积
        #     P300_TDC(1, 8, kernel_size=32),
        #     # Multi_Stage_P300_TDC(1, 8, kernel_size=32),  # 多阶段P300差分
        #     nn.BatchNorm2d(8),
        #     nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 4)),
        #     nn.Dropout(self.dropout)
        # )

         # ============ 替换原有的共享特征提取器 ============
        self.block_shared_feature_extractor = DualStreamSpatioTemporalExtractor(
            in_channels=1,

            out_channels=16,
            num_electrodes=channels,
            kernel_size=32,
            dropout=dropout
        )


        # ---------------------- 原有特征提取器保留，仅修改输出适配图卷积 ----------------------
        # 共享特征提取器（输出：[batch, 16, 1, 64] → 调整为 [batch, 64, 16] 适配图卷积）
        # self.block_shared_feature_extractor = nn.Sequential(
        #     nn.ZeroPad2d((15, 16, 0, 0)),
        #     nn.Conv2d(1, 8, (1, 32), bias=False),
        #     nn.BatchNorm2d(8),
        #     # 关键修改：空间卷积保持电极维度
        #     nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),  # 输出: [batch, 16, 1, time_dim]
        #     nn.BatchNorm2d(16),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 4)),
        #     nn.Dropout(self.dropout)
        # )



        # MTCN 特征提取器
        # self.block_shared_feature_extractor = nn.Sequential(
        #     # 原block1
        #     nn.ZeroPad2d((15, 16, 0, 0)),
        #     nn.Conv2d(1, 8, (1, 32), bias=False),
        #     nn.BatchNorm2d(8),
        #     # 原block2
        #     nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 4)),
        #     nn.Dropout(self.dropout)
        # )

        ############### 主任务特征提取器使用差分卷积 #################
        # self.block_specific_main_feature_extractor = nn.Sequential(
        #     # 原block1
        #     nn.ZeroPad2d((15, 16, 0, 0)),
        #     # nn.Conv2d(1, 8, (1, 32), bias=False),  # 保持原有卷积
        #     # EEG_TDC_1D(1, 8, kernel_size=32, step=1),  # 替换原始时间卷积
        #     P300_TDC(1, 8, kernel_size=32),
        #     # Multi_Stage_P300_TDC(1, 8, kernel_size=32),  # 多阶段P300差分
        #     nn.BatchNorm2d(8),
        #     # 原block2
        #     nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 4)),
        #     nn.Dropout(self.dropout)
        # )

        ############### 主任务特征提取器也使用双流 #################
        self.block_specific_main_feature_extractor = DualStreamSpatioTemporalExtractor(
            in_channels=1,
            out_channels=16,
            num_electrodes=channels,
            kernel_size=32,
            dropout=dropout
        )

        '''
        self.block_specific_mtr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_msr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_ftr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        '''

         # 增强时域掩码任务的特征提取器，使用多尺度时序卷积增强时域特征提取能力
        self.block_specific_mtr_feature_extractor = nn.Sequential(
            # 原block1 - 使用多尺度时序卷积增强时域特征提取
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 添加多尺度时序特征提取模块
            nn.Conv2d(8, 8, (1, 3), padding=(0, 1), bias=False),  # 小尺度感受野
            nn.ELU(),
            nn.Conv2d(8, 8, (1, 5), padding=(0, 2), bias=False),  # 中尺度感受野
            nn.ELU(),
            nn.Conv2d(8, 8, (1, 7), padding=(0, 3), bias=False),  # 大尺度感受野
            nn.ELU(),
            # 原block2 - 空间处理保持不变
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        # 增强空域掩码任务的特征提取器，使用空间注意力机制增强空间特征提取能力
        self.block_specific_msr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2 - 增强空间特征提取
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU()
        )

        # 添加空间注意力模块
        self.msr_spatial_attention = nn.Sequential(
            # 空间注意力模块
            nn.Conv2d(16, 1, kernel_size=7, padding=3),  # 大卷积核捕获空间依赖
            nn.Sigmoid()
        )

        # 添加空间特征后处理
        self.msr_post_process = nn.Sequential(
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        # 增强频域掩码任务的特征提取器，使用多尺度空洞卷积增强频域特征提取能力
        self.block_specific_ftr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8)
        )

        # 添加多尺度空洞卷积模块 (MDFA简化版) 用于频域特征提取
        self.ftr_mdfa = nn.ModuleDict({
            # 不同空洞率的卷积分支
            'branch1': nn.Sequential(
                nn.Conv2d(8, 4, 3, 1, padding=1, dilation=1),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True)
            ),
            'branch2': nn.Sequential(
                nn.Conv2d(8, 4, 3, 1, padding=2, dilation=2),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True)
            ),
            'branch3': nn.Sequential(
                nn.Conv2d(8, 4, 3, 1, padding=3, dilation=3),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True)
            ),
            'branch4': nn.Sequential(
                nn.Conv2d(8, 4, 1, 1, padding=0),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True)
            ),
            # 合并分支
            'fusion': nn.Sequential(
                nn.Conv2d(16, 16, 1, 1, padding=0),
                nn.BatchNorm2d(16),
                nn.ELU()
            )
        })

        # 添加频域特征后处理
        self.ftr_post_process = nn.Sequential(
            nn.Conv2d(16, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )

        self.block_feature_fusion = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//8-1, self.kernel_length//8, 0, 0)),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, self.kernel_length//4), groups=self.n_kernel_s, bias=False),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, 1), bias=False),
            nn.BatchNorm2d(self.n_kernel_s),
            nn.ELU()
        )
        self.main_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.vto_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.msp_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.ftr_task_projection_head = nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )

        # Fully-connected layer
        self.primary_task_classifier = nn.Sequential(
            # nn.Linear(self.n_kernel_s*T * 2 //32, self.n_class_primary)  # 图像用的这个
            nn.Linear(self.n_kernel_s*T //32, self.n_class_primary)
        )
        self.vto_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, 9)
        )
        self.msp_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, 8)
        )
        self.ftr_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T // 32, 10)
        )

        self.eca = nn.Sequential(
            ECAAttention(kernel_size=3)  # 实例化ECA注意力模块，指定核大小为3
        )
        self.cbam = nn.Sequential(
            CBAM(64)
        )




    def calculate_orthogonal_constraint(self, feature_1, feature_2):
        '''
        计算两个特征矩阵之间的正交约束损失，以确保它们在统计上尽可能正交（即相互独立）
        数学原理：若两个特征完全正交（即 feature_1^T feature_2 = 0），则它们的协方差为零，彼此独立。通过最小化非对角线元素的平方和，间接约束两者的内积接近零，从而实现正交性。
        确保共享特征专注于通用性，任务特定特征专注于任务差异性，从而提升特征的可区分性。
        '''
        assert feature_1.shape == feature_2.shape, "the dimension of two matrix is not equal"  # 确保输入的两个特征矩阵维度一致。
        N, channels, H, W = feature_1.shape
        feature_1, feature_2 = torch.reshape(feature_1, (N*channels, H, W)), torch.reshape(feature_2, (N*channels, H, W))  # 原始形状 (N, channels, H, W) → reshape为 (N*channels, H, W)。
        weight_squared = torch.bmm(feature_1, feature_2.permute(0, 2, 1)) #使用 torch.bmm 计算两个三维矩阵的批量矩阵乘积： permute()就是转置，将矩阵转置为 (N*channels, W, H)。
        # weight_squared = torch.norm(weight_squared, p=2)
        # ones = torch.ones(N*channels, H, H, dtype=torch.float32).to(torch.device('cuda:0')) # 创建全1矩阵 ones 和单位矩阵 diag
        # diag = torch.eye(H, dtype=torch.float32).to(torch.device('cuda:0'))
        ones = torch.ones(N*channels, H, H, dtype=torch.float32, device=feature_1.device)
        diag = torch.eye(H, dtype=torch.float32, device=feature_1.device)

        loss = ((weight_squared * (ones - diag)) ** 2).sum() # 计算非对角线元素的平方和：ones - diag:将单位矩阵的对角线置零，保留非对角线元素。 weight_squared * (ones - diag)：仅保留 weight_squared 的非对角线部分。
        return loss  # 返回一个标量 loss，表示两个特征矩阵的非正交程度。该损失函数会作为训练的惩罚项，通过反向传播优化模型参数，迫使 feature_1 和 feature_2 的统计相关性降低。


    # def calculate_orthogonal_constraint(self, feature_1, feature_2):
    #     assert feature_1.shape == feature_2.shape
    #     N, channels, H, W = feature_1.shape

    #     # 使用宽度维度而不是高度维度
    #     feature_1 = feature_1.reshape(N * channels, W, H)  # 转置H和W
    #     feature_2 = feature_2.reshape(N * channels, W, H)

    #     weight_squared = torch.bmm(feature_1, feature_2.permute(0, 2, 1))

    #     ones = torch.ones(N * channels, W, W, device=feature_1.device)
    #     diag = torch.eye(W, device=feature_1.device)
    #     mask = ones - diag

    #     # 计算非对角线元素的数量
    #     num_off_diagonal = mask.sum()

    #     loss = ((weight_squared * mask) ** 2).sum() / num_off_diagonal
    #     return loss

     # ---------------------- 新增：多任务特征收集（用于对比损失）----------------------
    def collect_task_features(self, task_features, task_name, feat):
        """收集各任务的高层特征，用于跨任务对比"""
        # 特征展平：[batch, 16, 1, 8] → [batch, 128]
        feat_flat = feat.view(feat.size(0), -1)
        task_features[task_name] = feat_flat
        return task_features

    def extract_electrode_features(self, x):
        """提取每个电极的特征表示（向量化实现，避免 Python for 循环）
        x: [batch, 1, 64, 256]
        返回: [batch, 64, 32]，每个电极 32 维特征
        """
        # 通过二维卷积在时间维上提取特征，并保持电极维度
        # conv: (N, 1, 64, 256) -> (N, 32, 64, T_out)
        if not hasattr(self, 'electrode_conv2d'):
            self.electrode_conv2d = nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(1, 32),
                stride=(1, 8),
                padding=(0, 12),
                bias=False
            ).to(x.device)

        feat_2d = self.electrode_conv2d(x)              # [N, 32, 64, T_out]
        # 在时间维上做自适应平均池化，得到每个电极的全局时间特征
        feat_2d = F.adaptive_avg_pool2d(feat_2d, (64, 1))  # [N, 32, 64, 1]
        feat_2d = feat_2d.squeeze(-1)                   # [N, 32, 64]

        # 调整维度为 [batch, 64, 32] 以适配 GraphConv
        electrode_features = feat_2d.permute(0, 2, 1).contiguous()
        return electrode_features


    def project_graph_features(self, graph_features):
        """将图卷积特征投影回原始特征空间"""
        # graph_features: [batch, 64, 32]
        if not hasattr(self, 'proj_conv'):
            self.proj_conv = nn.Conv2d(32, 16, 1).to(graph_features.device)

        graph_reshaped = graph_features.unsqueeze(2)  # [batch, 64, 1, 32]
        graph_projected = self.proj_conv(graph_reshaped.permute(0, 3, 2, 1))  # [batch, 16, 1, 64]
        return graph_projected.permute(0, 1, 2, 3)


    def forward(self, x, task_name, mode = 'train'):
        '''
        @description: Complete the corresponding task according to the task tag
        X； [32, 64, 256]
        x_pic:(32,64,256,256)
        mode: 'train'（训练，返回损失组件）/ 'eval'（推理，返回预测）
        task_features: 用于收集多任务特征（训练时使用）
        '''
        # 初始化多任务特征字典（仅训练时使用）
        task_features = {} if mode == 'train' else None
        # extract features
        # print("送进来： x.shape: ")
        # print(x.shape) # X； [32, 64, 256]
        # B,C,T = x.size()
        # x = x.view(1, B, C, T)
        # x = x.ca(x)
        # x = x.view(B, C, T)
        # x_pic_tmp = x # 为了转图像用
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2])) # [32,1,64,256]

        # x = torch.reshape(x, (x.shape[1], x.shape[2], x.shape[0], x.shape[3])) # [1,64,32,256]
        # x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1], x.shape[3]))  # 感觉上面的调整不对，应该调整成 [32,64,1,256]
        # x = self.eca(x)
        # x = self.cbam(x)
        # x = torch.reshape(x, (x.shape[2], x.shape[0], x.shape[1], x.shape[3]))  # [32, 1, 64, 256]
        # x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1], x.shape[3]))  # [32, 1, 64, 256]


        # fea_shared_extract = self.block_shared_feature_extractor(x)  # [32, 16, 1, 64]
        # fea_after_fusion = self.block_feature_fusion(fea_shared_extract) # [32, 16, 1, 64] # vto(288,16,1,64) # msp(256,16,1,64) # ftr(320,16,1,64)

        ####################使用图卷积#####################
        # 先进行基础特征提取
        fea_shared_extract = self.block_shared_feature_extractor(x)  # [B,16,1,64]
        fea_after_fusion = self.block_feature_fusion(fea_shared_extract)  # [B,16,1,64]


        # # 在共享特征上构建图，并将图卷积结果回注入融合特征（使用门控机制控制融合强度）
        # if self.use_graph_conv and self.graph_conv is not None and hasattr(self, 'adj') and self.adj is not None:
        #     # 将共享特征视为图节点特征：[B,16,1,64] -> [B,64,16]
        #     # 先将形状调整为 [B, time(64), channel(16)] 再作为每个电极的节点特征
        #     node_feat = fea_shared_extract.permute(0, 3, 1, 2).contiguous()  # [B,64,16,1]
        #     node_feat = node_feat.view(node_feat.size(0), node_feat.size(1), node_feat.size(2))  # [B,64,16]
        #     node_enhanced = self.graph_conv(node_feat, self.adj)                                  # [B,64,16]
        #     # 还原为卷积特征形状：[B,64,16] -> [B,16,1,64]
        #     graph_contribution = node_enhanced.permute(0, 2, 1).unsqueeze(2)        # [B,16,1,64]
        #     # 使用sigmoid门控，将graph_alpha限制在[0, 0.1]范围内，避免过度干扰原特征
        #     gate_weight = torch.sigmoid(self.graph_alpha) * 0.1
        #     fea_after_fusion = fea_after_fusion + gate_weight * graph_contribution


        if task_name == "main":
            # print("main ---- x.shape: ")
            # print(x.shape)   # [32, 1, 64, 256]
            # x = torch.reshape(x, (x.shape[1], x.shape[2], x.shape[0], x.shape[3])) # [1, 64, 32, 256]
            # x = self.eca(x)
            # x = torch.reshape(x, (x.shape[2], x.shape[0], x.shape[1], x.shape[3]))  # [32, 1, 64, 256]


            # 推理过程
            fea_specific_main = self.block_specific_main_feature_extractor(x) # [32, 16, 1, 64]
            # # 在特定任务特征提取后添加时空自注意力模块
            # fea_specific_main = self.sp_temp_attention(fea_specific_main)
            fea_main = fea_specific_main + fea_after_fusion #[32, 16, 1, 64]
            # fea_main = fea_specific_main   # 消融实验，消掉主任务的通用特征提取器

            fea_main = self.main_task_projection_head(fea_main)  # ([32, 16, 1, 8])   (32,32,1,8)
             # 收集主任务特征（训练时）
            if mode == 'train':
                task_features = self.collect_task_features(task_features, 'main', fea_main)
            # fea_main = self.eca(fea_main)
            fea_main = fea_main.view(fea_main.size(0), -1) # [32, 128]  (32,256)
            logits_main = self.primary_task_classifier(fea_main) # [32, 2]
            pred_main = F.softmax(logits_main, dim = 1) # [32, 2]
            # 损失计算
            # 过eca
            # fea_specific_main = self.eca(fea_specific_main)
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_main, fea_shared_extract) # 消融实验，消掉主任务的通用特征提取器，这里不用算正交损失了

            # return pred_main, orthogonal_constraint
            # return pred_main,0
            if mode == 'eval':
                return pred_main
            else:
                return pred_main, orthogonal_constraint, task_features

        elif task_name == "vto":
            # print("vto ---- x.shape: ")
            # print(x.shape)  # [288, 1, 64, 256]
            # x = torch.reshape(x, (x.shape[1], x.shape[2], x.shape[0], x.shape[3]))
            # x = self.eca(x)
            # x = torch.reshape(x, (x.shape[2], x.shape[0], x.shape[1], x.shape[3]))  # [32, 1, 64, 256]
            # 推理过程
            fea_specific_vto = self.block_specific_mtr_feature_extractor(x) #(288,16,1,64)
            fea_vto = (fea_specific_vto + fea_after_fusion)  # (288,16,1,64) = (288,16,1,64) + (288,16,1,64)

            fea_vto = self.vto_task_projection_head(fea_vto) # ([288, 16, 1, 8])
            if mode == 'train':
                task_features = self.collect_task_features(task_features, 'vto', fea_vto)
            fea_vto = fea_vto.view(fea_vto.size(0), -1) #(288, 128)
            logits_vto = self.vto_task_classifier(fea_vto)  #(288,9)
            pred_vto = F.softmax(logits_vto, dim = 1) #(288,9)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_vto, fea_shared_extract)

            # return pred_vto, orthogonal_constraint
            if mode == 'eval':
                return pred_vto
            else:
                return pred_vto, orthogonal_constraint, task_features

        elif task_name == "msp":
            # print("msp ---- x.shape: ")
            # print(x.shape)  # [256, 1, 64, 256]
            # x = torch.reshape(x, (x.shape[1], x.shape[2], x.shape[0], x.shape[3]))
            # x = self.eca(x)
            # x = torch.reshape(x, (x.shape[2], x.shape[0], x.shape[1], x.shape[3]))  # [32, 1, 64, 256]
            # 推理过程
            fea_specific_msp_pre = self.block_specific_msr_feature_extractor(x)  # (256,16,1,64)
            # 应用空间注意力
            spatial_attention = self.msr_spatial_attention(fea_specific_msp_pre)
            fea_specific_msp_enhanced = fea_specific_msp_pre * spatial_attention
            # 应用后处理
            fea_specific_msp = self.msr_post_process(fea_specific_msp_enhanced)

            fea_msp = (fea_specific_msp + fea_after_fusion)  #(256,16,1,64) = (256,16,1,64) + (256,16,1,64)

            fea_msp = self.msp_task_projection_head(fea_msp) # ([256, 16, 1, 8])
            if mode == 'train':
                task_features = self.collect_task_features(task_features, 'msp', fea_msp)
            fea_msp = fea_msp.view(fea_msp.size(0), -1) #(256,128)
            logits_msp = self.msp_task_classifier(fea_msp) #(256,8)
            pred_msp = F.softmax(logits_msp, dim = 1) #(256,8)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_msp, fea_shared_extract)

            # return pred_msp, orthogonal_constraint
            if mode == 'eval':
                return pred_msp
            else:
                return pred_msp, orthogonal_constraint, task_features

        elif task_name == "ftr":
            # print("ftr ---- x.shape: ")
            # print(x.shape)  # [320, 1, 64, 256]
            # 推理过程
            fea_specific_ftr_pre = self.block_specific_ftr_feature_extractor(x)  # (320,16,1,64)
             # 应用多尺度空洞卷积模块
            branch1_out = self.ftr_mdfa['branch1'](fea_specific_ftr_pre)
            branch2_out = self.ftr_mdfa['branch2'](fea_specific_ftr_pre)
            branch3_out = self.ftr_mdfa['branch3'](fea_specific_ftr_pre)
            branch4_out = self.ftr_mdfa['branch4'](fea_specific_ftr_pre)
              # 合并多尺度特征
            mdfa_out = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
            mdfa_out = self.ftr_mdfa['fusion'](mdfa_out)

            # 应用后处理
            fea_specific_ftr = self.ftr_post_process(mdfa_out)  # (320,16,1,64)

            fea_ftr = (fea_specific_ftr + fea_after_fusion)  # (320,16,1,64) = (320,16,1,64) + (320,16,1,64)

            fea_ftr = self.ftr_task_projection_head(fea_ftr)  # ([320, 16, 1, 8])
            if mode == 'train':
                task_features = self.collect_task_features(task_features, 'ftr', fea_ftr)
            fea_ftr = fea_ftr.view(fea_ftr.size(0), -1)  # (320,128)
            logits_ftr = self.ftr_task_classifier(fea_ftr)  # (320,8)
            pred_ftr = F.softmax(logits_ftr, dim=1)  # (320,8)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_ftr, fea_shared_extract)

            # return pred_ftr, orthogonal_constraint
            if mode == 'eval':
                return pred_ftr
            else:
                return pred_ftr, orthogonal_constraint, task_features

        else:
            assert("TaskName Error!")

'''
class EEGInception(nn.Module):
    def __init__(self, num_classes, channels=64, T=256, dropout=0.5):
        super(EEGInception, self).__init__()
        self.T = T
        self.channels = channels
        self.dropout = 0.5
        # input size: (N, 1, channels, T)
        self.time_block_11 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 8, (1, 64)),  # 时序分析
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout),
            nn.Conv2d(8, 16, (self.channels, 1), groups=8),  # 空间分析，片卷积（depthwse）
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout)
        )
        self.time_block_12 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout),
            nn.Conv2d(8, 16, (self.channels, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout)
        )
        self.time_block_13 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(1, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout),
            nn.Conv2d(8, 16, (self.channels, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout)
        )

        self.time_block_21 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(48, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout)
        )
        self.time_block_22 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(48, 8, (1, 8)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout)
        )
        self.time_block_23 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(48, 8, (1, 4)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout)
        )

        self.time_block_3 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(24, 12, (1, 8)),
            nn.BatchNorm2d(12),
            nn.Dropout(self.dropout)
        )

        self.time_block_4 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(12, 6, (1, 4)),
            nn.BatchNorm2d(6),
            nn.Dropout(self.dropout)
        )

        self.pool_1 = nn.AvgPool2d((1, 4))
        self.pool_2 = nn.AvgPool2d((1, 2))

        self.fc = nn.Linear(self.T // (4 * 2 * 2 * 2) * 6, num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x_11 = self.time_block_11(x)
        x_12 = self.time_block_12(x)
        x_13 = self.time_block_13(x)
        x = torch.cat((x_11, x_12, x_13), dim=1)
        x = self.pool_1(x)

        x_21 = self.time_block_21(x)
        x_22 = self.time_block_22(x)
        x_23 = self.time_block_23(x)
        x = torch.cat((x_21, x_22, x_23), dim=1)
        x = self.pool_2(x)

        x = self.time_block_3(x)
        x = self.pool_2(x)
        x = self.time_block_4(x)
        x = self.pool_2(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas
'''


'''
class VisionEagle(nn.Module):
    def __init__(self, n_class_primary, T=256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32):
        super(VisionEagle, self).__init__()
        self.n_class_primary = n_class_primary
        self.channels = channels
        self.n_kernel_t = n_kernel_t
        self.n_kernel_s = n_kernel_s
        self.dropout = dropout
        self.kernel_length = kernel_length

        self.block_shared_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )

        self.block_feature_fusion = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length // 8 - 1, self.kernel_length // 8, 0, 0)),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, self.kernel_length // 4), groups=self.n_kernel_s,
                      bias=False),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, 1), bias=False),
            nn.BatchNorm2d(self.n_kernel_s),
            nn.ELU()
        )

        self.block_specific_main_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_mtr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_msr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )

        self.main_task_projection_head = nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.vto_task_projection_head = nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.msp_task_projection_head = nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )

        # Fully-connected layer
        self.primary_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s * T // 32, self.n_class_primary)
        )
        self.vto_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s * T // 32, 9)
        )
        self.msp_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s * T // 32, 8)
        )


        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18_2 = models.resnet18(pretrained=True)
        # 修改第一层卷积层以处理 64 通道的输入
        # self.conv1 = self.resnet18.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = self.resnet18.bn1
        self.relu = self.resnet18.relu
        self.maxpool = self.resnet18.maxpool

        # 修改第一层卷积层以处理 64 通道的输入
        # self.conv12 = self.resnet18_2.conv1
        self.conv12 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn12 = self.resnet18_2.bn1
        self.relu2 = self.resnet18_2.relu
        self.maxpool2 = self.resnet18_2.maxpool

        self.layer1 = self.resnet18.layer1
        self.layer2 = self.resnet18.layer2
        self.layer3 = self.resnet18.layer3
        self.layer4 = self.resnet18.layer4
        self.layer12 = self.resnet18_2.layer1
        self.layer22 = self.resnet18_2.layer2
        self.layer32 = self.resnet18_2.layer3
        self.layer42 = self.resnet18_2.layer4

        self.scan_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.scan_attention = nn.Conv2d(128, 1, kernel_size=1)
        self.scan_conv22 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.scan_attention2 = nn.Conv2d(128, 1, kernel_size=1)
        self.scan_conv23 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.scan_attention3 = nn.Conv2d(256, 1, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, n_class_primary)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])  # (32,1,64,256)

        x1 = self.conv1(x) #(32,64,32,128)
        x1 = self.bn1(x1) #(32,64,32,128)
        x1 = self.relu(x1) #(32,64,32,128)
        x1 = self.maxpool(x1) #(32,64,16,64)

        scan_out = F.relu(self.scan_conv2(x1)) #(32,128,8,32)
        attention_map = torch.sigmoid(self.scan_attention(scan_out)) #(32,1,8,32)
        attention_map = F.interpolate(attention_map, size=x.size()[2:], mode='bilinear', align_corners=False) #(32,1,64,256)
        x = x * attention_map #(32,1,64,256)
        x = self.conv12(x) #(32,64,32,128)
        x = self.bn12(x) #(32,64,32,128)
        x = self.relu2(x) #(32,64,32,128)
        x = self.maxpool2(x) #(32,64,16,64)
        x1 = self.layer1(x) #(32,64,16,64)

        scan_out = F.relu(self.scan_conv22(x1)) #(32,128,8,32)
        attention_map = torch.sigmoid(self.scan_attention2(scan_out)) #(32,1,8,32)
        attention_map = F.interpolate(attention_map, size=x.size()[2:], mode='bilinear', align_corners=False) #(32,1,16,64)
        x = x * attention_map #(32,64,16,64)
        x = self.layer12(x) #(32,64,16,64)
        x1 = self.layer2(x) #(32,128,8,32)

        scan_out = F.relu(self.scan_conv23(x1)) #(32,256,4,16)
        attention_map = torch.sigmoid(self.scan_attention3(scan_out)) # (32,1,4,16)
        attention_map = F.interpolate(attention_map, size=x.size()[2:], mode='bilinear', align_corners=False) # (32,1,16,64)
        x = x * attention_map #(32,64,16,64)
        x = self.layer22(x) #(32,128,8,32)
        x = self.layer32(x) #(32,256,4,16)
        x = self.layer42(x)  #(32,512,2,8)

        # 推理部分
        x = self.avgpool(x) #(32,512,1,1)
        x = torch.flatten(x, 1) #(32,512)
        x = self.fc(x) #(32,2)
        probas = F.softmax(x, dim=1) #(32,2)
        return probas

'''

if __name__ == "__main__":
    data = torch.tensor(np.random.rand(64, 64, 256)).to(torch.float32)
    # 假设你有一个模型 model
    model = MTCN(n_class_primary=2, T = 256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32)
    # model = EEGInception(num_classes=2)
    a = model(data, "main")

