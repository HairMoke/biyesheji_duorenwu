import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleEEGModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 多尺度卷积分支（捕获不同感受野的时空特征）
        self.conv_short = nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 2), bias=False)  # 短期时间特征
        self.conv_medium = nn.Conv2d(1, 8, kernel_size=(1, 11), padding=(0, 5), bias=False)  # 中期时间特征
        self.conv_long = nn.Conv2d(1, 8, kernel_size=(1, 21), padding=(0, 10), bias=False)  # 长期时间特征

        self.bn = nn.BatchNorm2d(24)  # 三个分支合并后为24通道
        self.conv_out = nn.Conv2d(24, 1, kernel_size=1, bias=False)  # 压缩回1通道

        # 增强版时空注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # 空间特征提取
            nn.Sigmoid()
        )
        self.temporal_attn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2)),  # 时间特征提取
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征提取
        short = self.conv_short(x)
        medium = self.conv_medium(x)
        long = self.conv_long(x)

        # 合并多尺度特征
        concat = torch.cat([short, medium, long], dim=1)
        concat = F.relu(self.bn(concat))

        # 压缩回原始通道数
        feat = self.conv_out(concat)

        # 计算空间和时间注意力图
        spatial_mask = self.spatial_attn(x)  # 通道间注意力
        temporal_mask = self.temporal_attn(x)  # 时间点间注意力

        # 应用注意力并残差连接
        out = feat * spatial_mask * temporal_mask
        out = out + x  # 残差连接
        return out


class FrequencyAwareEEGModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 时域卷积分支
        self.time_conv = nn.Conv2d(1, 8, kernel_size=(3, 5), padding=(1, 2), bias=False)
        self.time_bn = nn.BatchNorm2d(8)

        # 频域卷积分支（使用短时傅里叶变换思想）
        self.freq_conv1 = nn.Conv2d(1, 8, kernel_size=(3, 7), padding=(1, 3), bias=False)  # 低频特征
        self.freq_conv2 = nn.Conv2d(1, 8, kernel_size=(3, 15), padding=(1, 7), bias=False)  # 高频特征
        self.freq_bn = nn.BatchNorm2d(16)

        # 特征融合与注意力
        self.fusion_conv = nn.Conv2d(24, 1, kernel_size=1, bias=False)
        self.attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 时域特征
        time_feat = F.relu(self.time_bn(self.time_conv(x)))

        # 频域特征（通过不同尺寸卷积核近似不同频段）
        freq_low = F.relu(self.freq_conv1(x))
        freq_high = F.relu(self.freq_conv2(x))
        freq_feat = torch.cat([freq_low, freq_high], dim=1)
        freq_feat = self.freq_bn(freq_feat)

        # 融合时空频特征
        concat = torch.cat([time_feat, freq_feat], dim=1)
        fused = self.fusion_conv(concat)

        # 注意力加权与残差连接
        attn_mask = self.attention(fused)
        out = fused * attn_mask + x
        return out


class EEGPreprocessingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 通道归一化（类似BatchNorm，但针对EEG通道）
        self.channel_norm = nn.BatchNorm2d(1)

        # 自适应特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道级归一化
        normed = self.channel_norm(x)

        # 增强有信息的特征，抑制噪声
        enhancement = self.enhance(normed)
        return normed * enhancement

class EEGEnhanceModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess = EEGPreprocessingLayer()
        self.attention = MultiScaleEEGModule()
        self.frequency = FrequencyAwareEEGModule()

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x):
        # 预处理增强
        preprocessed = self.preprocess(x)

        # 多模块特征提取
        attn_feat = self.attention(preprocessed)
        freq_feat = self.frequency(preprocessed)

        # 融合原始特征和增强特征
        concat = torch.cat([x, attn_feat, freq_feat], dim=1)
        fused = self.fusion(concat)

        # 最终残差连接
        return fused + x


import torch
import torch.nn as nn
import torch.nn.functional as F


class ImbalancedEEGModule(nn.Module):
    def __init__(self, alpha=2.0, beta=0.75):
        super().__init__()
        # 多尺度特征提取（保留原始设计）
        self.conv_short = nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.conv_medium = nn.Conv2d(1, 8, kernel_size=(1, 11), padding=(0, 5), bias=False)
        self.conv_long = nn.Conv2d(1, 8, kernel_size=(1, 21), padding=(0, 10), bias=False)
        self.bn = nn.BatchNorm2d(24)
        self.conv_out = nn.Conv2d(24, 1, kernel_size=1, bias=False)

        # 少数类增强注意力（关键改进）
        self.positive_enhance = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围[-1,1]，允许抑制或增强
        )

        # 自适应阈值机制（减少对负样本的依赖）
        self.threshold_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 残差连接权重（可学习的跳接强度）
        self.res_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # 多尺度特征提取
        short = self.conv_short(x)
        medium = self.conv_medium(x)
        long = self.conv_long(x)
        concat = torch.cat([short, medium, long], dim=1)
        feat = self.conv_out(F.relu(self.bn(concat)))

        # 少数类增强注意力（关键改进点）
        enhance_map = self.positive_enhance(feat)
        # 对正样本特征进行增强（通过非线性变换提升弱信号）
        enhanced_feat = feat * (1 + enhance_map)

        # 自适应阈值注意力（忽略低置信度区域，减少负样本干扰）
        threshold_map = self.threshold_attention(enhanced_feat)
        thresholded_feat = enhanced_feat * threshold_map

        # 可调节残差连接（控制原始信息与增强信息的比例）
        out = self.res_weight * thresholded_feat + (1 - self.res_weight) * x
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeAutoEnhanceModule(nn.Module):
    def __init__(self, adapt_threshold=0.8, pos_scale=1.5):
        super().__init__()
        # 1. 特征编码器（提取时空特征）
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        # 2. 样本类型检测器（自动区分正/负样本特征）
        self.sample_detector = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.Sigmoid()  # 输出样本属于正类别的概率
        )

        # 3. 正样本增强器（放大疑似正样本的特征）
        self.pos_enhancer = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=(1, 7), padding=(0, 3), bias=False),  # 时间维度增强
            nn.Tanh()  # 输出范围[-1,1]，可抑制或增强
        )

        # 4. 负样本抑制器（弱化疑似负样本的特征）
        self.neg_suppressor = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=(5, 1), padding=(2, 0), bias=False),  # 通道维度抑制
            nn.Sigmoid()  # 输出抑制权重（0-1）
        )

        # 5. 自适应阈值（控制样本类型判断的敏感度）
        self.adapt_threshold = adapt_threshold
        self.pos_scale = pos_scale  # 正样本增强系数

        # 6. 可学习的正样本原型（初始化时用随机值，训练中自动更新）
        self.register_buffer('pos_prototype', torch.randn(1, 4, 64, 256) * 0.01)
        self.prototype_momentum = 0.9  # 原型更新动量

        # 7. 安全通道恢复层（使用1x1卷积确保维度匹配）
        self.channel_recover = nn.Conv2d(4, 1, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)

        # 步骤1：提取特征
        feat = self.encoder(x)  # (batch,4,64,256)

        # 步骤2：计算样本与正样本原型的相似度（判断样本类型）
        sim_to_pos = F.cosine_similarity(
            feat.view(batch_size, -1),
            self.pos_prototype.view(1, -1).expand(batch_size, -1),
            dim=1
        ).view(batch_size, 1, 1, 1)  # (batch,1,1,1)

        # 步骤3：样本类型概率（相似度越高，越可能是正样本）
        pos_prob = torch.sigmoid(sim_to_pos * 5)  # 放大差异，使判断更明确

        # 步骤4：正样本增强（对疑似正样本进行特征强化）
        pos_enhance_map = 1 + self.pos_enhancer(feat) * pos_prob * self.pos_scale

        # 步骤5：负样本抑制（对疑似负样本进行特征弱化）
        neg_suppress_map = 1 - self.neg_suppressor(feat) * (1 - pos_prob)

        # 步骤6：应用增强和抑制
        enhanced_feat = feat * pos_enhance_map * neg_suppress_map

        # 步骤7：安全恢复原始通道数（使用1x1卷积）
        out = self.channel_recover(enhanced_feat)  # (batch,1,64,256)

        # 残差连接，确保信息不丢失
        out = out + x

        # 步骤8：自适应更新正样本原型（仅在训练时更新，且不影响主计算图）
        if self.training:
            # 找出最可能是正样本的样本（相似度最高的top-k%）
            k = max(1, int(batch_size * 0.2))  # 取前20%的样本
            _, top_indices = torch.topk(sim_to_pos.squeeze(), k)
            top_pos_samples = feat[top_indices]

            # 关键修复：使用detach()切断梯度流，避免干扰主训练过程
            self.pos_prototype = (self.prototype_momentum * self.pos_prototype +
                                  (1 - self.prototype_momentum) * top_pos_samples.detach().mean(dim=0, keepdim=True))

        return out

if __name__ == '__main__':
    # 创建一个SAFM实例并对一个随机输入进行处理
    x = torch.randn(32, 1, 64, 256)
    # Model = EEGEnhanceModule()
    # Model = ImbalancedEEGModule()
    Model = SafeAutoEnhanceModule()
    out = Model(x)
    print(out.shape)




