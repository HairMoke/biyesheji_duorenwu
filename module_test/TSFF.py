import torch
import torch.nn as nn

'''
def raw_depth_attention(x):
    """ x: input features with shape [N, C, H, W] """
    N, C, H, W = x.size()
    # K = W if W % 2 else W + 1
    k = 7
    adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
    conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True).to(x.device)  # original kernel k
    softmax = nn.Softmax(dim=-2)
    x_pool = adaptive_pool(x)
    x_transpose = x_pool.transpose(-2, -3)
    y = conv(x_transpose)
    y = softmax(y)
    y = y.transpose(-2, -3)
    return y * C * x


class TSFF(nn.Module):

    def __init__(self, img_weight=0.02, width=224, length=224, num_classes=2, samples=1000, channels=3, avepool=25):
        super(TSFF, self).__init__()
        self.channel_weight = nn.Parameter(torch.randn(9, 1, channels), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.num_classes = num_classes
        self.img_weight = img_weight

        self.raw_time_conv = nn.Sequential(
            nn.Conv2d(9, 24, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=(1, 75), groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )

        self.raw_chanel_conv = nn.Sequential(
            nn.Conv2d(24, 9, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=(channels, 1), groups=9, bias=False),
            nn.BatchNorm2d(9),
            nn.GELU(),
        )

        self.raw_norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        # raw features
        raw_eeg = torch.ones((1, 1, channels, samples))
        raw_eeg = torch.einsum('bdcw, hdc->bhcw', raw_eeg, self.channel_weight)
        out_raw_eeg = self.raw_time_conv(raw_eeg)
        out_raw_eeg = self.raw_chanel_conv(out_raw_eeg)
        out_raw_eeg = self.raw_norm(out_raw_eeg)
        out_raw_eeg_shape = out_raw_eeg.cpu().data.numpy().shape
        print('out_raw_eeg_shape: ', out_raw_eeg_shape)
        n_out_raw_eeg = out_raw_eeg_shape[-1] * out_raw_eeg_shape[-2] * out_raw_eeg_shape[-3]

        self.frequency_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(4, 4), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8),
            nn.Dropout(p=0.25),

            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, out_raw_eeg_shape[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_raw_eeg_shape[-1]),
            nn.Conv2d(out_raw_eeg_shape[-1], out_raw_eeg_shape[-1], kernel_size=4,
                      groups=out_raw_eeg_shape[-1], bias=False, padding=2),
            nn.BatchNorm2d(out_raw_eeg_shape[-1]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.25),
        )

        img_eeg = torch.ones((1, 3, width, length))
        out_img = self.frequency_features(img_eeg)
        out_img_shape = out_img.cpu().data.numpy().shape
        n_out_img = out_img_shape[-1] * out_img_shape[-2] * out_img_shape[-3]
        print('n_out_img shape: ', out_img_shape)

        self.classifier = nn.Sequential(
            nn.Linear(n_out_img, num_classes),
        )

    def forward(self, x_raw, x_frequency):
        # features for frequency graph
        x_frequency = self.frequency_features(x_frequency)
        x_frequency = x_frequency.view(x_frequency.size(0), -1)

        x_raw = torch.einsum('bdcw, hdc->bhcw', x_raw, self.channel_weight)
        x_raw = self.raw_time_conv(x_raw)
        x_raw = self.raw_chanel_conv(x_raw)
        x_raw = raw_depth_attention(x_raw)
        x_raw = self.raw_norm(x_raw)
        # raw features and img features weighted fusion
        # Check the order of magnitudes for both features.
        x_raw_flatten = x_raw.view(x_raw.size(0), -1)

        weighted_features = x_raw_flatten * (1 - self.img_weight) + x_frequency * self.img_weight

        x = self.classifier(weighted_features)

        return x, x_raw_flatten, x_frequency


# 示例代码
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input = torch.randn(3, 1, 512, 512)  # 生成随机输入
    # input = torch.randn(32,64,256,256)  # 生成随机输入
    input = torch.randn(32,64,256)  # 生成随机输入
    model = TSFF()
    output = model(input)
    print(output.shape)

'''


import torch
import numpy as np
from torch import nn


# 引入模型代码中的函数和类
def raw_depth_attention(x):
    """ x: input features with shape [N, C, H, W] """
    N, C, H, W = x.size()
    k = 7
    adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
    conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True).to(x.device)
    softmax = nn.Softmax(dim=-2)
    x_pool = adaptive_pool(x)
    x_transpose = x_pool.transpose(-2, -3)
    y = conv(x_transpose)
    y = softmax(y)
    y = y.transpose(-2, -3)
    return y * C * x


class TSFF(nn.Module):
    def __init__(self, img_weight=0.02, width=224, length=224, num_classes=2,
                 samples=256, channels=64, time_kernel_size=25, avepool=None):
        """
        参数调整说明:
        - samples: 输入时间点数(如256)
        - channels: EEG通道数(如64)
        - time_kernel_size: 时间卷积核大小(默认25，可自动计算)
        - avepool: 最终平均池化大小(自动计算保持输出尺寸合理)
        """
        super(TSFF, self).__init__()

        # 自动计算池化大小(保持与原始模型相似的降采样比例)
        if avepool is None:
            avepool = max(1, samples // 40)  # 原1000/25=40

        # 自动调整时间卷积核大小(保持与原始模型相似的感受野比例)
        if time_kernel_size is None:
            time_kernel_size = max(3, samples // 10)  # 原1000/75≈13

        self.channel_weight = nn.Parameter(torch.randn(9, 1, channels), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.num_classes = num_classes
        self.img_weight = img_weight

        # 时间卷积层(调整kernel_size适应新输入长度)
        self.raw_time_conv = nn.Sequential(
            nn.Conv2d(9, 24, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=(1, time_kernel_size),
                      groups=24, bias=False, padding=(0, time_kernel_size // 2)),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )

        # 通道卷积层(自动适应输入通道数)
        self.raw_chanel_conv = nn.Sequential(
            nn.Conv2d(24, 9, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=(channels, 1), groups=9, bias=False),
            nn.BatchNorm2d(9),
            nn.GELU(),
        )

        # 归一化层(使用自动计算的avepool)
        self.raw_norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )

        # 计算原始EEG特征输出形状(动态适应输入尺寸)
        raw_eeg = torch.ones((1, 1, channels, samples))
        raw_eeg = torch.einsum('bdcw, hdc->bhcw', raw_eeg, self.channel_weight)
        out_raw_eeg = self.raw_time_conv(raw_eeg)
        out_raw_eeg = self.raw_chanel_conv(out_raw_eeg)
        out_raw_eeg = self.raw_norm(out_raw_eeg)
        out_raw_eeg_shape = out_raw_eeg.shape
        print('动态计算的原始EEG特征形状:', out_raw_eeg_shape)

        # 时频特征网络(保持不变，仍输出224x224)
        self.frequency_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(4, 4), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8),
            nn.Dropout(p=0.25),

            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, out_raw_eeg_shape[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_raw_eeg_shape[-1]),
            nn.Conv2d(out_raw_eeg_shape[-1], out_raw_eeg_shape[-1], kernel_size=4,
                      groups=out_raw_eeg_shape[-1], bias=False, padding=2),
            nn.BatchNorm2d(out_raw_eeg_shape[-1]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.25),
        )

        # 分类器(动态适应特征维度)
        self.classifier = nn.Sequential(
            nn.Linear(out_raw_eeg_shape[-1] * out_raw_eeg_shape[-2] * out_raw_eeg_shape[-3], num_classes),
        )

    def forward(self, x_raw, x_frequency):
        x_frequency = self.frequency_features(x_frequency)
        x_frequency = x_frequency.view(x_frequency.size(0), -1)

        x_raw = torch.einsum('bdcw, hdc->bhcw', x_raw, self.channel_weight)
        x_raw = self.raw_time_conv(x_raw)
        x_raw = self.raw_chanel_conv(x_raw)
        x_raw = raw_depth_attention(x_raw)
        x_raw = self.raw_norm(x_raw)
        x_raw_flatten = x_raw.view(x_raw.size(0), -1)

        weighted_features = x_raw_flatten * (1 - self.img_weight) + x_frequency * self.img_weight
        x = self.classifier(weighted_features)

        return x, x_raw_flatten, x_frequency


class TSFFFeatureExtractor(nn.Module):
    def __init__(self, img_weight=0.5, samples=256, channels=64,
                 time_kernel_size=25, avepool=None):
        super(TSFFFeatureExtractor, self).__init__()

        # 初始化参数
        self.img_weight = img_weight
        self.channels = channels

        # 原始EEG特征提取路径（保持不变）
        self.channel_weight = nn.Parameter(torch.randn(9, 1, channels), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.raw_time_conv = nn.Sequential(
            nn.Conv2d(9, 24, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=(1, time_kernel_size),
                      groups=24, bias=False, padding=(0, time_kernel_size // 2)),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )

        self.raw_chanel_conv = nn.Sequential(
            nn.Conv2d(24, 9, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=(1, 1), groups=9, bias=False),
            nn.BatchNorm2d(9),
            nn.GELU(),
        )

        # 维度恢复层（保持不变）
        self.dim_recover = nn.Conv2d(9, 1, kernel_size=1)

        # 时频特征提取路径（调整为处理64通道输入）
        self.frequency_features = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(4, 4), stride=1, padding=2),  # 输入通道改为64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, 16, kernel_size=(4, 4), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.25),

            nn.Conv2d(16, 9, kernel_size=1, bias=False),  # 最终输出9通道
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=4, groups=9, bias=False, padding=2),
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            nn.Dropout(p=0.25),
        )

        # 维度对齐层（调整输入通道数）
        self.freq_feat_adjust = nn.Sequential(
            nn.Conv2d(9, 1, kernel_size=1),  # 将9通道降为1通道
            nn.Upsample(size=(channels, samples))  # 调整空间维度
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x_raw, x_frequency):
        # 原始EEG特征提取（保持不变）
        x_raw = torch.einsum('bdcw, hdc->bhcw', x_raw, self.channel_weight)
        x_raw = self.raw_time_conv(x_raw)
        x_raw = self.raw_chanel_conv(x_raw)
        x_raw = raw_depth_attention(x_raw)  # 假设这是自定义的注意力模块
        x_raw = self.dim_recover(x_raw)  # [batch, 1, channels, samples]

        # 时频特征提取（适配64通道输入）
        x_frequency = self.frequency_features(x_frequency)  # [batch, 9, H, W]
        x_frequency = self.freq_feat_adjust(x_frequency)    # [batch, 1, channels, samples]

        # 特征融合（确保维度匹配）
        assert x_raw.shape == x_frequency.shape == torch.Size([x_raw.size(0), 1, self.channels, x_raw.size(-1)])
        fused_features = x_raw * (1 - self.img_weight) + x_frequency * self.img_weight

        return fused_features  # 形状保持: [batch, 1, channels, samples]

import numpy as np
import pywt
import torch
from scipy import signal
from matplotlib import pyplot as plt

import numpy as np
import pywt
import torch
from scipy import signal


def generate_time_frequency_map(eeg_signal, fs=250, target_size=(224, 224)):
    """
    修正后的时频图生成函数

    参数:
        eeg_signal: 输入EEG信号,形状为[channels, time_points]
        fs: 采样频率(默认为250Hz)
        target_size: 目标时频图大小

    返回:
        time_freq_map: 时频图张量,形状为[3, 224, 224]
    """
    # 设置复数Morlet小波参数
    bandwidth = 1.5  # 带宽参数B
    center_freq = 1.0  # 中心频率参数C
    wavelet_name = f'cmor{bandwidth}-{center_freq}'

    # 频率范围设置(4-38Hz，与论文一致)
    frequencies = np.arange(4, 38, 0.5)

    # 为每个通道生成时频图
    channel_maps = []
    for channel_data in eeg_signal:
        # 计算小波尺度
        scales = fs * (1 / frequencies)

        # 连续小波变换(使用修正的小波名称)
        coefficients, _ = pywt.cwt(channel_data, scales, wavelet_name, sampling_period=1 / fs)

        # 计算幅度并平方(得到功率)
        power = np.abs(coefficients) ** 2

        # 转换为dB尺度
        time_freq = 10 * np.log10(power + 1e-12)

        # 归一化到[0,1]范围
        time_freq = (time_freq - time_freq.min()) / (time_freq.max() - time_freq.min())
        channel_maps.append(time_freq)

    # 合并三个通道的时频图
    merged_map = np.stack(channel_maps, axis=0)

    # 调整尺寸到目标大小
    resized_map = []
    for ch in range(3):
        resized = signal.resample(merged_map[ch], target_size[0], axis=0)
        resized = signal.resample(resized, target_size[1], axis=1)
        resized_map.append(resized)

    return torch.FloatTensor(np.stack(resized_map, axis=0))


import torch
import torch.nn.functional as F
from kymatio.torch import Scattering1D


def generate_time_frequency_map_kymatio(eeg_signal, fs=250, target_size=(224, 224)):
    """
    使用 `kymatio` 库实现 GPU 加速的时频图生成
    Args:
        eeg_signal: (C, T) EEG 信号（GPU Tensor）
        fs: 采样率（Hz）
        target_size: 输出时频图大小（H, W）
    Returns:
        time_freq_map: (3, H, W) 时频图（GPU Tensor）
    """
    device = eeg_signal.device
    T = eeg_signal.shape[-1]  # 时间点数

    # 初始化 Scattering1D（小波散射变换）
    J = 8  # 最大尺度（覆盖 4-38Hz）
    Q = 12  # 每个尺度的滤波器数
    scattering = Scattering1D(J=J, Q=Q, shape=T).to(device)

    # 计算每个通道的时频特征
    channel_maps = []
    for channel_data in eeg_signal:
        # 计算散射变换（返回功率谱）
        Sx = scattering(channel_data.unsqueeze(0))  # (1, n_scales, n_time)
        power = torch.abs(Sx) ** 2  # 计算功率

        # 转换为 dB 并归一化
        time_freq = 10 * torch.log10(power + 1e-12)
        time_freq = (time_freq - time_freq.min()) / (time_freq.max() - time_freq.min())
        channel_maps.append(time_freq.squeeze(0))

    # 合并通道并调整大小
    merged_map = torch.stack(channel_maps, dim=0)  # (C, n_scales, n_time)
    resized_map = F.interpolate(
        merged_map.unsqueeze(0),  # (1, C, n_scales, n_time)
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (C, H, W)

    return resized_map

def main():
    # 初始化特征提取器
    feature_extractor = TSFFFeatureExtractor(
        img_weight=0.3,  # 时频特征权重
        samples=256,  # 输入时间点数
        channels=64  # 输入通道数
    )

    # 模拟输入数据
    batch_size = 32
    x_raw = torch.randn(batch_size, 1, 64, 256)  # 原始EEG
    # 时频图数据：形状[N, 3, width, length]
    x_freq = torch.stack([
        generate_time_frequency_map(x_raw[i, 0].numpy())
        for i in range(batch_size)
    ])
    # x_freq = torch.randn(batch_size, 3, 224, 224)  # 时频图

    # 特征提取
    fused_features = feature_extractor(x_raw, x_freq)

    # 检查输出维度
    print(f"融合特征形状: {fused_features.shape}")  # 应为 torch.Size([32, 1, 64, 256])

if __name__ == "__main__":
    main()

