import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from typing import List, Tuple


class EEGFreqFeatureExtractor(nn.Module):
    """
    脑电频域特征提取模块（通道扩展版）
    输入：[32,1,64,256] → 输出：[32,8,64,256]
    核心：MACB交叉耦合特征 + 频段功率谱特征，单通道→8通道映射
    """

    def __init__(
            self,
            fs: int = 128,  # 采样频率（Hz）
            freq_bands: dict = None,  # 脑电频段定义
            freq_pairs: List[Tuple[str, str]] = None,  # 交叉频率对
            tau: int = 5,  # 耦合滞后值（时间点）
            dropout: float = 0.1
    ):
        super().__init__()
        self.fs = fs
        self.tau = tau
        self.dropout = nn.Dropout(dropout)

        # 1. 脑电频段定义（默认常用频段，可自定义）
        self.freq_bands = freq_bands or {
            'delta': (1, 4),  # δ波
            'theta': (4, 8),  # θ波
            'alpha': (8, 13),  # α波
            'beta': (13, 30),  # β波
            'gamma': (30, 45)  # γ波
        }
        self.band_names = list(self.freq_bands.keys())
        self.num_bands = len(self.band_names)  # 5个频段

        # 2. 交叉频率对（基于脑电生理机制的关键耦合组合）
        self.freq_pairs = freq_pairs or [
            ('delta', 'theta'), ('theta', 'alpha'), ('alpha', 'beta'),
            ('beta', 'gamma'), ('alpha', 'gamma'), ('delta', 'alpha')
        ]
        self.num_pairs = len(self.freq_pairs)  # 6组频率对

        # 3. 特征融合+通道扩展层（11类特征→8通道，保持其他维度）
        # 输入特征数：num_pairs（6） + num_bands（5）= 11
        self.feature_fusion = nn.Sequential(
            # 第一层：11类特征融合，保持空间/时间维度
            nn.Conv2d(
                in_channels=self.num_pairs + self.num_bands,
                out_channels=32,  # 中间特征增强
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 第二层：映射到8通道（目标输出通道数）
            nn.Conv2d(
                in_channels=32,
                out_channels=8,  # 输出通道=8
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        # 4. 残差连接（单通道→8通道，确保维度匹配）
        self.residual_conv = nn.Conv2d(
            in_channels=1,  # 输入通道=1
            out_channels=8,  # 输出通道=8
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _get_band_omega(self, band_name: str) -> Tuple[float, float]:
        """将频段名称转换为角频率（rad/s）"""
        f_min, f_max = self.freq_bands[band_name]
        omega_min = 2 * torch.pi * f_min
        omega_max = 2 * torch.pi * f_max
        return omega_min, omega_max

    def _power_spectrum_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取各频段功率谱特征（基础频域特征）
        输入：[32,1,64,256] → 输出：[32,5,1,64,256]（5个频段特征）
        """
        batch, ch, feat_dim, time = x.shape  # ch=1
        x_fft = fft.fft(x, dim=-1)  # [32,1,64,256]（复张量）
        freq_axis = fft.fftfreq(time, 1 / self.fs)  # [256]
        omega_axis = 2 * torch.pi * freq_axis  # [256]

        power_features = []
        for band in self.band_names:
            omega_min, omega_max = self._get_band_omega(band)
            band_mask = (omega_axis >= omega_min) & (omega_axis <= omega_max)
            band_fft = x_fft[..., band_mask]  # [32,1,64,K]
            band_power = torch.abs(band_fft) ** 2 / time  # 功率计算
            # 频段内平均+广播到时间维度
            band_power = band_power.mean(dim=-1, keepdim=True).repeat(1, 1, 1, time)  # [32,1,64,256]
            power_features.append(band_power)

        return torch.stack(power_features, dim=1)  # [32,5,1,64,256]

    def _macb_coupling_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        基于MACB提取交叉频率耦合特征（核心非线性特征）
        输入：[32,1,64,256] → 输出：[32,6,1,64,256]（6组频率对特征）
        """
        batch, ch, feat_dim, time = x.shape  # ch=1
        # 处理滞后避免越界
        if self.tau > 0 and time > self.tau:
            x_lag = x[..., self.tau:]
            x_lead = x[..., :-self.tau]
            min_time = min(x_lag.shape[-1], x_lead.shape[-1])
            x_lag = x_lag[..., :min_time]
            x_lead = x_lead[..., :min_time]
            pad_size = time - min_time
            x_lag = F.pad(x_lag, (0, pad_size), mode='constant', value=0)
            x_lead = F.pad(x_lead, (0, pad_size), mode='constant', value=0)
        else:
            x_lag = x
            x_lead = x

        x_lag_fft = fft.fft(x_lag, dim=-1)
        x_lead_fft = fft.fft(x_lead, dim=-1)
        freq_axis = fft.fftfreq(time, 1 / self.fs)
        omega_axis = 2 * torch.pi * freq_axis

        coupling_features = []
        for (band1, band2) in self.freq_pairs:
            # 频段角频率范围
            omega1_min, omega1_max = self._get_band_omega(band1)
            omega2_min, omega2_max = self._get_band_omega(band2)
            omega_sum_min = omega1_min + omega2_min
            omega_sum_max = omega1_max + omega2_max

            # 频率掩码
            mask1 = (omega_axis >= omega1_min) & (omega_axis <= omega1_max)
            mask2 = (omega_axis >= omega2_min) & (omega_axis <= omega2_max)
            mask_sum = (omega_axis >= omega_sum_min) & (omega_axis <= omega_sum_max)

            # 频域信号提取+平均
            fft1 = x_lag_fft[..., mask1].mean(dim=-1, keepdim=True)  # [32,1,64,1]
            fft2 = x_lag_fft[..., mask2].mean(dim=-1, keepdim=True)  # [32,1,64,1]
            fft_sum = x_lead_fft[..., mask_sum].mean(dim=-1, keepdim=True)  # [32,1,64,1]

            # MACB反对称构造+归一化
            cross_bispec = fft1 * fft2 * fft_sum.conj()
            cross_bispec_anti = cross_bispec - cross_bispec.conj()
            power1 = torch.abs(fft1) ** 2 / time
            power2 = torch.abs(fft2) ** 2 / time
            power_sum = torch.abs(fft_sum) ** 2 / time
            norm_term = torch.sqrt(power1 * power2 * power_sum + 1e-8)
            coupling = torch.abs(cross_bispec_anti) / norm_term  # [32,1,64,1]

            # 广播到时间维度
            coupling = coupling.repeat(1, 1, 1, time)  # [32,1,64,256]
            coupling_features.append(coupling)

        return torch.stack(coupling_features, dim=1)  # [32,6,1,64,256]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：输入→功率谱特征+耦合特征→融合扩展→残差连接→输出
        输入：[32,1,64,256] → 输出：[32,8,64,256]
        """
        # 残差分支（1通道→8通道，保持其他维度）
        residual = self.residual_conv(x)  # [32,8,64,256]

        # 1. 提取基础频域特征（5个频段）→ [32,5,1,64,256]
        power_feats = self._power_spectrum_feature(x)

        # 2. 提取交叉耦合特征（6组频率对）→ [32,6,1,64,256]
        coupling_feats = self._macb_coupling_feature(x)

        # 3. 特征拼接：11类特征合并 → [32,11,1,64,256]
        combined_feats = torch.cat([power_feats, coupling_feats], dim=1)

        # 4. 维度调整：适配卷积层输入（[batch, feat_num, ch, feat_dim, time] → [batch×ch, feat_num, feat_dim, time]）
        combined_feats = combined_feats.permute(0, 2, 1, 3, 4)  # [32,1,11,64,256]
        combined_feats = combined_feats.reshape(
            x.shape[0] * x.shape[1],  # 32×1=32
            self.num_pairs + self.num_bands,  # 11
            x.shape[2],  # 64
            x.shape[3]  # 256
        )  # 调整后：[32,11,64,256]

        # 5. 特征融合+通道扩展（11→8通道）→ [32,8,64,256]
        fused_feats = self.feature_fusion(combined_feats)

        # 6. 恢复原始batch维度（32→32,8,64,256）
        fused_feats = fused_feats.reshape(
            x.shape[0],  # 32
            8,  # 目标通道数
            x.shape[2],  # 64
            x.shape[3]  # 256
        )  # 最终特征：[32,8,64,256]

        # 7. 残差连接+激活+dropout（增强泛化性）
        output = F.relu(fused_feats + residual)
        output = self.dropout(output)

        return output


# ------------------------------ 模块测试 ------------------------------
if __name__ == "__main__":
    # 生成输入数据 [32,1,64,256]
    torch.manual_seed(42)
    input_data = torch.randn(32, 1, 64, 256).cuda()

    # 初始化特征提取器
    extractor = EEGFreqFeatureExtractor(
        fs=128,
        dropout=0.1
    ).cuda()

    # 前向传播测试
    with torch.no_grad():
        output = extractor(input_data)

    # 验证维度
    print(f"输入维度：{input_data.shape}")
    print(f"输出维度：{output.shape}")
    print(f"维度符合要求：{output.shape == torch.Size([32, 8, 64, 256])}")  # 应输出 True

    # 验证可训练参数
    total_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"可训练参数总数：{total_params:,}")  # 约 5k 参数，轻量化