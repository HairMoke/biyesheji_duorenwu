import torch
import torch.nn as nn
import torch.fft

import torch
import torch.nn as nn
import torch.fft


class SpatioTemporalFreqTransformer(nn.Module):
    def __init__(self, input_channels, seq_len, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.d_model = d_model

        # 1. 时空分支（输出维度: d_model//2）
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_channels, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU()
        )

        # 2. 频域分支（修复维度问题）
        self.fft_size = seq_len // 2 + 1  # 实信号FFT输出维度 (129)
        self.freq_conv = nn.Sequential(
            nn.Linear(self.fft_size, seq_len),  # 关键修复: 输入129 → 输出256
            nn.GELU()
        )

        # 3. 双分支融合层（静态定义避免动态建层）
        self.fusion_proj = nn.Linear(d_model // 2 + input_channels, d_model)  # 拼接后投影到d_model

        # 4. Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # 5. 残差连接与输出
        self.res_proj = nn.Conv1d(input_channels, d_model, 1) if input_channels != d_model else nn.Identity()
        self.output_proj = nn.Conv1d(d_model, input_channels, 1)

    def forward(self, x):
        residual = x  # (B, C, L)

        # ---- 时空分支 ----
        temporal_feat = self.temporal_conv(x)  # (B, d_model//2, L)
        temporal_feat = temporal_feat.permute(0, 2, 1)  # (B, L, d_model//2)

        # ---- 频域分支（修复后）----
        freq_feat = torch.fft.rfft(x, dim=-1, norm='ortho')  # (B, C, F) F=129
        freq_feat = torch.abs(freq_feat)  # 幅度谱
        freq_feat = self.freq_conv(freq_feat)  # (B, C, L) [修复1: 输出256]
        freq_feat = freq_feat.permute(0, 2, 1)  # (B, L, C)

        # ---- 双分支拼接 ----
        fused = torch.cat([temporal_feat, freq_feat], dim=-1)  # (B, L, d_model//2 + C)
        fused = self.fusion_proj(fused)  # (B, L, d_model) [修复2: 静态投影层]

        # ---- Transformer融合 ----
        encoded = self.transformer_encoder(fused)  # (B, L, d_model)
        encoded = encoded.permute(0, 2, 1)  # (B, d_model, L)

        # ---- 残差连接 ----
        residual = self.res_proj(residual)  # (B, d_model, L)
        output = self.output_proj(encoded + residual)  # (B, C, L)
        return output

if __name__ == '__main__':
    # 创建一个SAFM实例并对一个随机输入进行处理
    x = torch.randn(32, 64, 256)
    Model = SpatioTemporalFreqTransformer(input_channels=64,
            seq_len=256,
            d_model=128,   # 隐层维度（建议64-256）
            nhead=8,       # 注意力头数
            num_layers=2    # Transformer层数
    )
    out = Model(x)
    print(out.shape)