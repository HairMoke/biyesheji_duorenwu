import torch.nn as nn
from einops.layers.torch import Rearrange
import torch


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, original_width=64, target_width=32, num_heads=4, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.original_width = original_width
        self.target_width = target_width

        # 将输入特征图转换为序列（Batch, Channels, H, W）→ (Batch, H*W, Channels)
        self.to_sequence = Rearrange('b c h w -> b (h w) c')

        # Transformer编码器层（输入输出维度均为 in_channels）
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_channels,
                nhead=num_heads,
                dim_feedforward=in_channels * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # 新增线性层：将宽度从 original_width 降到 target_width（保持通道数不变）
        self.projection = nn.Linear(in_channels * original_width, in_channels * target_width)

    def forward(self, x):
        # x: [B, C, H, W] → 例如 [288, 16, 1, 64]
        B, C, H, W = x.shape

        # 转换为序列（H=1，W=64 → 序列长度=64）
        x_seq = self.to_sequence(x)  # [B, 64, C]

        # Transformer编码器处理（保持维度不变）
        x_encoded = self.transformer_encoder(x_seq)  # [B, 64, C]

        # 将序列展平为 [B, C*64]，通过线性层压缩宽度到32 → [B, C*32]
        x_flatten = x_encoded.view(B, -1)  # [B, 64*C]
        x_proj = self.projection(x_flatten)  # [B, 32*C]

        # 恢复形状 [B, C, 1, 32]
        x_lowdim = x_proj.view(B, C, 1, self.target_width)
        return x_lowdim


if __name__ == '__main__':
    model = TransformerEncoder(
        in_channels=16,
        original_width=64,
        target_width=32,
        num_heads=4,
        num_layers=2).cuda()
    # [32, 16, 1, 64] # vto(288,16,1,64) # msp(256,16,1,64) # ftr(320,16,1,64)
    t1 = torch.randn(256, 16, 1, 64).cuda()
    output = model(t1)
    print(output.shape)