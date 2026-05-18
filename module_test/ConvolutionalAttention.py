import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as trunc_normal_init
class ConvolutionalAttention(nn.Module):
    def __init__(self, in_channels, inter_channels, num_heads=8):
        """
        构造函数：
        - in_channels：输入特征图的通道数
        - inter_channels：内部处理时使用的通道数
        - num_heads：多头注意力机制中的头数（默认8）
        """
        super(ConvolutionalAttention, self).__init__()
        out_channels = in_channels # 输出通道数与输入保持一致
        assert out_channels % num_heads == 0, \
            "out_channels ({}) must be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        # 使用同步批归一化确保跨设备一致性
        self.norm = nn.SyncBatchNorm(in_channels)

        # 初始化纵向卷积核参数（处理高度方向特征）
        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        # 初始化横向卷积核参数（处理宽度方向特征）
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        trunc_normal_init(self.kv, std=0.001) # 用截断正态分布初始化核参数
        trunc_normal_init(self.kv3, std=0.001)

    def _act_dn(self, x):
        # 获取 x 的形状：(B, c_inter, H, W)
        x_shape = x.shape
        h, w = x_shape[2], x_shape[3]
        # 重塑为 (B, head, c_per_head, L)，其中 L 为每个头的空间元素总数
        x = x.reshape([x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])
        # 对最后一个维度执行 softmax，以获得概率分布
        x = F.softmax(x, dim=3)
        # 对每个头的通道进行归一化，防止数值过大
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        # 恢复为原始形状 (B, c_inter, H, W)
        x = x.reshape([x_shape[0], self.inter_channels, h, w])
        return x

    def forward(self, x):
        # 首先进行批归一化处理
        x = self.norm(x)

        """
        处理纵向特征：
        使用 self.kv 进行卷积，沿高度方向填充 (3, 0) 确保输出尺寸与输入一致
        """
        x1 = F.conv2d(x, self.kv, bias=None, stride=1, padding=(3, 0))
        x1 = self._act_dn(x1) # 对卷积输出进行 softmax 归一化处理
        x1 = F.conv2d(x1, self.kv.transpose(1, 0), bias=None, stride=1, padding=(3, 0))
        # ai缝合大王

        """
        处理横向特征：
        利用 self.kv3 进行卷积，设置填充为 (0, 3) 以维持高度不变，宽度填充保证输出尺寸正确
        """
        x3 = F.conv2d(x, self.kv3, bias=None, stride=1, padding=(0, 3))
        x3 = self._act_dn(x3)
        x3 = F.conv2d(x3, self.kv3.transpose(1, 0), bias=None, stride=1, padding=(0, 3))

        # 将纵向和横向卷积的结果叠加，实现不同方向特征的融合
        x = x1 + x3
        return x

if __name__ == "__main__":
    model = ConvolutionalAttention(64, 32, 8) # 创建模型实例：输入64通道，中间32通道，8头注意力
    # input = torch.randn(1, 64, 32, 32) # 随机生成输入张量，尺寸为 (1, 64, 32, 32)
    input = torch.randn(1, 64, 32, 256) # 随机生成输入张量，尺寸为 (1, 64, 32, 32)
    output = model(input) # 前向传播得到输出
    print("Output shape:", output.shape)