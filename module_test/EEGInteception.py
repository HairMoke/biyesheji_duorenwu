import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchvision.models as models
import numpy as np


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

    def forward(self, x): # x(64,64,256)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2]) # (64,1,64,256)
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


# 示例代码
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input = torch.randn(3, 1, 512, 512)  # 生成随机输入
    # model = VisionEagle()  # 实例化TripletAttention
    # print(output.shape)  # 打印输出形状
    # output = model(input)  # 应用TripletAttention

    model = EEGInception(num_classes=2).to(device)
    from torchinfo import summary
    summary(model, (64, 64, 256))