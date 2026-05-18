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



class VisionEagle(nn.Module):
    def __init__(self, num_classes=2, channels=16, T=256, dropout=0.5):
        super(VisionEagle, self).__init__()
        self.T = T
        self.channels = channels
        self.dropout = 0.5
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18_2 = models.resnet18(pretrained=True)
        # self.conv1 = self.resnet18.conv1
        self.conv1 = nn.Conv2d(self.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = self.resnet18.bn1
        self.relu = self.resnet18.relu
        self.maxpool = self.resnet18.maxpool
        # self.conv12 = self.resnet18_2.conv1
        self.conv12 = nn.Conv2d(self.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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
        self.fc = nn.Linear(512, num_classes)
        # 添加新的层来调整输出形状
        self.shape_adjuster = nn.Sequential(
            nn.Conv2d(512, 16, kernel_size=1),  # 将通道数从512减少到16
            nn.AdaptiveAvgPool2d((1, 64))  # 将空间维度调整为(1, 64)
        )

    def forward(self, x):
        # 第一个 ResNet 分支
        x1 = self.conv1(x)   # 卷积层
        x1 = self.bn1(x1)    # 批归一化
        x1 = self.relu(x1)   # ReLU 激活
        x1 = self.maxpool(x1)  # 最大池化

        # 扫描卷积和注意力机制
        scan_out = F.relu(self.scan_conv2(x1)) # 扫描卷积
        attention_map = torch.sigmoid(self.scan_attention(scan_out)) # 注意力机制
        attention_map = F.interpolate(attention_map, size=x.size()[2:], mode='bilinear', align_corners=False) # 插值
        x = x * attention_map# 应用注意力图

        # 第二个 ResNet 分支
        x = self.conv12(x) #卷积层
        x = self.bn12(x) # 批归一化
        x = self.relu2(x)  # ReLU 激活
        x = self.maxpool2(x) # 最大池化

        # 扫描卷积和注意力机制
        x1 = self.layer1(x) # 第一个残差块 ResNet层
        scan_out = F.relu(self.scan_conv22(x1))  # 扫描卷积
        attention_map = torch.sigmoid(self.scan_attention2(scan_out)) # 注意力机制
        attention_map = F.interpolate(attention_map, size=x.size()[2:], mode='bilinear', align_corners=False) # 插值
        x = x * attention_map # 应用注意力图

        x = self.layer12(x) # 第二个残差块 ResNet层
        x1 = self.layer2(x) # 第三个残差块 ResNet层
        scan_out = F.relu(self.scan_conv23(x1)) # 扫描卷积
        attention_map = torch.sigmoid(self.scan_attention3(scan_out)) # 注意力机制
        attention_map = F.interpolate(attention_map, size=x.size()[2:], mode='bilinear', align_corners=False) # 插值
        x = x * attention_map # 应用注意力图（32，64，64，64）

        x = self.layer22(x) # ResNet 层  (32，128，32，32）
        x = self.layer32(x) # ResNet 层  (32,256,16,16)
        x = self.layer42(x) # ResNet 层 (32,512,8,8)

        x = self.shape_adjuster(x) #(32,16,1,64)
        return x
        # 推理部分, 先注释掉，测试特征提取能力
        # x = self.avgpool(x) # 全局平均池化
        # x = torch.flatten(x, 1) # 展平特征
        # x = self.fc(x) # 全连接层
        # probas = F.softmax(x, dim=1) # Softmax 激活
        # return probas #输出概率分布


# 示例代码
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input = torch.randn(3, 1, 512, 512)  # 生成随机输入
    # input = torch.randn(32,64,256,256)  # 生成随机输入
    input = torch.randn(32,1,64,256)  # 生成随机输入
    model = VisionEagle()  # 实例化TripletAttention
    output = model(input)  # 应用TripletAttention
    print(output.shape)  # 打印输出形状

    # model = VisionEagle(num_classes=2).to(device)
    # from torchinfo import summary
    # summary(model, (64, 1, 64, 64))