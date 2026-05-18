import torch
import torch.nn as nn
import torch.nn.functional as F
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res



class EEGInception(nn.Module):
    def __init__(self, num_classes, C=32, T=256, drop_out=0.5):
        super(EEGInception, self).__init__()
        self.T = T
        self.C = C
        self.drop_out = 0.5
        # input size: (N, 1, C, T)
        self.time_block_11 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 8, (1, 64)),   #时序分析
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C*4, 1), groups=8),    #空间分析，片卷积（depthwse）
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.time_block_12 = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C*4, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.time_block_13 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(1, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (self.C*4, 1), groups=8),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_21 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(48, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_22 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(48, 8, (1, 8)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_23 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(48, 8, (1, 4)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_3 = nn.Sequential(
            # nn.ZeroPad2d((3, 4, 0, 0)),
            # nn.Conv2d(24, 12, (1, 8)),
            nn.Conv2d(48, 16, (self.C*2, 1), groups=8),     #这里的分片group卷积有问题不能一直是8
            # nn.BatchNorm2d(12),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_4 = nn.Sequential(
            # nn.ZeroPad2d((1, 2, 0, 0)),
            # nn.Conv2d(12, 6, (1, 4)),
            nn.Conv2d(16, 8, (self.C, 1), groups=8),
            # nn.BatchNorm2d(6),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.pool_2 = nn.AvgPool2d((1, 2))
        self.pool_3 = nn.AvgPool2d((2, 1))
        # self.fc = nn.Linear(self.T // (4 * 2 * 2 * 2) * 6, num_classes)
        self.fc = nn.Linear(18944, num_classes)
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x_11 = self.time_block_11(x)
        x_12 = self.time_block_12(x)
        x_13 = self.time_block_13(x)
        x = torch.cat((x_11, x_12, x_13), dim=1)
        # print('Inception',x.shape)
        x = self.pool_1(x)
        # print('1号池化后',x.shape)
        # x_21 = self.time_block_21(x)
        # x_22 = self.time_block_22(x)
        # x_23 = self.time_block_23(x)
        # x = torch.cat((x_21, x_22, x_23), dim=1)
        # x = self.pool_2(x)
        
        x = self.time_block_3(x)
        # print('通道提取，核长8*2',x.shape)
        x = self.pool_2(x)
        # print('2号池化后',x.shape)
        x = self.time_block_4(x)
        # print('通道提取，核长8*1',x.shape)
        x = self.pool_3(x)
        # print('3号池化后',x.shape)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas
    
class EEGChannel_1(nn.Module):
    ###模仿EEGInception，分不同长度的卷积核提取通道信息##
    def __init__(self, num_classes, C=32, T=256, drop_out=0.5):
        super(EEGChannel_1, self).__init__()
        self.T = T
        self.C = C
        self.drop_out = 0.5
        # # input size: (N, 1, C, T)
        # self.time_block_11 = nn.Sequential(
        #     nn.ZeroPad2d((31, 32, 0, 0)),
        #     nn.Conv2d(1, 8, (1, 64)),   #时序分析
        #     nn.BatchNorm2d(8),
        #     # nn.Dropout(self.drop_out), 
        #     nn.Conv2d(1, 8, (self.C*4, 1), groups=8),    #空间分析，片卷积（depthwse）
        #     nn.BatchNorm2d(16),
        #     nn.Dropout(self.drop_out)
        # )
        # self.time_block_12 = nn.Sequential(
        #     nn.ZeroPad2d((15, 16, 0, 0)),
        #     # nn.Conv2d(1, 8, (1, 32)),
        #     nn.BatchNorm2d(8),
        #     # nn.Dropout(self.drop_out), 
        #     nn.Conv2d(8, 16, (self.C*3, 1), groups=8),
        #     nn.BatchNorm2d(16),
        #     nn.Dropout(self.drop_out)
        # )
        # self.time_block_13 = nn.Sequential(
        #     nn.ZeroPad2d((7, 8, 0, 0)),
        #     # nn.Conv2d(1, 8, (1, 16)),
        #     nn.BatchNorm2d(8),
        #     # nn.Dropout(self.drop_out), 
        #     nn.Conv2d(8, 16, (self.C*2, 1), groups=8),
        #     nn.BatchNorm2d(16),
        #     nn.Dropout(self.drop_out)
        # )
        
        self.time_block_3 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 7, 8)),
            # nn.Conv2d(24, 12, (1, 8)),
            nn.Conv2d(48, 16, (self.C, 1), groups=8),     #这里的分片group卷积有问题不能一直是8
            # nn.BatchNorm2d(12),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_4 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 2)),
            # nn.Conv2d(12, 6, (1, 4)),
            nn.Conv2d(16, 8, (self.C, 1), groups=8),
            # nn.BatchNorm2d(6),
            nn.BatchNorm2d(8), 
            nn.Dropout(self.drop_out)
        )
        self.spatial_block11= nn.Sequential(
            nn.ZeroPad2d((0, 0, 3, 4)),
            nn.Conv2d(1, 8, (8, 1)),
            nn.Conv2d(8, 16, (1,self.T//4), groups=8),     
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.spatial_block12= nn.Sequential(
            nn.ZeroPad2d((0, 0, 7, 8)),
            nn.Conv2d(1, 8, (16, 1)),
            nn.Conv2d(8, 16, (1,self.T//4), groups=8),     
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.spatial_block13= nn.Sequential(
            nn.ZeroPad2d((0, 0, 15, 16)),
            nn.Conv2d(1, 8, (32, 1)),       #提取通道信息
            nn.Conv2d(8, 16, (1,self.T//4), groups=8),     
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
                
        
        self.time_block_21 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(48, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_22 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(48, 8, (1, 8)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_23 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(48, 8, (1, 4)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.pool_2 = nn.AvgPool2d((1, 2))
        self.pool_3 = nn.AvgPool2d((2, 1))      #通道可以卷积，但不要池化！！！！！！！！！所以后面的卷积还是以提取时序信息为主
        # self.fc = nn.Linear(self.T // (4 * 2 * 2 * 2) * 6, num_classes)
        
        self.fc = nn.Linear(40960 , num_classes)
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        # x_11 = self.time_block_11(x)
        # print('x_11',x_11.shape)
        # x_12 = self.time_block_12(x)
        # print('x_12',x_12.shape)
        # x_13 = self.time_block_13(x)
        # print('x_13',x_13.shape)
        # x = torch.cat((x_11, x_12, x_13), dim=1)
        # print('Inception',x.shape)
        
        x_11 = self.spatial_block11(x)
        x_12 = self.spatial_block12(x)
        x_13 = self.spatial_block13(x)
        x = torch.cat((x_11,x_12,x_13),dim=1)
        
        # x = self.pool_2(x)  #对通道的池化可以针对DPN做工作，即不是相邻两格数据平均，而是对应位置通道进行fusion
        # print('1号池化后',x.shape)
        x_21 = self.time_block_21(x)
        x_22 = self.time_block_22(x)
        x_23 = self.time_block_23(x)
        x = torch.cat((x_21, x_22, x_23), dim=1)
        # x = self.pool_2(x)
        
        x = self.time_block_3(x)
        print('通道提取，核长8*2',x.shape)
        x = self.pool_3(x)
        print('2号池化后',x.shape)
        x = self.time_block_4(x)
        print('通道提取，核长8*1',x.shape)
        x = self.pool_2(x)
        print('3号池化后',x.shape)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas    
    
    
class EEGChannel_2(nn.Module):
    #####
    def __init__(self, num_classes, C=8, T=256, drop_out=0.5):
        super(EEGChannel_2, self).__init__()
        self.T = T
        self.C = C
        self.drop_out = 0.5
        

        self.spatial_block11= nn.Sequential(
            nn.ZeroPad2d((0, 0, 3, 4)),
            nn.Conv2d(1, 8, (self.C, 1)),
            nn.Conv2d(8, 16, (1,self.T//4), groups=8),     
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        # self.spatial_block12= nn.Sequential(
        #     nn.ZeroPad2d((0, 0, 7, 8)),
        #     nn.Conv2d(1, 8, (16, 1)),
        #     nn.Conv2d(8, 16, (1,self.T//4), groups=8),     
        #     nn.BatchNorm2d(16),
        #     nn.Dropout(self.drop_out)
        # )
        # self.spatial_block13= nn.Sequential(
        #     nn.ZeroPad2d((0, 0, 15, 16)),
        #     nn.Conv2d(1, 8, (32, 1)),       #提取通道信息
        #     nn.Conv2d(8, 16, (1,self.T//4), groups=8),     
        #     nn.BatchNorm2d(16),
        #     nn.Dropout(self.drop_out)
        # )
        self.time_block_3 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 7, 8)),
            # nn.Conv2d(24, 12, (1, 8)),
            nn.Conv2d(48, 16, (self.C, 1), groups=8),     #这里的分片group卷积有问题不能一直是8
            # nn.BatchNorm2d(12),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_4 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 2)),
            # nn.Conv2d(12, 6, (1, 4)),
            nn.Conv2d(16, 8, (self.C, 1), groups=8),
            # nn.BatchNorm2d(6),
            nn.BatchNorm2d(8), 
            nn.Dropout(self.drop_out)
        )                
        
        self.time_block_21 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(48, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_22 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(48, 8, (1, 8)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_23 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(48, 8, (1, 4)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.pool_2 = nn.AvgPool2d((1, 2))
        self.pool_3 = nn.AvgPool2d((2, 1))      #通道可以卷积，但不要池化！！！！！！！！！所以后面的卷积还是以提取时序信息为主
        # self.fc = nn.Linear(self.T // (4 * 2 * 2 * 2) * 6, num_classes)
        
        self.fc = nn.Linear(40960 , num_classes)
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        
        #提取左视野脑电通道间特征
        x_r = self.spatial_block11(x_r)
        
        #提取右视野脑电通道间特征
        x_l = self.spatial_block11(x_l)
        
        #融合学习左右脑区效果
        x_lr = self.fusion_block(x_l,x_r)
        
        x_11 = self.spatial_block11(x)
        x_12 = self.spatial_block12(x)
        x_13 = self.spatial_block13(x)
        x = torch.cat((x_11,x_12,x_13),dim=1)
        
        # x = self.pool_2(x)  #对通道的池化可以针对DPN做工作，即不是相邻两格数据平均，而是对应位置通道进行fusion
        # print('1号池化后',x.shape)
        x_21 = self.time_block_21(x)
        x_22 = self.time_block_22(x)
        x_23 = self.time_block_23(x)
        x = torch.cat((x_21, x_22, x_23), dim=1)
        # x = self.pool_2(x)
        
        x = self.time_block_3(x)
        print('通道提取，核长8*2',x.shape)
        x = self.pool_3(x)
        print('2号池化后',x.shape)
        x = self.time_block_4(x)
        print('通道提取，核长8*1',x.shape)
        x = self.pool_2(x)
        print('3号池化后',x.shape)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas    
    
class EEGChannel_3(nn.Module):
    #####
    def __init__(self, num_classes, C=8, T=256, drop_out=0.5):
        super(EEGChannel_3, self).__init__()
        self.T = T
        self.C = C
        self.drop_out = 0.5
        self.spatial_block11= nn.Sequential(
            nn.ZeroPad2d((0, 0, 3, 4)),
            nn.Conv2d(1, 8, (self.C, 1)),
            nn.Conv2d(8, 16, (1,self.T//4), groups=8),     
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.dotconv = nn.Conv2d(2,1,(1,1))
        self.seperateconv = nn.Conv2d(2, 2, (self.C,1), groups=2)       # self.seperateconv = nn.Conv2d(2, 1, (self.C,1), groups=2) 能不能输出通道是1，这种融合和点卷积有区别吗？   答：不能这样输出
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.pool_2 = nn.AvgPool2d((1, 2))
        self.pool_3 = nn.AvgPool2d((2, 1))      #通道可以卷积，但不要池化！！！！！！！！！所以后面的卷积还是以提取时序信息为主
        # self.fc = nn.Linear(self.T // (4 * 2 * 2 * 2) * 6, num_classes)
        
        self.fc = nn.Linear(40960 , num_classes)
    def forward(self, x):
        #输入维度（16,128,512）
        #先变成（16,2,64,512）      相当于把两层结构变成两个“图像通道”
        #再变成（）

        #-----------------------------
        #？？？需要确定  x = x.reshape(x.shape[0], 2, x.shape[1]//2, x.shape[2])  reshape是不是前64在第一维、后64在第二维      #已确定，是的
        
        #先逐通道卷积，提取各自的时间维度信息
        #再普通卷积。？？？需确定卷积后是否是对应位置直接相加，若是，会不会有影响。若不是，能否有方法讲两个维度利用卷积合并成1个维度
        #---------------------------------------------------
#方法一：视作两通道，利用点卷积融合通道（本质是直接相加）
        x = x.reshape(x.shape[0], 2, x.shape[1]//2, x.shape[2])
        x = self.dotconv(x)
        print('点卷积后尺度',x.shape)   #(16,1,64,512)
        x = x.squeeze(dim=(1))        #(16,64,512)
# #Deprecated    方法二：视作两通道，取绝对值融合。2.1点卷积  2.2自己相加   取绝对值我觉得很不合适
#         x_r = x[:,:64,:]
#         x_l = x[:,64:,:]
        
# #方法三：视作两通道，利用分片卷积。卷积后，concat
#         x = x.reshape(x.shape[0], 2, x.shape[1]//2, x.shape[2])
#         x = self.seperateconv(x)
#         xl = x[:,0,:,:]
#         xr = x[:,1,:,:]
#         x = torch.cat((xl,xr),dim=1)
#         # x = self.pool_2(x)  #对通道的池化可以针对DPN做工作，即不是相邻两格数据平均，而是对应位置通道进行fusion
#         print(x.shape)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas    
    
class EEGInception_test(nn.Module):
    def __init__(self, num_classes, C=8, T=256, drop_out=0.5):
        super(EEGInception_test, self).__init__()
        self.T = T
        self.C = C
        self.drop_out = 0.5
        # input size: (N, 1, C, T)
        self.dotconv = nn.Conv2d(2,1,(1,1))
        self.time_block_11 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 15, 16)),
            nn.Conv2d(2, 8, (self.C*4, 1),groups=2),   
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (1,64)),    #空间分析，片卷积（depthwse）
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.time_block_12 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 11, 12)),
            nn.Conv2d(2, 8, (self.C*3, 1),groups=2),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (1,64)),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        self.time_block_13 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 7, 8)),
            nn.Conv2d(2, 8, (self.C*2,1),groups=2),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out), 
            nn.Conv2d(8, 16, (1,64)),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
         
        self.time_block_21 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(48, 8, (1, 16)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_22 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(48, 8, (1, 8)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        self.time_block_23 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(48, 8, (1, 4)),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_3 = nn.Sequential(
            # nn.ZeroPad2d((3, 4, 0, 0)),
            # nn.Conv2d(24, 12, (1, 8)),
            # nn.Conv2d(48, 16, (self.C*2, 1), groups=8),     #这里的分片group卷积有问题不能一直是8
            nn.Conv2d(48, 16, (1,self.C*2), groups=8),     #这里的分片group卷积有问题不能一直是8
            # nn.BatchNorm2d(12),
            nn.BatchNorm2d(16),
            nn.Dropout(self.drop_out)
        )
        
        self.time_block_4 = nn.Sequential(
            # nn.ZeroPad2d((1, 2, 0, 0)),
            # nn.Conv2d(12, 6, (1, 4)),
            # nn.Conv2d(16, 8, (self.C, 1), groups=8),
            nn.Conv2d(16, 8, (1,self.C), groups=8),
            # nn.BatchNorm2d(6),
            nn.BatchNorm2d(8),
            nn.Dropout(self.drop_out)
        )
        
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.pool_2 = nn.AvgPool2d((1, 2))
        self.pool_3 = nn.AvgPool2d((2, 1))
        # self.fc = nn.Linear(self.T // (4 * 2 * 2 * 2) * 6, num_classes)
        # self.fc = nn.Linear(10752, num_classes)
        self.fc = nn.Linear(12544, num_classes)
        
    def forward(self, x):

        #方法一：视作两通道，利用点卷积融合通道（本质是直接相加）
        x = x.reshape(x.shape[0], 2, x.shape[1]//2, x.shape[2])
        # x = self.dotconv(x)
        # print('点卷积后尺度',x.shape)   #(16,1,64,512)
        # x = x.squeeze(dim=(1))        #(16,64,512)
        x_11 = self.time_block_11(x)
        x_12 = self.time_block_12(x)
        x_13 = self.time_block_13(x)
        x = torch.cat((x_11, x_12, x_13), dim=1)
        # print('Inception',x.shape)
        x = self.pool_1(x)
        # print('1号池化后',x.shape)
        # x_21 = self.time_block_21(x)
        # x_22 = self.time_block_22(x)
        # x_23 = self.time_block_23(x)
        # x = torch.cat((x_21, x_22, x_23), dim=1)
        # x = self.pool_2(x)
        
        x = self.time_block_3(x)
        # print('通道提取，核长8*2',x.shape)
        x = self.pool_2(x)
        # print('2号池化后',x.shape)
        x = self.time_block_4(x)
        # print('通道提取，核长8*1',x.shape)
        x = self.pool_3(x)
        # print('3号池化后',x.shape)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas
    