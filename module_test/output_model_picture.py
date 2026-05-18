'''
Author: Tammie li
Description: Define model
FilePath: \model.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchviz import make_dot
import os
os.environ["PATH"] += os.pathsep + 'E:/Program Files/Graphviz-12.2.1-win64/bin/'

class MTCN(nn.Module):
    def __init__(self, n_class_primary, T=256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32):
        super(MTCN, self).__init__()

        self.n_class_primary = n_class_primary
        self.channels = channels
        self.n_kernel_t = n_kernel_t
        self.n_kernel_s = n_kernel_s
        self.dropout = dropout
        self.kernel_length = kernel_length

        # MTCN 特征提取器
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
            nn.ZeroPad2d((self.kernel_length // 8 - 1, self.kernel_length // 8, 0, 0)),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, self.kernel_length // 4), groups=self.n_kernel_s,
                      bias=False),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, 1), bias=False),
            nn.BatchNorm2d(self.n_kernel_s),
            nn.ELU()
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
        self.ftr_task_projection_head = nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )

        # Fully-connected layer
        self.primary_task_classifier = nn.Sequential(
            # nn.Linear(self.n_kernel_s*T * 2 //32, self.n_class_primary)  # 图像用的这个
            nn.Linear(self.n_kernel_s * T // 32, self.n_class_primary)
        )
        self.vto_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s * T // 32, 9)
        )
        self.msp_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s * T // 32, 8)
        )
        self.ftr_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s * T // 32, 10)
        )


    def calculate_orthogonal_constraint(self, feature_1, feature_2):
        '''
        计算两个特征矩阵之间的正交约束损失，以确保它们在统计上尽可能正交（即相互独立）
        数学原理：若两个特征完全正交（即 feature_1^T feature_2 = 0），则它们的协方差为零，彼此独立。通过最小化非对角线元素的平方和，间接约束两者的内积接近零，从而实现正交性。
        确保共享特征专注于通用性，任务特定特征专注于任务差异性，从而提升特征的可区分性。
        '''
        assert feature_1.shape == feature_2.shape, "the dimension of two matrix is not equal"  # 确保输入的两个特征矩阵维度一致。
        N, channels, H, W = feature_1.shape
        feature_1, feature_2 = torch.reshape(feature_1, (N * channels, H, W)), torch.reshape(feature_2, (
        N * channels, H, W))  # 原始形状 (N, channels, H, W) → reshape为 (N*channels, H, W)。
        weight_squared = torch.bmm(feature_1, feature_2.permute(0, 2,
                                                                1))  # 使用 torch.bmm 计算两个三维矩阵的批量矩阵乘积： permute()就是转置，将矩阵转置为 (N*channels, W, H)。
        # weight_squared = torch.norm(weight_squared, p=2)
        ones = torch.ones(N * channels, H, H, dtype=torch.float32).to(torch.device('cuda:0'))  # 创建全1矩阵 ones 和单位矩阵 diag
        diag = torch.eye(H, dtype=torch.float32).to(torch.device('cuda:0'))

        loss = ((weight_squared * (
                    ones - diag)) ** 2).sum()  # 计算非对角线元素的平方和：ones - diag:将单位矩阵的对角线置零，保留非对角线元素。 weight_squared * (ones - diag)：仅保留 weight_squared 的非对角线部分。
        return loss  # 返回一个标量 loss，表示两个特征矩阵的非正交程度。该损失函数会作为训练的惩罚项，通过反向传播优化模型参数，迫使 feature_1 和 feature_2 的统计相关性降低。

    def forward(self, x, task_name):
        '''
        @description: Complete the corresponding task according to the task tag
        X； [32, 64, 256]
        x_pic:(32,64,256,256)
        '''
        # extract features
        # print("送进来： x.shape: ")
        # print(x.shape) # X； [32, 64, 256]
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))  # [32,1,64,256]


        fea_shared_extract = self.block_shared_feature_extractor(x)  # [32, 16, 1, 64]
        fea_after_fusion = self.block_feature_fusion(
            fea_shared_extract)  # [32, 16, 1, 64] # vto(288,16,1,64) # msp(256,16,1,64) # ftr(320,16,1,64)

        if task_name == "main":
            # print("main ---- x.shape: ")

            # 推理过程
            fea_specific_main = self.block_specific_main_feature_extractor(x)
            # 融合特征
            fea_main = fea_specific_main + fea_after_fusion  # [32, 16, 1, 64]


            fea_main = self.main_task_projection_head(fea_main)  # ([32, 16, 1, 8])   (32,32,1,8)
            # fea_main = self.eca(fea_main)
            fea_main = fea_main.view(fea_main.size(0), -1)  # [32, 128]  (32,256)
            logits_main = self.primary_task_classifier(fea_main)  # [32, 2]
            pred_main = F.softmax(logits_main, dim=1)  # [32, 2]
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_main, fea_shared_extract)

            return pred_main, orthogonal_constraint

        elif task_name == "vto":
            # 推理过程
            fea_specific_vto = self.block_specific_mtr_feature_extractor(x)  # (288,16,1,64)
            fea_vto = (fea_specific_vto + fea_after_fusion)  # (288,16,1,64) = (288,16,1,64) + (288,16,1,64)

            fea_vto = self.vto_task_projection_head(fea_vto)  # ([288, 16, 1, 8])
            # fea_vto = self.eca(fea_vto)
            fea_vto = fea_vto.view(fea_vto.size(0), -1)  # (288, 128)
            logits_vto = self.vto_task_classifier(fea_vto)  # (288,9)
            pred_vto = F.softmax(logits_vto, dim=1)  # (288,9)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_vto, fea_shared_extract)

            return pred_vto, orthogonal_constraint

        elif task_name == "msp":
            # print("msp ---- x.shape: ")
            # 推理过程 - 使用增强的空间特征提取器
            fea_specific_msp_pre = self.block_specific_msr_feature_extractor(x)  # (256,16,1,64)
            # 应用空间注意力
            spatial_attention = self.msr_spatial_attention(fea_specific_msp_pre)
            fea_specific_msp_enhanced = fea_specific_msp_pre * spatial_attention
            # 应用后处理
            fea_specific_msp = self.msr_post_process(fea_specific_msp_enhanced)
            fea_msp = (fea_specific_msp + fea_after_fusion)  # (256,16,1,64) = (256,16,1,64) + (256,16,1,64)

            fea_msp = self.msp_task_projection_head(fea_msp)  # ([256, 16, 1, 8])
            # fea_msp = self.eca(fea_msp)
            fea_msp = fea_msp.view(fea_msp.size(0), -1)  # (256,128)
            logits_msp = self.msp_task_classifier(fea_msp)  # (256,8)
            pred_msp = F.softmax(logits_msp, dim=1)  # (256,8)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_msp, fea_shared_extract)

            return pred_msp, orthogonal_constraint
        elif task_name == "ftr":
            # 推理过程 - 使用增强的频域特征提取器
            fea_specific_ftr_pre = self.block_specific_ftr_feature_extractor(x)  # 初始特征提取

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
            fea_ftr = fea_ftr.view(fea_ftr.size(0), -1)  # (320,128)
            logits_ftr = self.ftr_task_classifier(fea_ftr)  # (320,8)
            pred_ftr = F.softmax(logits_ftr, dim=1)  # (320,8)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_ftr, fea_shared_extract)

            return pred_ftr, orthogonal_constraint

        else:
            assert ("TaskName Error!")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(np.random.rand(64, 64, 256)).to(torch.float32).to(device)
    # 假设你有一个模型 model
    model = MTCN(n_class_primary=2, T=256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32).to(
        device)
    output, _ = model(data, "ftr")
    # 生成计算图
    dot = make_dot(output, params=dict(model.named_parameters()))

    # 保存为SVG文件
    dot.format = 'png'
    dot.render('model_structure_f')

