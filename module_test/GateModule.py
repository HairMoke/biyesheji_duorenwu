import torch
import torch.nn as nn
import torch.nn.functional as F

class GateModule(nn.Module):
    def __init__(self, features_shape):
        super(GateModule, self).__init__()
        self.features_shape = features_shape

        self.W1_f = nn.Parameter(torch.randn(features_shape * 2, features_shape))
        self.b1_f = nn.Parameter(torch.zeros(features_shape))
        self.W2_f = nn.Parameter(torch.randn(features_shape * 2, features_shape))
        self.b2_f = nn.Parameter(torch.zeros(features_shape))

        # 初始化权重
        nn.init.xavier_uniform_(self.W1_f)
        nn.init.xavier_uniform_(self.W2_f)

    def forward(self, input1, input2):
        # 确保输入是 2D (batch_size, features)
        # 如果输入是 4D (B, C, H, W)，需要先展平
        original_shape = input1.shape
        is_4d = len(original_shape) == 4

        if is_4d:
            # 假设特征维度是 C
            input1_flat = input1.permute(0, 2, 3, 1).reshape(-1, self.features_shape)
            input2_flat = input2.permute(0, 2, 3, 1).reshape(-1, self.features_shape)
        else:
            input1_flat = input1
            input2_flat = input2

        concat = torch.cat([input1_flat, input2_flat], dim=-1)
        
        G1 = torch.sigmoid(torch.matmul(concat, self.W1_f) + self.b1_f)
        G2 = torch.sigmoid(torch.matmul(concat, self.W2_f) + self.b2_f)
        
        gated_out1_flat = (1 - G1) * input1_flat + G1 * input2_flat
        gated_out2_flat = (1 - G2) * input2_flat + G2 * input1_flat

        if is_4d:
            # 恢复原始形状
            new_shape = (original_shape[0], original_shape[2], original_shape[3], original_shape[1])
            gated_out1 = gated_out1_flat.reshape(new_shape).permute(0, 3, 1, 2)
            gated_out2 = gated_out2_flat.reshape(new_shape).permute(0, 3, 1, 2)
        else:
            gated_out1 = gated_out1_flat
            gated_out2 = gated_out2_flat

        return gated_out1, gated_out2