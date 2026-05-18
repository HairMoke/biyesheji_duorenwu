import torch
import torch.nn as nn


class GAFGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def _time_series_to_gaf(self, time_series: torch.Tensor, type_: str = 'summation') -> torch.Tensor:
        # Normalize to [-1, 1]
        mean = time_series.mean(dim=1, keepdim=True)
        std = time_series.std(dim=1, keepdim=True) + 1e-8
        normalized = (time_series - mean) / std

        # Compute polar coordinates
        phi = torch.acos(torch.clamp(normalized, -1.0, 1.0))
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        if type_ == 'summation':
            gaf = cos_phi.unsqueeze(-1) * cos_phi.unsqueeze(-2) - sin_phi.unsqueeze(-1) * sin_phi.unsqueeze(-2)
        elif type_ == 'difference':
            gaf = sin_phi.unsqueeze(-1) * cos_phi.unsqueeze(-2) - cos_phi.unsqueeze(-1) * sin_phi.unsqueeze(-2)
        else:
            raise ValueError("Invalid GAF type")

        return gaf

    def generate_gaf(self, data: torch.Tensor) -> torch.Tensor:
        selected_channels = [26, 30, 39]  # 确保这些索引在输入数据的通道范围内

        # 提取指定通道的数据（形状: (batch_size, 3, time_steps)）
        selected_data = data[:, selected_channels, :]

        # 重组数据以进行批量处理（形状: (batch_size*3, time_steps)）
        time_series_batch = selected_data.permute(1, 0, 2).contiguous().view(-1, selected_data.size(2))

        # 生成GAF图像（形状: (batch_size*3, 256, 256)）
        gaf = self._time_series_to_gaf(time_series_batch, type_='summation')

        # 调整形状以匹配 (batch_size, 3, 256, 256)
        return gaf.view(data.size(0), 3, 256, 256)