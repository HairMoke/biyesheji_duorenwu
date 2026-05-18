

'''
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def time_series_to_gaf(time_series, type='summation'):
    # 对时间序列进行归一化到 [-1, 1] 的范围
    normalized = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-8)

    # 将归一化的时间序列映射到余弦和正弦值（极坐标）
    phi = np.arccos(np.clip(normalized, -1.0, 1.0))
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    if type == 'summation':
        gaf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
    elif type == 'difference':
        gaf = np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)
    else:
        raise ValueError("Type must be either 'summation' or 'difference'.")

    return gaf


# 创建随机的脑电数据集作为示例
num_samples = 2  # 样本数减少，以便快速测试
num_channels = 3  # 通道数减少，以便快速测试
num_timepoints = 64  # 采样点数减少，以便快速测试

# 构建随机数据
np.random.seed(42)  # 设置随机种子以保证结果可复现
eeg_data = np.random.rand(num_samples, num_channels, num_timepoints)
labels = np.random.randint(0, 2, size=num_samples)

# 创建一个目录来保存生成的图像
output_dir = Path('test_eeg_images')
output_dir.mkdir(parents=True, exist_ok=True)

for sample_idx in range(eeg_data.shape[0]):
    for channel_idx in range(eeg_data.shape[1]):
        # 获取单个通道的时间序列数据
        time_series = eeg_data[sample_idx, channel_idx]

        # 转换为GAF
        gaf_summation = time_series_to_gaf(time_series, type='summation')
        gaf_difference = time_series_to_gaf(time_series, type='difference')

        # 构造文件名
        file_name_base = f'sample_{sample_idx}_channel_{channel_idx}'
        summation_file = output_dir / f'{file_name_base}_gaf_summation.png'
        difference_file = output_dir / f'{file_name_base}_gaf_difference.png'

        # 保存GAF图像
        plt.imsave(summation_file, gaf_summation, cmap='gray')
        plt.imsave(difference_file, gaf_difference, cmap='gray')

print(f"Test images saved to {output_dir}")

# 如果你想要查看某个具体的GAF图像，可以取消下面两行的注释
plt.imshow(gaf_summation, cmap='gray')
plt.show()



'''



import numpy as np
from pathlib import Path

'''
def time_series_to_gaf(time_series, type='summation'):
    # 对时间序列进行归一化到 [-1, 1] 的范围
    normalized = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-8)

    # 将归一化的时间序列映射到余弦和正弦值（极坐标）
    phi = np.arccos(np.clip(normalized, -1.0, 1.0))
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    if type == 'summation':
        gaf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
    elif type == 'difference':
        gaf = np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)
    else:
        raise ValueError("Type must be either 'summation' or 'difference'.")

    return gaf

eeg_data = np.load("E:/code/python/PycharmProject/ZHB_BiYeSheJi/Dataset/THU/S01/x_train.npy")
# 假设 eeg_data 是已经读取的脑电数据，形状为 (256, 64, 256)
# labels 是对应的标签，形状为 (256,)
# 创建一个目录来保存生成的numpy数据
output_dir = Path('processed_eeg_data')
output_dir.mkdir(parents=True, exist_ok=True)

# 初始化一个空的列表来存储所有GAF图像
gaf_images = []

# 遍历每个样本和每个通道，转换为GAF并添加到列表中
for sample_idx in range(eeg_data.shape[0]):
    for channel_idx in range(eeg_data.shape[1]):
        # 获取单个通道的时间序列数据
        time_series = eeg_data[sample_idx, channel_idx]

        # 转换为GAF
        gaf_summation = time_series_to_gaf(time_series, type='summation')  # 或者 'difference'

        # 将GAF图像添加到列表中
        gaf_images.append(gaf_summation)

# 将列表中的所有GAF图像重新组织成所需的形状 (256, 64, 256, 256)
gaf_images = np.array(gaf_images).reshape((256, 64, 256, 256))

# 保存为单一的numpy文件
output_file = output_dir / 'all_gaf_images.npy'
np.save(output_file, gaf_images)

print(f"All GAF images saved to {output_file}")


'''
data = np.load("processed_eeg_data/all_gaf_images.npy")
print(data.shape)

