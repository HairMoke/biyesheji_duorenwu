import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体样式（可根据论文要求调整）
plt.rcParams['font.family'] = ['SimHei', 'Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2


# 数据定义
noise_levels = [0, 10, 20, 30]  # 噪声比例（%）
x_ticks = noise_levels
x_labels = [f'{n}%' for n in noise_levels]

# DST-GCN 数据
dst_bas = [0.8582, 0.8138, 0.7684, 0.7106]
dst_tpr = [0.8562, 0.8005, 0.7158, 0.6728]
dst_fpr = [0.1398, 0.1729, 0.1790, 0.2516]
dst_f1  = [0.1985, 0.1601, 0.1025, 0.0759]
dst_kappa = [0.5181, 0.4515, 0.3450, 0.2998]
dst_auc  = [0.9267, 0.8692, 0.7516, 0.7352]

# EMDCN 数据
emd_bas = [0.8669, 0.8487, 0.8082, 0.7928]
emd_tpr = [0.8714, 0.8467, 0.7928, 0.7795]
emd_fpr = [0.1376, 0.1493, 0.1765, 0.1939]
emd_f1  = [0.2051, 0.1836, 0.1409, 0.1232]
emd_kappa = [0.5238, 0.4912, 0.4308, 0.4012]
emd_auc  = [0.9363, 0.9085, 0.8423, 0.8186]


# 创建子图：2行3列
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()  # 将二维数组展平以便索引

# 指标名称及对应的数据对
metrics = [
    ('BA', dst_bas, emd_bas),
    ('TPR', dst_tpr, emd_tpr),
    ('FPR', dst_fpr, emd_fpr),
    ('F1', dst_f1, emd_f1),
    ('KAPPA', dst_kappa, emd_kappa),
    ('AUC', dst_auc, emd_auc)
]

# 为每个子图绘制折线
for i, (name, dst_data, emd_data) in enumerate(metrics):
    ax = axes[i]
    # 绘制DST-GCN折线（蓝色圆圈标记）
    ax.plot(x_ticks, dst_data, marker='o', linestyle='-', color='blue',
            linewidth=2, markersize=6, label='DST-GCN')
    # 绘制EMDCN折线（红色方块标记）
    ax.plot(x_ticks, emd_data, marker='s', linestyle='-', color='red',
            linewidth=2, markersize=6, label='EMDCN')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('噪声比例', fontsize=11)
    ax.set_ylabel(name, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc='best', fontsize=9)

# 调整整体布局，防止重叠
plt.tight_layout()
# 保存图片为高分辨率（如300 dpi），格式可选pdf或png
# plt.savefig('robustness_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('添加不同的噪声比例-鲁棒性实验.png', dpi=300, bbox_inches='tight')
plt.show()