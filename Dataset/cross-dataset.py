import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
methods = ['EEGNet', 'PPNN', 'DRL', 'EEG conformer', 'EMDCN']  # 添加随机对比方法
metrics = ['BA', 'KAPPA', 'AUC']

# CAS到THU数据
data1 = {
    'EEGNet': [0.6345, 0.4779, 0.6971],
    'PPNN': [0.6892, 0.4608, 0.7513],
    'DRL': [0.4635, 0.4301, 0.4901],
    'EEG conformer': [0.6820, 0.4638, 0.7415],
    'EMDCN': [0.7159, 0.4916, 0.7612]
}

# THU到CAS数据
data2 = {
    'EEGNet': [0.6573, 0.4669, 0.6732],
    'PPNN': [0.6945, 0.4526, 0.7674],
    'DRL': [0.5021, 0.4289, 0.5107],
    'EEG conformer': [0.7015, 0.4817, 0.7531],
    'EMDCN': [0.7246, 0.4869, 0.7834]
}

# 颜色设置
colors = ['#FCB2AF', '#9BDFDF', '#FFE2CE', '#C4D8E9', '#BEBCDF']

# 创建一行两列的图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 设置柱状图位置
x = np.arange(len(metrics))
width = 0.12

# 绘制CAS到THU子图
for i, method in enumerate(methods):
    offset = (i - (len(methods) - 1) / 2) * (width * 1.15)  # 增加1.2倍的偏移间距
    bars = ax1.bar(x + offset, data1[method], width, label=method, color=colors[i])
    for bar, value in zip(bars, data1[method]):
        height = bar.get_height()
        # ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
        #          f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        width_bar = bar.get_width()
        # 计算 Y 轴的起始位置（因为设置了 ylim 从 0.30 开始）
        y_bottom = 0.30
        y_top = height
        y_middle = (y_bottom + y_top) / 2  # 柱子的实际垂直中心位置
        # 将数字竖直旋转 90 度，精确放置在柱子正中央
        ax1.text(bar.get_x() + width_bar / 2, y_middle,
                 f'{value:.4f}', ha='center', va='center', fontsize=18, fontweight='bold', rotation=90,
                 color='black')

ax1.set_xlabel('CAS到THU数据集的跨数据集实验结果', fontsize=24)
# ax1.set_ylabel('数值', fontsize=12)
# ax1.set_title('CAS到THU数据集的跨集合实验结果', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics,fontsize=20)
ax1.set_ylim([0.30, 0.85])
ax1.tick_params(axis='y', labelsize=20)  # Y 轴刻度数字字号调大到 20
ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

# 绘制THU到CAS子图
for i, method in enumerate(methods):
    offset = (i - (len(methods) - 1) / 2) * (width * 1.15)  # 增加1.2倍的偏移间距
    bars = ax2.bar(x + offset, data2[method], width, label=method, color=colors[i])
    for bar, value in zip(bars, data2[method]):
        height = bar.get_height()
        # ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
        #          f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        width_bar = bar.get_width()
        # 计算 Y 轴的起始位置（因为设置了 ylim 从 0.30 开始）
        y_bottom = 0.30
        y_top = height
        y_middle = (y_bottom + y_top) / 2  # 柱子的实际垂直中心位置
        # 将数字竖直旋转 90 度，精确放置在柱子正中央
        ax2.text(bar.get_x() + width_bar / 2, y_middle,
                 f'{value:.4f}', ha='center', va='center', fontsize=18, fontweight='bold', rotation=90,
                 color='black')

ax2.set_xlabel('THU到CAS数据集的跨数据集实验结果', fontsize=24)
# ax2.set_ylabel('数值', fontsize=12)
# ax2.set_title('THU到CAS数据集的跨集合实验结果', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=20)
ax2.set_ylim([0.30, 0.85])
ax2.tick_params(axis='y', labelsize=20)  # Y 轴刻度数字字号调大到 20
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

# 为第一个子图添加图例
# ax1.legend(loc='upper right', fontsize=10)
# ax1.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0.01, 0.99))
# ax1.legend(loc='upper center', fontsize=10, bbox_to_anchor=(0.5, 1.02))

# 将图例放在第二个图的右侧
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=18, title='方法', title_fontsize=18)


plt.tight_layout()
plt.savefig('跨数据集双向验证泛化性.png', dpi=300, bbox_inches='tight')
plt.show()