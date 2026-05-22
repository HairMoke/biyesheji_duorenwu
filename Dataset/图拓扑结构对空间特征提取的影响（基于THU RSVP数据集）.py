import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Songti SC']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 20

# 1. 准备数据
metrics = ['BA', 'TPR', 'FPR', 'F1', 'KAPPA', 'AUC']
data = {
    'Random': [0.8305, 0.8234, 0.1624, 0.1756, 0.4897, 0.8943],
    'Distance': [0.8478, 0.8413, 0.1456, 0.1850, 0.5078, 0.9145],
    'KNN (k=6)': [0.8582, 0.8562, 0.1398, 0.1985, 0.5181, 0.9267]
}
methods_cn = {
    'Random': 'Random',
    'Distance': 'Distance',
    'KNN (k=6)': 'KNN(k=6)'
}

# 2. 设置学术风格配色
academic_colors = ['#F8CBAD', '#C5E0B4', '#B4C7E7']

# 3. 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

# 4. 绘制分组柱状图
x = np.arange(len(metrics))  # 指标的位置
width = 0.22  # 调整柱宽

# 绘制三组柱子，调整偏移量减少组间缝隙
bars1 = ax.bar(x - width*1.1, data['Random'], width,
               label=methods_cn['Random'], color=academic_colors[0],
               edgecolor='black', linewidth=1.2, alpha=0.9, zorder=3)
bars2 = ax.bar(x, data['Distance'], width,
               label=methods_cn['Distance'], color=academic_colors[1],
               edgecolor='black', linewidth=1.2, alpha=0.9, zorder=3)
bars3 = ax.bar(x + width*1.1, data['KNN (k=6)'], width,
               label=methods_cn['KNN (k=6)'], color=academic_colors[2],
               edgecolor='black', linewidth=1.2, alpha=0.9, zorder=3)

# 5. 添加数值标签
# def add_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         vertical_offset = 4
#         ax.annotate(f'{height:.4f}',
#                    xy=(bar.get_x() + bar.get_width() / 2, height),
#                    xytext=(0, vertical_offset),
#                    textcoords="offset points",
#                    ha='center', va='bottom', fontsize=9, fontweight='normal')

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        width = bar.get_width()
        # 将数字竖直旋转 90 度，放置在柱子内部靠上的位置
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + width / 2, height * 0.5),  # 垂直居中
                   xytext=(0, 0),
                   textcoords="offset points",
                   ha='center', va='center',
                   fontsize=12, fontweight='bold', rotation=90,  # 旋转 90 度，字号大幅调大
                   color='black')  # 确保数字颜色清晰可见


add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# 6. 设置图表属性
ax.set_xlabel('评估指标', fontsize=24, fontweight='bold', labelpad=10)
ax.set_ylabel('性能得分', fontsize=24, fontweight='bold', labelpad=10)
ax.set_title('图拓扑结构对空间特征提取的影响（基于THU RSVP数据集）',
             fontsize=28, pad=20, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=20, fontweight='bold')

# 优化Y轴范围
ax.set_ylim(0, 1.0)

# 7. 调整图例位置
# 将图例位置下移一点，避免遮挡标题
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02),
                   fontsize=18, title='拓扑结构类型', title_fontsize=18,
                   frameon=True, fancybox=False, edgecolor='gray',
                   ncol=1, columnspacing=1.0, handlelength=1.5)

# 8. 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# 9. 美化边框
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('gray')

# 10. 调整布局并保存
plt.tight_layout(rect=[0, 0, 1, 0.90])  # 为上方的图例和标题留出适当空间

output_filename = '图拓扑结构对空间特征提取的影响（基于THU RSVP数据集）.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='png',
            facecolor='white', edgecolor='none')
print(f"图表已保存为: {output_filename}")

plt.show()