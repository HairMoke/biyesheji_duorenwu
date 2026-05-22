import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体（根据你的系统调整）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Songti SC']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 16

# 1. 准备数据
k_values = [3, 6, 9, 12]
auc_values = [0.9123, 0.9267, 0.9215, 0.9187]

# 2. 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 3. 绘制折线图
line = ax.plot(k_values, auc_values, marker='o', linewidth=2.5, markersize=8,
               color='#2E86AB', markerfacecolor='white', markeredgecolor='#2E86AB',
               markeredgewidth=2, zorder=5)

# 4. 标注数据点
for i, (k, auc) in enumerate(zip(k_values, auc_values)):
    # 调整标注位置，避免重叠
    y_offset = 10 if i != 1 else -25  # 对于k=6的标注放在下方
    va = 'bottom' if i != 1 else 'top'  # 垂直对齐方式

    ax.annotate(f'{auc:.4f}',
                xy=(k, auc),
                xytext=(0, y_offset),
                textcoords='offset points',
                ha='center', va=va,
                fontsize=24, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='gray'))

# 5. 设置图表属性
ax.set_xlabel('K值', fontsize=24, fontweight='bold', labelpad=10)
ax.set_ylabel('AUC值', fontsize=24, fontweight='bold', labelpad=10)
ax.set_title('KNN图构造中K值选择的敏感性分析', fontsize=24, pad=20, fontweight='bold')

# 6. 设置坐标轴
ax.set_xticks(k_values)
ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=20)
ax.set_yticks(np.arange(0.910, 0.930, 0.005))
ax.set_ylim(0.910, 0.930)

# 7. 添加网格线
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# 8. 美化边框
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('gray')

# 9. 高亮最优值（k=6）
optimal_k_index = 1  # k=6是最优值
ax.scatter(k_values[optimal_k_index], auc_values[optimal_k_index],
           color='red', s=150, zorder=10, edgecolor='black', linewidth=2)

# 添加最优值箭头标注
# ax.annotate('最优点 (k=6)',
#             xy=(k_values[optimal_k_index], auc_values[optimal_k_index]),
#             xytext=(k_values[optimal_k_index] + 0.8, auc_values[optimal_k_index] - 0.0025),
#             fontsize=20, fontweight='bold', color='red',
#             arrowprops=dict(arrowstyle='->', color='red', lw=1.5, connectionstyle="arc3,rad=0.2"))
ax.annotate('最优点 (k=6)',
            xy=(k_values[optimal_k_index], auc_values[optimal_k_index]),
            xytext=(k_values[optimal_k_index] + 0.8, auc_values[optimal_k_index] + 0.00),
            fontsize=20, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5, connectionstyle="arc3,rad=0.2"))



#
# # 10. 添加分析说明（根据你的文本）
# analysis_text = (
#     "当 k=3 时，连接过于稀疏，消息传递不充分\n"
#     "当 k=12 时，过度连接引入远距离噪声\n"
#     "k=6 时达到最优，邻域覆盖同一大脑功能区"
# )
# ax.text(0.05, 0.02, analysis_text, transform=ax.transAxes, fontsize=10,
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
#         verticalalignment='bottom', horizontalalignment='left')

# 11. 调整布局并保存
plt.tight_layout()

output_filename = 'KNN图构造中K值选择的敏感性分析.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='png',
            facecolor='white', edgecolor='none')
print(f"图表已保存为: {output_filename}")

# 显示图表
plt.show()

# 打印文件信息
import os

file_size = os.path.getsize(output_filename) / 1024  # KB
print(f"文件大小: {file_size:.2f} KB")
print(f"分辨率: 300 DPI")