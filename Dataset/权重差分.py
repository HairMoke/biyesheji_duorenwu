import matplotlib.pyplot as plt
import numpy as np

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模拟学到的权重
time = np.arange(32)
w_learned = np.random.randn(32) * 0.1 + 0.1  # 随机小权重

# 各阶段差分变换
w_early = 0.3 * w_learned[0:8]      # 基线期：抑制
w_n200 = np.diff(w_learned[8:16])   # N200期：一阶差分
w_p300 = w_learned[16:24] - w_learned[14:22]  # P300期：二阶差分
w_late = 0.5 * np.diff(w_learned[24:32])  # 晚期：衰减差分

# 计算最终融合权重
w_fused = np.zeros_like(w_learned)
w_fused[:8] = w_early
w_fused[8:15] = w_n200
w_fused[16:24] = w_p300
w_fused[24:31] = w_late

# 绘制对比
plt.figure(figsize=(15, 10))

plt.subplot(2,3,1)
plt.plot(time, w_learned, 'b-', label='原始权重')
plt.title("学到的权重（平坦无特异性）")
plt.legend()

plt.subplot(2,3,2)
plt.plot(time[:8], w_early, 'r-', label='基线增强')
plt.title("基线期：噪声抑制")
plt.legend()

plt.subplot(2,3,3)
plt.plot(time[9:16], w_n200, 'g-', label='N200增强')      # 修复：时间轴调整
plt.title("N200期：一阶差分")
plt.legend()

plt.subplot(2,3,4)
plt.plot(time[16:24], w_p300, 'm-', label='P300增强')
plt.title("P300期：二阶差分")
plt.legend()

plt.subplot(2,3,5)
plt.plot(time[25:32], w_late, 'k-', label='晚期衰减')     # 修复：时间轴调整
plt.title("晚期：能量耗散")
plt.legend()

plt.subplot(2,3,6)
plt.plot(time, w_fused, 'purple', label='融合权重')
plt.title("最终融合权重")
plt.legend()

plt.tight_layout()  # 添加布局调整
plt.savefig('权重差分.png', dpi=300, bbox_inches='tight')
plt.show()
