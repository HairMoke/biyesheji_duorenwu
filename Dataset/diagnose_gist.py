import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
from scipy import signal

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# # 定义处理好的数据路径
# data_path = r"D:\ZHB_BiYeSheJi\Dataset\GIST\S01"
#
#
# def print_data_info():
#     """打印处理后的数据信息"""
#     print("=" * 80)
#     print("GIST数据集处理结果 - 第一个被试 (S01)")
#     print("=" * 80)
#
#     # 加载数据
#     try:
#         x_train = np.load(os.path.join(data_path, 'x_train.npy'))
#         x_test = np.load(os.path.join(data_path, 'x_test.npy'))
#         y_train = np.load(os.path.join(data_path, 'y_train.npy'))
#         y_test = np.load(os.path.join(data_path, 'y_test.npy'))
#
#         # 打印基本信息
#         print("\n1. 数据维度信息:")
#         print(f"   x_train shape: {x_train.shape}")  # [样本数, 通道数, 时间点]
#         print(f"   x_test shape: {x_test.shape}")
#         print(f"   y_train shape: {y_train.shape}")
#         print(f"   y_test shape: {y_test.shape}")
#
#         print("\n2. 样本数量统计:")
#         print(f"   训练集总样本数: {len(x_train)}")
#         print(f"   测试集总样本数: {len(x_test)}")
#         print(f"   训练集中目标样本数: {np.sum(y_train == 1)}")
#         print(f"   训练集中非目标样本数: {np.sum(y_train == 0)}")
#         print(f"   测试集中目标样本数: {np.sum(y_test == 1)}")
#         print(f"   测试集中非目标样本数: {np.sum(y_test == 0)}")
#
#         print("\n3. 数据统计信息 (x_train):")
#         print(f"   最小值: {np.min(x_train):.4f}")
#         print(f"   最大值: {np.max(x_train):.4f}")
#         print(f"   平均值: {np.mean(x_train):.4f}")
#         print(f"   标准差: {np.std(x_train):.4f}")
#
#         print("\n4. 标签统计 (y_train):")
#         print(f"   标签值: {np.unique(y_train)}")
#         print(f"   标签分布: {np.bincount(y_train.astype(int))}")
#
#         print("\n5. 详细数据查看 (前5个样本):")
#         print("\n   x_train 前5个样本的第一个通道数据 (前10个时间点):")
#         for i in range(min(5, len(x_train))):
#             print(f"   样本{i + 1} (标签={y_train[i]}): {x_train[i, 0, :10]}")
#
#         print("\n   y_train 前10个标签:")
#         print(f"   {y_train[:10]}")
#
#         print("\n   x_test 前5个样本的第一个通道数据 (前10个时间点):")
#         for i in range(min(5, len(x_test))):
#             print(f"   样本{i + 1} (标签={y_test[i]}): {x_test[i, 0, :10]}")
#
#         print("\n   y_test 前10个标签:")
#         print(f"   {y_test[:10]}")
#
#         return x_train, x_test, y_train, y_test
#
#     except Exception as e:
#         print(f"加载数据时出错: {e}")
#         return None, None, None, None
#
#
# def plot_erp(x_train, y_train):
#     """绘制目标和非目标的平均ERP"""
#     if x_train is None or y_train is None:
#         print("无法绘制ERP，数据加载失败")
#         return
#
#     print("\n" + "=" * 80)
#     print("开始绘制ERP图像...")
#     print("=" * 80)
#
#     # 分离目标和刺激
#     target_idx = np.where(y_train == 1)[0]
#     nontarget_idx = np.where(y_train == 0)[0]
#
#     print(f"目标试次数: {len(target_idx)}")
#     print(f"非目标试次数: {len(nontarget_idx)}")
#
#     if len(target_idx) == 0 or len(nontarget_idx) == 0:
#         print("错误: 目标或非目标试次数为0")
#         return
#
#     # 计算平均ERP
#     # 注意: 我们只使用前32个真实的通道 (GIST数据集是32通道)
#     target_erp = np.mean(x_train[target_idx, :32, :], axis=0)  # [32, 256]
#     nontarget_erp = np.mean(x_train[nontarget_idx, :32, :], axis=0)
#
#     print(f"目标ERP形状: {target_erp.shape}")
#     print(f"非目标ERP形状: {nontarget_erp.shape}")
#
#     # 选择要绘制的电极 (根据GIST数据集的电极位置)
#     # GIST数据集使用32个电极，我们选择Fz, Cz, Pz对应的电极
#     # 由于GIST数据集的电极顺序可能不同，我们使用常见的电极位置
#     # 注意: 这里可能需要根据实际的电极映射调整
#
#     # 假设电极顺序是标准的10-20系统的32导联
#     # 常见索引: Fz=4, Cz=12, Pz=20 (在32导联中)
#     # 但为了更准确，我们可以绘制多个电极
#
#     electrodes_to_plot = [4, 12, 20]  # Fz, Cz, Pz
#     electrode_names = ['Fz', 'Cz', 'Pz']
#
#     # 创建时间轴 (0-1000ms, 256Hz采样率)
#     time_ms = np.linspace(0, 1000, target_erp.shape[1])
#
#     # 创建图形
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#     fig.suptitle('GIST数据集 - 第一个被试 (S01) ERP分析', fontsize=16, fontweight='bold')
#
#     # 1. 绘制Fz, Cz, Pz电极的ERP
#     ax = axes[0, 0]
#     for i, elec_idx in enumerate(electrodes_to_plot):
#         if elec_idx < target_erp.shape[0]:
#             ax.plot(time_ms, target_erp[elec_idx, :], label=f'目标-{electrode_names[i]}',
#                     linewidth=2, alpha=0.8)
#             ax.plot(time_ms, nontarget_erp[elec_idx, :], label=f'非目标-{electrode_names[i]}',
#                     linewidth=2, alpha=0.8, linestyle='--')
#
#     ax.set_xlabel('时间 (ms)')
#     ax.set_ylabel('幅值 (μV)')
#     ax.set_title('Fz, Cz, Pz电极的ERP')
#     ax.legend(loc='upper right')
#     ax.grid(True, alpha=0.3)
#     ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#     ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
#
#     # 2. 绘制所有电极平均的ERP
#     ax = axes[0, 1]
#     # 计算所有电极的平均
#     avg_target_erp = np.mean(target_erp, axis=0)
#     avg_nontarget_erp = np.mean(nontarget_erp, axis=0)
#
#     ax.plot(time_ms, avg_target_erp, label='目标刺激',
#             color='red', linewidth=3, alpha=0.9)
#     ax.plot(time_ms, avg_nontarget_erp, label='非目标刺激',
#             color='blue', linewidth=3, alpha=0.9, linestyle='--')
#
#     ax.set_xlabel('时间 (ms)')
#     ax.set_ylabel('幅值 (μV)')
#     ax.set_title('所有电极平均ERP')
#     ax.legend(loc='upper right')
#     ax.grid(True, alpha=0.3)
#     ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#     ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
#
#     # 添加P300标记 (通常在300ms左右)
#     p300_time = 300
#     ax.axvline(x=p300_time, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='P300 (300ms)')
#     ax.legend(loc='upper right')
#
#     # 3. 绘制地形图差异 (目标-非目标) 在P300时刻
#     ax = axes[1, 0]
#
#     # 找到P300附近的时间点 (250-350ms)
#     p300_start = int(250 / 1000 * 256)  # 250ms
#     p300_end = int(350 / 1000 * 256)  # 350ms
#
#     # 计算P300时间窗内的平均差异
#     diff_erp = target_erp - nontarget_erp
#     p300_diff = np.mean(diff_erp[:, p300_start:p300_end], axis=1)
#
#     # 由于我们没有电极位置信息，简单绘制条形图显示各电极的P300差异
#     electrodes = np.arange(min(32, len(p300_diff)))
#     ax.bar(electrodes, p300_diff[:len(electrodes)], alpha=0.7)
#     ax.set_xlabel('电极索引')
#     ax.set_ylabel('目标-非目标差异 (μV)')
#     ax.set_title('P300时间窗 (250-350ms) 各电极差异')
#     ax.grid(True, alpha=0.3, axis='y')
#     ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#
#     # 标记Fz, Cz, Pz电极
#     for elec_idx in electrodes_to_plot:
#         if elec_idx < len(p300_diff):
#             ax.bar(elec_idx, p300_diff[elec_idx], color='red', alpha=0.9,
#                    label='关键电极' if elec_idx == electrodes_to_plot[0] else "")
#
#     if electrodes_to_plot[0] < len(p300_diff):
#         ax.legend(['Fz/Cz/Pz电极'])
#
#     # 4. 绘制单个试次的示例
#     ax = axes[1, 1]
#
#     # 选择一个目标试次和一个非目标试次
#     target_sample_idx = target_idx[0] if len(target_idx) > 0 else 0
#     nontarget_sample_idx = nontarget_idx[0] if len(nontarget_idx) > 0 else 0
#
#     # 选择Cz电极 (索引12)
#     cz_idx = 12 if 12 < x_train.shape[1] else 0
#
#     ax.plot(time_ms, x_train[target_sample_idx, cz_idx, :],
#             label=f'单个目标试次 (Cz)', color='red', linewidth=2, alpha=0.7)
#     ax.plot(time_ms, x_train[nontarget_sample_idx, cz_idx, :],
#             label=f'单个非目标试次 (Cz)', color='blue', linewidth=2, alpha=0.7, linestyle='--')
#
#     ax.set_xlabel('时间 (ms)')
#     ax.set_ylabel('幅值 (μV)')
#     ax.set_title('单个试次示例 (Cz电极)')
#     ax.legend(loc='upper right')
#     ax.grid(True, alpha=0.3)
#     ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#     ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
#
#     plt.tight_layout()
#     plt.show()
#
#     # 打印ERP统计信息
#     print("\nERP统计信息:")
#     print(f"目标ERP平均值: {np.mean(target_erp):.4f} μV")
#     print(f"非目标ERP平均值: {np.mean(nontarget_erp):.4f} μV")
#     print(f"目标ERP标准差: {np.std(target_erp):.4f} μV")
#     print(f"非目标ERP标准差: {np.std(nontarget_erp):.4f} μV")
#
#     # 计算P300幅值 (250-350ms时间窗内的平均差异)
#     p300_amplitude_target = np.mean(target_erp[:, p300_start:p300_end])
#     p300_amplitude_nontarget = np.mean(nontarget_erp[:, p300_start:p300_end])
#     p300_amplitude_diff = p300_amplitude_target - p300_amplitude_nontarget
#
#     print(f"\nP300幅值 (250-350ms):")
#     print(f"  目标刺激: {p300_amplitude_target:.4f} μV")
#     print(f"  非目标刺激: {p300_amplitude_nontarget:.4f} μV")
#     print(f"  差异 (目标-非目标): {p300_amplitude_diff:.4f} μV")
#
#     # 绘制目标和非目标刺激的对比图 (类似论文中的图)
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#
#     # 计算所有电极平均的ERP
#     ax2.plot(time_ms, avg_target_erp, label='目标刺激', color='red', linewidth=3)
#     ax2.plot(time_ms, avg_nontarget_erp, label='非目标刺激', color='blue', linewidth=3, linestyle='--')
#
#     # 添加P300和N200标记
#     ax2.axvline(x=200, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='N200 (~200ms)')
#     ax2.axvline(x=300, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='P300 (~300ms)')
#
#     ax2.set_xlabel('时间 (ms)', fontsize=12)
#     ax2.set_ylabel('幅值 (μV)', fontsize=12)
#     ax2.set_title('GIST数据集 - 目标与非目标刺激ERP对比', fontsize=14, fontweight='bold')
#     ax2.legend(loc='upper right', fontsize=11)
#     ax2.grid(True, alpha=0.3)
#     ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#     ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
#
#     # 设置坐标轴范围
#     y_min = min(np.min(avg_target_erp), np.min(avg_nontarget_erp)) * 1.1
#     y_max = max(np.max(avg_target_erp), np.max(avg_nontarget_erp)) * 1.1
#     ax2.set_ylim(y_min, y_max)
#
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_data_distribution(y_train, y_test):
#     """绘制数据分布图"""
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
#     # 训练集分布
#     train_counts = np.bincount(y_train.astype(int))
#     train_labels = ['非目标', '目标']
#     axes[0].bar(train_labels, train_counts, color=['blue', 'red'], alpha=0.7)
#     axes[0].set_title('训练集样本分布')
#     axes[0].set_ylabel('样本数量')
#     for i, v in enumerate(train_counts):
#         axes[0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
#
#     # 测试集分布
#     test_counts = np.bincount(y_test.astype(int))
#     axes[1].bar(train_labels, test_counts, color=['blue', 'red'], alpha=0.7)
#     axes[1].set_title('测试集样本分布')
#     axes[1].set_ylabel('样本数量')
#     for i, v in enumerate(test_counts):
#         axes[1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
#
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_sample_waveforms(x_train, y_train, n_samples=5):
#     """绘制几个样本的波形图"""
#     fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
#
#     # 如果没有足够的样本，调整
#     if n_samples > len(x_train):
#         n_samples = len(x_train)
#
#     # 如果是单个子图，转换为列表
#     if n_samples == 1:
#         axes = [axes]
#
#     for i in range(n_samples):
#         # 选择Cz电极 (索引12)
#         cz_idx = 12 if 12 < x_train.shape[1] else 0
#
#         time_ms = np.linspace(0, 1000, x_train.shape[2])
#         label_type = "目标" if y_train[i] == 1 else "非目标"
#         color = 'red' if y_train[i] == 1 else 'blue'
#
#         axes[i].plot(time_ms, x_train[i, cz_idx, :], color=color, linewidth=1.5)
#         axes[i].set_ylabel(f'样本{i + 1}\n({label_type})', fontsize=10)
#         axes[i].grid(True, alpha=0.3)
#         axes[i].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#
#         # 只在最后一个子图显示x轴标签
#         if i == n_samples - 1:
#             axes[i].set_xlabel('时间 (ms)')
#         else:
#             axes[i].set_xticklabels([])
#
#     plt.suptitle(f'前{n_samples}个样本的Cz电极波形', fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.show()
#
#
# # 主程序
# if __name__ == "__main__":
#     print("开始分析GIST数据集处理结果...")
#
#     # 打印数据信息
#     x_train, x_test, y_train, y_test = print_data_info()
#
#     if x_train is not None:
#         # 绘制数据分布
#         plot_data_distribution(y_train, y_test)
#
#         # 绘制样本波形
#         plot_sample_waveforms(x_train, y_train, n_samples=5)
#
#         # 绘制ERP
#         plot_erp(x_train, y_train)
#
#         print("\n" + "=" * 80)
#         print("分析完成！")
#         print("=" * 80)
#
#         # 额外的数据质量检查
#         print("\n数据质量检查:")
#
#         # 检查NaN值
#         nan_count_train = np.isnan(x_train).sum()
#         nan_count_test = np.isnan(x_test).sum()
#         print(f"训练集中NaN值数量: {nan_count_train}")
#         print(f"测试集中NaN值数量: {nan_count_test}")
#
#         # 检查无限值
#         inf_count_train = np.isinf(x_train).sum()
#         inf_count_test = np.isinf(x_test).sum()
#         print(f"训练集中Inf值数量: {inf_count_train}")
#         print(f"测试集中Inf值数量: {inf_count_test}")
#
#         # 检查数据范围
#         print(f"\n数据范围检查:")
#         print(f"训练集范围: [{np.min(x_train):.4f}, {np.max(x_train):.4f}]")
#         print(f"测试集范围: [{np.min(x_test):.4f}, {np.max(x_test):.4f}]")
#
#         # 检查标签平衡
#         print(f"\n标签平衡检查:")
#         train_target_ratio = np.sum(y_train == 1) / len(y_train)
#         test_target_ratio = np.sum(y_test == 1) / len(y_test)
#         print(f"训练集目标比例: {train_target_ratio:.2%}")
#         print(f"测试集目标比例: {test_target_ratio:.2%}")
#     else:
#         print("数据加载失败，请检查路径和数据文件。")
#
#
#


"""
GIST数据集处理结果校验与分析
用于验证数据处理是否正确，并检查ERP特征
"""
#
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from scipy import signal
# from scipy.stats import ttest_ind
# import seaborn as sns
#
# # 设置matplotlib中文字体（如果需要）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 数据路径
# DATA_PATH = r"D:\ZHB_BiYeSheJi\Dataset\GIST\S01"
#
#
# def load_and_print_data_info():
#     """加载数据并打印详细信息"""
#     print("=" * 80)
#     print("GIST数据集处理结果校验 - 第一个被试 (S01)")
#     print("=" * 80)
#
#     try:
#         # 加载数据
#         x_train = np.load(os.path.join(DATA_PATH, 'x_train.npy'))
#         x_test = np.load(os.path.join(DATA_PATH, 'x_test.npy'))
#         y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
#         y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))
#
#         # 1. 打印基本信息
#         print("\n1. 数据维度信息:")
#         print(f"   x_train shape: {x_train.shape}")  # [样本数, 通道数, 时间点]
#         print(f"   x_test shape: {x_test.shape}")
#         print(f"   y_train shape: {y_train.shape}")
#         print(f"   y_test shape: {y_test.shape}")
#
#         # 2. 详细数据查看
#         print("\n2. 详细数据值 (前5个样本):")
#         print("\n   x_train (前5个样本，第一个通道，前10个时间点):")
#         for i in range(min(5, len(x_train))):
#             label = "目标" if y_train[i] == 1 else "非目标"
#             print(f"   样本{i + 1} [{label}]: {x_train[i, 0, :10]}")
#
#         print("\n   y_train (前10个标签):")
#         print(f"   {y_train[:10]}")
#         print(f"   (1=目标, 0=非目标)")
#
#         print("\n   x_test (前5个样本，第一个通道，前10个时间点):")
#         for i in range(min(5, len(x_test))):
#             label = "目标" if y_test[i] == 1 else "非目标"
#             print(f"   样本{i + 1} [{label}]: {x_test[i, 0, :10]}")
#
#         print("\n   y_test (前10个标签):")
#         print(f"   {y_test[:10]}")
#         print(f"   (1=目标, 0=非目标)")
#
#         # 3. 统计信息
#         print("\n3. 数据统计信息:")
#         print("\n   x_train:")
#         print(f"     最小值: {np.min(x_train):.6f}")
#         print(f"     最大值: {np.max(x_train):.6f}")
#         print(f"     平均值: {np.mean(x_train):.6f}")
#         print(f"     标准差: {np.std(x_train):.6f}")
#         print(f"     NaN值数量: {np.isnan(x_train).sum()}")
#         print(f"     Inf值数量: {np.isinf(x_train).sum()}")
#
#         print("\n   x_test:")
#         print(f"     最小值: {np.min(x_test):.6f}")
#         print(f"     最大值: {np.max(x_test):.6f}")
#         print(f"     平均值: {np.mean(x_test):.6f}")
#         print(f"     标准差: {np.std(x_test):.6f}")
#         print(f"     NaN值数量: {np.isnan(x_test).sum()}")
#         print(f"     Inf值数量: {np.isinf(x_test).sum()}")
#
#         # 4. 样本分布统计
#         print("\n4. 样本分布统计:")
#         print(f"\n   训练集:")
#         print(f"     总样本数: {len(x_train)}")
#         print(f"     目标样本数: {np.sum(y_train == 1)} ({np.sum(y_train == 1) / len(y_train) * 100:.1f}%)")
#         print(f"     非目标样本数: {np.sum(y_train == 0)} ({np.sum(y_train == 0) / len(y_train) * 100:.1f}%)")
#
#         print(f"\n   测试集:")
#         print(f"     总样本数: {len(x_test)}")
#         print(f"     目标样本数: {np.sum(y_test == 1)} ({np.sum(y_test == 1) / len(y_test) * 100:.1f}%)")
#         print(f"     非目标样本数: {np.sum(y_test == 0)} ({np.sum(y_test == 0) / len(y_test) * 100:.1f}%)")
#
#         # 5. 数据质量检查
#         print("\n5. 数据质量检查:")
#
#         # 检查是否有异常值
#         threshold = 100  # 假设超过100μV为异常
#         abnormal_train = np.sum(np.abs(x_train) > threshold)
#         abnormal_test = np.sum(np.abs(x_test) > threshold)
#         print(f"   训练集中幅值超过{threshold}μV的异常值数量: {abnormal_train}")
#         print(f"   测试集中幅值超过{threshold}μV的异常值数量: {abnormal_test}")
#
#         # 检查通道数据
#         print(f"\n   通道信息:")
#         print(f"     总通道数: {x_train.shape[1]}")
#         print(f"     实际通道数 (GIST): 32")
#         print(f"     补充通道数: {x_train.shape[1] - 32}")
#
#         # 检查补充通道是否全为0
#         if x_train.shape[1] > 32:
#             sup_channels_train = np.sum(np.abs(x_train[:, 32:, :]) > 1e-10)
#             sup_channels_test = np.sum(np.abs(x_test[:, 32:, :]) > 1e-10)
#             print(f"     训练集补充通道非零值数量: {sup_channels_train}")
#             print(f"     测试集补充通道非零值数量: {sup_channels_test}")
#
#         return x_train, x_test, y_train, y_test
#
#     except Exception as e:
#         print(f"加载数据时出错: {e}")
#         return None, None, None, None
#
#
# def plot_data_distribution(y_train, y_test):
#     """绘制数据分布图"""
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
#     # 训练集分布
#     train_counts = np.bincount(y_train.astype(int))
#     train_labels = ['非目标', '目标']
#     colors = ['blue', 'red']
#
#     axes[0].bar(train_labels, train_counts, color=colors, alpha=0.7)
#     axes[0].set_title('训练集样本分布', fontsize=14, fontweight='bold')
#     axes[0].set_ylabel('样本数量', fontsize=12)
#     axes[0].set_ylim(0, max(train_counts) * 1.1)
#
#     for i, v in enumerate(train_counts):
#         axes[0].text(i, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=12)
#
#     # 测试集分布
#     test_counts = np.bincount(y_test.astype(int))
#     axes[1].bar(train_labels, test_counts, color=colors, alpha=0.7)
#     axes[1].set_title('测试集样本分布', fontsize=14, fontweight='bold')
#     axes[1].set_ylabel('样本数量', fontsize=12)
#     axes[1].set_ylim(0, max(test_counts) * 1.1)
#
#     for i, v in enumerate(test_counts):
#         axes[1].text(i, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=12)
#
#     plt.suptitle('GIST数据集样本分布', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.show()
#
#     # 打印比例信息
#     print("\n样本分布比例:")
#     print(f"  训练集 - 目标:非目标 = {train_counts[1]}:{train_counts[0]} = 1:{train_counts[0] / train_counts[1]:.1f}")
#     print(f"  测试集 - 目标:非目标 = {test_counts[1]}:{test_counts[0]} = 1:{test_counts[0] / test_counts[1]:.1f}")
#
#
# def compute_and_plot_erp(x_data, y_data, title_prefix="训练集", use_channels=32):
#     """计算并绘制ERP"""
#     # 分离目标和非目标
#     target_idx = np.where(y_data == 1)[0]
#     nontarget_idx = np.where(y_data == 0)[0]
#
#     if len(target_idx) == 0 or len(nontarget_idx) == 0:
#         print(f"  {title_prefix}: 没有足够的目标或非目标试次")
#         return None, None
#
#     # 只使用实际的EEG通道（GIST是32通道）
#     target_data = x_data[target_idx, :use_channels, :]
#     nontarget_data = x_data[nontarget_idx, :use_channels, :]
#
#     # 计算平均ERP
#     target_erp = np.mean(target_data, axis=0)  # [通道, 时间点]
#     nontarget_erp = np.mean(nontarget_data, axis=0)
#
#     # 计算所有通道平均的ERP
#     avg_target_erp = np.mean(target_erp, axis=0)
#     avg_nontarget_erp = np.mean(nontarget_erp, axis=0)
#
#     # 创建时间轴 (0-1000ms, 256Hz采样率)
#     time_ms = np.linspace(0, 1000, target_erp.shape[1])
#
#     print(f"\n{title_prefix} ERP统计:")
#     print(f"  目标试次数: {len(target_idx)}")
#     print(f"  非目标试次数: {len(nontarget_idx)}")
#     print(f"  目标ERP平均值: {np.mean(target_erp):.4f} μV")
#     print(f"  非目标ERP平均值: {np.mean(nontarget_erp):.4f} μV")
#
#     return target_erp, nontarget_erp, avg_target_erp, avg_nontarget_erp, time_ms
#
#
# def plot_erp_comparison(x_train, y_train, x_test, y_test):
#     """绘制训练集和测试集的ERP对比"""
#     # 计算训练集ERP
#     train_target_erp, train_nontarget_erp, train_avg_target, train_avg_nontarget, time_ms = compute_and_plot_erp(
#         x_train, y_train, "训练集"
#     )
#
#     # 计算测试集ERP
#     test_target_erp, test_nontarget_erp, test_avg_target, test_avg_nontarget, _ = compute_and_plot_erp(
#         x_test, y_test, "测试集"
#     )
#
#     if train_target_erp is None or test_target_erp is None:
#         print("无法绘制ERP图，数据不足")
#         return
#
#     # 创建图形
#     fig, axes = plt.subplots(2, 3, figsize=(18, 10))
#     fig.suptitle('GIST数据集ERP分析 - 第一个被试 (S01)', fontsize=16, fontweight='bold')
#
#     # 1. 训练集所有通道平均ERP
#     ax = axes[0, 0]
#     ax.plot(time_ms, train_avg_target, label='目标刺激', color='red', linewidth=2.5)
#     ax.plot(time_ms, train_avg_nontarget, label='非目标刺激', color='blue', linewidth=2.5, linestyle='--')
#
#     # 标记关键时间点
#     ax.axvline(x=300, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='P300 (~300ms)')
#     ax.axvline(x=200, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='N200 (~200ms)')
#
#     ax.set_xlabel('时间 (ms)', fontsize=11)
#     ax.set_ylabel('幅值 (μV)', fontsize=11)
#     ax.set_title('训练集 - 所有通道平均ERP', fontsize=13, fontweight='bold')
#     ax.legend(loc='upper right', fontsize=10)
#     ax.grid(True, alpha=0.3)
#     ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#     ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
#
#     # 2. 测试集所有通道平均ERP
#     ax = axes[0, 1]
#     ax.plot(time_ms, test_avg_target, label='目标刺激', color='red', linewidth=2.5)
#     ax.plot(time_ms, test_avg_nontarget, label='非目标刺激', color='blue', linewidth=2.5, linestyle='--')
#
#     # 标记关键时间点
#     ax.axvline(x=300, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='P300 (~300ms)')
#     ax.axvline(x=200, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='N200 (~200ms)')
#
#     ax.set_xlabel('时间 (ms)', fontsize=11)
#     ax.set_ylabel('幅值 (μV)', fontsize=11)
#     ax.set_title('测试集 - 所有通道平均ERP', fontsize=13, fontweight='bold')
#     ax.legend(loc='upper right', fontsize=10)
#     ax.grid(True, alpha=0.3)
#     ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#     ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
#
#     # 3. 训练集关键电极ERP (Fz, Cz, Pz)
#     ax = axes[0, 2]
#     # 假设的电极索引 (需要根据实际的电极映射调整)
#     # GIST数据集是32导联，我们假设标准顺序
#     electrodes = {'Fz': 4, 'Cz': 12, 'Pz': 20}
#
#     for name, idx in electrodes.items():
#         if idx < 32:  # 确保索引在范围内
#             ax.plot(time_ms, train_target_erp[idx, :], label=f'目标-{name}', linewidth=1.5, alpha=0.8)
#             ax.plot(time_ms, train_nontarget_erp[idx, :], label=f'非目标-{name}', linewidth=1.5, alpha=0.8,
#                     linestyle='--')
#
#     ax.set_xlabel('时间 (ms)', fontsize=11)
#     ax.set_ylabel('幅值 (μV)', fontsize=11)
#     ax.set_title('训练集 - 关键电极ERP', fontsize=13, fontweight='bold')
#     ax.legend(loc='upper right', fontsize=9, ncol=2)
#     ax.grid(True, alpha=0.3)
#     ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#
#     # 4. P300幅值对比 (250-350ms时间窗)
#     ax = axes[1, 0]
#
#     # 计算P300时间窗内的平均幅值
#     p300_start = int(250 / 1000 * 256)  # 250ms
#     p300_end = int(350 / 1000 * 256)  # 350ms
#
#     train_p300_target = np.mean(train_avg_target[p300_start:p300_end])
#     train_p300_nontarget = np.mean(train_avg_nontarget[p300_start:p300_end])
#     test_p300_target = np.mean(test_avg_target[p300_start:p300_end])
#     test_p300_nontarget = np.mean(test_avg_nontarget[p300_start:p300_end])
#
#     categories = ['训练集-目标', '训练集-非目标', '测试集-目标', '测试集-非目标']
#     p300_values = [train_p300_target, train_p300_nontarget, test_p300_target, test_p300_nontarget]
#     colors = ['red', 'lightcoral', 'darkred', 'pink']
#
#     bars = ax.bar(categories, p300_values, color=colors, alpha=0.7)
#     ax.set_ylabel('P300幅值 (μV)', fontsize=11)
#     ax.set_title('P300幅值对比 (250-350ms)', fontsize=13, fontweight='bold')
#     ax.grid(True, alpha=0.3, axis='y')
#
#     # 添加数值标签
#     for bar, value in zip(bars, p300_values):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1 * np.sign(height),
#                 f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
#
#     # 5. 目标-非目标差异地形图 (在P300时刻)
#     ax = axes[1, 1]
#
#     # 计算P300时间窗内的差异
#     train_diff = np.mean(train_target_erp[:, p300_start:p300_end] - train_nontarget_erp[:, p300_start:p300_end], axis=1)
#     test_diff = np.mean(test_target_erp[:, p300_start:p300_end] - test_nontarget_erp[:, p300_start:p300_end], axis=1)
#
#     x = np.arange(len(train_diff))
#     width = 0.35
#
#     ax.bar(x - width / 2, train_diff, width, label='训练集差异', alpha=0.7)
#     ax.bar(x + width / 2, test_diff, width, label='测试集差异', alpha=0.7)
#
#     ax.set_xlabel('电极索引', fontsize=11)
#     ax.set_ylabel('目标-非目标差异 (μV)', fontsize=11)
#     ax.set_title('P300时间窗各电极差异', fontsize=13, fontweight='bold')
#     ax.legend(loc='upper right', fontsize=10)
#     ax.grid(True, alpha=0.3, axis='y')
#     ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#
#     # 6. 统计显著性检验
#     ax = axes[1, 2]
#
#     # 选择Cz电极进行统计检验
#     cz_idx = 12 if 12 < 32 else 0
#
#     # 提取Cz电极在P300时间窗的数据
#     train_target_cz = x_train[y_train == 1, cz_idx, p300_start:p300_end].flatten()
#     train_nontarget_cz = x_train[y_train == 0, cz_idx, p300_start:p300_end].flatten()
#
#     test_target_cz = x_test[y_test == 1, cz_idx, p300_start:p300_end].flatten()
#     test_nontarget_cz = x_test[y_test == 0, cz_idx, p300_start:p300_end].flatten()
#
#     # 进行t检验
#     try:
#         t_stat_train, p_val_train = ttest_ind(train_target_cz, train_nontarget_cz, equal_var=False)
#         t_stat_test, p_val_test = ttest_ind(test_target_cz, test_nontarget_cz, equal_var=False)
#
#         results = {
#             '训练集': {'t统计量': t_stat_train, 'p值': p_val_train},
#             '测试集': {'t统计量': t_stat_test, 'p值': p_val_test}
#         }
#
#         # 创建文本显示
#         text_content = "Cz电极P300时间窗统计检验\n\n"
#         for dataset, vals in results.items():
#             text_content += f"{dataset}:\n"
#             text_content += f"  t统计量 = {vals['t统计量']:.3f}\n"
#             text_content += f"  p值 = {vals['p值']:.6f}\n"
#             text_content += f"  显著性: {'显著' if vals['p值'] < 0.05 else '不显著'}\n\n"
#
#         ax.text(0.1, 0.5, text_content, fontsize=11, verticalalignment='center')
#         ax.set_title('统计显著性检验', fontsize=13, fontweight='bold')
#         ax.axis('off')
#
#     except Exception as e:
#         ax.text(0.1, 0.5, f"统计检验失败:\n{str(e)}", fontsize=11, verticalalignment='center')
#         ax.set_title('统计显著性检验', fontsize=13, fontweight='bold')
#         ax.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#     # 打印P300统计信息
#     print("\nP300幅值统计 (250-350ms时间窗):")
#     print(f"  训练集 - 目标: {train_p300_target:.4f} μV, 非目标: {train_p300_nontarget:.4f} μV")
#     print(f"  测试集 - 目标: {test_p300_target:.4f} μV, 非目标: {test_p300_nontarget:.4f} μV")
#     print(f"  训练集差异: {train_p300_target - train_p300_nontarget:.4f} μV")
#     print(f"  测试集差异: {test_p300_target - test_p300_nontarget:.4f} μV")
#
#
# def plot_single_trials(x_train, y_train, n_trials=3):
#     """绘制单个试次的波形"""
#     # 找到目标和非目标试次的索引
#     target_idx = np.where(y_train == 1)[0]
#     nontarget_idx = np.where(y_train == 0)[0]
#
#     if len(target_idx) == 0 or len(nontarget_idx) == 0:
#         print("没有足够的目标或非目标试次")
#         return
#
#     # 选择几个试次
#     selected_targets = target_idx[:min(n_trials, len(target_idx))]
#     selected_nontargets = nontarget_idx[:min(n_trials, len(nontarget_idx))]
#
#     # 创建时间轴
#     time_ms = np.linspace(0, 1000, x_train.shape[2])
#
#     fig, axes = plt.subplots(n_trials, 2, figsize=(14, 3 * n_trials))
#     fig.suptitle(f'单个试次波形示例 - 前{n_trials}个目标和非目标试次', fontsize=16, fontweight='bold')
#
#     # 如果是单个子图，转换为2D数组
#     if n_trials == 1:
#         axes = axes.reshape(1, -1)
#
#     # 选择Cz电极 (索引12)
#     cz_idx = 12 if 12 < x_train.shape[1] else 0
#
#     for i in range(n_trials):
#         # 目标试次
#         if i < len(selected_targets):
#             idx = selected_targets[i]
#             axes[i, 0].plot(time_ms, x_train[idx, cz_idx, :], color='red', linewidth=1.5)
#             axes[i, 0].set_ylabel(f'目标试次 {i + 1}\n幅值 (μV)', fontsize=10)
#             axes[i, 0].grid(True, alpha=0.3)
#             axes[i, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#
#         # 非目标试次
#         if i < len(selected_nontargets):
#             idx = selected_nontargets[i]
#             axes[i, 1].plot(time_ms, x_train[idx, cz_idx, :], color='blue', linewidth=1.5)
#             axes[i, 1].set_ylabel(f'非目标试次 {i + 1}\n幅值 (μV)', fontsize=10)
#             axes[i, 1].grid(True, alpha=0.3)
#             axes[i, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#
#         # 只在最后一行显示x轴标签
#         if i == n_trials - 1:
#             axes[i, 0].set_xlabel('时间 (ms)', fontsize=11)
#             axes[i, 1].set_xlabel('时间 (ms)', fontsize=11)
#         else:
#             axes[i, 0].set_xticklabels([])
#             axes[i, 1].set_xticklabels([])
#
#     # 设置标题
#     axes[0, 0].set_title('目标试次 (Cz电极)', fontsize=12, fontweight='bold')
#     axes[0, 1].set_title('非目标试次 (Cz电极)', fontsize=12, fontweight='bold')
#
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_data_quality_heatmap(x_train, x_test):
#     """绘制数据质量热图"""
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#     fig.suptitle('数据质量检查 - 热图分析', fontsize=16, fontweight='bold')
#
#     # 1. 训练集通道平均幅值
#     ax = axes[0, 0]
#     channel_means_train = np.mean(np.mean(np.abs(x_train[:, :32, :]), axis=2), axis=0)
#     im1 = ax.imshow(channel_means_train.reshape(1, -1), aspect='auto', cmap='hot')
#     ax.set_title('训练集各通道平均绝对幅值', fontsize=13)
#     ax.set_xlabel('通道索引', fontsize=11)
#     ax.set_ylabel('幅值 (μV)', fontsize=11)
#     plt.colorbar(im1, ax=ax, orientation='vertical')
#
#     # 添加通道数值
#     for i, val in enumerate(channel_means_train):
#         ax.text(i, 0, f'{val:.2f}', ha='center', va='center',
#                 color='white' if val > np.max(channel_means_train) / 2 else 'black', fontsize=8)
#
#     # 2. 测试集通道平均幅值
#     ax = axes[0, 1]
#     channel_means_test = np.mean(np.mean(np.abs(x_test[:, :32, :]), axis=2), axis=0)
#     im2 = ax.imshow(channel_means_test.reshape(1, -1), aspect='auto', cmap='hot')
#     ax.set_title('测试集各通道平均绝对幅值', fontsize=13)
#     ax.set_xlabel('通道索引', fontsize=11)
#     ax.set_ylabel('幅值 (μV)', fontsize=11)
#     plt.colorbar(im2, ax=ax, orientation='vertical')
#
#     # 添加通道数值
#     for i, val in enumerate(channel_means_test):
#         ax.text(i, 0, f'{val:.2f}', ha='center', va='center',
#                 color='white' if val > np.max(channel_means_test) / 2 else 'black', fontsize=8)
#
#     # 3. 训练集时间点平均幅值
#     ax = axes[1, 0]
#     time_means_train = np.mean(np.mean(np.abs(x_train[:, :32, :]), axis=0), axis=0)
#     ax.plot(np.linspace(0, 1000, len(time_means_train)), time_means_train, color='red', linewidth=2)
#     ax.set_title('训练集时间点平均绝对幅值', fontsize=13)
#     ax.set_xlabel('时间 (ms)', fontsize=11)
#     ax.set_ylabel('幅值 (μV)', fontsize=11)
#     ax.grid(True, alpha=0.3)
#
#     # 标记P300时间窗
#     ax.axvspan(250, 350, alpha=0.2, color='green', label='P300时间窗')
#     ax.legend(loc='upper right', fontsize=10)
#
#     # 4. 测试集时间点平均幅值
#     ax = axes[1, 1]
#     time_means_test = np.mean(np.mean(np.abs(x_test[:, :32, :]), axis=0), axis=0)
#     ax.plot(np.linspace(0, 1000, len(time_means_test)), time_means_test, color='blue', linewidth=2)
#     ax.set_title('测试集时间点平均绝对幅值', fontsize=13)
#     ax.set_xlabel('时间 (ms)', fontsize=11)
#     ax.set_ylabel('幅值 (μV)', fontsize=11)
#     ax.grid(True, alpha=0.3)
#
#     # 标记P300时间窗
#     ax.axvspan(250, 350, alpha=0.2, color='green', label='P300时间窗')
#     ax.legend(loc='upper right', fontsize=10)
#
#     plt.tight_layout()
#     plt.show()
#
#     # 打印通道幅值统计
#     print("\n各通道平均绝对幅值统计:")
#     print(f"  训练集 - 最大值: {np.max(channel_means_train):.4f} μV (通道{np.argmax(channel_means_train)})")
#     print(f"          最小值: {np.min(channel_means_train):.4f} μV (通道{np.argmin(channel_means_train)})")
#     print(f"          平均值: {np.mean(channel_means_train):.4f} μV")
#     print(f"  测试集 - 最大值: {np.max(channel_means_test):.4f} μV (通道{np.argmax(channel_means_test)})")
#     print(f"          最小值: {np.min(channel_means_test):.4f} μV (通道{np.argmin(channel_means_test)})")
#     print(f"          平均值: {np.mean(channel_means_test):.4f} μV")
#
#
# def comprehensive_data_validation(x_train, x_test, y_train, y_test):
#     """综合数据验证"""
#     print("\n" + "=" * 80)
#     print("综合数据验证")
#     print("=" * 80)
#
#     # 验证1: 检查目标和非目标试次是否重叠
#     print("\n1. 数据划分验证:")
#
#     # 理论上，目标试次应该在训练集和测试集中不重叠
#     # 但由于我们处理的是特征数据，无法直接验证原始试次
#     # 我们可以检查比例是否合理
#     train_target_ratio = np.sum(y_train == 1) / len(y_train)
#     test_target_ratio = np.sum(y_test == 1) / len(y_test)
#
#     print(f"   训练集目标比例: {train_target_ratio:.2%}")
#     print(f"   测试集目标比例: {test_target_ratio:.2%}")
#
#     # 理想情况下，训练集应该平衡，测试集保持原始比例
#     if 0.45 <= train_target_ratio <= 0.55:
#         print("   ✓ 训练集基本平衡")
#     else:
#         print(f"   ⚠ 训练集可能不平衡: {train_target_ratio:.2%}")
#
#     if test_target_ratio < 0.2:  # 测试集中目标应该少于非目标
#         print(f"   ✓ 测试集目标比例合理")
#     else:
#         print(f"   ⚠ 测试集目标比例可能过高: {test_target_ratio:.2%}")
#
#     # 验证2: 检查数据范围
#     print("\n2. 数据范围验证:")
#
#     # EEG数据通常在-100到100μV之间
#     train_min, train_max = np.min(x_train[:, :32, :]), np.max(x_train[:, :32, :])
#     test_min, test_max = np.min(x_test[:, :32, :]), np.max(x_test[:, :32, :])
#
#     print(f"   训练集范围: [{train_min:.2f}, {train_max:.2f}] μV")
#     print(f"   测试集范围: [{test_min:.2f}, {test_max:.2f}] μV")
#
#     if -150 < train_min < -10 and 10 < train_max < 150:
#         print("   ✓ 训练集数据范围合理")
#     else:
#         print(f"   ⚠ 训练集数据范围可能异常")
#
#     if -150 < test_min < -10 and 10 < test_max < 150:
#         print("   ✓ 测试集数据范围合理")
#     else:
#         print(f"   ⚠ 测试集数据范围可能异常")
#
#     # 验证3: 检查P300特征
#     print("\n3. P300特征验证:")
#
#     # 计算P300时间窗的平均差异
#     p300_start = int(250 / 1000 * 256)
#     p300_end = int(350 / 1000 * 256)
#
#     # 训练集
#     train_target_p300 = np.mean(x_train[y_train == 1, :32, p300_start:p300_end])
#     train_nontarget_p300 = np.mean(x_train[y_train == 0, :32, p300_start:p300_end])
#     train_p300_diff = train_target_p300 - train_nontarget_p300
#
#     # 测试集
#     test_target_p300 = np.mean(x_test[y_test == 1, :32, p300_start:p300_end])
#     test_nontarget_p300 = np.mean(x_test[y_test == 0, :32, p300_start:p300_end])
#     test_p300_diff = test_target_p300 - test_nontarget_p300
#
#     print(f"   训练集P300差异: {train_p300_diff:.4f} μV")
#     print(f"   测试集P300差异: {test_p300_diff:.4f} μV")
#
#     # P300差异通常为正数，目标刺激的P300幅值更大
#     if train_p300_diff > 0:
#         print("   ✓ 训练集P300特征方向正确")
#     else:
#         print(f"   ⚠ 训练集P300特征方向可能异常")
#
#     if test_p300_diff > 0:
#         print("   ✓ 测试集P300特征方向正确")
#     else:
#         print(f"   ⚠ 测试集P300特征方向可能异常")
#
#     # 验证4: 检查数据一致性
#     print("\n4. 数据一致性验证:")
#
#     # 检查训练集和测试集的数据统计是否相似
#     train_mean = np.mean(x_train[:, :32, :])
#     train_std = np.std(x_train[:, :32, :])
#     test_mean = np.mean(x_test[:, :32, :])
#     test_std = np.std(x_test[:, :32, :])
#
#     print(f"   训练集均值: {train_mean:.4f}, 标准差: {train_std:.4f}")
#     print(f"   测试集均值: {test_mean:.4f}, 标准差: {test_std:.4f}")
#
#     mean_diff_ratio = abs(train_mean - test_mean) / (abs(train_mean) + 1e-10)
#     std_diff_ratio = abs(train_std - test_std) / (train_std + 1e-10)
#
#     if mean_diff_ratio < 0.1:
#         print("   ✓ 训练集和测试集均值一致")
#     else:
#         print(f"   ⚠ 训练集和测试集均值差异较大: {mean_diff_ratio:.2%}")
#
#     if std_diff_ratio < 0.1:
#         print("   ✓ 训练集和测试集标准差一致")
#     else:
#         print(f"   ⚠ 训练集和测试集标准差差异较大: {std_diff_ratio:.2%}")
#
#     return {
#         'train_target_ratio': train_target_ratio,
#         'test_target_ratio': test_target_ratio,
#         'train_p300_diff': train_p300_diff,
#         'test_p300_diff': test_p300_diff,
#         'data_range_ok': (-150 < train_min < -10 and 10 < train_max < 150) and (
#                     -150 < test_min < -10 and 10 < test_max < 150),
#         'p300_direction_ok': (train_p300_diff > 0) and (test_p300_diff > 0),
#         'data_consistency_ok': (mean_diff_ratio < 0.1) and (std_diff_ratio < 0.1)
#     }
#
#
# def main():
#     """主函数"""
#     print("开始GIST数据集处理结果校验...")
#
#     # 1. 加载数据并打印信息
#     x_train, x_test, y_train, y_test = load_and_print_data_info()
#
#     if x_train is None:
#         print("数据加载失败，请检查路径和数据文件。")
#         return
#
#     # 2. 绘制数据分布
#     plot_data_distribution(y_train, y_test)
#
#     # 3. 绘制单个试次波形
#     plot_single_trials(x_train, y_train, n_trials=3)
#
#     # 4. 绘制数据质量热图
#     plot_data_quality_heatmap(x_train, x_test)
#
#     # 5. 绘制ERP对比
#     plot_erp_comparison(x_train, y_train, x_test, y_test)
#
#     # 6. 综合数据验证
#     validation_results = comprehensive_data_validation(x_train, x_test, y_train, y_test)
#
#     # 7. 生成总结报告
#     print("\n" + "=" * 80)
#     print("数据校验总结报告")
#     print("=" * 80)
#
#     print("\n数据处理基本正确性:")
#
#     checks_passed = 0
#     total_checks = 0
#
#     # 检查1: 数据维度
#     total_checks += 1
#     if x_train.shape[1:] == (64, 256) and x_test.shape[1:] == (64, 256):
#         print("  ✓ 数据维度正确 (64通道, 256时间点)")
#         checks_passed += 1
#     else:
#         print(f"  ✗ 数据维度异常: 训练集{x_train.shape}, 测试集{x_test.shape}")
#
#     # 检查2: 标签类型
#     total_checks += 1
#     if set(np.unique(y_train)) == {0, 1} and set(np.unique(y_test)) == {0, 1}:
#         print("  ✓ 标签类型正确 (0和1)")
#         checks_passed += 1
#     else:
#         print(f"  ✗ 标签类型异常: 训练集{np.unique(y_train)}, 测试集{np.unique(y_test)}")
#
#     # 检查3: 训练集平衡性
#     total_checks += 1
#     if validation_results['train_target_ratio'] >= 0.45 and validation_results['train_target_ratio'] <= 0.55:
#         print(f"  ✓ 训练集基本平衡 (目标比例: {validation_results['train_target_ratio']:.2%})")
#         checks_passed += 1
#     else:
#         print(f"  ✗ 训练集不平衡 (目标比例: {validation_results['train_target_ratio']:.2%})")
#
#     # 检查4: P300特征方向
#     total_checks += 1
#     if validation_results['p300_direction_ok']:
#         print(f"  ✓ P300特征方向正确 (目标>非目标)")
#         checks_passed += 1
#     else:
#         print(f"  ✗ P300特征方向可能异常")
#
#     # 检查5: 数据范围
#     total_checks += 1
#     if validation_results['data_range_ok']:
#         print(f"  ✓ 数据范围合理")
#         checks_passed += 1
#     else:
#         print(f"  ✗ 数据范围可能异常")
#
#     # 检查6: 数据一致性
#     total_checks += 1
#     if validation_results['data_consistency_ok']:
#         print(f"  ✓ 训练集和测试集数据一致")
#         checks_passed += 1
#     else:
#         print(f"  ✗ 训练集和测试集数据不一致")
#
#     print(f"\n总检查项: {total_checks}")
#     print(f"通过项: {checks_passed}")
#     print(f"通过率: {checks_passed / total_checks * 100:.1f}%")
#
#     if checks_passed == total_checks:
#         print("\n✅ 所有检查通过，数据处理正确!")
#     else:
#         print(f"\n⚠ 有{total_checks - checks_passed}项检查未通过，请检查数据处理过程。")
#
#     print("\n" + "=" * 80)
#     print("校验完成!")
#     print("=" * 80)
#
#
# if __name__ == "__main__":
#     main()


"""
GIST数据集处理结果验证 - 使用原始论文的可视化逻辑
用于验证处理后的数据是否正确保持了ERP特征
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
import scipy.io as matReader
import mat73

# 设置matplotlib
import matplotlib

matplotlib.use('Qt5Agg')

# 数据路径
PROCESSED_DATA_PATH = r"D:\ZHB_BiYeSheJi\Dataset\GIST\S01"
RAW_DATA_PATH = r"D:\EEG-dataset-for-RSVP-P300-speller\Python\data"


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """巴特沃斯带通滤波器 - 与原始论文相同的滤波器"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data, axis=1)
    return y


def extractEpoch3D(data, event_indices, srate, baseline, frame, apply_baseline=True):
    """
    提取3D epoch数据 - 模拟原始论文的提取函数
    参数:
        data: [通道, 时间点] 原始EEG数据
        event_indices: 事件索引数组
        srate: 采样率
        baseline: 基线时间 [开始, 结束]，单位ms
        frame: 提取时间窗 [开始, 结束]，单位ms
        apply_baseline: 是否应用基线校正
    返回:
        epochs: [通道, 时间点, 试次] 3D数组
    """
    # 转换为样本点
    baseline_samples = [int(baseline[0] * srate / 1000), int(baseline[1] * srate / 1000)]
    frame_samples = [int(frame[0] * srate / 1000), int(frame[1] * srate / 1000)]

    # 计算epoch长度
    epoch_length = frame_samples[1] - frame_samples[0]
    n_channels = data.shape[0]
    n_trials = len(event_indices)

    # 初始化epochs数组
    epochs = np.zeros((n_channels, epoch_length, n_trials))

    for i, idx in enumerate(event_indices):
        start_idx = idx + frame_samples[0]
        end_idx = idx + frame_samples[1]

        # 确保索引在有效范围内
        if start_idx >= 0 and end_idx <= data.shape[1]:
            epoch = data[:, start_idx:end_idx]

            # 基线校正
            if apply_baseline:
                baseline_start = idx + baseline_samples[0]
                baseline_end = idx + baseline_samples[1]
                if baseline_start >= 0 and baseline_end <= data.shape[1]:
                    baseline_mean = np.mean(data[:, baseline_start:baseline_end], axis=1, keepdims=True)
                    epoch = epoch - baseline_mean

            epochs[:, :, i] = epoch

    return epochs


def plot_original_paper_style_with_processed_data():
    """使用原始论文的绘制逻辑，但使用我们处理好的数据"""
    print("=" * 80)
    print("使用原始论文可视化逻辑验证处理后的数据")
    print("=" * 80)

    # 1. 首先加载原始数据，按照原始论文的方式处理并绘制
    print("\n1. 加载原始数据并按照原始论文方式处理...")

    # 预定义参数 (与原始论文相同)
    baseline = [-200, 0]  # in ms
    frame = [-200, 1000]  # in ms

    # 加载原始数据
    raw_file_path = os.path.join(RAW_DATA_PATH, 's01.mat')

    try:
        # 尝试使用mat73加载（支持v7.3格式）
        try:
            EEG_raw = mat73.loadmat(raw_file_path)
        except:
            # 如果mat73失败，尝试使用scipy.io
            EEG_raw = matReader.loadmat(raw_file_path)

        # 提取RSVP数据
        cur_EEG = EEG_raw['RSVP']
        data_raw = np.asarray(cur_EEG['data'])
        srate_raw = cur_EEG['srate']  # 512 Hz
        markers_raw = cur_EEG['markers_target']

        print(f"原始数据信息:")
        print(f"  数据形状: {data_raw.shape}")
        print(f"  采样率: {srate_raw} Hz")
        print(f"  标记数量: {len(markers_raw)}")
        print(f"  目标标记数: {np.sum(markers_raw == 1)}")
        print(f"  非目标标记数: {np.sum(markers_raw == 2)}")

        # 检查并确保数据维度正确
        if len(data_raw.shape) > 2:
            data_raw = np.squeeze(data_raw)

        # 转置数据，使形状为 [channels, time]
        if data_raw.shape[0] > data_raw.shape[1]:
            data_raw = data_raw.T

        # 应用1-10Hz带通滤波 (与原始论文相同)
        print(f"  应用1-10Hz带通滤波...")
        data_raw_filtered = butter_bandpass_filter(data_raw, 1, 10, srate_raw, 4)

        # 提取目标和非目标试次
        targetID = np.where(markers_raw == 1)[0]
        nontargetID = np.where(markers_raw == 2)[0]

        print(f"  目标试次数: {len(targetID)}")
        print(f"  非目标试次数: {len(nontargetID)}")

        # 提取epochs
        targetEEG_raw = extractEpoch3D(data_raw_filtered, targetID, srate_raw, baseline, frame, True)
        nontargetEEG_raw = extractEpoch3D(data_raw_filtered, nontargetID, srate_raw, baseline, frame, True)

        # 计算试次平均
        avg_target_raw = np.mean(targetEEG_raw, axis=2)  # 试次平均 [通道, 时间点]
        avg_nontarget_raw = np.mean(nontargetEEG_raw, axis=2)

        print(f"  原始数据ERP形状: 目标={avg_target_raw.shape}, 非目标={avg_nontarget_raw.shape}")

    except Exception as e:
        print(f"加载原始数据失败: {e}")
        avg_target_raw = None
        avg_nontarget_raw = None

    # 2. 加载我们处理好的数据
    print("\n2. 加载处理后的数据...")

    try:
        # 加载处理后的数据
        x_train = np.load(os.path.join(PROCESSED_DATA_PATH, 'x_train.npy'))
        x_test = np.load(os.path.join(PROCESSED_DATA_PATH, 'x_test.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'))
        y_test = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'))

        # 合并训练集和测试集，得到所有数据
        x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)

        print(f"处理后的数据信息:")
        print(f"  总样本数: {len(x_all)}")
        print(f"  目标样本数: {np.sum(y_all == 1)}")
        print(f"  非目标样本数: {np.sum(y_all == 0)}")
        print(f"  数据形状: {x_all.shape}")

        # 分离目标和刺激
        target_idx = np.where(y_all == 1)[0]
        nontarget_idx = np.where(y_all == 0)[0]

        # 只使用实际的EEG通道 (前32个通道)
        target_data_processed = x_all[target_idx, :32, :]
        nontarget_data_processed = x_all[nontarget_idx, :32, :]

        # 计算平均ERP (试次平均)
        avg_target_processed = np.mean(target_data_processed, axis=0)  # [通道, 时间点]
        avg_nontarget_processed = np.mean(nontarget_data_processed, axis=0)

        print(f"  处理后数据ERP形状: 目标={avg_target_processed.shape}, 非目标={avg_nontarget_processed.shape}")

    except Exception as e:
        print(f"加载处理后数据失败: {e}")
        avg_target_processed = None
        avg_nontarget_processed = None

    # 3. 绘制对比图
    print("\n3. 绘制对比图...")

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GIST数据集处理结果验证 - 原始论文可视化风格', fontsize=16, fontweight='bold')

    # 3.1 原始数据 - 中线电极平均ERP
    if avg_target_raw is not None:
        # 通道选择 (与原始论文相同: Fz, Cz, Pz)
        # 注意: 原始论文中电极索引是31, 32, 13 (1-based索引)
        # 转换为0-based索引: 30, 31, 12
        elec_midline_raw = [30, 31, 12]  # Fz, Cz, Pz

        # 确保索引在范围内
        valid_electrodes = [idx for idx in elec_midline_raw if idx < avg_target_raw.shape[0]]

        if len(valid_electrodes) > 0:
            # 计算中线电极平均
            ch_avg_target_raw = np.mean(avg_target_raw[valid_electrodes, :], axis=0)
            ch_avg_nontarget_raw = np.mean(avg_nontarget_raw[valid_electrodes, :], axis=0)

            # 创建时间轴 (-200到1000ms)
            t_raw = np.linspace(-200, 1000, avg_target_raw.shape[1])

            ax = axes[0, 0]
            ax.plot(t_raw, ch_avg_target_raw.transpose(), color=[1, 0.5, 0], linewidth=2.5, label='目标')
            ax.plot(t_raw, ch_avg_nontarget_raw.transpose(), color=[0, 0, 0], linewidth=2.5, label='非目标')
            ax.set_xlabel('时间 (ms)', fontsize=12)
            ax.set_ylabel(r'$\mu V$', fontsize=12)
            ax.set_title('原始数据 - 中线电极平均ERP\n(原始论文处理方式)', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_xlim([-200, 1000])

            # 标记P300
            ax.axvline(x=300, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='P300')
            ax.legend(loc='upper right', fontsize=11)

    # 3.2 处理后的数据 - 中线电极平均ERP (假设的电极索引)
    if avg_target_processed is not None:
        # 假设的电极索引 (根据标准10-20系统32导联)
        # Fz, Cz, Pz的索引
        elec_midline_processed = [4, 12, 20]  # 假设的索引

        # 确保索引在范围内
        valid_electrodes = [idx for idx in elec_midline_processed if idx < avg_target_processed.shape[0]]

        if len(valid_electrodes) > 0:
            # 计算中线电极平均
            ch_avg_target_processed = np.mean(avg_target_processed[valid_electrodes, :], axis=0)
            ch_avg_nontarget_processed = np.mean(avg_nontarget_processed[valid_electrodes, :], axis=0)

            # 创建时间轴 (0到1000ms，因为我们去掉了基线期)
            t_processed = np.linspace(0, 1000, avg_target_processed.shape[1])

            ax = axes[0, 1]
            ax.plot(t_processed, ch_avg_target_processed.transpose(), color=[1, 0.5, 0], linewidth=2.5, label='目标')
            ax.plot(t_processed, ch_avg_nontarget_processed.transpose(), color=[0, 0, 0], linewidth=2.5, label='非目标')
            ax.set_xlabel('时间 (ms)', fontsize=12)
            ax.set_ylabel(r'$\mu V$', fontsize=12)
            ax.set_title('处理后数据 - 中线电极平均ERP\n(0-1000ms，无基线期)', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_xlim([0, 1000])

            # 标记P300
            ax.axvline(x=300, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='P300')
            ax.legend(loc='upper right', fontsize=11)

    # 3.3 原始数据 - 所有电极平均ERP
    if avg_target_raw is not None:
        # 计算所有电极平均
        all_ch_avg_target_raw = np.mean(avg_target_raw, axis=0)
        all_ch_avg_nontarget_raw = np.mean(avg_nontarget_raw, axis=0)

        t_raw = np.linspace(-200, 1000, avg_target_raw.shape[1])

        ax = axes[1, 0]
        ax.plot(t_raw, all_ch_avg_target_raw.transpose(), color='red', linewidth=2.5, label='目标')
        ax.plot(t_raw, all_ch_avg_nontarget_raw.transpose(), color='blue', linewidth=2.5, linestyle='--',
                label='非目标')
        ax.set_xlabel('时间 (ms)', fontsize=12)
        ax.set_ylabel(r'$\mu V$', fontsize=12)
        ax.set_title('原始数据 - 所有电极平均ERP', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlim([-200, 1000])

        # 标记P300
        ax.axvline(x=300, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='P300')
        ax.legend(loc='upper right', fontsize=11)

    # 3.4 处理后数据 - 所有电极平均ERP
    if avg_target_processed is not None:
        # 计算所有电极平均
        all_ch_avg_target_processed = np.mean(avg_target_processed, axis=0)
        all_ch_avg_nontarget_processed = np.mean(avg_nontarget_processed, axis=0)

        t_processed = np.linspace(0, 1000, avg_target_processed.shape[1])

        ax = axes[1, 1]
        ax.plot(t_processed, all_ch_avg_target_processed.transpose(), color='red', linewidth=2.5, label='目标')
        ax.plot(t_processed, all_ch_avg_nontarget_processed.transpose(), color='blue', linewidth=2.5, linestyle='--',
                label='非目标')
        ax.set_xlabel('时间 (ms)', fontsize=12)
        ax.set_ylabel(r'$\mu V$', fontsize=12)
        ax.set_title('处理后数据 - 所有电极平均ERP', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlim([0, 1000])

        # 标记P300
        ax.axvline(x=300, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='P300')
        ax.legend(loc='upper right', fontsize=11)

    # 调整子图间距
    plt.tight_layout()
    plt.show()

    # 4. 绘制与原始论文完全相同的图 (单图)
    print("\n4. 绘制与原始论文完全相同的单图...")

    if avg_target_raw is not None and avg_nontarget_raw is not None:
        # 使用原始论文的电极索引
        elec_midline_raw = [30, 31, 12]  # Fz, Cz, Pz (1-based: 31, 32, 13)

        # 确保索引在范围内
        valid_electrodes = [idx for idx in elec_midline_raw if idx < avg_target_raw.shape[0]]

        if len(valid_electrodes) > 0:
            # 计算中线电极平均
            ch_avg_target_raw = np.mean(avg_target_raw[valid_electrodes, :], axis=0)
            ch_avg_nontarget_raw = np.mean(avg_nontarget_raw[valid_electrodes, :], axis=0)

            # 创建时间轴
            t = np.linspace(-200, 1000, avg_target_raw.shape[1])

            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制曲线 (使用原始论文的颜色)
            ax.plot(t, ch_avg_target_raw.transpose(), color=[1, 0.5, 0], linewidth=3, label='目标刺激')
            ax.plot(t, ch_avg_nontarget_raw.transpose(), color=[0, 0, 0], linewidth=3, label='非目标刺激')

            # 设置标签和标题
            ax.set_xlabel('时间 (ms)', fontsize=14)
            ax.set_ylabel(r'幅值 ($\mu V$)', fontsize=14)
            ax.set_title('GIST数据集 - 原始论文ERP可视化', fontsize=16, fontweight='bold')

            # 网格和轴线
            ax.yaxis.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8)

            # 设置范围
            ax.set_xlim([-200, 1000])

            # 设置字体大小 (与原始论文相同)
            plt.rcParams.update({'font.size': 13})

            # 图例
            ax.legend(loc='upper right', fontsize=12)

            # 设置纵横比 (与原始论文相同)
            ratio = 0.6
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            plt.tight_layout()
            plt.show()

    # 5. 数据处理质量分析
    print("\n5. 数据处理质量分析:")

    if avg_target_raw is not None and avg_target_processed is not None:
        # 计算P300幅值 (250-350ms时间窗)
        # 原始数据
        t_raw = np.linspace(-200, 1000, avg_target_raw.shape[1])
        p300_start_raw = np.argmin(np.abs(t_raw - 250))
        p300_end_raw = np.argmin(np.abs(t_raw - 350))

        # 处理后数据
        t_processed = np.linspace(0, 1000, avg_target_processed.shape[1])
        p300_start_processed = np.argmin(np.abs(t_processed - 250))
        p300_end_processed = np.argmin(np.abs(t_processed - 350))

        # 计算所有电极平均的P300幅值
        all_ch_avg_target_raw = np.mean(avg_target_raw, axis=0)
        all_ch_avg_nontarget_raw = np.mean(avg_nontarget_raw, axis=0)

        all_ch_avg_target_processed = np.mean(avg_target_processed, axis=0)
        all_ch_avg_nontarget_processed = np.mean(avg_nontarget_processed, axis=0)

        # 计算P300幅值
        p300_target_raw = np.mean(all_ch_avg_target_raw[p300_start_raw:p300_end_raw])
        p300_nontarget_raw = np.mean(all_ch_avg_nontarget_raw[p300_start_raw:p300_end_raw])
        p300_diff_raw = p300_target_raw - p300_nontarget_raw

        p300_target_processed = np.mean(all_ch_avg_target_processed[p300_start_processed:p300_end_processed])
        p300_nontarget_processed = np.mean(all_ch_avg_nontarget_processed[p300_start_processed:p300_end_processed])
        p300_diff_processed = p300_target_processed - p300_nontarget_processed

        print(f"  原始数据P300幅值:")
        print(f"    目标: {p300_target_raw:.4f} μV")
        print(f"    非目标: {p300_nontarget_raw:.4f} μV")
        print(f"    差异: {p300_diff_raw:.4f} μV")

        print(f"  处理后数据P300幅值:")
        print(f"    目标: {p300_target_processed:.4f} μV")
        print(f"    非目标: {p300_nontarget_processed:.4f} μV")
        print(f"    差异: {p300_diff_processed:.4f} μV")

        # 计算相对差异
        if p300_diff_raw != 0:
            rel_diff = abs(p300_diff_processed - p300_diff_raw) / abs(p300_diff_raw) * 100
            print(f"  P300差异相对变化: {rel_diff:.2f}%")

            if rel_diff < 20:
                print(f"  ✅ P300特征保持良好 (变化 < 20%)")
            else:
                print(f"  ⚠ P300特征变化较大 (变化 = {rel_diff:.2f}%)")

        # 检查数据范围
        print(f"\n  数据范围检查:")
        print(f"    原始数据范围: [{np.min(avg_target_raw):.2f}, {np.max(avg_target_raw):.2f}] μV")
        print(f"    处理后数据范围: [{np.min(avg_target_processed):.2f}, {np.max(avg_target_processed):.2f}] μV")

        # 检查数据形状
        print(f"\n  数据形状:")
        print(f"    原始数据: {avg_target_raw.shape} (通道, 时间点)")
        print(f"    处理后数据: {avg_target_processed.shape} (通道, 时间点)")

        # 原始数据是614个时间点 (-200到1000ms @ 512Hz)
        # 处理后数据是256个时间点 (0到1000ms @ 256Hz)

        # 计算时间分辨率
        time_res_raw = 1200 / avg_target_raw.shape[1]  # ms/点
        time_res_processed = 1000 / avg_target_processed.shape[1]  # ms/点

        print(f"\n  时间分辨率:")
        print(f"    原始数据: {time_res_raw:.2f} ms/点 (约{1000 / time_res_raw:.0f} Hz)")
        print(f"    处理后数据: {time_res_processed:.2f} ms/点 ({1000 / time_res_processed:.0f} Hz)")

    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)


def plot_processed_data_with_original_style():
    """只使用处理后的数据，但使用原始论文的绘制风格"""
    print("=" * 80)
    print("使用处理后数据的原始论文风格可视化")
    print("=" * 80)

    # 加载处理后的数据
    print("\n加载处理后的数据...")

    try:
        # 加载处理后的数据
        x_train = np.load(os.path.join(PROCESSED_DATA_PATH, 'x_train.npy'))
        x_test = np.load(os.path.join(PROCESSED_DATA_PATH, 'x_test.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'))
        y_test = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'))

        # 合并训练集和测试集，得到所有数据
        x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)

        print(f"数据信息:")
        print(f"  总样本数: {len(x_all)}")
        print(f"  目标样本数: {np.sum(y_all == 1)}")
        print(f"  非目标样本数: {np.sum(y_all == 0)}")
        print(f"  数据形状: {x_all.shape}")

        # 分离目标和刺激
        target_idx = np.where(y_all == 1)[0]
        nontarget_idx = np.where(y_all == 0)[0]

        # 只使用实际的EEG通道 (前32个通道)
        target_data = x_all[target_idx, :32, :]
        nontarget_data = x_all[nontarget_idx, :32, :]

        # 计算平均ERP (试次平均)
        avg_target = np.mean(target_data, axis=0)  # [通道, 时间点]
        avg_nontarget = np.mean(nontarget_data, axis=0)

        # 假设的电极索引 (Fz, Cz, Pz)
        elec_midline = [4, 12, 20]

        # 确保索引在范围内
        valid_electrodes = [idx for idx in elec_midline if idx < avg_target.shape[0]]

        if len(valid_electrodes) > 0:
            # 计算中线电极平均
            ch_avg_target = np.mean(avg_target[valid_electrodes, :], axis=0)
            ch_avg_nontarget = np.mean(avg_nontarget[valid_electrodes, :], axis=0)

            # 创建时间轴 (0到1000ms)
            t = np.linspace(0, 1000, avg_target.shape[1])

            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制曲线 (使用原始论文的颜色)
            ax.plot(t, ch_avg_target.transpose(), color=[1, 0.5, 0], linewidth=3, label='目标刺激')
            ax.plot(t, ch_avg_nontarget.transpose(), color=[0, 0, 0], linewidth=3, label='非目标刺激')

            # 设置标签和标题
            ax.set_xlabel('时间 (ms)', fontsize=14)
            ax.set_ylabel(r'幅值 ($\mu V$)', fontsize=14)
            ax.set_title('GIST数据集处理后数据 - ERP可视化', fontsize=16, fontweight='bold')

            # 网格和轴线
            ax.yaxis.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8)

            # 设置范围
            ax.set_xlim([0, 1000])

            # 标记P300
            ax.axvline(x=300, color='green', linestyle=':', linewidth=2, alpha=0.7, label='P300 (~300ms)')

            # 设置字体大小
            plt.rcParams.update({'font.size': 13})

            # 图例
            ax.legend(loc='upper right', fontsize=12)

            # 设置纵横比
            ratio = 0.6
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            plt.tight_layout()
            plt.show()

            # 计算P300幅值
            p300_start = np.argmin(np.abs(t - 250))
            p300_end = np.argmin(np.abs(t - 350))

            p300_target = np.mean(ch_avg_target[p300_start:p300_end])
            p300_nontarget = np.mean(ch_avg_nontarget[p300_start:p300_end])
            p300_diff = p300_target - p300_nontarget

            print(f"\nP300幅值分析 (中线电极平均):")
            print(f"  目标: {p300_target:.4f} μV")
            print(f"  非目标: {p300_nontarget:.4f} μV")
            print(f"  差异: {p300_diff:.4f} μV")

            if p300_diff > 0:
                print(f"  ✅ P300特征方向正确 (目标 > 非目标)")
            else:
                print(f"  ⚠ P300特征方向异常 (目标 < 非目标)")

            # 检查典型P300幅值 (通常在3-20μV之间)
            if 1 < p300_diff < 30:
                print(f"  ✅ P300幅值在合理范围内")
            else:
                print(f"  ⚠ P300幅值可能异常")

    except Exception as e:
        print(f"加载数据失败: {e}")


if __name__ == "__main__":
    print("GIST数据集处理结果验证")
    print("=" * 80)

    # 选择要运行的函数
    choice = input("选择验证方式:\n1. 对比原始数据和处理后数据\n2. 仅使用处理后数据\n请输入选择 (1或2): ").strip()

    if choice == "1":
        # 运行完整的对比验证
        plot_original_paper_style_with_processed_data()
    elif choice == "2":
        # 只运行处理后数据的验证
        plot_processed_data_with_original_style()
    else:
        print("无效选择，默认运行完整对比验证")
        plot_original_paper_style_with_processed_data()