import matplotlib

# 必须在导入pyplot之前设置后端
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')


def set_chinese_font_safe():
    """
    安全设置中文字体，避免字体缺失错误
    """
    # 获取系统中可用的字体
    import matplotlib.font_manager as fm

    # 尝试的中文字体列表
    chinese_fonts = [
        'SimHei',  # Windows 黑体
        'Microsoft YaHei',  # Windows 微软雅黑
        'DejaVu Sans',  # Linux 通用字体
        'Arial Unicode MS',  # Mac
        'sans-serif'  # 备用通用字体
    ]

    # 检查系统中可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 选择第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    if selected_font:
        plt.rcParams["font.family"] = selected_font
        print(f"使用字体: {selected_font}")
    else:
        # 如果没有中文字体，使用默认字体
        plt.rcParams["font.family"] = "sans-serif"
        print("未找到中文字体，使用默认字体")

    plt.rcParams["axes.unicode_minus"] = False


def inspect_rsvp_data(file_path, sample_rate=256, visualize=True, save_figures=True):
    """
    读取并可视化脑电RSVP数据
    参数:
        file_path: 数据文件路径
        sample_rate: 采样率（Hz）
        visualize: 是否显示图形
        save_figures: 是否保存图形到文件
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}")
            return

        data = np.load(file_path)
        n_trials, n_channels, n_timepoints = data.shape

        # 打印基础信息
        print("=" * 60)
        print(f"文件: {file_path}")
        print(f"数据形状: 试次={n_trials}, 通道={n_channels}, 时间点={n_timepoints}")
        print(f"时间窗口: {n_timepoints / sample_rate * 1000:.1f}ms (采样率={sample_rate}Hz)")
        print(f"数据类型: {data.dtype}")
        print(f"数值范围: 最小值={data.min():.2f}, 最大值={data.max():.2f}, 均值={data.mean():.2f}μV")
        print(f"标准差: {data.std():.2f}μV")

        # 基本统计
        print("\n各通道均值:")
        channel_means = data.mean(axis=(0, 2))
        for ch in range(min(5, n_channels)):
            print(f"  通道{ch}: {channel_means[ch]:.2f}μV")

        if visualize and save_figures:
            figures = visualize_rsvp_safe(data, sample_rate, save_figures)
            return figures
        elif visualize:
            visualize_rsvp_safe(data, sample_rate, save_figures=False)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def visualize_rsvp_safe(data, sample_rate, save_figures=True):
    """
    安全的可视化函数，保存图形到文件而不显示
    """
    n_trials, n_channels, n_timepoints = data.shape
    time_axis = np.arange(n_timepoints) / sample_rate * 1000

    # 创建输出目录
    output_dir = "rsvp_visualization"
    if save_figures and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    figures = []

    # 1. 单试次多通道时序图
    print("\n生成图1: 单试次多通道时序图...")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    trial_idx = 0
    channel_offsets = np.arange(n_channels) * 5

    for ch in range(min(20, n_channels)):  # 只显示前20个通道
        ax1.plot(time_axis, data[trial_idx, ch, :] + channel_offsets[ch],
                 linewidth=0.8, alpha=0.8)

    ax1.axvline(x=0, color='r', linestyle='--', linewidth=1.5, label='刺激呈现')
    ax1.set_xlabel("时间 (ms)", fontsize=12)
    ax1.set_ylabel("脑电信号 (μV + 偏移)", fontsize=12)
    ax1.set_title(f"试次{trial_idx}的前20个通道脑电时序", fontsize=14, pad=20)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    if save_figures:
        fig1_path = os.path.join(output_dir, "trial_channels_timeline.png")
        fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {fig1_path}")

    # 2. 通道×时间热力图
    print("生成图2: 通道×时间热力图...")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    mean_trial = data[:min(100, n_trials)].mean(axis=0)

    # 计算合适的颜色范围
    vmax = np.percentile(np.abs(mean_trial), 95)  # 使用95%分位数
    vmin = -vmax

    im = ax2.imshow(mean_trial, aspect='auto', cmap='RdBu_r',
                    vmin=vmin, vmax=vmax,
                    extent=[time_axis[0], time_axis[-1], n_channels - 1, 0])

    ax2.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
    ax2.set_xlabel("时间 (ms)", fontsize=12)
    ax2.set_ylabel("通道索引", fontsize=12)
    ax2.set_title("前100个试次的平均脑电（热力图）", fontsize=14, pad=20)

    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('平均电位 (μV)', fontsize=12)

    if save_figures:
        fig2_path = os.path.join(output_dir, "channel_time_heatmap.png")
        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {fig2_path}")

    # 3. ERP波形
    print("生成图3: ERP波形图...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # 选择有代表性的通道
    if n_channels >= 20:
        key_channels = [0, 7, 13, 20, 30]  # 分布在不同脑区
    else:
        key_channels = list(range(min(5, n_channels)))

    for ch in key_channels:
        erp = data[:, ch, :].mean(axis=0)
        ax3.plot(time_axis, erp, linewidth=2, label=f"通道{ch}")

    ax3.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.set_xlabel("时间 (ms)", fontsize=12)
    ax3.set_ylabel("平均电位 (μV)", fontsize=12)
    ax3.set_title("关键通道的平均诱发电位（ERP）", fontsize=14, pad=20)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    if save_figures:
        fig3_path = os.path.join(output_dir, "erp_waveforms.png")
        fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {fig3_path}")

    # 4. 通道分布图
    print("生成图4: 通道分布图...")
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    # 找到300ms附近的时间点
    time_diff = np.abs(time_axis - 300)
    target_time_idx = np.argmin(time_diff)
    target_time = time_axis[target_time_idx]

    channel_means = data[:, :, target_time_idx].mean(axis=0)
    channels = range(n_channels)

    bars = ax4.bar(channels, channel_means, alpha=0.7, edgecolor='black', linewidth=0.5)

    # 标记正值和负值
    for i, (ch, val) in enumerate(zip(channels, channel_means)):
        if val > 0:
            bars[i].set_color('red')
        else:
            bars[i].set_color('blue')

    ax4.set_xlabel("通道索引", fontsize=12)
    ax4.set_ylabel(f"{target_time:.0f}ms时的平均电位 (μV)", fontsize=12)
    ax4.set_title(f"刺激后{target_time:.0f}ms的各通道平均电位分布", fontsize=14, pad=20)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_axisbelow(True)

    if save_figures:
        fig4_path = os.path.join(output_dir, "channel_distribution.png")
        fig4.savefig(fig4_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {fig4_path}")

    # 5. 试次间变异性
    print("生成图5: 试次间变异性...")
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))

    # 5a. 所有试次在通道0的叠加
    channel_to_plot = 0
    for trial in range(min(50, n_trials)):
        ax5a.plot(time_axis, data[trial, channel_to_plot, :],
                  alpha=0.1, linewidth=0.5, color='blue')

    # 平均ERP
    avg_erp = data[:, channel_to_plot, :].mean(axis=0)
    ax5a.plot(time_axis, avg_erp, 'r-', linewidth=2, label='平均ERP')

    ax5a.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
    ax5a.set_xlabel("时间 (ms)", fontsize=12)
    ax5a.set_ylabel("电位 (μV)", fontsize=12)
    ax5a.set_title(f"通道{channel_to_plot}的试次叠加", fontsize=12)
    ax5a.legend()
    ax5a.grid(True, alpha=0.3)

    # 5b. 试次间标准差
    std_across_trials = data.std(axis=0)  # 形状: (n_channels, n_timepoints)
    channel_std = std_across_trials[channel_to_plot, :]

    ax5b.plot(time_axis, channel_std, 'g-', linewidth=2)
    ax5b.axvline(x=0, color='k', linestyle='--', linewidth=1.5)
    ax5b.fill_between(time_axis, 0, channel_std, alpha=0.3, color='green')
    ax5b.set_xlabel("时间 (ms)", fontsize=12)
    ax5b.set_ylabel("标准差 (μV)", fontsize=12)
    ax5b.set_title(f"通道{channel_to_plot}的试次间变异性", fontsize=12)
    ax5b.grid(True, alpha=0.3)

    fig5.suptitle("试次间变异性分析", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_figures:
        fig5_path = os.path.join(output_dir, "trial_variability.png")
        fig5.savefig(fig5_path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {fig5_path}")

    figures = [fig1, fig2, fig3, fig4, fig5]

    if not save_figures:
        # 尝试显示图形（可能仍然失败）
        try:
            plt.show()
        except:
            print("警告: 无法显示图形，请检查图形后端设置")

    return figures


def batch_process_datasets():
    """
    批量处理多个数据集的函数
    """
    datasets = {
        "THU_S01_train": r"D:\ZHB_BiYeSheJi\Dataset\THU\S01\x_train.npy",
        "THU_S01_test": r"D:\ZHB_BiYeSheJi\Dataset\THU\S01\x_test.npy",
        "GIST_S01_train": r"D:\ZHB_BiYeSheJi\Dataset\GIST\S01\x_train.npy",
        "GIST_S01_test": r"D:\ZHB_BiYeSheJi\Dataset\GIST\S01\x_test.npy",
    }

    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"\n{'=' * 60}")
            print(f"处理数据集: {name}")
            print(f"路径: {path}")
            print(f"{'=' * 60}")

            inspect_rsvp_data(path, sample_rate=256, visualize=True, save_figures=True)
        else:
            print(f"\n数据集不存在: {name} - {path}")


if __name__ == "__main__":
    # 设置中文字体（安全模式）
    set_chinese_font_safe()

    # 单个文件测试
    npy_file = r"D:\ZHB_BiYeSheJi\Dataset\GIST\S01\x_train.npy"

    if os.path.exists(npy_file):
        inspect_rsvp_data(npy_file, sample_rate=256, visualize=True, save_figures=True)
    else:
        print(f"文件不存在: {npy_file}")
        print("尝试使用测试文件...")

        # 创建测试数据
        test_file = "test_rsvp_data.npy"
        if not os.path.exists(test_file):
            print(f"创建测试数据: {test_file}")
            # 生成模拟RSVP数据
            np.random.seed(42)
            test_data = np.random.randn(100, 64, 256) * 10  # 100试次，64通道，256时间点
            test_data[:, :, 100:150] += 5  # 添加模拟的P300成分
            np.save(test_file, test_data)

        inspect_rsvp_data(test_file, sample_rate=256, visualize=True, save_figures=True)

    # 可选：批量处理
    # batch_process_datasets()

    print("\n处理完成！图形已保存到 'rsvp_visualization' 目录")