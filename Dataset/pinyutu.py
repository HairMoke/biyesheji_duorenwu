import numpy as np
import matplotlib.pyplot as plt

# 中文和负号显示设置
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


def debug_frequency_analysis(x_train, y_train, sample_rate=256):
    """
    调试版本的频域分析，找出根本问题
    """
    print("=== 开始调试分析 ===")

    # 分离数据
    target_data = x_train[y_train == 1]
    non_target_data = x_train[y_train == 0]

    print(f"目标数据形状: {target_data.shape}")
    print(f"非目标数据形状: {non_target_data.shape}")

    # 计算全局平均
    target_avg = target_data.mean(axis=(0, 1))
    non_target_avg = non_target_data.mean(axis=(0, 1))

    print(f"目标平均形状: {target_avg.shape}")
    print(f"非目标平均形状: {non_target_avg.shape}")

    # 计算频谱
    freqs_target, psd_target = compute_psd_debug(target_avg, sample_rate)
    freqs_non_target, psd_non_target = compute_psd_debug(non_target_avg, sample_rate)

    print(f"目标频率形状: {freqs_target.shape if hasattr(freqs_target, 'shape') else '无形状'}")
    print(f"目标PSD形状: {psd_target.shape if hasattr(psd_target, 'shape') else '无形状'}")
    print(f"非目标频率形状: {freqs_non_target.shape if hasattr(freqs_non_target, 'shape') else '无形状'}")
    print(f"非目标PSD形状: {psd_non_target.shape if hasattr(psd_non_target, 'shape') else '无形状'}")

    # 检查数组内容
    print(
        f"目标频率类型: {type(freqs_target)}, 长度: {len(freqs_target) if hasattr(freqs_target, '__len__') else '无长度'}")
    print(f"目标PSD类型: {type(psd_target)}, 长度: {len(psd_target) if hasattr(psd_target, '__len__') else '无长度'}")

    return freqs_target, psd_target, freqs_non_target, psd_non_target


def compute_psd_debug(signal, sample_rate):
    """
    调试版本的PSD计算，添加详细检查
    """
    print(f"输入信号形状: {signal.shape}")
    print(f"输入信号长度: {len(signal)}")

    n = len(signal)

    if n == 0:
        print("警告: 信号长度为0")
        return np.array([]), np.array([])

    # 执行FFT
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)

    print(f"FFT结果形状: {fft_vals.shape}")
    print(f"频率数组形状: {freqs.shape}")

    # 限制在0-50Hz
    mask = (freqs >= 0) & (freqs <= 50)
    freqs = freqs[mask]
    fft_amp = np.abs(fft_vals[mask])

    print(f"滤波后频率形状: {freqs.shape}")
    print(f"滤波后幅度形状: {fft_amp.shape}")

    if len(fft_amp) == 0:
        print("警告: 滤波后无数据")
        return np.array([]), np.array([])

    # 计算PSD (dB)
    psd = (fft_amp ** 2) / (sample_rate / 2)
    psd_db = 10 * np.log10(psd + 1e-12)

    print(f"最终PSD形状: {psd_db.shape}")
    print("=== PSD计算完成 ===\n")

    return freqs, psd_db


def compute_psd_safe(signal, sample_rate):
    """
    安全版本的PSD计算，确保返回正确的数组
    """
    n = len(signal)

    if n == 0:
        return np.array([]), np.array([])

    try:
        # 执行FFT
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=1 / sample_rate)

        # 限制在0-50Hz
        mask = (freqs >= 0) & (freqs <= 50)
        freqs_filtered = freqs[mask]
        fft_amp_filtered = np.abs(fft_vals[mask])

        if len(fft_amp_filtered) == 0:
            return np.array([]), np.array([])

        # 计算PSD (dB)
        psd = (fft_amp_filtered ** 2) / (sample_rate / 2)
        psd_db = 10 * np.log10(psd + 1e-12)

        # 确保返回numpy数组
        return np.asarray(freqs_filtered), np.asarray(psd_db)

    except Exception as e:
        print(f"PSD计算错误: {e}")
        return np.array([]), np.array([])


def plot_separate_frequency_plots(x_train, y_train, sample_rate=256):
    """
    绘制两张独立的频率图：一张目标刺激，一张非目标刺激
    """
    # 分离数据
    target_data = x_train[y_train == 1]
    non_target_data = x_train[y_train == 0]

    print("开始绘制独立频率图...")

    # 计算全局平均
    target_avg = target_data.mean(axis=(0, 1))
    non_target_avg = non_target_data.mean(axis=(0, 1))

    # 计算频谱
    freqs_target, psd_target = compute_psd_safe(target_avg, sample_rate)
    freqs_non_target, psd_non_target = compute_psd_safe(non_target_avg, sample_rate)

    # ========== 第一张图：目标刺激频率图 ==========
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    if len(freqs_target) > 0 and len(psd_target) > 0 and len(freqs_target) == len(psd_target):
        # 绘制目标刺激频谱
        ax1.plot(freqs_target, psd_target, 'r-', linewidth=2, label='目标刺激', alpha=0.8)
        # 填充区域增强可视化
        y_min = np.min(psd_target) - 5
        ax1.fill_between(freqs_target, psd_target, y_min,
                         where=(psd_target > y_min),
                         alpha=0.2, color='red', interpolate=True)

        # 添加频段标记
        bands = [(1, 4, 'Delta', 'purple'), (4, 8, 'Theta', 'blue'),
                 (8, 13, 'Alpha', 'green'), (13, 30, 'Beta', 'orange')]
        for f_low, f_high, name, color in bands:
            ax1.axvspan(f_low, f_high, alpha=0.1, color=color, label=f'{name}频段')

    # 设置第一张图属性
    ax1.set_xlim(0, 30)
    ax1.set_xlabel('频率 (Hz)', fontsize=12)
    ax1.set_ylabel('功率谱密度 (dB)', fontsize=12)
    ax1.set_title('目标刺激频率响应', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    # 保存第一张图
    fig1.savefig('target_frequency_analysis.png', dpi=300, bbox_inches='tight')
    print("目标刺激频率图已保存为: target_frequency_analysis.png")

    # ========== 第二张图：非目标刺激频率图 ==========
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    if len(freqs_non_target) > 0 and len(psd_non_target) > 0 and len(freqs_non_target) == len(psd_non_target):
        # 绘制非目标刺激频谱
        ax2.plot(freqs_non_target, psd_non_target, 'b-', linewidth=2, label='非目标刺激', alpha=0.8)
        # 填充区域增强可视化
        y_min = np.min(psd_non_target) - 5
        ax2.fill_between(freqs_non_target, psd_non_target, y_min,
                         where=(psd_non_target > y_min),
                         alpha=0.2, color='blue', interpolate=True)

        # 添加频段标记
        for f_low, f_high, name, color in bands:
            ax2.axvspan(f_low, f_high, alpha=0.1, color=color, label=f'{name}频段')

    # 设置第二张图属性
    ax2.set_xlim(0, 30)
    ax2.set_xlabel('频率 (Hz)', fontsize=12)
    ax2.set_ylabel('功率谱密度 (dB)', fontsize=12)
    ax2.set_title('非目标刺激频率响应', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    # 保存第二张图
    fig2.savefig('non_target_frequency_analysis.png', dpi=300, bbox_inches='tight')
    print("非目标刺激频率图已保存为: non_target_frequency_analysis.png")

    return fig1, fig2


# 主函数
def main():
    try:
        # 读取数据
        x_train = np.load(r"D:\ZHB_BiYeSheJi\Dataset\THU\S01\x_train.npy")
        y_train = np.load(r"D:\ZHB_BiYeSheJi\Dataset\THU\S01\y_train.npy")

        print(f"数据形状: x_train {x_train.shape}, y_train {y_train.shape}")
        print(f"标签分布: 目标 {np.sum(y_train == 1)}, 非目标 {np.sum(y_train == 0)}")

        # 首先进行调试分析
        print("\n=== 调试分析开始 ===")
        debug_frequency_analysis(x_train, y_train)
        print("=== 调试分析结束 ===\n")

        # 绘制两张独立的频率图（核心修改）
        print("=== 绘制独立频率图 ===")
        fig1, fig2 = plot_separate_frequency_plots(x_train, y_train)

        # 显示图片
        plt.show()

    except Exception as e:
        print(f"\n主程序错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()