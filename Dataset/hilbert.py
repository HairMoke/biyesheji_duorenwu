import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 显示负号


# 切换Matplotlib后端，解决tostring_rgb报错
plt.switch_backend('TkAgg')  # 改用TkAgg后端（兼容Windows/Linux/Mac）
from scipy.signal import hilbert
from scipy.stats import norm

# ===================== 1. 核心参数设置 =====================
n_channels = 64  # 脑电通道数
n_samples = 256  # 每个通道的采样点数
fs = 128  # 采样频率(Hz)，EEG常用
t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)  # 时间轴（0-2秒）
stim_time = 1.0  # 刺激呈现时间（1秒处）
p300_latency = 0.3  # P300潜伏期（刺激后300ms）
p300_peak_time = stim_time + p300_latency  # P300峰值时间（1.3秒）
np.random.seed(42)  # 固定随机种子，结果可复现

# ===================== 2. 模拟含P300的脑电数据 =====================
eeg_data = np.zeros((n_channels, n_samples))

# 基础脑电节律参数
theta_freq = 6  # θ波(4-8Hz)
alpha_freq = 10  # α波(8-13Hz)
beta_freq = 20  # β波(13-30Hz)


# 生成P300波形（高斯窗函数模拟P300的钟形特征）
def generate_p300_waveform(t, peak_time, sigma=0.05, amplitude=3):
    """生成P300波形（高斯调制的正相波）"""
    gaussian = norm.pdf(t, loc=peak_time, scale=sigma)
    # 归一化并缩放振幅
    p300_wave = amplitude * (gaussian / np.max(gaussian))
    return p300_wave


# 为每个通道生成含P300和相位变化的EEG信号
for ch in range(n_channels):
    # 随机基础节律振幅（通道间差异）
    theta_amp = np.random.uniform(0.5, 2.0)
    alpha_amp = np.random.uniform(1.0, 3.0)
    beta_amp = np.random.uniform(0.2, 1.0)

    # 生成基础节律（刺激前正常相位，刺激后相位偏移）
    # 相位掩码：刺激前相位=0，刺激后相位偏移π/3（60°）
    phase_shift = np.pi / 3  # P300相关的相位偏移量
    phase_mask = np.where(t < stim_time, 0, phase_shift)

    # 带相位变化的基础节律
    theta = theta_amp * np.sin(2 * np.pi * theta_freq * t + phase_mask)
    alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t + phase_mask)
    beta = beta_amp * np.sin(2 * np.pi * beta_freq * t + phase_mask)

    # 生成P300成分（通道间振幅随机）
    p300_amp = np.random.uniform(2.0, 5.0)  # P300振幅（比基础节律大）
    p300_wave = generate_p300_waveform(t, p300_peak_time, amplitude=p300_amp)

    # 背景噪声
    noise = np.random.normal(0, 0.8, n_samples)

    # 组合最终信号：基础节律 + P300 + 噪声
    eeg_data[ch] = theta + alpha + beta + p300_wave + noise

# ===================== 3. 希尔伯特变换 =====================
# 对每个通道沿时间轴做希尔伯特变换，得到解析信号
analytic_signal = hilbert(eeg_data, axis=1)
hilbert_transform = np.imag(analytic_signal)  # 希尔伯特变换结果（虚部）
envelope = np.abs(analytic_signal)  # 振幅包络
instant_phase = np.angle(analytic_signal)  # 瞬时相位

# 修复瞬时频率计算的维度问题（避免广播错误）
instant_freq = np.zeros_like(instant_phase)
for ch in range(n_channels):
    # 相位差分计算瞬时频率，补全第一个点
    phase_diff = np.diff(instant_phase[ch])
    instant_freq[ch, :-1] = phase_diff * fs / (2 * np.pi)
    instant_freq[ch, -1] = instant_freq[ch, -2]  # 最后一个点用前一个值填充

# ===================== 4. 可视化（突出P300和相位变化） =====================
plt.figure(figsize=(18, 14))
selected_channels = [0, 15, 30, 45]  # 选4个代表性通道

# 子图1：原始EEG信号（含P300标注）
plt.subplot(3, 3, 1)
for i, ch in enumerate(selected_channels):
    plt.plot(t, eeg_data[ch] + i * 8, label=f'通道{ch}', linewidth=1.5)
# 标注刺激时间和P300峰值时间
plt.axvline(x=stim_time, color='red', linestyle='--', alpha=0.7, label='刺激呈现（1.0秒）')
plt.axvline(x=p300_peak_time, color='orange', linestyle=':', alpha=0.7, label='P300峰值（1.3秒）')
plt.title('含P300的原始脑电信号（选定通道）', fontsize=12)
plt.xlabel('时间（秒）', fontsize=10)
plt.ylabel('振幅（偏移以区分通道）', fontsize=10)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)

# 子图2：原始EEG热力图（标注P300区域）
plt.subplot(3, 3, 2)
im1 = plt.imshow(eeg_data, aspect='auto', extent=[0, 2, 0, 64], cmap='viridis')
plt.colorbar(im1, label='振幅')
plt.axvline(x=stim_time, color='red', linestyle='--', alpha=0.7)
plt.axvline(x=p300_peak_time, color='orange', linestyle=':', alpha=0.7)
plt.title('原始脑电数据热力图（P300区域）', fontsize=12)
plt.xlabel('时间（秒）', fontsize=10)
plt.ylabel('通道编号', fontsize=10)

# 子图3：单通道P300细节（通道0）
plt.subplot(3, 3, 3)
ch0_data = eeg_data[0]
plt.plot(t, ch0_data, color='blue', linewidth=1.5, label='通道0原始信号')
# 提取P300时间段（1.0-1.6秒）放大
p300_window = (t >= 1.0) & (t <= 1.6)
plt.fill_between(t[p300_window], ch0_data[p300_window], alpha=0.3, color='orange', label='P300时间窗')
plt.axvline(x=p300_peak_time, color='orange', linestyle=':', alpha=0.7, label='P300峰值')
plt.title('P300波形细节（通道0）', fontsize=12)
plt.xlabel('时间（秒）', fontsize=10)
plt.ylabel('振幅', fontsize=10)
plt.legend()
plt.grid(alpha=0.3)

# 子图4：希尔伯特变换结果（时域）
plt.subplot(3, 3, 4)
for i, ch in enumerate(selected_channels):
    plt.plot(t, hilbert_transform[ch] + i * 8, label=f'通道{ch}', linewidth=1.5)
plt.axvline(x=stim_time, color='red', linestyle='--', alpha=0.7)
plt.axvline(x=p300_peak_time, color='orange', linestyle=':', alpha=0.7)
plt.title('希尔伯特变换结果（时域）', fontsize=12)
plt.xlabel('时间（秒）', fontsize=10)
plt.ylabel('振幅（偏移以区分通道）', fontsize=10)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)

# 子图5：希尔伯特变换热力图
plt.subplot(3, 3, 5)
im2 = plt.imshow(hilbert_transform, aspect='auto', extent=[0, 2, 0, 64], cmap='plasma')
plt.colorbar(im2, label='振幅')
plt.axvline(x=stim_time, color='red', linestyle='--', alpha=0.7)
plt.axvline(x=p300_peak_time, color='orange', linestyle=':', alpha=0.7)
plt.title('希尔伯特变换结果热力图', fontsize=12)
plt.xlabel('时间（秒）', fontsize=10)
plt.ylabel('通道编号', fontsize=10)

# 子图6：瞬时相位（突出P300相位变化）
plt.subplot(3, 3, 6)
for i, ch in enumerate(selected_channels):
    plt.plot(t, instant_phase[ch] + i * 4, label=f'通道{ch}', linewidth=1.5)
plt.axvline(x=stim_time, color='red', linestyle='--', alpha=0.7, label='刺激呈现（相位偏移）')
plt.title('瞬时相位（P300诱导的相位偏移）', fontsize=12)
plt.xlabel('时间（秒）', fontsize=10)
plt.ylabel('相位（弧度，偏移以区分通道）', fontsize=10)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)

# 子图7：原始信号 + P300包络
plt.subplot(3, 3, 7)
for i, ch in enumerate(selected_channels):
    plt.plot(t, eeg_data[ch] + i * 8, color='gray', alpha=0.6, label=f'通道{ch}原始信号')
    plt.plot(t, envelope[ch] + i * 8, 'r--', linewidth=2, label=f'通道{ch}振幅包络')
plt.axvline(x=p300_peak_time, color='orange', linestyle=':', alpha=0.7)
plt.title('原始信号 + P300振幅包络', fontsize=12)
plt.xlabel('时间（秒）', fontsize=10)
plt.ylabel('振幅（偏移以区分通道）', fontsize=10)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)

# 子图8：瞬时频率（P300相关频率变化）
plt.subplot(3, 3, 8)
for i, ch in enumerate(selected_channels):
    plt.plot(t, instant_freq[ch] + i * 5, label=f'通道{ch}', linewidth=1.5)
plt.axvline(x=stim_time, color='red', linestyle='--', alpha=0.7)
plt.title('瞬时频率（P300效应）', fontsize=12)
plt.xlabel('时间（秒）', fontsize=10)
plt.ylabel('频率（赫兹，偏移以区分通道）', fontsize=10)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)

# 子图9：P300包络对比（通道0）
plt.subplot(3, 3, 9)
ch0_envelope = envelope[0]
ch0_raw = eeg_data[0]
plt.plot(t, ch0_raw, color='blue', alpha=0.6, label='通道0原始信号')
plt.plot(t, ch0_envelope, 'r--', linewidth=2, label='通道0振幅包络')
plt.fill_between(t[p300_window], ch0_envelope[p300_window], alpha=0.3, color='red', label='P300包络区域')
plt.axvline(x=p300_peak_time, color='orange', linestyle=':', alpha=0.7)
plt.title('P300振幅包络（通道0）', fontsize=12)
plt.xlabel('时间（秒）', fontsize=10)
plt.ylabel('振幅', fontsize=10)
plt.legend()
plt.grid(alpha=0.3)

# 调整布局，避免重叠
plt.tight_layout()
plt.show()

# ===================== 结果输出 =====================
print("=== 含P300的脑电希尔伯特变换分析 ===")
print(f"原始数据维度：{eeg_data.shape} (通道数 × 采样点数)")
print(f"刺激呈现时间：{stim_time}秒 | P300峰值时间：{p300_peak_time}秒")
print(f"P300峰值采样点：{int(p300_peak_time * fs)} (采样频率{fs}Hz)")
print("\n关键特征：")
print("1. 刺激后300毫秒出现P300正相波（橙色点线标注）")
print("2. 刺激后相位发生明显偏移（瞬时相位图中1.0秒处可见突变）")
print("3. 希尔伯特包络可清晰捕捉P300的振幅峰值特征")
print("4. 瞬时频率在P300出现区域有小幅波动（与认知加工相关）")