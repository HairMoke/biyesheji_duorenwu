
# 画脑电电极
#
# import mne
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 使用 mne 内置的标准电极布局 (比如 'biosemi32')
# montage = mne.channels.make_standard_montage('biosemi64')
#
# # 创建一个 info 对象，指定采样频率和电极类型
# info = mne.create_info(montage.ch_names, sfreq=250, ch_types='eeg')
#
# # 创建 RawArray 对象
# raw_data = np.random.randn(len(montage.ch_names), 1000)  # 随机数据
# raw = mne.io.RawArray(raw_data, info)
#
# # 将标准电极布局应用到 Raw 数据中
# raw.set_montage(montage)
#
# # 绘制电极位置图
# fig, ax = plt.subplots(figsize=(8, 8))
#
# # 使用 plot_sensors 绘制电极位置图
# raw.plot_sensors(show_names=True, axes=ax, show=False)
#
# # 美化图形
# ax.set_title('EEG Electrode Locations (BioSemi64)', fontsize=16, weight='bold')  # 设置标题
# ax.set_facecolor('white')  # 设置背景颜色为白色
# ax.set_xticks([])  # 隐藏x轴
# ax.set_yticks([])  # 隐藏y轴
#
# # 设置电极标记字体大小
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(12)
#     label.set_weight('bold')
#
# # 去除边框
# for spine in ax.spines.values():
#     spine.set_visible(False)
#
# # 展示图像
# plt.tight_layout()
# plt.show()
#

#################################################
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm
#
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False
#
# # -------------------------------------------------
# # 1. 读取数据（用你的路径）
# # -------------------------------------------------
# file_path = r"D:\ZHB_BiYeSheJi\Dataset\THU\S01\x_train.npy"
# sample_rate = 256
# data = np.load(file_path)          # (trials, channels, time)
# n_trials, n_channels, n_timepoints = data.shape
# time_axis = np.arange(n_timepoints) / sample_rate * 1000   # ms
#
# print(f"数据形状: {data.shape}")
# print(f"时间窗: {time_axis[0]:.0f} ~ {time_axis[-1]:.0f} ms")
#
# # -------------------------------------------------
# # 2. 原来 4 幅图（函数直接搬过来，略）
# # -------------------------------------------------
# def plot_original_four_figs(data, time_axis):
#     # 2.1 单试次多通道
#     trial_idx = 0
#     plt.figure(figsize=(12, 6))
#     offset = np.arange(n_channels) * 5
#     for ch in range(n_channels):
#         plt.plot(time_axis, data[trial_idx, ch] + offset[ch],
#                  lw=1, color='k', alpha=.7)
#     plt.axvline(0, color='r', ls='--')
#     plt.title(f"试次 {trial_idx}  64 通道波形（带偏移）")
#     plt.xlabel("时间 (ms)")
#     plt.ylabel("电位 (μV + 偏移)")
#     plt.show()
#
#     # 2.2 通道×时间热力图
#     mean_trial = data[:100].mean(0)
#     plt.figure(figsize=(12, 4))
#     norm = TwoSlopeNorm(vcenter=0)
#     im = plt.imshow(mean_trial, cmap='coolwarm', norm=norm, aspect='auto',
#                     extent=[time_axis[0], time_axis[-1], n_channels-1, 0])
#     plt.colorbar(im, label='μV')
#     plt.axvline(0, color='k', ls='--')
#     plt.title("前 100 次平均：通道×时间")
#     plt.xlabel("时间 (ms)")
#     plt.ylabel("通道索引")
#     plt.show()
#
#     # 2.3 ERP
#     key_ch = [0, 10, 20]
#     plt.figure(figsize=(8, 4))
#     for ch in key_ch:
#         erp = data[:, ch].mean(0)
#         plt.plot(time_axis, erp, label=f'Ch{ch}')
#     plt.axvline(0, color='r', ls='--')
#     plt.axhline(0, color='k', alpha=.3)
#     plt.title("关键通道 ERP")
#     plt.xlabel("时间 (ms)")
#     plt.ylabel("μV")
#     plt.legend()
#     plt.grid(alpha=.3)
#     plt.show()
#
#     # 2.4 scalp-topo（300 ms）
#     t300 = np.argmin(np.abs(time_axis - 300))
#     plt.figure(figsize=(6, 4))
#     plt.bar(range(n_channels), data[:, :, t300].mean(0))
#     plt.title("刺激后 300 ms 各通道平均电位")
#     plt.xlabel("通道")
#     plt.ylabel("μV")
#     plt.grid(alpha=.3, axis='y')
#     plt.show()
#
# plot_original_four_figs(data, time_axis)
#
# # -------------------------------------------------
# # 3. 对 ERP 做傅里叶变换并画幅度谱
# # -------------------------------------------------
# # 3.1 把“全试次平均”再平均一次，得到一条 1-D 时间序列
# global_erp = data.mean(axis=(0, 1))      # shape = (n_timepoints,)
#
# # 3.2 FFT
# n = len(global_erp)
# fft_vals = np.fft.rfft(global_erp)          # 单边谱
# fft_amp  = np.abs(fft_vals) / n * 2         # 幅度归一化
# freqs    = np.fft.rfftfreq(n, d=1/sample_rate)
#
# # 3.3 画图
# plt.figure(figsize=(7, 4))
# plt.stem(freqs, fft_amp, basefmt=' ')
# plt.xlim(0, sample_rate/2)
# plt.xlabel("频率 (Hz)")
# plt.ylabel("幅度 (μV)")
# plt.title("平均 ERP 的傅里叶幅度谱")
# plt.grid(alpha=.3)
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号（-）显示为方块的问题


# ================= 1. 读取数据 =================
file_path = r"D:\ZHB_BiYeSheJi\Dataset\THU\S01\x_train.npy"
sample_rate = 256                       # Hz
data = np.load(file_path)               # shape: (trials, channels, time)
n_trials, n_channels, n_time = data.shape
print(f"数据加载完成：{n_trials} 试次 × {n_channels} 通道 × {n_time} 时间点")

# ================= 2. 计算平均 ERP =============
global_erp = data.mean(axis=(0, 1))     # 一条平均时间序列 (n_time,)

# ================= 3. FFT + 0-50 Hz ============
fft_vals = np.fft.rfft(global_erp)              # 单边复数谱
fft_amp  = np.abs(fft_vals) / n_time * 2        # 幅度归一化
freqs    = np.fft.rfftfreq(n_time, d=1/sample_rate)

mask = freqs <= 50
freqs, fft_amp = freqs[mask], fft_amp[mask]

# 功率谱密度 → dB
psd     = (fft_amp ** 2) / (sample_rate / 2)    # μV²/Hz
psd_db  = 10 * np.log10(psd + 1e-12)            # 加小常数防 log(0)
# ================= 4. 学术风画图（终极修复） ================
fig, ax = plt.subplots(figsize=(7, 3))

# 1) 主曲线
ax.plot(freqs, psd_db, color='k', lw=1.2, label='Grand-average ERP')

# 2) 填充
x = np.concatenate([freqs, freqs[::-1]])
y = np.concatenate([psd_db, np.full_like(freqs, -200)])
ax.fill(x, y, color='k', alpha=0.15, lw=0)

# 3) 每 5 Hz 一条虚线（0-50 Hz）
for f in range(0, 51, 5):
    # 仅给 5 Hz、10 Hz 加图例标签，其余空字符串
    label = '5 Hz' if f == 5 else '10 Hz' if f == 10 else ""
    ax.axvline(f, color='gray', ls='--', lw=0.8, alpha=0.7, label=label)

# 4) 坐标轴
ax.set_xlim(0, 50)
ax.set_ylim(bottom=np.floor(psd_db.min() / 5) * 5 - 5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (dB μV²/Hz)')
ax.grid(alpha=0.2, linewidth=0.3)
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
plt.show()

# ================= 4. 彩色学术风（无分段 fill_between） ======
cmap = plt.cm.plasma
norm = plt.Normalize(0, 50)

fig, ax = plt.subplots(figsize=(7, 3))

# 主曲线：一条彩色线（逐点颜色）
for i in range(len(freqs)-1):
    ax.plot(freqs[i:i+2], psd_db[i:i+2],
            color=cmap(norm(freqs[i])), lw=2)

# 填充：一次性 polygon，避开 contiguous_regions
# 构建闭合多边形顶点
x_poly = np.concatenate([freqs, freqs[::-1]])
y_poly = np.concatenate([psd_db, np.full_like(freqs, -200)])
# 用 plt.fill 直接画，不经过 fill_between
ax.fill(x_poly, y_poly, color=cmap(0.5), alpha=0.25, lw=0)

# 参考线：5 Hz/10 Hz 高亮
for f in range(0, 51, 5):
    if f in (5, 10):
        ax.axvline(f, color=cmap(norm(f)), ls='--', lw=1.5, alpha=0.9,
                   label=f'{f} Hz' if f == 5 else '')
    else:
        ax.axvline(f, color='gray', ls='--', lw=0.8, alpha=0.4)

ax.set_xlim(0, 50)
ax.set_ylim(bottom=np.floor(psd_db.min() / 5) * 5 - 5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (dB μV^2/Hz)')
ax.grid(alpha=0.2)
ax.legend(frameon=False, fontsize=9)
fig.colorbar(plt.cm.ScalarMappable(norm, cmap),
             ax=ax, label='Frequency (Hz)', ticks=np.arange(0, 51, 10))
fig.tight_layout()
plt.show()

# ================= 5. 彩色「原始」脑电时序图 =================
plt.figure(figsize=(12, 6))
time_ms = np.arange(n_time) / sample_rate * 1000
offset  = np.arange(n_channels) * 5
trial_idx = 0

# 每条通道用 viridis 彩色
viridis = plt.cm.viridis
c_norm = plt.Normalize(0, n_channels-1)

for ch in range(n_channels):
    plt.plot(time_ms, data[trial_idx, ch, :] + offset[ch],
             lw=1, color=viridis(c_norm(ch)), alpha=0.8)
plt.axvline(0, color='w', ls='--', lw=2, label='刺激 onset')
plt.title(f'试次 {trial_idx}（原始）64 通道脑电（彩色）')
plt.xlabel('时间 (ms)')
plt.ylabel('电位 (μV + 通道偏移)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# -------------- ② 确保掩码变量存在 ----------
# 放在「原始」图之后、「掩码后」图之前，千万别删
fft_spec = np.fft.rfft(global_erp)
freqs_full = np.fft.rfftfreq(n_time, d=1/sample_rate)
mu, sigma = 2.5, 1.0
gauss_mask = 1 - np.exp(-0.5 * ((freqs_full - mu) / sigma) ** 2)
fft_spec_masked = fft_spec * gauss_mask
global_erp_masked = np.fft.irfft(fft_spec_masked, n=n_time)

# ================= 6. 彩色「0–5 Hz 掩码后」脑电时序图 =========
plt.figure(figsize=(12, 6))
time_ms = np.arange(n_time) / sample_rate * 1000
offset = np.arange(n_channels) * 5
for ch in range(n_channels):
    plt.plot(time_ms, global_erp_masked + offset[ch],
             lw=1, color=viridis(c_norm(ch)), alpha=0.8)
plt.axvline(0, color='w', ls='--', lw=2, label='刺激 onset')
plt.title(f'试次 {trial_idx}（0–5 Hz 掩码后）64 通道脑电（彩色）')
plt.xlabel('时间 (ms)')
plt.ylabel('电位 (μV + 通道偏移)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# =========================================================
# ① 45-50 Hz 高斯掩码（复用前面 FFT）
# =========================================================
mu45, sigma45 = 47.5, 1.0
gauss_mask45 = 1 - np.exp(-0.5 * ((freqs_full - mu45) / sigma45) ** 2)
fft_spec_masked45 = fft_spec * gauss_mask45
global_erp_masked45 = np.fft.irfft(fft_spec_masked45, n=n_time)

# =========================================================
# ② 45-50 Hz 掩码后频谱图（彩色）
# =========================================================
# 重新计算掩码后的功率谱
fft_amp45 = np.abs(fft_spec_masked45[:len(freqs)]) / n_time * 2
psd45 = (fft_amp45 ** 2) / (sample_rate / 2)
psd_db45 = 10 * np.log10(psd45 + 1e-12)

cmap = plt.cm.plasma
norm = plt.Normalize(0, 50)

fig, ax = plt.subplots(figsize=(7, 3))
# 主曲线：彩色渐变
for i in range(len(freqs)-1):
    ax.plot(freqs[i:i+2], psd_db45[i:i+2],
            color=cmap(norm(freqs[i])), lw=2)

# 一次性填充
x_poly = np.concatenate([freqs, freqs[::-1]])
y_poly = np.concatenate([psd_db45, np.full_like(freqs, -200)])
ax.fill(x_poly, y_poly, color=cmap(0.5), alpha=0.25, lw=0)

# 参考线：45 Hz/50 Hz 红色高亮，其余淡灰
for f in range(0, 51, 5):
    if f in (45, 50):
        ax.axvline(f, color='red', ls='--', lw=1.5, alpha=0.9,
                   label=f'{f} Hz' if f == 45 else '')
    else:
        ax.axvline(f, color='gray', ls='--', lw=0.8, alpha=0.4)

ax.set_xlim(0, 50)
ax.set_ylim(bottom=np.floor(psd_db45.min() / 5) * 5 - 5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (dB μV^2/Hz)')
ax.grid(alpha=0.2)
ax.legend(frameon=False, fontsize=9)
fig.colorbar(plt.cm.ScalarMappable(norm, cmap),
             ax=ax, label='Frequency (Hz)', ticks=np.arange(0, 51, 10))
fig.tight_layout()
plt.show()

# =========================================================
# ③ 彩色「原始」脑电时序图（复用前面代码即可，已存在）
# =========================================================

# =========================================================
# ④ 彩色「45-50 Hz 掩码后」脑电时序图
# =========================================================
plt.figure(figsize=(12, 6))
time_ms = np.arange(n_time) / sample_rate * 1000
offset = np.arange(n_channels) * 5
trial_idx = 0
for ch in range(n_channels):
    plt.plot(time_ms, global_erp_masked45 + offset[ch],
             lw=1, color=viridis(c_norm(ch)), alpha=0.8)
plt.axvline(0, color='w', ls='--', lw=2, label='刺激 onset')
plt.title(f'试次 {trial_idx}（45–50 Hz 掩码后）64 通道脑电（彩色）')
plt.xlabel('时间 (ms)')
plt.ylabel('电位 (μV + 通道偏移)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()