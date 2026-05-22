import os
import json
import numpy as np
import pandas as pd
import mne
import matplotlib

# 强制使用非交互式后端，避免GUI问题
matplotlib.use('Agg')  # 关键修复：使用Agg后端，不显示图形
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 文件路径设置
bids_root = r'D:\Won2022_BIDS'
subject_id = '001'
task_name = 'RSVPtask'
run_id = '3'

# 构建完整的文件路径
files = {
    'eeg_set': os.path.join(bids_root, f'sub-{subject_id}', 'eeg',
                            f'sub-{subject_id}_task-{task_name}_run-{run_id}_eeg.set'),
    'eeg_fdt': os.path.join(bids_root, f'sub-{subject_id}', 'eeg',
                            f'sub-{subject_id}_task-{task_name}_run-{run_id}_eeg.fdt'),
    'eeg_json': os.path.join(bids_root, f'sub-{subject_id}', 'eeg',
                             f'sub-{subject_id}_task-{task_name}_run-{run_id}_eeg.json'),
    'channels_tsv': os.path.join(bids_root, f'sub-{subject_id}', 'eeg',
                                 f'sub-{subject_id}_task-{task_name}_run-{run_id}_channels.tsv'),
    'electrodes_tsv': os.path.join(bids_root, f'sub-{subject_id}', 'eeg',
                                   f'sub-{subject_id}_task-{task_name}_run-{run_id}_electrodes.tsv'),
    'coords_json': os.path.join(bids_root, f'sub-{subject_id}', 'eeg',
                                f'sub-{subject_id}_task-{task_name}_run-{run_id}_coordsystem.json')
}

# 修复1：检查文件路径，注意coordsystem.json的文件名
print("=" * 60)
print("检查BIDS格式文件是否存在：")
print("=" * 60)

# 修正coordsystem.json的文件名（根据您的图片）
coords_file = os.path.join(bids_root, f'sub-{subject_id}', 'eeg',
                           f'sub-{subject_id}_task-{task_name}_run-{run_id}_coordsystem.json')
files['coords_json'] = coords_file  # 更新为正确的文件名

for file_type, file_path in files.items():
    exists = os.path.exists(file_path)
    status = "✓ 存在" if exists else "✗ 不存在"
    print(f"{file_type:20} {status}")
    if exists:
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb > 1000:
            print(f"                   大小: {size_kb / 1024:.1f} MB")
        else:
            print(f"                   大小: {size_kb:.1f} KB")

# 2. 读取并显示BIDS元数据
print("\n" + "=" * 60)
print("读取BIDS元数据文件：")
print("=" * 60)

# 2.1 读取EEG JSON文件
if os.path.exists(files['eeg_json']):
    with open(files['eeg_json'], 'r', encoding='utf-8') as f:
        eeg_info = json.load(f)
    print("EEG JSON 文件内容：")
    for key, value in eeg_info.items():
        print(f"  {key}: {value}")
else:
    print("警告：EEG JSON文件不存在")

# 2.2 读取通道信息
if os.path.exists(files['channels_tsv']):
    channels_df = pd.read_csv(files['channels_tsv'], sep='\t')
    print(f"\n通道信息 (共 {len(channels_df)} 个通道)：")
    print(channels_df.head())
    print(f"列名: {list(channels_df.columns)}")
else:
    print("警告：通道TSV文件不存在")

# 2.3 读取电极位置信息
if os.path.exists(files['electrodes_tsv']):
    electrodes_df = pd.read_csv(files['electrodes_tsv'], sep='\t')
    print(f"\n电极位置信息：")
    print(electrodes_df.head())
else:
    print("警告：电极TSV文件不存在")

# 2.4 读取坐标系统信息
if os.path.exists(files['coords_json']):
    with open(files['coords_json'], 'r', encoding='utf-8') as f:
        coords_info = json.load(f)
    print(f"\n坐标系统信息：")
    for key, value in coords_info.items():
        print(f"  {key}: {value}")
else:
    print("警告：坐标系统JSON文件不存在")

# 3. 读取EEG数据
print("\n" + "=" * 60)
print("读取EEG原始数据：")
print("=" * 60)

if os.path.exists(files['eeg_set']):
    try:
        # 使用MNE读取EEGLAB格式数据
        print(f"正在读取EEGLAB格式数据: {files['eeg_set']}")
        raw = mne.io.read_raw_eeglab(files['eeg_set'], preload=True)

        # 显示基本信息
        print("\n✓ EEG数据读取成功！")
        print(f"数据形状: {raw.get_data().shape}")
        print(f"  通道数: {len(raw.ch_names)}")
        print(f"  时间点数: {raw.n_times}")
        print(f"  采样频率: {raw.info['sfreq']} Hz")
        print(f"  总时长: {raw.times[-1]:.2f} 秒")

        # 获取前10个通道的前20个时间点的数据
        data, times = raw[:10, :20]

        print(f"\n=== 前10个通道的前20个时间点 ===")
        for i in range(min(10, len(raw.ch_names))):
            channel_name = raw.ch_names[i]
            channel_data = data[i, :]
            print(f"{channel_name:10}: ", end="")
            for val in channel_data:
                print(f"{val:8.3f}", end=" ")
            print()

        # 4. 保存图表到文件（而不是显示）
        print("\n" + "=" * 60)
        print("生成数据可视化并保存为文件：")
        print("=" * 60)

        # 获取更多数据用于可视化
        all_data, all_times = raw[:10, :1000]  # 前10通道，前1000时间点

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 4.1 时域波形
        ax1 = axes[0, 0]
        for i in range(min(5, len(raw.ch_names))):
            ax1.plot(all_times[:100], all_data[i, :100], label=raw.ch_names[i], linewidth=1)
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('幅值 (µV)')
        ax1.set_title('前5个通道的时域波形')
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(True, alpha=0.3)

        # 4.2 通道平均值
        ax2 = axes[0, 1]
        all_channel_data, _ = raw[:, :]
        channel_means = all_channel_data.mean(axis=1)
        channel_stds = all_channel_data.std(axis=1)
        channels = np.arange(len(raw.ch_names))
        ax2.errorbar(channels[:20], channel_means[:20], yerr=channel_stds[:20],
                     fmt='o', capsize=5, capthick=2, alpha=0.7)
        ax2.set_xlabel('通道索引')
        ax2.set_ylabel('平均幅值 (µV)')
        ax2.set_title('前20个通道的平均幅值')
        ax2.grid(True, alpha=0.3)

        # 4.3 数据分布
        ax3 = axes[1, 0]
        ax3.hist(all_channel_data.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('幅值 (µV)')
        ax3.set_ylabel('频数')
        ax3.set_title('EEG数据幅值分布')
        ax3.grid(True, alpha=0.3)

        # 4.4 简单通道功率谱
        ax4 = axes[1, 1]
        # 计算第一个通道的功率谱
        from scipy import signal

        channel_data = all_channel_data[0, :]
        f, Pxx = signal.welch(channel_data, fs=raw.info['sfreq'], nperseg=1024)
        ax4.semilogy(f, Pxx)
        ax4.set_xlabel('频率 (Hz)')
        ax4.set_ylabel('功率谱密度')
        ax4.set_title(f'通道 {raw.ch_names[0]} 的功率谱')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 50])  # 显示0-50Hz

        plt.tight_layout()

        # 保存图表到文件
        output_path = os.path.join(bids_root, f'sub-{subject_id}', 'eeg_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存到: {output_path}")

        # 不要调用plt.show()，因为使用了Agg后端

        # 5. 保存文本摘要
        print("\n" + "=" * 60)
        print("数据摘要：")
        print("=" * 60)

        # 创建摘要文件
        summary_path = os.path.join(bids_root, f'sub-{subject_id}', 'eeg_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EEG数据分析摘要\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"被试: sub-{subject_id}\n")
            f.write(f"任务: {task_name}\n")
            f.write(f"Run: {run_id}\n\n")

            f.write("数据基本信息:\n")
            f.write(f"  通道数: {len(raw.ch_names)}\n")
            f.write(f"  时间点数: {raw.n_times}\n")
            f.write(f"  采样频率: {raw.info['sfreq']} Hz\n")
            f.write(f"  总时长: {raw.times[-1]:.2f} 秒\n\n")

            f.write("前10个通道:\n")
            for i, ch in enumerate(raw.ch_names[:10]):
                f.write(f"  {i + 1:2}. {ch}\n")

            f.write("\n数据统计:\n")
            f.write(f"  最小值: {all_channel_data.min():.6f} µV\n")
            f.write(f"  最大值: {all_channel_data.max():.6f} µV\n")
            f.write(f"  平均值: {all_channel_data.mean():.6f} µV\n")
            f.write(f"  标准差: {all_channel_data.std():.6f} µV\n")

        print(f"✓ 摘要已保存到: {summary_path}")

        # 打印前3个通道的详细统计
        print("\n=== 前3个通道的详细统计 ===")
        for i in range(min(3, len(raw.ch_names))):
            ch_data = all_channel_data[i, :]
            print(f"{raw.ch_names[i]:10}: "
                  f"min={ch_data.min():7.3f}µV, "
                  f"max={ch_data.max():7.3f}µV, "
                  f"mean={ch_data.mean():7.3f}µV, "
                  f"std={ch_data.std():7.3f}µV")

    except Exception as e:
        print(f"读取EEG数据时出错: {e}")
        import traceback

        traceback.print_exc()
else:
    print("错误：EEG SET文件不存在，无法读取数据")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)