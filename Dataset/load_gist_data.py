
'''

# GIST数据集处理程序 - 修改版
# 按照THU-RSVP数据集的格式进行处理，便于代码复用
# 按照8:2比例划分训练集和测试集，确保目标试次不重叠

import os
import scipy.io as matReader
import numpy as np
from sklearn import preprocessing
from scipy import signal
from tqdm import tqdm, trange
import gc
import mat73  # 用于读取.mat v7.3格式文件


def scale_data(data):
    """归一化数据 data = [C, T]"""
    # scaler = preprocessing.StandardScaler()
    # data = scaler.fit_transform(data)
    return data


def band_pass_filter(data, freq_low, freq_high, fs):
    """带通滤波 data = [C, T]"""
    # 参考论文中的滤波参数：1-10Hz，4阶巴特沃斯滤波器
    wn = [freq_low * 2 / fs, freq_high * 2 / fs]
    b, a = signal.butter(4, wn, 'bandpass')
    for channel in range(data.shape[0]):
        data[channel, ...] = signal.filtfilt(b, a, data[channel, ...])
    return data


def extract_epochs(data, markers, srate, baseline_ms, frame_ms, apply_baseline=True):
    """
    提取事件相关的epoch
    参数:
        data: [C, T] 原始数据
        markers: 标记数组，目标为1，非目标为2
        srate: 采样率
        baseline_ms: 基线时间范围 [start, end]，单位ms
        frame_ms: 提取时间范围 [start, end]，单位ms
        apply_baseline: 是否进行基线校正
    """
    # 找到目标和非目标事件的索引
    target_indices = np.where(markers == 1)[0]
    nontarget_indices = np.where(markers == 2)[0]

    # 转换为样本点
    baseline_samples = [int(baseline_ms[0] * srate / 1000), int(baseline_ms[1] * srate / 1000)]
    frame_samples = [int(frame_ms[0] * srate / 1000), int(frame_ms[1] * srate / 1000)]

    # 计算每个epoch的长度
    epoch_length = frame_samples[1] - frame_samples[0]

    # 分别提取目标和刺激，确保顺序一致
    target_epochs = []
    nontarget_epochs = []

    # 提取目标事件
    for idx in target_indices:
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
                    baseline = np.mean(data[:, baseline_start:baseline_end], axis=1, keepdims=True)
                    epoch = epoch - baseline

            target_epochs.append(epoch)

    # 提取非目标事件
    for idx in nontarget_indices:
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
                    baseline = np.mean(data[:, baseline_start:baseline_end], axis=1, keepdims=True)
                    epoch = epoch - baseline

            nontarget_epochs.append(epoch)

    return np.array(target_epochs), np.array(nontarget_epochs)


def subject_process(rawdata, sub_id, apply_filter=True, train_ratio=0.8):
    """
    以subject为单位进行预处理
    参数:
        rawdata: 从.mat文件加载的原始数据
        sub_id: 被试ID
        apply_filter: 是否应用滤波
        train_ratio: 训练集比例
    """
    # 提取数据
    cur_EEG = rawdata['RSVP']
    data = np.asarray(cur_EEG['data'])  # 形状: [32, T]
    srate = cur_EEG['srate']  # 512 Hz
    markers = cur_EEG['markers_target']

    print(f"\nProcessing subject {sub_id}: data shape = {data.shape}, srate = {srate}")

    # 检查并确保数据维度正确
    if len(data.shape) > 2:
        data = np.squeeze(data)

    # 转置数据，使形状为 [channels, time]
    if data.shape[0] > data.shape[1]:
        data = data.T

    # 参考论文中的预处理：1-10Hz带通滤波
    if apply_filter:
        print(f"  Applying 1-10Hz bandpass filter...")
        data = band_pass_filter(data, 1, 10, srate)

    # 提取事件epochs
    # 使用论文中的参数：基线[-200, 0]ms，时间窗[-200, 1000]ms
    baseline = [-200, 0]
    frame = [-200, 1000]

    target_epochs, nontarget_epochs = extract_epochs(data, markers, srate, baseline, frame, apply_baseline=True)

    print(f"  Extracted {len(target_epochs)} target epochs, {len(nontarget_epochs)} nontarget epochs")

    # 检查数据量是否符合预期
    if len(target_epochs) != 40:
        print(f"  Warning: Expected 40 target epochs, but got {len(target_epochs)}")

    if len(nontarget_epochs) != 800:
        print(f"  Warning: Expected 800 nontarget epochs, but got {len(nontarget_epochs)}")

    # 原始数据是1200ms（-200到1000ms），采样率512Hz，所以614个点
    # 我们需要降采样到256Hz，并且只保留刺激后0-1000ms的数据（与THU-RSVP保持一致）

    # 降采样到256Hz
    # 原始采样率512Hz -> 目标256Hz，降采样因子=2
    # 但epoch长度是614点（1200ms @ 512Hz），我们需要提取刺激后0-1000ms（512个点）
    # 然后降采样到256个点

    # 首先，找到刺激开始点（0ms对应第200ms处，因为基线是-200到0ms）
    stim_start_sample = int(200 * srate / 1000)  # 200ms处
    post_stim_samples = int(1000 * srate / 1000)  # 1000ms

    # 提取刺激后0-1000ms的数据并降采样
    def process_epochs(epochs):
        """处理epochs: 提取刺激后数据并降采样"""
        processed = []
        for epoch in epochs:
            # 提取刺激后0-1000ms的数据
            post_stim_data = epoch[:, stim_start_sample:stim_start_sample + post_stim_samples]

            # 降采样到256Hz（每2个点取平均值）
            downsampled = np.zeros((post_stim_data.shape[0], 256))
            for i in range(256):
                downsampled[:, i] = (post_stim_data[:, 2 * i] + post_stim_data[:, 2 * i + 1]) / 2
            processed.append(downsampled)

        return np.array(processed)  # 形状: [n_epochs, 32, 256]

    # 处理目标和非目标epochs
    target_data = process_epochs(target_epochs)
    nontarget_data = process_epochs(nontarget_epochs)

    print(f"  After processing - Target shape: {target_data.shape}, Nontarget shape: {nontarget_data.shape}")

    # 按照8:2比例划分训练集和测试集
    # 确保训练集和测试集的目标试次不重叠
    n_target = len(target_data)
    n_nontarget = len(nontarget_data)

    # 计算训练集样本数
    n_train_target = int(n_target * train_ratio)
    n_train_nontarget = int(n_nontarget * train_ratio)

    # 随机打乱数据，确保随机性但可重复
    np.random.seed(sub_id)  # 使用被试ID作为随机种子，确保可重复性

    # 打乱目标数据
    target_indices = np.random.permutation(n_target)
    target_data_shuffled = target_data[target_indices]

    # 打乱非目标数据
    nontarget_indices = np.random.permutation(n_nontarget)
    nontarget_data_shuffled = nontarget_data[nontarget_indices]

    # 划分训练集和测试集
    # 目标数据
    train_target = target_data_shuffled[:n_train_target]
    test_target = target_data_shuffled[n_train_target:]

    # 非目标数据
    train_nontarget = nontarget_data_shuffled[:n_train_nontarget]
    test_nontarget = nontarget_data_shuffled[n_train_nontarget:]

    print(f"  Split - Train: {len(train_target)} targets, {len(train_nontarget)} nontargets")
    print(f"          Test: {len(test_target)} targets, {len(test_nontarget)} nontargets")

    # 创建训练集
    train_data = np.concatenate([train_target, train_nontarget], axis=0)
    train_labels = np.concatenate([
        np.ones(len(train_target)),  # 目标为1
        np.zeros(len(train_nontarget))  # 非目标为0
    ])

    # 创建测试集
    test_data = np.concatenate([test_target, test_nontarget], axis=0)
    test_labels = np.concatenate([
        np.ones(len(test_target)),
        np.zeros(len(test_nontarget))
    ])

    print(f"  Combined - Train: {len(train_data)} samples, Test: {len(test_data)} samples")

    # 平衡训练数据集（使目标和非目标数量相等）
    train_target_idx = np.where(train_labels == 1)[0]
    train_nontarget_idx = np.where(train_labels == 0)[0]

    if len(train_target_idx) > 0 and len(train_nontarget_idx) > 0:
        print(f"  Balancing training set...")
        print(f"    Before balancing - Targets: {len(train_target_idx)}, Nontargets: {len(train_nontarget_idx)}")

        # 确定平衡后的样本数（取较小值）
        n_balanced = min(len(train_target_idx), len(train_nontarget_idx))

        # 如果非目标多于目标，从非目标中随机选择与目标数量相同的样本
        if len(train_nontarget_idx) > len(train_target_idx):
            # 随机选择非目标样本
            selected_nontarget_idx = np.random.choice(
                train_nontarget_idx,
                size=n_balanced,
                replace=False
            )

            # 合并平衡后的数据
            balanced_train_data = np.concatenate([
                train_data[train_target_idx],
                train_data[selected_nontarget_idx]
            ])

            balanced_train_labels = np.concatenate([
                np.ones(n_balanced),  # 目标
                np.zeros(n_balanced)  # 非目标
            ])
        else:
            # 如果目标多于非目标（不太可能）
            selected_target_idx = np.random.choice(
                train_target_idx,
                size=n_balanced,
                replace=False
            )

            # 合并平衡后的数据
            balanced_train_data = np.concatenate([
                train_data[selected_target_idx],
                train_data[train_nontarget_idx]
            ])

            balanced_train_labels = np.concatenate([
                np.ones(n_balanced),  # 目标
                np.zeros(n_balanced)  # 非目标
            ])

        # 打乱平衡后的训练集
        shuffle_idx = np.random.permutation(len(balanced_train_data))
        train_data = balanced_train_data[shuffle_idx]
        train_labels = balanced_train_labels[shuffle_idx]

        print(f"    After balancing - Total: {len(train_data)} samples")

    # 补充通道到64个，与THU-RSVP保持一致
    n_channels = train_data.shape[1]
    if n_channels < 64:
        # 创建补充通道（全0）
        sup_channels = 64 - n_channels
        train_data_sup = np.zeros((train_data.shape[0], sup_channels, train_data.shape[2]))
        test_data_sup = np.zeros((test_data.shape[0], sup_channels, test_data.shape[2]))

        train_data = np.concatenate([train_data, train_data_sup], axis=1)
        test_data = np.concatenate([test_data, test_data_sup], axis=1)

    print(f"  Final shapes - Train: {train_data.shape}, Test: {test_data.shape}")

    return train_data, test_data, train_labels, test_labels


def process_gist(raw_path, tar_path, train_ratio=0.8):
    """
    处理GIST数据集
    参数:
        raw_path: 原始数据路径
        tar_path: 目标保存路径
        train_ratio: 训练集比例
    """
    # 获取所有.mat文件
    filenames = [f for f in os.listdir(raw_path) if f.endswith('.mat') and f.startswith('s')]
    filenames.sort()  # 确保按顺序处理

    print(f"Found {len(filenames)} subject files")

    for i, filename in enumerate(tqdm(filenames, desc="Processing subjects")):
        sub_id = i + 1

        # 构建完整的文件路径
        filepath = os.path.join(raw_path, filename)

        try:
            # 加载.mat文件
            print(f"\n{'=' * 60}")
            print(f"Loading {filename}...")

            # 尝试使用mat73加载（支持v7.3格式）
            try:
                mat_data = mat73.loadmat(filepath)
            except:
                # 如果mat73失败，尝试使用scipy.io
                mat_data = matReader.loadmat(filepath)

            # 处理单个被试
            x_train, x_test, y_train, y_test = subject_process(
                mat_data, sub_id, apply_filter=True, train_ratio=train_ratio
            )

            # 创建保存目录
            save_dir = os.path.join(tar_path, f'S{sub_id:>02d}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 保存为.npy文件
            np.save(os.path.join(save_dir, 'x_train.npy'), x_train)
            np.save(os.path.join(save_dir, 'x_test.npy'), x_test)
            np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

            print(f"  Saved subject {sub_id} data to {save_dir}")
            print(f"  Train set: {x_train.shape}, Test set: {x_test.shape}")
            print(f"  Train labels: {np.sum(y_train == 1)} targets, {np.sum(y_train == 0)} nontargets")
            print(f"  Test labels: {np.sum(y_test == 1)} targets, {np.sum(y_test == 0)} nontargets")

            # 清理内存
            del mat_data, x_train, x_test, y_train, y_test
            gc.collect()

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue


# 测试脚本：检查处理后的数据
def check_processed_data(data_path, sub_id=1):
    """检查处理后的数据"""
    print(f"\n{'=' * 80}")
    print(f"检查处理后的数据 - 被试 S{sub_id:>02d}")
    print('=' * 80)

    sub_path = os.path.join(data_path, f'S{sub_id:>02d}')

    if not os.path.exists(sub_path):
        print(f"路径不存在: {sub_path}")
        return

    # 加载数据
    try:
        x_train = np.load(os.path.join(sub_path, 'x_train.npy'))
        x_test = np.load(os.path.join(sub_path, 'x_test.npy'))
        y_train = np.load(os.path.join(sub_path, 'y_train.npy'))
        y_test = np.load(os.path.join(sub_path, 'y_test.npy'))

        # 打印基本信息
        print(f"\n1. 数据维度信息:")
        print(f"   x_train shape: {x_train.shape}")  # [样本数, 通道数, 时间点]
        print(f"   x_test shape: {x_test.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_test shape: {y_test.shape}")

        print(f"\n2. 样本数量统计:")
        print(f"   训练集总样本数: {len(x_train)}")
        print(f"   测试集总样本数: {len(x_test)}")
        print(f"   训练集中目标样本数: {np.sum(y_train == 1)}")
        print(f"   训练集中非目标样本数: {np.sum(y_train == 0)}")
        print(f"   测试集中目标样本数: {np.sum(y_test == 1)}")
        print(f"   测试集中非目标样本数: {np.sum(y_test == 0)}")

        print(f"\n3. 数据统计信息 (x_train):")
        print(f"   最小值: {np.min(x_train):.4f}")
        print(f"   最大值: {np.max(x_train):.4f}")
        print(f"   平均值: {np.mean(x_train):.4f}")
        print(f"   标准差: {np.std(x_train):.4f}")

        print(f"\n4. 标签统计:")
        print(f"   训练集标签分布: 目标={np.sum(y_train == 1)}, 非目标={np.sum(y_train == 0)}")
        print(f"   测试集标签分布: 目标={np.sum(y_test == 1)}, 非目标={np.sum(y_test == 0)}")

        print(f"\n5. 检查是否有重叠:")
        # 注意：由于我们处理的是特征数据，无法直接检查原始试次是否重叠
        # 但我们可以通过统计信息推断
        train_target_ratio = np.sum(y_train == 1) / len(y_train)
        test_target_ratio = np.sum(y_test == 1) / len(y_test)
        print(f"   训练集目标比例: {train_target_ratio:.2%}")
        print(f"   测试集目标比例: {test_target_ratio:.2%}")

        if train_target_ratio > 0.5 or test_target_ratio > 0.5:
            print(f"   警告: 目标比例过高，可能划分不合理")

        return x_train, x_test, y_train, y_test

    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None, None, None


if __name__ == "__main__":
    # 设置路径
    raw_path = r"D:\EEG-dataset-for-RSVP-P300-speller\Python\data"
    tar_path = r"D:\ZHB_BiYeSheJi\Dataset\GIST"

    # 检查路径是否存在
    if not os.path.exists(raw_path):
        print(f"Raw data path does not exist: {raw_path}")
        exit(1)

    # 创建目标目录
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)
        print(f"Created target directory: {tar_path}")

    # 为每个被试创建子目录
    for i in range(1, 56):  # GIST有55个被试
        sub_dir = os.path.join(tar_path, f'S{i:>02d}')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    # 处理数据 (8:2划分)
    print("开始处理GIST数据集，按照8:2比例划分训练集和测试集...")
    process_gist(raw_path, tar_path, train_ratio=0.8)

    print("\n" + "=" * 80)
    print("GIST dataset processing completed!")
    print("=" * 80)

    # 检查处理后的数据
    check_processed_data(tar_path, sub_id=1)

'''


'''

# GIST数据集处理程序 - 修改版
# 按照THU-RSVP数据集的格式进行处理，便于代码复用
# 使用THU-RSVP的滤波参数(1-40Hz)和预处理方法
# 保持32通道，不填充到64通道
# 按照8:2比例划分训练集和测试集，确保目标试次不重叠

import os
import scipy.io as matReader
import numpy as np
from sklearn import preprocessing
from scipy import signal
from tqdm import tqdm, trange
import gc
import mat73  # 用于读取.mat v7.3格式文件


def scale_data(data):
    """归一化数据 data = [C, T] - 参考THU-RSVP的处理方式"""
    # THU-RSVP中实际注释掉了归一化，但保留了函数结构
    # 为了与THU-RSVP保持一致，我们也注释掉归一化
    # scaler = preprocessing.StandardScaler()
    # data = scaler.fit_transform(data)
    return data


def band_pass_filter(data, freq_low, freq_high, fs):
    """带通滤波 data = [C, T] - 参考THU-RSVP的处理方式"""
    # THU-RSVP中使用1-40Hz带通滤波，3阶巴特沃斯滤波器
    # 但实际代码中注释掉了滤波，为了保持一致我们也注释掉
    # wn = [freq_low * 2 / fs, freq_high * 2 / fs]
    # b, a = signal.butter(3, wn, 'bandpass')
    # for channel in range(data.shape[0]):
    #     data[channel, ...] = signal.filtfilt(b, a, data[channel, ...])
    return data


def extract_epochs(data, markers, srate, baseline_ms, frame_ms, apply_baseline=True):
    """
    提取事件相关的epoch
    参数:
        data: [C, T] 原始数据
        markers: 标记数组，目标为1，非目标为2
        srate: 采样率
        baseline_ms: 基线时间范围 [start, end]，单位ms
        frame_ms: 提取时间范围 [start, end]，单位ms
        apply_baseline: 是否进行基线校正
    """
    # 找到目标和非目标事件的索引
    target_indices = np.where(markers == 1)[0]
    nontarget_indices = np.where(markers == 2)[0]

    # 转换为样本点
    baseline_samples = [int(baseline_ms[0] * srate / 1000), int(baseline_ms[1] * srate / 1000)]
    frame_samples = [int(frame_ms[0] * srate / 1000), int(frame_ms[1] * srate / 1000)]

    # 计算每个epoch的长度
    epoch_length = frame_samples[1] - frame_samples[0]

    # 分别提取目标和刺激，确保顺序一致
    target_epochs = []
    nontarget_epochs = []

    # 提取目标事件
    for idx in target_indices:
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
                    baseline = np.mean(data[:, baseline_start:baseline_end], axis=1, keepdims=True)
                    epoch = epoch - baseline

            target_epochs.append(epoch)

    # 提取非目标事件
    for idx in nontarget_indices:
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
                    baseline = np.mean(data[:, baseline_start:baseline_end], axis=1, keepdims=True)
                    epoch = epoch - baseline

            nontarget_epochs.append(epoch)

    return np.array(target_epochs), np.array(nontarget_epochs)


def subject_process(rawdata, sub_id, apply_filter=True, train_ratio=0.8):
    """
    以subject为单位进行预处理
    参数:
        rawdata: 从.mat文件加载的原始数据
        sub_id: 被试ID
        apply_filter: 是否应用滤波（为了与THU-RSVP一致，实际不应用）
        train_ratio: 训练集比例
    """
    # 提取数据
    cur_EEG = rawdata['RSVP']
    data = np.asarray(cur_EEG['data'])  # 形状: [32, T]
    srate = cur_EEG['srate']  # 512 Hz
    markers = cur_EEG['markers_target']

    print(f"\nProcessing subject {sub_id}: data shape = {data.shape}, srate = {srate}")

    # 检查并确保数据维度正确
    if len(data.shape) > 2:
        data = np.squeeze(data)

    # 转置数据，使形状为 [channels, time]
    if data.shape[0] > data.shape[1]:
        data = data.T

    # 参考THU-RSVP的预处理方式：注释掉滤波
    # 这里为了与THU-RSVP完全一致，不应用滤波
    if apply_filter:
        print(f"  Note: Following THU-RSVP, filter is commented out (1-40Hz would be used if enabled)")
        # data = band_pass_filter(data, 1, 40, srate)  # THU-RSVP使用1-40Hz

    # 提取事件epochs
    # 使用论文中的参数：基线[-200, 0]ms，时间窗[-200, 1000]ms
    baseline = [-200, 0]
    frame = [-200, 1000]

    target_epochs, nontarget_epochs = extract_epochs(data, markers, srate, baseline, frame, apply_baseline=True)

    print(f"  Extracted {len(target_epochs)} target epochs, {len(nontarget_epochs)} nontarget epochs")

    # 检查数据量是否符合预期
    if len(target_epochs) != 40:
        print(f"  Warning: Expected 40 target epochs, but got {len(target_epochs)}")

    if len(nontarget_epochs) != 800:
        print(f"  Warning: Expected 800 nontarget epochs, but got {len(nontarget_epochs)}")

    # 原始数据是1200ms（-200到1000ms），采样率512Hz，所以614个点
    # 我们需要降采样到256Hz，并且只保留刺激后0-1000ms的数据（与THU-RSVP保持一致）

    # 降采样到256Hz
    # 原始采样率512Hz -> 目标256Hz，降采样因子=2
    # 但epoch长度是614点（1200ms @ 512Hz），我们需要提取刺激后0-1000ms（512个点）
    # 然后降采样到256个点

    # 首先，找到刺激开始点（0ms对应第200ms处，因为基线是-200到0ms）
    stim_start_sample = int(200 * srate / 1000)  # 200ms处
    post_stim_samples = int(1000 * srate / 1000)  # 1000ms

    # 提取刺激后0-1000ms的数据并降采样
    def process_epochs(epochs):
        """处理epochs: 提取刺激后数据并降采样"""
        processed = []
        for epoch in epochs:
            # 提取刺激后0-1000ms的数据
            post_stim_data = epoch[:, stim_start_sample:stim_start_sample + post_stim_samples]

            # 降采样到256Hz（每2个点取平均值）- 与THU-RSVP相同的降采样方式
            downsampled = np.zeros((post_stim_data.shape[0], 256))
            for i in range(256):
                downsampled[:, i] = (post_stim_data[:, 2 * i] + post_stim_data[:, 2 * i + 1]) / 2
            processed.append(downsampled)

        return np.array(processed)  # 形状: [n_epochs, 32, 256]

    # 处理目标和非目标epochs
    target_data = process_epochs(target_epochs)
    nontarget_data = process_epochs(nontarget_epochs)

    print(f"  After processing - Target shape: {target_data.shape}, Nontarget shape: {nontarget_data.shape}")

    # 按照8:2比例划分训练集和测试集
    # 确保训练集和测试集的目标试次不重叠
    n_target = len(target_data)
    n_nontarget = len(nontarget_data)

    # 计算训练集样本数
    n_train_target = int(n_target * train_ratio)
    n_train_nontarget = int(n_nontarget * train_ratio)

    # 随机打乱数据，确保随机性但可重复
    np.random.seed(sub_id)  # 使用被试ID作为随机种子，确保可重复性

    # 打乱目标数据
    target_indices = np.random.permutation(n_target)
    target_data_shuffled = target_data[target_indices]

    # 打乱非目标数据
    nontarget_indices = np.random.permutation(n_nontarget)
    nontarget_data_shuffled = nontarget_data[nontarget_indices]

    # 划分训练集和测试集
    # 目标数据
    train_target = target_data_shuffled[:n_train_target]
    test_target = target_data_shuffled[n_train_target:]

    # 非目标数据
    train_nontarget = nontarget_data_shuffled[:n_train_nontarget]
    test_nontarget = nontarget_data_shuffled[n_train_nontarget:]

    print(f"  Split - Train: {len(train_target)} targets, {len(train_nontarget)} nontargets")
    print(f"          Test: {len(test_target)} targets, {len(test_nontarget)} nontargets")

    # 创建训练集
    train_data = np.concatenate([train_target, train_nontarget], axis=0)
    train_labels = np.concatenate([
        np.ones(len(train_target)),  # 目标为1
        np.zeros(len(train_nontarget))  # 非目标为0
    ])

    # 创建测试集
    test_data = np.concatenate([test_target, test_nontarget], axis=0)
    test_labels = np.concatenate([
        np.ones(len(test_target)),
        np.zeros(len(test_nontarget))
    ])

    print(f"  Combined - Train: {len(train_data)} samples, Test: {len(test_data)} samples")

    # 平衡训练数据集（使目标和非目标数量相等）- 与THU-RSVP相同的平衡方式
    train_target_idx = np.where(train_labels == 1)[0]
    train_nontarget_idx = np.where(train_labels == 0)[0]

    if len(train_target_idx) > 0 and len(train_nontarget_idx) > 0:
        print(f"  Balancing training set...")
        print(f"    Before balancing - Targets: {len(train_target_idx)}, Nontargets: {len(train_nontarget_idx)}")

        # 确定平衡后的样本数（取较小值）
        n_balanced = min(len(train_target_idx), len(train_nontarget_idx))

        # 如果非目标多于目标，从非目标中随机选择与目标数量相同的样本
        if len(train_nontarget_idx) > len(train_target_idx):
            # 随机选择非目标样本
            selected_nontarget_idx = np.random.choice(
                train_nontarget_idx,
                size=n_balanced,
                replace=False
            )

            # 合并平衡后的数据
            balanced_train_data = np.concatenate([
                train_data[train_target_idx],
                train_data[selected_nontarget_idx]
            ])

            balanced_train_labels = np.concatenate([
                np.ones(n_balanced),  # 目标
                np.zeros(n_balanced)  # 非目标
            ])
        else:
            # 如果目标多于非目标（不太可能）
            selected_target_idx = np.random.choice(
                train_target_idx,
                size=n_balanced,
                replace=False
            )

            # 合并平衡后的数据
            balanced_train_data = np.concatenate([
                train_data[selected_target_idx],
                train_data[train_nontarget_idx]
            ])

            balanced_train_labels = np.concatenate([
                np.ones(n_balanced),  # 目标
                np.zeros(n_balanced)  # 非目标
            ])

        # 打乱平衡后的训练集
        shuffle_idx = np.random.permutation(len(balanced_train_data))
        train_data = balanced_train_data[shuffle_idx]
        train_labels = balanced_train_labels[shuffle_idx]

        print(f"    After balancing - Total: {len(train_data)} samples")

    # 注意：这里不补充通道到64，保持32个通道
    print(f"  Final shapes - Train: {train_data.shape}, Test: {test_data.shape}")
    print(f"  Note: Using original 32 channels, not padded to 64")
    print(f"  Note: Following THU-RSVP processing (1-40Hz filter commented out, no scaling)")

    return train_data, test_data, train_labels, test_labels


def process_gist(raw_path, tar_path, train_ratio=0.8):
    """
    处理GIST数据集
    参数:
        raw_path: 原始数据路径
        tar_path: 目标保存路径
        train_ratio: 训练集比例
    """
    # 获取所有.mat文件
    filenames = [f for f in os.listdir(raw_path) if f.endswith('.mat') and f.startswith('s')]
    filenames.sort()  # 确保按顺序处理

    print(f"Found {len(filenames)} subject files")

    for i, filename in enumerate(tqdm(filenames, desc="Processing subjects")):
        sub_id = i + 1

        # 构建完整的文件路径
        filepath = os.path.join(raw_path, filename)

        try:
            # 加载.mat文件
            print(f"\n{'=' * 60}")
            print(f"Loading {filename}...")

            # 尝试使用mat73加载（支持v7.3格式）
            try:
                mat_data = mat73.loadmat(filepath)
            except:
                # 如果mat73失败，尝试使用scipy.io
                mat_data = matReader.loadmat(filepath)

            # 处理单个被试
            x_train, x_test, y_train, y_test = subject_process(
                mat_data, sub_id, apply_filter=True, train_ratio=train_ratio
            )

            # 创建保存目录
            save_dir = os.path.join(tar_path, f'S{sub_id:>02d}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 保存为.npy文件
            np.save(os.path.join(save_dir, 'x_train.npy'), x_train)
            np.save(os.path.join(save_dir, 'x_test.npy'), x_test)
            np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

            print(f"  Saved subject {sub_id} data to {save_dir}")
            print(f"  Train set: {x_train.shape}, Test set: {x_test.shape}")
            print(f"  Train labels: {np.sum(y_train == 1)} targets, {np.sum(y_train == 0)} nontargets")
            print(f"  Test labels: {np.sum(y_test == 1)} targets, {np.sum(y_test == 0)} nontargets")

            # 清理内存
            del mat_data, x_train, x_test, y_train, y_test
            gc.collect()

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue


# 测试脚本：检查处理后的数据
def check_processed_data(data_path, sub_id=1):
    print(f"\n{'=' * 80}")
    print(f"检查处理后的数据 - 被试 S{sub_id:>02d}")
    print('=' * 80)

    sub_path = os.path.join(data_path, f'S{sub_id:>02d}')

    if not os.path.exists(sub_path):
        print(f"路径不存在: {sub_path}")
        return

    # 加载数据
    try:
        x_train = np.load(os.path.join(sub_path, 'x_train.npy'))
        x_test = np.load(os.path.join(sub_path, 'x_test.npy'))
        y_train = np.load(os.path.join(sub_path, 'y_train.npy'))
        y_test = np.load(os.path.join(sub_path, 'y_test.npy'))

        # 打印基本信息
        print(f"\n1. 数据维度信息:")
        print(f"   x_train shape: {x_train.shape}")  # [样本数, 通道数, 时间点]
        print(f"   x_test shape: {x_test.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_test shape: {y_test.shape}")

        print(f"\n2. 样本数量统计:")
        print(f"   训练集总样本数: {len(x_train)}")
        print(f"   测试集总样本数: {len(x_test)}")
        print(f"   训练集中目标样本数: {np.sum(y_train == 1)}")
        print(f"   训练集中非目标样本数: {np.sum(y_train == 0)}")
        print(f"   测试集中目标样本数: {np.sum(y_test == 1)}")
        print(f"   测试集中非目标样本数: {np.sum(y_test == 0)}")

        print(f"\n3. 数据统计信息 (x_train):")
        print(f"   最小值: {np.min(x_train):.4f}")
        print(f"   最大值: {np.max(x_train):.4f}")
        print(f"   平均值: {np.mean(x_train):.4f}")
        print(f"   标准差: {np.std(x_train):.4f}")

        print(f"\n4. 标签统计:")
        print(f"   训练集标签分布: 目标={np.sum(y_train == 1)}, 非目标={np.sum(y_train == 0)}")
        print(f"   测试集标签分布: 目标={np.sum(y_test == 1)}, 非目标={np.sum(y_test == 0)}")

        print(f"\n5. 检查是否有重叠:")
        # 注意：由于我们处理的是特征数据，无法直接检查原始试次是否重叠
        # 但我们可以通过统计信息推断
        train_target_ratio = np.sum(y_train == 1) / len(y_train)
        test_target_ratio = np.sum(y_test == 1) / len(y_test)
        print(f"   训练集目标比例: {train_target_ratio:.2%}")
        print(f"   测试集目标比例: {test_target_ratio:.2%}")

        if train_target_ratio > 0.5 or test_target_ratio > 0.5:
            print(f"   警告: 目标比例过高，可能划分不合理")

        return x_train, x_test, y_train, y_test

    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None, None, None


if __name__ == "__main__":
    # 设置路径
    raw_path = r"D:\EEG-dataset-for-RSVP-P300-speller\Python\data"
    tar_path = r"D:\ZHB_BiYeSheJi\Dataset\GIST"

    # 检查路径是否存在
    if not os.path.exists(raw_path):
        print(f"Raw data path does not exist: {raw_path}")
        exit(1)

    # 创建目标目录
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)
        print(f"Created target directory: {tar_path}")

    # 为每个被试创建子目录
    for i in range(1, 56):  # GIST有55个被试
        sub_dir = os.path.join(tar_path, f'S{i:>02d}')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    # 处理数据 (8:2划分)
    print("开始处理GIST数据集，按照8:2比例划分训练集和测试集...")
    print("注意: 只使用原始32通道，不填充到64通道")
    print("注意: 使用THU-RSVP的预处理方式（1-40Hz滤波注释掉，无归一化）")
    process_gist(raw_path, tar_path, train_ratio=0.8)

    print("\n" + "=" * 80)
    print("GIST dataset processing completed!")
    print("=" * 80)

    # 检查处理后的数据
    check_processed_data(tar_path, sub_id=1)
'''

# GIST数据集处理程序 - 混合所有被试版本（补充到64通道）
# 将所有55个被试的数据混合后按8:2划分训练集和测试集
# 按照THU-RSVP数据集的格式进行处理，便于代码复用
# 使用THU-RSVP的滤波参数(1-40Hz)和预处理方法
# 将32通道补充到64通道以兼容THU-RSVP代码

import os
import scipy.io as matReader
import numpy as np
from sklearn import preprocessing
from scipy import signal
from tqdm import tqdm, trange
import gc
import mat73  # 用于读取.mat v7.3格式文件


def scale_data(data):
    """归一化数据 data = [C, T] - 参考THU-RSVP的处理方式"""
    # THU-RSVP中实际注释掉了归一化，但保留了函数结构
    # 为了与THU-RSVP保持一致，我们也注释掉归一化
    # scaler = preprocessing.StandardScaler()
    # data = scaler.fit_transform(data)
    return data


def band_pass_filter(data, freq_low, freq_high, fs):
    """带通滤波 data = [C, T] - 参考THU-RSVP的处理方式"""
    # THU-RSVP中使用1-40Hz带通滤波，3阶巴特沃斯滤波器
    # 但实际代码中注释掉了滤波，为了保持一致我们也注释掉
    # wn = [freq_low * 2 / fs, freq_high * 2 / fs]
    # b, a = signal.butter(3, wn, 'bandpass')
    # for channel in range(data.shape[0]):
    #     data[channel, ...] = signal.filtfilt(b, a, data[channel, ...])
    return data


def extract_epochs(data, markers, srate, baseline_ms, frame_ms, apply_baseline=True):
    """
    提取事件相关的epoch
    参数:
        data: [C, T] 原始数据
        markers: 标记数组，目标为1，非目标为2
        srate: 采样率
        baseline_ms: 基线时间范围 [start, end]，单位ms
        frame_ms: 提取时间范围 [start, end]，单位ms
        apply_baseline: 是否进行基线校正
    """
    # 找到目标和非目标事件的索引
    target_indices = np.where(markers == 1)[0]
    nontarget_indices = np.where(markers == 2)[0]

    # 转换为样本点
    baseline_samples = [int(baseline_ms[0] * srate / 1000), int(baseline_ms[1] * srate / 1000)]
    frame_samples = [int(frame_ms[0] * srate / 1000), int(frame_ms[1] * srate / 1000)]

    # 计算每个epoch的长度
    epoch_length = frame_samples[1] - frame_samples[0]

    # 分别提取目标和刺激，确保顺序一致
    target_epochs = []
    nontarget_epochs = []

    # 提取目标事件
    for idx in target_indices:
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
                    baseline = np.mean(data[:, baseline_start:baseline_end], axis=1, keepdims=True)
                    epoch = epoch - baseline

            target_epochs.append(epoch)

    # 提取非目标事件
    for idx in nontarget_indices:
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
                    baseline = np.mean(data[:, baseline_start:baseline_end], axis=1, keepdims=True)
                    epoch = epoch - baseline

            nontarget_epochs.append(epoch)

    return np.array(target_epochs), np.array(nontarget_epochs)


def process_subject_data(rawdata, sub_id, pad_to_64=True):
    """
    处理单个被试的数据，返回目标和非目标数据
    参数:
        rawdata: 从.mat文件加载的原始数据
        sub_id: 被试ID
        pad_to_64: 是否将通道补充到64
    返回:
        target_data: 目标数据 [n_target, 64, 256] (如果pad_to_64=True)
        nontarget_data: 非目标数据 [n_nontarget, 64, 256] (如果pad_to_64=True)
    """
    # 提取数据
    cur_EEG = rawdata['RSVP']
    data = np.asarray(cur_EEG['data'])  # 形状: [32, T]
    srate = cur_EEG['srate']  # 512 Hz
    markers = cur_EEG['markers_target']

    print(f"Processing subject {sub_id}: data shape = {data.shape}, srate = {srate}")

    # 检查并确保数据维度正确
    if len(data.shape) > 2:
        data = np.squeeze(data)

    # 转置数据，使形状为 [channels, time]
    if data.shape[0] > data.shape[1]:
        data = data.T

    # 提取事件epochs
    # 使用论文中的参数：基线[-200, 0]ms，时间窗[-200, 1000]ms
    baseline = [-200, 0]
    frame = [-200, 1000]

    target_epochs, nontarget_epochs = extract_epochs(data, markers, srate, baseline, frame, apply_baseline=True)

    print(f"  Extracted {len(target_epochs)} target epochs, {len(nontarget_epochs)} nontarget epochs")

    # 检查数据量是否符合预期
    if len(target_epochs) != 40:
        print(f"  Warning: Expected 40 target epochs, but got {len(target_epochs)}")

    if len(nontarget_epochs) != 800:
        print(f"  Warning: Expected 800 nontarget epochs, but got {len(nontarget_epochs)}")

    # 原始数据是1200ms（-200到1000ms），采样率512Hz，所以614个点
    # 我们需要降采样到256Hz，并且只保留刺激后0-1000ms的数据

    # 降采样到256Hz
    # 原始采样率512Hz -> 目标256Hz，降采样因子=2
    # 但epoch长度是614点（1200ms @ 512Hz），我们需要提取刺激后0-1000ms（512个点）
    # 然后降采样到256个点

    # 首先，找到刺激开始点（0ms对应第200ms处，因为基线是-200到0ms）
    stim_start_sample = int(200 * srate / 1000)  # 200ms处
    post_stim_samples = int(1000 * srate / 1000)  # 1000ms

    # 提取刺激后0-1000ms的数据并降采样
    def process_epochs(epochs):
        """处理epochs: 提取刺激后数据并降采样"""
        processed = []
        for epoch in epochs:
            # 提取刺激后0-1000ms的数据
            post_stim_data = epoch[:, stim_start_sample:stim_start_sample + post_stim_samples]

            # 降采样到256Hz（每2个点取平均值）
            downsampled = np.zeros((post_stim_data.shape[0], 256))
            for i in range(256):
                downsampled[:, i] = (post_stim_data[:, 2 * i] + post_stim_data[:, 2 * i + 1]) / 2
            processed.append(downsampled)

        return np.array(processed)  # 形状: [n_epochs, 32, 256]

    # 处理目标和非目标epochs
    target_data = process_epochs(target_epochs)  # [n_target, 32, 256]
    nontarget_data = process_epochs(nontarget_epochs)  # [n_nontarget, 32, 256]

    print(f"  After processing - Target shape: {target_data.shape}, Nontarget shape: {nontarget_data.shape}")

    # 将通道补充到64，以便与THU-RSVP兼容
    if pad_to_64:
        n_target = target_data.shape[0]
        n_nontarget = nontarget_data.shape[0]

        # 创建补充通道（全0）
        target_data_padded = np.zeros((n_target, 64, 256))
        nontarget_data_padded = np.zeros((n_nontarget, 64, 256))

        # 将原始32通道数据复制到前32个通道
        target_data_padded[:, :32, :] = target_data
        nontarget_data_padded[:, :32, :] = nontarget_data

        target_data = target_data_padded
        nontarget_data = nontarget_data_padded

        print(
            f"  After padding to 64 channels - Target shape: {target_data.shape}, Nontarget shape: {nontarget_data.shape}")

    return target_data, nontarget_data


def mix_all_subjects_and_split(raw_path, train_ratio=0.8, random_seed=42, pad_to_64=True):
    """
    混合所有被试的数据，然后按比例划分训练集和测试集
    参数:
        raw_path: 原始数据路径
        train_ratio: 训练集比例
        random_seed: 随机种子，确保结果可重复
        pad_to_64: 是否将通道补充到64
    返回:
        x_train, x_test, y_train, y_test
    """
    # 获取所有.mat文件
    filenames = [f for f in os.listdir(raw_path) if f.endswith('.mat') and f.startswith('s')]
    filenames.sort()  # 确保按顺序处理

    print(f"Found {len(filenames)} subject files")
    print("Mixing all subjects' data together...")

    # 初始化列表，用于存储所有被试的数据
    all_target_data = []
    all_nontarget_data = []

    # 处理所有被试
    for i, filename in enumerate(tqdm(filenames, desc="Processing subjects")):
        sub_id = i + 1

        # 构建完整的文件路径
        filepath = os.path.join(raw_path, filename)

        try:
            # 加载.mat文件
            # 尝试使用mat73加载（支持v7.3格式）
            try:
                mat_data = mat73.loadmat(filepath)
            except:
                # 如果mat73失败，尝试使用scipy.io
                mat_data = matReader.loadmat(filepath)

            # 处理单个被试
            target_data, nontarget_data = process_subject_data(mat_data, sub_id, pad_to_64=pad_to_64)

            # 添加到总数据中
            all_target_data.append(target_data)
            all_nontarget_data.append(nontarget_data)

            # 清理内存
            del mat_data, target_data, nontarget_data
            gc.collect()

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # 合并所有被试的数据
    print("\nMerging all subjects' data...")
    all_target_data = np.concatenate(all_target_data, axis=0)
    all_nontarget_data = np.concatenate(all_nontarget_data, axis=0)

    print(f"Total target data shape: {all_target_data.shape}")
    print(f"Total nontarget data shape: {all_nontarget_data.shape}")

    # 创建标签
    target_labels = np.ones(all_target_data.shape[0])  # 目标为1
    nontarget_labels = np.zeros(all_nontarget_data.shape[0])  # 非目标为0

    # 合并所有数据和标签
    all_data = np.concatenate([all_target_data, all_nontarget_data], axis=0)
    all_labels = np.concatenate([target_labels, nontarget_labels], axis=0)

    print(f"Total data shape: {all_data.shape}")
    print(f"Total labels shape: {all_labels.shape}")

    # 设置随机种子，确保可重复性
    np.random.seed(random_seed)

    # 打乱数据
    print("\nShuffling all data...")
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]

    # 按照比例划分训练集和测试集
    print(f"\nSplitting data with train ratio {train_ratio}...")
    split_idx = int(len(all_data) * train_ratio)

    x_train = all_data[:split_idx]
    y_train = all_labels[:split_idx]
    x_test = all_data[split_idx:]
    y_test = all_labels[split_idx:]

    print(f"Train set shape: {x_train.shape}")
    print(f"Test set shape: {x_test.shape}")
    print(f"Train labels - Target: {np.sum(y_train == 1)}, Nontarget: {np.sum(y_train == 0)}")
    print(f"Test labels - Target: {np.sum(y_test == 1)}, Nontarget: {np.sum(y_test == 0)}")

    # 平衡训练集（可选）
    print("\nBalancing training set...")
    train_target_idx = np.where(y_train == 1)[0]
    train_nontarget_idx = np.where(y_train == 0)[0]

    # 确定平衡后的样本数（取较小值）
    n_balanced = min(len(train_target_idx), len(train_nontarget_idx))

    print(f"  Before balancing - Targets: {len(train_target_idx)}, Nontargets: {len(train_nontarget_idx)}")

    # 如果非目标多于目标，从非目标中随机选择与目标数量相同的样本
    if len(train_nontarget_idx) > len(train_target_idx):
        # 随机选择非目标样本
        selected_nontarget_idx = np.random.choice(
            train_nontarget_idx,
            size=n_balanced,
            replace=False
        )

        # 合并平衡后的数据
        balanced_x_train = np.concatenate([
            x_train[train_target_idx],
            x_train[selected_nontarget_idx]
        ])

        balanced_y_train = np.concatenate([
            np.ones(n_balanced),  # 目标
            np.zeros(n_balanced)  # 非目标
        ])
    else:
        # 如果目标多于非目标（不太可能）
        selected_target_idx = np.random.choice(
            train_target_idx,
            size=n_balanced,
            replace=False
        )

        # 合并平衡后的数据
        balanced_x_train = np.concatenate([
            x_train[selected_target_idx],
            x_train[train_nontarget_idx]
        ])

        balanced_y_train = np.concatenate([
            np.ones(n_balanced),  # 目标
            np.zeros(n_balanced)  # 非目标
        ])

    # 打乱平衡后的训练集
    shuffle_idx = np.random.permutation(len(balanced_x_train))
    x_train = balanced_x_train[shuffle_idx]
    y_train = balanced_y_train[shuffle_idx]

    print(f"  After balancing - Train: {len(x_train)} samples")
    print(f"  Final - Train targets: {np.sum(y_train == 1)}, nontargets: {np.sum(y_train == 0)}")
    print(f"  Final - Test targets: {np.sum(y_test == 1)}, nontargets: {np.sum(y_test == 0)}")

    return x_train, x_test, y_train, y_test


def process_gist_mixed(raw_path, tar_path, train_ratio=0.8, random_seed=42, pad_to_64=True):
    """
    处理GIST数据集（混合所有被试版本）
    参数:
        raw_path: 原始数据路径
        tar_path: 目标保存路径
        train_ratio: 训练集比例
        random_seed: 随机种子
        pad_to_64: 是否将通道补充到64
    """
    # 检查路径是否存在
    if not os.path.exists(raw_path):
        print(f"Raw data path does not exist: {raw_path}")
        exit(1)

    # 创建目标目录
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)
        print(f"Created target directory: {tar_path}")

    # 混合所有被试的数据并划分
    print("=" * 80)
    print("开始处理GIST数据集（混合所有被试）")
    print(f"训练集比例: {train_ratio}")
    print(f"随机种子: {random_seed}")
    print(f"补充通道到64: {pad_to_64}")
    print("=" * 80)

    x_train, x_test, y_train, y_test = mix_all_subjects_and_split(
        raw_path, train_ratio=train_ratio, random_seed=random_seed, pad_to_64=pad_to_64
    )

    # 保存数据
    print("\nSaving data...")
    np.save(os.path.join(tar_path, 'x_train.npy'), x_train)
    np.save(os.path.join(tar_path, 'x_test.npy'), x_test)
    np.save(os.path.join(tar_path, 'y_train.npy'), y_train)
    np.save(os.path.join(tar_path, 'y_test.npy'), y_test)

    print(f"\nSaved to {tar_path}:")
    print(f"  x_train.npy: {x_train.shape}")
    print(f"  x_test.npy: {x_test.shape}")
    print(f"  y_train.npy: {y_train.shape}")
    print(f"  y_test.npy: {y_test.shape}")


# 测试脚本：检查处理后的数据
def check_mixed_data(data_path):
    """检查混合后的数据"""
    print(f"\n{'=' * 80}")
    print(f"检查混合后的数据")
    print('=' * 80)

    # 加载数据
    try:
        x_train = np.load(os.path.join(data_path, 'x_train.npy'))
        x_test = np.load(os.path.join(data_path, 'x_test.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))

        # 打印基本信息
        print(f"\n1. 数据维度信息:")
        print(f"   x_train shape: {x_train.shape}")  # [样本数, 通道数, 时间点]
        print(f"   x_test shape: {x_test.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_test shape: {y_test.shape}")

        print(f"\n2. 样本数量统计:")
        print(f"   训练集总样本数: {len(x_train)}")
        print(f"   测试集总样本数: {len(x_test)}")
        print(f"   训练集中目标样本数: {np.sum(y_train == 1)}")
        print(f"   训练集中非目标样本数: {np.sum(y_train == 0)}")
        print(f"   测试集中目标样本数: {np.sum(y_test == 1)}")
        print(f"   测试集中非目标样本数: {np.sum(y_test == 0)}")

        print(f"\n3. 数据统计信息 (x_train):")
        print(f"   最小值: {np.min(x_train):.4f}")
        print(f"   最大值: {np.max(x_train):.4f}")
        print(f"   平均值: {np.mean(x_train):.4f}")
        print(f"   标准差: {np.std(x_train):.4f}")

        print(f"\n4. 标签统计:")
        print(f"   训练集标签分布: 目标={np.sum(y_train == 1)}, 非目标={np.sum(y_train == 0)}")
        print(f"   测试集标签分布: 目标={np.sum(y_test == 1)}, 非目标={np.sum(y_test == 0)}")

        print(f"\n5. 比例分析:")
        train_target_ratio = np.sum(y_train == 1) / len(y_train)
        test_target_ratio = np.sum(y_test == 1) / len(y_test)
        print(f"   训练集目标比例: {train_target_ratio:.2%}")
        print(f"   测试集目标比例: {test_target_ratio:.2%}")

        # 计算总目标数和非目标数
        total_targets = np.sum(y_train == 1) + np.sum(y_test == 1)
        total_nontargets = np.sum(y_train == 0) + np.sum(y_test == 0)
        print(f"\n   总计:")
        print(f"     总样本数: {len(x_train) + len(x_test)}")
        print(f"     总目标数: {total_targets} (理论值: 55 * 40 = 2200)")
        print(f"     总非目标数: {total_nontargets} (理论值: 55 * 800 = 44000)")

        # 检查通道信息
        print(f"\n6. 通道信息:")
        print(f"   数据通道数: {x_train.shape[1]}")
        print(f"   实际EEG通道数: 32")
        print(f"   补充通道数: {x_train.shape[1] - 32}")

        # 检查补充通道是否全为0
        if x_train.shape[1] > 32:
            sup_channels_train = np.sum(np.abs(x_train[:, 32:, :]) > 1e-10)
            sup_channels_test = np.sum(np.abs(x_test[:, 32:, :]) > 1e-10)
            print(f"   训练集补充通道非零值数量: {sup_channels_train}")
            print(f"   测试集补充通道非零值数量: {sup_channels_test}")

        # 检查是否有NaN或Inf值
        print(f"\n7. 数据质量检查:")
        print(f"   训练集中NaN值数量: {np.isnan(x_train).sum()}")
        print(f"   训练集中Inf值数量: {np.isinf(x_train).sum()}")
        print(f"   测试集中NaN值数量: {np.isnan(x_test).sum()}")
        print(f"   测试集中Inf值数量: {np.isinf(x_test).sum()}")

        return x_train, x_test, y_train, y_test

    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None, None, None


if __name__ == "__main__":
    # 设置路径
    raw_path = r"D:\EEG-dataset-for-RSVP-P300-speller\Python\data"
    tar_path = r"D:\ZHB_BiYeSheJi\Dataset\GIST_Mixed_64ch"

    # 检查路径是否存在
    if not os.path.exists(raw_path):
        print(f"Raw data path does not exist: {raw_path}")
        exit(1)

    # 处理数据 (混合所有被试，8:2划分，补充到64通道)
    process_gist_mixed(raw_path, tar_path, train_ratio=0.8, random_seed=42, pad_to_64=True)

    print("\n" + "=" * 80)
    print("GIST dataset processing completed (mixed all subjects, padded to 64 channels)!")
    print("=" * 80)

    # 检查处理后的数据
    check_mixed_data(tar_path)