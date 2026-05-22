import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.channels import make_standard_montage
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_eeg_sensor_layout():
    """创建脑电传感器位置图，兼容不同MNE版本"""

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. 使用plot_sensors绘制传感器位置
    ax1 = axes[0]

    try:
        # 创建标准10-20蒙太奇
        montage = make_standard_montage('standard_1020')

        # 创建模拟的info对象
        ch_names = montage.ch_names[:32]  # 使用前32个电极
        info = mne.create_info(ch_names, sfreq=250, ch_types='eeg')
        info.set_montage(montage)

        # 使用plot_sensors绘制传感器位置
        mne.viz.plot_sensors(info, axes=ax1, show_names=True,
                             sphere=(0, 0, 0, 0.1))

        ax1.set_title('10-20系统传感器位置', fontsize=14, fontweight='bold')

    except Exception as e:
        print(f"使用plot_sensors时出错: {e}")
        # 如果出错，手动绘制
        ax1.text(0.5, 0.5, "无法绘制传感器位置",
                 ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('传感器位置图', fontsize=14, fontweight='bold')
        ax1.axis('off')

    # 2. 手动绘制电极位置
    ax2 = axes[1]
    plot_manual_sensor_positions(ax2)

    # 3. 绘制脑区分区
    ax3 = axes[2]
    plot_brain_regions(ax3)

    plt.suptitle('脑电传感器位置示意图', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()

    return fig, axes


def plot_manual_sensor_positions(ax):
    """手动绘制标准10-20系统电极位置"""

    # 绘制头部轮廓
    circle = plt.Circle((0, 0), 1.0, color='lightblue', alpha=0.2,
                        linewidth=2, edgecolor='black', fill=False)
    ax.add_patch(circle)

    # 绘制鼻子
    ax.plot([-0.1, 0, 0.1], [1.0, 1.15, 1.0], 'black', linewidth=2)

    # 绘制耳朵
    left_ear = plt.Circle((-1.0, 0.3), 0.1, color='gray', alpha=0.3)
    right_ear = plt.Circle((1.0, 0.3), 0.1, color='gray', alpha=0.3)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

    # 标准10-20系统电极位置
    # 前额叶
    fp_positions = [(-0.3, 0.8), (0.3, 0.8)]
    fp_labels = ['Fp1', 'Fp2']

    # 额叶
    f_positions = [(-0.5, 0.5), (0.5, 0.5), (-0.6, 0.4), (0.6, 0.4)]
    f_labels = ['F3', 'F4', 'F7', 'F8']

    # 中央区
    c_positions = [(-0.5, 0.0), (0.5, 0.0), (-0.6, -0.1), (0.6, -0.1)]
    c_labels = ['C3', 'C4', 'T7', 'T8']

    # 顶叶
    p_positions = [(-0.5, -0.5), (0.5, -0.5), (-0.4, -0.6), (0.4, -0.6)]
    p_labels = ['P3', 'P4', 'P7', 'P8']

    # 枕叶
    o_positions = [(-0.3, -1.0), (0.3, -1.0)]
    o_labels = ['O1', 'O2']

    # 中线电极
    midline_positions = [(0, 0.4), (0, 0), (0, -0.7), (0, -1.0)]
    midline_labels = ['Fz', 'Cz', 'Pz', 'Oz']

    # 合并所有位置
    all_positions = fp_positions + f_positions + c_positions + p_positions + o_positions + midline_positions
    all_labels = fp_labels + f_labels + c_labels + p_labels + o_labels + midline_labels

    # 绘制所有电极
    for pos, label in zip(all_positions, all_labels):
        # 绘制电极点
        electrode = plt.Circle(pos, 0.05, color='white',
                               edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(electrode)

        # 添加标签
        ax.text(pos[0], pos[1], label, fontsize=8, fontweight='bold',
                ha='center', va='center')

    # 添加连接线
    # 定义一些典型的连接
    connections = [
        ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
        ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
        ('Fz', 'Cz'), ('Cz', 'Pz'), ('Pz', 'Oz'),
    ]

    # 创建位置字典
    pos_dict = dict(zip(all_labels, all_positions))

    for ch1, ch2 in connections:
        if ch1 in pos_dict and ch2 in pos_dict:
            x1, y1 = pos_dict[ch1]
            x2, y2 = pos_dict[ch2]
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.8)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title('手动绘制的10-20系统', fontsize=14, fontweight='bold')
    ax.axis('off')


def plot_brain_regions(ax):
    """绘制脑区分区图"""

    # 绘制头部轮廓
    circle = plt.Circle((0, 0), 1.0, color='lightgray', alpha=0.2,
                        linewidth=2, edgecolor='black', fill=False)
    ax.add_patch(circle)

    # 绘制脑区
    regions = {
        '前额叶\n(Prefrontal)': {
            'pos': (0, 0.8),
            'color': '#FF6B6B',
            'area': plt.Circle((0, 0.8), 0.3, color='#FF6B6B', alpha=0.3)
        },
        '额叶\n(Frontal)': {
            'pos': (0, 0.3),
            'color': '#4ECDC4',
            'area': plt.Rectangle((-0.7, 0.1), 1.4, 0.4, color='#4ECDC4', alpha=0.3)
        },
        '中央区\n(Central)': {
            'pos': (0, -0.1),
            'color': '#45B7D1',
            'area': plt.Rectangle((-0.7, -0.3), 1.4, 0.4, color='#45B7D1', alpha=0.3)
        },
        '顶叶\n(Parietal)': {
            'pos': (0, -0.6),
            'color': '#96CEB4',
            'area': plt.Rectangle((-0.7, -0.8), 1.4, 0.4, color='#96CEB4', alpha=0.3)
        },
        '枕叶\n(Occipital)': {
            'pos': (0, -1.0),
            'color': '#FFEAA7',
            'area': plt.Circle((0, -1.0), 0.3, color='#FFEAA7', alpha=0.3)
        },
        '左颞叶\n(L Temporal)': {
            'pos': (-0.8, 0.0),
            'color': '#DDA0DD',
            'area': plt.Circle((-0.8, 0.0), 0.25, color='#DDA0DD', alpha=0.3)
        },
        '右颞叶\n(R Temporal)': {
            'pos': (0.8, 0.0),
            'color': '#DDA0DD',
            'area': plt.Circle((0.8, 0.0), 0.25, color='#DDA0DD', alpha=0.3)
        }
    }

    # 添加脑区
    for region, info in regions.items():
        ax.add_patch(info['area'])
        ax.text(info['pos'][0], info['pos'][1], region,
                fontsize=9, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 添加一些示例电极
    sample_electrodes = [
        ('Fp1', (-0.3, 0.8)), ('Fp2', (0.3, 0.8)),
        ('F3', (-0.5, 0.5)), ('F4', (0.5, 0.5)),
        ('C3', (-0.5, 0.0)), ('C4', (0.5, 0.0)),
        ('P3', (-0.5, -0.5)), ('P4', (0.5, -0.5)),
        ('O1', (-0.3, -1.0)), ('O2', (0.3, -1.0)),
        ('T7', (-0.9, 0.0)), ('T8', (0.9, 0.0)),
    ]

    for label, pos in sample_electrodes:
        electrode = plt.Circle(pos, 0.03, color='black', alpha=0.7)
        ax.add_patch(electrode)
        ax.text(pos[0], pos[1], label, fontsize=7,
                ha='center', va='center', color='white', fontweight='bold')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title('脑功能分区示意图', fontsize=14, fontweight='bold')
    ax.axis('off')


def create_publication_quality_figure():
    """创建适合论文发表的高质量图形"""

    fig = plt.figure(figsize=(15, 10))

    # 1. 左侧：电极位置图
    ax1 = plt.subplot(1, 2, 1)

    # 绘制头部
    head = plt.Circle((0, 0), 1.0, color='lightblue', alpha=0.1,
                      linewidth=2, edgecolor='black', fill=False)
    ax1.add_patch(head)

    # 标准10-20系统电极位置
    # 定义电极位置 (基于标准10-20系统)
    electrodes_1020 = {
        'Fp1': (-0.3, 0.85), 'Fp2': (0.3, 0.85),
        'AF3': (-0.2, 0.7), 'AF4': (0.2, 0.7), 'AFz': (0, 0.75),
        'F1': (-0.4, 0.5), 'F2': (0.4, 0.5), 'F3': (-0.6, 0.4), 'F4': (0.6, 0.4),
        'F5': (-0.7, 0.3), 'F6': (0.7, 0.3), 'F7': (-0.8, 0.2), 'F8': (0.8, 0.2), 'Fz': (0, 0.4),
        'FC1': (-0.4, 0.2), 'FC2': (0.4, 0.2), 'FC3': (-0.6, 0.1), 'FC4': (0.6, 0.1), 'FCz': (0, 0.1),
        'C1': (-0.4, 0.0), 'C2': (0.4, 0.0), 'C3': (-0.6, -0.1), 'C4': (0.6, -0.1), 'Cz': (0, -0.1),
        'CP1': (-0.4, -0.2), 'CP2': (0.4, -0.2), 'CP3': (-0.6, -0.3), 'CP4': (0.6, -0.3), 'CPz': (0, -0.3),
        'P1': (-0.4, -0.4), 'P2': (0.4, -0.4), 'P3': (-0.6, -0.5), 'P4': (0.6, -0.5), 'Pz': (0, -0.5),
        'PO3': (-0.3, -0.7), 'PO4': (0.3, -0.7), 'POz': (0, -0.7),
        'O1': (-0.2, -0.9), 'O2': (0.2, -0.9), 'Oz': (0, -0.9),
        'T7': (-0.9, 0.0), 'T8': (0.9, 0.0),
        'FT7': (-0.9, 0.2), 'FT8': (0.9, 0.2),
        'TP7': (-0.9, -0.3), 'TP8': (0.9, -0.3),
    }

    # 按脑区分色
    region_colors = {
        'Fp': '#FF6B6B', 'AF': '#FF8E8E',
        'F': '#4ECDC4', 'FC': '#6ED4D1',
        'C': '#45B7D1', 'CP': '#67C5DF',
        'P': '#96CEB4', 'PO': '#B0DCC6',
        'O': '#FFEAA7',
        'T': '#DDA0DD', 'FT': '#E6BBE6', 'TP': '#EEC6EE'
    }

    # 绘制电极
    for label, pos in electrodes_1020.items():
        # 确定颜色
        color = 'gray'  # 默认颜色
        for prefix, col in region_colors.items():
            if label.startswith(prefix):
                color = col
                break

        # 绘制电极点
        electrode = plt.Circle(pos, 0.035, color=color, alpha=0.8,
                               edgecolor='black', linewidth=1)
        ax1.add_patch(electrode)

        # 添加标签（只显示部分，避免重叠）
        if label in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'Fz', 'Cz', 'Pz', 'Oz', 'T7', 'T8']:
            ax1.text(pos[0], pos[1], label, fontsize=8, fontweight='bold',
                     ha='center', va='center')

    # 添加鼻子
    ax1.plot([-0.05, 0, 0.05], [1.0, 1.1, 1.0], 'black', linewidth=2)

    # 添加耳朵
    left_ear = plt.Circle((-1.0, 0.2), 0.08, color='gray', alpha=0.3)
    right_ear = plt.Circle((1.0, 0.2), 0.08, color='gray', alpha=0.3)
    ax1.add_patch(left_ear)
    ax1.add_patch(right_ear)

    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.set_title('10-20脑电电极系统', fontsize=16, fontweight='bold')
    ax1.axis('off')

    # 2. 右侧：说明和图例
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')

    # 添加说明文本
    explanation = (
        "10-20脑电电极系统\n\n"
        "国际10-20系统是脑电记录的标准电极放置方法。\n\n"
        "命名规则：\n"
        "• 字母表示大脑区域：\n"
        "  F - 额叶 (Frontal lobe)\n"
        "  C - 中央区 (Central region)\n"
        "  P - 顶叶 (Parietal lobe)\n"
        "  O - 枕叶 (Occipital lobe)\n"
        "  T - 颞叶 (Temporal lobe)\n\n"
        "• 数字表示位置：\n"
        "  奇数 - 左侧半球\n"
        "  偶数 - 右侧半球\n"
        "  z    - 中线位置\n\n"
        "系统特点：\n"
        "• 电极间距为10%或20%的头围距离\n"
        "• 确保电极位置可重复、标准化\n"
        "• 适用于研究和临床脑电记录\n\n"
        "常用配置：\n"
        "• 19通道：基本临床配置\n"
        "• 32通道：标准研究配置\n"
        "• 64通道：高密度研究配置\n"
        "• 128+通道：超高密度研究"
    )

    ax2.text(0.05, 0.95, explanation, fontsize=11, va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    # 添加图例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='#FF6B6B', label='前额叶 (FP/AF)', alpha=0.8),
        Patch(facecolor='#4ECDC4', label='额叶 (F)', alpha=0.8),
        Patch(facecolor='#45B7D1', label='中央区 (C/FC)', alpha=0.8),
        Patch(facecolor='#96CEB4', label='顶叶 (P/CP)', alpha=0.8),
        Patch(facecolor='#FFEAA7', label='枕叶 (O/PO)', alpha=0.8),
        Patch(facecolor='#DDA0DD', label='颞叶 (T/FT/TP)', alpha=0.8),
    ]

    ax2.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=10)

    # 添加引用信息
    ax2.text(0.5, 0.02, "图1. 标准10-20脑电电极系统示意图",
             fontsize=10, ha='center', style='italic')

    plt.suptitle('脑电传感器位置标准化图示', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    return fig, (ax1, ax2)


def create_simple_sensor_layout():
    """创建一个简单的传感器位置图，避免MNE版本问题"""

    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制头部
    head = plt.Circle((0, 0), 1.0, color='lightblue', alpha=0.1,
                      linewidth=2, edgecolor='black')
    ax.add_patch(head)

    # 绘制鼻子
    ax.plot([-0.05, 0, 0.05], [1.0, 1.1, 1.0], 'black', linewidth=2)

    # 绘制耳朵
    left_ear = plt.Circle((-1.0, 0.2), 0.1, color='gray', alpha=0.3)
    right_ear = plt.Circle((1.0, 0.2), 0.1, color='gray', alpha=0.3)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

    # 标准19个电极位置（基本的10-20系统）
    electrodes_19 = {
        'Fp1': (-0.3, 0.8), 'Fp2': (0.3, 0.8),
        'F3': (-0.5, 0.5), 'F4': (0.5, 0.5), 'F7': (-0.7, 0.4), 'F8': (0.7, 0.4), 'Fz': (0, 0.5),
        'C3': (-0.5, 0.0), 'C4': (0.5, 0.0), 'T3': (-0.8, 0.0), 'T4': (0.8, 0.0), 'Cz': (0, 0.0),
        'P3': (-0.5, -0.5), 'P4': (0.5, -0.5), 'T5': (-0.7, -0.4), 'T6': (0.7, -0.4), 'Pz': (0, -0.5),
        'O1': (-0.3, -0.8), 'O2': (0.3, -0.8), 'Oz': (0, -0.8)
    }

    # 绘制电极
    for label, pos in electrodes_19.items():
        electrode = plt.Circle(pos, 0.04, color='white',
                               edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.add_patch(electrode)

        # 添加标签
        ax.text(pos[0], pos[1], label, fontsize=8, fontweight='bold',
                ha='center', va='center')

    # 添加脑区标签
    brain_regions = [
        ('前额叶', (0, 0.9)),
        ('额叶', (0, 0.5)),
        ('中央区', (0, 0)),
        ('顶叶', (0, -0.5)),
        ('枕叶', (0, -0.9)),
        ('左颞叶', (-0.9, 0)),
        ('右颞叶', (0.9, 0)),
    ]

    for region, pos in brain_regions:
        ax.text(pos[0], pos[1], region, fontsize=10, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title('标准10-20脑电电极系统（19通道）', fontsize=16, fontweight='bold')
    ax.axis('off')

    return fig, ax


# 主程序
if __name__ == "__main__":
    print("开始绘制脑电传感器位置图...")
    print("=" * 60)

    try:
        # 1. 创建简单传感器位置图
        print("1. 创建简单传感器位置图...")
        fig1, ax1 = create_simple_sensor_layout()
        plt.savefig('simple_eeg_sensors.png', dpi=300, bbox_inches='tight')
        print("   已保存: simple_eeg_sensors.png")

        # 2. 创建综合传感器图
        print("\n2. 创建综合传感器图...")
        fig2, axes2 = create_eeg_sensor_layout()
        plt.savefig('eeg_sensor_layout.png', dpi=300, bbox_inches='tight')
        print("   已保存: eeg_sensor_layout.png")

        # 3. 创建论文质量图
        print("\n3. 创建论文质量图...")
        fig3, axes3 = create_publication_quality_figure()
        plt.savefig('publication_quality_eeg.png', dpi=300, bbox_inches='tight')
        print("   已保存: publication_quality_eeg.png")

        print("\n" + "=" * 60)
        print("所有图形已成功生成！")
        print("\n生成的文件：")
        print("1. simple_eeg_sensors.png - 简单的19通道电极图")
        print("2. eeg_sensor_layout.png - 综合传感器布局图")
        print("3. publication_quality_eeg.png - 论文质量图")

        print("\n提示：")
        print("- 所有图形均为300DPI，适合论文发表")
        print("- 电极位置基于国际10-20标准系统")
        print("- 可以修改代码中的电极位置和颜色")

        # 显示图形
        plt.show()

    except Exception as e:
        print(f"运行过程中出错: {e}")
        print("\n尝试基本绘图...")

        # 如果出错，尝试最基本的绘图
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "绘图出错，请检查MNE安装\n\n错误信息:\n" + str(e),
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('绘图错误', fontsize=16)
        ax.axis('off')
        plt.savefig('error_fallback.png', dpi=300, bbox_inches='tight')
        print("已保存错误回退图: error_fallback.png")
        plt.show()