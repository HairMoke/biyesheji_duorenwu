# file: compare_training_results.py
"""
训练结果对比可视化工具
用于比较不同方法的训练损失和准确率曲线
"""

import matplotlib.pyplot as plt
import json
import os
import numpy as np
from datetime import datetime
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingResultComparator:
    """训练结果对比可视化器"""

    def __init__(self, figsize=(16, 12)):
        """
        初始化对比可视化器

        Args:
            figsize: 图像大小
        """
        self.figsize = figsize
        self.results = {}

    def load_json_data(self, filepath):
        """
        从JSON文件加载训练数据

        Args:
            filepath: JSON文件路径

        Returns:
            dict: 加载的数据
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def add_result(self, name, filepath):
        """
        添加一个训练结果

        Args:
            name: 结果名称（用于图例）
            filepath: JSON文件路径
        """
        data = self.load_json_data(filepath)
        self.results[name] = data
        print(f"已加载 {name} 的数据: {filepath}")

    def find_latest_json_in_folder(self, folder_path, pattern="*.json"):
        """
        在文件夹中查找最新的JSON文件

        Args:
            folder_path: 文件夹路径
            pattern: 文件模式

        Returns:
            str: 最新文件路径
        """
        json_files = glob.glob(os.path.join(folder_path, pattern))
        if not json_files:
            raise FileNotFoundError(f"在 {folder_path} 中未找到JSON文件")

        # 根据文件名中的时间戳找到最新文件
        latest_file = max(json_files, key=os.path.getctime)
        return latest_file

    def compare_losses_and_accuracies(self, save_plot=True, show_plot=False,
                                      save_dir="Results/Comparison",
                                      filename_suffix=""):
        """
        对比不同方法的损失和准确率曲线，使用2x2子图布局
        第一行：第一个方法的损失和准确率曲线
        第二行：第二个方法的损失和准确率曲线

        Args:
            save_plot: 是否保存图像
            show_plot: 是否显示图像
            save_dir: 保存目录
            filename_suffix: 文件名后缀

        Returns:
            str: 保存路径
        """
        if len(self.results) < 2:
            raise ValueError("至少需要两个结果进行对比")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 创建2x2图形布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)

        # 定义颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))

        # 获取方法名称列表
        method_names = list(self.results.keys())

        # 子图1: 第一个方法的训练损失
        name1 = method_names[0]
        data1 = self.results[name1]
        epochs1 = data1['epochs']
        train_losses1 = data1['train_losses']
        ax1.plot(epochs1, train_losses1,
                 label=name1,
                 color=colors[0],
                 linewidth=2,
                 marker='o',
                 markersize=4,
                 alpha=0.8)
        ax1.set_xlabel('轮次 (Epochs)')
        ax1.set_ylabel('训练损失 (Training Loss)')
        ax1.set_title(f'{name1} - 训练损失', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 第一个方法的训练准确率
        train_accs1 = data1['train_accs']
        ax2.plot(epochs1, train_accs1,
                 label=name1,
                 color=colors[0],
                 linewidth=2,
                 marker='s',
                 markersize=4,
                 alpha=0.8)
        ax2.set_xlabel('轮次 (Epochs)')
        ax2.set_ylabel('训练准确率 (%)')
        ax2.set_title(f'{name1} - 训练准确率', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: 第二个方法的训练损失
        name2 = method_names[1]
        data2 = self.results[name2]
        epochs2 = data2['epochs']
        train_losses2 = data2['train_losses']
        ax3.plot(epochs2, train_losses2,
                 label=name2,
                 color=colors[1],
                 linewidth=2,
                 marker='^',
                 markersize=4,
                 alpha=0.8)
        ax3.set_xlabel('轮次 (Epochs)')
        ax3.set_ylabel('训练损失 (Training Loss)')
        ax3.set_title(f'{name2} - 训练损失', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 子图4: 第二个方法的训练准确率
        train_accs2 = data2['train_accs']
        ax4.plot(epochs2, train_accs2,
                 label=name2,
                 color=colors[1],
                 linewidth=2,
                 marker='D',
                 markersize=4,
                 alpha=0.8)
        ax4.set_xlabel('轮次 (Epochs)')
        ax4.set_ylabel('训练准确率 (%)')
        ax4.set_title(f'{name2} - 训练准确率', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TrainingComparison_2x2_Separate_{timestamp}{filename_suffix}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"对比图像已保存至: {filepath}")
        else:
            filepath = ""

        # 显示图像
        if show_plot:
            plt.show()
        else:
            plt.close()

        return filepath

    def compare_losses_and_accuracies_side_by_side(self, save_plot=True, show_plot=False,
                                                   save_dir="Results/Comparison",
                                                   filename_suffix=""):
        """
        对比不同方法的损失和准确率曲线，左右对比布局（每个指标显示两个方法）

        Args:
            save_plot: 是否保存图像
            show_plot: 是否显示图像
            save_dir: 保存目录
            filename_suffix: 文件名后缀

        Returns:
            str: 保存路径
        """
        if len(self.results) < 2:
            raise ValueError("至少需要两个结果进行对比")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 创建2x2图形布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)

        # 定义颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))

        # 子图1: 两个方法的训练损失对比
        for i, (name, data) in enumerate(self.results.items()):
            epochs = data['epochs']
            train_losses = data['train_losses']
            ax1.plot(epochs, train_losses,
                     label=f'{name}',
                     color=colors[i],
                     linewidth=2,
                     marker='o',
                     markersize=4,
                     alpha=0.8)
        ax1.set_xlabel('轮次 (Epochs)')
        ax1.set_ylabel('训练损失 (Training Loss)')
        ax1.set_title('训练损失对比', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 两个方法的训练准确率对比
        for i, (name, data) in enumerate(self.results.items()):
            epochs = data['epochs']
            train_accs = data['train_accs']
            ax2.plot(epochs, train_accs,
                     label=f'{name}',
                     color=colors[i],
                     linewidth=2,
                     marker='s',
                     markersize=4,
                     alpha=0.8)
        ax2.set_xlabel('轮次 (Epochs)')
        ax2.set_ylabel('训练准确率 (%)')
        ax2.set_title('训练准确率对比', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: 第一个方法的损失和准确率
        for i, (name, data) in enumerate(list(self.results.items())[:1]):
            epochs = data['epochs']
            train_losses = data['train_losses']
            train_accs = data['train_accs']
            ax3_twin = ax3.twinx()  # 创建共享x轴的第二个y轴
            p1, = ax3.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2, marker='o', markersize=3, alpha=0.8)
            p2, = ax3_twin.plot(epochs, train_accs, 'r-', label='训练准确率', linewidth=2, marker='s', markersize=3, alpha=0.8)
        ax3.set_xlabel('轮次 (Epochs)')
        ax3.set_ylabel('训练损失', color='b')
        ax3_twin.set_ylabel('训练准确率 (%)', color='r')
        ax3.set_title(f'{list(self.results.keys())[0]} - 损失和准确率', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(handles=[p1, p2])

        # 子图4: 第二个方法的损失和准确率
        for i, (name, data) in enumerate(list(self.results.items())[1:2]):
            epochs = data['epochs']
            train_losses = data['train_losses']
            train_accs = data['train_accs']
            ax4_twin = ax4.twinx()  # 创建共享x轴的第二个y轴
            p1, = ax4.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2, marker='o', markersize=3, alpha=0.8)
            p2, = ax4_twin.plot(epochs, train_accs, 'r-', label='训练准确率', linewidth=2, marker='s', markersize=3, alpha=0.8)
        ax4.set_xlabel('轮次 (Epochs)')
        ax4.set_ylabel('训练损失', color='b')
        ax4_twin.set_ylabel('训练准确率 (%)', color='r')
        ax4.set_title(f'{list(self.results.keys())[1]} - 损失和准确率', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(handles=[p1, p2])

        # 调整布局
        plt.tight_layout()

        # 保存图像
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TrainingComparison_2x2_{timestamp}{filename_suffix}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"对比图像已保存至: {filepath}")
        else:
            filepath = ""

        # 显示图像
        if show_plot:
            plt.show()
        else:
            plt.close()

        return filepath

    def compare_from_folders(self, folder1_path, folder2_path,
                             folder1_name="方法1", folder2_name="方法2",
                             save_plot=True, show_plot=False,
                             save_dir="Results/Comparison", filename_suffix="",
                             layout_type="separate", accuracy_only=False, smooth_window=5):
        """
        从两个文件夹中读取最新的JSON文件并进行对比

        Args:
            folder1_path: 第一个文件夹路径
            folder2_path: 第二个文件夹路径
            folder1_name: 第一个文件夹的名称
            folder2_name: 第二个文件夹的名称
            save_plot: 是否保存图像
            show_plot: 是否显示图像
            save_dir: 保存目录
            filename_suffix: 文件名后缀
            layout_type: 布局类型 ("separate" 或 "side_by_side")
            accuracy_only: 是否只绘制准确率曲线
            smooth_window: 平滑窗口大小

        Returns:
            str: 保存路径
        """
        # 查找第一个文件夹中的最新JSON文件
        json1_path = self.find_latest_json_in_folder(folder1_path)
        self.add_result(folder1_name, json1_path)

        # 查找第二个文件夹中的最新JSON文件
        json2_path = self.find_latest_json_in_folder(folder2_path)
        self.add_result(folder2_name, json2_path)

        # 进行对比
        if accuracy_only:
            return self.compare_accuracy_curves(
                save_plot=save_plot,
                show_plot=show_plot,
                save_dir=save_dir,
                filename_suffix=filename_suffix,
                smooth_window=smooth_window
            )
        elif layout_type == "separate":
            return self.compare_losses_and_accuracies(
                save_plot=save_plot,
                show_plot=show_plot,
                save_dir=save_dir,
                filename_suffix=filename_suffix
            )
        else:
            return self.compare_losses_and_accuracies_side_by_side(
                save_plot=save_plot,
                show_plot=show_plot,
                save_dir=save_dir,
                filename_suffix=filename_suffix
            )

    def compare_accuracy_curves(self, save_plot=True, show_plot=False,
                                save_dir="Results/Comparison",
                                filename_suffix="", smooth_window=5):
        """
        对比不同方法的准确率曲线，并添加平滑线

        Args:
            save_plot: 是否保存图像
            show_plot: 是否显示图像
            save_dir: 保存目录
            filename_suffix: 文件名后缀
            smooth_window: 平滑窗口大小

        Returns:
            str: 保存路径
        """
        if len(self.results) < 2:
            raise ValueError("至少需要两个结果进行对比")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))

        # 定义颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))

        # 绘制每个方法的准确率曲线和平滑线
        for i, (name, data) in enumerate(self.results.items()):
            epochs = data['epochs']
            train_accs = data['train_accs']

            # 绘制原始准确率曲线
            ax.plot(epochs, train_accs,
                    label=f'{name} - 原始准确率',
                    color=colors[i],
                    linewidth=1.5,
                    alpha=0.6)

            # 计算并绘制平滑线（移动平均）
            if len(train_accs) >= smooth_window:
                smoothed_accs = np.convolve(train_accs, np.ones(smooth_window) / smooth_window, mode='valid')
                # 对齐x轴坐标
                smoothed_epochs = epochs[(smooth_window - 1) // 2:-(smooth_window // 2)]
                if len(smoothed_epochs) > len(smoothed_accs):
                    smoothed_epochs = smoothed_epochs[:len(smoothed_accs)]
                elif len(smoothed_epochs) < len(smoothed_accs):
                    smoothed_accs = smoothed_accs[:len(smoothed_epochs)]

                ax.plot(smoothed_epochs, smoothed_accs,
                        label=f'{name} - 平滑准确率',
                        color=colors[i],
                        linewidth=3,
                        alpha=1.0)

        ax.set_xlabel('轮次 (Epochs)', fontsize=12)
        ax.set_ylabel('训练准确率 (%)', fontsize=12)
        ax.set_title('训练准确率对比', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"AccuracyComparison_{timestamp}{filename_suffix}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"准确率对比图像已保存至: {filepath}")
        else:
            filepath = ""

        # 显示图像
        if show_plot:
            plt.show()
        else:
            plt.close()

        return filepath


def main():
    """主函数 - 演示如何使用对比功能"""
    print("开始比较不同方法的训练结果...")

    # 创建对比器实例
    comparator = TrainingResultComparator()

    # 定义文件夹路径
    folder1_path = r"D:\ZHB_BiYeSheJi\Results\图卷积_P300"
    folder2_path = r"D:\ZHB_BiYeSHi\Results\添加频域分支"

    # 进行准确率对比
    try:
        filepath = comparator.compare_from_folders(
            folder1_path=folder1_path,
            folder2_path=folder2_path,
            folder1_name="THU_MTCN_Sub01_TrainingData_20260112_221440.json",
            folder2_name="THU_MTCN_Sub01_TrainingData_20260113_111813.json",
            save_plot=True,
            show_plot=True,
            save_dir="Results/Comparison",
            filename_suffix="_accuracy_comparison",
            accuracy_only=True,  # 只绘制准确率曲线
            smooth_window=5      # 平滑窗口大小
        )

        print(f"准确率对比完成！图像已保存至: {filepath}")
    except Exception as e:
        print(f"发生错误: {e}")
        print("请确保指定的文件夹中包含JSON格式的训练数据文件")



if __name__ == "__main__":
    main()
