"""
训练过程可视化工具
用于绘制和保存训练损失曲线、准确率曲线等
Author: Assistant
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple
import seaborn as sns


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, subject_id: int, model_name: str, dataset_name: str, 
                 save_dir: str = "Results/图卷积_P300", figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器
        
        Args:
            subject_id: 被试ID
            model_name: 模型名称
            dataset_name: 数据集名称
            save_dir: 保存目录
            figsize: 图像大小
        """
        self.subject_id = subject_id
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.figsize = figsize
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化数据存储
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.epochs = []
        
        # 多任务特定数据
        self.task_losses = {
            'main': [],
            'vto': [],
            'msp': [],
            'ftr': []
        }
        self.task_accs = {
            'vto': [],
            'msp': [],
            'ftr': []
        }

        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
    def add_epoch_data(self, epoch: int, train_loss: float, train_acc: float,
                      val_loss: Optional[float] = None, val_acc: Optional[float] = None,
                      task_losses: Optional[Dict[str, float]] = None,
                      task_accs: Optional[Dict[str, float]] = None):
        """
        添加一个epoch的数据
        
        Args:
            epoch: 轮次
            train_loss: 训练损失
            train_acc: 训练准确率
            val_loss: 验证损失（可选）
            val_acc: 验证准确率（可选）
            task_losses: 各任务损失字典（可选）
            task_accs: 各任务准确率字典（可选）
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_acc is not None:
            self.val_accs.append(val_acc)
            
        # 添加任务特定数据
        if task_losses:
            for task, loss in task_losses.items():
                if task in self.task_losses:
                    self.task_losses[task].append(loss)
                    
        if task_accs:
            for task, acc in task_accs.items():
                if task in self.task_accs:
                    self.task_accs[task].append(acc)
    
    def plot_basic_curves(self, save_plot: bool = True, show_plot: bool = False) -> str:
        """
        绘制基础损失和准确率曲线
        
        Args:
            save_plot: 是否保存图像
            show_plot: 是否显示图像
            
        Returns:
            保存路径
        """
        # 绘图设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # 损失曲线
        ax1.plot(self.epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        if self.val_losses:
            ax1.plot(self.epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.set_title(f'{self.dataset_name} - 被试{self.subject_id:02d} - 损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(self.epochs, self.train_accs, 'b-', label='训练准确率', linewidth=2)
        if self.val_accs:
            ax2.plot(self.epochs, self.val_accs, 'r-', label='验证准确率', linewidth=2)
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title(f'{self.dataset_name} - 被试{self.subject_id:02d} - 准确率曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 损失下降率
        if len(self.train_losses) > 1:
            loss_diff = np.diff(self.train_losses)
            ax3.plot(self.epochs[1:], loss_diff, 'g-', label='损失下降量', linewidth=2)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('轮次')
            ax3.set_ylabel('损失下降量')
            ax3.set_title('损失下降趋势')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 收敛分析
        if len(self.train_losses) > 10:
            window_size = min(10, len(self.train_losses) // 4)
            moving_avg = np.convolve(self.train_losses, np.ones(window_size)/window_size, mode='valid')
            ax4.plot(self.epochs[window_size-1:], moving_avg, 'purple', 
                    label=f'{window_size}轮移动平均', linewidth=2)
            ax4.set_xlabel('轮次')
            ax4.set_ylabel('损失')
            ax4.set_title('收敛趋势分析')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.dataset_name}_{self.model_name}_Sub{self.subject_id:02d}_BasicCurves_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"基础曲线已保存至: {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return filepath if save_plot else ""
    
    def plot_task_specific_curves(self, save_plot: bool = True, show_plot: bool = False) -> str:
        """
        绘制任务特定的损失和准确率曲线
        
        Args:
            save_plot: 是否保存图像
            show_plot: 是否显示图像
            
        Returns:
            保存路径
        """
        # 绘图设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        if not any(self.task_losses.values()) and not any(self.task_accs.values()):
            print("没有任务特定数据可供绘制")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        # 绘制任务损失
        colors = ['red', 'blue', 'green', 'orange']
        if any(self.task_losses.values()):
            ax = axes[0]
            for i, (task, losses) in enumerate(self.task_losses.items()):
                if losses:
                    ax.plot(self.epochs[:len(losses)], losses, 
                           color=colors[i], label=f'{task.upper()}任务', linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('轮次')
            ax.set_ylabel('损失')
            ax.set_title('各任务损失对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 绘制任务准确率
        if any(self.task_accs.values()):
            ax = axes[1]
            for i, (task, accs) in enumerate(self.task_accs.items()):
                if accs:
                    ax.plot(self.epochs[:len(accs)], accs, 
                           color=colors[i], label=f'{task.upper()}任务', linewidth=2, marker='s', markersize=4)
            ax.set_xlabel('轮次')
            ax.set_ylabel('准确率 (%)')
            ax.set_title('各任务准确率对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 损失热力图
        if any(self.task_losses.values()):
            ax = axes[2]
            task_names = []
            task_loss_arrays = []
            for task, losses in self.task_losses.items():
                if losses:
                    task_names.append(task.upper())
                    task_loss_arrays.append(losses)
            
            if task_loss_arrays:
                loss_matrix = np.array(task_loss_arrays)
                im = ax.imshow(loss_matrix, aspect='auto', cmap='YlOrRd')
                ax.set_yticks(range(len(task_names)))
                ax.set_yticklabels(task_names)
                ax.set_xlabel('轮次')
                ax.set_title('任务损失热力图')
                plt.colorbar(im, ax=ax)
        
        # 准确率分布
        if any(self.task_accs.values()):
            ax = axes[3]
            acc_data = []
            task_labels = []
            for task, accs in self.task_accs.items():
                if accs:
                    acc_data.extend(accs)
                    task_labels.extend([task.upper()] * len(accs))
            
            if acc_data:
                import pandas as pd
                df = pd.DataFrame({'Accuracy': acc_data, 'Task': task_labels})
                sns.boxplot(data=df, x='Task', y='Accuracy', ax=ax)
                ax.set_title('任务准确率分布')
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.dataset_name}_{self.model_name}_Sub{self.subject_id:02d}_TaskCurves_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"任务特定曲线已保存至: {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return filepath if save_plot else ""
    
    def plot_comparison_curves(self, other_visualizers: List['TrainingVisualizer'], 
                              metric: str = 'loss', save_plot: bool = True, show_plot: bool = False) -> str:
        """
        绘制多个被试的对比曲线
        
        Args:
            other_visualizers: 其他被试的可视化器列表
            metric: 对比指标 ('loss' 或 'acc')
            save_plot: 是否保存图像
            show_plot: 是否显示图像
            
        Returns:
            保存路径
        """
        # 绘图设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 收集所有被试的数据
        all_visualizers = [self] + other_visualizers
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_visualizers)))
        
        # 绘制损失对比
        if metric == 'loss':
            for i, viz in enumerate(all_visualizers):
                if viz.train_losses:
                    ax1.plot(viz.epochs, viz.train_losses, color=colors[i], 
                            label=f'被试{viz.subject_id:02d}', linewidth=2, alpha=0.8)
            ax1.set_xlabel('轮次')
            ax1.set_ylabel('训练损失')
            ax1.set_title(f'{self.dataset_name} - 多被试训练损失对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 绘制准确率对比
        if metric == 'acc':
            for i, viz in enumerate(all_visualizers):
                if viz.train_accs:
                    ax2.plot(viz.epochs, viz.train_accs, color=colors[i], 
                            label=f'被试{viz.subject_id:02d}', linewidth=2, alpha=0.8)
            ax2.set_xlabel('轮次')
            ax2.set_ylabel('训练准确率 (%)')
            ax2.set_title(f'{self.dataset_name} - 多被试训练准确率对比')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.dataset_name}_{self.model_name}_MultiSubject_Comparison_{metric}_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"多被试对比曲线已保存至: {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return filepath if save_plot else ""
    
    def save_data(self, filename: Optional[str] = None) -> str:
        """
        保存训练数据到JSON文件
        
        Args:
            filename: 文件名（可选）
            
        Returns:
            保存路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.dataset_name}_{self.model_name}_Sub{self.subject_id:02d}_TrainingData_{timestamp}.json"
        
        data = {
            'subject_id': self.subject_id,
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'task_losses': self.task_losses,
            'task_accs': self.task_accs,
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"训练数据已保存至: {filepath}")
        return filepath
    
    def load_data(self, filepath: str):
        """
        从JSON文件加载训练数据
        
        Args:
            filepath: JSON文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.subject_id = data['subject_id']
        self.model_name = data['model_name']
        self.dataset_name = data['dataset_name']
        self.epochs = data['epochs']
        self.train_losses = data['train_losses']
        self.train_accs = data['train_accs']
        self.val_losses = data.get('val_losses', [])
        self.val_accs = data.get('val_accs', [])
        self.task_losses = data.get('task_losses', {})
        self.task_accs = data.get('task_accs', {})
        
        print(f"训练数据已从 {filepath} 加载")


# 便捷函数用于快速绘制
def quick_plot_training_curves(train_losses: List[float], train_accs: List[float],
                             val_losses: Optional[List[float]] = None,
                             val_accs: Optional[List[float]] = None,
                             subject_id: int = 1,
                             model_name: str = "Model",
                             dataset_name: str = "Dataset",
                             save_dir: str = "Results/图卷积_P300") -> str:
    """
    快速绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        val_losses: 验证损失列表（可选）
        val_accs: 验证准确率列表（可选）
        subject_id: 被试ID
        model_name: 模型名称
        dataset_name: 数据集名称
        save_dir: 保存目录
        
    Returns:
        保存路径
    """
    visualizer = TrainingVisualizer(
        subject_id=subject_id,
        model_name=model_name,
        dataset_name=dataset_name,
        save_dir=save_dir
    )
    
    # 添加数据
    for i, (loss, acc) in enumerate(zip(train_losses, train_accs)):
        val_loss = val_losses[i] if val_losses else None
        val_acc = val_accs[i] if val_accs else None
        visualizer.add_epoch_data(i, loss, acc, val_loss, val_acc)
    
    # 绘制并保存
    filepath = visualizer.plot_basic_curves(save_plot=True, show_plot=False)
    return filepath


if __name__ == "__main__":
    # # 测试代码
    # print("训练可视化工具测试...")
    #
    # # 创建模拟数据
    # epochs = 50
    # train_losses = [np.exp(-x/20) + 0.1 * np.random.randn() for x in range(epochs)]
    # train_losses = [max(0, loss) for loss in train_losses]
    # train_accs = [50 + 40 * (1 - np.exp(-x/15)) + 2 * np.random.randn() for x in range(epochs)]
    # train_accs = [min(95, max(50, acc)) for acc in train_accs]
    #
    # # 快速绘制
    # filepath = quick_plot_training_curves(
    #     train_losses=train_losses,
    #     train_accs=train_accs,
    #     subject_id=1,
    #     model_name="MTCN",
    #     dataset_name="THU"
    # )
    #
    # print(f"测试图像已保存至: {filepath}")
    # print("训练可视化工具测试完成！")

    # 从真实数据文件读取
    print("从真实数据文件加载训练可视化...")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建可视化器实例
    visualizer = TrainingVisualizer(
        subject_id=1,
        model_name="MTCN",
        dataset_name="THU",
        save_dir="Results/TrainingCurves"
    )

    # 从JSON文件加载真实数据
    data_file_path = r"/Results/图卷积_P300\THU_MTCN_Sub01_TrainingData_20260113_020544.json"
    visualizer.load_data(data_file_path)

    # 绘制基础曲线
    basic_filepath = visualizer.plot_basic_curves(save_plot=True, show_plot=False)

    # 如果有多任务数据，绘制任务特定曲线
    task_filepath = visualizer.plot_task_specific_curves(save_plot=True, show_plot=False)

    print(f"基础曲线已保存至: {basic_filepath}")
    print(f"任务特定曲线已保存至: {task_filepath}")
    print("真实数据可视化完成！")