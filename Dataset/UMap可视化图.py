import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os
import gc
from umap import UMAP
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Manage.model import MTCN


def extract_features_batch(model, X_data, device, batch_size=256):
    """
    分批提取特征，避免显存溢出

    参数:
    - model: 模型
    - X_data: 输入数据，形状为 (n_samples, 1, channels, time_points)
    - device: 设备
    - batch_size: 批处理大小

    返回:
    - features: 提取的特征，形状为 (n_samples, feature_dim)
    """
    features_list = []
    n_samples = len(X_data)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_data[i:end_idx]

            # 转换为张量
            X_tensor = torch.FloatTensor(X_batch).to(device)

            # 特征提取
            fea_shared = model.block_shared_feature_extractor(X_tensor)
            fea_specific = model.block_specific_main_feature_extractor(X_tensor)
            fea_fusion = model.block_feature_fusion(fea_shared)
            fea_main = fea_specific + fea_fusion
            fea_main = model.main_task_projection_head(fea_main)
            batch_features = fea_main.view(fea_main.size(0), -1).cpu().numpy()

            features_list.append(batch_features)

            # 清理显存
            del X_tensor, fea_shared, fea_specific, fea_fusion, fea_main, batch_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 显示进度
            if (i // batch_size) % 10 == 0 or end_idx == n_samples:
                print(f"    已提取特征 {end_idx}/{n_samples} 个样本...")

    # 合并所有批次的特征
    features = np.vstack(features_list)
    return features


def plot_umap_comparison(X_raw_flat, X_features, y_labels,
                         subject_id="S50", save_path="umap_comparison.png"):
    """
    绘制UMAP对比图：原始数据vs模型特征

    参数:
    - X_raw_flat: 展平的原始数据
    - X_features: 模型提取的特征
    - y_labels: 标签
    - subject_id: 被试ID
    - save_path: 保存路径
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'被试 {subject_id} - UMAP可视化对比', fontsize=16, fontweight='bold', y=0.98)

    # 1. 原始数据UMAP
    print("正在进行原始数据UMAP降维...")
    umap_raw = UMAP(
        n_components=2,
        n_neighbors=15,  # 邻居数，控制局部与全局平衡
        min_dist=0.1,  # 点之间的最小距离
        random_state=42,
        metric='euclidean',
        n_epochs=1000
    )

    # 如果原始数据维度太高，先进行PCA预处理
    if X_raw_flat.shape[1] > 100:
        print("  原始数据维度太高，先进行PCA降维到50维...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50, random_state=42)
        X_raw_pca = pca.fit_transform(X_raw_flat)
        X_umap_raw = umap_raw.fit_transform(X_raw_pca)
        print(f"  PCA解释方差比例: {np.sum(pca.explained_variance_ratio_):.3f}")
    else:
        X_umap_raw = umap_raw.fit_transform(X_raw_flat)

    print("原始数据UMAP完成!")

    # 2. 模型特征UMAP
    print("正在进行模型特征UMAP降维...")
    umap_features = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        metric='euclidean',
        n_epochs=1000
    )
    X_umap_features = umap_features.fit_transform(X_features)
    print("模型特征UMAP完成!")

    # 颜色和标记设置
    colors = {0: 'red', 1: 'green'}  # 目标:红色, 非目标:绿色
    markers = {0: 'o', 1: 's'}  # 目标:圆形, 非目标:方形
    sizes = {0: 80, 1: 20}  # 目标:大点, 非目标:小点
    alphas = {0: 0.9, 1: 0.3}  # 目标:不透明, 非目标:半透明
    edgecolors = {0: 'black', 1: 'none'}  # 目标有黑色边框
    linewidths = {0: 1.0, 1: 0.0}

    labels_map = {0: '目标', 1: '非目标'}

    # 计算每个类别的样本数
    n_target = np.sum(y_labels == 0)
    n_nontarget = np.sum(y_labels == 1)

    # 左图：原始数据UMAP
    ax = axes[0]
    for label in [1, 0]:  # 先绘制非目标，再绘制目标
        mask = y_labels == label
        if np.any(mask):
            ax.scatter(
                X_umap_raw[mask, 0], X_umap_raw[mask, 1],
                c=colors[label],
                marker=markers[label],
                s=sizes[label],
                alpha=alphas[label],
                edgecolors=edgecolors[label],
                linewidths=linewidths[label],
                label=labels_map[label]
            )

    ax.set_title('原始脑电数据', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 维度1', fontsize=12)
    ax.set_ylabel('UMAP 维度2', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2, linestyle='--')

    # 添加统计信息
    info_text = f'总样本: {len(y_labels)}\n目标: {n_target}\n非目标: {n_nontarget}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 右图：模型特征UMAP
    ax = axes[1]
    for label in [1, 0]:  # 先绘制非目标，再绘制目标
        mask = y_labels == label
        if np.any(mask):
            ax.scatter(
                X_umap_features[mask, 0], X_umap_features[mask, 1],
                c=colors[label],
                marker=markers[label],
                s=sizes[label],
                alpha=alphas[label],
                edgecolors=edgecolors[label],
                linewidths=linewidths[label],
                label=labels_map[label]
            )

    ax.set_title('EMDCN提取特征', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 维度1', fontsize=12)
    ax.set_ylabel('UMAP 维度2', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2, linestyle='--')

    # 添加统计信息
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n图像已保存为: {save_path}")

    plt.show()

    return X_umap_raw, X_umap_features


def plot_separate_umap(X_raw_flat, X_features, y_labels,
                       subject_id="S50", save_dir="umap_plots"):
    """
    分别绘制并保存原始数据和模型特征的UMAP图

    参数:
    - X_raw_flat: 展平的原始数据
    - X_features: 模型提取的特征
    - y_labels: 标签
    - subject_id: 被试ID
    - save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 颜色和标记设置
    colors = {0: 'red', 1: 'green'}  # 目标:红色, 非目标:绿色
    markers = {0: 'o', 1: 's'}  # 目标:圆形, 非目标:方形
    sizes = {0: 100, 1: 20}  # 目标:大点, 非目标:小点
    alphas = {0: 0.9, 1: 0.3}  # 目标:不透明, 非目标:半透明
    edgecolors = {0: 'black', 1: 'none'}  # 目标有黑色边框
    linewidths = {0: 1.0, 1: 0.0}

    labels_map = {0: '目标', 1: '非目标'}

    # 计算每个类别的样本数
    n_target = np.sum(y_labels == 0)
    n_nontarget = np.sum(y_labels == 1)
    info_text = f'总样本: {len(y_labels)}\n目标: {n_target}\n非目标: {n_nontarget}'

    # 1. 原始数据UMAP
    print("正在进行原始数据UMAP降维...")
    umap_raw = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        metric='euclidean',
        n_epochs=1000
    )

    # 如果原始数据维度太高，先进行PCA预处理
    if X_raw_flat.shape[1] > 100:
        print("  原始数据维度太高，先进行PCA降维到50维...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50, random_state=42)
        X_raw_pca = pca.fit_transform(X_raw_flat)
        X_umap_raw = umap_raw.fit_transform(X_raw_pca)
        print(f"  PCA解释方差比例: {np.sum(pca.explained_variance_ratio_):.3f}")
    else:
        X_umap_raw = umap_raw.fit_transform(X_raw_flat)

    print("原始数据UMAP完成!")

    # 绘制原始数据UMAP图
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    for label in [1, 0]:  # 先绘制非目标，再绘制目标
        mask = y_labels == label
        if np.any(mask):
            ax1.scatter(
                X_umap_raw[mask, 0], X_umap_raw[mask, 1],
                c=colors[label],
                marker=markers[label],
                s=sizes[label],
                alpha=alphas[label],
                edgecolors=edgecolors[label],
                linewidths=linewidths[label],
                label=labels_map[label]
            )

    ax1.set_title(f'被试 {subject_id} - 原始脑电数据UMAP可视化', fontsize=16, fontweight='bold')
    ax1.set_xlabel('UMAP 维度1', fontsize=12)
    ax1.set_ylabel('UMAP 维度2', fontsize=12)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.2, linestyle='--')

    # 添加统计信息
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    raw_umap_path = os.path.join(save_dir, f'{subject_id}_原始数据_UMAP.png')
    plt.savefig(raw_umap_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"原始数据UMAP图已保存: {raw_umap_path}")
    plt.close(fig1)

    # 2. 模型特征UMAP
    print("正在进行模型特征UMAP降维...")
    umap_features = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        metric='euclidean',
        n_epochs=1000
    )
    X_umap_features = umap_features.fit_transform(X_features)
    print("模型特征UMAP完成!")

    # 绘制模型特征UMAP图
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    for label in [1, 0]:  # 先绘制非目标，再绘制目标
        mask = y_labels == label
        if np.any(mask):
            ax2.scatter(
                X_umap_features[mask, 0], X_umap_features[mask, 1],
                c=colors[label],
                marker=markers[label],
                s=sizes[label],
                alpha=alphas[label],
                edgecolors=edgecolors[label],
                linewidths=linewidths[label],
                label=labels_map[label]
            )

    ax2.set_title(f'被试 {subject_id} - EMDCN提取特征UMAP可视化', fontsize=16, fontweight='bold')
    ax2.set_xlabel('UMAP 维度1', fontsize=12)
    ax2.set_ylabel('UMAP 维度2', fontsize=12)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.2, linestyle='--')

    # 添加统计信息
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    feature_umap_path = os.path.join(save_dir, f'{subject_id}_模型特征_UMAP.png')
    plt.savefig(feature_umap_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"模型特征UMAP图已保存: {feature_umap_path}")
    plt.close(fig2)

    return raw_umap_path, feature_umap_path, X_umap_raw, X_umap_features


def main():
    # 配置
    MODEL_PATH = '../Checkpoint/THU_MTCN_64.pth'
    DATA_PATH = 'THU/S50/'

    # 参数设置
    batch_size = 256  # 批处理大小，根据GPU显存调整
    subject_id = 'S50'

    print("=" * 60)
    print(f"开始处理被试: {subject_id}")
    print("=" * 60)

    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. 加载测试数据
    print("1. 加载测试数据...")
    X_test = np.load(f'{DATA_PATH}x_test.npy')
    y_test = np.load(f'{DATA_PATH}y_test.npy')

    print(f"原始数据形状: {X_test.shape}")
    print(f"标签形状: {y_test.shape}")
    print(f"目标样本数: {np.sum(y_test == 0)}, 非目标样本数: {np.sum(y_test == 1)}")

    # 2. 数据预处理
    if X_test.ndim == 3:
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
        print(f"调整后数据形状: {X_test.shape}")

    # 3. 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 4. 加载模型
    print("2. 加载模型...")
    model = MTCN(n_class_primary=2, T=256, channels=64).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)
    model.eval()
    print("模型加载成功!")

    # 5. 分批提取特征
    print("3. 分批提取特征...")
    features = extract_features_batch(model, X_test, device, batch_size=batch_size)
    print(f"特征提取完成! 特征形状: {features.shape}")

    # 6. 清理模型
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    # 7. 原始数据展平
    X_flat = X_test.reshape(len(X_test), -1)
    print(f"原始数据展平后形状: {X_flat.shape}")

    # 8. 保存UMAP对比图（一张图）
    print("\n4. 生成UMAP对比图...")
    comparison_path = f'{subject_id}_UMAP对比图.png'
    X_umap_raw, X_umap_features = plot_umap_comparison(
        X_flat, features, y_test,
        subject_id=subject_id,
        save_path=comparison_path
    )

    # 9. 分别保存UMAP图（两张单独的图）
    print("\n5. 分别保存UMAP图...")
    save_dir = 'umap_results'
    raw_path, feature_path, _, _ = plot_separate_umap(
        X_flat, features, y_test,
        subject_id=subject_id,
        save_dir=save_dir
    )

    # 10. 保存UMAP数据（可选，用于后续分析）
    print("\n6. 保存UMAP数据...")
    data_dir = 'umap_data'
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, f'{subject_id}_X_umap_raw.npy'), X_umap_raw)
    np.save(os.path.join(data_dir, f'{subject_id}_X_umap_features.npy'), X_umap_features)
    np.save(os.path.join(data_dir, f'{subject_id}_y_test.npy'), y_test)
    np.save(os.path.join(data_dir, f'{subject_id}_features.npy'), features)
    print(f"UMAP数据已保存到: {data_dir}")

    # 11. 计算分离度指标
    print("\n7. 计算分离度指标...")
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    # 原始数据分离度
    if len(np.unique(y_test)) > 1:
        sil_score_raw = silhouette_score(X_umap_raw, y_test)
        ch_score_raw = calinski_harabasz_score(X_umap_raw, y_test)
        print(f"原始数据UMAP:")
        print(f"  轮廓系数: {sil_score_raw:.4f} (越接近1表示分类越好)")
        print(f"  Calinski-Harabasz指数: {ch_score_raw:.2f} (越高表示聚类越好)")

    # 模型特征分离度
    if len(np.unique(y_test)) > 1:
        sil_score_feat = silhouette_score(X_umap_features, y_test)
        ch_score_feat = calinski_harabasz_score(X_umap_features, y_test)
        print(f"模型特征UMAP:")
        print(f"  轮廓系数: {sil_score_feat:.4f} (越接近1表示分类越好)")
        print(f"  Calinski-Harabasz指数: {ch_score_feat:.2f} (越高表示聚类越好)")

        if sil_score_feat > sil_score_raw:
            improvement = (sil_score_feat - sil_score_raw) / sil_score_raw * 100
            print(f"✓ 模型特征的轮廓系数提高了 {improvement:.1f}%")

    print(f"\n✓ 被试 {subject_id} 处理完成!")
    print(f"  对比图: {comparison_path}")
    print(f"  原始数据图: {raw_path}")
    print(f"  模型特征图: {feature_path}")


if __name__ == "__main__":
    main()