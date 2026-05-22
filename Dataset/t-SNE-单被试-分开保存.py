import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Manage.model import MTCN

# 显卡跑不动放雾浮起上跑


# 配置
MODEL_PATH = '../Checkpoint/THU_MTCN_01.pth'
DATA_PATH = 'THU/S01/'

# 加载测试数据
X_test = np.load(f'{DATA_PATH}x_test.npy')
y_test = np.load(f'{DATA_PATH}y_test.npy')

print(f"数据形状: {X_test.shape}")

if X_test.ndim == 3:
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MTCN(n_class_primary=2, T=256, channels=64).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(filtered_state_dict)
model.load_state_dict(model_dict)
model.eval()

# 提取特征 - 使用完整的特征提取流程
print("正在提取特征...")
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_test.squeeze(1)).to(device)
    X_tensor = X_tensor.unsqueeze(1)

    # 完整的特征提取流程
    fea_shared = model.block_shared_feature_extractor(X_tensor)
    fea_specific = model.block_specific_main_feature_extractor(X_tensor)
    fea_fusion = model.block_feature_fusion(fea_shared)
    fea_main = fea_specific + fea_fusion
    fea_main = model.main_task_projection_head(fea_main)
    features = fea_main.view(fea_main.size(0), -1).cpu().numpy()

# 原始数据展平
X_flat = X_test.reshape(len(X_test), -1)
y_subset = y_test

# t-SNE降维 - 分别降维
print("正在进行t-SNE降维...")
tsne_raw = TSNE(n_components=2, random_state=42, perplexity=5,
                early_exaggeration=100, learning_rate=50, n_iter=1000)
X_tsne = tsne_raw.fit_transform(X_flat)

tsne_feat = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
features_tsne = tsne_feat.fit_transform(features)

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 定义类别颜色和标签
class_info = [
    {'label': 0, 'color': 'red', 'name': '目标'},
    {'label': 1, 'color': 'green', 'name': '非目标'}
]


# 函数：创建并保存单个图片
def save_single_plot(tsne_data, labels, class_info, filename, figsize=(8, 6)):
    """创建并保存单个t-SNE分布图"""
    plt.figure(figsize=figsize)

    # 绘制散点图
    for cls_info in class_info:
        mask = labels == cls_info['label']
        plt.scatter(tsne_data[mask, 0], tsne_data[mask, 1],
                    c=cls_info['color'], label=cls_info['name'],
                    alpha=0.6, s=20, edgecolors='none')

    # 设置图表属性
    plt.grid(True, alpha=0.3)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)

    # 添加图例在右侧外部
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # 调整布局，为图例留出空间
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 右侧留出15%的空间给图例

    # 保存图片
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"图片已保存为 {filename}")
    plt.show()


# 1. 保存原始脑电数据分布图
print("\n正在绘制原始脑电数据分布图...")
save_single_plot(X_tsne, y_subset, class_info, '原始脑电数据.png')

# 2. 保存模型提取特征分布图
print("\n正在绘制模型提取特征分布图...")
save_single_plot(features_tsne, y_subset, class_info, 'EMDCN提取特征.png')

# 3. 创建两张图片合并在一张图上的对比图
print("\n正在绘制对比图（两张图合并在一张上）...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 原始数据
for cls_info in class_info:
    mask = y_subset == cls_info['label']
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=cls_info['color'], label=cls_info['name'],
                    alpha=0.6, s=20, edgecolors='none')
axes[0].set_title('原始脑电数据', fontsize=16, fontweight='bold')
axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
axes[0].legend(loc='upper right', fontsize=12)
axes[0].grid(True, alpha=0.3)

# 提取特征
for cls_info in class_info:
    mask = y_subset == cls_info['label']
    axes[1].scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                    c=cls_info['color'], label=cls_info['name'],
                    alpha=0.6, s=20, edgecolors='none')
axes[1].set_title('EMDCN提取特征', fontsize=16, fontweight='bold')
axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
axes[1].legend(loc='upper right', fontsize=12)
axes[1].grid(True, alpha=0.3)

# 添加总体标题
fig.suptitle('特征提取能力对比', fontsize=18, fontweight='bold', y=0.95)

plt.tight_layout()
plt.savefig('特征提取对比图.png', dpi=300, bbox_inches='tight', facecolor='white')
print("对比图已保存为 特征提取对比图.png")
plt.show()

# 打印统计信息
print("\n统计信息:")
print(f"样本总数: {len(y_test)}")
print(f"目标样本数: {np.sum(y_test == 0)}")
print(f"非目标样本数: {np.sum(y_test == 1)}")
print(f"原始数据维度: {X_test.shape}")
print(f"提取特征维度: {features.shape}")