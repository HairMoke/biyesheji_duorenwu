
# 以前可以跑，但是现在报显存溢出
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Manage.model import MTCN

# 配置
subjects = ['S64', 'S51', 'S21']  # 三个被试
models_path = '../Checkpoint/'  # 模型文件夹路径
display_names = ['S01', 'S02', 'S03']  # 显示名称

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建 3x2 的子图
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

# 遍历每个被试
for idx, subject in enumerate(subjects):
    # 设置数据路径
    DATA_PATH = f'THU/{subject}/'

    # 加载测试数据
    X_test = np.load(f'{DATA_PATH}x_test.npy')
    y_test = np.load(f'{DATA_PATH}y_test.npy')

    print(f'{subject} 数据形状: {X_test.shape}')

    if X_test.ndim == 3:
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

    # 加载对应被试的模型
    # MODEL_PATH = f'{models_path}THU_MTCN_64.pth'  # 假设使用相同模型，如需不同模型请调整路径
    # 加载对应被试的模型
    MODEL_PATH = f'{models_path}THU_MTCN_{subject[1:]}.pth'  # 每个被试使用对应模型

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MTCN(n_class_primary=2, T=256, channels=64).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)
    model.eval()

    # 提取特征 - 使用完整的特征提取流程
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
    print(f"正在进行{subject}的t-SNE降维...")
    tsne_raw = TSNE(n_components=2, random_state=42, perplexity=5,
                    early_exaggeration=100, learning_rate=50, max_iter=1000)
    X_tsne = tsne_raw.fit_transform(X_flat)

    tsne_feat = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_tsne = tsne_feat.fit_transform(features)

    # 绘制原始数据
    for label, color, name in [(1, 'green', '非目标'), (0, 'red', '目标')]:
        mask = y_subset == label
        axes[idx, 0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name, alpha=0.6, s=20, edgecolors='none')
    axes[idx, 0].set_title(f'{display_names[idx]} 原始脑电数据', fontsize=14, fontweight='bold')
    if idx == 0:  # 只在第一行显示图例
        axes[idx, 0].legend(fontsize=10)
    axes[idx, 0].grid(True, alpha=0.3)

    # 绘制提取特征
    for label, color, name in [(1, 'green', '非目标'), (0, 'red', '目标')]:
        mask = y_subset == label
        axes[idx, 1].scatter(features_tsne[mask, 0], features_tsne[mask, 1], c=color, label=name, alpha=0.6, s=20,
                             edgecolors='none')
    axes[idx, 1].set_title(f'{display_names[idx]} EMDCN提取特征', fontsize=14, fontweight='bold')
    if idx == 0:  # 只在第一行显示图例
        axes[idx, 1].legend(fontsize=10)
    axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('三被试_tsne_全部样本.png', dpi=300, bbox_inches='tight')
print("图像已保存为 三被试_tsne_全部样本.png")
plt.show()
