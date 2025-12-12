# run_new_arch.py
import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from collections import Counter
import math
import hoggorm as ho
import scanpy as sc
from new_arch import *
from utils import *
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import pickle

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 提取 RNA 特征
def extract_rna_feat(adata, num_feat=2048, dim_reduction_method='high_var'):
    if type(adata.X) is np.ndarray:
        data = adata.X
    else:
        data = adata.X.toarray()
    if dim_reduction_method == 'pca':
        pca = PCA(n_components=num_feat)
        rna_df = pca.fit_transform(data)
    elif dim_reduction_method == 'high_variable':
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=num_feat)
        rna_df = adata[:, adata.var['highly_variable']].X#.toarray()
    elif dim_reduction_method == 'high_var':
        variances = np.var(data, axis=0)
        top_n = np.argsort(variances)[-num_feat:]
        high_var_genes = adata.var.index[top_n]
        rna_df = adata[:,high_var_genes].X.toarray()
    elif dim_reduction_method == 'none':
        rna_df = adata.obsm['X_pca']
    
    return pd.DataFrame(rna_df)

# 计算熵
def Entropy(DataList):
    '''
        计算随机变量 DataList 的熵
    '''
    counts = len(DataList)      # 总数量
    try:
        counter = Counter(DataList.values.flatten())
    except:
        counter = Counter(DataList.flatten())
    prob = {i[0]:i[1]/counts for i in counter.items()}      # 计算每个变量出现的比例 p
    H = - sum([i[1]*math.log2(i[1]) for i in prob.items()]) # 计算熵
    
    print("熵：",H)
    
    return H

# 计算聚类相似性
def calculate_clustering_similarity(df, columns, metrics=['ARI', 'NMI'], save_path=None):
    # 初始化结果存储
    results = {}
    
    # 计算 ARI
    if 'ARI' in metrics:
        ari_matrix = np.zeros((len(columns), len(columns)))
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                ari_matrix[i, j] = adjusted_rand_score(df[col1], df[col2])
        results['ARI'] = pd.DataFrame(ari_matrix, index=columns, columns=columns)
    
    # 计算 NMI
    if 'NMI' in metrics:
        nmi_matrix = np.zeros((len(columns), len(columns)))
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                nmi_matrix[i, j] = normalized_mutual_info_score(df[col1], df[col2])
        results['NMI'] = pd.DataFrame(nmi_matrix, index=columns, columns=columns)
    
    # 绘制热图
    if save_path is not None:
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            sns.heatmap(
                results[metric], 
                annot=True, 
                fmt=".2f", 
                cmap="YlGnBu", 
                cbar=True,
                square=True,
                ax=axes[idx]
            )
            axes[idx].set_title(f'{metric} between Clustering Results')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return results

def disentangle_loss_jsd(model_output,hparams):
    loss_sim_rna = abs_cos_sim(model_output['rna_shared'],
                                            model_output['rna_distinct'])
    loss_sim_img = abs_cos_sim(model_output['img_shared'],
                                            model_output['img_distinct'])

    jsd = JSD()
    loss_jsd = jsd(model_output['rna_shared'].sigmoid(),
                   model_output['img_shared'].sigmoid())

    loss_disentanglement = (hparams.lambda_disentangle_shared * loss_jsd +
                            hparams.lambda_disentangle_rna * loss_sim_rna +
                            hparams.lambda_disentangle_img * loss_sim_img)

    return loss_disentanglement

def auto_normalize_log1p(adata):
    # 检查数据是否已经过log处理
    # log处理后的数据通常最大值较小(如<30)，且大部分值为非整数
    if np.max(adata.X) > 30 or np.all(np.mod(adata.X.data, 1) == 0):
        print("数据未检测到log处理，正在执行normalize_total和log1p...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("数据可能已经过log处理，跳过normalize步骤")
    
    return adata

# 主程序
def main(args):
    set_seed(42)
    # 创建保存结果的目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    img_feat = pd.read_pickle(args.image_feature_path)
    adata = sc.read(args.adata_path)
    auto_normalize_log1p(adata)
    
    # 提取 RNA 特征
    rna_feat = extract_rna_feat(adata, num_feat=args.num_rna_features, dim_reduction_method=args.dim_reduction_method)
    
    rna_feat = rna_feat.to_numpy() if hasattr(rna_feat, 'to_numpy') else np.array(rna_feat)
    img_feat = img_feat.to_numpy() if hasattr(img_feat, 'to_numpy') else np.array(img_feat)
    
    # 如果需要，对图像特征进行 PCA 降维
    if args.pca:
        n_pca = args.pca_components
        img_pca = PCA(n_components=n_pca)
        img_feat = img_pca.fit_transform(img_feat)
        
        rna_pca = PCA(n_components=n_pca)
        rna_feat = rna_pca.fit_transform(rna_feat)
    
    # 保存 RNA 和图像特征到 adata
    adata.obsm['rna_features'] = rna_feat
    adata.obsm['img_features'] = img_feat
    
    # 进行空间聚类分析
    sc.pp.neighbors(adata, use_rep='rna_features')
    sc.tl.leiden(adata, key_added='rna_feat')
    sc.pl.spatial(adata, color='rna_feat', spot_size=100, show=False)
    plt.savefig(os.path.join(args.output_dir, f'rna_feat_leiden.png'))
    
    sc.pp.neighbors(adata, use_rep='img_features')
    sc.tl.leiden(adata, key_added='img_feat')
    sc.pl.spatial(adata, color='img_feat', spot_size=100)
    plt.savefig(os.path.join(args.output_dir, f'img_feat_leiden.png'))
    
#     # 计算熵
#     Entropy(rna_feat)
#     Entropy(img_feat)
    
#     # 计算 RV 系数
#     rna_feat = rna_feat
#     img_feat = img_feat
#     rna_cent = rna_feat - np.mean(rna_feat, axis=0)
#     img_cent = img_feat - np.mean(img_feat, axis=0)
#     rv_results = ho.RVcoeff([rna_cent, img_cent])
#     print("RV Coefficients:", rv_results)
    
    # 计算生态位特征
    spot_adj, _, _ = calculate_neighborhood_graph(
        adata, feature_key='spatial', k_cutoff=args.spatial_k, metric='euclidean',
    )
    
    st = time.time()
    rna_niche_features, _, _, _, _ = aggregate_nichi_features(
        adata, feature_key='rna_features', spatial_k=args.spatial_k, feature_k=args.feature_k,
        spatial_metric='euclidean', aggregation_method='mean', remove_duplicates=True, verbose=True
    )
    end = time.time()
    print(f"RNA Niche Features Computation Time: {end - st:.2f} seconds")
    print("RNA Niche Features Shape:", rna_niche_features.shape)
    
    st = time.time()
    img_niche_features, _, _, _, _ = aggregate_nichi_features(
        adata, feature_key='img_features', spatial_k=args.spatial_k, feature_k=args.feature_k,
        spatial_metric='euclidean', aggregation_method='mean', remove_duplicates=True, verbose=True
    )
    end = time.time()
    print(f"Image Niche Features Computation Time: {end - st:.2f} seconds")
    print("Image Niche Features Shape:", img_niche_features.shape)
    
    # 准备邻接矩阵
    spot_edge_index, _ = prepare_adj(spot_adj)
    
    # 初始化模型
    model = FinalModalNetwork(
        rna_feature_dim=adata.obsm['rna_features'].shape[-1],
        img_feature_dim=adata.obsm['img_features'].shape[-1],
        rna_niche_dim=rna_niche_features.shape[-1],
        img_niche_dim=img_niche_features.shape[-1]
    )
    
    # 将模型移动到 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 准备数据张量
    rna_features = torch.tensor(adata.obsm['rna_features'], dtype=torch.float32).to(device)
    img_features = torch.tensor(adata.obsm['img_features'], dtype=torch.float32).to(device)
    rna_niche = torch.tensor(rna_niche_features, dtype=torch.float32).to(device)
    img_niche = torch.tensor(img_niche_features, dtype=torch.float32).to(device)
    spot_edge_index = spot_edge_index.to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    
    # 初始化用于存储损失值的列表
    epoch_losses = []
    img_distinct_losses = []
    img_share_losses = []
    rna_distinct_losses = []
    rna_share_losses = []
    disentanglement_losses = []
    
    # 训练循环
    num_epochs = args.num_epochs
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    
    for epoch in epoch_pbar:
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(
            rna_features=rna_features,
            img_features=img_features,
            rna_niche_features=rna_niche,
            img_niche_features=img_niche,
            spot_edge_index=spot_edge_index
        )
        
        # 计算损失
        loss_img_distinct = loss_fn(outputs['img_distinct_recon'], img_features)
        loss_img_share = loss_fn(outputs['img_share_recon'], img_features)
        loss_rna_distinct = loss_fn(outputs['rna_distinct_recon'], rna_features)
        loss_rna_share = loss_fn(outputs['rna_share_recon'], rna_features)
        loss_disentanglement = disentangle_loss_jsd(outputs,args)
        
        total_loss = loss_img_distinct + loss_img_share + loss_rna_distinct + loss_rna_share + loss_disentanglement
        
        # 反向传播和优化
        total_loss.backward()
        optimizer.step()
        
        # 存储损失值
        epoch_losses.append(total_loss.item())
        img_distinct_losses.append(loss_img_distinct.item())
        img_share_losses.append(loss_img_share.item())
        rna_distinct_losses.append(loss_rna_distinct.item())
        rna_share_losses.append(loss_rna_share.item())
        disentanglement_losses.append(loss_disentanglement.item())
        
        # 更新进度条描述
        epoch_pbar.set_postfix({
            'Total Loss': f"{total_loss.item():.4f}",
            'ImgDist': f"{loss_img_distinct.item():.4f}",
            'ImgShare': f"{loss_img_share.item():.4f}",
            'RNADist': f"{loss_rna_distinct.item():.4f}",
            'RNAShare': f"{loss_rna_share.item():.4f}",
            'Disentanglement': f"{loss_disentanglement.item():.4f}"
        })
    
    # 绘制并保存损失曲线
    plt.figure(figsize=(12, 6))
    
    # 总损失
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses, label='Total Loss')
    plt.title('Total Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 组分损失
    plt.subplot(1, 2, 2)
    plt.plot(img_distinct_losses, label='Img Distinct')
    plt.plot(img_share_losses, label='Img Share')
    plt.plot(rna_distinct_losses, label='RNA Distinct')
    plt.plot(rna_share_losses, label='RNA Share')
    plt.plot(disentanglement_losses, label='Disentanglement')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{args.dim_reduction_method}_loss_curves.png'))
    plt.close()
    
    # 将损失值保存到文件
    loss_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Total_Loss': epoch_losses,
        'Img_Distinct': img_distinct_losses,
        'Img_Share': img_share_losses,
        'RNA_Distinct': rna_distinct_losses,
        'RNA_Share': rna_share_losses,
        'Disentanglement': disentanglement_losses
    })
    loss_df.to_csv(os.path.join(args.output_dir, f'{args.dim_reduction_method}_loss_values.csv'), index=False)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.dim_reduction_method}_final_modal_dict.pth'))
    torch.save(model, os.path.join(args.output_dir, f'{args.dim_reduction_method}_final_modal_model.pth'))
    print("Training completed. Results saved in 'training_results' directory.")
    
    # 评估模型
    inputs = {
        'rna_features': torch.tensor(adata.obsm['rna_features'], dtype=torch.float32).to(device),
        'img_features': torch.tensor(adata.obsm['img_features'], dtype=torch.float32).to(device),
        'rna_niche_features': torch.tensor(rna_niche_features, dtype=torch.float32).to(device),
        'img_niche_features': torch.tensor(img_niche_features, dtype=torch.float32).to(device),
        'spot_edge_index': spot_edge_index.to(device)
    }
    
    with torch.no_grad():
        outputs = model(**inputs)  # 解包输入字典
        
    # 提取所有输出（字典形式）
    print(outputs.keys())  # 查看所有可用的输出键
    
    # 将所有张量转移到 CPU
    outputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                   for k, v in outputs.items()}
    # 保存为 .pt 或 .pkl 文件
    torch.save(outputs_cpu, os.path.join(args.output_dir,"outputs.pt"))  # 推荐 .pt 扩展名

    
    # 示例：提取特定输出并转换为 numpy 数组
    img_distinct_recon = outputs['img_distinct_recon'].cpu().numpy()
    inter_fused = outputs['inter_fused'].cpu().numpy()
    rna_distinct = outputs['rna_distinct'].cpu().numpy()
    img_distinct = outputs['img_distinct'].cpu().numpy()
    
    adata.obsm['fuse_feat'] = inter_fused  # +rna_distinct+img_distinct
    sc.pp.neighbors(adata, use_rep='fuse_feat')
    #sc.tl.leiden(adata, resolution=0.4, key_added='fuse_feat')
    resolution, adata = find_res_binary(adata, resolution_min=0.1, resolution_max=1.2, num_clusters=7, key_added='fuse_feat')
    sc.pl.spatial(adata, color='fuse_feat', spot_size=100, show=False)
    plt.savefig(os.path.join(args.output_dir, f'{args.dim_reduction_method}_spatial_domain.png'))
    
    # 计算聚类相似性
    results = calculate_clustering_similarity(
        adata.obs,
        columns=['rna_feat', 'img_feat', 'fuse_feat'],
        metrics=['ARI', 'NMI'],
        save_path=os.path.join(args.output_dir, 'clustering_similarity.png')
    )
    
    # 查看 ARI 矩阵
    print("ARI Matrix:")
    print(results['ARI'])
    
    # 查看 NMI 矩阵
    print("\nNMI Matrix:")
    print(results['NMI'])
    
    print(torch.cuda.memory_summary())

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Run the new architecture for spatial transcriptomics analysis.")
    parser.add_argument('--image_feature_path', type=str, required=True, help='Path to the image feature file.')
    parser.add_argument('--adata_path', type=str, required=True, help='Path to the adata file.')
    parser.add_argument('--output_dir', type=str, default='training_results', help='Directory to save the results.')
    parser.add_argument('--num_rna_features', type=int, default=2048, help='Number of RNA features to extract.')
    parser.add_argument('--dim_reduction_method', type=str, default='high_var', help='Dimensionality reduction method for RNA features e.g., pca, high_var.')
    parser.add_argument('--pca', action='store_true', help='Whether to perform PCA on image features.')
    parser.add_argument('--pca_components', type=int, default=200, help='Number of PCA components for image features.')
    parser.add_argument('--spatial_k', type=int, default=8, help='Spatial k for neighborhood graph.')
    parser.add_argument('--feature_k', type=int, default=16, help='Feature k for niche features.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lambda_disentangle_shared', type=float, default=1)
    parser.add_argument('--lambda_disentangle_rna', type=float, default=1)
    parser.add_argument('--lambda_disentangle_img', type=float, default=1)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)