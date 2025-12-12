import torch
import torch.nn as nn
import scanpy as sc
import anndata
from anndata import AnnData
import pandas as pd
import numpy as np
import sklearn.neighbors
from anndata import AnnData
from typing import Optional, Union, List, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected
from torch_geometric.nn import GCNConv, GATConv, GINConv, TransformerConv
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def calculate_neighborhood_graph(
    adata: AnnData,
    feature_key: str = 'features',
    rad_cutoff: Optional[float] = None,
    k_cutoff: Optional[int] = None,
    metric: str = 'cosine',
    verbose: bool = True,
    **kwargs
) -> AnnData:
    # Input validation
    if feature_key not in adata.obsm:
        raise ValueError(f"Features not found in adata.obsm['{feature_key}']")
    
    if not (rad_cutoff or k_cutoff):
        raise ValueError("Must provide either rad_cutoff or k_cutoff")
    if rad_cutoff and k_cutoff:
        raise ValueError("Provide only one of rad_cutoff or k_cutoff")

    # Get features
    X = adata.obsm[feature_key]
    if hasattr(X, 'toarray'):  # Handle sparse matrices
        X = X.toarray()

    # Calculate neighbors
    if rad_cutoff:
        if verbose:
            print(f"Calculating {feature_key} neighbors within radius {rad_cutoff}...")
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff, metric=metric).fit(X)
        distances, indices = nbrs.radius_neighbors(X, return_distance=True)
    else:
        if verbose:
            print(f"Calculating {feature_key} {k_cutoff}-nearest neighbors...")
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1, metric=metric).fit(X)  # +1 to include self
        distances, indices = nbrs.kneighbors(X)

    adj_net = pd.DataFrame({'source': np.repeat(np.arange(len(indices)), [len(x) for x in indices]),
                            'target': np.concatenate(indices),
                            'distance': np.concatenate(distances)})
    
    if verbose:
        print(f"Graph contains {len(adj_net)} edges among {adata.n_obs} cells.")
        print(f"Average neighbors per cell: {len(adj_net)/adata.n_obs:.2f}")
    
    return adj_net, distances, indices

def aggregate_nichi_features(
    adata: AnnData,
    spatial_k: int = 8,
    feature_k: int = 8,
    feature_key: Union[str, np.ndarray] = 'X_pca',
    spatial_metric: str = 'euclidean',
    feature_metric: str = 'cosine',
    aggregation_method: str = 'mean',
    remove_duplicates: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """Robust feature aggregation with type checking"""
    # 获取特征矩阵
    if isinstance(feature_key, str):
        try:
            features = adata.obsm[feature_key] if feature_key in adata.obsm else adata.X
        except KeyError:
            raise ValueError(f"Feature key '{feature_key}' not found in adata.obsm")
    else:
        features = feature_key  # 允许直接传入特征矩阵
    
    # 转换为numpy数组
    if hasattr(features, 'toarray'):
        features = features.toarray()
    #features = np.asarray(features)
    print(f'features shape: {features.shape}')

    # 查找邻居
    try:
        print(f'Finding {feature_key} niche spot')
        # 空间邻居
        _, _, spatial_idx = calculate_neighborhood_graph(
            adata, feature_key='spatial', k_cutoff=spatial_k, metric=spatial_metric, verbose=False
        )
        if verbose:
            print(f"Finding {spatial_k} spatial neighbors...    shape of neighbors {spatial_idx.shape}")

        # 特征邻居
        _, _, feature_idx = calculate_neighborhood_graph(
            adata, feature_key=feature_key, k_cutoff=feature_k, metric=feature_metric, verbose=False
        )
        if verbose:
            print(f"Finding {feature_k} feature neighbors...    shape of neighbors {feature_idx.shape}")

    except Exception as e:
        raise RuntimeError(f"Failed to find neighbors: {str(e)}")

    # 组合所有索引
    all_indices = np.concatenate([spatial_idx,feature_idx], axis=1)
    # print(f'spatial_idx: \n {spatial_idx}');print(f'feature_idx: \n {feature_idx}')
    # print(f'all_idx: \n {all_indices}')
    
    if remove_duplicates:
        all_indices = np.unique(all_indices, axis=1) #去重后按首行升序排列

    # 特征聚合
    neighbor_features = features[all_indices]
    print(f'Niche neighbor_features shape: {neighbor_features.shape}')
    
    if aggregation_method == 'mean':
        return np.mean(neighbor_features, axis=1),all_indices,neighbor_features, spatial_idx,feature_idx
    elif aggregation_method == 'max':
        return np.max(neighbor_features, axis=1),all_indices,neighbor_features
    elif aggregation_method == 'sum':
        return np.sum(neighbor_features, axis=1),all_indices,neighbor_features
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation_method}")

def prepare_adj(
    net_df,
    add_self_loops_flag: bool = True,
    make_undirected: bool = True,
    remove_existing_loops: bool = False,
) -> torch.Tensor:
    
    edge_index = torch.stack([
        torch.tensor(net_df['source'].values, dtype=torch.long),
        torch.tensor(net_df['target'].values, dtype=torch.long)
    ])
    edge_attr = torch.tensor(net_df['distance'].values, dtype=torch.long)
    
    if make_undirected:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    
    if add_self_loops_flag:
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        
    if remove_existing_loops:
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    
    return edge_index, edge_attr

class ExplicitModalityAttention(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_feat, 1),
            nn.Flatten(start_dim=-2)  
        )

    def forward(self, emb1, emb2):
        # Stack inputs along new modality dimension
        stacked = torch.stack([emb1, emb2], dim=-2)  # (..., 2, in_feat)
        
        # Compute raw attention scores
        raw_scores = self.attention(stacked)  # (..., 2)
        
        # Softmax normalization
        weights = torch.softmax(raw_scores, dim=-1)  # (..., 2)
        
        # Weighted fusion
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=-2)  # (..., in_feat)
        
        return fused, weights
    
class GatedModalityAttention(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        # 门控权重生成器（输入是 stacked 的拼接特征）
        self.gate = nn.Sequential(
            nn.Linear(in_feat, 1),  # 对每个模态独立计算门控分数
            nn.Sigmoid()            # 输出0~1的标量权重
        )
    
    def forward(self, emb1, emb2):
        # 保持原版输入形式：(..., 2, in_feat)
        stacked = torch.stack([emb1, emb2], dim=-2)
        
        # 计算门控权重 (..., 2, 1) -> (..., 2)
        gate_scores = self.gate(stacked).squeeze(-1)
        
        # 对 emb1 和 emb2 分别生成独立权重
        g1 = gate_scores[..., 0]  # emb1 的权重 (..., )
        g2 = gate_scores[..., 1]  # emb2 的权重 (..., )
        
        # 加权融合（门控公式）
        fused = g1.unsqueeze(-1) * emb1 + g2.unsqueeze(-1) * emb2
        
        # 返回融合结果和权重（可选项）
        weights = torch.stack([g1, g2], dim=-1)  # (..., 2)
        return fused, weights

class GraphEncoder(nn.Module):
    def __init__(
        self,
        channels: Union[int, List[int]],
        conv_type: str = "GCN",
        activation: str = "relu",
        final_activation: Optional[str] = None,  # 新增：最后一层激活函数
        heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # 统一处理 channels 参数
        channels = [channels] if isinstance(channels, int) else channels
        channels = channels * 2 if len(channels) == 1 else channels
        
        self.conv_type = conv_type
        self.activation_name = activation
        self.final_activation_name = final_activation  # 存储最后一层激活类型
        self.dropout = dropout
        self.heads = heads
        
        # 网络层构建
        self.conv_layers = self._make_conv_layers(channels, conv_type, heads, dropout)
        self.activation = self._get_activation(activation)
        self.final_activation = self._get_activation(final_activation)  # 最后一层激活
        
    def _make_conv_layers(self, channels, conv_type, heads, dropout) -> nn.ModuleList:
        layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_dim, out_dim = channels[i], channels[i+1]
            layers.append(self._get_conv_layer(conv_type, in_dim, out_dim, heads, dropout))
        return layers

    def _get_conv_layer(self, conv_type, in_dim, out_dim, heads, dropout):
        if conv_type == "GCN":
            return GCNConv(in_dim, out_dim)
        elif conv_type == "GAT":
            return GATConv(in_dim, out_dim//heads, heads=heads, dropout=dropout)
        elif conv_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            return GINConv(mlp)
        elif conv_type == "Transformer":
            return TransformerConv(in_dim, out_dim//heads, heads=heads, dropout=dropout)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

    def _get_activation(self, activation: Optional[str]) -> nn.Module:
        if activation is None:
            return nn.Identity()
            
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        
        if activation.lower() not in activations:
            warnings.warn(f"Unsupported activation: '{activation}'. Using Identity.", UserWarning)
            return nn.Identity()
        return activations[activation.lower()]

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:  
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:  # 最后一层处理
                x = self.final_activation(x)  
        return x

    def __repr__(self):
        layers_str = []
        for i, layer in enumerate(self.conv_layers):
            layers_str.append(f"  ({i}): {layer}")
            if i < len(self.conv_layers) - 1:  # 中间层标注
                layers_str.append(f"  (act_{i}): {self.activation}")
                if self.dropout > 0:
                    layers_str.append(f"  (dropout_{i}): Dropout(p={self.dropout})")
            elif self.final_activation_name is not None:  # 最后一层标注
                layers_str.append(f"  (final_act): {self.final_activation}")
        
        return (
            f"{self.__class__.__name__}(\n"
            f"  conv_type: {self.conv_type}\n"
            f"  activation: {self.activation_name}\n"
            f"  final_activation: {self.final_activation_name}\n"
            f"  dropout: {self.dropout}\n"
            f"  heads: {self.heads}\n"
            + "\n".join(layers_str) + "\n)"
        )



class IntraModelFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        conv_type: str = "GCN",
    ):
        super().__init__()
        self.encoder = GraphEncoder(channels=[hidden_dim],conv_type=conv_type)
        #self.attention = ExplicitModalityAttention(in_feat=hidden_dim)
        self.attention = GatedModalityAttention(in_feat=hidden_dim)
        

    def forward(
        self, 
        spot_features: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        niche_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        encoded_spot_features = self.encoder(spot_features, spatial_edge_index)
        fused, weights = self.attention(encoded_spot_features, niche_features)
        
        return fused
    
    def get_outputs(
        self,
        spot_features: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        niche_features: torch.Tensor,
    ) -> dict:
        """Convenience method to return all outputs"""
        fused = self(features, edge_index, niche_features)
        return {
            'fused_features': fused,
            'attention_weights': weights,
            'encoded_features': self.encoder(features, edge_index)
        }

class DualPathProject(nn.Module):
    def __init__(
        self,
        hidden_dim: List[int],
        conv_type: str = "GCN",
        hidden_dim_scale: float = 2.0
    ):
        super().__init__()
        # 1. Initialize intra-modality fusion models
        self.intra_fusion = IntraModelFusion(hidden_dim=hidden_dim[0],conv_type=conv_type)
        
        # 2. Define dual-path GNNs
        # Shared GNN path (2 layers)
        self.share_gnn = GraphEncoder(channels=hidden_dim,conv_type=conv_type)
        # Distinct GNN path (1 layers)
        self.distinct_gnn = GraphEncoder(channels=[hidden_dim[0],hidden_dim[-1]],conv_type=conv_type)

    def forward(
        self,
        spot_features: torch.Tensor,
        niche_features: torch.Tensor,
        spatial_edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Intra-modality fusion
        intra_fused = self.intra_fusion(spot_features, spatial_edge_index,niche_features)
        
        # Shared path processing
        shared_feature = self.share_gnn(intra_fused, spatial_edge_index)
        
        # Distinct path processing
        distinct_feature = self.distinct_gnn(intra_fused, spatial_edge_index)
        
        return shared_feature, distinct_feature

class GraphDecoder(GraphEncoder):
    def __init__(self, **kwargs):
        kwargs['conv_type'] = kwargs.get('conv_type', 'GCN')  # 默认使用GAT
        super().__init__(**kwargs)
        
class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).mean()
    
def abs_cos_sim(x, y):
    alignment_cos_sim = nn.CosineSimilarity(dim=1)
    return (alignment_cos_sim(x, y).abs()).mean() 

class FinalModalNetwork(nn.Module):
    def __init__(self, 
                 rna_feature_dim, 
                 img_feature_dim,
                 rna_niche_dim,
                 img_niche_dim,
                 hidden_dims=[64,32]):
        super(FinalModalNetwork, self).__init__()
        
        # RNA pathway
        self.rna_model = DualPathProject(hidden_dim=[rna_niche_dim] + hidden_dims)
        
        # Image pathway
        self.img_model = DualPathProject(hidden_dim=[img_niche_dim] + hidden_dims)
        
        # # Modality attention
        # self.inter_modal_attn = ExplicitModalityAttention(in_feat=hidden_dims[-1])
        
        # Decoders
        self.img_decoder = GraphDecoder(
            channels=[hidden_dims[-1], 64, img_feature_dim],
            conv_type="GCN"
        )
        
        self.rna_decoder = GraphDecoder(
            channels=[hidden_dims[-1], 64, rna_feature_dim],
            conv_type="GCN"
        )
    
    def forward(self, 
                rna_features, 
                img_features,
                rna_niche_features,
                img_niche_features,
                spot_edge_index):#zheg
        
        # Process RNA data
        rna_shared, rna_distinct = self.rna_model(
            spot_features=rna_features,
            niche_features=rna_niche_features,
            spatial_edge_index=spot_edge_index
        )
        
        # Process image data
        img_shared, img_distinct = self.img_model(
            spot_features=img_features,
            niche_features=img_niche_features,
            spatial_edge_index=spot_edge_index
        )
        
        # # Cross-modal attention
        # inter_fused, _ = self.inter_modal_attn(rna_shared, img_shared)
        
        # logitpool
        inter_fused = torch.logsumexp(torch.stack([rna_shared + img_shared, 
                                               rna_shared + img_shared, 
                                               rna_shared, 
                                               img_shared], dim=2), dim=2) - \
                  torch.logsumexp(torch.stack([torch.zeros_like(rna_shared), 
                                               torch.zeros_like(rna_shared), 
                                               rna_shared, 
                                               img_shared], dim=2), dim=2)
        
        # Reconstructions
        # img_distinct_recon = self.img_decoder(img_distinct, spot_edge_index)
        # img_share_recon = self.rna_decoder(inter_fused, spot_edge_index)
        # rna_distinct_recon = self.rna_decoder(rna_distinct, spot_edge_index)
        # rna_share_recon = self.img_decoder(inter_fused, spot_edge_index)
        img_distinct_recon = self.img_decoder(img_distinct, spot_edge_index)
        img_share_recon = self.img_decoder(inter_fused, spot_edge_index)
        rna_distinct_recon = self.rna_decoder(rna_distinct, spot_edge_index)
        rna_share_recon = self.rna_decoder(inter_fused, spot_edge_index)
        
        return {
            'img_distinct_recon': img_distinct_recon,
            'img_share_recon': img_share_recon,
            'rna_distinct_recon': rna_distinct_recon,
            'rna_share_recon': rna_share_recon,
            'inter_fused': inter_fused,
            'rna_shared': rna_shared,
            'rna_distinct': rna_distinct,
            'img_shared': img_shared,
            'img_distinct': img_distinct
        }
    
if __name__ == "__main__":
    # 创建测试数据
    cell_unm = 1000
    adata = AnnData(np.random.randn(cell_unm, 3000))
    adata.obsm['spatial'] = np.random.rand(cell_unm, 2) * 100
    adata.obsm['rna_features'] = np.random.randn(cell_unm, 200)
    adata.obsm['img_features'] = np.random.randn(cell_unm, 200)
    adata.obs_names = [f"cell_{i}" for i in range(cell_unm)]

    spot_adj, _, _ = calculate_neighborhood_graph(
        adata, feature_key='spatial', k_cutoff=8, metric='euclidean',
    )

    st = time.time()
    rna_niche_features,_,_,_,_ = aggregate_nichi_features(adata,feature_key='rna_features', spatial_k=8,feature_k=8,
                                                      spatial_metric='euclidean',aggregation_method='mean',remove_duplicates=True,verbose=True)
    end = time.time()
    print(end-st)
    print("Test passed. Output shape:", rna_niche_features.shape)

    st = time.time()
    img_niche_features,_,_,_,_ = aggregate_nichi_features(adata,feature_key='img_features', spatial_k=8,feature_k=8,
                                                      spatial_metric='euclidean',aggregation_method='mean',remove_duplicates=True,verbose=True)
    end = time.time()
    print(end-st)
    print("Test passed. Output shape:", img_niche_features.shape)

    spot_edge_index, _ = prepare_adj(spot_adj)
    
    # Initialize model
    model = FinalModalNetwork(
        rna_feature_dim=adata.obsm['rna_features'].shape[-1],
        img_feature_dim=adata.obsm['img_features'].shape[-1],
        rna_niche_dim=rna_niche_features.shape[-1],
        img_niche_dim=img_niche_features.shape[-1]
    )

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Prepare data tensors
    rna_features = torch.tensor(adata.obsm['rna_features'], dtype=torch.float32).to(device)
    img_features = torch.tensor(adata.obsm['img_features'], dtype=torch.float32).to(device)
    rna_niche = torch.tensor(rna_niche_features, dtype=torch.float32).to(device)
    img_niche = torch.tensor(img_niche_features, dtype=torch.float32).to(device)
    spot_edge_index = spot_edge_index.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Create directory for saving results
    os.makedirs('training_results', exist_ok=True)

    # Initialize lists to store loss values
    epoch_losses = []
    img_distinct_losses = []
    img_share_losses = []
    rna_distinct_losses = []
    rna_share_losses = []

    # Training loop with tqdm
    num_epochs = 10
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)

    for epoch in epoch_pbar:
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            rna_features=rna_features,
            img_features=img_features,
            rna_niche_features=rna_niche,
            img_niche_features=img_niche,
            spot_edge_index=spot_edge_index
        )

        # Calculate losses
        loss_img_distinct = loss_fn(outputs['img_distinct_recon'], img_features)
        loss_img_share = loss_fn(outputs['img_share_recon'], img_features)
        loss_rna_distinct = loss_fn(outputs['rna_distinct_recon'], rna_features)
        loss_rna_share = loss_fn(outputs['rna_share_recon'], rna_features)

        total_loss = loss_img_distinct + loss_img_share + loss_rna_distinct + loss_rna_share

        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()

        # Store loss values
        epoch_losses.append(total_loss.item())
        img_distinct_losses.append(loss_img_distinct.item())
        img_share_losses.append(loss_img_share.item())
        rna_distinct_losses.append(loss_rna_distinct.item())
        rna_share_losses.append(loss_rna_share.item())

        # Update progress bar description
        epoch_pbar.set_postfix({
            'Total Loss': f"{total_loss.item():.4f}",
            'ImgDist': f"{loss_img_distinct.item():.4f}",
            'ImgShare': f"{loss_img_share.item():.4f}",
            'RNADist': f"{loss_rna_distinct.item():.4f}",
            'RNAShare': f"{loss_rna_share.item():.4f}"
        })

    # Plot and save loss curves
    plt.figure(figsize=(12, 6))

    # Total loss
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses, label='Total Loss')
    plt.title('Total Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Component losses
    plt.subplot(1, 2, 2)
    plt.plot(img_distinct_losses, label='Img Distinct')
    plt.plot(img_share_losses, label='Img Share')
    plt.plot(rna_distinct_losses, label='RNA Distinct')
    plt.plot(rna_share_losses, label='RNA Share')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results/loss_curves.png')
    plt.close()

    # Save loss values to file
    import pandas as pd
    loss_df = pd.DataFrame({
        'Epoch': range(1, num_epochs+1),
        'Total_Loss': epoch_losses,
        'Img_Distinct': img_distinct_losses,
        'Img_Share': img_share_losses,
        'RNA_Distinct': rna_distinct_losses,
        'RNA_Share': rna_share_losses
    })
    loss_df.to_csv('training_results/loss_values.csv', index=False)

    # Save model
    torch.save(model.state_dict(), 'training_results/final_modal_model.pth')

    print("Training completed. Results saved in 'training_results' directory.")

#     # Initialize
#     model = FinalModalNetwork(
#         rna_feature_dim=adata.obsm['rna_features'].shape[-1],
#         img_feature_dim=adata.obsm['img_features'].shape[-1],
#         rna_niche_dim=rna_niche_features.shape[-1],
#         img_niche_dim=img_niche_features.shape[-1]
#     )

#     # Forward pass
#     outputs = model(
#         rna_features=torch.tensor(adata.obsm['rna_features'], dtype=torch.float32),
#         img_features=torch.tensor(adata.obsm['img_features'], dtype=torch.float32),
#         rna_niche_features=torch.tensor(rna_niche_features, dtype=torch.float32),
#         img_niche_features=torch.tensor(img_niche_features, dtype=torch.float32),
#         spot_edge_index=spot_edge_index
#     )

#     import torch.optim as optim
#     from tqdm import tqdm  # for progress bar

#     # Initialize model
#     model = FinalModalNetwork(
#         rna_feature_dim=adata.obsm['rna_features'].shape[-1],
#         img_feature_dim=adata.obsm['img_features'].shape[-1],
#         rna_niche_dim=rna_niche_features.shape[-1],
#         img_niche_dim=img_niche_features.shape[-1]
#     )

#     # Move model to GPU if available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.float().to(device)

#     # Prepare data tensors
#     rna_features = torch.tensor(adata.obsm['rna_features'], dtype=torch.float32).to(device)
#     img_features = torch.tensor(adata.obsm['img_features'], dtype=torch.float32).to(device)
#     rna_niche = torch.tensor(rna_niche_features, dtype=torch.float32).to(device)
#     img_niche = torch.tensor(img_niche_features, dtype=torch.float32).to(device)
#     spot_edge_index = spot_edge_index.to(device)

#     # Define optimizer and loss function
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     loss_fn = nn.MSELoss()  # Mean Squared Error for reconstruction

#     # Training loop
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(
#             rna_features=rna_features,
#             img_features=img_features,
#             rna_niche_features=rna_niche,
#             img_niche_features=img_niche,
#             spot_edge_index=spot_edge_index
#         )

#         # Calculate individual losses
#         loss_img_distinct = loss_fn(outputs['img_distinct_recon'], img_features)
#         loss_img_share = loss_fn(outputs['img_share_recon'], img_features)
#         loss_rna_distinct = loss_fn(outputs['rna_distinct_recon'], rna_features)
#         loss_rna_share = loss_fn(outputs['rna_share_recon'], rna_features)

#         # Total loss (sum of all four losses)
#         total_loss = loss_img_distinct + loss_img_share + loss_rna_distinct + loss_rna_share

#         # Backward pass and optimize
#         total_loss.backward()
#         optimizer.step()

#         # Print progress
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}, '
#               f'ImgDistinct: {loss_img_distinct.item():.4f}, '
#               f'ImgShare: {loss_img_share.item():.4f}, '
#               f'RNADistinct: {loss_rna_distinct.item():.4f}, '
#               f'RNAShare: {loss_rna_share.item():.4f}')
        
    print(torch.cuda.memory_summary())

    # After training, you can save the model if needed
    # torch.save(model.state_dict(), 'final_modal_model.pth')

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)