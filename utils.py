import scanpy as sc
import pandas as pd
import sklearn.neighbors
import numpy as np
import anndata
import random
import torch
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler,scale

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def generate_adata(X, spatial=None, n_neighbors=20, n_pcs=30):
    adata = sc.AnnData(X)
    if spatial is not None:
        #adata.obsm['spatial'] = np.array(obs.loc[:,["img_x","img_y"]])
        adata.obsm['spatial'] = np.array(spatial)
    sc.pp.neighbors(adata,use_rep='X',n_neighbors=n_neighbors, n_pcs=n_pcs)
    return adata

def find_res_binary(adata, resolution_min, resolution_max, num_clusters, method = 'leiden',key_added = 'cluster'):
    # Use binary search to find the resolution parameter that satisfies the condition
    if method == 'leiden':
        sc.tl.leiden(adata, resolution=resolution_max, key_added=key_added)
    elif method == 'louvain':
        sc.tl.louvain(adata, resolution=resolution_max, key_added=key_added)
    if int(len(np.unique(adata.obs[key_added]))) < int(num_clusters):
        resolution_max = resolution_max+1
        print('Number of clusters at the maximum resolution is less than %s, adjust maximum resolution to %s' % (num_clusters,resolution_max))
    
    while resolution_min <= resolution_max:
        # Perform Leiden clustering
        resolution = (resolution_min + resolution_max) / 2
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added)

        # Check the number of unique clusters in the clustering result
        unique_clusters = np.unique(adata.obs[key_added])
        
        # Print the current progress
        print(f"Resolution: {resolution_min, resolution_max, resolution}, Unique Clusters: {len(unique_clusters)}")

        if len(unique_clusters) == num_clusters:
            break
        elif len(unique_clusters) < num_clusters:
            resolution_min = resolution
            # resolution = (resolution + resolution_max) / 2
        else:
            resolution_max = resolution
            # resolution = (resolution_min + resolution) / 2
            
        if resolution_max - resolution_min < 1e-6:
            break
            
    return resolution, adata

def plot_spatial(adata,save_path,title=None,group='cluster',set_scale = 6):
    scale = (adata.obsm['spatial'][:,0].max()-adata.obsm['spatial'][:,0].min())/(adata.obsm['spatial'][:,1].max()-adata.obsm['spatial'][:,1].min())
    plt.rcParams["figure.figsize"] = (set_scale*scale,set_scale)
    if 'spatial' in adata.uns:
        sc.pl.spatial(adata, img_key="hires", color=[group],basis="spatial", title = title,size=1, alpha_img=0.8, alpha=0.5, show=False)
    else:
        sc.pl.embedding(adata, color=[group],basis="spatial", title = title,size=20, show=False)
    plt.savefig(os.path.join(save_path,str(title+'.png')), bbox_inches='tight')
    plt.show()

def cal_spatial_net(adata, rad_cutoff=None, k_cutoff=None, map_id=True, verbose=True):
    #assert(model in ['Radius', 'KNN'])
    if verbose:
        print('Calculating spatial location graph......')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    #coor.columns = ['imagerow', 'imagecol']

    if rad_cutoff is not None:
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if k_cutoff is not None:
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))
        
    if map_id:
        #Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
        id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
        Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
        Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
        return Spatial_Net,id_cell_trans
    else:
        return Spatial_Net
    
def edgelist2adj(edgelist, weight = None):
    import networkx as nx
    G = nx.Graph()

    # Add the connections to the graph
    for _, row in edgelist.iterrows():
        cell1 = row['Cell1']
        cell2 = row['Cell2']
        
        if weight is not None:
            G.add_edge(cell1, cell2, weight=weight)
        else:
            G.add_edge(cell1, cell2)

    # Convert the graph to a connection matrix
    adjacency_matrix = nx.to_numpy_array(G)
    
    return adjacency_matrix

    
def pruning_knn(knn_df,clustet_df):
    filtered_df = pd.DataFrame()
    if clustet_df.shape[1] == 2:
        print('pruning use 2 modality')
        for index, row in knn_df.iterrows():
            cell1 = row['Cell1']
            cell2 = row['Cell2']

            rna = clustet_df.loc[cell1,'cluster_x']==clustet_df.loc[cell2,'cluster_x']
            img = clustet_df.loc[cell1,'cluster_y']==clustet_df.loc[cell2,'cluster_y']
            #print(cell1,cell2,rna,img)
            if (rna or img):
                filtered_df = filtered_df._append(row, ignore_index=True)
    elif clustet_df.shape[1] == 1:
        print('pruning use 1 modality')
        for index, row in knn_df.iterrows():
            cell1 = row['Cell1']
            cell2 = row['Cell2']

            col_name = clustet_df.columns
            #print(cell1,cell2,int(clustet_df.loc[cell1,col_name]),int(clustet_df.loc[cell2,col_name]),(clustet_df.loc[cell1,col_name]==clustet_df.loc[cell2,col_name]).any())
            if (clustet_df.loc[cell1,col_name]==clustet_df.loc[cell2,col_name]).any():
                filtered_df = filtered_df._append(row, ignore_index=True)
    return filtered_df 

def index_knn(knn_df,id_cell_trans):
    trans = {value: key for key, value in id_cell_trans.items()}
    index_knn_df = knn_df.copy()
    index_knn_df ['Cell1'] = index_knn_df ['Cell1'].map(trans)
    index_knn_df ['Cell2'] = index_knn_df ['Cell2'].map(trans)
    return index_knn_df

def get_cluster_id(adata,res,method='leiden'):
    adata = select_res(adata,res,method,plot=False,title=None)
    cluter_id = pd.DataFrame(adata.obs.loc[:,'cluster'])
    return cluter_id
    
def purning_by_cluster(knn_df,rna_data,img_data,init_res):
    if isinstance(rna_data, anndata.AnnData):
        rna_adata = rna_data
    else:
        rna_adata = generate_adata(rna_data)
        
    if isinstance(img_data, anndata.AnnData):
        img_adata = img_data
    else:
        img_adata = generate_adata(img_data)
    
    if rna_data is not None and img_data is not None:
        rna_cluster = get_cluster_id(rna_adata,res=init_res,method='leiden')
        img_cluster = get_cluster_id(img_adata,res=init_res,method='leiden')
        clus_df = pd.merge(rna_cluster,img_cluster,left_index=True,right_index=True, how='left')
    elif rna_data is None:
        clus_df = get_cluster_id(img_adata,res=init_res,method='leiden')
    elif img_data is None:
        clus_df = get_cluster_id(rna_adata,res=init_res,method='leiden')
    
    prun_knn_df = pruning_knn(knn_df,clus_df)
    print('edges of raw knn graph: %s. edges of purning knn graph: %s' % (knn_df.shape[0],prun_knn_df.shape[0]))
    return prun_knn_df

def extract_rna_feat(adata,num_feat=2048,dim_reduction_method = 'high_var'):
    if type(adata.X) is np.ndarray:
        data = adata.X
    else:
        data = adata.X.toarray()
    if dim_reduction_method == 'pca':
        pca = PCA(n_components=num_feat)
        rna_df = pca.fit_transform(data)
    elif dim_reduction_method == 'high_variable':
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=num_feat)
        rna_df = adata[:, adata.var['highly_variable']].X
    elif dim_reduction_method == 'high_var':
        variances = np.var(data, axis=0)
        top_n = np.argsort(variances)[-num_feat:]
        high_var_genes = adata.var.index[top_n]
        rna_df = adata[:,high_var_genes].X.toarray()
    elif dim_reduction_method == None:
        rna_df = adata.X
    
    return(pd.DataFrame(rna_df))

def scale_data(data,scaler='zscore'):
    if scaler=='zscore':
        scaler = StandardScaler()
    elif scaler=='ninmax':
        scaler = MinMaxScaler()
    
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def load_graph_edgelist(edgelist_path):
    edgelist = []
    with open(edgelist_path, 'r') as edgelist_file:
        edgelist = [(int(item.split()[0]), int(item.split()[1])) for item in edgelist_file.readlines()]
    return edgelist

def plot_loss_curve(loss_values,out_dir):
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'b', label='Training Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir,'Train_loss_curve.png'))
    plt.show()

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

def find_res_sigmoid(adata, cluster_range=(12,13), by=0.1, res=1, verbose=False):
    if verbose:
        print("Find suitable resolution, start with", res)

    if isinstance(cluster_range, int):
        cluster_range = [cluster_range, cluster_range]
    elif isinstance(cluster_range, tuple) and len(cluster_range) == 2:
        cluster_range = cluster_range

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x = -np.log(10/res - 1)
    plus_counter = minus_counter = 0
    n_cluster = 1

    while n_cluster < cluster_range[0] or n_cluster > cluster_range[1]:
        sc.tl.leiden(adata, resolution=res, key_added='clusters')
        n_cluster = len(np.unique(adata.obs['clusters']))

        if n_cluster < cluster_range[0]:
            x = x + by
            plus_counter = plus_counter + 1
        elif n_cluster > cluster_range[1]:
            x = x - by
            minus_counter = minus_counter + 1
        else:
            break

        res = round(sigmoid(x) * 10, 3)

        if plus_counter and minus_counter:
            print()
            raise ValueError("Specific cluster range was skipped! Try expanding the cluster range or reducing the resolution step size.")

        if verbose:
            print("resolution", res, "... ",n_cluster, "clusters.")

    adata.uns['best_resolution'] = res

    if verbose:
        print("Final resolution:", res, "with", n_cluster, "clusters.")

    return adata

def select_res(adata,res,method='leiden',plot=False,save_path=None,title=None):
    if method == 'leiden':
        sc.tl.leiden(adata, resolution=res, key_added='cluster')
    elif method == 'louvain':
        sc.tl.louvain(adata, resolution=res, key_added='cluster')
    #print(adata.obs.cluster.unique())
        
    if plot == True:
        if 'spatial' in adata.obsm.keys():
            scale = (adata.obsm['spatial'][:,0].max()-adata.obsm['spatial'][:,0].min())/(adata.obsm['spatial'][:,1].max()-adata.obsm['spatial'][:,1].min())
            plt.rcParams["figure.figsize"] = (8*scale,8)
            sc.pl.embedding(adata, color=["cluster"],basis="spatial", title = title,size=20, show=False)
        else:
            sc.tl.umap(adata)
            sc.pl.embedding(adata, color=["cluster"], title = title,size=20, show=False)

        # plt.savefig('./resolution %s for purning.png' % res, bbox_inches='tight')
        plt.savefig(os.path.join(save_path,str(title+'.png')), bbox_inches='tight')
        plt.show()
    return adata

def find_res_step(adata, cluster_num = 12, res_range = np.around(np.arange(0.3,0.9,0.04), 3), determine_clus_num = True,criterion = 'CH_score'):
    if determine_clus_num == True:
        print('choose res by number of clusters')
        # adata = find_clusters(adata, cluster_range=cluster_num, by=0.1, res=1, verbose=False)
        ## choose res by number of cluster
        cluster_num = cluster_num
        found = False

        for res in res_range:
            sc.tl.leiden(adata, resolution=res)#,key_added=str("res"+str(res))
            num = len(adata.obs.leiden.unique())
            print(str(res),":", num)
            if num == cluster_num:
                print('res',str(res),'reach number of cluser',str(cluster_num))
                found = True
                return res

        if not found:
            print("No suitable resolution found") 
            return res
            
    else:
        print('choose res by best cluster scores')
        ## choose res by best cluster scores
        #CH_scores = DB_scores = sil_scores = []
        res_outs = []

        for res in res_range:
            sc.tl.leiden(adata, resolution=res)#,key_added=str("res"+str(res))
            labels=vgae.obs['leiden']
            CH_score = metrics.calinski_harabasz_score(emb, labels)
            DB_score = metrics.davies_bouldin_score(emb, labels)
            sil_score = metrics.silhouette_score(emb, labels)
            num = str(len(vgae.obs.leiden.unique()))
            print(str(str(res)+"_"+num),":",DB_score,CH_score,sil_score)
            res_out = [res,num,DB_score,CH_score,sil_score]
            res_outs.append(res_out)

        res_df = pd.DataFrame(res_outs,columns=['res','DB_score','CH_score','sil_score'])
        criterion = criterion
        if criterion in ['CH_score','sil_score']:
            best_res = res_df.iloc[res_df[criterion].idxmax(),0]
            print('best res choosen by',criterion,':',str(best_res))
        elif criterion == 'DB_score':
            best_res = res_df.iloc[res_df[criterion].idxmin(),0]
            print('best res choosen by',criterion,':',str(best_res))
        return res
        
            
def definite_res(adata,res,save_path=None,plot_file='choose_res_spatial_plot.png',title=None):
    sc.tl.leiden(adata, resolution=res)
    #print(adata.obs.leiden.unique())
    scale = (adata.obsm['spatial'][:,0].max()-adata.obsm['spatial'][:,0].min())/(adata.obsm['spatial'][:,1].max()-adata.obsm['spatial'][:,1].min())
    plt.rcParams["figure.figsize"] = (8*scale,8)
    sc.pl.embedding(adata, basis="spatial", color=["leiden"],title = title,size=20, show=False)
    #plt.savefig(os.path.join(save_path,plot_file))
    #plt.show()
    return adata
