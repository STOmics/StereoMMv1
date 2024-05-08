import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from utils import *
from models import *
from trainer import *
import sys
import random
import pickle
import warnings
warnings.filterwarnings("ignore")

###### Version and Date
PROG_VERSION = '1.0'
PROG_DATE = '2023-09-20'

###### Usage
usage = '''

Version %s  by Luo Bingying  %s

Usage: %s --rna_data <adata_file> --image_data <image_file> -o <output_dir> [...]

''' % (PROG_VERSION, PROG_DATE, os.path.basename(sys.argv[0]))

def main():
    import argparse
    def int_or_none(value):
        if value is None:
            return None
        return int(value)

    ArgParser = argparse.ArgumentParser(usage=usage,description='run model')
    ArgParser.add_argument("--version", action="version", version=PROG_VERSION)
    ArgParser.add_argument("--sessioninfo", dest="sessioninfo",required=False, action='store_true', default=False, help="Print conda list.")
    ArgParser.add_argument("--verbose", dest="verbose",required=False, action='store_true', default=False, help="Print cuda memory use.")

    ArgParser.add_argument("-t", "--toy", action="store_true", dest="toy_data", default=False, help="Whether to use toy data.")
    ArgParser.add_argument("--rna_data", action="store", dest="rna_data", required=False, type=str, help="Path of rna feature file.")
    ArgParser.add_argument("--image_data", action="store", dest="image_data", required=False, type=str, help="Path of H&E image feature file.")
    ArgParser.add_argument("-o", "--output", action="store", dest="output", required=True, type=str, help="Output folder.")
    ArgParser.add_argument("--epochs", dest="epochs",required=True, type=int, default=100, help="Number of training epochs.")
    ArgParser.add_argument("--lr", dest="lr", required=False, type=float, default=0.0001, help="Learning rate for training.")
    ArgParser.add_argument("--opt", dest="opt", required=False, type=str, default="adam", help="Optimizer for training.")
    ArgParser.add_argument("--dim_reduction_method", dest="dim_reduction_method",required=False, type=str, default="high_var", help="RNA data dim reduction method.")
    ArgParser.add_argument("--scale", dest="scale",required=False, type=str, default="zscore", help="Feature normalization method.")
    ArgParser.add_argument("--feat_pca", dest="feat_pca",required=False, action='store_true', default=False, help="PCA when ectract feature.")
    ArgParser.add_argument("--radiu_cutoff", dest="radiu_cutoff",required=False, type=int_or_none, default=None,help="Radiu for KNN graph.")
    ArgParser.add_argument("--knn", dest="knn",required=False, type=int_or_none, default=None, help="K for KNN graph.")
    ArgParser.add_argument("--purning", dest="purning_knn",required=False, action='store_true', default=False, help="Prune the knn graph")
    ArgParser.add_argument("--num_heads", dest="num_heads",required=False, type=int, default=1, help="Number of attention heads for cross attention layer.")
    ArgParser.add_argument("--hidden_dims", dest="hidden_dims",required=False, type=int, default=[1024, 512], help="Hidden dimension for each hidden layer.", nargs="*")
    ArgParser.add_argument("--latent_dim", dest="latent_dim",required=False, type=int, default=100, help="Latent dimension (output dimension for node embeddings).")
    ArgParser.add_argument("--brief_att", dest="brief_att",required=False, action='store_true', default=False, help="Attention layer with fewer parameters.")
    ArgParser.add_argument("--att_mode", dest="att_mode",required=False, action='store', default='cross', help="Attention layer type.")
    ArgParser.add_argument("--decoder", dest="customize_decoder",required=False, action='store_true', default=False, help="Construct the neural network decoder.")
    ArgParser.add_argument("--gnn_type", dest="gnn_type",required=False, type=str, default="GCN", help="Graph Neural Network type.")
    ArgParser.add_argument("--rna_weight", dest="rna_weight",required=False, type=int, default=1, help="Weight of rna attention to concat.")
    ArgParser.add_argument("--img_weight", dest="img_weight",required=False, type=int, default=1, help="Weight of img attention to concat.")
    ArgParser.add_argument("--dec", action="store_true", dest="dec", default=False, help="Whether to add DEC for cluster.")
    ArgParser.add_argument("--n_cluster", dest="n_cluster",required=False, type=int, default=10, help="number of clustered categories.")
    # ArgParser.add_argument("--res_range", dest="res_range", required=False, default=np.around(np.arange(0.2,1.2,0.03), 3), type=float,help="resolution range for clustering.")
    
    (para, args) = ArgParser.parse_known_args()
    print(para)
    
    set_seed(77)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("number of GPUs：", num_gpus)
        current_gpu = torch.cuda.current_device()
        print("The GPU index currently in use：", current_gpu)
        print(f'CUDA available, using {torch.cuda.get_device_name(device)}.')
    else:
        print('CUDA not available, use CPU')
        
    out_dir = para.output
    makedir(out_dir)

    if para.toy_data:
        # Create the number of nodes and feature dimensions
        num_nodes = 100
        num_features = 2000

        img_tensor = torch.randn(num_nodes, num_features).double().to(device)
        rna_tensor = torch.randn(num_nodes, num_features).double().to(device)

        edge_index = torch.randint(num_nodes, (2, num_nodes * 2))
        edge_index = edge_index.to(device)

        # single_model_size=50
        # hidden_size=[32,16] 
        # latent_size=10 

    else:
        ###========== Feature processing of each modality ==========###
        img_feat = pd.read_pickle(para.image_data)
        rna_adata = sc.read_h5ad(para.rna_data)

        ### Single modality preprocessing
        rna_feat = extract_rna_feat(rna_adata,num_feat=2048,dim_reduction_method = para.dim_reduction_method)
        if not para.dim_reduction_method == 'pca':
            rna_feat = scale_data(rna_feat,scaler=para.scale)
        img_feat = scale_data(img_feat,scaler=para.scale)
        
        if para.feat_pca == True:
            print('use PCA when excrate single feature')
            pca = PCA(n_components=200)
            rna_feat = pca.fit_transform(rna_feat)
            img_feat = pca.fit_transform(img_feat)
            print('final rna feature shape: %s. image feature shape: %s' % (rna_feat.shape,img_feat.shape))

        rna_tensor = torch.from_numpy(np.array(rna_feat)).double().to(device)
        img_tensor = torch.from_numpy(np.array(img_feat)).double().to(device)
        
        print('input image feature shape: %s. \n input rna feature shape: %s.' % (img_tensor.shape,rna_tensor.shape))
        
        ### knn
        graph_file_path = "knn.txt"
        if os.path.exists(os.path.join(out_dir,'knn.txt')):
            print('The KNN file already exists in %s, read it directly' % os.path.join(out_dir,'knn.txt')) 
        else:
            knn_df,id_cell_trans = cal_spatial_net(rna_adata,rad_cutoff=para.radiu_cutoff,k_cutoff=para.knn, map_id=True)

            # Whether to prune
            if para.purning_knn:
                knn_df = purning_by_cluster(knn_df=knn_df,rna_data=rna_adata,img_data=img_feat,init_res=0.4)

            index_knn_df = index_knn(knn_df,id_cell_trans)
            index_knn_df  
            index_knn_df.iloc[:,0:2].to_csv(os.path.join(out_dir,'knn.txt'),index=False,sep=' ',header=False)

        edgelist = load_graph_edgelist(os.path.join(out_dir,'knn.txt'))
        edge_index = np.array(edgelist).astype(int).T
        edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long))
        edge_index = edge_index.to(device)

    single_model_size=img_tensor.shape[1]
    hidden_size=para.hidden_dims
    latent_size=para.latent_dim

    # Inspecting the data used for graph convolution
    data_obj = Data(edge_index=edge_index, x=img_tensor)
    print('Data inspection results for graph convolutional networks: %s. \n' % data_obj.validate(raise_on_error=True))

    ###========== CONSTRUCT ==========###
    import time
    start_time = time.time()
    
    if para.customize_decoder:
        decoder_hidden_size = hidden_size.copy()
        decoder_hidden_size.reverse()
        decoder_hidden_size.insert(0,latent_size)

        decoder = Decoder(output_size=single_model_size*2,hidden_sizes=decoder_hidden_size, gnn_type = para.gnn_type)
        fimodal = FinalModal(single_model_size, hidden_size, latent_size, num_heads=para.num_heads, brief_att = para.brief_att, attn_mode=para.att_mode,gnn_type = para.gnn_type, img_weight = para.img_weight, rna_weight = para.rna_weight, decoder = decoder)

    else:
        fimodal = FinalModal(single_model_size, hidden_size, latent_size, num_heads=para.num_heads, brief_att = para.brief_att, attn_mode=para.att_mode)

    fimodal = fimodal.double().to(device)
    fimodal.print_networks()
    current_memory = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
    print(f"CUDA memory consumption at the current stage: {current_memory} MB")

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # num_params = count_parameters(fimodal)
    # print("Number of trainable parameters: ", num_params)

    ###========== TRAINING ==========###
    torch.cuda.empty_cache()
    #attn_out_dir = os.path.join(out_dir,'attn_weight')
    trainer = train(img_tensor, rna_tensor, edge_index, fimodal, custom_decoder = para.customize_decoder,n_epochs=para.epochs, opt=para.opt, lr=para.lr, weight_decay=0.0001, save_att=False, verbose=True)#, attn_out = attn_out_dir
    if para.dec :
        loss_values, img_emb, rna_emb, attention_weight = trainer.train_dec(init = "kmeans", n_cluster = para.n_cluster, n_neighbors = 20, max_epochs = 100, update_interval = 10, tol = 1e-5, alpha = 0.9)
        flag = 'dec'
    else:
        loss_values, img_emb, rna_emb, attention_weight = trainer.train_ae()
        flag = 'ae'
    emb = trainer.predict()
    
    if para.verbose:
        # print(u'Current memory usage：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        print(torch.cuda.max_memory_allocated())
        print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()
    
    end_time = time.time()
    run_time = end_time - start_time

    # Save data
    emb = pd.DataFrame(emb)
    emb.to_pickle(os.path.join(out_dir,'embedding.pkl'))
    emb.to_csv(os.path.join(out_dir,'embedding.csv'))
    with open(os.path.join(out_dir,'att_weight.pkl'), 'wb') as file:
        pickle.dump(attention_weight, file)
    img_emb = pd.DataFrame(img_emb.cpu().detach().numpy())
    img_emb.to_pickle(os.path.join(out_dir,'img_embedding.pkl'))
    rna_emb = pd.DataFrame(rna_emb.cpu().detach().numpy())
    rna_emb.to_pickle(os.path.join(out_dir,'rna_embedding.pkl'))

    # Draw the loss curve
    loss_values = pd.DataFrame(loss_values)
    loss_values.to_csv(os.path.join(out_dir,'loss.csv'))
    try:
        plot_loss_curve(loss_values,out_dir)
    except:
        print('Error when plot_loss_curve')
    
    # clustering
    if not para.toy_data:
        emb.index = rna_adata.obs.index
        emb.to_pickle(os.path.join(out_dir,'embedding.pkl'))
        emb.to_csv(os.path.join(out_dir,'embedding.csv'))
        #spatial_coord = pd.read_csv('./spatial.csv',index_col=0)
        stereomm_adata = generate_adata(emb , spatial=rna_adata.obsm['spatial'])
        
        resolution, stereomm_adata = find_res_binary(stereomm_adata, resolution_min=0.1, resolution_max=1.2, num_clusters=para.n_cluster)
        print(f"Final Resolution: {resolution}")
        plot_spatial(stereomm_adata,out_dir,'StereoMM_domain')
        
        stereomm_adata.write(os.path.join(out_dir,'emb_adata.h5ad'))

    print('======= Finish!!!!!!======')
    print('RUNING TIME: %s' % (run_time))
    
    if para.sessioninfo:     
        os.system('pip list')
    
if __name__ == "__main__":
    main()
