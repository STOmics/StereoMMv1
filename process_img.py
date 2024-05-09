import os 
import argparse
import matplotlib.pyplot as plt
import scanpy as sc
#import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm
import anndata as ad
from matplotlib import pyplot as plt
from sklearn import metrics
from torchvision import transforms
from torchvision import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import warnings
import sys
from torch.utils.data import Dataset
from utils import find_res_binary,plot_spatial
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
print(device)

Image.MAX_IMAGE_PIXELS = None

###### Version and Date
PROG_VERSION = '1.0'
PROG_DATE = '2023-08-31'

###### Usage
usage = '''

Version %s  by Luo Bingying  %s

Usage: %s -a <adata_file> -i <gem_file> -o <output_dir> [...]

''' % (PROG_VERSION, PROG_DATE, os.path.basename(sys.argv[0]))


def add_img2adata(img,adata,bin_size = 100, library_id = "cancer",spatial_key = "spatial"):
    adata.var_names_make_unique()
    adata.uns[spatial_key] = {library_id: {}}
    adata.uns[spatial_key][library_id]["images"] = {}
    img_arr =  np.array(img)
    adata.uns[spatial_key][library_id]["images"] = {"hires": img_arr}
    adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 100}
    adata.obsm['spatial'][:,0] = adata.obsm['spatial'][:,0]-adata.obsm['spatial'][:,0].min()+bin_size/2
    adata.obsm['spatial'][:,1] = adata.obsm['spatial'][:,1]-adata.obsm['spatial'][:,1].min()+bin_size/2
    adata.obs['img_x'] = adata.obsm['spatial'][:,0]
    adata.obs['img_y'] = adata.obsm['spatial'][:,1]
    adata.obs['cell_id'] = adata.obs.index
    split_data_1=adata.obs['cell_id'].astype('str').str.split('_',expand=True)
    split_data_1.columns=['array_row','array_col']
    adata.obs=adata.obs.join(split_data_1)
    adata.uns["spatial"][library_id]["use_quality"] = 'hires'
    return adata

def image_crop(image,adata,slide_gem=False,save_path='./',crop_size=128,verbose=False,):
    tile_names = []

    with tqdm(total=len(adata),
              desc="Tiling image",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        
        if slide_gem == False:
            crop_coord = zip(adata.obs["img_x"], adata.obs["img_y"])
        else:
            crop_coord = zip(adata.obs["x"], adata.obs["y"])
        
        for img_x, img_y in crop_coord:
            
            tile_name = str(img_x) + "-" + str(img_y) + "-" + str(crop_size)
            tile_path = os.path.join(save_path,tile_name+'.tiff')
            #print(tile_path)
            tile_names.append(str(tile_path))
            
            try:
                x1 = img_x - crop_size / 2 
                x2 = img_x + crop_size / 2 
                y1 = img_y - crop_size / 2 
                y2 = img_y + crop_size / 2 
                # tile = image[int(y1):int(y2),int(x1):int(x2)]#[y1:y2, x1:x2]
                # cv2.imwrite(tile_path, tile)
                box = (int(x1), int(y1), int(x2), int(y2))# (left, upper, right, lower)
                tile = image.crop(box)  
                tile.save(tile_path)
            except:
                print(tile_path," error, generate black image")
                # tile = np.zeros((crop_size, crop_size, 3), np.uint8)
                # tile[:] = [49,0,0]
                # cv2.imwrite(tile_path, tile)  
                tile = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
                tile.save(tile_path)
            
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(
                        str(img_x), str(img_y)))
            pbar.update(1)
        adata.obs["slices_path"] = tile_names
    return adata

def plot_spatial_with_img(adata,save_path,file):
    scale = (adata.obsm['spatial'][:,0].max()-adata.obsm['spatial'][:,0].min())/(adata.obsm['spatial'][:,1].max()-adata.obsm['spatial'][:,1].min())
    plt.rcParams["figure.figsize"] = (8*scale,8)
    #sc.pl.embedding(adata, basis="spatial", title='VGAE_domain',color=["leiden"],size=20, show=False)
    sc.pl.spatial(adata, img_key="hires", color="leiden",basis="spatial", size=1.5,alpha_img=0.4)#,crop_coord=[0,11550,0,12600]
    plt.savefig(os.path.join(save_path,file), bbox_inches='tight')
    plt.show()
    return None
    
def save_data(adata,save_path,label_txt_file,adata_file):
    label_txt = adata.obs[['slices_path','leiden']]
    label_txt.to_csv(os.path.join(save_path,label_txt_file),sep = ' ', header=0, index= False)
    adata.write(os.path.join(save_path,adata_file))
    
class MyDataset(Dataset): 
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r') 
        imgs = []  
        for line in fh:
            line = line.rstrip() 
            words = line.split() 
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs       
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # img = Image.open(fn).convert('RGB')    
        img = Image.open(fn) 

        if self.transform is not None:
            img = self.transform(img)   

        return img, label

    def __len__(self):
        # assert len(self.image_list) == len(self.label_list)
        return len(self.imgs)   
    
def load_model():
    model_com = models.resnet50(pretrained=False)
    current_path = os.getcwd()
    if 'StereoMMv1' in current_path:
        pth_path = os.path.join(current_path, "torch_pths/resnet50-19c8e357.pth")
    else:
        pth_path = os.path.join(current_path, "StereoMMv1/torch_pths/resnet50-19c8e357.pth")
    model_com.load_state_dict(torch.load(pth_path))
    num_features = model_com.fc.in_features

    ### strip the last layer
    feature_extractor = torch.nn.Sequential(*list(model_com.children())[:-1])
    # Get the feature extractor up to layer4
    # feature_extractor_layer4 = torch.nn.Sequential(*list(model.children())[:-2])

    model_com.to(device)
    model_com.eval()

    feature_extractor.to(device)
    feature_extractor.eval()
    return(feature_extractor,model_com)

def feature_extractor(model,dataset):
    feat_outputs = []
    model.eval()
    # Disable gradient calculation to improve inference speed
    with torch.no_grad():
        with tqdm(total=len(dataset),
              desc="calculate image feature",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            # Iterate over the test set samples
            for i in range(len(dataset)):
                # Get the i-th sample from the test set
                inputs, target = dataset[i][0], torch.tensor(dataset[i][1])

                # Move the data to the device (e.g. GPU)
                inputs, target = inputs.to(device), target.to(device)

                # Compute the model prediction for the sample
                output = model(inputs.unsqueeze(0))
                #_, predicted = torch.max(output.data, 1)
                # Store the FC layer output
                output = output.data.cpu().numpy().ravel()
                feat_outputs.append(output)

                pbar.update(1)
    return feat_outputs
    
    
def extract_img_feat(tile_file,extractor):
    transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = MyDataset(tile_file, transforms_val)
    
    # if extractor=='feature_extractor':
    #     model = feature_extractor
    # else:
    #     model = model_com
        
    feat_outputs = feature_extractor(model=extractor,dataset=dataset)
    feat_outputs = pd.DataFrame(feat_outputs)
    return feat_outputs
    
                
def save_img_feat(data,save_path,feat_file):
    print('shape of image_feature：',data.shape)
    data.to_pickle(os.path.join(save_path,feat_file))
    
def generate_adata(X,spatial=None):
    # adata = ad.AnnData(X,obs=obs)
    # adata.obsm['spatial'] = np.array(obs.loc[:,["img_x","img_y"]])
    adata = ad.AnnData(X)
    adata.obsm['spatial'] = np.array(spatial)
    sc.pp.neighbors(adata,use_rep='X',n_neighbors=20, n_pcs=30)
    return adata
  
def choose_res(adata,res_range,method = 'num_cluster',cluster_num = 10,criterion = 'CH_score'):
    if method == 'score':
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
    
    if method == 'num_cluster':
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

def definite_res(adata,res,save_path,plot_file='choose_res_spatial_plot.png',title=None):
    sc.tl.leiden(adata, resolution=res)
    print(adata.obs.leiden.unique())
    scale = (adata.obsm['spatial'][:,0].max()-adata.obsm['spatial'][:,0].min())/(adata.obsm['spatial'][:,1].max()-adata.obsm['spatial'][:,1].min())
    plt.rcParams["figure.figsize"] = (8*scale,8)
    sc.pl.embedding(adata, basis="spatial", color=["leiden"],title = title,size=20, show=False)
    plt.savefig(os.path.join(save_path,plot_file))
    plt.show()
    return adata

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    import argparse
    ArgParser = argparse.ArgumentParser(usage=usage,description='Process some integers.')
    ArgParser.add_argument("--version", action="version", version=PROG_VERSION)

    ArgParser.add_argument("-a", "--adata", action="store", dest="input", required=True, type=str, help="path of adata file")
    ArgParser.add_argument("-i", "--image", action="store", dest="image", required=True, type=str, help="path of H&E image file")
    ArgParser.add_argument("-o", "--output", action="store", dest="output", required=True, type=str, help="output folder")
    ArgParser.add_argument("-b", "--bin_size", action="store", dest="bin_size", required=True, type=int, help="bin size of adata")
    ArgParser.add_argument("-c", "--crop_size", action="store", dest="crop_size", required=True, type=int, help="crop size of image")
    ArgParser.add_argument("-n", "--num_cluster", action="store", dest="num_cluster", required=False, default=10, type=int, help="number of cluster")
    ArgParser.add_argument("-g", "--slide_gem",  action="store_true", dest="slide_gem", default=False, help="gem file for whole slide")
    # ArgParser.add_argument("-r", "--res_range", dest="res_range", required=False, default=np.around(np.arange(0.3,0.9,0.02), 3), type=float,help="resolution range for clustering")
    (para, args) = ArgParser.parse_known_args()
    print(para,'\n')
    #print(type(para.bin_size))
    
    num_gpus = torch.cuda.device_count()
    print("number of GPUs：", num_gpus)
    if num_gpus!=0:
        current_gpu = torch.cuda.current_device()
        print("The GPU index currently in use：", current_gpu)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f'CUDA available, using {torch.cuda.get_device_name(device)}.')
    
    name = str(para.bin_size)+'_crop_',str(para.crop_size)
    name = "".join(name)
    label_txt_file=str(name+'_label.txt')
    adata_file=str(name+'_adata.h5ad')
    
    rna_adata = sc.read_h5ad(para.input)
    
    if os.path.exists(os.path.join(para.output,label_txt_file)):
        print('The image tile has been completed. %s, read it directly' % os.path.join(para.output,label_txt_file)) 
    else:
        #img = cv2.imread(para.image)
        image_raw = Image.open(para.image)
        print(type(image_raw), image_raw.size)
        if para.slide_gem:
        #if int(rna_adata.obs.x.max()-rna_adata.obs.x.min()+para.bin_size)< img.size[0]:
            cbox = (int(rna_adata.obs.x.min()-50), int(rna_adata.obs.y.min()-50), int(rna_adata.obs.x.max()+50), int(rna_adata.obs.y.max()+50))
            image_added = image_raw.crop(cbox)  
        else:
            image_added = image_raw.copy()
        
        if 'spatial'  not in rna_adata.uns:
            print("add image to adata.uns['spatial']")
            rna_adata = add_img2adata(image_added,rna_adata,bin_size = para.bin_size,library_id = "cancer",spatial_key = "spatial")
        else:
            pass

        tile_path = os.path.join(para.output,name)
        mkdir(tile_path)
        crop_size = int(para.crop_size)  
        rna_adata = image_crop(image_added,rna_adata,save_path=tile_path,crop_size=crop_size,verbose=False)
        # rna_adata = image_crop(image_raw,rna_adata,save_path=tile_path,crop_size=crop_size,verbose=False)
        plot_spatial_with_img(rna_adata,para.output,file='rna_spatial_with_img.png')

        save_data(rna_adata,para.output,label_txt_file=label_txt_file,adata_file=adata_file)
        print('=======================finish generate tiles=============================')
        
    if os.path.exists(os.path.join(para.output,'img_feat.pkl')):
        print('The image feature extract has been completed. %s, read it directly' % os.path.join(para.output,'img_feat.pkl')) 
        feat_outputs = pd.read_pickle(os.path.join(para.output,'img_feat.pkl'))
    else:
        feature_extractor,model_com = load_model()
        # if torch.cuda.device_count() > 1:
        #     print("Turn on parallelism: use multiple GPUs for training")
        #     feature_extractor = torch.nn.DataParallel(feature_extractor)
        print('The device where the model is located:',next(feature_extractor.parameters()).device)

        label_txt_path = os.path.join(para.output,label_txt_file)
        feat_outputs = extract_img_feat(label_txt_path,extractor=feature_extractor)
        save_img_feat(feat_outputs,para.output,feat_file = 'img_feat.pkl')
        print('=======================finish generate image feature=============================')
    
    img_adata = generate_adata(feat_outputs.values,spatial=rna_adata.obsm['spatial'])
    print(f"Clustering H&E image feature at number of clusters: {para.num_cluster}")
    resolution, img_adata = find_res_binary(img_adata, resolution_min=0.1, resolution_max=1.2, num_clusters=para.num_cluster,key_added='cluster')
    print(f"Final Resolution: {resolution}")
    plot_spatial(img_adata,para.output,title ='H&E morphology featuer domain',group='cluster')
    # res_range = para.res_range
    # res = choose_res(img_adata,res_range,method = 'num_cluster',cluster_num = 10,criterion = 'CH_score')
    # img_adata = definite_res(img_adata,res,para.output,plot_file='image_spatial_plot.png',title='image_feat_leiden')
    # img_adata = find_clusters(img_adata,cluster_range=12,verbose=True)
    # img_adata = definite_res(img_adata, img_adata.uns['best_resolution'],para.output,plot_file='choose_res_spatial_plot.png',title=None)
    img_adata.write(os.path.join(para.output,'img_adata.h5ad'))
    
if __name__ == "__main__":
    main()
