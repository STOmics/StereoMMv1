# StereoMM

[![My Skills](https://skillicons.dev/icons?i=python,pytorch,r,bash,linux)](https://skillicons.dev)

![Static Badge](https://img.shields.io/badge/MultiModal-StereoMM-red)

![](https://komarev.com/ghpvc/?username=hah2468)

[Overview](#Overview)

[Usage](#Usage)

Citation
---------

If you use `stereoMM`_ in your work, please cite the publication as follows:

    **StereoMM: A Graph Fusion Model for Integrating Spatial Transcriptomic Data and Pathological Images**

    Bingying Luo, Fei Teng, Guo Tang, Weixuan Chen, Chi Qu, Xuanzhu Liu, Xin Liu, Xing Liu, Huaqiang Huang, Yu Feng, Xue Zhang, Min Jian, Mei Li, Feng Xi, Guibo Li, Sha Liao, Ao Chen, Xun Xu, Jiajun Zhang

    bioRxiv 2024.05.07.592486; doi: https://doi.org/10.1101/2024.05.04.592486


## Overview

**StereoMM** is a graph fusion model that can **integrates gene expression, histological images, and spatial location**. And the information interaction within modalities is strengthened by introducing an attention mechanism. 

StereoMM firstly performs information interaction on transcriptomic and imaging features through the attention module. The interactive features are put into the graph autoencoder together with the graph of spatial position, so that multimodal features are fused in a self-supervised manner. Finally, a low-dimensional, noise-reducing, higher-quality feature representation is obtained by extracting the features of the latent space. StereoMM contributes to accurately identifying domains, uncovering the more significant molecule characteristics among different domains and paving ways for downstream analysis.



## Environment preparation and package download

StereoMM relies on pytorch and pyg. In order to create a working environment, we recommend the following installation code：

```
conda create -n StereoMM python==3.9
conda activate StereoMM
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
# check gpu availiable
python print_cuda.py

## essential packages
conda install -c conda-forge scanpy python-igraph leidenalg
```



## Usage

**Project directory structure**

Users should upload their spatial transcriptomics data (in the form of an h5ad file) and the corresponding registered image files. For this example, the files should be located at `./example/adata.h5ad` and `./example/image.tif`, respectively. If the files are uploaded to different paths, the relevant parameters must be adjusted accordingly.

```
StereoMMv1/
├── models.py
├── trainer.py
├── process_img.py
├── print_cuda.py
├── torch_pths
│   └── resnet50-19c8e357.pth
├── example
│   ├── image.tif
│   └── adata.h5ad
├── Tutorials
│   └── tutorial.ipynb
└── utils.py
```

Clone the project to your computer:

```
git clone https://github.com/STOmics/StereoMMv1
```



### Method 1: Run from command line

#### For real data： 

Step1：Process imaging data, including tiling and feature extraction through CNN.

```
python StereoMMv1/process_img.py -a StereoMMv1/example/adata.h5ad -i StereoMMv1/example/image.tif -o StereoMMv1/example/image_out -b 100 -c 150
```



Step2：Construct a StereoMM model and perform feature fusion.

```
python StereoMMv1 --rna_data StereoMMv1/example/adata.h5ad --image_data StereoMMv1/example/image_out/img_feat.pkl -o StereoMMv1/example/real_test_nn --epochs 100 --lr 0.0001 --radiu_cutoff 100 --hidden_dims 2048 256 --latent_dim 100 --gnn_type GCN --opt adam --dim_reduction_method high_var --scale zscore --num_head 1 --decoder --brief_att
```



#### For toy data:

We also provide simulated data for model testing, using the -t parameter:

```
python StereoMMv1 -t -o StereoMMv1/example/toy_test --epochs 30 --lr 0.001 --radiu_cutoff 100 --hidden_dims 1028 256 --latent_dim 100 --gnn_type GCN --opt adam --dim_reduction_method high_var --scale zscore --num_heads 4 --decoder --brief_att
```



### Method 2: Run step by step in jupyter-notebook

More details at: [StereoMM tutorial](./Tutorials/tutorial.ipynb)



The output folder contains **embedding.pkl/csv**, that is the new feature representation file.



### Parameter Description

(1) --version: Boolean switch, print the current software version; 

(2) --sessioninfo: Boolean switch, print the version of each reference library in the current environment; 

(3) --verbose: Boolean switch, print the monitoring status of CUDA memory usage during the training process;

 (4) --toy: Boolean switch, the input uses simulation data to run the program; 

(5) --rna_data: string, reading path of transcriptome data;

 (6) --image_data: string, the reading path of H&E imaging data; 

(7) --output: string, output result storage path; 

(8) --epochs: integer, the number of iterations of model training, the default is 100; 

(9) --lr: floating point number, learning rate for model training, default is 0.0001; 

(10) --opt: string, optimizer for model training, default is 'adam'; 

(11) --dim_reduction_method: string, feature extraction method for transcriptome data, default is 'high_var'; 

(12) --scale: string, standardization method of single-modal features, the default is 'zscore'; 

(13) --radiu_cutoff: integer or empty data, the radius threshold when constructing the position coordinate KNN, the default is empty; 

(14) --purning: Boolean switch, input to prune the position coordinate KNN graph; 

(15) --num_heads: integer, the number of attention heads of the attention module, the default is 1;

 (16) --hidden_dims: integer, the number of nodes in the hidden layer of the VGAE module, the default is [2048,128]; 

(17) --latent_dim: integer, the dimension of the mean and variance of the latent space of the VGAE module, the default is 100;

 (18) --brief_att: Boolean switch, the input uses the attention version with a reduced number of parameters; 

(19) --att_mode: string, the running mode of the attention module, optional 'cross', 'img', 'rna', 'self', the default is 'cross';

 (20) --decoder: Boolean switch, the input is to reconstruct the graph node characteristics in the VGAE module, otherwise the graph structure is reconstructed; 

(21) --gnn_type: string, the type of graph neural network in the VGAE module, optional 'GAT', 'GCN', 'GraphSAGE', the default is 'GCN'; 

(22) --dec: Boolean switch, the input is to add the DEC module during the training process;

(23) --rna_weight: integer, the weight of rna feature during feature concat, the default is 1. 

(24) --n_cluster: integer, the final number of categories in the clustering process, the default is 10;
