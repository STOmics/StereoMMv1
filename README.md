# StereoMM
StereoMM is a graph fusion model that can integrate gene expression, histological images, and spatial location. And the information interaction within modalities is strengthened by introducing an attention mechanism. 

This framework utilizes a self-supervised generative neural network model. It generates a feature representation that combines multiple modalities, which can be utilized for various downstream tasks to enhance task accuracy. The learning process is guided by a combination of minimizing the self-supervised reconstruction loss and a regularization loss that forces the latent space representation. The reconstruction loss in an autoencoder encourages the generated outputs ((X¬¬¬¬) ̂) to closely resemble the original input matrix (X). In other words, it ensures that the latent features learned by the encoder preserve the maximum information from the original input, then the decoder can reconstruct the original input through these latent features. The regularization loss, also known as the Kullback-Leibler (KL) divergence, encourages the model to learn a compact and smooth latent space representation.

Specifically, the training process is divided into the following steps: (1) For the transcriptome and H&E image, a unimodal feature extractor is employed to extract s-dimensional unimodal features, generate two feature matrices (X_t∈R^(n×s) for transcriptome, and X_m∈R^(n×s) for morphology, where n represents the number of bins or spots). (2) These features are then fed into the attention module, where the information between modalities is integrated using the attention mechanism (Fig. 1b). This integration results in an s-dimensional output that enhances the interaction between modalities (X_ta∈R^(n×s) for transcriptome, and X_ma∈R^(n×s) for morphology). (3) The feature matrices from both modalities are concatenated (X = X_ta⊕X_ma) and used as input for the node features of the graph autoencoder. (4) To incorporate spatial location information, a spatial neighbor graph (SNG) is generated based on the physical distance. This SNG serves as the input for the adjacency matrix in the graph autoencoder.

The generative model for graph data utilizes a graph neural network to learn a distribution of node vector representations (Fig. 1c). These representations are then sampled from the distribution, and the graph is reconstructed using a decoder. By extracting the latent representation from the Variational Graph Autoencoder (VGAE), a high-quality, low-dimensional representation (Z∈R^(n×d), Where d represents the feature dimension after dimensionality reduction) of the graph data is obtained. This feature representation Z can be effectively utilized for various downstream analyses, including clustering, trajectory analysis, and more.


First, the process of image should be done.

python process_img.py -a example/adata.h5ad -i example/image.tif -o example/image_out -b 100 -c 150 > example/process_img.log

Second, the feature fusion should be run

python __main__.py --rna_data example/adata.h5ad --image_data example/image_out/img_feat.pkl -o example/real_test_nn --epochs 100 --lr 0.0001 --radiu_cutoff 100 --hidden_dims 2048 256 --latent_dim 100 --docoder_type GAT --opt adam --dim_reduction_method high_var --scale zscore --num_head 1 --decoder --brief_att>example/real_test.log 

Last, the data for feature fusion

python __main__.py -t -o example/toy_test --epochs 30 --lr 0.001 --radiu_cutoff 100 --hidden_dims 1028 256 --latent_dim 100 --decoder --gnn_type GCN --opt adam --dim_reduction_method pca --scale zscore --num_heads 4 --brief_att > example/toy_test.log 



