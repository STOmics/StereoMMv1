import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import MinMaxScaler,StandardScaler,scale
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv, ResGatedGraphConv, GINConv, TransformerConv, RGATConv, TAGConv, VGAE, InnerProductDecoder
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import negative_sampling
from math import sqrt

EPS = 1e-15
MAX_LOGSTD = 10

class ActivateLayer(nn.Module):
    def __init__(self, activation_func):
        super(ActivateLayer, self).__init__()
        self.activation_func = activation_func

    def forward(self, x):
        x = self.activation_func(x)
        return x
    
class GraphConvLayer(nn.Module):
    def __init__(self, conv_func):#in_channels, out_channels
        super(GraphConvLayer, self).__init__()
        #self.conv = conv_func(in_channels, out_channels)
        self.conv = conv_func

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x
    
class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        with torch.no_grad():
            x1 = torch.matmul(Q,torch.transpose(K, -1, -2))
            # use mask
            if mask is not None:
                x1 = x1.masked_fill_(mask, -1e9)
            x1.div_(sqrt(Q.size(-1)))
            x2 = torch.softmax(x1, dim=-1); del x1;
            x3 = torch.matmul(x2,V); del x2;  
        return x3
    
class Multi_CrossAttention(nn.Module):
    def __init__(self,embed_dim,all_head_dim,num_heads):
        super().__init__()
        self.embed_dim    = embed_dim      
        self.all_head_dim  = all_head_dim     
        self.num_heads      = num_heads         
        self.h_size         = all_head_dim // num_heads

        assert all_head_dim % num_heads == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(embed_dim, all_head_dim , bias=False)
        self.linear_k = nn.Linear(embed_dim, all_head_dim , bias=False)
        self.linear_v = nn.Linear(embed_dim, all_head_dim , bias=False)
        self.linear_output = nn.Linear(all_head_dim, embed_dim)

        # normalization
        self.norm = sqrt(all_head_dim)

    def print(self):
        print(self.embed_dim,self.all_head_dim)
        print(self.linear_k,self.linear_q,self.linear_v)
    
    def forward(self,q,k,v,attention_mask=None):
        batch_size = q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(q).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(k).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(v).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        if attention_mask is not None:
            attention_mask = attention_mask.eq(0)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output, attention

class Attention_module(nn.Module):
    def __init__(self, embed_dim, num_heads, brief_att = True, all_head_dim=256, mode='cross', extract_att_weight=True):
        super(Attention_module, self).__init__()
        self.mode = mode
        self.extract_att_weight = extract_att_weight
            
        if self.mode in ['cross','rna','img']:
            print('Attention_module %s.\n number of heads in Attention_module: %s' % (mode,num_heads))
            if brief_att:
                self.attention = Multi_CrossAttention(embed_dim,all_head_dim,num_heads)
            else:
                self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        else:
            print('NO Attention modal!')

    def forward(self, img_tensor, rna_tensor):
        ## unsqueeze: batch_size, seq_length, embed_dim
        img_tensor = img_tensor.unsqueeze(0)
        rna_tensor = rna_tensor.unsqueeze(0)
        
        if self.mode == 'cross':
            attended_rna, attn_weights_rna = self.attention(rna_tensor, img_tensor, img_tensor)
            rna_emb = rna_tensor+attended_rna
            
            attended_img, attn_weights_img = self.attention(img_tensor, rna_tensor, rna_tensor)
            img_emb = img_tensor+attended_img
            
            attn_weight = (attn_weights_rna,attn_weights_img)
        
        elif self.mode == 'rna':
            attended_rna, attn_weights_rna = self.attention(rna_tensor, img_tensor, img_tensor) 
            rna_emb = rna_tensor+attended_rna  
            img_emb = img_tensor
            attn_weight = attn_weights_rna
        
        elif self.mode=='img':
            attended_img, attn_weights_img = self.attention(img_tensor, rna_tensor, rna_tensor)
            img_emb = img_tensor+attended_img   
            rna_emb = rna_tensor
            attn_weight = attn_weights_img

        # else:
        #     img_emb = img_tensor
        #     rna_emb = rna_tensor
        #     attn_weight = 'NO Attention modal!'
            
        rna_emb = torch.squeeze(rna_emb,0)
        img_emb = torch.squeeze(img_emb,0)
        return img_emb,rna_emb,attn_weight

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, gnn_type = 'GCN', activation_func = nn.ReLU()):
        super(Encoder, self).__init__()
        self.stack_net = self._make_encoder(input_size, hidden_sizes, gnn_type, activation_func)

    def _build_layer(self, conv_func, activation_func, drop_p = 0):
        activation_layer = ActivateLayer(activation_func)
        conv_layer = GraphConvLayer(conv_func)

        layers = []
        layers.append(conv_layer)
        layers.append(activation_layer)
        if drop_p > 0:
            layers.append(nn.Dropout(drop_p))

        return nn.ModuleList(layers)
    
    def _make_layer(self, block, in_size, growth_rate, num_layers, droprate):
        layers = []
        for i in range(num_layers):
            layers.append(block(in_size, in_size-i*growth_rate, droprate))
        return nn.Sequential(*layers)
    
    def _make_encoder(self, input_size, hidden_sizes, gnn_type, activation_func):
        encoder = nn.ModuleList()
        # Define activation function
        #activation_func = activation_func
        
        # Define the convolution function
        for i in range(len(hidden_sizes)):
            if gnn_type == 'GCN': 
                if i == 0:
                    # First layer, input size is the initial input size
                    conv_func = GCNConv(input_size, hidden_sizes[i])#.double()
                else:
                    # Subsequent layers, input size is the output size of the previous layer
                    conv_func = GCNConv(hidden_sizes[i-1], hidden_sizes[i])#.double()
                    
            elif gnn_type == 'GAT': 
                if i == 0:
                    conv_func = GATConv(input_size, hidden_sizes[i])#.double()
                else:
                    conv_func = GATConv(hidden_sizes[i-1], hidden_sizes[i])#.double()
                    
            elif gnn_type == 'SAGE': 
                if i == 0:
                    conv_func = SAGEConv(input_size, hidden_sizes[i])#.double()
                else:
                    conv_func = SAGEConv(hidden_sizes[i-1], hidden_sizes[i])#.double()
                    
            elif gnn_type == 'ResGate': 
                if i == 0:
                    conv_func = ResGatedGraphConv(input_size, hidden_sizes[i])#.double()
                else:
                    conv_func = ResGatedGraphConv(hidden_sizes[i-1], hidden_sizes[i])#.double()
                    
            elif gnn_type == 'GIN': 
                if i == 0:
                    conv_func = GINConv(input_size, hidden_sizes[i])#.double()
                else:
                    conv_func = GINConv(hidden_sizes[i-1], hidden_sizes[i])#.double()
                    
            elif gnn_type == 'Transformer': 
                if i == 0:
                    conv_func = TransformerConv(input_size, hidden_sizes[i])#.double()
                else:
                    conv_func = TransformerConv(hidden_sizes[i-1], hidden_sizes[i])#.double()
                    
            elif gnn_type == 'RGAT': 
                if i == 0:
                    conv_func = RGATConv(input_size, hidden_sizes[i])#.double()
                else:
                    conv_func = RGATConv(hidden_sizes[i-1], hidden_sizes[i])#.double()
                    
            elif gnn_type == 'TAG': 
                if i == 0:
                    conv_func = TAGConv(input_size, hidden_sizes[i])#.double()
                else:
                    conv_func = TAGConv(hidden_sizes[i-1], hidden_sizes[i])#.double()

            encoder.add_module(f'encoder_L{i}', 
                    self._build_layer(conv_func,activation_func,drop_p = 0))

        return encoder 
    
    def forward(self, x, edge_index):
        for layer in self.stack_net: 
            x = layer[0](x, edge_index)
            x = layer[1](x)
        return x
    
    # def _make_layer(self, input_size, hidden_size, conv_type, num_layers):
    #     layers = []
    #     layers.append(GraphConvLayer(input_size, hidden_size, conv_type))
    #     for _ in range(num_layers - 1):
    #         layers.append(GraphConvLayer(hidden_size, hidden_size, conv_type))
    #     return nn.ModuleList(layers)
    
class Decoder(Encoder):
    def __init__(self, output_size, hidden_sizes, gnn_type = 'GCN', activation_func = nn.ReLU()):
        super().__init__(output_size, hidden_sizes)#在用super继承的时候参数只能加不能减
        self.stack_net = self._make_decoder(output_size, hidden_sizes, gnn_type, activation_func)
        
    def _make_decoder(self, output_size, hidden_sizes, gnn_type, activation_func): 
        decoder = nn.ModuleList()
        
        # Define activation function
        # activation_func = activation_func
        
        # Define the convolution function
        for i in range(len(hidden_sizes)):
            if gnn_type == 'GCN': 
                if i < len(hidden_sizes)-1:
                    # First layer, input size is the initial input size
                    conv_func = GCNConv(hidden_sizes[i], hidden_sizes[i+1])#.double()
                else:
                    # Subsequent layers, input size is the output size of the previous layer
                    conv_func = GCNConv(hidden_sizes[i], output_size)#.double()
                    
            elif gnn_type == 'GAT': 
                if i < len(hidden_sizes)-1:
                    conv_func = GATConv(hidden_sizes[i], hidden_sizes[i+1])#.double()
                else:
                    conv_func = GATConv(hidden_sizes[i], output_size)#.double()
                    
            elif gnn_type == 'SAGE': 
                if i < len(hidden_sizes)-1:
                    conv_func = SAGEConv(hidden_sizes[i], hidden_sizes[i+1])#.double()
                else:
                    conv_func = SAGEConv(hidden_sizes[i], output_size)#.double()
            
            elif gnn_type == 'ResGated': 
                if i < len(hidden_sizes)-1:
                    conv_func = ResGatedGraphConv(hidden_sizes[i], hidden_sizes[i+1])#.double()
                else:
                    conv_func = ResGatedGraphConv(hidden_sizes[i], output_size)#.double()
                    
            elif gnn_type == 'GIN': 
                if i < len(hidden_sizes)-1:
                    conv_func = GINConv(hidden_sizes[i], hidden_sizes[i+1])#.double()
                else:
                    conv_func = GINConv(hidden_sizes[i], output_size)#.double()
                    
            elif gnn_type == 'Transformer': 
                if i < len(hidden_sizes)-1:
                    conv_func = TransformerConv(hidden_sizes[i], hidden_sizes[i+1])#.double()
                else:
                    conv_func = TransformerConv(hidden_sizes[i], output_size)#.double()
                    
            elif gnn_type == 'RGAT': 
                if i < len(hidden_sizes)-1:
                    conv_func = RGATConv(hidden_sizes[i], hidden_sizes[i+1])#.double()
                else:
                    conv_func = RGATConv(hidden_sizes[i], output_size)#.double()
                    
            elif gnn_type == 'TAG': 
                if i < len(hidden_sizes)-1:
                    conv_func = TAGConv(hidden_sizes[i], hidden_sizes[i+1])#.double()
                else:
                    conv_func = TAGConv(hidden_sizes[i], output_size)#.double()

            decoder.add_module(f'decoder_L{i}', 
                    self._build_layer(conv_func,activation_func,drop_p = 0))

        return decoder 
    
class CustomVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, gnn_type = 'GAT', decoder = None, activation_func = nn.ReLU()):
        super(CustomVAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, gnn_type , activation_func)
        # self.fc_mean = GCNConv(hidden_size[-1], latent_size)
        # self.fc_logvar = GCNConv(hidden_size[-1], latent_size)
        
        if gnn_type == 'GCN':
            self.fc_mean = GCNConv(hidden_size[-1], latent_size)
            self.fc_logvar = GCNConv(hidden_size[-1], latent_size)
            
        elif gnn_type == 'GAT':
            self.fc_mean = GATConv(hidden_size[-1], latent_size)
            self.fc_logvar = GATConv(hidden_size[-1], latent_size)
            
        elif gnn_type == 'SAGE':
            self.fc_mean = SAGEConv(hidden_size[-1], latent_size)
            self.fc_logvar = SAGEConv(hidden_size[-1], latent_size)
            
        self.decoder = InnerProductDecoder() if decoder is None else decoder
            
    def encode(self, x, edge_index):
        encoded = self.encoder(x,edge_index)
        mean = self.fc_mean(encoded, edge_index)
        logvar = self.fc_logvar(encoded, edge_index)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, *args, **kwargs):
        #decoded = self.decoder(z, edge_index)
        return self.decoder(*args, **kwargs)

    def forward(self, x, edge_index):
        mean, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mean, logvar)
        try:
            x_hat = self.decode(z.long(), edge_index.long())
        except:
            x_hat = self.decode(z, edge_index)
        return z, mean, logvar, x_hat
    
class FinalModal(nn.Module):
    def __init__(self, single_model_size, hidden_size, latent_size, num_heads=1, brief_att = True, attn_mode='cross', gnn_type = 'GCN', img_weight = 1, rna_weight = 1, decoder = None):
        super(FinalModal, self).__init__()
        self.img_weight = img_weight
        self.rna_weight = rna_weight
        self.attn_mode = attn_mode
        #self.atten = Attention_module(single_model_size*num_heads, num_heads, brief_att = brief_att, mode=attn_mode, all_head_dim=256, extract_att_weight=True)
        self.atten = Attention_module(single_model_size, num_heads, brief_att = brief_att, mode=attn_mode, all_head_dim=256, extract_att_weight=True)
        self.vae = CustomVAE(2*single_model_size, hidden_size, latent_size, gnn_type = gnn_type, decoder=decoder)
         
    def forward(self, img_tensor, rna_tensor, edge_index):
        if self.attn_mode in ['cross','rna','img']:
            img_emb,rna_emb,attn_weight = self.atten(img_tensor, rna_tensor)
            concat_emb = torch.cat((self.img_weight*img_emb,self.rna_weight*rna_emb),1)
        else:
            img_emb = img_tensor
            rna_emb = rna_tensor
            concat_emb = torch.cat((self.img_weight*img_tensor, self.rna_weight*rna_tensor),1)
            attn_weight = 'No attention model for modalitys fusion'
        
        z, mean, logvar, x_hat = self.vae(concat_emb,edge_index)
        return z, mean, logvar, x_hat, img_emb, rna_emb, attn_weight
       
    def innerproduct_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            # self.vae.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
            self.vae.decoder(z, pos_edge_index) + EPS).mean()

        # remove self-loops 
        # pos_edge_index, _ = remove_self_loops(pos_edge_index)
        # pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              # self.vae.decoder(z, neg_edge_index, sigmoid=True) +
                              self.vae.decoder(z, neg_edge_index) +
                              EPS).mean()
        return pos_loss + neg_loss
    
    def kl_loss(self, mu = None,logstd = None) -> Tensor:
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    
    def recon_loss(self, input, output):
        reconstruction_loss = F.mse_loss(output, input, reduction='mean')
        return reconstruction_loss
    
    def print_networks(self, verbose=True):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for i,layer in enumerate(self.named_children()): 
            name = layer[0]
            if isinstance(name, str):
                net = getattr(self,name)
                num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def test_modal():
    num_nodes = 10
    num_features = 50

    img_tensor = torch.randn(num_nodes, num_features)
    rna_tensor = torch.randn(num_nodes, num_features)

    edge_index = torch.randint(num_nodes, (2, num_nodes * 2))

    edge_index = to_undirected(edge_index)

    decoder = Decoder(output_size=50,hidden_sizes=[10,16,32])
    fimodal = FinalModal(50,[32,16],10,decoder=decoder)
    print(fimodal)
    z, mean, logvar, x_hat, img_emb, rna_emb, attn_weight = fimodal(img_tensor, rna_tensor, edge_index)
    print(z.shape, attn_weight)
    
    inner_loss = fimodal.innerproduct_loss(z,edge_index)
    #recon_loss = fimodal.recon_loss(x_hat,img_emb)
    kl_loss = fimodal.kl_loss(mean, logvar)
    print(inner_loss,kl_loss)

if __name__ == '__main__':
    test_modal()
