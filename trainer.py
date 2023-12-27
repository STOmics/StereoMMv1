import torch
from tqdm import tqdm
import random
import numpy as np
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter
from utils import *
import gc

class train():
    def __init__(self, img_tensor, rna_tensor, edge_index, model, custom_decoder = True,n_epochs=100, opt="adam", lr=0.0001, weight_decay=0.0001, save_att=False, verbose=True):#, attn_out = None
        self.img_tensor = img_tensor
        self.rna_tensor = rna_tensor
        self.edge_index = edge_index
        self.model = model
        self.custom_decoder = custom_decoder
        self.n_epochs = n_epochs
        self.save_att = save_att
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if opt=="sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif opt=="adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr, weight_decay=weight_decay)  

    def train_ae(self, gradient_clipping=5.):
        print('tarining AE')
        loss_list = []
        pbar = tqdm(range(1, self.n_epochs+1),desc='Training model...')
        current_memory = torch.cuda.memory_allocated() / 1024**2 
        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad()
            z, mean, logvar, x_hat, img_emb,rna_emb, attn_weight = self.model(self.img_tensor, self.rna_tensor, self.edge_index)
            if self.custom_decoder == True:
                concat_tensor = torch.cat((self.img_tensor,self.rna_tensor),1)
                recon_loss = self.model.recon_loss(x_hat,concat_tensor)
                inner_loss = self.model.innerproduct_loss(z,self.edge_index)
                kl_loss = self.model.kl_loss(mean, logvar)
                loss = recon_loss + (1 / self.img_tensor.shape[0]) * kl_loss #+ inner_loss
            else:
                inner_loss = self.model.innerproduct_loss(z,self.edge_index)
                kl_loss = self.model.kl_loss(mean, logvar)
                loss = inner_loss + (1 / self.img_tensor.shape[0]) * kl_loss
            loss_list.append(loss.item())
            loss.backward()
            #print(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)
            self.optimizer.step()

            gc.collect()
            torch.cuda.empty_cache()

            if epoch % 2 == 0:
                pbar.set_postfix_str("loss: {:.4f}".format(loss.item()))
            if self.save_att == True:
                # print('attention weiget saved at: %s' % (attn_out))
                # if not os.path.exists(attn_out):  
                #     os.makedirs(attn_out)
                self.save_attn_weight(epoch, attn_weight, inter=10, out_dir='./')
                #tqdm.set_description("loss: {:.4f}".format(loss.item()))
                #tqdm.set_postfix(loss=loss.item())
                #tqdm.set_postfix_str("loss: {:.4f}".format(loss.item()))

#         self.model.eval()
#         with torch.no_grad():
#             z, mean, logvar, x_hat, attn_weight = self.model(self.img_tensor, self.rna_tensor, self.edge_index)
#         vae_emb = z.to('cpu').detach().numpy()

#         return vae_emb,loss_list
        return loss_list, img_emb, rna_emb, attn_weight
    
    def train_dec(self, init = "kmeans", init_spa = True, n_cluster = 10, n_neighbors = 20, max_epochs = 100, update_interval = 3, tol = 1e-5, alpha = 0.9):
        loss_list, img_emb, rna_emb, attn_weight = self.train_ae()
        x = get_embedding(self.img_tensor, self.rna_tensor, self.edge_index, self.model)
        
        self.model.dec = Parameter(torch.Tensor(n_cluster, x.shape[1])).to(self.device)
        print('training use DEC')
        # self.model.dec = Parameter(torch.Tensor(n_cluster, x.shape[0])).to(self.device)
        # torch.nn.init.xavier_normal_(self.dec.data)
            
        if torch.is_tensor(x) :
            x_tensor = x.to(self.device)
        elif  isinstance(x, pd.DataFrame):
            x_array = x.values.astype(np.float32)
            x_tensor = torch.tensor(x_array).to(self.device)
        else:
            x_tensor = torch.tensor(x).to(self.device)

        if init=="kmeans":
            print(str("Initializing cluster centers with kmeans, n_clusters known: "+str(n_cluster)))
            # self.n_clusters=n_clusters
            kmeans = KMeans(n_cluster, n_init=20)

            if init_spa:
                #------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(x)
            else:
                #------Kmeans only use exp info, no spatial
                concat_tensor = torch.cat((self.img_tensor,self.rna_tensor),1)
                y_pred = kmeans.fit_predict(concat_tensor.numpy())  
                
        elif init=="leiden":
            print("Initializing cluster centers with leiden, resolution = ", res)
            if init_spa:
                adata=sc.AnnData(x)
            else:
                concat_tensor = torch.cat((self.img_tensor,self.rna_tensor),1)
                adata=sc.AnnData(concat_tensor.numpy())
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            
            res = choose_res(adata, cluster_num = n_cluster ,res_range = np.around(np.arange(0.3,0.9,0.04), 3), determine_clus_num = True)
            adata = definite_res(adata,res,'./',plot_file='init_spatial_plot.png',title='init_leiden')
            try :
                select_res(adata,res,method='leiden',plot=True,title='init_plot')
            except :
                print('try select_res error')
            y_pred=adata.obs['leiden'].astype(int).to_numpy()
            
        y_pred_last = y_pred

        Group=pd.Series(y_pred,index=np.arange(0,x.shape[0]),name="Group")
        Mergefeature=pd.concat([pd.DataFrame(x),Group],axis=1) #detach().numpy()
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        #print(cluster_centers.shape,cluster_centers)
        
        self.model.dec.data.copy_(torch.Tensor(cluster_centers)).to(self.device)
        self.model.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                zq, _, _, _, _, _, _ = self.model.forward(self.img_tensor, self.rna_tensor, self.edge_index, atten_model = True)
                
                q = 1.0 / ((1.0 + torch.sum((zq.unsqueeze(1) - self.model.dec)**2, dim=2) / alpha) + 1e-8)
                q = q**(alpha+1.0)/2.0
                q = q / torch.sum(q, dim=1, keepdim=True)
                
                p = self.target_distribution(q).data
            # else :
            #     p = p
            self.optimizer.zero_grad()
            z, mean, logvar, x_hat, img_emb,rna_emb, attn_weight = self.model.forward(self.img_tensor, self.rna_tensor, self.edge_index, atten_model = True)
            if self.custom_decoder == True:
                concat_tensor = torch.cat((self.img_tensor,self.rna_tensor),1)
                recon_loss = self.model.recon_loss(x_hat,concat_tensor)
                kl_loss = self.model.kl_loss(mean, logvar)
                loss = 10*recon_loss + (1 / self.img_tensor.shape[0]) * kl_loss
            else:
                inner_loss = self.model.innerproduct_loss(z,self.edge_index)
                kl_loss = self.model.kl_loss(mean, logvar)
                loss = inner_loss + (1 / self.img_tensor.shape[0]) * kl_loss
            # total_loss = loss + dec_kl_loss
            loss.backward()
            self.optimizer.step()
            if epoch%10==0:
                print("Epoch ", epoch, " loss:",loss) 

            #Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / x.shape[0]
            y_pred_last = y_pred
            if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break
        
        return loss_list, img_emb, rna_emb, attn_weight
                
    def target_distribution(self, q):
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def save_attn_weight(self, epoch, attn_weight, inter=10, out_dir='./'):
        import pickle
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if epoch % inter == 0:
            save_filename = 'epoch_%s_att_weight.pkl' % (epoch)
            with open(os.path.join(out_dir, save_filename), 'wb') as file:
                pickle.dump(attn_weight, file)

    def predict(self, ):
        self.model.eval()
        with torch.no_grad():
            z, mean, logvar, x_hat, img_emb,rna_emb, attn_weight = self.model(self.img_tensor, self.rna_tensor, self.edge_index)
        emb = z.to('cpu').detach().numpy()

        return emb


def get_embedding(img_tensor, rna_tensor, edge_index, model):
    model.eval()
    with torch.no_grad():
        z, mean, logvar, x_hat, img_emb,rna_emb, attn_weight = model(img_tensor, rna_tensor, edge_index)
    vae_emb = z.to('cpu').detach().numpy()

    return vae_emb

