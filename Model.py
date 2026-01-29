from torch import nn
import torch.nn.functional as F
import torch
from Params import args
from copy import deepcopy
import numpy as np
import math
import scipy.sparse as sp
from Utils.Utils import contrastLoss, calcRegLoss, pairPredict
import time
import torch_sparse
import pickle
from scipy.sparse import coo_matrix
import torch as t
from Louvain import Louvain
from Diffusion_process import DiffusionProcess
init = nn.init.xavier_uniform_


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.uEmbeds = torch.nn.Embedding(
            num_embeddings=args.user, embedding_dim=args.latdim)
        self.iEmbeds = torch.nn.Embedding(
            num_embeddings=args.item, embedding_dim=args.latdim)
        nn.init.normal_(self.uEmbeds.weight, std=0.1)
        nn.init.normal_(self.iEmbeds.weight, std=0.1)

        # output_dims = [args.dims] + [args.latdim]
        # input_dims = output_dims[::-1]
        # self.SDNet = SDNet(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(self.device)
        # self.DiffProcess=DiffusionProcess(args.noise_schedule,args.noise_scale, args.noise_min, args.noise_max, args.steps,self.device).to(self.device)
        self.louvain=Louvain()
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
        self.gcn = GCNLayer()
        self.gcn_social = GCNLayer_social()
        self.MLP0 = MLP_0(args.latdim * 2, 2)
        self.MLP1 = MLP_1(args.latdim * 2, 2)
        self.MLP2 = MLP_2(args.latdim * 2, 2)
        self.MLP3 = MLP_3(args.latdim * 2, 2)
        self.MLP4 = MLP_4(args.latdim * 2, 2)
        self.MLP01 = MLP_01(args.latdim * 2, 2)
        self.MLP11 = MLP_11(args.latdim * 2, 2)
        self.MLP21 = MLP_21(args.latdim * 2, 2)
        self.MLP31 = MLP_31(args.latdim * 2, 2)
        self.MLP41 = MLP_41(args.latdim * 2, 2)
        self.MLPL1 = MLP_L1(args.latdim * 5, 5)
        self.MLPL2 = MLP_L2(args.latdim * 5, 5)
        self.M1 = MLP_4(args.latdim * 3, 3)
        self.M2 = MLP_4(args.latdim * 2, 2)
        self.M3 = MLP_4(args.latdim * 2, 2)
   

    def forward_gcn(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds.weight, self.iEmbeds.weight], dim=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)
        return mainEmbeds[:args.user], mainEmbeds[args.user:]

    def forward_graphcl(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds.weight, self.iEmbeds.weight], dim=0)
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)
        return mainEmbeds

    #新的融合
    def forward_fusion(self, interaction_adj, social_adj):
        socail_adj = social_adj.to(self.device)
        iniEmbeds = torch.concat((self.uEmbeds.weight, self.iEmbeds.weight), dim=0)#全部用户嵌入和商品嵌入
        iniEmbeds_s = iniEmbeds[:args.user]#用户嵌入
        embedsLst_interaction = [iniEmbeds]#全部嵌入矩阵
        abation = [iniEmbeds]#全部嵌入矩阵
        embedsLst_social = [iniEmbeds_s]#用户嵌入矩阵
    
        
        for i in range(args.uu_layers):#UU GNN
            embeds_social = self.gcn(socail_adj, embedsLst_social[-1])
            embedsLst_social.append(embeds_social)
        for i in range(args.ui_layers):#UI GNN
            embeds_inter = self.gcn(interaction_adj, abation[-1])
            abation.append(embeds_inter)
        
        ui =sum(abation)
        uu =sum(embedsLst_social)#1.是否加原始嵌入，2.聚合几层
        inter_u = ui[:args.user]#UI图用户嵌入

        self.inter_uu=inter_u#交互图用户嵌入出去
        self.social_uu=uu#社交图用户嵌入
        # m0 = self.MLP0(uu, inter_u)#两用户嵌入融合
        # m01 = torch.reshape(m0[:, 0], [args.user, 1])
        # m02 = torch.reshape(m0[:, 1], [args.user, 1])
        # embeds = torch.cat((uu * m01 + inter_u * m02 , ui[args.user:]), dim=0)#融合用户嵌入再拼接商品嵌入
       
        # return embeds
        return ui,uu

    def forward_fusion2(self, interaction_adj, socail_adj, a, b):  
        socail_adj = socail_adj.to(self.device)
        iniEmbeds = torch.concat((self.uEmbeds.weight, self.iEmbeds.weight), dim=0)#全部用户嵌入和商品嵌入
        iniEmbeds_s = iniEmbeds[:args.user]#用户嵌入
        embedsLst_interaction = [iniEmbeds]#全部嵌入矩阵
        abation = [iniEmbeds]#全部嵌入矩阵
        embedsLst_social = [iniEmbeds_s]#用户嵌入矩阵
        embedsLst = []
        if a == 1:
            for i in range(args.gnn_layer):#UU GNN
                embeds_social = self.gcn(socail_adj, embedsLst_social[-1])
                embedsLst_social.append(embeds_social)
            for i in range(args.gnn_layer):#UI GNN
                embeds_inter = self.gcn(interaction_adj, abation[-1])
                abation.append(embeds_inter)

            uu_0 = embedsLst_social[0]#原始UU图用户嵌入
            ui_0 = embedsLst_interaction[0]#原始UI图用户和商品嵌入
            inter_u = ui_0[:args.user]#UI图用户嵌入
            m0 = self.MLP0(uu_0, inter_u)#两用户嵌入融合
            m01 = torch.reshape(m0[:, 0], [args.user, 1])
            m02 = torch.reshape(m0[:, 1], [args.user, 1])
            embeds = torch.cat((uu_0 * m01 + inter_u * m02, ui_0[args.user:]), dim=0)#融合用户嵌入再拼接商品嵌入
            embeds_inter = self.gcn(interaction_adj, embeds)#原始用户嵌入融合后gcn
            embedsLst_interaction[0] = embeds#替换为用户融合后用户及商品嵌入
            embedsLst_interaction.append(embeds_inter)#添加融合后的用户嵌入和商品嵌入GNN后的UI嵌入
            embedsLst.append(embeds)#结果：原始UI图用户嵌入添加融合后的嵌入

            uu_1 = embedsLst_social[1]
            ui_1 = embedsLst_interaction[1]
            inter_u = ui_1[:args.user]
            m1 = self.MLP1(uu_1, inter_u)
            m11 = torch.reshape(m1[:, 0], [args.user, 1])
            m12 = torch.reshape(m1[:, 1], [args.user, 1])
            embeds = torch.cat((uu_1 * m11 + inter_u * m12, ui_1[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)#结果： GNN后 用户嵌入融合后的嵌入及商品嵌入

            uu_2 = embedsLst_social[2]
            ui_2 = embedsLst_interaction[2]
            inter_u = ui_2[:args.user]
            m2 = self.MLP2(uu_2, inter_u)
            m21 = torch.reshape(m2[:, 0], [args.user, 1])
            m22 = torch.reshape(m2[:, 1], [args.user, 1])
            embeds = torch.cat((uu_2 * m21 + inter_u * m22, ui_2[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)
            
            uu_3 = embedsLst_social[3]
            ui_3 = embedsLst_interaction[3]
            inter_u = ui_3[:args.user]
            m3 = self.MLP3(uu_3, inter_u)
            m31 = torch.reshape(m3[:, 0], [args.user, 1])
            m32 = torch.reshape(m3[:, 1], [args.user, 1])
            embeds = torch.cat((uu_3 * m31 + inter_u * m32, ui_3[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)

    
        else:
            for i in range(args.gnn_layer):
                embeds_social = self.gcn(socail_adj, embedsLst_social[-1])
                embedsLst_social.append(embeds_social)
            for i in range(args.gnn_layer):
                embeds_inter = self.gcn(interaction_adj, abation[-1])
                abation.append(embeds_inter)

            uu_0 = embedsLst_social[0]
            ui_0 = embedsLst_interaction[0]
            inter_u = ui_0[:args.user]
            m0 = self.MLP01(uu_0, inter_u)
            m01 = torch.reshape(m0[:, 0], [args.user, 1])
            m02 = torch.reshape(m0[:, 1], [args.user, 1])
            embeds = torch.cat((uu_0 * m01 + inter_u * m02, ui_0[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction[0] = embeds
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)

            uu_1 = embedsLst_social[1]
            ui_1 = embedsLst_interaction[1]
            inter_u = ui_1[:args.user]
            m1 = self.MLP11(uu_1, inter_u)
            m11 = torch.reshape(m1[:, 0], [args.user, 1])
            m12 = torch.reshape(m1[:, 1], [args.user, 1])
            embeds = torch.cat((uu_1 * m11 + inter_u * m12, ui_1[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)

            uu_2 = embedsLst_social[2]
            ui_2 = embedsLst_interaction[2]
            inter_u = ui_2[:args.user]
            m2 = self.MLP21(uu_2, inter_u)
            m21 = torch.reshape(m2[:, 0], [args.user, 1])
            m22 = torch.reshape(m2[:, 1], [args.user, 1])
            embeds = torch.cat((uu_2 * m21 + inter_u * m22, ui_2[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)
            
            uu_3 = embedsLst_social[3]
            ui_3 = embedsLst_interaction[3]
            inter_u = ui_3[:args.user]
            m3 = self.MLP31(uu_3, inter_u)
            m31 = torch.reshape(m3[:, 0], [args.user, 1])
            m32 = torch.reshape(m3[:, 1], [args.user, 1])
            embeds = torch.cat((uu_3 * m31 + inter_u * m32, ui_3[args.user:]), dim=0)
            embeds_inter = self.gcn(interaction_adj, embeds)
            embedsLst_interaction.append(embeds_inter)
            embedsLst.append(embeds)

        main_embeds = sum(embedsLst_interaction)
        out = sum(abation)

        return main_embeds, out


    def loss_graphcl(self, user_embeddings1,user_embeddings2,user):  
            T = args.temp
            user_embeddings1 = F.normalize(user_embeddings1, dim=1)
            
            user_embeddings2 = F.normalize(user_embeddings2, dim=1)


            user_embs1 = F.embedding(user, user_embeddings1)
       
            user_embs2 = F.embedding(user, user_embeddings2)
   

           

            all_embs1_abs = user_embs1.norm(dim=1)
            all_embs2_abs = user_embs2.norm(dim=1)

            sim_matrix = torch.einsum('ik,jk->ij', user_embs1, user_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
                                                                                        all_embs2_abs)
            sim_matrix = torch.exp(sim_matrix / T)  
            pos_sim = sim_matrix[np.arange(user_embs1.shape[0]), np.arange(user_embs2.shape[0])]  
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  
            loss = - torch.log(loss)  


            return loss

    # def loss_graphcl(self, x1, x2, users, items,T):  
    #     user_embeddings1, item_embeddings1 = torch.split(x1, [args.user, args.item], dim=0)
    #     user_embeddings2, item_embeddings2 = torch.split(x2, [args.user, args.item], dim=0)

    #     user_embeddings1 = F.normalize(user_embeddings1, dim=1)
    #     item_embeddings1 = F.normalize(item_embeddings1, dim=1)
    #     user_embeddings2 = F.normalize(user_embeddings2, dim=1)
    #     item_embeddings2 = F.normalize(item_embeddings2, dim=1)

    #     user_embs1 = F.embedding(users, user_embeddings1)
    #     item_embs1 = F.embedding(items, item_embeddings1)
    #     user_embs2 = F.embedding(users, user_embeddings2)
    #     item_embs2 = F.embedding(items, item_embeddings2)

    #     all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
    #     all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

    #     all_embs1_abs = all_embs1.norm(dim=1)
    #     all_embs2_abs = all_embs2.norm(dim=1)

    #     sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
    #                                                                                 all_embs2_abs)
    #     sim_matrix = torch.exp(sim_matrix / T)  
    #     pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]  
    #     loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  
    #     loss = - torch.log(loss)  


    #     return loss

    def loss_graphcl_2(self, x1, x2, users, items):  # 对比损失函数
        T = args.temp
        user_embeddings1, item_embeddings1 = torch.split(x1, [args.user, args.item], dim=0)
        user_embeddings2, item_embeddings2 = torch.split(x2, [args.user, args.item], dim=0)

        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

        all_embs1_abs = all_embs1.norm(dim=1)
        all_embs2_abs = all_embs2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
                                                                                    all_embs2_abs)

        sim_matrix = torch.exp(sim_matrix / T)  
        pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]  

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  
        loss = - torch.log(loss)  
        loss_user = loss[:len(users)]
        loss_item = loss[len(items)]
        return loss_user, loss_item
    
# class Inter_denoise_gate(torch.nn.Module):
#     def __init__(self):
#         super(Inter_denoise_gate, self).__init__()
#         self.W_cf = nn.Linear(args.latdim, args.latdim)
#         self.W_soc = nn.Linear(args.latdim, args.latdim)
#         self.W_mix = nn.Linear(args.latdim, args.latdim)
        
#         self.W_en = nn.Linear(int(2*args.latdim), args.latdim)
        
#         self.sig = nn.Sigmoid()
#         self.tanh = nn.Tanh()
        
#     def forward(self, embs_cf, embs_soc):
#         embs = self.W_cf(embs_cf)+self.W_soc(embs_soc)+self.W_mix(embs_soc*embs_cf)
#         forget = self.sig(embs)
        
#         embs = self.W_en(torch.cat((embs_cf, embs_soc), -1))
#         enhance = self.sig(embs)
#         out = forget * embs_soc + enhance*self.tanh(self.W_soc(embs_soc))
#         return out

class MLP_L1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_L1, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, concatenated):
        
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output
class MLP_L2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_L2, self).__init__()
       
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, concatenated):
        
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output

class MLP_0(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_0, self).__init__()
       
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output
class MLP_1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_1, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
       
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output
class MLP_2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_2, self).__init__()
       
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
       
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output

class MLP_3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_3, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
       
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output

class MLP_4(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_4, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, y, z):
        
        concatenated = torch.cat((x, y, z), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output



class MLP_01(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_01, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output
class MLP_11(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_11, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
    
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output
class MLP_21(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_21, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output

class MLP_31(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_31, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output

class MLP_41(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_41, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim).to('cuda:0')
        
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, Zi, Hi):
        
        concatenated = torch.cat((Zi, Hi), dim=1)
        weight_output = F.softmax(self.leaky_relu(self.linear(concatenated)), dim=1)
        return weight_output


class EmbeddingMapper(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers):
        super(EmbeddingMapper, self).__init__()
        self.num_layers = num_layers
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)

        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])
        
        self.fc_last = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, x):
        x = F.elu(self.fc1(x))

        for layer in self.hidden_layers:
            x = F.elu(layer(x))

        x_mapped = self.fc_last(x)

        return x_mapped



class GCNLayer(nn.Module):  
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):

        return torch.spmm(adj, embeds)

class GCNLayer_social(nn.Module):
    def __init__(self):
        super(GCNLayer_social, self).__init__()

    def forward(self, adj, embeds):

        return torch.spmm(adj, embeds)



class vgae_encoder(Model):
    def __init__(self):
        super(vgae_encoder, self).__init__()
        hidden = args.latdim
        self.encoder_mean = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.encoder_std = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                         nn.Softplus())

    def forward(self, adj):  
        x = self.forward_graphcl(adj)
        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)
        gaussian_noise = torch.randn(x_mean.shape).cuda()
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std


class vgae_decoder(nn.Module):
    def __init__(self, hidden=""):
        super(vgae_decoder, self).__init__()
        hidden=args.latdim
        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                                     nn.Linear(hidden, 1))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='none')

    def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
        x_user, x_item = torch.split(x, [args.user, args.item], dim=0)

        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).cuda())
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).cuda())
        loss_rec = loss_edge_pos + loss_edge_neg

        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean ** 2 - x_std ** 2).sum(dim=1)

        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
        regLoss = calcRegLoss(encoder) * args.reg

        beta = 0.1 #0.1
        loss = (loss_rec + beta * kl_divergence.mean()).mean()
        return loss


class vgae(nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, users, items, neg_items):
        x, x_mean, x_std = self.encoder(data)
        loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)
        return loss

    def generate(self, data, edge_index, adj):
        x, _, _ = self.encoder(data)

        edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))

        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        edge_pred = edge_pred[:, 0]
        mask = ((edge_pred + args.mask).floor()).type(torch.bool)#lastfm 0.7  ciao 0.5

        newVals = vals[mask]

        newVals = newVals / (newVals.shape[0] / edgeNum[0])
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class SocialDenoising(Model):
    """相似度计算器：多种相似度度量"""
    def __init__(self, social_adj):
        super(SocialDenoising, self).__init__()
        self.social_adj = social_adj #社交图
          # 从稀疏张量中提取索引和值
        self.indices = self.social_adj.indices().cpu().numpy()
        self.values = self.social_adj.values().cpu().numpy()
        #获得社交对列表
        self.social_pairs , self.strengths = self.get_all_social_pairs(social_adj)
        self.weights  = nn.Parameter(torch.tensor([0.5, 0.5]))  # [结构,特征,时序]
        #self.edgePredictor = EdgePredictor()
        self.user_embedding = self.uEmbeds.weight
    def get_all_social_pairs(self, social_adj):
    #"""获取所有的社交对和社交强度"""
        if social_adj is None:
            return [], []
        
        # 从稀疏张量中提取索引和值
        indices = self.social_adj.indices().cpu().numpy()
        values = self.social_adj.values().cpu().numpy()
        
        # 组织成社交对列表
        social_pairs = []
        social_strengths = []
        
        for i in range(indices.shape[1]):
            node1 = indices[0, i]
            node2 = indices[1, i]
            strength = values[i]
            
            # 避免重复（无向图）
            if node1 <= node2:
                social_pairs.append((node1, node2))
                social_strengths.append(strength)
    
        return social_pairs, social_strengths
    
    def get_direct_neighbors(self, node):
        """获取直接邻居（一跳邻居）"""
        if self.social_adj is None:
            return set()
        
        indices = self.indices
        values = self.values
        
        neighbors = set()
        for i in range(indices.shape[1]):
            if len(neighbors)==50:
                break
            elif indices[0, i] == node:
                neighbors.add(indices[1, i])
            elif indices[1, i] == node:  # 无向图，双向考虑
                neighbors.add(indices[0, i])
            
        
        return neighbors
    
    def get_second_hop_neighbors(self, node):
        """获取邻居的邻居（二跳邻居）"""
        direct_neighbors = self.get_direct_neighbors(node)
        second_hop_neighbors = set()
        
        # 获取每个直接邻居的邻居
        for neighbor in direct_neighbors:
            neighbor_neighbors = self.get_direct_neighbors(neighbor)
            second_hop_neighbors.update(neighbor_neighbors)
        
        # 排除自身和直接邻居
        second_hop_neighbors.discard(node)
        second_hop_neighbors = second_hop_neighbors - direct_neighbors
        
        return second_hop_neighbors
    
    def structural_similarity(self, edge_index, node_pairs):
        """基于图结构的相似度"""
        similarities = []
        for (i, j) in node_pairs:
            # 检查是否为PyTorch张量，如果是则转为NumPy数组，否则直接使用
            if isinstance(edge_index, torch.Tensor):
                neighbors_i = set(edge_index[1][edge_index[0] == i].cpu().numpy())
            else:
                neighbors_i = set(edge_index[1][edge_index[0] == i])

            if isinstance(edge_index, torch.Tensor):
                neighbors_j = set(edge_index[1][edge_index[0] == j].cpu().numpy())
            else:
                neighbors_j = set(edge_index[1][edge_index[0] == j])
            # Jaccard相似度（使用邻居集合）
            # neighbors_i = set((edge_index[1][edge_index[0] == i]).cpu().numpy())
            # neighbors_j = set((edge_index[1][edge_index[0] == j]).cpu().numpy())
            
            if len(neighbors_i) == 0 and len(neighbors_j) == 0:
                sim = 0.0
            else:
                intersection = len(neighbors_i.intersection(neighbors_j))
                union = len(neighbors_i.union(neighbors_j))
                sim = intersection / union if union > 0 else 0.0
            similarities.append(sim)
        
        return torch.tensor(similarities, dtype=torch.float32)
    
    def embedding_similarity(self, user_embedding, node_pairs):
        """基于节点嵌入的相似度"""
        indices_i = torch.tensor([pair[0] for pair in node_pairs])
        indices_j = torch.tensor([pair[1] for pair in node_pairs])
        
        z_i = user_embedding[indices_i]
        z_j = user_embedding[indices_j]
        
        # 余弦相似度
        cosine_sim = F.cosine_similarity(z_i, z_j)
        return cosine_sim
    
    def combined_similarity(self, user_embedding, edge_index, node_pairs, temporal_data=None):
        """组合相似度计算"""
        struct_sim = self.structural_similarity(edge_index, node_pairs)
        embed_sim = self.embedding_similarity(user_embedding, node_pairs)
        
        # 归一化权重
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # 组合相似度
        combined = (normalized_weights[0] * struct_sim + 
                   normalized_weights[1] * embed_sim)
            
        return combined
    
    def calculate_dynamic_threshold(self, node_pairs):
        """基于深度学习的动态阈值计算"""
        with torch.no_grad():
            # 计算节点度数特征
            degrees = []
            for (i, j) in node_pairs:
                deg_i = (node_pairs[0] == i).sum().item()
                deg_j = (node_pairs[0] == j).sum().item()
                degrees.append((deg_i + deg_j) / 2)
            
            degrees = torch.tensor(degrees, dtype=torch.float32).to(self.device)
            
            # 基于度数的动态阈值
            base_threshold = 0.1
            # 度数越小，阈值越低（更保守）
            #adjusted_threshold = base_threshold * (1 +  0.1*torch.tanh(degrees / 10 - 1))
            adjusted_threshold = base_threshold * (1 +  5*degrees) 
            # adjusted_threshold = base_threshold * (1 - 0.1 * torch.tanh(degrees / 10 - 1))
            return adjusted_threshold.cpu().numpy()
        
    def deep_edge_removal(self, user_embedding,removal_strategy='false'):
        """基于深度学习的边删除"""
        edges_to_remove = []
        edge_index = self.indices

        node_pairs = self.social_pairs
        
        with torch.no_grad():

            # similarities = self.combined_similarity(self.user_embedding,
            #     edge_index, node_pairs)
            # similarities = self.structural_similarity(
            #     edge_index, node_pairs)#笛卡及 结构相似度
            similarities = self.embedding_similarity(
                user_embedding, node_pairs)
            if removal_strategy == 'adaptive':
                thresholds = self.calculate_dynamic_threshold(node_pairs)
                for idx, (edge, similarity) in enumerate(zip(node_pairs, similarities)):
                    if similarity < thresholds[idx]:
                        edges_to_remove.append(edge)
            else:
                # 固定阈值
                for edge, similarity in zip(node_pairs, similarities):
                    if similarity < 0.4:#0.4
                        edges_to_remove.append(edge)

        return edges_to_remove #返回需要删除的边集合
    
    def deep_edge_addition(self, addition_threshold=0.4):
        """基于深度学习的边添加"""
        indices = self.indices
        node_set = set(indices[0] | indices[1])
        node_list = list(node_set)
        edges_to_add = []
        candidate_pairs = []
        # 生成候选边

        for i in range(len(node_list)):
            node = node_list[i]
            node_second = list(self.get_second_hop_neighbors(node))
            if node_second == set():
                continue
            for i in range(len(node_second)):
                candidate_pairs.append((node,node_second[i]))
        
        user, user_second = zip(*candidate_pairs)
        new_indices = np.vstack([user, user_second])

        with torch.no_grad():
            # 计算相似度
            edges_to_add = []
            similarities = self.combined_similarity(self.user_embedding, new_indices, candidate_pairs)            
            # 使用边预测器预测存在概率
            #candidate_edges = torch.tensor(candidate_pairs).t().to(self.device)
            # z_src = z[candidate_edges[0]]
            # z_dst = z[candidate_edges[1]]
            #edge_embeddings = torch.cat([z_src, z_dst], dim=1)
            #existence_probs = self.edge_predictor.predictor(edge_embeddings).squeeze()
            # 综合决策：相似度 + 存在概率
            #combined_scores = 0.7 * similarities + 0.3 * existence_probs
            for idx, (edge, similarity) in enumerate(zip(candidate_pairs, similarities)):
                    if similarity > addition_threshold:
                        edges_to_add.append(edge)
        
        return edges_to_add
    def normalizeAdj_torch(self, mat):
        """
        归一化 PyTorch 稀疏邻接矩阵（对称归一化：D^(-1/2) * A * D^(-1/2)）
        :param mat: PyTorch 稀疏张量（shape: [n, n]，需在 CPU 上）
        :return: 归一化后的 PyTorch 稀疏张量
        """
        device = mat.device
        n = mat.shape[0]
        # 1. 计算节点度数（稀疏矩阵求和，转为 dense 1D 张量）
        degree = torch.sparse.sum(mat, dim=1).to_dense().squeeze()  # [n,]
        # 2. 计算 D^(-1/2)，处理度数为 0 的情况
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[degree == 0] = 0.0  # 孤立节点设为 0
        # 3. 构建对角矩阵（PyTorch 稀疏格式）
        indices = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)  # [2, n]
        d_inv_sqrt_mat = torch.sparse.FloatTensor(indices, d_inv_sqrt, [n, n])
        # 4. 对称归一化：D^(-1/2) * A * D^(-1/2)（稀疏矩阵乘法用 torch.spmm）
        normalized_mat = torch.spmm(torch.spmm(d_inv_sqrt_mat, mat), d_inv_sqrt_mat)
        return normalized_mat
    
    # def normalizeAdj(self, mat):  #归一化邻接矩阵
	# 	degree = np.array(mat.sum(axis=-1))
	# 	dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
	# 	dInvSqrt[np.isinf(dInvSqrt)] = 0.0
	# 	dInvSqrtMat = sp.diags(dInvSqrt) #将处理过的度数组转换为对角矩阵
	# 	return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def build_socialgraph(self,user_embedding):
        shape = self.social_adj.shape
        num_nodes = shape[0]
        edges_to_remove = self.deep_edge_removal(user_embedding)#得到删除边
        #edges_to_add =self.deep_edge_addition()#得到增加边
        node_pairs = self.social_pairs #原社交边
        #social_strengths = self.strengths #原所有强度
        new_node_pairs = list(set(node_pairs) - set(edges_to_remove))
        #new_node_pairs = list(set(new_node_pairs) | set(edges_to_add))
        
        
        #new_node_pairs = node_pairs - edges_to_remove #删除边后的社交边集合
        #new_social_strengths = []
        # for edge, strength in zip(node_pairs, social_strengths):
        #     if edge not in edges_to_remove:
        #         new_social_strengths.append(strength)
        if not new_node_pairs:
            # 处理空的情况（根据业务逻辑选择）：
            # 情况1：允许空社交图，初始化空的rows和cols
            rows, cols = (), ()
            # 情况2：如果必须有节点对，抛出明确错误并提示原因
            # raise ValueError("new_node_pairs为空，请检查节点对生成逻辑")
        else:
            rows, cols = zip(*new_node_pairs)
        #rows, cols = zip(*new_node_pairs)
        data = np.ones(len(new_node_pairs), dtype=np.float32)
        # 创建COO格式稀疏矩阵
        coo_mat = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        # 转换为PyTorch稀疏张量
        indices = np.vstack([coo_mat.row, coo_mat.col])
        indices = t.from_numpy(indices.astype(np.int64))
        values = t.from_numpy(coo_mat.data.astype(np.float32))
        social_mat = t.sparse.FloatTensor(indices, values, t.Size(coo_mat.shape))
        norm_social_mat = self.normalizeAdj_torch(social_mat)
        return norm_social_mat
    

    def denoise_graph(self, S,R,inter_user_emb,inter_item_emb, threshold1=0.8,threshold2=0.6):
        #计算相似度矩阵
        user_user_sim_matrix = self.uu_cosine_similarity(inter_user_emb)
        user_user_sim_matrix = 1/2*(user_user_sim_matrix + torch.tensor(1.0, device=inter_user_emb.device).cuda()) 
        S_indices = S.indices().cpu().numpy()  # [2, N]，第0行是行索引，第1行是列索引
        S_values = S.values().cpu().numpy()    # 非零元素的值（CPU 上的 NumPy 数组）
        S_rows = S_indices[0, :]
        S_cols = S_indices[1, :]
        #将相似度矩阵小于阈值的位置用0mask
        valid_sim_values = user_user_sim_matrix[S_rows,S_cols].cpu()
        mask_denoise = (valid_sim_values<threshold1)
         # 保留符合条件的边（不符合的置0），注意 S_values 是 NumPy 数组，直接用布尔索引
        remained_sim_values = S_values.copy()
        remained_sim_values[mask_denoise] = 0.0
        weighted_edge = S_values * remained_sim_values
        # 构建去噪后的社交矩阵（SciPy CSR 格式）
        shape_S = (S.shape[0], S.shape[1])
        denoised_S_adj = sp.csr_matrix((weighted_edge, (S_rows, S_cols)), shape=shape_S)
        denoised_S_adj.eliminate_zeros()  # 删除值为0的元素（压缩稀疏矩阵）
        # weighted_edge = nonzero_values * remained_sim_values##weigh
        # # sim_matrix = torch.FloatTensor(sim_matrix)
        # shape = (S.shape[0],S.shape[1])
        # #用更新过的元素值创建新的稀疏矩阵，称为去燥后的邻接矩阵
        # denoised_S_adj = sp.csr_matrix((weighted_edge,(rows,cols)),shape)
        # denoised_S_adj.eliminate_zeros()

        denoised_norm_S_adj = self.normalize_graph_mat(denoised_S_adj)

        ###用u-i嵌入乘以denoised S去计算社交增强的偏好嵌入
        torch_S = self.convert_sparse_mat_to_tensor(denoised_norm_S_adj).cuda()

        social_user_emb = torch.sparse.mm(torch_S,inter_user_emb) ## cuda
        # #计算u-i的u-u相似度矩阵
        user_item_sim_matrix = self.ui_cosine_similarity(social_user_emb,inter_item_emb)
        user_item_sim_matrix = 1/2*(user_item_sim_matrix + torch.tensor(1.0,device=inter_user_emb.device).cuda()) 

        R_indices = R.indices().cpu().numpy()  # [2, M]，M为 R 的非零元素数
        R_values = R.values().cpu().numpy()    # R 的非零元素值
        rows_R = R_indices[0, :]
        cols_R = R_indices[1, :]
        # 裁剪行索引到 [0, row_bound]
        row_bound = user_item_sim_matrix.shape[0] - 1
        rows_R = np.clip(rows_R,0, row_bound)  # PyTorch张量用clamp，numpy数组用np.clip

        # 裁剪列索引到 [0, col_bound]
        col_bound = user_item_sim_matrix.shape[1] - 1
        cols_R = np.clip(cols_R,0, col_bound)  # PyTorch张量用clamp，numpy数组用np.clip
         # 关键修复：过滤超出范围的索引
        # max_user = args.user - 1  # 最大有效用户索引
        # max_item = args.item - 1  # 最大有效物品索引
        # # 保留用户索引 < args.user 且 物品索引 < args.item 的索引
        # valid_mask = (rows_R <= max_user) & (cols_R <= max_item)
        # # 过滤后的有效索引
        # rows_R = rows_R[valid_mask]
        # cols_R = cols_R[valid_mask]
        # ---------------------- 5. 过滤 UI 矩阵的低相似度边（逻辑不变） ----------------------
        valid_sim_values_R = user_item_sim_matrix[rows_R, cols_R].detach().cpu().numpy()
        mask_denoise_R = (valid_sim_values_R < threshold2)
        remained_sim_values_R = R_values.copy()
        remained_sim_values_R[mask_denoise_R] = 0.0
        #remained_sim_values = remained_sim_values.detach().numpy()
        weighted_edge_R = R_values * remained_sim_values_R
        shape_R = (R.shape[0], R.shape[1])
        denoised_R_adj = sp.csr_matrix((weighted_edge_R, (rows_R, cols_R)), shape=shape_R)
        denoised_R_adj.eliminate_zeros()
        # #用相似度给u-u边加权
        # weighted_edge = nonzero_values * remained_sim_values##weigh
        # # sim_matrix = torch.FloatTensor(sim_matrix)
        # shape = (R.shape[0],R.shape[1])
        # #用更新过的元素值创建新的稀疏矩阵，称为去燥后的邻接矩阵
        # denoised_R_adj = sp.csr_matrix((weighted_edge,(rows,cols)),shape)
        # denoised_R_adj.eliminate_zeros()

        denoised_norm_R_adj = self.normalize_graph_mat(denoised_R_adj)
        torch_R = self.convert_sparse_mat_to_tensor(denoised_norm_R_adj).cuda()

        
        
        return torch_S,torch_R
    
    def ui_cosine_similarity(self,user_emb, item_emb):
        """计算用户-物品余弦相似度（user_emb: [U, D], item_emb: [I, D]）"""
        # 归一化嵌入（避免重复计算范数）
        user_emb_norm = F.normalize(user_emb, dim=1)
        item_emb_norm = F.normalize(item_emb, dim=1)
        # 计算相似度矩阵 [U, I]，并归一化到 [0,1]
        sim_matrix = torch.matmul(user_emb_norm, item_emb_norm.T)
        sim_matrix = 0.5 * (sim_matrix + torch.tensor(1.0, device=user_emb.device))
        return sim_matrix
    
    def uu_cosine_similarity(self, X):
        """
        计算输入矩阵中每一行之间的余弦相似度，并将对角线元素置为0。

        参数：
        - X: 输入矩阵，每一行是一个向量。

        返回：
        - similarities: 余弦相似度矩阵，每个元素表示对应向量之间的余弦相似度。
        """
        # 将对角线元素置为0

        # 计算矩阵的内积
        dot_product = torch.matmul(X, X.T)

        # 计算每个向量的范数
        norm = torch.norm(X, dim=1, keepdim=True)
        norm[norm==0] = 1e-9
        # 计算余弦相似度
        similarities = dot_product / (norm * norm.T)
        # print(similarities)
        diagonal = torch.eye(similarities.shape[0], dtype=similarities.dtype).cuda()
        # print(diagonal)
        similarities = similarities - diagonal
        # print(similarities)
        return similarities
    
    def normalize_graph_mat(self,adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat
    
    def convert_sparse_mat_to_tensor(self,X):
        coo = X.tocoo()

        edge_indices_np = np.array([coo.row, coo.col], dtype=np.int64)
        # 2. 用torch.from_numpy转张量（避免列表解析，效率更高）
        i = torch.from_numpy(edge_indices_np).long()
        #i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SDNet(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(SDNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            if x.is_sparse:
            # 稀疏转密集（仅当非零元素少、显存足够时可行）
                x = x.to_dense()  
                x = F.normalize(x)  # 此时x是密集张量，支持任意dim归一化
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


    def timestep_embedding(timesteps, dim, max_period=10000):

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


    def mean_flat(tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))