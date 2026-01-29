import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import community as community_louvain
import networkx as nx
from Params import args


class Louvain(nn.Module):
    """基于Louvain社区检测和超图GCN的社交推荐模型"""
    
    def __init__(self):
        super(Louvain, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = args.latdim
        self.num_users = args.user
        # 超图GCN层
        #self.hyper_gcn = HyperGCN(embedding_dim, hidden_dim, embedding_dim).to(self.device)
        self.hyper_gcn = HyperGCN(self.embedding_dim, self.embedding_dim*2,self.embedding_dim//2,self.embedding_dim).to(self.device)

    def get_all_social_pairs(self, social_adj):
    #"""获取所有的社交对和社交强度"""
        if social_adj is None:
            return [], []
        
        # 从稀疏张量中提取索引和值
        indices = social_adj.indices().cpu().numpy()
        values = social_adj.values().cpu().numpy()
        
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
            
    def build_community_hypergraph(self, social_relations):
        """使用Louvain检测社区并构建超图"""
        # 构建社交图
        G = nx.Graph()
        G.add_edges_from(social_relations)
        
        # Louvain社区检测
        partition = community_louvain.best_partition(G)
        
        # 构建社区到用户的映射
        communities = defaultdict(list)
        for user, comm_id in partition.items():
            communities[comm_id].append(user)
        
        # 构建超图关联矩阵 H: [num_users, num_communities]
        num_communities = len(communities)
        H = torch.zeros(self.num_users, num_communities)
        
        for comm_id, users in communities.items():
            for user in users:
                if user < self.num_users:  # 确保用户ID在范围内
                    H[user, comm_id] = 1.0
        
        return H, communities
    
    def forward(self, user_embedding, social_relations, H, communities):
        # 构建社区超图
        #H, communities = self.build_community_hypergraph(social_relations)
        H = H
        communities = communities
        H = H.to(self.device)
        
        # 超图GCN聚合
        community_enhanced_emb = self.hyper_gcn(user_embedding, H)  # [num_users, emb_dim]
        return community_enhanced_emb
        
        

class HyperGCN(nn.Module):
    """超图卷积网络层"""
    
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, feature_pre = True, layer_num = 2, dropout=True):
        super(HyperGCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(feature_dim, output_dim, bias=True)
        self.prelu = nn.PReLU()
        # self.gcn1 = nn.Linear(in_dim, hidden_dim)
        # self.gcn2 = nn.Linear(hidden_dim, out_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.activation = nn.ReLU()
       
    def forward(self, x, H):
        """
        x: 节点特征 [num_nodes, in_dim]
        H: 超图关联矩阵 [num_nodes, num_hyperedges]
        """
        # 计算超边特征: 超边内节点的平均特征
        H_T = H.T  # [num_hyperedges, num_nodes]
        hyperedge_degrees = torch.sum(H_T, dim=1, keepdim=True)  # [num_hyperedges, 1]
        hyperedge_features = torch.matmul(H_T, x) / torch.clamp(hyperedge_degrees, min=1)  # [num_hyperedges, in_dim]
        
        # 计算节点的新特征: 节点所属超边的平均特征
        node_degrees = torch.sum(H, dim=1, keepdim=True)  # [num_nodes, 1]
        aggregated_features = torch.matmul(H, hyperedge_features) / torch.clamp(node_degrees, min=1)  # [num_nodes, in_dim]
        
    
        if self.feature_pre:
            x = self.linear_pre(aggregated_features)
        x = self.prelu(x)
        for i in range(self.layer_num - 2):
            x =  self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x
        #return aggregated_features


