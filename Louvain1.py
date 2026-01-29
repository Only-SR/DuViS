import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from Params import args
class MultiScaleHyperGraphAttention(nn.Module):
    def __init__(self, in_dim, hidden_dims, num_heads, dropout=0.2, feature_pre=True):
        """
        in_dim: 输入特征维度
        hidden_dims: 各隐藏层维度列表
        num_heads: 注意力头数
        dropout: dropout率
        feature_pre: 是否使用特征预处理
        """
        super(MultiScaleHyperGraphAttention, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.feature_pre = feature_pre
        self.device = torch.device('cuda:'+args.device if torch.cuda.is_available() else 'cpu')

        # 多尺度特征提取
        self.multi_scale_layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dims[0] // 4),
            nn.Linear(in_dim, hidden_dims[0] // 4),
            nn.Linear(in_dim, hidden_dims[0] // 4),
            nn.Linear(in_dim, hidden_dims[0] // 4)
        ])
        
        # 注意力机制
        self.node_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.hyperedge_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征预处理
        if feature_pre:
            self.linear_pre = nn.Linear(hidden_dims[0], hidden_dims[0])
            self.pre_norm = nn.LayerNorm(hidden_dims[0])
        
        # 残差连接的隐藏层
        self.residual_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.residual_layers.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout=dropout)
            )
        
        # 输出层
        self.linear_out = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.out_norm = nn.LayerNorm(hidden_dims[-1])
        
        # 超边权重学习
        self.hyperedge_weight = nn.Parameter(torch.ones(1))
        
        # 门控机制
        self.gate = nn.Linear(hidden_dims[0] * 2, hidden_dims[0])
        
    def forward(self, x, H, adj_mask=None):
        """
        x: 节点特征 [num_nodes, in_dim]
        H: 超图关联矩阵 [num_nodes, num_hyperedges]
        adj_mask: 邻接矩阵掩码（可选）
        """
        # #1. 多尺度特征提取
        scale_features = []
        for layer in self.multi_scale_layers:
            scale_feat = F.leaky_relu(layer(x))
            scale_features.append(scale_feat)
        
        # 拼接多尺度特征
        x_multi_scale = torch.cat(scale_features, dim=1)  # [num_nodes, hidden_dims[0]]
        H = H.to(self.device) 
        # 2. 节点级注意力聚合
        H_T = H.T  # [num_hyperedges, num_nodes]
        
        # 计算超边特征（带注意力）
        hyperedge_degrees = torch.sum(H_T, dim=1, keepdim=True)
        hyperedge_features_base = torch.matmul(H_T, x_multi_scale) / torch.clamp(hyperedge_degrees, min=1)
        
        # 超边注意力增强
        hyperedge_features_attn, _ = self.hyperedge_attention(
            hyperedge_features_base.unsqueeze(0),
            hyperedge_features_base.unsqueeze(0),
            hyperedge_features_base.unsqueeze(0)
        )
        hyperedge_features = hyperedge_features_attn.squeeze(0)
        
        # 3. 节点特征更新（带注意力）
        node_degrees = torch.sum(H, dim=1, keepdim=True)
        aggregated_features_base = torch.matmul(H, hyperedge_features) / torch.clamp(node_degrees, min=1)
        
        # 节点注意力增强
        aggregated_features_attn, attention_weights = self.node_attention(
            x_multi_scale.unsqueeze(0),
            aggregated_features_base.unsqueeze(0),
            aggregated_features_base.unsqueeze(0)
        )
        aggregated_features_attn = aggregated_features_attn.squeeze(0)
        
        # 4. 门控融合原始特征和聚合特征
        gate_input = torch.cat([x_multi_scale, aggregated_features_attn], dim=1)
        gate_signal = torch.sigmoid(self.gate(gate_input))
        x_enhanced = gate_signal * x_multi_scale + (1 - gate_signal) * aggregated_features_attn
        
        # 5. 特征预处理（如果启用）
        if self.feature_pre:
            x_enhanced = self.linear_pre(x_enhanced)
            x_enhanced = self.pre_norm(x_enhanced)
        
        x_enhanced = F.leaky_relu(x_enhanced)
        
        # 6. 残差多层处理
        for layer in self.residual_layers:
            x_enhanced = layer(x_enhanced)
        
        # 7. 输出处理
        x_out = self.linear_out(x_enhanced)
        x_out = self.out_norm(x_out)
        x_out = F.normalize(x_out, p=2, dim=-1)
        
        return x_out, attention_weights

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 如果输入输出维度不同，需要投影
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.linear1(x)
        out = self.norm1(out)
        out = F.leaky_relu(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.norm2(out)
        
        out += identity  # 残差连接
        out = F.leaky_relu(out)
        out = self.dropout(out)
        
        return out

class Louvain1(nn.Module):
    """增强的超图模型，结合传统GCN和超图卷积"""
    def __init__(self, in_dim, hyper_hidden_dims, gcn_hidden_dims, num_heads, dropout=0.2):
        super(Louvain1, self).__init__()
        
        # 超图分支
        self.hyper_graph = MultiScaleHyperGraphAttention(
            in_dim, hyper_hidden_dims, num_heads, dropout
        )
        
        # GCN分支（处理成对关系）
        # self.gcn_layers = nn.ModuleList()
        # gcn_dims = [in_dim] + gcn_hidden_dims
        # for i in range(len(gcn_dims) - 1):
        #     self.gcn_layers.append(
        #         GATConv(gcn_dims[i], gcn_dims[i+1] // num_heads, heads=num_heads, dropout=dropout)
        #     )
        
        # # 特征融合
        # self.fusion_gate = nn.Linear(hyper_hidden_dims[-1] + gcn_hidden_dims[-1], 2)
        #+ gcn_hidden_dims[-1]
        # 最终输出层
        self.output_layer = nn.Linear(hyper_hidden_dims[-1] , hyper_hidden_dims[-1])
        self.output_norm = nn.LayerNorm(hyper_hidden_dims[-1])
    
    def forward(self, x, H):
        """
        x: 节点特征
        H: 超图关联矩阵
        edge_index: 传统图的边索引
        """
        # 超图分支
        hyper_features, attention_weights = self.hyper_graph(x, H)
        # # GCN分支
        # gcn_features = x
        # for gcn_layer in self.gcn_layers:
        #     gcn_features = F.leaky_relu(gcn_layer(gcn_features, edge_index))
        
        # # 特征融合
        # combined_features = torch.cat([hyper_features, gcn_features], dim=1)
        # fusion_weights = F.softmax(self.fusion_gate(combined_features), dim=1)
        
        # fused_features = (fusion_weights[:, 0:1] * hyper_features + 
        #                  fusion_weights[:, 1:2] * gcn_features)
        
        # 输出
        output = self.output_layer(hyper_features)
        output = self.output_norm(output)
        output = F.normalize(output, p=2, dim=-1)
        
        return output
    #, attention_weights, fusion_weights