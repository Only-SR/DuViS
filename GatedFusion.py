import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    """门控融合机制"""
    
    def __init__(self, embedding_dim, num_inputs=3):
        super(GatedFusion, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 门控权重
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * num_inputs, num_inputs),
            nn.Sigmoid()
        )
        
        # 转换层
        self.transform = nn.Linear(embedding_dim * num_inputs, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, *embeddings):
        """融合多个嵌入"""
        if len(embeddings) == 1:
            return embeddings[0]
            
        # 拼接所有嵌入
        concated = torch.cat(embeddings, dim=-1)  # [batch_size, emb_dim * num_inputs]
        
        # 计算门控权重
        gate_weights = self.gate(concated)  # [batch_size, emb_dim]
        
        # 转换和融合
        transformed = self.transform(concated)  # [batch_size, emb_dim]
        
        # 加权融合
        if len(embeddings) == 2:
            # 对于两个嵌入的情况，使用门控机制
            fused_emb = gate_weights * embeddings[0] + (1 - gate_weights) * embeddings[1]
        else:
            # 对于多个嵌入，使用平均加权
            weights = F.softmax(gate_weights, dim=-1)
            stacked_embs = torch.stack(embeddings, dim=1)  # [batch_size, num_inputs, emb_dim]
            fused_emb = torch.sum(weights.unsqueeze(-1) * stacked_embs, dim=1)
        
        #return self.layer_norm(fused_emb + transformed)
        return self.layer_norm(fused_emb)