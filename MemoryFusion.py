import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryFusion(nn.Module):
    """基于记忆网络的融合"""
    
    def __init__(self, embedding_dim, memory_size=64, num_heads=4):
        super(MemoryFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        
        # 记忆矩阵
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, user1, user2, user3):
        """融合用户、社交和兴趣嵌入"""
        # 基础融合
        base_fused = (user1 + user2 + user3) / 3
        
        # 与记忆交互
        memory_bank = self.memory.unsqueeze(1).repeat(1, user1.size(0), 1)  # [memory_size, batch_size, emb_dim]
        query = base_fused.unsqueeze(0)  # [1, batch_size, emb_dim]
        
        # 注意力读取记忆
        attended_memory, _ = self.attention(query, memory_bank, memory_bank)
        attended_memory = attended_memory.squeeze(0)  # [batch_size, emb_dim]
        
        # 融合原始嵌入和记忆增强嵌入
        final_emb = self.output_layer(torch.cat([base_fused, attended_memory], dim=-1))
        
        return final_emb