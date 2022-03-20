import torch
import torch.nn as nn
import torch.nn.functional as F

from .vp_att import FeedForward
from .Sparsemax import Sparsemax

"""
该文件下包含三个类的定义与实现：
1）SCAttEnc - 核心的注意力实现（通过qkv，计算注意力权重及加权求和）
2）MultiHeadAttentionEnc - 多头注意力机制实现
3）VP_Refine_Module - 视觉特征增强层
关系：3）-调用-> 2）-调用-> 1）
"""

############################################
# 用于Encoder部分，全局特征和目标特征的特征增强
############################################
# --------------   XLAN   --------------
# XLAN SCAtt 模块
class SCAttEnc(nn.Module):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAttEnc, self).__init__()
        self.attention_basic = nn.Sequential(
            nn.Linear(mid_dims[0], mid_dims[1]), 
            nn.ReLU(), 
            nn.Dropout(mid_dropout)
        )
        
        self.attention_last = nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])
        
    def forward(self, query, key, att_mask, value1, value2):
        # query [B, 8, 128]
        # key [B, 8, M, 128]
        # att_mask [B, M]
        # value1 [B, 8, 128]
        # value2 [B, 8, M, 128]
        
        att_map = query.unsqueeze(-2) * key  # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map_pool = att_map.mean(-2)  # [B, 8, 64]
        
        # Spatial Attention
        alpha_spatial = self.attention_last(att_map)  # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1)     # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)
        
        if len(alpha_spatial.shape) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            value2 = torch.matmul(alpha_spatial, value2)
        else:
            value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)  # [B, 8, 128]

        # Channel Attention
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)  # [B, 8, 128]
        
        attn = value1 * value2 * alpha_channel
        
        return attn
    
    
class MultiHeadAttentionEnc(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout):
        super(MultiHeadAttentionEnc, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        output_dim = embed_dim

        # query1 用于全局特征增强的query
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        sequential.append(nn.CELU(1.3))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q = nn.Sequential(*sequential)

        # keys
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        sequential.append(nn.CELU(1.3))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_k = nn.Sequential(*sequential)

        # values1 用于通道注意力的query
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        sequential.append(nn.CELU(1.3))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v1 = nn.Sequential(*sequential)

        # values2 作为真正的value，同时用于空间注意力和通道注意力
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        sequential.append(nn.CELU(1.3))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)

        self.attn_net = SCAttEnc(att_mid_dim, att_mid_drop)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, query, key, mask, value1, value2, precompute=False):
        """
        输入数据：
        query1: [B, 1024]
        query2: [B, M, 1024]
        key: [B, M, 1024]
        mask: [B, M]
        value1: [B, M, 1024]
        value2: [B, M, 1024]
        """
        # 输入数据全连接层
        batch_size = query.size()[0]
        q = self.in_proj_q(query)
        
        key = key.view(-1, key.size()[-1])
        k = self.in_proj_k(key)
        
        # value1 = value1.view(-1, value1.size()[-1])
        v1 = self.in_proj_v1(value1)
        
        value2 = value2.view(-1, value2.size()[-1])
        v2 = self.in_proj_v2(value2)
        
        # 输入数据维度变换，用于多头注意力
        # [B, 8, 128]
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k  = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 调用注意力机制核心操作函数
        # 将attn_map的计算置于attn_net中进行
        attn = self.attn_net(q, k, mask, v1, v2)

        # 将输出从多头维度上恢复为正确维度
        # [B, 8, 128] --> [B, 1024]
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        
        if self.dropout is not None:
            attn = self.dropout(attn)
        
        return attn

    
# 方式一：收集每一层的gv_feat，concat之后进行Linear投影
# 用于图像目标特征的增强（用于Encoder）
# 及Visual Persistence in Encoder体现（获取主要目标）
class VP_Refine_Module(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Refine_Module, self).__init__()
        
        self.layers = nn.ModuleList([])
        self.bifeat_emb = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([]) 
        for _ in range(layer_num):
            sublayer = MultiHeadAttentionEnc(
                embed_dim = embed_dim, 
                att_type = att_type, 
                att_heads = att_heads, 
                att_mid_dim = att_mid_dim, 
                att_mid_drop = att_mid_drop,
                dropout = dropout)
            self.layers.append(sublayer)
            
            self.bifeat_emb.append(nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ))

            self.layer_norms.append(torch.nn.LayerNorm(embed_dim))

        # gv_feat 投影
        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
        
        feat_arr = [gv_feat]
        for i, layer in enumerate(self.layers):
            # q, key, mask, v1, v2
            gv_feat = layer(gv_feat, att_feats, att_mask, gv_feat, att_feats)
            att_feats_cat = torch.cat([gv_feat.unsqueeze(1).expand_as(att_feats), att_feats], dim=-1)
            
            # att_feats 残差连接
            att_feats = self.bifeat_emb[i](att_feats_cat) + att_feats
            att_feats = self.layer_norms[i](att_feats)
            feat_arr.append(gv_feat)

        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        return gv_feat, att_feats

# 无特征增强操作
class VP_Refine_Module_init(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Refine_Module_init, self).__init__()

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
        
        return gv_feat, att_feats
    
# 门控自注意力机制
class VP_Refine_Module_gated_self_attn(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Refine_Module_gated_self_attn, self).__init__()
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.head_num = att_heads
        self.head_dim = embed_dim // att_heads
        self.scaling = self.head_dim ** -0.5
        """
        self.q_linears = nn.ModuleList([])
        self.k_linears = nn.ModuleList([])
        self.v_linears = nn.ModuleList([])
        """
        self.memories_linears = nn.ModuleList([])
        self.updata_linears   = nn.ModuleList([])
        # """
        self.gate_linears     = nn.ModuleList([])
        # """
        self.layer_norms      = nn.ModuleList([])
        
        for _ in range(layer_num):
            """
            self.q_linears.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.CELU(1.3),
                    # nn.GroupNorm(att_heads, embed_dim)
                )
            )
            self.k_linears.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.CELU(1.3),
                    # nn.GroupNorm(att_heads, embed_dim)
                )
            )
            self.v_linears.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.CELU(1.3),
                    # nn.GroupNorm(att_heads, embed_dim)
                )
            )
            """
            
            # """
            self.memories_linears.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.CELU(1.3),
                    nn.GroupNorm(att_heads, embed_dim)
                )
            )
            # """
            self.updata_linears.append(
                nn.Sequential(
                    nn.Linear(2*self.head_dim, self.head_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
            )
            # """
            self.gate_linears.append(
                nn.Linear(2*self.head_dim, self.head_dim)
            )
            # """
            self.layer_norms.append(
                nn.LayerNorm(embed_dim)
            )

        self.dropout = nn.Dropout(dropout)
        # gv_feat 投影
        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            
        feat_arr = [gv_feat]
        # for i, layer in enumerate(self.layers):
        for i in range(self.layer_num):
            # 门控自注意力对att_feats进行自增强
            """
            q = self.q_linears[i](att_feats)
            k = self.k_linears[i](att_feats)
            v = self.v_linears[i](att_feats)
            """
            att_feats_mem = att_feats.view(-1, self.embed_dim) # [B*M, 1024]
            memories = self.memories_linears[i](att_feats_mem) # [B*M, 1024]
            
            B = att_feats.size()[0]
            
            # 转换为多头模式
            att_feats = att_feats.view(B, -1, self.head_num, self.head_dim).transpose(1, 2)  # [B, 8, M, 128]
            memories = memories.view(B, -1, self.head_num, self.head_dim).transpose(1, 2)  # [B, 8, M, 128]
            
            attn_map = torch.matmul(att_feats, memories.transpose(-1, -2)) * self.scaling   # [B, 8, M, M]
            # attn_map = torch.matmul(q, k.transpose(-1, -2))
            
            if att_mask is not None:
                mask = att_mask.unsqueeze(1)  # [B, 1, M]
                mask_ext = mask.unsqueeze(1)  # [B, 1, 1, M]
                attn_map = attn_map.masked_fill(mask_ext==0, -1e9) # [B, 8, M, M]
            attn_scores = F.softmax(attn_map, dim=-1) # [B, 8, M, M]
            
            context = torch.matmul(attn_scores, att_feats)  # [B, 8, M, 128]
            inputs = torch.cat([att_feats, context], dim=-1) # [B, 8, M, 128*2]
            # """
            # 门控残差
            f_t = torch.tanh(self.updata_linears[i](inputs))  # [B, 8, M, 128]
            g_t = torch.sigmoid(self.gate_linears[i](inputs)) # [B, 8, M, 128]
            att_feats = g_t * f_t + (1 - g_t) * att_feats     # [B, 8, M, 128]
            # """

            # 普通残差
            """
            f_t = self.updata_linears[i](inputs)              # [B, M, 1024]
            att_feats = f_t + att_feats
            """
            
            # 维度还原 [B, M, 1024]
            att_feats = att_feats.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
            
            att_feats = self.layer_norms[i](att_feats)
            
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            feat_arr.append(gv_feat)

        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        
        return gv_feat, att_feats
    
# 方式一：收集每一层的gv_feat，concat之后进行Linear投影
# 用于图像目标特征的增强（用于Encoder）
# 及Visual Persistence in Encoder体现（获取主要目标）
# 在残差连接部分，加入门控机制
class VP_Refine_Module_gated_residual(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Refine_Module_gated_residual, self).__init__()
        
        self.layers = nn.ModuleList([])
        self.bifeat_emb = nn.ModuleList([])
        self.gate_linears = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([]) 
        for _ in range(layer_num):
            sublayer = MultiHeadAttentionEnc(
                embed_dim = embed_dim, 
                att_type = att_type, 
                att_heads = att_heads, 
                att_mid_dim = att_mid_dim, 
                att_mid_drop = att_mid_drop,
                dropout = dropout)
            self.layers.append(sublayer)
            
            self.bifeat_emb.append(nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ))
            
            self.gate_linears.append(
                nn.Linear(2 * embed_dim, embed_dim)
            )

            self.layer_norms.append(torch.nn.LayerNorm(embed_dim))

        # gv_feat 投影
        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
        
        feat_arr = [gv_feat]
        for i, layer in enumerate(self.layers):
            # q, key, mask, v1, v2
            gv_feat = layer(gv_feat, att_feats, att_mask, gv_feat, att_feats)
            att_feats_cat = torch.cat([gv_feat.unsqueeze(1).expand_as(att_feats), att_feats], dim=-1)
            
            # att_feats 残差连接
            # att_feats = self.bifeat_emb[i](att_feats_cat) + att_feats
            
            # att_feats 门控残差
            g_i = torch.sigmoid(self.gate_linears[i](att_feats_cat)) # [B, M, 1024]
            att_feats = g_i * self.bifeat_emb[i](att_feats_cat) + (1 - g_i) * att_feats
            att_feats = self.layer_norms[i](att_feats)
            feat_arr.append(gv_feat)

        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        return gv_feat, att_feats