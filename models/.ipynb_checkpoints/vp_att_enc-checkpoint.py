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
"""
class SCAttEnc(nn.Module):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAttEnc, self).__init__()
        # 用于全局特征的通道注意力
        self.attention1_channel = nn.Linear(mid_dims[-1], mid_dims[-1])
        # 用于目标特征的通道注意力
        self.attention2_channel = nn.Linear(mid_dims[-1], mid_dims[-1])
        self.sparsemax = Sparsemax(dim = -1)
    
    def forward(self, query1, query2, key, att_mask, value1, value2):
        # query1 [B, 8, 128]       用于全局特征增强的spatial注意力
        # query2 [B, 8, M, 128]    用于目标特征增强的spatial注意力
        # key [B, 8, M, 128]
        # att_mask [B, M]
        # value1 [B, 8, M, 128]    用作channel注意力的query（全局特征和目标特征）
        # value2 [B, 8, M, 128]
        scaling = query1.shape[-1] ** -0.5      # (128) ** -0.5，即 1 / sqrt(128)
        
        ######################################
        # 1 先对目标特征增强
        ######################################
        # 1.1 空间注意力
        # 矩阵乘法： [B, 8, M, 128] x [B, 8, 128, M]  -->  [B, 8, M, M]
        att_map2_spatial = torch.matmul(query2, key.transpose(-1, -2)) * scaling

        # 1.2 通道注意力（同时用于目标特征和全局特征增强）
        # 矩阵点乘： [B, 8, M, 128] * [B, 8, M, 128]  -->  [B, 8, M, 128]
        query_channel = value1   # value1作为通道注意力的query
        # （1）[B, 8, M, 128]，用于目标特征
        att_map2_channel = query_channel * key  
        # （2）求att_map2_channel均值 [B, 8, 128]，用于全局变量的增强
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)  # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map2_channel_pool = torch.sum(att_map2_channel * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map2_channel_pool = att_map2_channel.mean(-2)
            
        # 【目标特征】空间注意力权重计算
        # [B, 8, M, M]
        if att_mask is not None:
            tmp_att_mask = att_mask.unsqueeze(-2) # [B, 1, 1, M]
            att_map2_spatial = att_map2_spatial.masked_fill(tmp_att_mask == 0, -1e9)
        alpha2_spatial = F.softmax(att_map2_spatial, dim=-1)
        # 矩阵乘法[B, 8, M, M] x [B, 8, M, 128] --> [B, 8, M, 128]
        attn2 = torch.matmul(alpha2_spatial, value2)
        
        # 【目标特征】通道注意力权重计算
        # att_map2_channel [B, 8, M, 128]  -->  [B, 8, M, 128]
        alpha2_channel = self.attention2_channel(att_map2_channel)
        alpha2_channel = torch.sigmoid(alpha2_channel)
        attn2 = alpha2_channel * attn2  # 增强后的目标特征
        
        ######################################
        # 2 再基于增强后的目标特征对全局特征进行增强
        ######################################
        # 2.1 空间注意力
        # 矩阵乘法： [B, 8, 1, 128] [B, 8, 128, M]  -->  [B, 8, 1, M]
        att_map1_spatial = torch.matmul(query1.unsqueeze(-2), key.transpose(-1, -2)) * scaling
        # [B, 8, 1, M]  -->  [B, 8, M]
        att_map1_spatial = att_map1_spatial.squeeze(-2)
        
        # 2.2 通道注意力
        att_map1_channel = att_map2_channel_pool
        
        # 【全局特征】空间注意力权重计算，此处使用的是Sparsemax
        # [B, 8, M]
        if att_mask is not None:
            att_map1_spatial = att_map1_spatial.masked_fill(att_mask == 0, -1e9)
        alpha1_spatial = self.sparsemax(att_map1_spatial)
        # 矩阵乘法[B, 8, 1, M] [B, 8, M, 128] --> [B, 8, 1, 128]  --> [B, 8, 128]
        attn1 = torch.matmul(alpha1_spatial.unsqueeze(-2), attn2).squeeze(-2)
        
        # 【全局特征】通道注意力权重计算
        # att_map1_channel [B, 8, 128]  -->  [B, 8, 128]
        alpha1_channel = self.attention1_channel(att_map1_channel)
        alpha1_channel = torch.sigmoid(alpha1_channel)
        attn1 = alpha1_channel * attn1
        
        # [B, 8, 128], [B, 8, M, 128]
        return attn1, attn2
"""

'''
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
        self.in_proj_q1 = nn.Sequential(*sequential)
        
        # query2 用于目标特征增强的query
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        sequential.append(nn.CELU(1.3))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q2 = nn.Sequential(*sequential)

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
        
        # 新增一个Linear层，用于最后获取的attended feats上
        sequential = []
        sequential.append(nn.Linear(output_dim, embed_dim))
        sequential.append(nn.CELU(1.3))
        self.fc1 = nn.Sequential(*sequential)
        
        sequential = []
        sequential.append(nn.Linear(output_dim, embed_dim))
        sequential.append(nn.CELU(1.3))
        self.fc2 = nn.Sequential(*sequential)

        self.attn_net = SCAttEnc(att_mid_dim, att_mid_drop)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, query1, query2, key, mask, value1, value2, precompute=False):
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
        batch_size = query1.size()[0]
        q1 = self.in_proj_q1(query1)
        
        query2 = query2.view(-1, query2.size()[-1])
        q2 = self.in_proj_q2(query2)
        
        key = key.view(-1, key.size()[-1])
        k = self.in_proj_k(key)
        
        value1 = value1.view(-1, value1.size()[-1])
        v1 = self.in_proj_v1(value1)
        
        value2 = value2.view(-1, value2.size()[-1])
        v2 = self.in_proj_v2(value2)
        
        # 输入数据维度变换，用于多头注意力
        # [B, 8, 128]
        q1 = q1.view(batch_size, self.num_heads, self.head_dim)
        # [B, M, 8, 128]
        q2 = q2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k  = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 调用注意力机制核心操作函数
        # 将attn_map的计算置于attn_net中进行
        attn_q1, attn_q2 = self.attn_net(q1, q2, k, mask, v1, v2)

        # 将输出从多头维度上恢复为正确维度
        # [B, 8, 128] --> [B, 1024]
        attn_q1 = attn_q1.view(batch_size, self.num_heads * self.head_dim)
        # [B, 8, M, 128] --> [B, M, 8, 128]  -->  [B, M, 1024]
        attn_q2 = attn_q2.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # 新增的Linear层
        attn_q1 = self.fc1(attn_q1)
        attn_q2 = self.fc2(attn_q2)
        if self.dropout is not None:
            attn_q1 = self.dropout(attn_q1)
            attn_q2 = self.dropout(attn_q2)
        return attn_q1, attn_q2
'''

    
"""
# 方式一：收集每一层的gv_feat，concat之后进行Linear投影
# 用于图像目标特征的增强（用于Encoder）
# 及Visual Persistence in Encoder体现（获取主要目标）
class VP_Refine_Module(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Refine_Module, self).__init__()
        
        self.layers = nn.ModuleList([])
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

            self.layer_norms.append(torch.nn.LayerNorm(embed_dim))

        # gv_feat 投影
        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(1024)

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
        
        feat_arr = [gv_feat]
        for i, layer in enumerate(self.layers):
            # q1, q2, key, mask, v1, v2
            # 对目标特征att_feats进行增强，得到att_feats_
            # 同时基于att_feats_获取gv_feat
            gv_feat, att_feats_ = layer(gv_feat, att_feats, att_feats, att_mask, att_feats, att_feats)
            # att_feats 残差连接
            att_feats = self.layer_norms[i](att_feats + att_feats_)
            feat_arr.append(gv_feat)

        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        return gv_feat, att_feats
"""

"""
# 方式二：与目标特征相似，对每一层的gv_feat进行残差连接，与layer_norm
# 用于图像目标特征的增强（用于Encoder）
# 及Visual Persistence in Encoder体现（获取主要目标）
class VP_Refine_Module(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Refine_Module, self).__init__()
        
        self.layers = nn.ModuleList([])
        # feed forward layer
        self.feedforwards_1 = nn.ModuleList([])
        self.feedforwards_2 = nn.ModuleList([])
        # Layer Norm，分别用于att_feats 和 gv_feat
        self.layer_norms_1 = nn.ModuleList([])
        self.layer_norms_2 = nn.ModuleList([])
        for _ in range(layer_num):
            # 核心层，进行att_feats的增强，与gv_feat的获取
            sublayer = MultiHeadAttentionEnc(
                embed_dim = embed_dim, 
                att_type = att_type, 
                att_heads = att_heads, 
                att_mid_dim = att_mid_dim, 
                att_mid_drop = att_mid_drop,
                dropout = dropout)
            self.layers.append(sublayer)
            
            # FeedForward
            subFF_1 = FeedForward(
                d_in = embed_dim, 
                d_hid = embed_dim * 2, 
                dropout = 0.1)
            self.feedforwards_1.append(subFF_1)
            subFF_2 = FeedForward(
                d_in = embed_dim, 
                d_hid = embed_dim * 2, 
                dropout = 0.1)
            self.feedforwards_2.append(subFF_2)

            # Layer Norm
            self.layer_norms_1.append(torch.nn.LayerNorm(embed_dim))
            self.layer_norms_2.append(torch.nn.LayerNorm(embed_dim))

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
        
        for i, layer in enumerate(self.layers):
            # q1, q2, key, mask, v1, v2
            # 对目标特征att_feats进行增强，得到att_feats_
            # 同时基于att_feats_获取gv_feat_
            gv_feat_, att_feats_ = layer(gv_feat, att_feats, att_feats, att_mask, att_feats, att_feats)
            # FeedForward
            att_feats_ = self.feedforwards_1[i](att_feats_)
            gv_feat_   = self.feedforwards_2[i](gv_feat_)
            # att_feats 残差连接 + LayerNorm
            att_feats = self.layer_norms_1[i](att_feats + att_feats_)
            # gv_feat 残差连接 + LayerNorm
            gv_feat   = self.layer_norms_2[i](gv_feat + gv_feat_)

        return gv_feat, att_feats
"""


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

"""
class tmp_Layer(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout):
        super(tmp_Layer, self).__init__()
        self.encoder_attn = MultiHeadAttentionEnc(
            embed_dim = embed_dim, 
            att_type = att_type, 
            att_heads = att_heads, 
            att_mid_dim = att_mid_dim, 
            att_mid_drop = att_mid_drop,
            dropout = dropout)
        
    def forward(self, query, key, mask, value1, value2, precompute=False):
        return self.encoder_attn(query, key, mask, value1, value2, precompute)
"""

    
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
        self.layer_norm = torch.nn.LayerNorm(1024)

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
