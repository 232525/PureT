import torch
import torch.nn as nn
import torch.nn.functional as F

from .vp_att import FeedForward
from .Sparsemax import Sparsemax

"""
该文件下包含三个类的定义与实现：
1）SCAttDec - 核心的注意力实现（通过qkv，计算注意力权重及加权求和）
2）MultiHeadAttentionDec - 多头注意力机制实现
3）VP_Attention_Module - 注意力层定义
关系：3）-调用-> 2）-调用-> 1）
"""

############################################
# 用于Decoder部分，目标特征的注意力机制
############################################
# """
# XLAN SCAtt 模块
class SCAttDec(nn.Module):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAttDec, self).__init__()
        self.attention_basic = nn.Sequential(
            nn.Linear(mid_dims[0], mid_dims[1]), 
            nn.ReLU(), 
            nn.Dropout(mid_dropout)
        )
        
        self.attention_spatial = nn.Linear(mid_dims[-2], 1)
        self.attention_channel = nn.Linear(mid_dims[-2], mid_dims[-1])
        
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
            att_map_pool = att_map.mean(-2)
        
        # Spatial Attention
        alpha_spatial = self.attention_spatial(att_map)
        alpha_spatial = alpha_spatial.squeeze(-1)
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)
        
        if len(alpha_spatial.shape) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            value2 = torch.matmul(alpha_spatial, value2)
        else:
            value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)

        # Channel Attention
        alpha_channel = self.attention_channel(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)
        
        attn = value1 * value2 * alpha_channel
        
        return attn
    

# 多头注意力模块
class MultiHeadAttentionDec(nn.Module):
    # 默认参数：
    # embed_dim=1024, att_heads=8, att_mid_dim=[128, 64, 128], att_mid_drop=0.1, dropout=0.5
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout):
        super(MultiHeadAttentionDec, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        output_dim = embed_dim

        # query 用于空间注意力的query
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

        # values2 用于真正的value
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        sequential.append(nn.CELU(1.3))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)
        
        # 新增Linear
        """
        sequential = []
        sequential.append(nn.Linear(output_dim, embed_dim))
        sequential.append(nn.CELU(1.3))
        self.fc = nn.Sequential(*sequential)
        """

        self.attn_net = SCAttDec(att_mid_dim, att_mid_drop)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    # query -- batch_size * qdim
    # value -- batch_size * att_num * vdim
    def forward(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = key
            v2 = value2
        
        # 将attn_map的计算置于attn_net中进行
        attn = self.attn_net(q, k, mask, v1, v2)
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        
        # attn = self.fc(attn)
        if self.dropout is not None:
            attn = self.dropout(attn)
        return attn
    
    # 预计算
    def precompute(self, key, value2):
        batch_size = value2.size()[0]
        key = key.view(-1, key.size()[-1])
        value2 = value2.view(-1, value2.size()[-1])

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v2
    
# """
# 用于LSTM每一步的注意力模块（用于Decoder）
# 与XLAN结构相同
class VP_Attention_Module(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layer_num):
            sublayer = MultiHeadAttentionDec(
                embed_dim = embed_dim, 
                att_type = att_type, 
                att_heads = att_heads, 
                att_mid_dim = att_mid_dim, 
                att_mid_drop = att_mid_drop,
                dropout = dropout)
            self.layers.append(sublayer)
        
        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(1024)
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        keys = []
        value2s = []
        for layer in self.layers:
            k, v = layer.precompute(key, value2)
            keys.append(k)
            value2s.append(v)
        return torch.cat(keys, dim=-1), torch.cat(value2s, dim=-1)

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # p_att_feats: [B, 8, M, 256]
        # 当使用p_att_feats时，att_feats不参与计算
        if precompute == True:
            dim = p_att_feats.size()[-1]
            keys = p_att_feats.narrow(-1, 0, dim // 2)
            value2s = p_att_feats.narrow(-1, dim // 2, dim // 2)
            dim = keys.size()[-1] // len(self.layers)
        
        feat_arr = [h_state]
        for i, layer in enumerate(self.layers):
            key = keys.narrow(-1, i * dim, dim) if precompute else att_feats
            value2 = value2s.narrow(-1, i * dim, dim) if precompute else att_feats
            
            # h_state作为query
            h_state_ = layer(h_state, key, att_mask, h_state, value2, precompute)
            # 残差连接
            # h_state = h_state + h_state_
            h_state = h_state_
            feat_arr.append(h_state)

        att = torch.cat(feat_arr, dim=-1)   # [B, 1024 * 2]
        att = self.proj(att)                # [B, 1024 * 2] --> [B, 1024]
        att = self.layer_norm(att)
        return att, None
    

"""
# 非多头注意力机制
class VP_Attention_Module_init(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_init, self).__init__()
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.alpha_linear1 = nn.Linear(embed_dim, 512)
        self.alpha_linear2 = nn.Linear(512, 1)
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        p_key = self.k_linear(key)
        p_value = self.v_linear(value2)
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
            
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        alpha = p_query * p_key           # [B, M, 1024]
        alpha = self.alpha_linear1(alpha) # [B, M, 512]
        alpha = self.alpha_linear2(alpha) # [B, M, 1]
        alpha = alpha.squeeze(-1)         # [B, M]
        
        # 计算权重
        if att_mask is not None:
            alpha = alpha.masked_fill(att_mask == 0, -1e9)
        alpha = F.softmax(alpha, dim=-1)  # [B, M]
        
        att = torch.matmul(alpha.unsqueeze(1), p_value).squeeze(1) # [B, 1024]
        
        return att, None
"""

# 模拟多头注意力
class VP_Attention_Module_init(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_init, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        # 通道注意力
        self.c_linear = nn.Linear(embed_dim, embed_dim)
        
        self.alpha_linear1 = nn.Linear(self.head_dim, self.head_dim // 2)
        self.alpha_linear2 = nn.Linear(self.head_dim // 2, 1)
        # 通道注意力
        self.alpha_linear3 = nn.Linear(self.head_dim // 2, self.head_dim)
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        p_key = self.k_linear(key)
        p_value = self.v_linear(value2)
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
            
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        # 转换为多头模式 [B, H, M, 1024 // H]
        batch_size = h_state.size()[0]
        p_query = p_query.view(batch_size, -1, self.num_heads, self.head_dim)
        p_key = p_key.view(batch_size, -1, self.num_heads, self.head_dim)
        p_value = p_value.view(batch_size, -1, self.num_heads, self.head_dim)
        
        p_query = p_query.transpose(1, 2)  # [B, H, 1, 1024 // H]
        p_key = p_key.transpose(1, 2)      # [B, H, M, 1024 // H]
        p_value = p_value.transpose(1, 2)  # [B, H, M, 1024 // H]
        
        _alpha = p_query * p_key           # [B, H, M, 1024 // H]
        _alpha = self.alpha_linear1(_alpha) # [B, H, M, (1024 // H) // 2]
        # >>> 空间注意力
        alpha = self.alpha_linear2(_alpha) # [B, H, M, 1]
        alpha = alpha.squeeze(-1)         # [B, H, M]
        
        # 计算权重
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            alpha = alpha.masked_fill(att_mask == 0, -1e9)
        alpha = F.softmax(alpha, dim=-1)  # [B, H, M]
        
        att = torch.matmul(alpha.unsqueeze(-2), p_value) # [B, H, 1, 1024 // H]
        att = att.squeeze(-2) # [B, H, 1024 // H]
        
        # >>> 通道注意力
        if att_mask is not None:
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            c_alpha = torch.sum(_alpha * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            c_alpha = _alpha.mean(-2)
            
        # 通道注意力权重
        c_alpha = self.alpha_linear3(c_alpha) # [B, H, 1024 // H]
        c_alpha = torch.sigmoid(c_alpha)      # [B, H, 1024 // H]
        
        # 通道注意力value
        p_channel = self.c_linear(h_state)  # [B, 1024]
        # 转换为多头模式，[B, H, 1024 // H]
        p_channel = p_channel.view(batch_size, self.num_heads, self.head_dim)
        
        att = c_alpha * p_channel * att  # [B, H, 1024 // H]
        
        # 维度还原
        att = att.view(batch_size, self.embed_dim) # [B, 1024]
        
        return att, None
    
# 传统add注意力
class VP_Attention_Module_add(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_add, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim // 2)
        self.k_linear = nn.Linear(embed_dim, embed_dim // 2)
        
        self.alpha_linear1 = nn.Linear(embed_dim // 2, 1)
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        p_key = self.k_linear(key)
        
        return p_key, p_key

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 512 * 2]
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 512]
            
        p_query = self.q_linear(h_state)  # [B, 512]
        p_query = p_query.unsqueeze(1)    # [B, 1, 512]
        
        alpha = torch.tanh(p_query + p_key)           # [B, M, 512]
        alpha = self.alpha_linear1(alpha) # [B, M, 1]
        alpha = alpha.squeeze(-1)         # [B, M]
        
        # 计算权重
        if att_mask is not None:
            alpha = alpha.masked_fill(att_mask == 0, -1e9)
        alpha = F.softmax(alpha, dim=-1)  # [B, M]
        
        att = torch.matmul(alpha.unsqueeze(-2), att_feats) # [B, 1, 1024]
        att = att.squeeze(-2) # [B, 1024]
        
        return att, None
    
# SCA CNN
class VP_Attention_Module_add_sca(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_add_sca, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 空间注意力
        self.q_linear = nn.Linear(embed_dim, embed_dim // 2)
        self.k_linear = nn.Linear(embed_dim, embed_dim // 2)
        
        self.alpha_linear1 = nn.Linear(embed_dim // 2, 1)
        
        # 通道注意力
        self.qc_linear = nn.Linear(embed_dim, embed_dim // 2)
        self.kc_linear = nn.Linear(1, embed_dim // 2)
        
        self.alpha_linear2 = nn.Linear(embed_dim // 2, 1)
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        _ = torch.zeros(1, device='cuda')
        
        return _, _

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 512 * 2]
        """
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 512]
        """
        
        # 通道注意力
        p_query_c = self.qc_linear(h_state)  # [B, 512]
        p_query_c = p_query_c.unsqueeze(1)     # [B, 1, 512]
        
        if att_mask is not None:
            att_feats_mean = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)  # [B, 1024]
            att_feats_mean = att_feats_mean.unsqueeze(-1) # [B, 1024, 1]
           
        p_key_c = self.kc_linear(att_feats_mean)  # [B, 1024, 512]
        
        c_alpha = torch.tanh(p_query_c + p_key_c) # [B, 1024, 512]
        c_alpha = self.alpha_linear2(c_alpha)     # [B, 1024, 1]
        c_alpha = c_alpha.squeeze(-1)             # [B, 1024]
        
        c_alpha = F.softmax(c_alpha, dim=-1)      # [B, 1024]
        c_alpha = c_alpha.unsqueeze(1)            # [B, 1, 1024]
        
        att_feats = att_feats * c_alpha           # [B, M, 1024]
        
        # 空间注意力
        p_query = self.q_linear(h_state)  # [B, 512]
        p_query = p_query.unsqueeze(1)    # [B, 1, 512]
        
        p_key = self.k_linear(att_feats)  # [B, M, 512]
        
        alpha = torch.tanh(p_query + p_key)           # [B, M, 512]
        alpha = self.alpha_linear1(alpha) # [B, M, 1]
        alpha = alpha.squeeze(-1)         # [B, M]
        
        # 计算权重
        if att_mask is not None:
            alpha = alpha.masked_fill(att_mask == 0, -1e9)
        alpha = F.softmax(alpha, dim=-1)  # [B, M]
        
        att = torch.matmul(alpha.unsqueeze(-2), att_feats) # [B, 1, 1024]
        att = att.squeeze(-2) # [B, 1024]
        
        return att, None
    
# self attention
class VP_Attention_Module_self(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_self, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        p_key = self.k_linear(key)
        p_value = self.v_linear(value2)
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
            
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        p_value = p_value.view(b, -1, self.num_heads, self.head_dim)
        p_value = p_value.transpose(1, 2) # [B, 8, M, 128]
        
        # 自注意力
        alpha = torch.matmul(p_query, p_key.transpose(-1, -2)) # [B, 8, 1, M]
        alpha = alpha * self.scaling
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, M]
            alpha = alpha.masked_fill(att_mask == 0, -1e9)
        alpha = F.softmax(alpha, dim=-1) # [B, 8, 1, M]
        att = torch.matmul(alpha, p_value) # [B, 8, 1, 128]
        
        att = att.squeeze(-2).view(-1, self.num_heads * self.head_dim)  # [B, 1024]
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
class VP_Attention_Module_xlan_simple(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            nn.ReLU(), 
            nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        
        # Dropout
        # self.dropout = None
        self.dropout = nn.Dropout(dropout)
        
        # 和h_state再次concat
        # self.tail_proj = None
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        
        att = att_spatial * att_channel
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 极简版本，只保留核心操作，去除CELU激活函数、GroupNorm、Dropout及tail_proj
# 去除通道注意力
class VP_Attention_Module_xlan_simple_ablation(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_ablation, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        """
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        """
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            # nn.ReLU(), 
            # nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        """
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        """
        
        # Dropout
        self.dropout = None
        """
        self.dropout = nn.Dropout(dropout)
        """
        
        # 和h_state再次concat
        self.tail_proj = None
        """
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        """
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        """
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        """
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        """
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        """
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        """
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
        """
        # For Test
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        """
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        
        att = att_spatial * att_channel
        """
        att = att_spatial  # For Test
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 极简版本
# 添加进来CELU激活函数和GroupNorm
# 及注意力linear层部分的ReLU激活函数和Dropout
class VP_Attention_Module_xlan_simple_ablation_w_norm(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_ablation_w_norm, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        """
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        """
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            nn.ReLU(), 
            nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        """
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        """
        
        # Dropout
        # self.dropout = None
        # """
        self.dropout = nn.Dropout(dropout)
        # """
        
        # 和h_state再次concat
        self.tail_proj = None
        """
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        """
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        """
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        """
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        """
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        """
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        """
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
        """
        # For Test
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        """
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        
        att = att_spatial * att_channel
        """
        att = att_spatial  # For Test
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 极简版本，保留空间注意力核心操作，及tail_proj
# 去除CELU激活函数、GroupNorm、Dropout
# 去除通道注意力
class VP_Attention_Module_xlan_simple_ablation_w_tail(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_ablation_w_tail, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        """
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        """
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            # nn.ReLU(), 
            # nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        """
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        """
        
        # Dropout
        self.dropout = None
        """
        self.dropout = nn.Dropout(dropout)
        """
        
        # 和h_state再次concat
        # self.tail_proj = None
        # """
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        # """
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        """
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        """
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        """
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        """
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        """
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
        """
        # For Test
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        """
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        
        att = att_spatial * att_channel
        """
        att = att_spatial  # For Test
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 极简版本，保留空间注意力核心操作，及通道注意力
# 去除CELU激活函数、GroupNorm、Dropout
# 去除tail_proj
class VP_Attention_Module_xlan_simple_ablation_w_channel(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_ablation_w_channel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        # """
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        # """
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            # nn.ReLU(), 
            # nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        # """
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        # """
        
        # Dropout
        self.dropout = None
        """
        self.dropout = nn.Dropout(dropout)
        """
        
        # 和h_state再次concat
        self.tail_proj = None
        """
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        """
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        # """
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        # """
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        # """
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        # """
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        # """
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
        # """
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        # """
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        
        att = att_spatial * att_channel
        # """
        
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 极简版本，保留空间注意力核心操作，及CELU激活函数、GroupNorm、Dropout及tail_proj
# 去除通道注意力
class VP_Attention_Module_xlan_simple_ablation_w_norm_tail(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_ablation_w_norm_tail, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        """
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        """
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            nn.ReLU(), 
            nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        """
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        """
        
        # Dropout
        # self.dropout = None
        # """
        self.dropout = nn.Dropout(dropout)
        # """
        
        # 和h_state再次concat
        # self.tail_proj = None
        # """
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        # """
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        """
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        """
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        """
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        """
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        """
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
        """
        # For Test
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        """
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        
        att = att_spatial * att_channel
        """
        att = att_spatial  # For Test
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 极简版本，保留空间注意力核心操作，及通道注意力
# 去除CELU激活函数、GroupNorm、Dropout
# 去除tail_proj
class VP_Attention_Module_xlan_simple_ablation_w_channel_norm(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_ablation_w_channel_norm, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        # """
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        # """
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            nn.ReLU(), 
            nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        # """
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        # """
        
        # Dropout
        # self.dropout = None
        # """
        self.dropout = nn.Dropout(dropout)
        # """
        
        # 和h_state再次concat
        self.tail_proj = None
        """
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        """
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        # """
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        # """
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        # """
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        # """
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        # """
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
        # """
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        # """
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        
        att = att_spatial * att_channel
        # """
        
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 极简版本，保留空间注意力核心操作，及通道注意力
# 去除CELU激活函数、GroupNorm、Dropout
# 去除tail_proj
class VP_Attention_Module_xlan_simple_ablation_w_channel_tail(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_ablation_w_channel_tail, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        # """
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        # """
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            # nn.ReLU(), 
            # nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        # """
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        # """
        
        # Dropout
        self.dropout = None
        """
        self.dropout = nn.Dropout(dropout)
        """
        
        # 和h_state再次concat
        # self.tail_proj = None
        # """
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        # """
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        # """
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        # """
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        # """
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        # """
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        # """
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
        # """
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        # """
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        
        att = att_spatial * att_channel
        # """
        
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None


# LAMA注意力，简化版本，把所有操作都在forward中实现
class VP_Attention_Module_lama_simple(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_lama_simple, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, self.num_heads),
            nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, self.num_heads),
            nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力线性层
        self.attention_linear = nn.Sequential(
            nn.Linear(self.num_heads * self.embed_dim, self.embed_dim), 
            nn.ReLU()
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 和h_state再次concat
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        p_key = self.k_linear(key)  # [B, M, 8]
        # p_key = p_key.transpose(-1, -2)   # [B, 8, M]
        
        batch_size = value2.size()[0]
        p_value = self.v_linear(value2.view(-1, self.embed_dim))
        p_value = p_value.view(batch_size, -1, self.embed_dim) #[B, M, D]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        batch_size = h_state.size()[0]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 8]
        p_query = p_query.unsqueeze(-1)    # [B, 8, 1]
        
        """
        p_key = self.k_linear(att_feats)  # [B, M, 8]
        p_key = p_key.transpose(-1, -2)   # [B, 8, M]
        
        p_value = self.v_linear(att_feats.view(-1, self.embed_dim))
        p_value = p_value.view(batch_size, -1, self.embed_dim) #[B, M, D]
        """
        if precompute == True:
            p_key = p_att_feats.narrow(-1, 0, self.num_heads)          # [B, M, 8]
            p_key = p_key.transpose(-1, -2)  # [B, 8, M]
            p_value = p_att_feats.narrow(-1, self.num_heads, self.embed_dim) # [B, M, 1024]
            
        
        alignment = torch.tanh(p_query * p_key)  # [B, 8, M]
        alignment = alignment / torch.norm(alignment, dim=1, keepdim=True)
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)  # [B, 1, M]
            alignment = alignment.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alignment, dim=-1) # [B, 8, M]
        
        att = torch.matmul(alpha_spatial, p_value) # [B, 8, D]
        att = att.view(batch_size, -1)      # [B, 8 * D]
        att = self.attention_linear(att)    # [B, D]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
class VP_Attention_Module_xlan_simple_addatt(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_addatt, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            nn.ReLU(), 
            nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        
        # Dropout
        # self.dropout = None
        self.dropout = nn.Dropout(dropout)
        
        # 和h_state再次concat
        # self.tail_proj = None
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力 替换为 add操作
        # att_map = p_query * p_key # [B, 8, M, 128]
        att_map = torch.tanh(p_query + p_key) # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        
        att = att_spatial * att_channel
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 将激活函数改为ReLU、标准化改为LayerNorm
class VP_Attention_Module_xlan_simple_relu_ln(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_relu_ln, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
            # nn.CELU(1.3),
            # nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            nn.ReLU(),
            # nn.Dropout(att_mid_drop)
            nn.LayerNorm(att_mid_dim[1])
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        
        # Dropout
        # self.dropout = None
        self.dropout = nn.Dropout(dropout)
        
        # 和h_state再次concat
        # self.tail_proj = None
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        
        att = att_spatial * att_channel
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 修改通道注意力机制部分，直接导入h_state全部信息，不进行sigmoid门控加权
class VP_Attention_Module_xlan_simple_wo_sigmoid(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_wo_sigmoid, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            nn.ReLU(), 
            nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        # self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        
        # Dropout
        # self.dropout = None
        self.dropout = nn.Dropout(dropout)
        
        # 和h_state再次concat
        self.tail_proj = None
        """
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        """
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        """
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        """
        # 直接把p_value1融入到att_spatial中，不进行sigmoid门控
        att_channel = p_value1.squeeze(-2)
        
        att = att_spatial * att_channel
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 先将grid feature与每一个时间步的h_state进行融合，然后进行注意力计算
class VP_Attention_Module_xlan_simple_fusion_h(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_fusion_h, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            nn.ReLU(), 
            nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        
        # Dropout
        # self.dropout = None
        self.dropout = nn.Dropout(dropout)
        
        # 和h_state再次concat
        # self.tail_proj = None
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # 与论文中Figure 2保持一致
        # xlan注意力，用于grid feature与h_state的融合
        fusion_att_feats = p_value2 * p_value1 # [B, 8, M, 128]
        
        # xlan注意力，用于相似度计算
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]        
        
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att = torch.matmul(alpha_spatial.unsqueeze(-2), fusion_att_feats).squeeze(-2) # [B, 8, 128]
        
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        
        att = att * alpha_channel                  # [B, 8, 128]
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 先将grid feature与每一个时间步的h_state进行融合，然后进行注意力计算
class VP_Attention_Module_xlan_simple_fusion_h_1(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_fusion_h_1, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            nn.ReLU(), 
            nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        
        # Dropout
        # self.dropout = None
        self.dropout = nn.Dropout(dropout)
        
        # 和h_state再次concat
        # self.tail_proj = None
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        # 线性变换，预计算
        p_key = self.k_linear(key)        
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
        
        # 准备输入query value1
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力，用于grid feature与h_state的融合
        fusion_att_feats = torch.tanh(p_value2 + p_value1) # [B, 8, M, 128]
        # fusion_att_feats = p_value2 * p_value1 # [B, 8, M, 128]
        # fusion_att_feats = self.k_linear(torch.cat([p_value2, p_value1.repeat(1, 1, p_value2.size()[-2], 1)], -1))  # [B, 8, M, 128]
        
        # xlan注意力，用于相似度计算
        # att_map = torch.tanh(p_query + p_key) # [B, 8, M, 128]
        att_map = p_query * p_key # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att = torch.matmul(alpha_spatial.unsqueeze(-2), fusion_att_feats).squeeze(-2) # [B, 8, 128]
        
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        
        att = att * alpha_channel                  # [B, 8, 128]
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None
    
# xlan注意力，简化版本，把所有操作都在forward中实现
# 与lstm输入替换为上一步的attend feature结构组合使用
# 1、保留空间注意力、通道注意力和激活函数+GroupNorm，去除tail_proj
# 2、通道注意力去除sigmoid门控机制
# 3、h_state和att_feats的交互计算使用add替换Hadamard积，包括query和key的相似度计算，及空间注意力和通道注意力结构的融合
class VP_Attention_Module_xlan_simple_attinlstm(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Attention_Module_xlan_simple_attinlstm, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        
        # 线性变换
        self.q_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.CELU(1.3),
            nn.GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            nn.Linear(att_mid_dim[0], att_mid_dim[1]), 
            nn.ReLU(), 
            nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = nn.Linear(att_mid_dim[-2], 1)
        # self.attention_channel = nn.Linear(att_mid_dim[-2], att_mid_dim[-1])
        
        # Dropout
        # self.dropout = None
        self.dropout = nn.Dropout(dropout)
        
        # 和h_state再次concat
        self.tail_proj = None
        """
        self.tail_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        """
                
    # 用于注意力模块的预计算
    def precompute(self, key, value2):
        # 为了和多头注意力函数调用兼容，此处的key，value2输入均为att_feats
        b = key.size()[0]
        # 维度变换，GroupNorm需要
        key = key.view(-1, key.size()[-1])          # [-1, 1024]
        value2 = value2.view(-1, value2.size()[-1]) # [-1, 1024]
        
        p_key = self.k_linear(key)
        p_value = self.v2_linear(value2)
        
        # 维度还原
        p_key = p_key.view(b, -1, self.embed_dim)     # [B, M, 1024]
        p_value = p_value.view(b, -1, self.embed_dim) # [B, M, 1024]
        
        return p_key, p_value

    def forward(self, h_state, att_feats, att_mask, p_att_feats=None, precompute=False):
        # 输入
        # h_state:     [B, 1024]
        # att_feats:   [B, M, 1024]
        # att_mask:    [B, M]
        # p_att_feats: [B, M, 1024 * 2]
        
        # 线性变换
        # 准备输入key value2
        if precompute == True:
            dim = p_att_feats.size()[-1]
            p_key = p_att_feats.narrow(-1, 0, dim // 2)          # [B, M, 1024]
            p_value2 = p_att_feats.narrow(-1, dim // 2, dim // 2) # [B, M, 1024]
          
        # 准备输入query value1，均来自于h_state
        p_query = self.q_linear(h_state)  # [B, 1024]
        p_query = p_query.unsqueeze(1)    # [B, 1, 1024]
        
        p_value1 = self.v1_linear(h_state) # [B, 1024]
        p_value1 = p_value1.unsqueeze(1)   # [B, 1, 1024]
        
        # 转换为多头模式
        b = h_state.size()[0]
        p_query = p_query.view(b, -1, self.num_heads, self.head_dim)
        p_query = p_query.transpose(1, 2) # [B, 8, 1, 128]
        p_key = p_key.view(b, -1, self.num_heads, self.head_dim)
        p_key = p_key.transpose(1, 2)     # [B, 8, M, 128]
        p_value1 = p_value1.view(b, -1, self.num_heads, self.head_dim)
        p_value1 = p_value1.transpose(1, 2)     # [B, 8, 1, 128]
        p_value2 = p_value2.view(b, -1, self.num_heads, self.head_dim)
        p_value2 = p_value2.transpose(1, 2) # [B, 8, M, 128]
        
        # xlan注意力
        # att_map = p_query * p_key # [B, 8, M, 128]
        att_map = torch.tanh(p_query + p_key) # [B, 8, M, 128]
        att_map = self.attention_basic(att_map) # [B, 8, M, 64]
        
        """
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            att_mask_ext = att_mask.unsqueeze(-1) # [B, 1, M, 1]
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, dim=-2)  # [B, 8, 64]
        else:
            att_map_pool = att_map.mean(-2)
        """
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1) # [B, 1, M]
            
        # 空间注意力
        alpha_spatial = self.attention_spatial(att_map) # [B, 8, M, 1]
        alpha_spatial = alpha_spatial.squeeze(-1) # [B, 8, M]
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask==0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1) # [B, 8, M]
        
        att_spatial = torch.matmul(alpha_spatial.unsqueeze(-2), p_value2).squeeze(-2) # [B, 8, 128]
        
        # 通道注意力
        """
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        att_channel = p_value1.squeeze(-2) * alpha_channel       # [B, 8, 128]
        """
        # 通道注意力直接来自于p_value1，不进行sigmoid门控
        att_channel = p_value1.squeeze(-2)
        
        # 融合空间注意力结果+通道注意力结果
        # att = att_spatial * att_channel
        att = torch.tanh(att_spatial + att_channel)
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
          
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
        
        return att, None