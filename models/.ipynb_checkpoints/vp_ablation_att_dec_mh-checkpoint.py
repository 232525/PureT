import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.utils as utils
from lib.config import cfg

from .vp_att import FeedForward
from .Sparsemax import Sparsemax

"""
该文件下包含三个类的定义与实现：
1）SCAttDec - 核心的注意力实现（通过qkv，计算注意力权重及加权求和）
2）MultiHeadAttentionDec - 多头注意力机制实现
3）VP_Attention_Module - 注意力层定义
关系：3）-调用-> 2）-调用-> 1）
"""

import math

class MH_Linear(nn.Module):
    def __init__(self, in_channel=128, out_channel=64, heads=8):
        super(MH_Linear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(heads, in_channel, out_channel))
        self.bias   = nn.Parameter(torch.Tensor(heads, out_channel))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x): 
        if len(x.size()) == 3:
            # (B, H, I) --> (B, H, O)
            x = torch.einsum('bhi,hio->bho', (x, self.weight))
            x = x + self.bias.unsqueeze(0)
        else:
            # (B, H, M, I) --> (B, H, M, O)
            x = torch.einsum('bhmi,hio->bhmo', (x, self.weight))
            x = x + self.bias.unsqueeze(0).unsqueeze(-2)
        return x
    
class GroupNorm(nn.Module):
    def __init__(self, num_groups=8, num_features=1024, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C)
        return x * self.weight + self.bias

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
            nn.ReLU(inplace=True), 
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
        # sequential.append(nn.CELU(1.3, inplace=True))
        sequential.append(utils.activation(cfg.MODEL.BILINEAR.ACT))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q = nn.Sequential(*sequential)

        # keys
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        # sequential.append(nn.CELU(1.3, inplace=True))
        sequential.append(utils.activation(cfg.MODEL.BILINEAR.ACT))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_k = nn.Sequential(*sequential)

        # values1 用于通道注意力的query
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        # sequential.append(nn.CELU(1.3, inplace=True))
        sequential.append(utils.activation(cfg.MODEL.BILINEAR.ACT))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v1 = nn.Sequential(*sequential)

        # values2 用于真正的value
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        # sequential.append(nn.CELU(1.3, inplace=True))
        sequential.append(utils.activation(cfg.MODEL.BILINEAR.ACT))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)
        
        # 新增Linear
        """
        sequential = []
        sequential.append(nn.Linear(output_dim, embed_dim))
        # sequential.append(nn.CELU(1.3))
        sequential.append(utils.activation(cfg.MODEL.BILINEAR.ACT))
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
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        
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
            # nn.CELU(1.3, inplace=True),
            utils.activation(cfg.MODEL.BILINEAR.ACT),
            # nn.GroupNorm(self.num_heads, embed_dim)
            GroupNorm(self.num_heads, embed_dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3, inplace=True),
            utils.activation(cfg.MODEL.BILINEAR.ACT),
            # nn.GroupNorm(self.num_heads, embed_dim)
            GroupNorm(self.num_heads, embed_dim)
        )
        self.v1_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3, inplace=True),
            utils.activation(cfg.MODEL.BILINEAR.ACT),
            # nn.GroupNorm(self.num_heads, embed_dim)
            GroupNorm(self.num_heads, embed_dim)
        )
        self.v2_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.CELU(1.3, inplace=True),
            utils.activation(cfg.MODEL.BILINEAR.ACT),
            # nn.GroupNorm(self.num_heads, embed_dim)
            GroupNorm(self.num_heads, embed_dim)
        )
        
        # 注意力机制
        self.attention_basic = nn.Sequential(
            MH_Linear(att_mid_dim[0], att_mid_dim[1], 8), 
            nn.ReLU(inplace=True), 
            nn.Dropout(att_mid_drop)
        )
        self.attention_spatial = MH_Linear(att_mid_dim[-2], 1, 8)
        self.attention_channel = MH_Linear(att_mid_dim[-2], att_mid_dim[-1], 8)
        
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

        # 空间注意力加权
        att = torch.matmul(alpha_spatial.unsqueeze(-2), fusion_att_feats).squeeze(-2) # [B, 8, 128]
        
        # 通道注意力
        alpha_channel = self.attention_channel(att_map_pool) # [B, 8, 128]
        alpha_channel = torch.sigmoid(alpha_channel) # [B, 8, 128]
        
        # 通道注意力加权
        att = att * alpha_channel                  # [B, 8, 128]
        
        # 维度还原
        att = att.view(-1, self.num_heads * self.head_dim) # [B, 1024]
        
        if self.dropout is not None:
            att = self.dropout(att)
         
        # 附加处理
        if self.tail_proj is not None:
            att = self.tail_proj(torch.cat([h_state, att], dim=-1))
           
        return att, None