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
"""

class FFN(nn.Module):
    def __init__(self, embed_dim):
        super(FFN, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.linear2 = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


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

        # values2 作为真正的value，同时用于空间注意力和通道注意力
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        sequential.append(nn.CELU(1.3))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)
        
        # linear 合并
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        # sequential.append(nn.CELU(1.3))
        # sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.linear_merge = nn.Sequential(*sequential)

        # self.attn_net = SCAttEnc(att_mid_dim, att_mid_drop)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, query, key, mask, value1, value2, precompute=False):
        """
        输入数据：
        query1: [B, 1024] / [B, M, 1024]
        query2: [B, M, 1024]
        key:    [B, M, 1024]
        mask:   [B, M]
        value1: [B, 1024] 无用，不进行操作
        value2: [B, M, 1024]
        """
        # 输入数据全连接层
        batch_size = query.size()[0]
        
        if len(query.size()) == 2:
            # [B, 1024] --> [B, 1, 1024]
            query = query.unsqueeze(1)
            
        # 如果使用Group Normalization，则必须将输入进行维度变换
        # [B, M, 1024] --> [-1, 1024]
        query  = query.view(-1, query.size()[-1])
        key    = key.view(-1, key.size()[-1])
        value2 = value2.view(-1, value2.size()[-1])
            
        # 投影变换
        # [B, M, 1024]  -->  [B, 8, M, 128]
        q = self.in_proj_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.in_proj_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = self.in_proj_v2(value2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 调用注意力机制核心操作函数
        # 将attn_map的计算置于attn_net中进行
        attn = self.attn_net(q, k, mask, None, v2)

        # 将输出从多头维度上恢复为正确维度
        # [B, 8, M, 128] --> [B, M, 1024]
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # linear 合并
        attn = self.linear_merge(attn)
        
        # 如果长度为1，则删除该维度
        if attn.size()[1] == 1:
            attn = attn.squeeze(1)
        
        return attn
    
    def attn_net(self, q, k, mask, v1, v2):
        # [B, 8, M, M]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask==0, -1e9)

        att_map = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            att_map = self.dropout(att_map)

        return torch.matmul(att_map, v2)

    
# 完整的self-attentino模块，包含残差连接和FeedForward Layer
class SelfAttentionEnc(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout):
        super(SelfAttentionEnc, self).__init__()
        self.mh_att = MultiHeadAttentionEnc(
            embed_dim = embed_dim, 
            att_type = att_type, 
            att_heads = att_heads, 
            att_mid_dim = att_mid_dim, 
            att_mid_drop = att_mid_drop,
            dropout = dropout)
        self.ffn = FFN(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, q, k, mask, v1, v2):
        attn = self.norm1(q + self.dropout1(
            self.mh_att(q, k, mask, v1, v2)
        ))

        attn = self.norm2(attn + self.dropout2(
            self.ffn(attn)
        ))
        return attn

# 结构一
class VP_Refine_Module(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(VP_Refine_Module, self).__init__()
        self.layer_num = layer_num
        
        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        self.bifeat_emb = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([])
        for _ in range(layer_num):
            # 图像全局特征预获取
            en_sublayer = SelfAttentionEnc(
                embed_dim = embed_dim, 
                att_type = att_type, 
                att_heads = att_heads, 
                att_mid_dim = att_mid_dim, 
                att_mid_drop = att_mid_drop,
                dropout = dropout)
            self.encoder_layers.append(en_sublayer)
            
            # 图像特征增强
            de_sublayer = SelfAttentionEnc(
                embed_dim = embed_dim, 
                att_type = att_type, 
                att_heads = att_heads, 
                att_mid_dim = att_mid_dim, 
                att_mid_drop = att_mid_drop,
                dropout = dropout)
            self.decoder_layers.append(de_sublayer)
            
            # 图像特征增强部分，接受全局特征的指导
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
        
        # Encoder-Decoder 结构
        # 1、先对gv_feat进行增强（即Encoder）
        feat_arr = [gv_feat]
        for i, layer in enumerate(self.encoder_layers):
            # q, key, mask, v1, v2
            gv_feat = layer(gv_feat, att_feats, att_mask, gv_feat, att_feats)
            feat_arr.append(gv_feat)

        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        
        # 2、然后进行att_feats增强（即Decoder）
        for i, layer in enumerate(self.decoder_layers):
            # 先对att_feats进行自注意力增强
            att_feats = layer(att_feats, att_feats, att_mask, att_feats, att_feats)
            # 然后接受gv_feat的指导，进行进一步增强
            att_feats_cat = torch.cat([gv_feat.unsqueeze(1).expand_as(att_feats), att_feats], dim=-1)
            
            # att_feats 残差连接
            att_feats = self.bifeat_emb[i](att_feats_cat) + att_feats
            att_feats = self.layer_norms[i](att_feats)
        
        return gv_feat, att_feats
