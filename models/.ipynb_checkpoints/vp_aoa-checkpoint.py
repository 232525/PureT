import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_model import BasicModel
from models.att_basic_model import AttBasicModel
# from layers.attention import Attention
from lib.config import cfg
import lib.utils as utils
from models.vp_att_dec import VP_Attention_Module


class VP_AoA(AttBasicModel):
    def __init__(self, vp_en=False, vp_de=False):
        super(VP_AoA, self).__init__()
        self.vp_en, self.vp_de = vp_en, vp_de
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE * 2 + cfg.MODEL.WORD_EMBED_DIM
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)

        # (1) Visual Persistence in Encoder
        # self.vp_en为False时，特征增强模块重载，使用AoA中的特征增强（也不完全一致）
        # self.vp_en为True时，使用AttBasicModel中的特征增强
        if not self.vp_en:
            del self.encoder_layers
            self.encoder_layers = AoA_Refine_Module(
                embed_dim=cfg.MODEL.BILINEAR.DIM,
                att_type=cfg.MODEL.BILINEAR.ATTTYPE,
                att_heads=cfg.MODEL.BILINEAR.HEAD,
                att_mid_dim=cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM,
                att_mid_drop=cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT,
                dropout=cfg.MODEL.BILINEAR.ENCODE_DROPOUT,
                layer_num=cfg.MODEL.BILINEAR.ENCODE_LAYERS
            )

        # 注意力机制
        # self.att = Attention()
        self.attention = VP_Attention_Module(
            embed_dim = cfg.MODEL.BILINEAR.DIM,
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS
        )

        # (2) Visual Persistence in Decoder
        # self.vp_de为Fasle时，不定义self.p_prev_att层
        # self.vp_de为True时，定义self.p_prev_att层
        if self.vp_de:
            self.p_prev_att = nn.Sequential(
                nn.Linear(cfg.MODEL.RNN_SIZE*3, 1),
                nn.Sigmoid()
            )

        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE),
            nn.GLU()
        )

    # state[0] -- h, state[1] -- c
    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        state = kwargs[cfg.PARAM.STATE]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # 如果使用Visual Persistence，则需要从state中还原prev_att
        if self.vp_de:
            prev_att = state[1][1]

        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
        xt = self.word_embed(wt)

        # lstm1
        input1 = torch.cat([xt, gv_feat, self.ctx_drop(state[0][1])], 1)
        h1_t, c1_t = self.att_lstm(input1, (state[0][0], state[1][0]))
        att, _ = self.attention(h1_t, att_feats, att_mask, p_att_feats, precompute=True)

        # 是否Visual Persistence in Encoder在父类AttBasicModel中
        # 是否使用Visual Persistence in Decoder
        if self.vp_de:
            assert prev_att is not None, 'prev_att is None'
            p = self.p_prev_att(torch.cat([prev_att, att, h1_t], 1))
            att = (1-p)*att + p*prev_att

        ctx_input = torch.cat([att, h1_t], 1)

        output = self.att2ctx(ctx_input)

        # 判断是否Visual Persistence，并分情况保存state
        if self.vp_de:
            # 如果使用Visual Persistence in Decoder，将att保存在state[1][1]中（为了代码兼容）
            # ([2, B, 1024], [2, B, 1024])
            state = [torch.stack([h1_t, output]), torch.stack([c1_t, att])]
        else:
            state = [torch.stack([h1_t, output]), torch.stack([c1_t, state[1][1]])]
        return output, state


# 用于图像目标特征的增强（用于Encoder）
class AoA_Refine_Module(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num):
        super(AoA_Refine_Module, self).__init__()

        self.layers = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([])
        for _ in range(layer_num):
            sublayer = AoAMultiHeadAttentionEnc(
                embed_dim=embed_dim,
                att_type=att_type,
                att_heads=att_heads,
                att_mid_dim=att_mid_dim,
                att_mid_drop=att_mid_drop,
                dropout=dropout)
            self.layers.append(sublayer)
            self.layer_norms.append(torch.nn.LayerNorm(embed_dim))

    def forward(self, _, att_feats, att_mask, p_att_feats=None):
        """
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
        """
        # 先对目标特征进行增强
        for i, layer in enumerate(self.layers):
            # q1, q2, key, mask, v1, v2
            _, att_feats_ = layer(_, att_feats, att_feats, att_mask, att_feats, att_feats)
            # 残差连接
            att_feats = self.layer_norms[i](att_feats + att_feats_)

        # AoA在对att_feats进行增强后，求均值作为全局特征gv_feat
        if att_mask is not None:
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
        else:
            gv_feat = torch.mean(att_feats, 1)

        return gv_feat, att_feats


class AoAMultiHeadAttentionEnc(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout):
        super(AoAMultiHeadAttentionEnc, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        output_dim = embed_dim

        """
        # AoA不需要对全局特征进行Linear
        # query1 用于全局特征增强的query
        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        sequential.append(nn.CELU(1.3))
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q1 = nn.Sequential(*sequential)
        """

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

        self.attn_net = AoASCAttEnc(att_mid_dim, att_mid_drop)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, _, query2, key, mask, value1, value2, precompute=False):
        """
        输入数据：
        # query1: [B, 1024]
        query2: [B, M, 1024]
        key: [B, M, 1024]
        mask: [B, M]
        value1: [B, M, 1024]
        value2: [B, M, 1024]
        """
        # 输入数据全连接层
        batch_size = query2.size()[0]

        query2 = query2.view(-1, query2.size()[-1])
        q2 = self.in_proj_q2(query2)

        key = key.view(-1, key.size()[-1])
        k = self.in_proj_k(key)

        value1 = value1.view(-1, value1.size()[-1])
        v1 = self.in_proj_v1(value1)

        value2 = value2.view(-1, value2.size()[-1])
        v2 = self.in_proj_v2(value2)

        # 输入数据维度变换，用于多头注意力
        # [B, M, 8, 128]
        q2 = q2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 调用注意力机制核心操作函数
        # 将attn_map的计算置于attn_net中进行
        _, attn_q2 = self.attn_net(_, q2, k, mask, v1, v2)

        # 将输出从多头维度上恢复为正确维度
        # [B, 8, M, 128] --> [B, M, 8, 128]  -->  [B, M, 1024]
        attn_q2 = attn_q2.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        if self.dropout is not None:
            # attn_q1 = self.dropout(attn_q1)
            attn_q2 = self.dropout(attn_q2)
        return _, attn_q2


class AoASCAttEnc(nn.Module):
    def __init__(self, mid_dims, mid_dropout):
        super(AoASCAttEnc, self).__init__()
        """
        # 用于全局特征的通道注意力
        self.attention1_channel = nn.Linear(mid_dims[-1], mid_dims[-1])
        """
        # 用于目标特征的通道注意力
        self.attention2_channel = nn.Linear(mid_dims[-1], mid_dims[-1])
        self.sparsemax = torch.nn.Softmax(dim=-1)

    def forward(self, _, query2, key, att_mask, value1, value2):
        # query1 [B, 8, 128]       用于全局特征增强的spatial注意力
        # query2 [B, 8, M, 128]    用于目标特征增强的spatial注意力
        # key [B, 8, M, 128]
        # att_mask [B, M]
        # value1 [B, 8, M, 128]    用作channel注意力的query（全局特征和目标特征）
        # value2 [B, 8, M, 128]

        # scaling = query1.shape[-1] ** -0.5  # (128) ** -0.5，即 1 / sqrt(128)
        scaling = query2.shape[-1] ** -0.5

        ######################################
        # 1 先对目标特征增强
        ######################################
        # 1.1 空间注意力
        # 矩阵乘法： [B, 8, M, 128] x [B, 8, 128, M]  -->  [B, 8, M, M]
        att_map2_spatial = torch.matmul(query2, key.transpose(-1, -2)) * scaling

        # 1.2 通道注意力（同时用于目标特征和全局特征增强）
        # 矩阵点乘： [B, 8, M, 128] * [B, 8, M, 128]  -->  [B, 8, M, 128]
        query_channel = value1  # value1作为通道注意力的query
        # （1）[B, 8, M, 128]，用于目标特征
        att_map2_channel = query_channel * key

        # 【目标特征】空间注意力权重计算
        # [B, 8, M, M]
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)  # [B, 1, M]
            tmp_att_mask = att_mask.unsqueeze(-2)  # [B, 1, 1, M]
            att_map2_spatial = att_map2_spatial.masked_fill(tmp_att_mask == 0, -1e9)
        alpha2_spatial = F.softmax(att_map2_spatial, dim=-1)
        # 矩阵乘法[B, 8, M, M] x [B, 8, M, 128] --> [B, 8, M, 128]
        attn2 = torch.matmul(alpha2_spatial, value2)

        # 实际AoA中并没有通道注意力
        # 【目标特征】通道注意力权重计算
        # att_map2_channel [B, 8, M, 128]  -->  [B, 8, M, 128]
        alpha2_channel = self.attention2_channel(att_map2_channel)
        alpha2_channel = torch.sigmoid(alpha2_channel)
        attn2 = alpha2_channel * attn2  # 增强后的目标特征

        # [B, 8, 128], [B, 8, M, 128]
        return _, attn2