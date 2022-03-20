import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_model import BasicModel
from models.att_basic_model import AttBasicModel
# from layers.attention import Attention
from lib.config import cfg
import lib.utils as utils
from models.vp_att_dec import VP_Attention_Module


class VP_UpDown(AttBasicModel):
    def __init__(self, vp_en=False, vp_de=False):
        super(VP_UpDown, self).__init__()
        self.vp_en, self.vp_de = vp_en, vp_de
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.WORD_EMBED_DIM + self.att_dim
        self.lstm1 = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        # Second LSTM Layer
        self.lstm2 = nn.LSTMCell(cfg.MODEL.RNN_SIZE + self.att_dim, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)

        # (1) Visual Persistence in Encoder
        # self.vp_en为False时，特征增强模块重载，不进行增强
        # self.vp_en为True时，使用AttBasicModel中的特征增强
        if not self.vp_en:
            del self.encoder_layers
            self.encoder_layers = lambda gv_feat, att_feats, att_mask: (gv_feat, att_feats)

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
            # 如果使用Visual Persistence in Decoder，则使用GLU取代lstm2
            del self.lstm2
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
        h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
        att, _ = self.attention(h1_t, att_feats, att_mask, p_att_feats, precompute=True)

        # 是否Visual Persistence in Encoder在父类AttBasicModel中
        # 是否使用Visual Persistence in Decoder
        if self.vp_de:
            assert prev_att is not None, 'prev_att is None'
            p = self.p_prev_att(torch.cat([prev_att, att, h1_t], 1))
            att = (1-p)*att + p*prev_att

        # lstm2
        if not self.vp_de:
            input2 = self.ctx_drop(torch.cat([att, h1_t], 1))
            h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))
        else:
            input2 = self.ctx_drop(torch.cat([att, h1_t], 1))
            h2_t = self.att2ctx(input2)

        # 判断是否Visual Persistence，并分情况保存state
        if self.vp_de:
            state = [torch.stack([h1_t, h2_t]), torch.stack([c1_t, att])]
        else:
            state = [torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t])]
        return h2_t, state