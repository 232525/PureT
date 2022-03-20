import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model import AttBasicModel

import blocks

# 特征增强  Refine Layer
# from models.vp_att_enc import VP_Refine_Module
# from models.vp_att_enc_ban import VP_Refine_Module
# from models.vp_att_enc_mca import VP_Refine_Module
# from models.vp_att_enc_mca1 import VP_Refine_Module
# from models.vp_att_enc_sge import VP_Refine_Module
from models.vp_att_enc_mh import VP_Refine_Module

# 语言模型注意力 Attention Layer
# from models.vp_att_dec import VP_Attention_Module
# from models.vp_att_dec_ban import VP_Attention_Module
from models.vp_att_dec_mh import VP_Attention_Module

class VPNet(AttBasicModel):
    def __init__(self):
        super(VPNet, self).__init__()
        self.num_layers = 2

        # First LSTM layer
        # 1、VPNet mode, vector "Concat"
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM * 2
        # 2、XLAN mode, vector "Add"
        # rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM
        # 3、Bilinear Pooling
        """
        rnn_input_size = cfg.MODEL.RNN_SIZE * 3
        self.fuse_in = nn.Sequential(
            nn.Linear(cfg.MODEL.RNN_SIZE * 3, 3 * cfg.MODEL.RNN_SIZE // 2), 
            nn.Tanh(),
            nn.Linear(3 * cfg.MODEL.RNN_SIZE // 2, cfg.MODEL.RNN_SIZE),
            nn.ReLU()
        )
        self.fuse_out = nn.Sequential(
            nn.Linear(cfg.MODEL.RNN_SIZE, 3 * cfg.MODEL.RNN_SIZE // 2), 
            nn.ReLU(),
            nn.Linear(3 * cfg.MODEL.RNN_SIZE // 2, cfg.MODEL.RNN_SIZE * 3)
        )
        """
        
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)
        
        # Re-build for VPNet
        # Refining Layer with Visual Persistence in Encoder
        del self.encoder_layers
        # """
        # Refining Layer for VPNet
        self.encoder_layers = VP_Refine_Module( 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
            layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
        )
        # """
        
        """
        # Refining Layer for XLAN
        self.encoder_layers = blocks.create(
            cfg.MODEL.BILINEAR.ENCODE_BLOCK, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT,
            layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
        )
        """

        # """
        # Attention Layer for VPNet
        self.attention = VP_Attention_Module(
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS
        )
        # """ 
        
        """
        # 使用XLAN的Attention Layer实验
        self.attention = blocks.create(
            cfg.MODEL.BILINEAR.DECODE_BLOCK, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS
        )
        """
        
        # """
        # Visual Persistence in Decoder
        self.p_prev_att = nn.Sequential(
            nn.Linear(cfg.MODEL.RNN_SIZE*3, 1), 
            nn.Sigmoid()
        )
        # """
        
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
        
        # initialization of gv_feat (redudant)
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
        
        xt = self.word_embed(wt)
        
        # LSTM Layer
        # 1、VPNet mode, vector "Concat"
        h_att, c_att = self.att_lstm(torch.cat([xt, gv_feat, self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        # 2、XLAN mode, vector "Add"
        # h_att, c_att = self.att_lstm(torch.cat([xt, gv_feat + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        # 3、Bilinear Pooling
        """
        lstm_input = torch.cat([xt, gv_feat, self.ctx_drop(state[0][1])], 1)
        lstm_input = self.fuse_in(lstm_input)
        lstm_input = self.fuse_out(lstm_input)
        h_att, c_att = self.att_lstm(lstm_input, (state[0][0], state[1][0]))
        """
        
        # Attention Layer
        att, _ = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        
        # """
        # Visual Persistence in Decoder
        if state[1][1].max() > 0:
            prev_att = state[1][1]
            p = self.p_prev_att(torch.cat([prev_att, att, h_att], 1))
            att = (1-p)*att + p*prev_att
        # """
        
        # Word Generation Layer
        ctx_input = torch.cat([att, h_att], 1)

        output = self.att2ctx(ctx_input)
        state = [torch.stack((h_att, output)), torch.stack((c_att, att))]

        return output, state