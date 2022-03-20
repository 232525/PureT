import torch
import torch.nn as nn

from lib.config import cfg
import lib.utils as utils
from models.att_basic_model import AttBasicModel

import blocks

# 特征增强  Refine Layer
# from models.vp_ablation_att_enc import VP_Refine_Module
# 无特征增强
# from models.vp_ablation_att_enc import VP_Refine_Module_init as VP_Refine_Module

# 使用门控自注意力机制进行特征增强，4层
# from models.vp_ablation_att_enc import VP_Refine_Module_gated_self_attn as VP_Refine_Module

# XLAN特征增强，残差连接部分加入了门控机制，4层
# from models.vp_ablation_att_enc import VP_Refine_Module_gated_residual as VP_Refine_Module

# 语言模型注意力 Attention Layer
# 1、XLAN 注意力
# from models.vp_ablation_att_dec import VP_Attention_Module

# ----------------------------
# 2、XLAN注意力，简化版，在同一个class中实现所有操作
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple as VP_Attention_Module

# ----------------------------
# 3、XLAN注意力，极简化版，去除了1）所有激活函数、正则化及Dropout；2）通道注意力；3）tail_proj，只保留了0）空间注意力
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_ablation as VP_Attention_Module
# 4、XLAN注意力，极简化版，去除了2）通道注意力；3）tail_proj，保留了0）空间注意力和1）激活函数、正则化及Dropout
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_ablation_w_norm as VP_Attention_Module
# 5、XLAN注意力，极简化版，去除了1）激活函数、正则化及Dropout；2）通道注意力；保留了0）空间注意力和，3）tail_proj
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_ablation_w_tail as VP_Attention_Module
# 6、XLAN注意力，极简化版，去除了1）激活函数、正则化及Dropout；3）tail_proj；保留了0）空间注意力和，2）通道注意力
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_ablation_w_channel as VP_Attention_Module
# 7、XLAN注意力，极简化版，去除了2）通道注意力；保留了0）空间注意力和；1）激活函数、正则化及Dropout；3）tail_proj
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_ablation_w_norm_tail as VP_Attention_Module
# 8、XLAN注意力，极简化版，去除了3）tail_proj；保留了0）空间注意力和；1）激活函数、正则化及Dropout；2）通道注意力
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_ablation_w_channel_norm as VP_Attention_Module
# 9、XLAN注意力，极简化版，去除了1）激活函数、正则化及Dropout；保留了0）空间注意力和；2）通道注意力；3）tail_proj
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_ablation_w_channel_tail as VP_Attention_Module

# ----------------------------
# xlan结构，但是query和key的交互由Hadamard积改为add操作
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_addatt as VP_Attention_Module

# ----------------------------
# xlan结构，配合lstm输入替换为上一步的attend feature结构组合使用
# 1、保留空间注意力、通道注意力和激活函数+GroupNorm，去除tail_proj
# 2、通道注意力去除sigmoid门控机制
# 3、h_state和att_feats的交互计算使用add替换Hadamard积，包括query和key的相似度计算，及空间注意力和通道注意力结构的融合
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_attinlstm as VP_Attention_Module

# ----------------------------
# xlan结构，但是激活函数改为ReLU，标准化改为LayerNorm
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_relu_ln as VP_Attention_Module

# ----------------------------
# xlan结构，但是通道注意力部分不进行sigmoid门控加权，且删除了tail_proj操作
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_wo_sigmoid as VP_Attention_Module

# ----------------------------
# LAMA attention
# from models.vp_ablation_att_dec import VP_Attention_Module_lama_simple as VP_Attention_Module

# ----------------------------
# XLAN注意力，简化版，在同一个class中实现所有操作
# 操作与论文中Figure 2保持一致，与开源代码有些许区别
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_fusion_h as VP_Attention_Module

# ----------------------------
# XLAN注意力，简化版，在同一个class中实现所有操作
# grid feature和h_state融合使用了glu
# from models.vp_ablation_att_dec import VP_Attention_Module_xlan_simple_fusion_h_1 as VP_Attention_Module

# ----------------------------
# 传统add注意力
# from models.vp_ablation_att_dec import VP_Attention_Module_add as VP_Attention_Module

# SCA-CNN注意力
# from models.vp_ablation_att_dec import VP_Attention_Module_add_sca as VP_Attention_Module

# 参考XLAN的结构，但是Linear层删掉了CELU激活函数和GroupNorm
# from models.vp_ablation_att_dec import VP_Attention_Module_init as VP_Attention_Module

# 自注意力机制
# from models.vp_ablation_att_dec import VP_Attention_Module_self as VP_Attention_Module

# 多头Linear层参数不共享
from models.vp_ablation_att_enc_mh import VP_Refine_Module_gated_residual as VP_Refine_Module
from models.vp_ablation_att_dec_mh import VP_Attention_Module_xlan_simple_fusion_h as VP_Attention_Module

# 特征增强部分采用M2 Transformer中的Encoder部分
# from models.m2_encoders import VP_Refine_Moduel_m2_encoder as VP_Refine_Module

class VPNet(AttBasicModel):
    def __init__(self):
        super(VPNet, self).__init__()
        self.num_layers = 2

        # First LSTM layer
        # 1、VPNet mode, vector "Concat"
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM * 2
        # 2、XLAN mode, vector "Add"
        # rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM
        
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)
        
        """
        # 两层LSTM结构
        self.lang_lstm = nn.LSTMCell(cfg.MODEL.RNN_SIZE*2, cfg.MODEL.RNN_SIZE)
        """
        
        # Re-build for VPNet
        # Refining Layer with Visual Persistence in Encoder
        del self.encoder_layers
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
        
        """
        # Visual Persistence in Decoder
        self.p_prev_att = nn.Sequential(
            nn.Linear(cfg.MODEL.RNN_SIZE*3, 1), 
            nn.Sigmoid()
        )
        """
        
        # """
        # Full
        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE), 
            nn.GLU()
        )
        # """
        
        """
        # w/o GLU
        self.att2ctx = nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 
                                 cfg.MODEL.RNN_SIZE)
        """
                
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
        
        # Attention Layer
        att, _ = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        
        """
        # Visual Persistence in Decoder
        if state[1][1].max() > 0:
            prev_att = state[1][1]
            p = self.p_prev_att(torch.cat([prev_att, att, h_att], 1))
            att = (1-p)*att + p*prev_att
        """
        
        # Word Generation Layer
        ctx_input = torch.cat([att, h_att], 1)
        
        """
        # 两层lstm结构
        output, _ = self.lang_lstm(ctx_input, (state[0][1], state[1][1]))
        state = [torch.stack((h_att, output)), torch.stack((c_att, _))]
        """
        
        # """
        # lstm + glu or linear结构
        output = self.att2ctx(ctx_input)
        state = [torch.stack((h_att, output)), torch.stack((c_att, att))]
        # """
        
        return output, state
    
class VPNet_Gated_Res(AttBasicModel):
    def __init__(self):
        super(VPNet_Gated_Res, self).__init__()
        self.num_layers = 2
        
        # Re-build for VPNet
        # Refining Layer with Visual Persistence in Encoder
        # 图像特征增强模块（重载）
        del self.encoder_layers
        self.encoder_layers = VP_Refine_Module( 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads = cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM,
            att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT,
            dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
            layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS
        )

        # First LSTM layer
        # 1、VPNet mode, vector "Concat"
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM * 2
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        
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
        
        """
        # Visual Persistence in Decoder
        self.p_prev_att = nn.Sequential(
            nn.Linear(cfg.MODEL.RNN_SIZE*3, 1), 
            nn.Sigmoid()
        )
        """
        
        """
        # 仅有GLU()
        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE), 
            nn.GLU()
        )
        """
        
        # 相当于GLU()加上残差
        # GLU改为门控残差
        self.att2ctx_feat = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, cfg.MODEL.RNN_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.att2ctx_gate = nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, cfg.MODEL.RNN_SIZE)
                
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
        
        # 词汇嵌入向量
        xt = self.word_embed(wt)
        
        # LSTM Layer
        # 1、VPNet mode, vector "Concat"
        h_att, c_att = self.att_lstm(torch.cat([xt, gv_feat, state[0][1]], 1), (state[0][0], state[1][0]))
        
        # Attention Layer
        att, _ = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        
        """
        # Visual Persistence in Decoder
        if state[1][1].max() > 0:
            prev_att = state[1][1]
            p = self.p_prev_att(torch.cat([prev_att, att, h_att], 1))
            att = (1-p)*att + p*prev_att
        """
        
        # Word Generation Layer
        ctx_input = torch.cat([att, h_att], 1)
        
        # 门控残差机制，与GLU相似
        gate = torch.sigmoid(self.att2ctx_gate(ctx_input))
        # 加号前可视为GLU激活函数
        output = gate * self.att2ctx_feat(ctx_input) + (1 - gate) * att
        # output = gate * self.att2ctx_feat(ctx_input) + (1 - gate) * h_state
        
        state = [torch.stack((h_att, output)), torch.stack((c_att, att))]
        
        return output, state
    
class VPNet_GRU(AttBasicModel):
    def __init__(self):
        super(VPNet_GRU, self).__init__()
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM * 2
        
        self.att_gru = nn.GRUCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)
        
        # Re-build for VPNet
        # Refining Layer with Visual Persistence in Encoder
        del self.encoder_layers
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
        
        """
        # Visual Persistence in Decoder
        self.p_prev_att = nn.Sequential(
            nn.Linear(cfg.MODEL.RNN_SIZE*3, 1), 
            nn.Sigmoid()
        )
        """
        
        # """
        # Full
        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE), 
            nn.GLU()
        )
        # """
                
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
        
        # GRU Layer
        h_att = self.att_gru(torch.cat([xt, gv_feat, self.ctx_drop(state[0][1])], 1), state[0][0])
        
        # Attention Layer
        att, _ = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        
        """
        # Visual Persistence in Decoder
        if state[1][1].max() > 0:
            prev_att = state[1][1]
            p = self.p_prev_att(torch.cat([prev_att, att, h_att], 1))
            att = (1-p)*att + p*prev_att
        """
        
        # Word Generation Layer
        ctx_input = torch.cat([att, h_att], 1)
        
        # """
        # lstm + glu or linear结构
        output = self.att2ctx(ctx_input)
        state = [torch.stack((h_att, output)), torch.stack((state[1][0], att))]
        # """
        
        return output, state
    
class VPNet_LSTM_Hadamard(AttBasicModel):
    def __init__(self):
        super(VPNet_LSTM_Hadamard, self).__init__()
        self.num_layers = 2

        # First LSTM layer
        # 1、VPNet mode, vector "Concat"
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM * 2
        # 2、XLAN mode, vector "Add"
        # rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM
        
        self.att_lstm = LSTMCell_Hadamard(rnn_input_size, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)
        
        """
        # 两层LSTM结构
        self.lang_lstm = nn.LSTMCell(cfg.MODEL.RNN_SIZE*2, cfg.MODEL.RNN_SIZE)
        """
        
        # Re-build for VPNet
        # Refining Layer with Visual Persistence in Encoder
        del self.encoder_layers
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
        
        """
        # Visual Persistence in Decoder
        self.p_prev_att = nn.Sequential(
            nn.Linear(cfg.MODEL.RNN_SIZE*3, 1), 
            nn.Sigmoid()
        )
        """
        
        # """
        # Full
        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE), 
            nn.GLU()
        )
        # """
        
        """
        # w/o GLU
        self.att2ctx = nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 
                                 cfg.MODEL.RNN_SIZE)
        """
                
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
        
        # Attention Layer
        att, _ = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        
        """
        # Visual Persistence in Decoder
        if state[1][1].max() > 0:
            prev_att = state[1][1]
            p = self.p_prev_att(torch.cat([prev_att, att, h_att], 1))
            att = (1-p)*att + p*prev_att
        """
        
        # Word Generation Layer
        ctx_input = torch.cat([att, h_att], 1)
        
        """
        # 两层lstm结构
        output, _ = self.lang_lstm(ctx_input, (state[0][1], state[1][1]))
        state = [torch.stack((h_att, output)), torch.stack((c_att, _))]
        """
        
        # """
        # lstm + glu or linear结构
        output = self.att2ctx(ctx_input)
        state = [torch.stack((h_att, output)), torch.stack((c_att, att))]
        # """
        
        return output, state
    
import math
class LSTMCell_Hadamard(nn.RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell_Hadamard, self).__init__(input_size, hidden_size, bias, num_chunks=4)
        
    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        
        # 输入：
        # input: [B, input_size]
        # hx[0]: [B, hidden_size] 即 h
        # hx[1]: [B, hidden_size] 即 c
        # 权重:
        # self.weight_ih: [4 * hidden_size, input_size]
        # self.bias_ih:   [4 * hidden_size]
        # self.weight_hh: [4 * hidden_size, hidden_size]
        # self.bias_hh:   [4 * hidden_size]
        
        input_ifgo = torch.matmul(self.weight_ih, input.unsqueeze(-1)).squeeze(-1) + self.bias_ih
        h_ifgo = torch.matmul(self.weight_hh, hx[0].unsqueeze(-1)).squeeze(-1) + self.bias_hh
        fuse_ifgo = input_ifgo + h_ifgo
        
        i,f,g,o = torch.chunk(fuse_ifgo, 4, dim=-1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_ = f * hx[1] + i * g
        h_ = o * torch.tanh(c_)
        
        return h_, c_
    
    
# 将lstm输入中的全局特征，替换为上一步的attended feature
class VPNet_AttInLSTM(AttBasicModel):
    def __init__(self):
        super(VPNet_AttInLSTM, self).__init__()
        self.num_layers = 2

        # First LSTM layer
        # 1、VPNet mode, vector "Concat"
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM * 2
        # 2、XLAN mode, vector "Add"
        # rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM
        
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)
        
        # Re-build for VPNet
        # Refining Layer with Visual Persistence in Encoder
        del self.encoder_layers
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
        
        """
        # Visual Persistence in Decoder
        self.p_prev_att = nn.Sequential(
            nn.Linear(cfg.MODEL.RNN_SIZE*3, 1), 
            nn.Sigmoid()
        )
        """
        
        # """
        # Full
        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE), 
            nn.GLU()
        )
        # """
        
        """
        # w/o GLU
        self.att2ctx = nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 
                                 cfg.MODEL.RNN_SIZE)
        """
                
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
        
        if state[1][1].max() > 0:
            prev_att = state[1][1]
        else:
            prev_att = gv_feat
            
        # LSTM Layer
        # 1、VPNet mode, vector "Concat"
        # LSTM输入中全局特征部分，改为上一时间步的attended feature
        h_att, c_att = self.att_lstm(torch.cat([xt, prev_att, self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        
        # Attention Layer
        att, _ = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)
        
        """
        # Visual Persistence in Decoder
        if state[1][1].max() > 0:
            prev_att = state[1][1]
            p = self.p_prev_att(torch.cat([prev_att, att, h_att], 1))
            att = (1-p)*att + p*prev_att
        """
        
        # Word Generation Layer
        ctx_input = torch.cat([att, h_att], 1)        
        # """
        # lstm + glu or linear结构
        output = self.att2ctx(ctx_input)
        state = [torch.stack((h_att, output)), torch.stack((c_att, att))]
        # """
        
        return output, state