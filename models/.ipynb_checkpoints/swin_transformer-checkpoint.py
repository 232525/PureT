import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

# from layers.low_rank import LowRank
# import blocks
# from layers.positional_encoding import PositionalEncoding

import lib.utils as utils
from models.basic_model import BasicModel
from models.swin_encoder import Encoder
from models.swin_decoder import Decoder

# from models.swin_encoder_decoder.swin_decoder2 import Decoder

from models.backbone.swin_transformer_backbone import SwinTransformer as STBackbone

# from models.m2_encoders import Encoder
# from models.tmp_encoder.x_encoder import Encoder

# 构造三角矩阵，用于生成训练时单词序列的mask
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# """
class SwinTransformer(BasicModel):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        
        # """
        # large 特征维度：1536
        self.backbone = STBackbone(
            img_size=384, 
            embed_dim=192, 
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            num_classes=1000
        )
        print('load pretrained weights!')
        self.backbone.load_weights(
            './swin_large_patch4_window12_384_22kto1k_no_head.pth'
        )
        # 固定参数
        for _name, _weight in self.backbone.named_parameters():
            _weight.requires_grad = False
            print(_name, _weight.requires_grad)
            
        # """
        """
        # base 特征维度：1024
        self.backbone = STBackbone(
            img_size=384, 
            embed_dim=128, 
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            num_classes=1000
        )
        print('load pretrained weights!')
        self.backbone.load_weights(
            './swin_base_patch4_window12_384_22kto1k_no_head.pth'
        )
        
        # 删除backbone最后一层BasicLayer，相当于最后一层替换成了encoder
        # self.backbone.layers[-1] = nn.Identity()
        # self.backbone.norm = nn.Identity()
        # """
        """
        self.backbone = STBackbone(
            img_size=384, 
            embed_dim=64, 
            depths=[2, 2, 18, 2],
            num_heads=[2, 4, 8, 16],
            window_size=12,
            num_classes=1000
        )
        """
        
        # att_feats encoder
        # """
        if cfg.MODEL.ATT_FEATS_DIM == cfg.MODEL.ATT_FEATS_EMBED_DIM:
            self.att_embed = nn.Identity()
        else:
            self.att_embed = nn.Sequential(
                nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM),
                utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT),
                nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM) if cfg.MODEL.ATT_FEATS_NORM == True else nn.Identity(),
                nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED)
            )
        

        """
        self.encoder = Encoder(
            dim=cfg.MODEL.ATT_FEATS_EMBED_DIM, 
            input_resolution=(12, 12), 
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS, 
            num_heads=cfg.MODEL.BILINEAR.HEAD, 
            window_size=6,
            mlp_ratio=4., 
            qkv_bias=True, 
            qk_scale=None, 
            drop=0.1, 
            attn_drop=0.1,
            drop_path=0.1, 
            norm_layer=nn.LayerNorm
        )
        # """
        
        use_gx = True
        # """
        # 修改后SwinTransformer Encoder
        self.encoder = Encoder(
            dim=cfg.MODEL.ATT_FEATS_EMBED_DIM, 
            input_resolution=(12, 12), 
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS, 
            num_heads=cfg.MODEL.BILINEAR.HEAD, 
            window_size=6,  # =12 SW-MSA/W-MSA退化为普通MSA
            shift_size=3,    # =0  无SW-MSA，仅W-MSA
            mlp_ratio=4,
            drop=0.1,
            use_gx = use_gx
        )
        # """
        
        """
        # m2 encoder
        self.encoder = Encoder(
            4, 
            d_in=1536, 
            d_model=1024, 
            d_k=128, 
            d_v=128, 
            h=8, 
            d_ff=4096
        )
        """
        
        """
        # XTransformer Encoder
        self.encoder = Encoder(
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            dropout = cfg.MODEL.BILINEAR.ENCODE_DROPOUT, 
            att_type = cfg.MODEL.BILINEAR.ATTTYPE, 
            att_heads = cfg.MODEL.BILINEAR.HEAD, 
            att_mid_dim = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM, 
            att_mid_drop = cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT, 
            bifeat_emb_act = cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT, 
            bifeat_emb_drop = cfg.MODEL.BILINEAR.ENCODE_BIFEAT_EMB_DROPOUT,
            ff_dropout = cfg.MODEL.BILINEAR.ENCODE_FF_DROPOUT,
            layer_num = cfg.MODEL.BILINEAR.ENCODE_LAYERS)
        # """

        """
        self.decoder = Decoder(
            dim=cfg.MODEL.ATT_FEATS_EMBED_DIM, 
            input_resolution=(12, 12), 
            depth=cfg.MODEL.BILINEAR.DECODE_LAYERS, 
            num_heads=cfg.MODEL.BILINEAR.HEAD, 
            window_size=12,
            mlp_ratio=4., 
            qkv_bias=True, 
            qk_scale=None, 
            drop=0.1, 
            attn_drop=0.1,
            drop_path=0.1, 
            norm_layer=nn.LayerNorm
        )
        """
        
        self.decoder = Decoder(
            vocab_size = self.vocab_size, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            att_heads = cfg.MODEL.BILINEAR.HEAD, 
            ff_dropout = cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS,
            use_gx = use_gx
        )
        
    def forward(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]
        
        # backbone forward
        att_feats = self.backbone(att_feats)
        
        # grid特征，无需att_mask
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        # 构造单词序列掩码 [B, L, L]
        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:,0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################

        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # decoder_out = self.decoder(seq, encoder_out, seq_mask)
        decoder_out = self.decoder(gx, seq, encoder_out, seq_mask, att_mask)
        return F.log_softmax(decoder_out, dim=-1)

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
        
        # 无需att_mask及图像全局特征gx
        # 预计算p_att_feats可以考虑保留，减少Linear层的调用
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        gx = kwargs[cfg.PARAM.GLOBAL_FEAT]
        # p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # state[0][0]: [B, seq_len-1]，用于保存截止到当前步，生成的单词序列
        # ys: [B, seq_len]，seq_len为当前步数，即单词个数
        if state is None:
            ys = wt.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
            
        # 依靠ys，计算seq_mask
        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(1)
        
        # [B, 1, Vocab_Size] --> [B, Vocab_Size]
        # decoder_out = self.decoder(ys[:, -1].unsqueeze(-1), encoder_out, seq_mask).squeeze(1)
        decoder_out = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, seq_mask, att_mask).squeeze(1)
        
        logprobs = F.log_softmax(decoder_out, dim=-1)
        # logprobs = torch.softmax(decoder_out, dim=-1).log()
        
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s
        return fn

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # p_att_feats = self.decoder.precompute(encoder_out)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        self.decoder.init_buffer(batch_size)
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            # [B*cur_beam_size, Vocab_size] --> [B, cur_beam_size, Vocab_size]
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            # 候选log概率，即已生成单词log概率之和，用于判断该步选择哪个单词
            # [B, cur_beam_size, Vocab_size]
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            # 基于candidate_logprob选择出前beam_size大的序列index及log概率（句子）
            # [B, beam_size], [B, beam_size]
            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            # selected_beam为选择的单词在哪个beam里面，[B, 3]
            # selected_words为选择的单词在词汇表中的index，[B, 3]
            selected_beam = selected_idx // candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            # 对decoder中的buffer进行更新
            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                # 相关输入复制扩展，用于下一步beam_search
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                gx = utils.expand_tensor(gx, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                # p_att_feats_tmp = []
                # for p_feat in p_att_feats:
                #     p_key, p_value2 = p_feat
                #     p_key = utils.expand_tensor(p_key, beam_size)
                #     p_value2 = utils.expand_tensor(p_value2, beam_size)
                #     p_att_feats_tmp.append((p_key, p_value2))

                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats_tmp
 
        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        att_feats = self.backbone(att_feats)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        # 预计算可考虑保留，减少Linear层调用次数
        # p_att_feats = self.decoder.precompute(encoder_out)
        self.decoder.init_buffer(batch_size)
        
        state = None
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        # 预计算可考虑保留，减少Linear层调用次数
        # kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        
        # 按时间步迭代进行推理计算
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            logprobs_t, state = self.get_logprobs_state(**kwargs)
            
            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        self.decoder.clear_buffer()
        return sents, logprobs
# """


# Decoder 使用了图像全局特征预融合
from models.swin_encoder_decoder.swin_decoder2 import Decoder as Decoder2

class D2SwinTransformer(SwinTransformer):
    def __init__(self):
        super(D2SwinTransformer, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        use_gx = True
        
        del self.decoder
        self.decoder = Decoder2(
            vocab_size = self.vocab_size, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            att_heads = cfg.MODEL.BILINEAR.HEAD, 
            ff_dropout = cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS,
            use_gx = use_gx
        )
        
from models.swin_encoder_decoder.swin_encoder_wo_window import Encoder as Encoder_wo_window
class Transformer_N3(SwinTransformer):
    def __init__(self):
        super(Transformer_N3, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        use_gx = False
        
        del self.encoder
        del self.decoder
        print('Transformer_N3')
        
        # 对于wo_window，部分参数无作用，仅为了API一致
        self.encoder = Encoder_wo_window(
            dim=cfg.MODEL.ATT_FEATS_EMBED_DIM, 
            input_resolution=(12, 12), 
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS, 
            num_heads=cfg.MODEL.BILINEAR.HEAD, 
            window_size=6,  # =12 SW-MSA/W-MSA退化为普通MSA
            shift_size=3,    # =0  无SW-MSA，仅W-MSA
            mlp_ratio=4,
            drop=0.1,
            use_gx = use_gx
        )
        
        self.decoder = Decoder(
            vocab_size = self.vocab_size, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            att_heads = cfg.MODEL.BILINEAR.HEAD, 
            ff_dropout = cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS,
            use_gx = use_gx
        )
        
from models.swin_encoder_decoder.swin_encoder_wo_window import Encoder as Encoder_wo_window
from models.swin_encoder_decoder.swin_decoder2 import Decoder as Decoder2

class Transformer_N3_w_pf(SwinTransformer):
    def __init__(self):
        super(Transformer_N3_w_pf, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        use_gx = True
        
        del self.encoder
        del self.decoder
        print('Transformer_N3_w_pf')
        
        # 对于wo_window，部分参数无作用，仅为了API一致
        self.encoder = Encoder_wo_window(
            dim=cfg.MODEL.ATT_FEATS_EMBED_DIM, 
            input_resolution=(12, 12), 
            depth=cfg.MODEL.BILINEAR.ENCODE_LAYERS, 
            num_heads=cfg.MODEL.BILINEAR.HEAD, 
            window_size=6,  # =12 SW-MSA/W-MSA退化为普通MSA
            shift_size=3,    # =0  无SW-MSA，仅W-MSA
            mlp_ratio=4,
            drop=0.1,
            use_gx = use_gx
        )
        
        self.decoder = Decoder2(
            vocab_size = self.vocab_size, 
            embed_dim = cfg.MODEL.BILINEAR.DIM, 
            dropout = cfg.MODEL.BILINEAR.DECODE_DROPOUT, 
            att_heads = cfg.MODEL.BILINEAR.HEAD, 
            ff_dropout = cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
            layer_num = cfg.MODEL.BILINEAR.DECODE_LAYERS,
            use_gx = use_gx
        )