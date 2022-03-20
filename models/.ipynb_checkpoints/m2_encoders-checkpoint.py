from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention, ScaledDotProductAttentionMemory


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None,
                 attention_module_kwargs={'m': 40}):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                          identity_map_reordering=identity_map_reordering,
                          attention_module=attention_module,
                          attention_module_kwargs=attention_module_kwargs)
             for _ in range(N)]
        )

    def forward(self, input, attention_mask, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask

class VP_Refine_Moduel_m2_encoder(MultiLevelEncoder):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop, dropout, layer_num, **kwargs):
        super(VP_Refine_Moduel_m2_encoder, self).__init__(layer_num, attention_module=ScaledDotProductAttentionMemory)
        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None):
        # 特征增强
        outs, _ = super(VP_Refine_Moduel_m2_encoder, self).forward(att_feats, att_mask==0)
        
        # 全局特征获取
        feat_arr = []
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            feat_arr.append(gv_feat)
            
        for i in range(outs.size()[1]):
            out = torch.sum(outs[:, i, :, :] * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            feat_arr.append(out)
        
        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        
        att_feats = outs[:, -1, :, :].contiguous()
        return gv_feat, att_feats
    
    
class Encoder(MultiLevelEncoder):
    def __init__(self, N, d_in=1536, **kwargs):
        """
        N, d_model=1024, d_k=128, d_v=128, h=8, d_ff=2048, dropout=.1,
        identity_map_reordering=False, attention_module=ScaledDotProductAttentionMemory,
        attention_module_kwargs={'m': 40}
        """
        super(Encoder, self).__init__(N, attention_module=ScaledDotProductAttentionMemory, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        att_mask = torch.ones(out.size()[:2], device='cuda').long()
        outs, _ = super(Encoder, self).forward(out, att_mask==0, attention_weights=attention_weights)
        
        att_feats = outs[:, -1, :, :].contiguous()
        return att_feats
    
    
