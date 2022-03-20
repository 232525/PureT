import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Encoder部分和Decoder部分通用操作
包含：
FeedForward层
"""

# FeedForward Layer 
# (without skip connection and LayerNorm)
class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.3):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        
    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        
        return x