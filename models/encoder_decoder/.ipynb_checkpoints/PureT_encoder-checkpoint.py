import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn.functional as F

# 加入全局特征共同处理
class Encoder(nn.Module):
    def __init__(
        self, 
        embed_dim=512, 
        input_resolution=(12, 12), 
        depth=3, 
        num_heads=8, 
        window_size=12,  # =12 退化为普通MSA结构
        shift_size=6,    # =0  无SW-MSA，仅W-MSA
        mlp_ratio=4,
        dropout=0.1,
        use_gx=False
    ):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_gx = use_gx
        
        # 构建 W-MSA / SW-MSA 层
        # 输入特征尺寸为 144 = 12 x 12，如果构建 SW-MSA 层，
        # 则需要将 window_size 设置得更小，比如设置为 6，且shift_size > 0
        # SW-MSA仅在偶数层被构造，W-MSA在奇数层构造
        # 如：W-MSA，SW-MSA，W-MSA，SW-MSA ......
        self.layers = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim, 
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_gx=use_gx
            ) for i in range(self.depth)
        ])
    
    def forward(self, x, att_mask=None):
        # x: [B, H*W, C]
        # 对于grid特征，att mask为None亦可
        # 全局特征初始化，图像特征均值 [B, C]
        if att_mask is not None:
            gx = (torch.sum(x * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1))
        else:
            gx = x.mean(1)
        
        # 如果使用全局特征，则需要将全局特征gx和grid特征x合并送入后续层处理
        if self.use_gx:
            O = torch.cat([x, gx.unsqueeze(1)], dim=1)  # [B, H*W+1, C]
        else:
            O = x
            
        # 核心操作层
        for layer in self.layers:
            O = layer(O, att_mask)
        
        if self.use_gx:
            gx = O[:, -1, :]
            x  = O[:, :-1, :]
        else:
            gx = O.mean(1)
            x = O
        return gx, x
    
    def flops(self):
        flops = 0
        for _l in self.layers:
            flops += _l.flops()
        return flops
    
class EncoderLayer(nn.Module):
    def __init__(
        self, 
        embed_dim=512, 
        input_resolution=(12, 12), 
        num_heads=8, 
        window_size=12,    # 窗口大小，如果窗口大小和输入一致，则退化为普通MSA
        shift_size=0,      # shift大小，0 OR window_size // 2
        mlp_ratio=4,       # FeedForward 中间层维度变换
        dropout=0.1,
        use_gx=False
    ):
        super(EncoderLayer, self).__init__()
        self.embed_dim = embed_dim                            # 1536
        self.input_resolution = input_resolution  # (12， 12)
        self.num_heads = num_heads                # 8
        self.window_size = window_size            # 12 / 6
        self.shift_size = shift_size   # shift_size可用于区分SW-MSA / W-MSA
        self.mlp_ratio = mlp_ratio     # 4
        self.use_gx = use_gx           # False
        self.nW = (input_resolution[0] // window_size)**2
        
        # if window size is larger than input resolution, 
        # we don't partition windows
        # 且window_size需要能够被input resolution整除，才能正确划分窗口
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # 构造注意力核心操作层
        self.encoder_attn = WindowAttention(
            embed_dim=embed_dim, 
            window_size=to_2tuple(self.window_size), 
            num_heads=num_heads,
            nW = self.nW
        )
        # dropout同时用于encoder_attn和ff_layer输出
        self.dropout = nn.Dropout(dropout) 
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        # 构造FeedForward层
        ffn_embed_dim = int(embed_dim * mlp_ratio)
        self.ff_layer = FeedForward(
            embed_dim = embed_dim, 
            ffn_embed_dim = ffn_embed_dim, 
            relu_dropout = dropout
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        # 此处mask为SW-MSA使用
        # [nW, w_s * w_s, w_s * w_s]
        # nW 为 window 数量，w_s 为 window_size
        # [4, 36, 36] 当input_resolution=(12, 12)，window_size=6时
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # [1, H, W, 1]
            # 对 [H, W] 大小进行分区
            # 分区的目的在于，shift之后，进行window划分时，一个window内包含多个区域，可能彼此不相临，需要进行标号区分
            # 数字相同表示在shift之前区域相邻
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            
            """
            # 也可以按如下分区
            # 数字相同表示在shift之前区域相邻
            h_slices = (slice(0, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.shift_size),
                        slice(-self.shift_size, None))
            """
            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1e9)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x, att_mask=None):
        # x: query / key / value  [B, L, C] 其中，L = H * W
        # x为grid特征，一个batch内每个样本特征数量一致，注意力计算时无需mask标注
        # att_mask 为 None 即可，不参与计算
        H, W = self.input_resolution
        B, L, C = x.shape
        short_cut = x
        
        # 如果使用全局特征，需要划分出全局特征和grid特征
        if self.use_gx:
            assert L == (H * W +1), "input feature has wrong size"
            gx = x[:, -1, :]   # [B, C]
            x  = x[:, :-1, :]  # [B, H * W, C]
        else:
            assert L == H * W, "input feature has wrong size"
            gx = None
            x = x

        x = x.view(B, H, W, C)

        # 循环移位，SW-MSA核心操作，W-MSA时不做处理
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # windows划分，比如 12x12 的区域被划分为4个 6 x 6 的windows
        # [B, H, W, C] --> [nW*B, window_size * window_size, C]
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        # 如果使用全局特征，需要嵌入 gx 到 x_windows 中，
        # 每个 window 内部的注意力机制加入了全局特征
        # 全局特征被复制了 nW 次，即窗口个数
        if self.use_gx:
            # [B, C] -> [B, 1, C] -> [B, nW, C] -> [B*nW, C]
            gx_ = gx.unsqueeze(1).repeat(1, x_windows.size()[0]//gx.size()[0], 1).view(x_windows.size()[0], -1)
            x_windows = torch.cat([x_windows, gx_.unsqueeze(1)], 1) # [B*nW, window_size*window_size + 1, C]
            
            # 对SW-MSA需要的mask进行扩充
            # 使用 torch.nn.functional.pad 填充
            if self.attn_mask is None:
                _mask = self.attn_mask
            else:
                _mask = F.pad(
                    self.attn_mask, 
                    pad=(0, 1, 0, 1, 0, 0),
                    mode='constant', 
                    value=0.0
                ) # [nW, window_size*window_size + 1, window_size*window_size + 1]
        else:
            _mask = self.attn_mask
            
        # W-MSA/SW-MSA
        # x_windows: [B*nW, Ww*Wh+1, C] 
        # mask: [nW, Ww*Wh+1, Ww*Wh+1,] or None
        attn_windows = self.encoder_attn(x_windows, mask=_mask)  # nW*B, window_size*window_size, C
        
        # 如果使用全局特征，需要从注意力机制输出（attn_windows）中拆分出 gx
        if self.use_gx:
            # 此处gx计算为4个windows内部的_gx的均值
            # 也可以在注意力机制内部进行处理（由注意力层的ind_gx参数控制
            # [B*nW, C] --> [B, nW, C] --> [B, C]
            gx = attn_windows[:, -1, :].view(-1, self.nW, 512).mean(1)
            # [B*nW, Ww*Wh, C]
            attn_windows = attn_windows[:, :-1, :]
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # 如果使用全局特征，需要再次合并gird特征和全局特征
        if self.use_gx:
            # [B, H*W+1, C]
            x = torch.cat([x, gx.unsqueeze(1)], dim=1)
        
        # 注意力后的残差
        x = self.dropout(x)
        x = self.layer_norm1(x + short_cut)
        
        # FeedForward及残差
        short_cut = x
        x = self.ff_layer(x)
        # dropout 残差 LayerNorm在此加入
        x = self.dropout(x)
        x = self.layer_norm2(x + short_cut)

        return x
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.embed_dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.encoder_attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.embed_dim * self.embed_dim * 4
        # norm2
        flops += self.embed_dim * H * W
        return flops
    
# 不包含残差连接和LayerNorm
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = nn.ReLU()    # ReLU / GELU / CELU
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, embed_dim=512, window_size=(12, 12), num_heads=8, 
                 nW=4, ind_gx=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.nW = nW
        self.ind_gx = ind_gx # 是否在注意力机制内部单独计算全局特征

        # 相对位置编码，用于grid特征的每一个window
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Wh-1 * 2*Ww-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.o_linear = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # N = Ww * Wh      仅window区域的grid特征，Ww和Wh为window_size
        # N = Ww * Wh + 1  window区域的grid特征加图像全局特征
        B_, N, C = x.size()
        """
        print('*'*30)
        print('raw gx', x[:, -1, :].min(), x[:, -1, :].max(), x[:, -1, :].mean())
        print('raw all', x.min(), x.max(), x.mean())
        # """
        
        # [B*nW, nH, N, C//nH]，其中nW为window数量，nH为num_heads
        q = self.q_linear(x).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        """
        print('gx q', q[:, :, -1, :].min(), q[:, :, -1, :].max(), q[:, :, -1, :].mean())
        print('gx k', k[:, :, -1, :].min(), k[:, :, -1, :].max(), k[:, :, -1, :].mean())
        print('gx v', v[:, :, -1, :].min(), v[:, :, -1, :].max(), v[:, :, -1, :].mean())
        print('all q', q.min(), q.max(), q.mean())
        print('all k', k.min(), k.max(), k.mean())
        print('all v', v.min(), v.max(), v.mean())
        # """
        
        # [B*nW, nH, N, N]，其中nW为window数量，nH为num_heads
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print(attn.min(), attn.max(), attn.mean())

        # 相对位置编码，仅window区域内的grid特征之间计算
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # [Wh*Ww, Wh*Ww, nH]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Wh*Ww, Wh*Ww]
        # 如果加入了全局特征，相对位置编码与全局特征无关
        if N == self.window_size[0] * self.window_size[1]:
            attn = attn + relative_position_bias.unsqueeze(0)
        else:
            # 仅对window区域的grid特征部分嵌入相对位置编码
            attn[:, :, :-1, :-1] = attn[:, :, :-1, :-1] + relative_position_bias.unsqueeze(0)
        
        """
        print(relative_position_bias.min(), relative_position_bias.max(), relative_position_bias.mean())
        print(attn.min(), attn.max(), attn.mean())
        # """
                    
        # 此处mask用于区分SW-MSA/W-MSA
        # mask: [nW, N, N]，
        # 其中nW为window数量，N=Ww*Wh or Ww*Wh+1，Ww和Wh为window_size
        if mask is not None:
            # mask = mask.masked_fill(mask == float(-100), float(-1e9))
            nW = mask.shape[0]
            # attn: [B*nW, nH, N, N] --> [B, nW, nH, N, N]
            # mask: [nW, N, N]       --> [1, nW,  1, N, N]
            # print('IN', attn.view(B_ // nW, nW, self.num_heads, N, N)[0, 2, 0, 0, :])
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # [B*nW, nH, N, N]
            attn = attn.view(-1, self.num_heads, N, N)
        else:
            attn = attn
        
        # TODO：代码精简
        # 单独处理全局特征，从attn中分离出全局特征权重，单独加权计算
        # [B*nW, nH, N]
        if self.ind_gx and N != self.window_size[0] * self.window_size[1]:
            # [B*nW, nH, N] -> [B, nW, nH, N] -> [B, nH, nW, N]
            gx_attn = attn[:, :, -1, :]
            gx_attn = gx_attn.view(B_ // self.nW, self.nW, self.num_heads, -1).permute(0, 2, 1, 3)
            # [B, nH, nW, N-1] --> [B, nH, nW * (N-1)] 即 [B, nH, H * W]
            gx_attn_1 = gx_attn[:, :, :, :-1].contiguous().view(B_ // self.nW, self.num_heads, -1)
            # [B, nH, nW, 1] --> [B, nH, 1]
            gx_attn_2 = gx_attn[:, :, :, -1:].mean(-2)
            # [B, nH, nW * (N-1) + 1] 即 [B, nH, H * W + 1]
            gx_attn = torch.cat([gx_attn_1, gx_attn_2], -1)
            # 全局特征权重 [B, nH, nW * (N-1) + 1] 即 [B, nH, H * W + 1]
            gx_attn = self.softmax(gx_attn)
            """
            if mask is None:
                # [B, 8, 145]
                # print(gx_attn.size())
                # print(gx_attn[0, 0, :-1].view(12, 12))
                print(gx_attn[:12, :, :].max(-1))
                if gx_attn[:, :, -1].max() > 0.001:
                    print('>>> gx alpha:', gx_attn[:, :, -1].max())
            # """
            # [B, nH, nW * (N-1)]
            gx_attn_1 = gx_attn[:, :, :-1] 
            # [B, nH, nW * (N-1)] --> [B, nH, nW, N-1] --> [B, nW, nH, N-1] --> [B*nW, nH, N-1]
            gx_attn_1 = gx_attn_1.view(B_ // self.nW, self.num_heads, self.nW , -1).permute(0, 2, 1, 3).contiguous().view(B_, self.num_heads, -1)
            # [B, nH, 1]
            gx_attn_2 = gx_attn[:, :, -1:] 
            # [B, nH, 1] --> [B, nH, nW, 1] --> [B, nW, nH, 1] --> [B*nW, nH, 1]
            gx_attn_2 = gx_attn_2.unsqueeze(-1).repeat(1, 1, self.nW, 1).permute(0, 2, 1, 3).contiguous().view(B_, self.num_heads, -1)
            # [B*nW, nH, N] --> [B*nW, nH, 1, N]
            gx_attn = torch.cat([gx_attn_1, gx_attn_2], -1).unsqueeze(-2)
            gx = (gx_attn @ v).transpose(1, 2).reshape(B_, C)
            # print(gx.size())
            gx = gx.view(B_ // self.nW, self.nW, -1).sum(1)
            # print(gx.size())
            gx = gx.unsqueeze(1).repeat(1, self.nW, 1).view(B_, C)
            # print(gx.size())
            
        # softmax计算权重
        attn = self.softmax(attn)
        """
        # [B*nW, 8, Ww*Wh, Ww*Wh]
        # print(attn.size())
        if attn[:12, :, :-1, -1].max() > 0.1:
            print(attn.size())
            print(attn[:12, :, 0, :].max(-1))
            # print(attn[:12, :, :-1, -1].max())
            # print(attn[:12, :, :-1, -1].max(-1))
            # print(attn[:8, :, :-1, -1].argmax(-1, keepdim=True))
        # """
        
        # 加权求和，[B*nW, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # 替换掉x中全局特征，[B*nW, C]
        if self.ind_gx and N != self.window_size[0] * self.window_size[1]:
            x[:, -1, :] = gx
        x = self.o_linear(x)
        return x
    
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.embed_dim * 3 * self.embed_dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.embed_dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.embed_dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.embed_dim * self.embed_dim
        return flops
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x