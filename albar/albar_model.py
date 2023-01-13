import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class AlbarModel(nn.Module):
    def __init__(self, config, albar_range=1, log_fn=print):
        super().__init__()
        self.config = config
        self.albar_range = albar_range
        self.log_fn = log_fn
        ch, ch_mult = config.model.ch, tuple(config.model.ch_mult)
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        out_ch = config.model.out_ch
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch*4
        self.ch_mult_len = len(ch_mult)
        self.num_res_blocks = config.model.num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        # The "dense" below is not a property; it's just an arbitrary name.
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution       # 32
        in_ch_mult = (1,)+ch_mult   # [1, 1, 2, 2, 2]
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.ch_mult_len):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:  # attn_resolutions: [16, ]
                    attn.append(AttnBlock(block_in))
            # for i_block
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.ch_mult_len-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        # for i_level

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.ch_mult_len)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def get_albar_embedding(self, albar):
        """
        Get alpha_bar embedding
        """
        assert len(albar.shape) == 1

        embedding_dim = self.ch
        bs = len(albar)  # batch size
        half_dim = embedding_dim // 2
        emb1 = torch.ones((bs, half_dim), dtype=torch.float32)
        emb1 = emb1.to(device=albar.device)
        emb1 *= albar.sqrt().view(-1, 1)
        emb2 = torch.ones((bs, embedding_dim - half_dim), dtype=torch.float32)
        emb2 = emb2.to(device=albar.device)
        emb2 *= (1 - albar).sqrt().view(-1, 1)
        emb = torch.cat([emb1, emb2], dim=1)
        if self.albar_range == 1: # range: [0, 1]
            if not hasattr(self, '_flag1'):
                setattr(self, '_flag1', True)
                self.log_fn(f"AlbarModel::self.albar_range == {self.albar_range}. Just keep old range.")
        else:
            emb = emb * 2 - 1   # range: [-1, 1]
            if not hasattr(self, '_flag1'):
                setattr(self, '_flag1', True)
                self.log_fn(f"AlbarModel::self.albar_range == {self.albar_range}. Adjust to new range.")
        return emb

    def forward(self, x, albar):
        assert x.shape[2] == x.shape[3] == self.resolution  # config.data.image_size

        # alpha_bar embedding.
        # self.ch: config.model.ch. Usually 128
        ab_emb = self.get_albar_embedding(albar)        # shape [250, 128]  batch size is 250
        ab_emb = self.temb.dense[0](ab_emb)             # shape [250, 512]
        ab_emb = nonlinearity(ab_emb)                   # shape [250, 512]
        ab_emb = self.temb.dense[1](ab_emb)             # shape [250, 512]

        # down-sampling
        hs = [self.conv_in(x)]
        for i_level in range(self.ch_mult_len):     # ch_mult = [1, 2, 2, 2]
            for i_block in range(self.num_res_blocks):  # num_res_blocks = 2
                h = self.down[i_level].block[i_block](hs[-1], ab_emb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.ch_mult_len-1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # after each level, the len(hs):
        #   i_level: len(hs)
        #        0 : 4
        #        1 : 7
        #        2 : 10
        #        3 : 12
        # Shape of each element in hs, from hs[0] to hs[11]
        #   [250, 128, 32, 32]
        #   [250, 128, 32, 32]
        #   [250, 128, 32, 32]
        #   [250, 128, 16, 16]
        #   [250, 256, 16, 16]
        #   [250, 256, 16, 16]
        #   [250, 256, 8,  8 ]
        #   [250, 256, 8,  8 ]
        #   [250, 256, 8,  8 ]
        #   [250, 256, 4,  4 ]
        #   [250, 256, 4,  4 ]
        #   [250, 256, 4,  4 ]

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, ab_emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, ab_emb)

        # up-sampling
        for i_level in reversed(range(self.ch_mult_len)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), ab_emb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # shape of h during up-sampling:
        #   [250, 256, 4, 4]
        #   [250, 256, 4, 4]
        #   [250, 256, 4, 4]
        #   [250, 256, 8, 8] ------ after self.up[i_level].upsample(h)
        #   [250, 256, 8, 8]
        #   [250, 256, 8, 8]
        #   [250, 256, 8, 8]
        #   [250, 256, 16, 16] ------ after self.up[i_level].upsample(h)
        #   [250, 256, 16, 16]
        #   [250, 256, 16, 16]
        #   [250, 256, 16, 16]
        #   [250, 256, 32, 32] ------ after self.up[i_level].upsample(h)
        #   [250, 128, 32, 32]
        #   [250, 128, 32, 32]
        #   [250, 128, 32, 32]

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
