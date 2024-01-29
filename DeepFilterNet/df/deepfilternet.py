from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from df.config import DfParams, config
from df.modules import DfOp, GroupedGRU, GroupedLinear, Mask, convkxf, erb_fb, get_device
from libdf import DF


class ModelParams(DfParams):
    section = "deepfilternet"

    def __init__(self):
        super().__init__()
        self.conv_lookahead: int = config(
            "CONV_LOOKAHEAD", cast=int, default=0, section=self.section
        )
        self.conv_k_enc: int = config("CONV_K_ENC", cast=int, default=2, section=self.section)
        self.conv_k_dec: int = config("CONV_K_DEC", cast=int, default=1, section=self.section)
        self.conv_ch: int = config("CONV_CH", cast=int, default=16, section=self.section)
        self.conv_width_f: int = config(
            "CONV_WIDTH_FACTOR", cast=int, default=1, section=self.section
        )
        self.conv_dec_mode: str = config(
            "CONV_DEC_MODE", default="transposed", section=self.section
        )
        self.conv_depthwise: bool = config(
            "CONV_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.convt_depthwise: bool = config(
            "CONVT_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.emb_hidden_dim: int = config(
            "EMB_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.emb_num_layers: int = config(
            "EMB_NUM_LAYERS", cast=int, default=1, section=self.section
        )
        self.df_hidden_dim: int = config(
            "DF_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.df_num_layers: int = config("DF_NUM_LAYERS", cast=int, default=3, section=self.section)
        self.gru_groups: int = config("GRU_GROUPS", cast=int, default=1, section=self.section)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=self.section)
        self.group_shuffle: bool = config(
            "GROUP_SHUFFLE", cast=bool, default=True, section=self.section
        )
        self.dfop_method: str = config(
            "DFOP_METHOD", cast=str, default="real_unfold", section=self.section
        )
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)


def init_model(df_state: Optional[DF] = None, run_df: bool = True, train_mask: bool = True):
    p = ModelParams()
    if df_state is None:
        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    model = DfNet(erb_inverse, run_df, train_mask)
    return model.to(device=get_device())


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        wf = p.conv_width_f
        assert p.nb_erb % 4 == 0, "erb_bins should be divisible by 4"

        k = p.conv_k_enc
        kwargs = {"batch_norm": True, "depthwise": p.conv_depthwise}
        k0 = 1 if k == 1 and p.conv_lookahead == 0 else max(2, k)
        cl = 1 if p.conv_lookahead > 0 else 0
        self.erb_conv0 = convkxf(1, layer_width, k=k0, fstride=1, lookahead=cl, **kwargs)
        cl = 1 if p.conv_lookahead > 1 else 0
        self.erb_conv1 = convkxf(
            layer_width * wf**0, layer_width * wf**1, k=k, lookahead=cl, **kwargs
        )
        cl = 1 if p.conv_lookahead > 2 else 0
        self.erb_conv2 = convkxf(
            layer_width * wf**1, layer_width * wf**2, k=k, lookahead=cl, **kwargs
        )
        self.erb_conv3 = convkxf(layer_width * wf**2, layer_width * wf**2, k=k, fstride=1, **kwargs)
        self.df_conv0 = convkxf(
            2, layer_width, fstride=1, k=k0, lookahead=p.conv_lookahead, **kwargs
        )
        self.df_conv1 = convkxf(layer_width, layer_width * wf**1, k=k, **kwargs)
        self.erb_bins = p.nb_erb
        self.emb_dim = layer_width * p.nb_erb // 4 * wf**2
        self.df_fc_emb = GroupedLinear(
            layer_width * p.nb_df // 2, self.emb_dim, groups=p.lin_groups
        )
        self.emb_out_dim = p.emb_hidden_dim
        self.emb_n_layers = p.emb_num_layers
        self.gru_groups = p.gru_groups
        self.emb_gru = GroupedGRU(
            self.emb_dim,
            self.emb_out_dim,
            num_layers=p.emb_num_layers,
            batch_first=False,
            groups=p.gru_groups,
            shuffle=p.group_shuffle,
            add_outputs=True,
        )
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Encodes erb; erb should be in dB scale + normalized; Fe are number of erb bands.
        # erb: [B, 1, T, Fe]
        # spec: [B, 2, T, Fc]
        b, _, t, _ = feat_erb.shape
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
        e1 = self.erb_conv1(e0)  # [B, C*2, T, F/2]
        e2 = self.erb_conv2(e1)  # [B, C*4, T, F/4]
        e3 = self.erb_conv3(e2)  # [B, C*4, T, F/4]
        c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C*2, T, Fc]
        cemb = c1.permute(2, 0, 1, 3).reshape(t, b, -1)  # [T, B, C * Fc/4]
        cemb = self.df_fc_emb(cemb)  # [T, B, C * F/4]
        emb = e3.permute(2, 0, 1, 3).reshape(t, b, -1)  # [T, B, C * F/4]
        emb = emb + cemb
        emb, _ = self.emb_gru(emb)
        emb = emb.transpose(0, 1)  # [B, T, C * F/4]
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        wf = p.conv_width_f
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"

        self.emb_width = layer_width * wf**2
        self.emb_dim = self.emb_width * (p.nb_erb // 4)
        self.fc_emb = nn.Sequential(
            GroupedLinear(
                p.emb_hidden_dim, self.emb_dim, groups=p.lin_groups, shuffle=p.group_shuffle
            ),
            nn.ReLU(inplace=True),
        )
        k = p.conv_k_dec
        kwargs = {"k": k, "batch_norm": True, "depthwise": p.conv_depthwise}
        tkwargs = {
            "k": k,
            "batch_norm": True,
            "depthwise": p.convt_depthwise,
            "mode": p.conv_dec_mode,
        }
        pkwargs = {"k": 1, "f": 1, "batch_norm": True}
        # convt: TransposedConvolution, convp: Pathway (encoder to decoder) convolutions
        self.conv3p = convkxf(layer_width * wf**2, self.emb_width, **pkwargs)
        self.convt3 = convkxf(self.emb_width, layer_width * wf**2, fstride=1, **kwargs)
        self.conv2p = convkxf(layer_width * wf**2, layer_width * wf**2, **pkwargs)
        self.convt2 = convkxf(layer_width * wf**2, layer_width * wf**1, **tkwargs)
        self.conv1p = convkxf(layer_width * wf**1, layer_width * wf**1, **pkwargs)
        self.convt1 = convkxf(layer_width * wf**1, layer_width * wf**0, **tkwargs)
        self.conv0p = convkxf(layer_width, layer_width, **pkwargs)
        self.conv0_out = convkxf(layer_width, 1, fstride=1, k=k, act=nn.Sigmoid())

    def forward(self, emb, e3, e2, e1, e0) -> Tensor:
        # Estimates erb mask
        b, _, t, f8 = e3.shape
        emb = self.fc_emb(emb)
        emb = emb.view(b, t, -1, f8).transpose(1, 2)  # [B, C*8, T, F/8]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C*4, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C*2, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        return m


class DfDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        self.emb_dim = p.emb_hidden_dim

        self.df_n_hidden = p.df_hidden_dim
        self.df_n_layers = p.df_num_layers
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_lookahead = p.df_lookahead
        self.gru_groups = p.gru_groups

        self.df_convp = convkxf(
            layer_width, self.df_order * 2, k=1, f=1, complex_in=True, batch_norm=True
        )
        self.df_gru = GroupedGRU(
            p.emb_hidden_dim,
            self.df_n_hidden,
            num_layers=self.df_n_layers,
            batch_first=False,
            groups=p.gru_groups,
            shuffle=p.group_shuffle,
            add_outputs=True,
        )
        self.df_fc_out = nn.Sequential(
            nn.Linear(self.df_n_hidden, self.df_bins * self.df_order * 2), nn.Tanh()
        )
        self.df_fc_a = nn.Sequential(nn.Linear(self.df_n_hidden, 1), nn.Sigmoid())

    def forward(self, emb: Tensor, c0: Tensor) -> Tuple[Tensor, Tensor]:
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb.transpose(0, 1))  # [T, B, H], H: df_n_hidden
        c0 = self.df_convp(c0).transpose(1, 2)  # [B, T, O*2, F]
        c = c.transpose(0, 1)  # [B, T, H]
        alpha = self.df_fc_a(c)  # [B, T, 1]
        c = self.df_fc_out(c)  # [B, T, F*O*2], O: df_order
        c = c.view(b, t, self.df_order * 2, self.df_bins)  # [B, T, O*2, F]
        c = c.add(c0).view(b, t, self.df_order, 2, self.df_bins).transpose(3, 4)  # [B, T, O, F, 2]
        return c, alpha


class DfNet(nn.Module):
    def __init__(
        self,
        erb_inv_fb: Tensor,
        run_df: bool = True,
        train_mask: bool = True,
    ):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.freq_bins = p.fft_size // 2 + 1
        self.emb_dim = layer_width * p.nb_erb
        self.erb_bins = p.nb_erb
        self.enc = Encoder()
        self.erb_dec = ErbDecoder()
        self.mask = Mask(erb_inv_fb, post_filter=p.mask_pf)

        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_lookahead = p.df_lookahead
        self.df_dec = DfDecoder()
        self.df_op = torch.jit.script(
            DfOp(
                p.nb_df,
                p.df_order,
                p.df_lookahead,
                freq_bins=self.freq_bins,
                method=p.dfop_method,
            )
        )

        self.run_df = run_df
        if not run_df:
            from loguru import logger

            logger.warning("Running without DF")
        self.train_mask = train_mask

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,  # Not used, take spec modified by mask instead
        atten_lim: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feat_spec = feat_spec.transpose(1, 4).squeeze(4)  # re/im into channel axis
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)
        spec = self.mask(spec, m, atten_lim)
        if self.run_df:
            df_coefs, df_alpha = self.df_dec(emb, c0)
            spec = self.df_op(spec, df_coefs, df_alpha)
        else:
            df_alpha = torch.zeros(spec.shape[0], spec.shape[2], 1, device=spec.device)
        return spec, m, lsnr, df_alpha
