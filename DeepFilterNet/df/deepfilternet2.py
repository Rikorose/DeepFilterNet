from functools import partial
from typing import Final, List, Optional, Tuple, Union

import torch
from loguru import logger
from torch import Tensor, nn

from df.config import Csv, DfParams, config
from df.modules import (
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    DfOp,
    GroupedGRU,
    GroupedLinear,
    GroupedLinearEinsum,
    Mask,
    SqueezedGRU,
    erb_fb,
    get_device,
)
from df.multiframe import MF_METHODS, MultiFrameModule
from libdf import DF


class ModelParams(DfParams):
    section = "deepfilternet"

    def __init__(self):
        super().__init__()
        self.conv_lookahead: int = config(
            "CONV_LOOKAHEAD", cast=int, default=0, section=self.section
        )
        self.conv_ch: int = config("CONV_CH", cast=int, default=16, section=self.section)
        self.conv_depthwise: bool = config(
            "CONV_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.convt_depthwise: bool = config(
            "CONVT_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.conv_kernel: List[int] = config(
            "CONV_KERNEL", cast=Csv(int), default=(1, 3), section=self.section  # type: ignore
        )
        self.conv_kernel_inp: List[int] = config(
            "CONV_KERNEL_INP", cast=Csv(int), default=(3, 3), section=self.section  # type: ignore
        )
        self.emb_hidden_dim: int = config(
            "EMB_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.emb_num_layers: int = config(
            "EMB_NUM_LAYERS", cast=int, default=2, section=self.section
        )
        self.df_hidden_dim: int = config(
            "DF_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.df_gru_skip: str = config("DF_GRU_SKIP", default="none", section=self.section)
        self.df_output_layer: str = config(
            "DF_OUTPUT_LAYER", default="linear", section=self.section
        )
        self.df_pathway_kernel_size_t: int = config(
            "DF_PATHWAY_KERNEL_SIZE_T", cast=int, default=1, section=self.section
        )
        self.enc_concat: bool = config("ENC_CONCAT", cast=bool, default=False, section=self.section)
        self.df_num_layers: int = config("DF_NUM_LAYERS", cast=int, default=3, section=self.section)
        self.df_n_iter: int = config("DF_N_ITER", cast=int, default=2, section=self.section)
        self.gru_type: str = config("GRU_TYPE", default="grouped", section=self.section)
        self.gru_groups: int = config("GRU_GROUPS", cast=int, default=1, section=self.section)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=self.section)
        self.group_shuffle: bool = config(
            "GROUP_SHUFFLE", cast=bool, default=True, section=self.section
        )
        self.dfop_method: str = config("DFOP_METHOD", cast=str, default="df", section=self.section)
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)


def init_model(df_state: Optional[DF] = None, run_df: bool = True, train_mask: bool = True):
    p = ModelParams()
    if df_state is None:
        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    model = DfNet(erb, erb_inverse, run_df, train_mask)
    return model.to(device=get_device())


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class Concat(nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=-1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        assert p.nb_erb % 4 == 0, "erb_bins should be divisible by 4"

        self.erb_conv0 = Conv2dNormAct(
            1, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True
        )
        conv_layer = partial(
            Conv2dNormAct,
            in_ch=p.conv_ch,
            out_ch=p.conv_ch,
            kernel_size=p.conv_kernel,
            bias=False,
            separable=True,
        )
        self.erb_conv1 = conv_layer(fstride=2)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=1)
        self.df_conv0 = Conv2dNormAct(
            2, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True
        )
        self.df_conv1 = conv_layer(fstride=2)
        self.erb_bins = p.nb_erb
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_out_dim = p.emb_hidden_dim
        if p.gru_type == "grouped":
            self.df_fc_emb = GroupedLinear(
                p.conv_ch * p.nb_df // 2, self.emb_in_dim, groups=p.lin_groups
            )
        else:
            df_fc_emb = GroupedLinearEinsum(
                p.conv_ch * p.nb_df // 2, self.emb_in_dim, groups=p.lin_groups
            )
            self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))
        if p.enc_concat:
            self.emb_in_dim *= 2
            self.combine = Concat()
        else:
            self.combine = Add()
        self.emb_out_dim = p.emb_hidden_dim
        self.emb_n_layers = p.emb_num_layers
        assert p.gru_type in ("grouped", "squeeze"), f"But got {p.gru_type}"
        if p.gru_type == "grouped":
            self.emb_gru = GroupedGRU(
                self.emb_in_dim,
                self.emb_out_dim,
                num_layers=1,
                batch_first=True,
                groups=p.gru_groups,
                shuffle=p.group_shuffle,
                add_outputs=True,
            )
        else:
            self.emb_gru = SqueezedGRU(
                self.emb_in_dim,
                self.emb_out_dim,
                num_layers=1,
                batch_first=True,
                linear_groups=p.lin_groups,
                linear_act_layer=partial(nn.ReLU, inplace=True),
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
        # b, _, t, _ = feat_erb.shape
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
        e1 = self.erb_conv1(e0)  # [B, C*2, T, F/2]
        e2 = self.erb_conv2(e1)  # [B, C*4, T, F/4]
        e3 = self.erb_conv3(e2)  # [B, C*4, T, F/4]
        c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C*2, T, Fc]
        cemb = c1.permute(0, 2, 3, 1).flatten(2)  # [B, T, -1]
        cemb = self.df_fc_emb(cemb)  # [T, B, C * F/4]
        emb = e3.permute(0, 2, 3, 1).flatten(2)  # [B, T, C * F/4]
        emb = self.combine(emb, cemb)
        emb, _ = self.emb_gru(emb)  # [B, T, -1]
        lsnr = self.lsnr_fc(emb) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr


class ErbDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"

        self.emb_out_dim = p.emb_hidden_dim

        if p.gru_type == "grouped":
            self.emb_gru = GroupedGRU(
                p.conv_ch * p.nb_erb // 4,  # For compat
                self.emb_out_dim,
                num_layers=p.emb_num_layers - 1,
                batch_first=True,
                groups=p.gru_groups,
                shuffle=p.group_shuffle,
                add_outputs=True,
            )
            # SqueezedGRU uses GroupedLinearEinsum, so let's use it here as well
            fc_emb = GroupedLinear(
                p.emb_hidden_dim,
                p.conv_ch * p.nb_erb // 4,
                groups=p.lin_groups,
                shuffle=p.group_shuffle,
            )
            self.fc_emb = nn.Sequential(fc_emb, nn.ReLU(inplace=True))
        else:
            self.emb_gru = SqueezedGRU(
                self.emb_out_dim,
                self.emb_out_dim,
                output_size=p.conv_ch * p.nb_erb // 4,
                num_layers=p.emb_num_layers - 1,
                batch_first=True,
                gru_skip_op=nn.Identity,
                linear_groups=p.lin_groups,
                linear_act_layer=partial(nn.ReLU, inplace=True),
            )
            self.fc_emb = nn.Identity()
        tconv_layer = partial(
            ConvTranspose2dNormAct,
            kernel_size=p.conv_kernel,
            bias=False,
            separable=True,
        )
        conv_layer = partial(
            Conv2dNormAct,
            bias=False,
            separable=True,
        )
        # convt: TransposedConvolution, convp: Pathway (encoder to decoder) convolutions
        self.conv3p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt3 = conv_layer(p.conv_ch, p.conv_ch, kernel_size=p.conv_kernel)
        self.conv2p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv1p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv0p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(
            p.conv_ch, 1, kernel_size=p.conv_kernel, activation_layer=nn.Sigmoid
        )

    def forward(self, emb, e3, e2, e1, e0) -> Tensor:
        # Estimates erb mask
        b, _, t, f8 = e3.shape
        emb, _ = self.emb_gru(emb)
        emb = self.fc_emb(emb)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)  # [B, C*8, T, F/8]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C*4, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C*2, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        return m


class DfOutputReshapeMF(nn.Module):
    """Coefficients output reshape for multiframe/MultiFrameModule

    Requires input of shape B, C, T, F, 2.
    """

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        coefs = coefs.view(*coefs.shape[:-1], -1, 2)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs


class DfDecoder(nn.Module):
    def __init__(self, out_channels: int = -1):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        self.emb_dim = p.emb_hidden_dim

        self.df_n_hidden = p.df_hidden_dim
        self.df_n_layers = p.df_num_layers
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.gru_groups = p.gru_groups
        self.df_out_ch = out_channels if out_channels > 0 else p.df_order * 2

        conv_layer = partial(Conv2dNormAct, separable=True, bias=False)
        kt = p.df_pathway_kernel_size_t
        self.df_convp = conv_layer(layer_width, self.df_out_ch, fstride=1, kernel_size=(kt, 1))
        if p.gru_type == "grouped":
            self.df_gru = GroupedGRU(
                p.emb_hidden_dim,
                p.df_hidden_dim,
                num_layers=self.df_n_layers,
                batch_first=True,
                groups=p.gru_groups,
                shuffle=p.group_shuffle,
                add_outputs=True,
            )
        else:
            self.df_gru = SqueezedGRU(
                p.emb_hidden_dim,
                p.df_hidden_dim,
                num_layers=self.df_n_layers,
                batch_first=True,
                gru_skip_op=nn.Identity,
                linear_act_layer=partial(nn.ReLU, inplace=True),
            )
        p.df_gru_skip = p.df_gru_skip.lower()
        assert p.df_gru_skip in ("none", "identity", "groupedlinear")
        self.df_skip: Optional[nn.Module]
        if p.df_gru_skip == "none":
            self.df_skip = None
        elif p.df_gru_skip == "identity":
            assert p.emb_hidden_dim == p.df_hidden_dim, "Dimensions do not match"
            self.df_skip = nn.Identity()
        elif p.df_gru_skip == "groupedlinear":
            self.df_skip = GroupedLinearEinsum(
                p.emb_hidden_dim, p.df_hidden_dim, groups=p.lin_groups
            )
        else:
            raise NotImplementedError()
        assert p.df_output_layer in ("linear", "groupedlinear")
        self.df_out: nn.Module
        out_dim = self.df_bins * self.df_out_ch
        if p.df_output_layer == "linear":
            df_out = nn.Linear(self.df_n_hidden, out_dim)
        elif p.df_output_layer == "groupedlinear":
            df_out = GroupedLinearEinsum(self.df_n_hidden, out_dim, groups=p.lin_groups)
        else:
            raise NotImplementedError
        self.df_out = nn.Sequential(df_out, nn.Tanh())
        self.df_fc_a = nn.Sequential(nn.Linear(self.df_n_hidden, 1), nn.Sigmoid())
        self.out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def forward(self, emb: Tensor, c0: Tensor) -> Tuple[Tensor, Tensor]:
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)  # [B, T, H], H: df_n_hidden
        if self.df_skip is not None:
            c += self.df_skip(emb)
        c0 = self.df_convp(c0).permute(0, 2, 3, 1)  # [B, T, F, O*2], channels_last
        alpha = self.df_fc_a(c)  # [B, T, 1]
        c = self.df_out(c)  # [B, T, F*O*2], O: df_order
        c = c.view(b, t, self.df_bins, self.df_out_ch) + c0  # [B, T, F, O*2]
        c = self.out_transform(c)
        return c, alpha


class DfNet(nn.Module):
    run_df: Final[bool]
    pad_specf: Final[bool]

    def __init__(
        self,
        erb_fb: Tensor,
        erb_inv_fb: Tensor,
        run_df: bool = True,
        train_mask: bool = True,
    ):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.df_lookahead = p.df_lookahead if p.pad_mode == "model" else 0
        self.nb_df = p.nb_df
        self.freq_bins: int = p.fft_size // 2 + 1
        self.emb_dim: int = layer_width * p.nb_erb
        self.erb_bins: int = p.nb_erb
        if p.conv_lookahead > 0 and p.pad_mode.startswith("input"):
            self.pad_feat = nn.ConstantPad2d((0, 0, -p.conv_lookahead, p.conv_lookahead), 0.0)
        else:
            self.pad_feat = nn.Identity()
        self.pad_specf = p.pad_mode.endswith("specf")
        if p.df_lookahead > 0 and self.pad_specf:
            self.pad_spec = nn.ConstantPad3d((0, 0, 0, 0, -p.df_lookahead, p.df_lookahead), 0.0)
        else:
            self.pad_spec = nn.Identity()
        if (p.conv_lookahead > 0 or p.df_lookahead > 0) and p.pad_mode.startswith("output"):
            assert p.conv_lookahead == p.df_lookahead
            pad = (0, 0, 0, 0, -p.conv_lookahead, p.conv_lookahead)
            self.pad_out = nn.ConstantPad3d(pad, 0.0)
        else:
            self.pad_out = nn.Identity()
        self.register_buffer("erb_fb", erb_fb)
        self.enc = Encoder()
        self.erb_dec = ErbDecoder()
        self.mask = Mask(erb_inv_fb, post_filter=p.mask_pf)

        self.df_order = p.df_order
        self.df_op: Union[DfOp, MultiFrameModule]
        if p.dfop_method == "real_unfold":
            raise ValueError("RealUnfold DF OP is now unsupported.")
        assert p.df_output_layer != "linear", "Must be used with `groupedlinear`"
        self.df_op = MF_METHODS[p.dfop_method](
            num_freqs=p.nb_df, frame_size=p.df_order, lookahead=self.df_lookahead
        )
        n_ch_out = self.df_op.num_channels()
        self.df_dec = DfDecoder(out_channels=n_ch_out)

        self.run_df = run_df
        if not run_df:
            logger.warning("Runing without DF")
        self.train_mask = train_mask
        assert p.df_n_iter == 1

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,  # Not used, take spec modified by mask instead
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward method of DeepFilterNet2.

        Args:
            spec (Tensor): Spectrum of shape [B, 1, T, F, 2]
            feat_erb (Tensor): ERB features of shape [B, 1, T, E]
            feat_spec (Tensor): Complex spectrogram features of shape [B, 1, T, F']

        Returns:
            spec (Tensor): Enhanced spectrum of shape [B, 1, T, F, 2]
            m (Tensor): ERB mask estimate of shape [B, 1, T, E]
            lsnr (Tensor): Local SNR estimate of shape [B, T, 1]
        """
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)

        feat_erb = self.pad_feat(feat_erb)
        feat_spec = self.pad_feat(feat_spec)
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)

        m = self.pad_out(m.unsqueeze(-1)).squeeze(-1)
        spec = self.mask(spec, m)

        if self.run_df:
            df_coefs, df_alpha = self.df_dec(emb, c0)
            df_coefs = self.pad_out(df_coefs)

            if self.pad_specf:
                # Only pad the lower part of the spectrum.
                spec_f = self.pad_spec(spec)
                spec_f = self.df_op(spec_f, df_coefs)
                spec[..., : self.nb_df, :] = spec_f[..., : self.nb_df, :]
            else:
                spec = self.pad_spec(spec)
                spec = self.df_op(spec, df_coefs)
        else:
            df_alpha = torch.zeros(())

        return spec, m, lsnr, df_alpha
