from functools import partial
from typing import Final, List, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor, nn

import df.multiframe as MF
from df.config import Csv, DfParams, config
from df.modules import (
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    GroupedLinearEinsum,
    Mask,
    SqueezedGRU_S,
    erb_fb,
    get_device,
)
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
        self.emb_gru_skip: str = config("EMB_GRU_SKIP", default="none", section=self.section)
        self.df_hidden_dim: int = config(
            "DF_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.df_gru_skip: str = config("DF_GRU_SKIP", default="none", section=self.section)
        self.df_pathway_kernel_size_t: int = config(
            "DF_PATHWAY_KERNEL_SIZE_T", cast=int, default=1, section=self.section
        )
        self.enc_concat: bool = config("ENC_CONCAT", cast=bool, default=False, section=self.section)
        self.df_num_layers: int = config("DF_NUM_LAYERS", cast=int, default=3, section=self.section)
        self.df_n_iter: int = config("DF_N_ITER", cast=int, default=1, section=self.section)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=self.section)
        self.enc_lin_groups: int = config(
            "ENC_LINEAR_GROUPS", cast=int, default=16, section=self.section
        )
        self.mfop_method: str = config(
            "MFOP_METHOD", cast=str, default="WF", section=self.section
        ).upper()
        self.mf_est_inverse: bool = config(
            "MF_ESTIMATE_INVERSE", cast=bool, default=True, section=self.section
        )
        self.mf_use_cholesky_decomp: bool = config(
            "MF_USE_CHOLESKY_DECOMP", cast=bool, default=False, section=self.section
        )
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
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4
        df_fc_emb = GroupedLinearEinsum(
            p.conv_ch * p.nb_df // 2, self.emb_in_dim, groups=p.enc_lin_groups
        )
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))
        if p.enc_concat:
            self.emb_in_dim *= 2
            self.combine = Concat()
        else:
            self.combine = Add()
        self.emb_n_layers = p.emb_num_layers
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=1,
            batch_first=True,
            gru_skip_op=None,
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

        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4

        if p.emb_gru_skip == "none":
            skip_op = None
        elif p.emb_gru_skip == "identity":
            assert self.emb_in_dim == self.emb_out_dim, "Dimensions do not match"
            skip_op = partial(nn.Identity)
        elif p.emb_gru_skip == "groupedlinear":
            skip_op = partial(
                GroupedLinearEinsum,
                input_size=self.emb_in_dim,
                hidden_size=self.emb_out_dim,
                groups=p.lin_groups,
            )
        else:
            raise NotImplementedError()
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=p.emb_num_layers - 1,
            batch_first=True,
            gru_skip_op=skip_op,
            linear_groups=p.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
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
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)  # [B, C*8, T, F/8]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C*4, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C*2, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        return m


class DfDecoder(nn.Module):
    """This model predicts the inverse noise/noisy covariance matrix and the speech intra-frame
    correlation (IFC) vector for usage in an MfWf or MfMvdr filter."""

    def __init__(self):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch

        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.df_hidden_dim

        self.df_n_hidden = p.df_hidden_dim
        self.df_n_layers = p.df_num_layers
        self.df_order = p.df_order
        self.df_bins = p.nb_df

        conv_layer = partial(Conv2dNormAct, separable=True, bias=False)
        kt = p.df_pathway_kernel_size_t
        self.cov_convp = conv_layer(layer_width, p.df_order**2 * 2, fstride=1, kernel_size=(kt, 1))
        self.ifc_convp = conv_layer(layer_width, p.df_order * 2, fstride=1, kernel_size=(kt, 1))

        self.df_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            num_layers=self.df_n_layers,
            batch_first=True,
            gru_skip_op=None,
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
            self.df_skip = GroupedLinearEinsum(self.emb_in_dim, self.emb_dim, groups=p.lin_groups)
        else:
            raise NotImplementedError()
        cov_out_dim = self.df_bins * p.df_order**2 * 2
        ifc_out_dim = self.df_bins * p.df_order * 2
        self.cov_out = GroupedLinearEinsum(self.df_n_hidden, cov_out_dim, groups=p.lin_groups)
        self.ifc_out = GroupedLinearEinsum(self.df_n_hidden, ifc_out_dim, groups=p.lin_groups)

    def forward(self, emb: Tensor, c0: Tensor) -> Tuple[Tensor, Tensor]:
        b, t, _ = emb.shape
        c, _ = self.df_gru(emb)  # [B, T, H], H: df_n_hidden
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        c0_ifc = self.ifc_convp(c0).permute(0, 2, 3, 1)  # [B, T, F, O*2], channels_last
        c0_cov = self.cov_convp(c0).permute(0, 2, 3, 1)  # [B, T, F, O**2*2], channels_last
        c_ifc = self.ifc_out(c)  # [B, T, F*O*2], O: df_order
        c_cov = self.cov_out(c)  # [B, T, F*O**2*2], O: df_order
        ifc = c_ifc.view(b, t, self.df_bins, self.df_order * 2) + c0_ifc  # [B, T, F, O*2]
        cov = c_cov.view(b, t, self.df_bins, self.df_order**2 * 2) + c0_cov  # [B, T, F, O**2*2]
        return ifc, cov


class DfNet(nn.Module):
    run_mff: Final[bool]

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
        self.df_lookahead = p.df_lookahead
        self.nb_df = p.nb_df
        self.freq_bins: int = p.fft_size // 2 + 1
        self.emb_dim: int = layer_width * p.nb_erb
        self.erb_bins: int = p.nb_erb
        if p.conv_lookahead > 0:
            assert p.conv_lookahead >= p.df_lookahead
            self.pad_feat = nn.ConstantPad2d((0, 0, -p.conv_lookahead, p.conv_lookahead), 0.0)
        else:
            self.pad_feat = nn.Identity()
        if p.df_lookahead > 0:
            self.pad_spec = nn.ConstantPad3d((0, 0, 0, 0, -p.df_lookahead, p.df_lookahead), 0.0)
        else:
            self.pad_spec = nn.Identity()
        self.register_buffer("erb_fb", erb_fb)
        self.enc = Encoder()
        self.erb_dec = ErbDecoder()
        self.mask = Mask(erb_inv_fb, post_filter=p.mask_pf)

        self.df_order = p.df_order
        assert p.mfop_method in ("WF", "MVDR")
        cholesky_decomp = p.mf_use_cholesky_decomp
        inverse = p.mf_est_inverse
        if p.mfop_method.startswith("WF"):
            self.mf_op = MF.MfWf(
                num_freqs=p.nb_df,
                frame_size=p.df_order,
                lookahead=self.df_lookahead,
                cholesky_decomp=cholesky_decomp,
                inverse=inverse,
            )
        elif p.mfop_method.startswith("MVDR"):
            self.mf_op = MF.MfMvdr(
                num_freqs=p.nb_df,
                frame_size=p.df_order,
                lookahead=self.df_lookahead,
                cholesky_decomp=cholesky_decomp,
                inverse=inverse,
            )
        else:
            raise NotImplementedError(p.mfop_method)
        self.df_dec = DfDecoder()

        self.run_mff = run_df
        if not run_df:
            logger.warning("Running without multi-frame filtering")
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
            feat_spec (Tensor): Complex spectrogram features of shape [B, 1, T, F', 2]

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

        spec_m = self.mask(spec, m)

        if self.run_mff:
            # ifc: speech intra-frame correlation vector
            # cov: correlation matrix of either noise (MVDR), or noisy (WF) signal
            ifc, cov = self.df_dec(emb, c0)
            spec = self.mf_op(spec, ifc, cov)
            spec[..., self.nb_df :, :] = spec_m[..., self.nb_df :, :]
            coefs = torch.cat((ifc, cov), dim=-1)
        else:
            coefs = torch.zeros((), device=spec.device)
            spec = spec_m

        return spec, m, lsnr, coefs
