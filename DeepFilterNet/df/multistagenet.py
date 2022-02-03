from typing import List, Optional, Tuple

import torch
from icecream import ic
from torch import Tensor, nn

from df.config import DfParams, config
from df.modules import Mask, PreNormShortcut, erb_fb
from df.utils import get_device
from libdf import DF


class Conv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int],
        lookahead: int = 0,
        stride: int = 1,
        dilation: int = 1,
    ):
        """Causal Conv2d by delaying the signal for any lookahead.

        Expected input format: [B, C, T, F]
        """
        super().__init__()
        # Padding on time axis
        fpad = kernel_size[1] // 2 + dilation - 1
        self.pad = nn.ConstantPad2d((0, 0, lookahead, kernel_size[0] - 1 - lookahead), 0.0)
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=(0, fpad),
            stride=(1, stride),  # Stride over time is always 1
            dilation=(1, dilation),  # Same for dilation
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = self.conv(x)
        return x


class ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        dilation: int = 1,
    ):
        """Causal ConvTranspose2d.

        Expected input format: [B, C, T, F]
        """
        super().__init__()
        # Padding on time axis, with lookahead = 0
        fpad = kernel_size[1] // 2
        self.pad = nn.ConstantPad2d((0, 0, 0, kernel_size[0] - 1), 0.0)
        self.conv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=(kernel_size[0] - 1, fpad + dilation - 1),
            output_padding=(0, fpad),
            stride=(1, stride),  # Stride over time is always 1
            dilation=(1, dilation),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = self.conv(x)
        return x


class GRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs["batch_first"] = True
        self.gru = nn.GRU(*args, **kwargs)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """GRU transposing [B, C, T, F] input shape to [B, T, C*F]."""
        _, c, _, f = x.shape
        x, h = self.gru(x.transpose(1, 2).flatten(2), h)
        x = x.unflatten(2, (-1, f)).transpose(1, 2)
        return x, h


class ModelParams(DfParams):
    section = "multistagenet"

    def __init__(self):
        super().__init__()
        self.n_stages: int = config("N_STAGES", cast=int, default=4, section=self.section)
        self.conv_lookahead: int = config(
            "CONV_LOOKAHEAD", cast=int, default=0, section=self.section
        )
        self.conv_ch: int = config("CONV_CH", cast=int, default=16, section=self.section)
        self.erb_hidden_dim: int = config(
            "ERB_HIDDEN_DIM", cast=int, default=64, section=self.section
        )
        self.refinement_hidden_dim: int = config(
            "REFINEMENT_HIDDEN_DIM", cast=int, default=96, section=self.section
        )
        self.refinement_act: str = config(
            "REFINEMENT_OUTPUT_ACT", default="tanh", section=self.section
        )
        self.gru_groups: int = config("GRU_GROUPS", cast=int, default=1, section=self.section)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=self.section)
        self.group_shuffle: bool = config(
            "GROUP_SHUFFLE", cast=bool, default=False, section=self.section
        )
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)


class ErbStage(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        self.lw = p.conv_ch  # Layer width
        self.fe = p.nb_erb  # Number of frequency bins in embedding
        self.hd = p.erb_hidden_dim
        if p.erb_hidden_dim % (p.nb_erb // 2) != 0:
            raise ValueError("`erb_hidden_dim` must be dividable by `nb_erb/2`")
        self.conv0 = nn.Sequential(
            nn.LayerNorm(self.fe), Conv2d(1, self.lw, (3, 3), lookahead=p.conv_lookahead), nn.GELU()
        )
        self.conv1 = PreNormShortcut(
            self.fe,
            nn.Sequential(Conv2d(self.lw, self.lw, (2, 3), lookahead=0, stride=2), nn.GELU()),
            Conv2d(self.lw, self.lw, (1, 1), lookahead=0, stride=2),
        )
        self.gru = PreNormShortcut(
            self.fe // 2,
            GRU(self.fe // 2 * self.lw, self.hd),
            shortcut=nn.Conv2d(self.lw, self.hd // (self.fe // 2), 1),
        )
        self.convt1 = nn.Sequential(
            nn.LayerNorm(self.fe // 2),
            ConvTranspose2d(self.hd // (self.fe // 2), self.lw, (2, 3), stride=2),
            nn.GELU(),
            nn.Conv2d(self.lw, self.lw, 1),
        )
        self.conv_out = nn.Sequential(
            nn.LayerNorm(self.fe), Conv2d(self.lw, 1, (2, 3)), nn.Sigmoid()
        )
        self.gru_snr = nn.GRU(self.hd, 16)
        self.fc_snr = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def forward(
        self, input: Tensor, h: Optional[Tensor] = None, h_snr: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # input shape: [B, 1, T, F]
        x0 = self.conv0(input)  # [B, C, T, F]
        x1 = self.conv1(x0)  # [B, C, T, F/2]
        x_rnn, h = self.gru(x1, h)
        x1 = self.convt1(x_rnn) + x0
        lsnr, h_snr = self.gru_snr(x_rnn.transpose(1, 2).flatten(2, 3), h_snr)
        lsnr = self.fc_snr(lsnr) * self.lsnr_scale + self.lsnr_offset
        m = self.conv_out(x1)
        return m, lsnr, h, h_snr


class SpectralRefinement(nn.Module):
    def __init__(self, kernel_size_t: int = 1, dilation: int = 1):
        super().__init__()
        p = ModelParams()
        self.lw = p.conv_ch  # Layer width
        self.fe = p.nb_df  # Number of frequency bins for complex refinement
        self.hd = p.refinement_hidden_dim
        if p.refinement_hidden_dim % (p.nb_df // 2) != 0:
            raise ValueError("`refinement_hidden_dim` must be dividable by `nb_df/2`")
        self.conv0 = nn.Sequential(
            nn.LayerNorm(self.fe),
            Conv2d(2, self.lw, (kernel_size_t, 3), dilation=dilation),
            nn.GELU(),
        )
        self.conv1 = PreNormShortcut(
            self.fe,
            nn.Sequential(
                Conv2d(self.lw, self.lw, (kernel_size_t, 3), dilation=dilation, stride=2), nn.GELU()
            ),
            Conv2d(self.lw, self.lw, (1, 1), lookahead=0, stride=2),
        )
        self.gru = PreNormShortcut(
            self.fe // 2,
            GRU(self.fe // 2 * self.lw, self.hd),
            shortcut=nn.Conv2d(self.lw, self.hd // (self.fe // 2), 1),
        )
        self.convt1 = nn.Sequential(
            nn.LayerNorm(self.fe // 2),
            ConvTranspose2d(
                self.hd // (self.fe // 2), self.lw, (kernel_size_t, 3), dilation=dilation, stride=2
            ),
            nn.GELU(),
            nn.Conv2d(self.lw, self.lw, 1),
        )
        self.conv_out = nn.Sequential(
            nn.LayerNorm(self.fe), Conv2d(self.lw, 2, (kernel_size_t, 3)), nn.Tanh()
        )
        self.reset()

    def reset(self):
        def _init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 0.01)

        self.apply(_init)

    def forward(
        self, input: Tensor, h_conv: Optional[Tensor] = None, h_rnn: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # input shape: [B, 1, T, F]
        x0 = self.conv0(input)  # [B, C, T, F]
        x1 = self.conv1(x0)  # [B, C, T, F/2]
        if h_conv is not None:
            x1 = x1 + h_conv
        x_rnn, h_rnn = self.gru(x1, h_rnn)
        x1t = self.convt1(x_rnn) + x0
        input = input + self.conv_out(x1t)
        return input, x1, h_rnn


class MSNet(nn.Module):
    def __init__(self, erb_inv_fb: Tensor):
        super().__init__()
        p = ModelParams()
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.n_stages = p.n_stages
        self.freq_bins = p.fft_size // 2 + 1
        self.erb_bins = p.nb_erb
        self.df_bins = p.nb_df
        self.erb_stage = ErbStage()
        self.mask = Mask(erb_inv_fb, post_filter=p.mask_pf)
        self.refinement_stages: List[SpectralRefinement]
        self.refinement_stages = nn.ModuleList(  # type: ignore
            SpectralRefinement(kernel_size_t=2 if i == 0 else 1, dilation=i + 1)
            for i in range(self.n_stages)
        )
        # SNR offsets on which each refinement layer is activated
        self.refinement_snr_min = -10
        self.refinement_snr_max = (100, 10, 5, 0, -5, -5, -5, -5)
        # Add a bunch of '-5' SNRs to support currently a maximum of 8 refinement layers.
        assert self.n_stages <= 8

    def forward(
        self, spec: Tensor, feat_erb: Tensor, feat_spec: Tensor, atten_lim: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        # Set memory format so stride represents NHWC order
        m, lsnr, _, _ = self.erb_stage(feat_erb)
        spec = self.mask(spec, m, atten_lim)  # [B, 1, T, F, 2]
        out_specs = [spec]
        # re/im into channel axis
        spec_f = spec.squeeze(1)[:, :, : self.df_bins].permute(0, 3, 1, 2)  # [B, 2, T, F_df]
        h_conv: Optional[Tensor] = None
        ic(spec_f.min(), spec_f.mean(), spec_f.max())
        for stage, _lim in zip(self.refinement_stages, self.refinement_snr_max):
            spec_f, h_conv, _ = stage(spec_f, h_conv)
            ic(spec_f.min(), spec_f.mean(), spec_f.max())
            # if lim >= 100:
            #     spec_f, _ = stage(spec_f)
            # else:
            #     idcs = torch.logical_and(lsnr < lim, lsnr > self.refinement_snr_min).squeeze(-1)
            #     for b in range(spec.shape[0]):
            #         spec_f_ = spec_f[b, :, idcs[b]].unsqueeze(0)
            #         ic(spec_f_.shape)
            #         if spec_f_.numel() > 0:
            #             spec_f[b, :, idcs[b]] = stage(spec_f_)[0].squeeze(0)
            #     out_specs.append(spec_f.unsqueeze(-1).transpose(1, -1))
        spec[..., : self.df_bins, :] = spec_f.unsqueeze(-1).transpose(1, -1)
        return spec, m, lsnr, out_specs


def init_model(df_state: Optional[DF] = None, run_df: bool = True, train_mask: bool = True):
    assert run_df and train_mask
    p = ModelParams()
    if df_state is None:
        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    model = MSNet(erb_inverse)
    return model.to(device=get_device())
