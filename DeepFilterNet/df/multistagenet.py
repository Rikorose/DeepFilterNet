from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from df.config import DfParams, config
from df.modules import ConvGRU, LongShortAttention, Mask, PreNormShortcut, erb_fb
from df.utils import get_device
from libdf import DF


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
            "REFINEMENT_HIDDEN_DIM", cast=int, default=64, section=self.section
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


class SpectralRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        self.lw = p.conv_ch  # Layer width
        self.fe = p.nb_df  # Number of frequency bins in embedding
        self.conv_in = nn.Sequential(nn.LayerNorm(self.fe), nn.Conv2d(2, self.lw, 1), nn.GELU())
        self.conv_gru = torch.jit.script(ConvGRU(self.lw, self.lw, 1, bias=False))
        self.ls_fatten = PreNormShortcut(
            self.fe, LongShortAttention(self.fe, self.lw, dim_head=p.refinement_hidden_dim, r=4)
        )
        self.conv_m = PreNormShortcut(
            self.fe,
            nn.Sequential(
                nn.Conv2d(self.lw, self.lw, 1), nn.GELU(), nn.Conv2d(self.lw, self.lw, 1)
            ),
        )
        p.refinement_act = p.refinement_act.lower()
        assert p.refinement_act in ("tanh", "none")
        out_act = nn.Tanh() if p.refinement_act == "tanh" else nn.Identity()
        self.conv_out = nn.Sequential(nn.LayerNorm(self.fe), nn.Conv2d(self.lw, 2, 1), out_act)

    def forward(self, input: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # input shape: [B, 1, T, F]
        x = self.conv_in(input)  # [B, C, T, F]
        x_rnn, h = self.conv_gru(x, h)
        x = self.ls_fatten(x + x_rnn)  # [B, C, T, F]
        x = self.conv_m(x)
        x = self.conv_out(x)
        input = input + x
        return input, h


class ErbStage(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        self.lw = p.conv_ch  # Layer width
        self.fe = p.nb_erb  # Number of frequency bins in embedding
        self.conv_in = nn.Sequential(nn.LayerNorm(self.fe), nn.Conv2d(1, self.lw, 1), nn.GELU())
        self.conv_gru = torch.jit.script(ConvGRU(self.lw, self.lw, 1, bias=False))
        self.ls_fatten = PreNormShortcut(
            self.fe, LongShortAttention(self.fe, self.lw, dim_head=p.erb_hidden_dim, r=4)
        )
        self.conv_m = PreNormShortcut(
            self.fe,
            nn.Sequential(
                nn.Conv2d(self.lw, self.lw, 1), nn.GELU(), nn.Conv2d(self.lw, self.lw, 1)
            ),
        )
        self.conv_out = nn.Sequential(nn.LayerNorm(self.fe), nn.Conv2d(self.lw, 1, 1), nn.Sigmoid())
        self.gru_snr = nn.GRU(self.lw * self.fe, 16)
        self.fc_snr = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def forward(
        self, input: Tensor, h: Optional[Tensor] = None, h_snr: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # input shape: [B, 1, T, F]
        x = self.conv_in(input)  # [B, C, T, F]
        x_rnn, h = self.conv_gru(x, h)
        x = self.ls_fatten(x + x_rnn)  # [B, C, T, F]
        x = self.conv_m(x)
        lsnr, h_snr = self.gru_snr(x.transpose(1, 2).flatten(2, 3), h_snr)
        lsnr = self.fc_snr(lsnr) * self.lsnr_scale + self.lsnr_offset
        m = self.conv_out(x)
        return m, lsnr, h, h_snr


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
            (SpectralRefinement() for _ in range(self.n_stages))
        )
        # SNR offsets on which each refinement layer is activated
        self.refinement_snr_min = -10
        self.refinement_snr_max = (20, 10, 5, 0, -5, -5, -5, -5)
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
        for stage, lim in zip(self.refinement_stages, self.refinement_snr_max):
            idcs = torch.logical_and(lsnr < lim, lsnr > self.refinement_snr_min).squeeze()
            for b in range(spec.shape[0]):
                spec_f_ = spec_f[b, :, idcs[b]].unsqueeze(0)
                if spec_f_.numel() > 0:
                    spec_f[b, :, idcs[b]] = stage(spec_f_)[0].squeeze(0)
                # spec_f_, _ = stage(spec_f[idcs].clone())
            out_specs.append(spec_f.unsqueeze(-1).transpose(1, -1))
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
