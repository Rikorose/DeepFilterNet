from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from df.config import DfParams, config
from df.modules import Mask, convkxf, erb_fb
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
        self.gru_groups: int = config("GRU_GROUPS", cast=int, default=1, section=self.section)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=self.section)
        self.group_shuffle: bool = config(
            "GROUP_SHUFFLE", cast=bool, default=False, section=self.section
        )
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)


class SpectalRecurrentAttention(nn.Module):
    def __init__(self, input_freqs: int, input_ch: int, hidden_dim: int):
        super().__init__()
        # TODO: maybe heads via GroupedGRU?
        self.input_freqs = input_freqs
        self.input_ch = input_ch
        self.hidden_dim = hidden_dim
        self.gru_q = nn.GRU(input_ch, hidden_dim)
        self.gru_k = nn.GRU(input_ch, hidden_dim)
        self.gru_v = nn.GRU(input_ch, hidden_dim)
        # self.gru_o = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_o = nn.Linear(hidden_dim, input_ch)

    def get_h0(self, batch_size: int = 1, device: torch.device = torch.device("cpu")):
        return torch.zeros(3, self.input_freqs * batch_size, self.hidden_dim, device=device)

    def forward(self, input: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Input shape: [T, B, F, C]
        if state is None:
            state = self.get_h0(batch_size=input.shape[1], device=input.device)
        h_q, h_k, h_v = torch.split(state, 1, dim=0)
        t, b, f, c = input.shape
        q, h_q = self.gru_q(input.flatten(1, 2), h_q)  # [T, B*F, H]
        k, h_k = self.gru_k(input.flatten(1, 2), h_k)  # [T, B*F, H]
        # TODO: Try fully connected value rnn via [T, B, F*C]
        v, h_v = self.gru_v(input.flatten(1, 2), h_v)  # [T, B*F, H]
        q = q.unflatten(1, (b, f))  # [T, B, F, H]
        k = k.unflatten(1, (b, f)).transpose(2, 3)  # [T, B, H, F]
        w = q.matmul(k)
        v = w.matmul(v.view(t, b, f, self.hidden_dim))  # [B, T, H]
        v = F.softmax(v, -2)
        o = self.fc_o(v)  # [B, T, F, H]
        state = torch.cat((h_q, h_k, h_v), dim=0)  # [B, T, F, C]
        return o, state


class SpectralRefinement(nn.Module):
    def __init__(self, dilation: int = 1):
        super().__init__()
        p = ModelParams()
        self.lw = p.conv_ch  # Layer width
        self.fe = p.nb_df // 2  # Number of frequency bins in embedding
        kwargs = {"k": 1, "f": 3, "norm": "layer_norm"}
        self.conv0 = convkxf(in_ch=2, fstride=1, out_ch=self.lw, n_freqs=p.nb_df, **kwargs)
        self.conv1 = convkxf(
            in_ch=self.lw, out_ch=self.lw, fstride=2, f_dilation=dilation, n_freqs=p.nb_df, **kwargs
        )
        self.ratten = SpectalRecurrentAttention(self.fe, self.lw, p.erb_hidden_dim)
        self.ln = nn.LayerNorm([p.nb_erb, self.lw])
        self.conv2 = convkxf(
            in_ch=self.lw,
            out_ch=self.lw,
            fstride=2,
            n_freqs=self.fe,
            mode="transposed",
            **kwargs,
        )
        self.conv3 = convkxf(
            in_ch=self.lw, out_ch=2, fstride=1, n_freqs=p.nb_df, act=nn.Tanh(), **kwargs
        )

    def forward(self, input: Tensor, h_atten: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # input shape: [B, 2, T, F]
        x = self.conv0(input)  # [B, C, T, F]
        x = self.conv1(x)  # [B, C, T, F]
        e = x.permute(2, 0, 3, 1)  # [T, B, F, C] (channels_last)
        e, h_atten = self.ratten.forward(e, h_atten)  # [T, B, F, C]
        e = self.ln(e)
        e = e.permute(1, 3, 0, 2)  # [B, C, T, F]
        x = self.conv2(x + e)  # [B, 1, T, F]
        x = self.conv3(x)
        input = input + x
        return x, h_atten


class ErbStage(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        self.lw = p.conv_ch  # Layer width
        self.fe = p.nb_erb  # Number of frequency bins in embedding
        kwargs = {"k": 1, "f": 3, "fstride": 1, "norm": "layer_norm", "n_freqs": p.nb_erb}
        self.conv0 = convkxf(in_ch=1, out_ch=self.lw, **kwargs)
        self.conv1 = convkxf(in_ch=self.lw, out_ch=self.lw, **kwargs)
        self.ratten = SpectalRecurrentAttention(p.nb_erb, self.lw, p.erb_hidden_dim)
        self.ln = nn.LayerNorm([p.nb_erb, self.lw])
        self.gru_snr = nn.GRU(self.lw * self.fe, 16)
        self.fc_snr = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min
        self.conv2 = convkxf(in_ch=self.lw, out_ch=1, act=nn.Sigmoid(), **kwargs)

    def forward(
        self, input: Tensor, h_atten: Optional[Tensor] = None, h_snr: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # input shape: [B, 1, T, F]
        x = self.conv0(input)  # [B, C, T, F]
        x = self.conv1(x)  # [B, C, T, F]
        e = x.permute(2, 0, 3, 1)  # [T, B, F, C] (channels_last)
        e, h_atten = self.ratten.forward(e, h_atten)  # [T, B, F, C]
        e = self.ln(e)
        lsnr, h_snr = self.gru_snr.forward(e.flatten(2, 3), h_snr)
        lsnr = self.fc_snr(lsnr) * self.lsnr_scale + self.lsnr_offset
        e = e.permute(1, 3, 0, 2)  # [B, C, T, F]
        m = self.conv2(x + e)  # [B, 1, T, F]
        return m, lsnr, h_atten, h_snr


class MSNet(nn.Module):
    def __init__(
        self,
        erb_inv_fb: Tensor,
    ):
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
            (SpectralRefinement(dilation=2 ** i) for i in range(self.n_stages))
        )

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,  # type: ignore, Not used, take spec modified by mask instead
        atten_lim: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Set memory format so stride represents NHWC order
        m, lsnr, _, _ = self.erb_stage(feat_erb.clone(memory_format=torch.channels_last))
        spec = self.mask(spec, m, atten_lim)  # [B, 1, T, F, 2]
        # re/im into channel axis
        spec_f = (
            spec.transpose(1, 4)
            .squeeze(4)[..., : self.df_bins]
            .clone(memory_format=torch.channels_last)
        )
        for stage in self.refinement_stages:
            spec_f, _ = stage.forward(spec_f)
        return spec, m, lsnr


def init_model(df_state: Optional[DF] = None, run_df: bool = True, train_mask: bool = True):
    assert run_df and train_mask
    p = ModelParams()
    if df_state is None:
        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    model = MSNet(erb_inverse)
    return model.to(device=get_device())
