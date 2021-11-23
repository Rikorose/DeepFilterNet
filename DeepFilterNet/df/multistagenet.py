from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init

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


def softmax2d(input: Tensor) -> Tensor:
    assert input.ndim == 4
    num = input.exp()
    denum = num.sum(dim=(2, 3), keepdim=True)
    return num / denum


class SpectalRecurrentAttention(nn.Module):
    def __init__(self, input_freqs: int, input_ch: int, hidden_dim: int):
        super().__init__()
        # TODO: maybe heads via GroupedGRU?
        self.input_freqs = input_freqs
        self.input_ch = input_ch
        self.hidden_dim = hidden_dim
        kwargs = {
            "in_ch": input_ch,
            "out_ch": hidden_dim,
            "k": 1,
            "f": 3,
            "f_stride": 1,
            "n_freqs": input_freqs,
            "norm": None,
            "act": None,
        }
        self.conv_q = convkxf(**kwargs)
        self.conv_k = convkxf(**kwargs)
        self.conv_v = convkxf(**kwargs)
        self.gru_o = nn.GRU(hidden_dim * input_freqs, input_ch * input_freqs, batch_first=True)

    def forward(self, input: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Input shape: [B, C, T, F]
        q = self.conv_q(input)  # [B, H, T, F]
        k = self.conv_k(input)  # [B, H, T, F]
        v = self.conv_v(input)  # [B, H, T, F]
        q = q.permute(0, 2, 3, 1)  # [B, T, F, H]
        k = k.transpose(1, 2)  # [B, T, H, F]
        w = q.matmul(k).div(self.hidden_dim ** 0.5)  # [B, T, F, F]
        w = softmax2d(w)
        v = v.permute(0, 2, 3, 1)  # [B, T, F, H]
        v = w.matmul(v)  # [B, T, F, H]
        o, h = self.gru_o.forward(v.flatten(2, 3), h)  # [B, T, F, H]
        o = o.unflatten(2, (self.input_ch, self.input_freqs))  # [B, T, C, F]
        o = o.transpose(1, 2)  # [B, C, T, F]
        return o, h


class SpectralRefinement(nn.Module):
    def __init__(self, dilation: int = 1):
        super().__init__()
        p = ModelParams()
        self.lw = p.conv_ch  # Layer width
        self.fe = p.nb_df // 2  # Number of frequency bins in embedding
        kwargs = {"k": 1, "f": 3, "norm": "layer_norm", "act": nn.PReLU()}
        self.conv0 = convkxf(in_ch=2, fstride=1, out_ch=self.lw, n_freqs=p.nb_df, **kwargs)
        self.conv1 = convkxf(
            in_ch=self.lw,
            out_ch=self.lw // 2,
            fstride=2,
            f_dilation=dilation,
            n_freqs=p.nb_df,
            **kwargs,
        )
        self.hd = p.erb_hidden_dim
        self.ratten = SpectalRecurrentAttention(self.fe, self.lw // 2, p.erb_hidden_dim)
        self.ln = nn.LayerNorm(self.fe)
        self.conv2 = convkxf(
            in_ch=self.lw // 2,
            out_ch=self.lw,
            fstride=2,
            n_freqs=self.fe,
            mode="transposed",
            **kwargs,
        )
        kwargs["norm"] = None
        kwargs["act"] = None
        # kwargs["act"] = nn.Tanh()
        self.conv3 = convkxf(in_ch=self.lw, out_ch=2, fstride=1, n_freqs=p.nb_df, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 0.01)
            elif isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=20)

        self.apply(init_weights)

    def forward(self, input: Tensor, h_atten: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # input shape: [B, 2, T, F]
        x0 = self.conv0(input)  # [B, C, T, F]
        x = self.conv1(x0)  # [B, C, T, F]
        e, h_atten = self.ratten.forward(x, h_atten)  # [B, C, T, F]
        e = self.ln(x + e)
        x = self.conv2(x)  # [B, C, T, F]
        x = self.conv3(x + x0)
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
        self.conv1 = convkxf(in_ch=self.lw, out_ch=self.lw // 2, **kwargs)
        self.ratten = SpectalRecurrentAttention(p.nb_erb, self.lw // 2, p.erb_hidden_dim)
        self.ln = nn.LayerNorm(self.fe)
        self.gru_snr = nn.GRU(self.lw // 2 * self.fe, 16)
        self.fc_snr = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min
        self.conv2 = convkxf(in_ch=self.lw // 2, out_ch=1, act=nn.Sigmoid(), **kwargs)

    def forward(
        self, input: Tensor, h_atten: Optional[Tensor] = None, h_snr: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # input shape: [B, 1, T, F]
        x = self.conv0(input)  # [B, C, T, F]
        x = self.conv1(x)  # [B, C, T, F]
        e, h_atten = self.ratten.forward(x, h_atten)  # [B, C, T, F]
        lsnr, h_snr = self.gru_snr.forward(e.transpose(1, 2).flatten(2, 3), h_snr)
        lsnr = self.fc_snr(lsnr) * self.lsnr_scale + self.lsnr_offset
        x = self.ln(x + e)
        m = self.conv2(x)  # [B, 1, T, F]
        return m, lsnr, h_atten, h_snr


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
            (SpectralRefinement(dilation=2 ** i) for i in range(self.n_stages))
        )

    def forward(
        self, spec: Tensor, feat_erb: Tensor, feat_spec: Tensor, atten_lim: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, None]:
        # Set memory format so stride represents NHWC order
        m, lsnr, _, _ = self.erb_stage(feat_erb)
        spec = self.mask(spec, m, atten_lim)  # [B, 1, T, F, 2]
        out_specs = [spec]
        # re/im into channel axis
        spec_f = spec.transpose(1, 4).squeeze(4)[..., : self.df_bins]
        for stage in self.refinement_stages:
            spec_f, _ = stage.forward(spec_f)
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
