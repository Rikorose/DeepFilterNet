import math
from functools import partial, reduce
from typing import Callable, Final, List, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import init
from torch.nn.parameter import Parameter

from df.config import Csv, DfParams, config
from df.modules import Conv2dNormAct, ConvTranspose2dNormAct, GroupedGRU, Mask, erb_fb
from df.utils import angle_re_im, get_device
from libdf import DF


class ModelParams(DfParams):
    section = "multistagenet"

    def __init__(self):
        super().__init__()
        self.stages: List[int] = config(
            "STAGES", cast=Csv(int), default=(3, 3, 9, 3), section=self.section  # type: ignore
        )
        self.conv_lookahead: int = config(
            "CONV_LOOKAHEAD", cast=int, default=0, section=self.section
        )
        self.erb_widths: List[int] = config(
            "ERB_WIDTHS", cast=Csv(int), default=[16, 16, 16], section=self.section
        )
        self.erb_hidden_dim: int = config(
            "ERB_HIDDEN_DIM", cast=int, default=64, section=self.section
        )
        self.refinement_widths: List[int] = config(
            "REFINEMENT_WIDTHS", cast=Csv(int), default=[32, 32, 32, 32], section=self.section
        )
        self.refinement_hidden_dim: int = config(
            "REFINEMENT_HIDDEN_DIM", cast=int, default=96, section=self.section
        )
        self.refinement_act: str = (
            config("REFINEMENT_OUTPUT_ACT", default="identity", section=self.section)
            .lower()
            .replace("none", "identity")
        )
        self.refinement_op: str = config(
            "REFINEMENT_OP", default="mul", section=self.section
        ).lower()
        self.refinement_out_layer: str = config(
            "REFINEMENT_OUTPUT_LAYER", default="LocallyConnected", section=self.section
        )
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)


# class GruSE(nn.Module):
#     """GRU with previous adaptive avg pooling like SqueezeExcitation"""

#     avg_dim: Final[int]

#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         groups: int = 1,
#         reduce_: str = "frequencies",
#         skip: Optional[Callable[..., torch.nn.Module]] = nn.Identity,
#         scale_activation: Optional[Callable[..., torch.nn.Module]] = None,
#     ):
#         super().__init__()
#         assert reduce_ in ("channels", "frequencies", "none")
#         if reduce_ == "channels":
#             self.avg_dim = 1
#         else:
#             self.avg_dim = 3
#         if groups == 1:
#             self.gru = nn.GRU(input_dim, hidden_dim)
#         else:
#             self.gru = GroupedGRU(input_dim, hidden_dim, groups=groups)
#         assert (
#             skip or scale_activation is None
#         ), "Can only either use a skip connection or SqueezeExcitation with `scale_activation`"
#         self.fc = nn.Linear(hidden_dim, input_dim)
#         self.skip = skip() if skip is not None else None
#         self.scale = scale_activation() if scale_activation is not None else None

#     def forward(self, input: Tensor, h=None) -> Tuple[Tensor, Tensor]:
#         # x: [B, C, T, F]
#         if self.avg_dim == 1:
#             x = input.mean(dim=self.avg_dim)  # [B, T, C]
#         else:
#             x = input.mean(dim=self.avg_dim).transpose(1, 2)  # [B, T, C]
#         x, h = self.gru(x, h)
#         x = self.fc(x)
#         if self.avg_dim == 1:
#             x = x.unsqueeze(1)
#         else:
#             x = x.transpose(1, 2).unsqueeze(-1)
#         if self.skip is not None:
#             x = self.skip(input) + x  # a regular skip connection
#         elif self.scale is not None:
#             x = input * self.scale(x)  # like in SqueezeExcitation
#         return x, h


class LSNRNet(nn.Module):
    lsnr_scale: Final[int]
    lsnr_offset: Final[int]

    def __init__(
        self, in_ch: int, hidden_dim: int = 16, fstride=2, lsnr_min: int = -15, lsnr_max: int = 40
    ):
        super().__init__()
        self.conv = Conv2dNormAct(in_ch, in_ch, kernel_size=(1, 3), fstride=fstride, separable=True)
        self.gru_snr = nn.GRU(in_ch, hidden_dim)
        self.fc_snr = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min

    def forward(self, x: Tensor, h=None) -> Tuple[Tensor, Tensor]:
        x = self.conv(x)
        x, h = self.gru_snr(x.mean(-1).transpose(1, 2), h)
        x = self.fc_snr(x) * self.lsnr_scale + self.lsnr_offset
        return x, h


class LocalLinearCF(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_freqs: int, bias: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.register_parameter(
            "weight", Parameter(torch.zeros(in_ch, out_ch, n_freqs), requires_grad=True)
        )
        if bias:
            self.bias: Optional[Tensor]
            self.register_parameter(
                "bias", Parameter(torch.zeros(out_ch, 1, n_freqs), requires_grad=True)
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, Ci, T, F]
        x = torch.einsum("bctf,cof->botf", x, self.weight)  # [B, Co, T, F]
        if self.bias is not None:
            x = x + self.bias
        return x


class GroupedLinearCF(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_freqs: int, n_groups: int, bias: bool = True):
        super().__init__()
        # self.weight: Tensor
        self.n_freqs = n_freqs
        n_groups = n_groups if n_groups > 0 else n_freqs
        if n_groups == n_freqs:
            logger.warning("Use more performant LocallyConnected since they are equivalent now.")
        assert (
            n_freqs % n_groups == 0
        ), "Number of frequencies must be dividable by the number of groups"
        self.n_groups = n_groups
        self.n_unfold = n_freqs // n_groups
        self.register_parameter(
            "weight",
            Parameter(
                torch.zeros(n_groups, n_freqs // n_groups, in_ch, out_ch), requires_grad=True
            ),
        )
        if bias:
            self.bias: Optional[Tensor]
            self.register_parameter(
                "bias", Parameter(torch.zeros(out_ch, 1, n_freqs), requires_grad=True)
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, Ci, T, F]
        x = x.permute(0, 2, 3, 1)  # [B, T, F, Ci]
        x = x.unflatten(2, (self.n_groups, self.n_unfold))  # [B, T, G, F/G, Ci]
        x = torch.einsum("btfgi,fgio->btfgo", x, self.weight)  # [B, T, G, F/G, Co]
        x = x.flatten(2, 3)  # [B, T, F, Co]
        x = x.permute(0, 3, 1, 2)  # [B, Co, T, F]
        if self.bias is not None:
            x = x + self.bias
        return x


class GroupedGRULayerMS(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_freqs: int, n_groups: int, bias: bool = True):
        super().__init__()
        assert n_freqs % n_groups == 0
        self.n_freqs = n_freqs
        self.g_freqs = n_freqs // n_groups
        self.n_groups = n_groups
        self.out_ch = self.g_freqs * out_ch
        self._in_ch = in_ch
        self.input_size = self.g_freqs * in_ch
        # self.weight_ih_l: Tensor
        self.register_parameter(
            "weight_ih_l",
            Parameter(torch.zeros(n_groups, 3 * self.out_ch, self.input_size), requires_grad=True),
        )
        # self.weight_hh_l: Tensor
        self.register_parameter(
            "weight_hh_l",
            Parameter(torch.zeros(n_groups, 3 * self.out_ch, self.out_ch), requires_grad=True),
        )
        if bias:
            # self.bias_ih_l: Tensor
            self.register_parameter(
                "bias_ih_l", Parameter(torch.zeros(n_groups, 3 * self.out_ch), requires_grad=True)
            )
            # self.bias_hh_l: Tensor
            self.register_parameter(
                "bias_hh_l", Parameter(torch.zeros(n_groups, 3 * self.out_ch), requires_grad=True)
            )
        else:
            self.bias_ih_l = None  # type: ignore
            self.bias_hh_l = None  # type: ignore

    def init_hidden(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tensor:
        return torch.zeros(batch_size, self.n_groups, self.out_ch, device=device)

    def forward(self, input: Tensor, h=None) -> Tuple[Tensor, Tensor]:
        # input: [B, Ci, T, F]
        assert self.n_freqs == input.shape[-1]
        assert self._in_ch == input.shape[1]
        if h is None:
            h = self.init_hidden(input.shape[0])
        input = input.permute(0, 2, 3, 1).unflatten(2, (self.n_groups, self.g_freqs)).flatten(3)
        input = torch.einsum("btgi,goi->btgo", input, self.weight_ih_l)
        if self.bias_ih_l is not None:
            input = input + self.bias_ih_l
        h_out: List[Tensor] = []
        for t in range(input.shape[1]):
            hh = torch.einsum("bgo,gpo->bgp", h, self.weight_hh_l)
            if self.bias_hh_l is not None:
                hh = hh + self.bias_hh_l
            ri, zi, ni = input[:, t].split(self.out_ch, dim=2)
            rh, zh, nh = hh.split(self.out_ch, dim=2)
            r = torch.sigmoid(ri + rh)
            z = torch.sigmoid(zi + zh)
            n = torch.tanh(ni + r * nh)
            h = (1 - z) * n + z * h
            h_out.append(h)
        out = torch.stack(h_out, dim=1)  # [B, T, G, F/G*Co]
        out = out.unflatten(3, (self.g_freqs, -1)).flatten(2, 3)  # [B, T, F, Co]
        out = out.permute(0, 3, 1, 2)  # [B, Co, T, F]
        return out, h


class GroupedGRUMS(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_freqs: int,
        n_groups: int,
        n_layers: int = 1,
        bias: bool = True,
        add_outputs: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.grus: List[GroupedGRULayerMS] = nn.ModuleList()  # type: ignore
        gru_layer = partial(
            GroupedGRULayerMS, out_ch=out_ch, n_freqs=n_freqs, n_groups=n_groups, bias=bias
        )
        self.gru0 = gru_layer(in_ch=in_ch)
        for _ in range(1, n_layers):
            self.grus.append(gru_layer(in_ch=out_ch))
        self.add_outputs = add_outputs

    def init_hidden(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tensor:
        return torch.stack(
            tuple(self.gru0.init_hidden(batch_size, device) for _ in range(self.n_layers))
        )

    def forward(self, input: Tensor, h=None) -> Tuple[Tensor, Tensor]:
        if h is None:
            h = self.init_hidden(input.shape[0], input.device)
        h_out = []
        # First layer
        input, hl = self.gru0(input, h[0])
        h_out.append(hl)
        output = input
        for i, gru in enumerate(self.grus, 1):
            input, hl = gru(input, h[i])
            h_out.append(hl)
            if self.add_outputs:
                output = output + input
        if not self.add_outputs:
            output = input
        return output, torch.stack(h_out)  # type: ignore


class FreqStage(nn.Module):
    in_ch: Final[int]
    out_ch: Final[int]
    depth: Final[int]

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        out_act: Optional[Callable[..., torch.nn.Module]],
        num_freqs: int,
        gru_dim: int,
        widths: List[int],
        fstrides: Optional[List[int]] = None,
        initial_kernel: Tuple[int, int] = (3, 3),
        kernel: Tuple[int, int] = (1, 3),
        separable_conv: bool = False,
        num_gru_layers: int = 3,
        num_gru_groups: int = 8,
        global_pathway: bool = False,
        decoder_out_layer: Optional[Callable[[int, int], torch.nn.Module]] = None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.depth = len(widths) - 1
        if fstrides is not None:
            assert len(fstrides) >= self.depth
            overall_stride = reduce(lambda x, y: x * y, fstrides)
        else:
            fstrides = [2] * self.depth
            overall_stride = 2 ** (len(widths) - 1)
        assert num_freqs % overall_stride == 0, f"num_freqs ({num_freqs}) must correspond to depth"
        norm_layer = nn.BatchNorm2d

        # Encoder
        self.enc0 = Conv2dNormAct(
            in_ch,
            widths[0],
            initial_kernel,
            fstride=1,
            norm_layer=norm_layer,
            separable=separable_conv,
        )
        self.enc = nn.ModuleList()

        fstrides = fstrides or [2] * self.depth
        freqs = num_freqs
        for i in range(self.depth):
            in_ch = widths[i]
            out_ch = widths[i + 1]
            fstride = fstrides[i]
            freqs = freqs // fstride
            self.enc.append(
                Conv2dNormAct(
                    in_ch, out_ch, kernel_size=kernel, fstride=fstride, separable=separable_conv
                )
            )
        self.inner_freqs = freqs
        self.lin_emb_in = LocalLinearCF(out_ch, gru_dim // freqs, n_freqs=freqs)
        if num_gru_groups == 1:
            self.gru = CFGRU(
                freqs=self.inner_freqs,
                input_size=gru_dim // freqs * freqs,
                hidden_size=gru_dim,
                num_layers=num_gru_layers,
            )
        else:
            self.gru = GroupedGRU(
                in_ch=gru_dim // freqs,
                out_ch=gru_dim // freqs,
                n_freqs=freqs,
                n_groups=min(freqs, num_gru_groups),
                n_layers=num_gru_layers,
            )
        self.lin_emb_out = LocalLinearCF(gru_dim // freqs, out_ch, n_freqs=freqs)
        self.gru_skip = nn.Conv2d(out_ch, out_ch, 1)

        self.dec = nn.ModuleList()
        for i in range(self.depth - 1, -1, -1):
            in_ch = widths[i + 1]
            out_ch = widths[i]
            fstride = fstrides[i]
            dec_layer = ConvTranspose2dNormAct if fstride > 1 else Conv2dNormAct
            self.dec.append(
                dec_layer(
                    in_ch, out_ch, kernel_size=kernel, fstride=fstride, separable=separable_conv
                )
            )
        self.global_pathway = (
            Conv2dNormAct(widths[0], widths[0], 1) if global_pathway else nn.Identity()
        )
        if decoder_out_layer is None:
            self.dec0 = Conv2dNormAct(
                widths[0],
                self.out_ch,
                kernel,
                fstride=1,
                norm_layer=None,
                activation_layer=out_act,
                separable=separable_conv,
            )
        else:
            self.dec0 = decoder_out_layer(widths[0], self.out_ch)

    def encode(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        intermediate = []
        x = self.enc0(x)
        for enc_layer in self.enc:
            intermediate.append(x)
            x = enc_layer(x)
        return x, intermediate

    def embed(self, input: Tensor, h=None) -> Tuple[Tensor, Tensor]:
        x = self.lin_emb_in(input)
        x_gru, h = self.gru(x, h)
        x_gru = self.lin_emb_out(x_gru)
        x = self.gru_skip(input) + x_gru
        return x, h

    def decode(self, x: Tensor, intermediate: List[Tensor]) -> Tensor:
        intermediate[0] = self.global_pathway(intermediate[0])
        for dec_layer, x_enc in zip(self.dec, reversed(intermediate)):
            x = dec_layer(x) + x_enc
        x = self.dec0(x)
        return x

    def forward(self, x: Tensor, h=None) -> Tuple[Tensor, Tensor, Tensor]:
        # input shape: [B, C, T, F]
        x_enc, intermediate = self.encode(x)
        x_inner, h = self.embed(x_enc, h)
        x = self.decode(x_inner, intermediate)
        return x, x_enc, h


class CFGRU(nn.Module):
    def __init__(self, freqs: int, *args, **kwargs):
        super().__init__()
        self.inner_freqs = freqs
        self.gru = nn.GRU(*args, **kwargs)

    def forward(self, x, h):
        x = x.permute(0, 2, 3, 1).flatten(2)
        x_gru, h = self.gru(x, h)
        x_gru = x.unflatten(2, (self.inner_freqs, -1)).permute(0, 3, 1, 2)
        return x_gru, h


class ComplexCompression(nn.Module):
    def __init__(self, n_freqs: int, init_value: float = 0.3):
        super().__init__()
        self.register_parameter(
            "c", Parameter(torch.full((n_freqs,), init_value), requires_grad=True)
        )

    def forward(self, x: Tensor):
        # x has shape x [B, 2, T, F]
        x_abs = (x[:, 0].square() + x[:, 1].square()).clamp_min(1e-10).pow(self.c)
        x_ang = angle_re_im.apply(x[:, 0], x[:, 1])
        x = torch.stack((x_abs * torch.cos(x_ang), x_abs * torch.sin(x_ang)), dim=1)
        # x_c = x_abs * torch.exp(1j * x_ang)
        # x_ = torch.view_as_complex(x.permute(0,2,3,1).contiguous())
        # torch.allclose(x_, x_c)
        return x


class MagCompression(nn.Module):
    def __init__(self, n_freqs: int, init_value: float = 0.3):
        super().__init__()
        self.c: Tensor
        self.register_parameter(
            "c", Parameter(torch.full((n_freqs,), init_value), requires_grad=True)
        )
        self.mn: Tensor
        self.register_parameter("mn", Parameter(torch.full((n_freqs,), -0.2), requires_grad=True))

    def forward(self, x: Tensor):
        # x has shape x [B, T, F, 2]
        x = x.pow(self.c) + self.mn
        return x


class ComplexAdd(nn.Module):
    def forward(self, a, b):
        return a + b


class ComplexMul(nn.Module):
    def forward(self, a, b):
        # [B, 2, *]
        re = a[:, :1] * b[:, :1] - a[:, 1:] * b[:, 1:]
        im = a[:, :1] * b[:, 1:] + a[:, :1] * b[:, 1:]
        return torch.cat((re, im), dim=1)


class MSNet(nn.Module):
    def __init__(self, erb_fb: Tensor, erb_inv_fb: Tensor):
        super().__init__()
        p = ModelParams()
        self.stages = p.stages
        self.freq_bins = p.fft_size // 2 + 1
        self.erb_bins = p.nb_erb
        self.df_bins = p.nb_df
        self.erb_fb: Tensor
        self.erb_comp = MagCompression(self.erb_bins)
        self.cplx_comp = ComplexCompression(self.df_bins)
        self.register_buffer("erb_fb", erb_fb, persistent=False)
        self.erb_stage = FreqStage(
            in_ch=1,
            out_ch=1,
            widths=p.erb_widths,
            out_act=nn.Sigmoid,
            num_freqs=p.nb_erb,
            gru_dim=p.erb_hidden_dim,
        )
        strides = [2, 2, 2, 2, 1, 1, 1]
        self.mask = Mask(erb_inv_fb, post_filter=p.mask_pf)
        refinement_act = {"tanh": nn.Tanh, "identity": nn.Identity}[p.refinement_act.lower()]
        assert p.refinement_out_layer.lower() in ("locallyconnected", "conv2d")
        refinement_out_layer = (
            partial(Conv2dNormAct, kernel_size=(3, 1), norm_layer=None, activation_layer=None)
            if p.refinement_out_layer.lower() == "conv2d"
            else partial(GroupedLinearCF, n_freqs=p.nb_df, n_groups=8)
        )
        self.refinement_stage = FreqStage(
            in_ch=2,
            out_ch=2,
            out_act=refinement_act,
            widths=p.refinement_widths,
            num_freqs=p.nb_df,
            gru_dim=p.refinement_hidden_dim,
            fstrides=strides,
            decoder_out_layer=refinement_out_layer,
        )
        self.refinement_op = ComplexMul() if p.refinement_op == "mul" else ComplexAdd()
        self.lsnr_net = LSNRNet(p.erb_widths[-1], lsnr_min=p.lsnr_min, lsnr_max=p.lsnr_max)

    def forward(
        self, spec: Tensor, atten_lim: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, None]:
        # Spec shape: [B, 1, T, F, 2]
        feat_erb = torch.view_as_complex(spec).abs().matmul(self.erb_fb)
        feat_erb = self.erb_comp(feat_erb)
        m, x_rnn, _ = self.erb_stage(feat_erb)
        spec = self.mask(spec, m, atten_lim)  # [B, 1, T, F, 2]
        lsnr, _ = self.lsnr_net(x_rnn)
        # re/im into channel axis
        spec_f = (
            spec.squeeze(1)[:, :, : self.df_bins].permute(0, 3, 1, 2).clone()
        )  # [B, 2, T, F_df]
        r, _, _ = self.refinement_stage(self.cplx_comp(spec_f))
        spec_f = self.refinement_op(spec_f, r)
        spec[..., : self.df_bins, :] = spec_f.unsqueeze(-1).transpose(1, -1)
        return spec, m, lsnr, None


def init_model(df_state: Optional[DF] = None, run_df: bool = True, train_mask: bool = True):
    assert run_df and train_mask
    p = ModelParams()
    if df_state is None:
        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb_width = df_state.erb_widths()
    erb = erb_fb(erb_width, p.sr)
    erb_inverse = erb_fb(erb_width, p.sr, inverse=True)
    model = MSNet(erb, erb_inverse)
    return model.to(device=get_device())
