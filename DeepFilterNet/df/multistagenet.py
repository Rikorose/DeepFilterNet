import math
from functools import partial, reduce
from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from icecream import ic  # noqa
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torchvision.models._utils import _make_divisible
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import SqueezeExcitation

from df.config import Csv, DfParams, config
from df.modules import Mask, erb_fb
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
        self.width_mult: int = config("WIDTH_MULT", cast=float, default=1, section=self.section)
        self.depth_mult: int = config("DEPTH_MULT", cast=float, default=1, section=self.section)
        self.erb_hidden_dim: int = config(
            "ERB_HIDDEN_DIM", cast=int, default=64, section=self.section
        )
        self.erb_depth: int = config("ERB_DEPTH", cast=int, default=3, section=self.section)
        self.refinement_hidden_dim: int = config(
            "REFINEMENT_HIDDEN_DIM", cast=int, default=96, section=self.section
        )
        self.refinement_depth: int = config(
            "REFINEMENT_DEPTH", cast=int, default=5, section=self.section
        )
        self.refinement_act: str = (
            config("REFINEMENT_OUTPUT_ACT", default="identity", section=self.section)
            .lower()
            .replace("none", "identity")
        )
        self.refinement_op: str = config(
            "REFINEMENT_OP", default="mul", section=self.section
        ).lower()
        assert self.refinement_op in ("mul", "add")
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)


class Conv2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Iterable[int]],
        fstride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        fpad: bool = True,
        bias: bool = True,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
    ):
        """Causal Conv2d by delaying the signal for any lookahead.

        Expected input format: [B, C, T, F]
        """
        super().__init__()
        lookahead = 0  # This needs to be handled on the input feature side
        # Padding on time axis
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        if fpad:
            fpad_ = kernel_size[1] // 2 + dilation - 1
        else:
            fpad_ = 0
        pad = (0, 0, lookahead, kernel_size[0] - 1 - lookahead)
        layers = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        layers.append(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=(0, fpad_),
                stride=(1, fstride),  # Stride over time is always 1
                dilation=(1, dilation),  # Same for dilation
                groups=groups,
                bias=bias,
            )
        )
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]],
        fstride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        fpad: bool = True,
        bias: bool = True,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
    ):
        """Causal ConvTranspose2d.

        Expected input format: [B, C, T, F]
        """
        super().__init__()
        # Padding on time axis, with lookahead = 0
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        if fpad:
            fpad_ = kernel_size[1] // 2
        else:
            fpad_ = 0
        pad = (0, 0, 0, kernel_size[0] - 1)
        layers = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        layers.append(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=(kernel_size[0] - 1, fpad_ + dilation - 1),
                output_padding=(0, fpad_),
                stride=(1, fstride),  # Stride over time is always 1
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class MbConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: float,
        kernel: Tuple[int, int],
        fstride: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
        dp: float,  # stochastic_depth_probability
        norm_layer: Callable[..., nn.Module],
        act_layer: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
        se_factor: float = 0.25,
        fused: bool = False,
        transpose: bool = False,
    ):
        super().__init__()
        assert 1 <= fstride <= 2
        self.use_res_connect = fstride == 1 and in_ch == out_ch
        layers: List[nn.Module] = []

        self.in_ch = self.adjust_channels(in_ch, width_mult)
        expanded_ch = self.adjust_channels(self.in_ch, expand_ratio)
        self.out_ch = self.adjust_channels(out_ch, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)
        # expand
        if expanded_ch != self.in_ch and not fused:
            layers.append(
                Conv2dNormAct(
                    self.in_ch,
                    expanded_ch,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=act_layer,
                    bias=False,
                )
            )
        # depthwise
        conv_l = Conv2dNormAct if not transpose else ConvTranspose2dNormAct
        layers.append(
            conv_l(
                self.in_ch if fused else expanded_ch,
                expanded_ch,
                kernel_size=kernel,
                fstride=fstride,
                groups=1 if fused else expanded_ch,
                bias=False,
                norm_layer=norm_layer,
                activation_layer=act_layer,
            )
        )
        # squeeze and excitation
        squeeze_ch = max(1, int(self.in_ch * se_factor))
        if se_factor < 1:
            layers.append(se_layer(expanded_ch, squeeze_ch, activation=act_layer))
        # project
        layers.append(
            nn.Sequential(
                Conv2dNormAct(
                    expanded_ch,
                    self.out_ch,
                    kernel_size=1,
                    bias=False,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
        )
        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(dp, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class GruMlp(nn.Module):
    def __init__(self, ch: int, freqs: int, hidden_size: int, *args, **kwargs):
        super().__init__()
        kwargs["batch_first"] = True
        self.ch = ch
        self.freqs = freqs
        io_size = ch * freqs
        self.gru = nn.GRU(io_size, hidden_size, *args, **kwargs)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, io_size)

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """GRU transposing [B, C, T, F] input shape to [B, T, C*F]."""
        _, _, _, f = x.shape
        x, h = self.gru(x.transpose(1, 2).flatten(2), h)
        x = self.fc(self.norm(x))
        x = x.unflatten(2, (-1, f)).transpose(1, 2)
        return x, h


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LSNRNet(nn.Module):
    def __init__(
        self, in_ch: int, hidden_dim: int = 16, fstride=2, lsnr_min: int = -15, lsnr_max: int = 40
    ):
        super().__init__()
        self.conv = Conv2dNormAct(in_ch, in_ch, kernel_size=(1, 3), fstride=fstride, groups=in_ch)
        self.gru_snr = nn.GRU(in_ch, hidden_dim)
        self.fc_snr = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x = self.conv(x)
        x, h = self.gru_snr(x.mean(-1).transpose(1, 2), h)
        x = self.fc_snr(x) * self.lsnr_scale + self.lsnr_offset
        return x, h


class FreqStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        out_act: Optional[Callable[..., torch.nn.Module]],
        num_freqs: int,
        hidden_dim: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        depth: int = 4,
        fstrides: Optional[List[int]] = None,
        stochastic_depth_prob: float = 0.2,
    ):
        super().__init__()
        self.fe = num_freqs  # Number of frequency bins in embedding
        self.in_ch = in_ch
        self.out_ch = out_ch
        if fstrides is not None:
            assert len(fstrides) == depth
            overall_stride = reduce(lambda x, y: x * y, fstrides)
        else:
            overall_stride = 2**depth
        assert num_freqs % overall_stride == 0, f"num_freqs ({num_freqs}) must correspond to depth"
        self.hd = hidden_dim
        norm_layer = nn.BatchNorm2d

        self.enc = nn.ModuleList()
        self.enc.append(Conv2dNormAct(in_ch, 16, (3, 3), fstride=1, norm_layer=norm_layer))

        dp = stochastic_depth_prob
        widths = [16, 32, 48, 64, 64, 64, 64, 64][: depth + 1]
        exp_ratio = [1, 4, 4, 4, 6, 6, 6][:depth]
        num_layers = [1, 2, 4, 4, 6, 9, 15][:depth]
        se_f = [1, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25][:depth]
        fstrides = fstrides or [2] * depth
        fused = [True, True, False, False, False, False, False][:depth]
        for i in range(depth):
            stage: List[nn.Module] = []
            in_ch = widths[i]
            out_ch = widths[i + 1]
            fstride = fstrides[i]
            for _ in range(num_layers[i]):
                if stage:
                    in_ch = out_ch
                    fstride = 1
                stage.append(
                    MbConv(
                        in_ch,
                        out_ch,
                        exp_ratio[i],
                        kernel=(3, 3),
                        fstride=fstride,
                        num_layers=num_layers[i],
                        se_factor=se_f[i],
                        fused=fused[i],
                        dp=dp / (depth - 1) * i,
                        width_mult=width_mult,
                        depth_mult=depth_mult,
                        norm_layer=norm_layer,
                    )
                )
            self.enc.append(nn.Sequential(*stage))

        self.max_width = widths[depth]
        self.rnn = GruMlp(self.max_width, num_freqs // overall_stride, self.hd)
        dp = stochastic_depth_prob
        self.dec = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            stage: List[nn.Module] = []
            in_ch = widths[i + 1]
            out_ch = widths[i]
            fstride = fstrides[i]
            for _ in range(num_layers[i]):
                if stage:
                    in_ch = out_ch
                    fstride = 1
                stage.append(
                    MbConv(
                        in_ch,
                        out_ch,
                        exp_ratio[i],
                        kernel=(3, 3),
                        fstride=fstride,
                        num_layers=num_layers[i],
                        se_factor=se_f[i],
                        fused=fused[i],
                        dp=dp / depth * i,
                        width_mult=width_mult,
                        depth_mult=depth_mult,
                        norm_layer=norm_layer,
                        transpose=True if fstride > 1 else False,
                    )
                )
            self.dec.append(nn.Sequential(*stage))
        self.dec.append(
            Conv2dNormAct(
                16, self.out_ch, (3, 3), fstride=1, norm_layer=norm_layer, activation_layer=out_act
            )
        )

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        # input shape: [B, C, T, F]
        intermediate = []
        for enc_layer in self.enc:
            x = enc_layer(x)
            intermediate.append(x)
        x_rnn, h = self.rnn(x, h)
        for dec_layer, x_enc in zip(self.dec, reversed(intermediate)):
            x = dec_layer(x + x_enc)
        return x, x_rnn, h


class ComplexCompression(nn.Module):
    def __init__(self, n_freqs: int, init_value: float = 0.5):
        super().__init__()
        self.c: Tensor
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
    def __init__(self, n_freqs: int, init_value: float = 0.5):
        super().__init__()
        self.c: Tensor
        self.register_parameter(
            "c", Parameter(torch.full((n_freqs,), init_value), requires_grad=True)
        )

    def forward(self, x: Tensor):
        # x has shape x [B, T, F, 2]
        x = x.pow(self.c)
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
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.stages = p.stages
        self.freq_bins = p.fft_size // 2 + 1
        self.erb_bins = p.nb_erb
        self.df_bins = p.nb_df
        self.erb_fb: Tensor
        self.erb_comp = MagCompression(self.erb_bins)
        self.cplx_comp = ComplexCompression(self.df_bins)
        self.register_buffer("erb_fb", erb_fb, persistent=False)
        assert p.erb_depth <= 6
        self.erb_stage = FreqStage(
            in_ch=1,
            out_ch=1,
            out_act=nn.Sigmoid,
            num_freqs=p.nb_erb,
            hidden_dim=p.erb_hidden_dim,
            width_mult=p.width_mult,
            depth_mult=p.depth_mult,
            depth=p.erb_depth,
        )
        self.mask = Mask(erb_inv_fb, post_filter=p.mask_pf)
        refinement_act = {"tanh": nn.Tanh, "identity": nn.Identity}[p.refinement_act.lower()]
        assert p.refinement_depth <= 6
        strides = [2, 2, 2, 2, 1, 2] if p.refinement_depth == 6 else None
        self.refinement_stage = FreqStage(
            in_ch=2,
            out_ch=2,
            out_act=refinement_act,
            num_freqs=p.nb_df,
            hidden_dim=p.refinement_hidden_dim,
            width_mult=p.width_mult,
            depth_mult=p.depth_mult,
            depth=p.refinement_depth,
            fstrides=strides,
        )
        self.refinement_op = ComplexMul() if p.refinement_op == "mul" else ComplexAdd()
        self.lsnr_net = LSNRNet(self.erb_stage.max_width, lsnr_min=p.lsnr_min, lsnr_max=p.lsnr_max)

    def forward(
        self, spec: Tensor, atten_lim: Optional[Tensor] = None, **kwargs  # type: ignore
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
