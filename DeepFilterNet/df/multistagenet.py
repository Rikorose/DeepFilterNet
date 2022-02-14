from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from timm.models.fx_features import register_notrace_module
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_
from timm.models.layers.drop import DropPath
from timm.models.layers.mlp import ConvMlp, Mlp
from torch import Tensor, nn

from df.config import Csv, DfParams, config
from df.modules import Mask, erb_fb
from df.utils import get_device
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
        self.conv_ch: int = config("CONV_CH", cast=int, default=16, section=self.section)
        self.erb_hidden_dim: int = config(
            "ERB_HIDDEN_DIM", cast=int, default=64, section=self.section
        )
        self.refinement_hidden_dim: int = config(
            "REFINEMENT_HIDDEN_DIM", cast=int, default=96, section=self.section
        )
        self.refinement_act: str = (
            config("REFINEMENT_OUTPUT_ACT", default="identity", section=self.section)
            .lower()
            .replace("none", "identity")
        )
        self.gru_groups: int = config("GRU_GROUPS", cast=int, default=1, section=self.section)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=self.section)
        self.group_shuffle: bool = config(
            "GROUP_SHUFFLE", cast=bool, default=False, section=self.section
        )
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]],
        lookahead: int = 0,
        fstride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        fpad: bool = True,
    ):
        """Causal Conv2d by delaying the signal for any lookahead.

        Expected input format: [B, C, T, F]
        """
        super().__init__()
        # Padding on time axis
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        if fpad:
            fpad_ = kernel_size[1] // 2 + dilation - 1
        else:
            fpad_ = 0
        self.pad = nn.ConstantPad2d((0, 0, lookahead, kernel_size[0] - 1 - lookahead), 0.0)
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=(0, fpad_),
            stride=(1, fstride),  # Stride over time is always 1
            dilation=(1, dilation),  # Same for dilation
            groups=groups,
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
        kernel_size: Union[int, Tuple[int, int]],
        fstride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        fpad: bool = True,
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
        self.pad = nn.ConstantPad2d((0, 0, 0, kernel_size[0] - 1), 0.0)
        self.conv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=(kernel_size[0] - 1, fpad_ + dilation - 1),
            output_padding=(0, fpad_),
            stride=(1, fstride),  # Stride over time is always 1
            dilation=(1, dilation),
            groups=groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = self.conv(x)
        return x


class GruMlp(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, *args, **kwargs):
        super().__init__()
        kwargs["batch_first"] = True
        self.gru = nn.GRU(input_size, hidden_size, *args, **kwargs)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

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


@register_notrace_module
class LayerNorm2d(nn.LayerNorm):
    r"""LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block copied and modified from `timm`.
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        drop_path=0.0,
        ls_init_value=1e-6,
        conv_mlp=False,
        mlp_ratio=4,
        norm_layer=None,
        kernel_size=7,
    ):
        super().__init__()
        if not norm_layer:
            norm_layer = (
                partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
            )
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp
        self.conv_dw = Conv2d(dim, dim, kernel_size=kernel_size, groups=dim)  # depthwise conv
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class ConvNeXtStage(nn.Module):
    """ConvNeXt Stage copied and modified from `timm`."""

    def __init__(
        self,
        in_chs,
        out_chs,
        stride=2,
        depth=2,
        dp_rates=None,
        ls_init_value=1.0,
        conv_mlp=False,
        norm_layer=None,
        cl_norm_layer=None,
        transpose=False,
    ):
        super().__init__()

        # Up/downsampling without grouping
        if in_chs != out_chs or stride > 1:
            assert norm_layer is not None
            # Either upsample or downsample
            if transpose:
                self.strided_conv = nn.Sequential(
                    norm_layer(in_chs),
                    ConvTranspose2d(
                        in_chs, out_chs, kernel_size=stride, fstride=stride, fpad=False
                    ),
                )
            else:
                self.strided_conv = nn.Sequential(
                    norm_layer(in_chs),
                    Conv2d(in_chs, out_chs, kernel_size=stride, fstride=stride, fpad=False),
                )
        else:
            self.strided_conv = nn.Identity()

        dp_rates = dp_rates or [0.0] * depth
        self.blocks = nn.Sequential(
            *(
                ConvNeXtBlock(
                    dim=out_chs,
                    drop_path=dp_rates[j],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    norm_layer=norm_layer if conv_mlp else cl_norm_layer,
                )
                for j in range(depth)
            )
        )

    def forward(self, x):
        x = self.strided_conv(x)
        x = self.blocks(x)
        return x


class ConvOut(nn.Module):
    def __init__(self, patch_size: int, in_ch: int, out_ch: int, act):
        super().__init__()
        self.convt0 = ConvTranspose2d(
            in_ch, in_ch, kernel_size=(1, patch_size), fstride=patch_size, fpad=False
        )
        self.norm = LayerNorm2d(in_ch)
        self.conv1 = Conv2d(in_ch, out_ch, kernel_size=(2, 3))
        self.act: nn.Module = act()

    def forward(self, x):
        x = self.convt0(x)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        return x


class LSNRNet(nn.Module):
    def __init__(self, in_ch: int, hidden_dim: int = 16, lsnr_min: int = -15, lsnr_max: int = 40):
        super().__init__()
        self.conv = Conv2d(in_ch, in_ch, kernel_size=(1, 3), fstride=2, groups=in_ch)
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
        out_act,
        width: int,
        num_freqs: int,
        hidden_dim: int,
        depth: int = 3,
        lookahead: int = 0,
        patch_size: int = 2,
        conv_mlp: bool = False,
        downsample_hprev: bool = False,
        layer_scale=1e-6,
        out_init_scale=1,
    ):
        super().__init__()
        self.lw = width  # Layer width
        self.fe = num_freqs  # Number of frequency bins in embedding
        self.hd = hidden_dim
        if self.fe % (patch_size * 2) != 0:
            raise ValueError(
                f"num_freqs ({num_freqs}) must be dividable by overall stride. "
                f"Match the number of frequencies to be a multiple of {patch_size*2} "
                "or reduce the number of stages."
            )

        norm_layer = partial(LayerNorm2d, eps=1e-6)
        cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        self.conv_in = nn.Sequential(
            Conv2d(
                in_ch,
                self.lw,
                kernel_size=(1, patch_size),
                lookahead=lookahead,
                fstride=patch_size,
                fpad=False,
            ),
            norm_layer(self.lw),
        )
        # Just one stage for ERB
        depth_up = depth // 2
        depth_down = depth - depth_up
        self.down_block = ConvNeXtStage(
            self.lw,
            self.lw * 2,
            stride=2,
            depth=depth_down,
            dp_rates=None,
            ls_init_value=layer_scale,
            conv_mlp=conv_mlp,
            norm_layer=norm_layer,
            cl_norm_layer=cl_norm_layer,
        )
        self.conv_hprev_down = None
        if downsample_hprev:
            self.conv_hprev_down = Conv2d(
                self.lw * 2, self.lw * 2, (1, 3), fstride=2, groups=self.lw * 2
            )
        in_hd = self.fe // patch_size // 2 * self.lw * 2
        self.gru = GruMlp(in_hd, self.hd, in_hd)
        self.up_block = ConvNeXtStage(
            self.lw * 2,
            self.lw,
            stride=2,
            depth=depth_up,
            dp_rates=None,
            ls_init_value=layer_scale,
            conv_mlp=conv_mlp,
            norm_layer=norm_layer,
            cl_norm_layer=cl_norm_layer,
            transpose=True,
        )
        self.conv_out = ConvOut(patch_size, self.lw, out_ch, out_act)
        named_apply(partial(_init_weights, out_init_scale=out_init_scale), self)

    def forward(
        self, input: Tensor, h_prev: Optional[Tensor] = None, h: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # input shape: [B, 1, T, F]
        x0 = self.conv_in(input)
        x1 = self.down_block(x0)
        if h_prev is not None:
            assert self.conv_hprev_down is not None
            h_prev = self.conv_hprev_down(h_prev)
            x1 = x1 + h_prev
        x_rnn, h = self.gru(x1, h)
        x_rnn = x_rnn + x1
        x1 = self.up_block(x_rnn)
        x1 = x1 + x0
        m = self.conv_out(x1)
        return m, x_rnn, h


def _init_weights(module, name=None, out_init_scale=1.0):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.LayerNorm)):
        trunc_normal_(module.weight, std=0.02)
        if hasattr(module, "bias"):
            nn.init.constant_(module.bias, 0)
        if name and "conv_out." in name and isinstance(module, nn.Conv2d):
            module.weight.data.mul_(out_init_scale)
            if hasattr(module, "bias"):
                module.bias.data.mul_(out_init_scale)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        nn.init.constant_(module.bias, 0)


class MSNet(nn.Module):
    def __init__(self, erb_inv_fb: Tensor):
        super().__init__()
        p = ModelParams()
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.stages = p.stages
        self.freq_bins = p.fft_size // 2 + 1
        self.erb_bins = p.nb_erb
        self.df_bins = p.nb_df
        self.erb_stage = FreqStage(1, 1, nn.Sigmoid, p.conv_ch, p.nb_erb, p.erb_hidden_dim, depth=3)
        self.mask = Mask(erb_inv_fb, post_filter=p.mask_pf)
        refinement_act = {"tanh": nn.Tanh, "identity": nn.Identity}[p.refinement_act.lower()]
        self.refinement_stages = nn.ModuleList(
            [
                FreqStage(
                    in_ch=2,
                    out_ch=2,
                    out_act=refinement_act,
                    width=p.conv_ch,
                    num_freqs=p.nb_df,
                    hidden_dim=p.refinement_hidden_dim,
                    depth=depth,
                    patch_size=2 ** (i + 1),
                    downsample_hprev=i >= 1,
                    out_init_scale=ic(10 ** -(i + 2)),
                )
                for i, depth in enumerate(self.stages)
            ]
        )
        self.lsnr_net = LSNRNet(p.conv_ch * 2, lsnr_min=p.lsnr_min, lsnr_max=p.lsnr_max)
        # SNR offsets on which each refinement layer is activated
        self.refinement_snr_min = -10
        self.refinement_snr_max = (100, 10, 5, 0, -5, -5, -5, -5)
        # Add a bunch of '-5' SNRs to support currently a maximum of 8 refinement layers.
        assert len(self.stages) <= 8

    def forward(
        self, spec: Tensor, feat_erb: Tensor, feat_spec: Tensor, atten_lim: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        # Set memory format so stride represents NHWC order
        m, x_rnn, h_erb = self.erb_stage(feat_erb)
        spec = self.mask(spec, m, atten_lim)  # [B, 1, T, F, 2]
        lsnr, _ = self.lsnr_net(x_rnn)
        out_specs = [spec]
        # re/im into channel axis
        spec_f = spec.squeeze(1)[:, :, : self.df_bins].permute(0, 3, 1, 2)  # [B, 2, T, F_df]
        h_conv: Optional[Tensor] = None
        for stage, _lim in zip(self.refinement_stages, self.refinement_snr_max):
            refinement, h_conv, _ = stage(spec_f, h_conv)
            spec_f = spec_f + refinement
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
