from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.fx import wrap as fx_wrap
from torch.nn import functional as F
from typing_extensions import Final


def df_real_loop(
    order: int,
    lookahead: int,
    n_df_bins: int,
    spec: Tensor,
    coefs: Tensor,
    alpha: Optional[Tensor] = None,
    freq_bins: int = 0,  # noqa
) -> Tensor:
    # Version 0: Manual loop over order, maybe best for onnx export?
    b, _, t, _, _ = spec.shape
    f = n_df_bins
    padded = spec_pad(spec[..., :n_df_bins, :].squeeze(1), order, lookahead, dim=-3)
    spec_f = torch.zeros((b, t, f, 2), device=spec.device)
    for i in range(order):
        spec_f[..., 0] += padded[:, i : i + t, ..., 0] * coefs[:, :, i, :, 0]
        spec_f[..., 0] -= padded[:, i : i + t, ..., 1] * coefs[:, :, i, :, 1]
        spec_f[..., 1] += padded[:, i : i + t, ..., 1] * coefs[:, :, i, :, 0]
        spec_f[..., 1] += padded[:, i : i + t, ..., 0] * coefs[:, :, i, :, 1]
    return assign_df(spec, spec_f.unsqueeze(1), n_df_bins, alpha)


def df_real_strided(
    order: int,
    lookahead: int,
    n_df_bins: int,
    spec: Tensor,
    coefs: Tensor,
    alpha: Optional[Tensor] = None,
    freq_bins: int = 0,  # noqa
) -> Tensor:
    # Version1: Use as_strided instead of unfold
    # spec (real) [B, 1, T, F, 2], O: order
    # coefs (real) [B, T, O, F, 2]
    # alpha (real) [B, T, 1]
    padded = as_strided(spec[..., :n_df_bins, :].squeeze(1), order, lookahead, dim=-3)
    # Complex numbers are not supported by onnx
    re = padded[..., 0] * coefs[..., 0]
    re -= padded[..., 1] * coefs[..., 1]
    im = padded[..., 1] * coefs[..., 0]
    im += padded[..., 0] * coefs[..., 1]
    spec_f = torch.stack((re, im), -1).sum(2)
    return assign_df(spec, spec_f.unsqueeze(1), n_df_bins, alpha)


def df_real_unfold(
    order: int,
    lookahead: int,
    n_df_bins: int,
    spec: Tensor,
    coefs: Tensor,
    alpha: Optional[Tensor] = None,
    freq_bins: int = 0,  # noqa
) -> Tensor:
    # Version2: Unfold
    # spec (real) [B, 1, T, F, 2], O: order
    # coefs (real) [B, T, O, F, 2]
    # alpha (real) [B, T, 1]
    padded = spec_pad(spec[..., :n_df_bins, :].squeeze(1), order, lookahead, dim=-3)
    padded = padded.unfold(dimension=1, size=order, step=1)  # [B, T, F, 2, O]
    padded = padded.permute(0, 1, 4, 2, 3)
    spec_f = torch.empty_like(padded)
    spec_f[..., 0] = padded[..., 0] * coefs[..., 0]  # re1
    spec_f[..., 0] -= padded[..., 1] * coefs[..., 1]  # re2
    spec_f[..., 1] = padded[..., 1] * coefs[..., 0]  # im1
    spec_f[..., 1] += padded[..., 0] * coefs[..., 1]  # im2
    spec_f = spec_f.sum(dim=2)
    return assign_df(spec, spec_f.unsqueeze(1), n_df_bins, alpha)


def df_complex_strided(
    order: int,
    lookahead: int,
    n_df_bins: int,
    spec: Tensor,
    coefs: Tensor,
    alpha: Optional[Tensor] = None,
    freq_bins: int = 0,  # noqa
) -> Tensor:
    # Version3: Complex strided; definatly nicest, no permute, no indexing, but complex gradient
    # spec (real) [B, 1, T, F, 2], O: order
    # coefs (real) [B, T, O, F, 2]
    # alpha (real) [B, T, 1]
    padded = as_strided(spec[..., :n_df_bins, :].squeeze(1), order, lookahead, dim=-3)
    spec_f = torch.sum(torch.view_as_complex(padded) * torch.view_as_complex(coefs), dim=2)
    spec_f = torch.view_as_real(spec_f)
    return assign_df(spec, spec_f.unsqueeze(1), n_df_bins, alpha)


def df_init_hidden_state_buffer(
    order: int, freq_bins: int, device: torch.device = torch.device("cpu")
) -> Tensor:
    return torch.zeros(order, freq_bins, 2, device=device)


def df_delay_spec(spec: Tensor, delay: int) -> Tensor:
    # spec (real) [B, 1, T, F', 2]
    return F.pad(spec, (0, 0, 0, 0, 0, delay))


def df_hidden_state_init(
    order: int,
    lookahead: int,
    spec: Tensor,
    freq_bins: int,
    spec_buf: Optional[Tensor] = None,
) -> Tensor:
    if spec_buf is None:
        # Init and pre-fill spec buffer
        spec_buf = df_init_hidden_state_buffer(order, freq_bins, spec.device)
        for i in range(lookahead):
            spec_buf[order - i - 1] = spec[i]
    return spec_buf


def df_real_hidden_state_loop(
    order: int,
    lookahead: int,
    n_df_bins: int,
    spec: Tensor,
    coefs: Tensor,
    alpha: Tensor,
    freq_bins: int,
    spec_buf: Optional[Tensor] = None,
) -> Tensor:
    # Version4: Designed for onnx export. `spec` buffer handling is done via a state (spec_buf).

    # spec (real) [T, F', 2]
    # coefs (real) [T, O, F, 2]
    # alpha [T, 1]
    t, _, _ = spec.shape

    spec = df_delay_spec(spec, lookahead)
    # spec_buf (real) [O, F, 2]
    spec_buf = df_hidden_state_init(order, lookahead, spec, freq_bins, spec_buf=spec_buf)
    spec_out = torch.empty((t, freq_bins, 2), device=spec.device)
    for i in range(t):
        spec_out[i], spec_buf = df_real_hidden_state_step(
            order,
            lookahead,
            n_df_bins,
            spec[i + lookahead],
            coefs[i],
            alpha[i],
            spec_buf=spec_buf,
        )
    return spec_out


def df_real_hidden_state_step(
    order: int,
    lookahead: int,
    n_df_bins: int,
    spec: Tensor,
    coefs: Tensor,
    alpha: Tensor,
    spec_buf: Tensor,
) -> Tuple[Tensor, Tensor]:
    # spec (real) [F, 2]
    # coefs (real) [O, F', 2]
    # alpha [1]
    # spec_buf (real) [O, F', 2]
    f = spec.shape[0]
    # TODO: This needs pytorch nightly (1.10)
    spec_buf = spec_buf.roll(-1, dims=0).detach()
    spec_buf[order - 1] = spec
    spec_in = spec_buf[:, :n_df_bins]
    sre, sim = spec_in.split(1, dim=2)
    cre, cim = coefs.split(1, dim=2)
    outr = torch.sum(sre * cre - sim * cim, dim=0)
    outi = torch.sum(sre * cim + sim * cre, dim=0)
    spec_f = torch.cat((outr, outi), dim=1)
    # TODO: This results in tract rank errors during scaternd
    # spec_out = spec_buf[order - 1 - lookahead].clone()
    # spec_out[:n_df_bins] = spec_f * alpha + spec_out[:n_df_bins] * (1 - alpha)
    spec_out_f, spec_out_g = spec_buf[order - 1 - lookahead].split(
        (n_df_bins, f - n_df_bins), dim=0
    )
    spec_out_f = spec_f * alpha + spec_out_f * (1 - alpha)
    spec_out = torch.cat((spec_out_f, spec_out_g), dim=0)
    return spec_out, spec_buf


def assign_df(spec: Tensor, spec_f: Tensor, df_bins: int, alpha: Optional[Tensor]):
    spec_out = spec.clone()
    if alpha is not None:
        b = spec.shape[0]
        alpha = alpha.view(b, 1, -1, 1, 1)
        spec_out[..., :df_bins, :] = spec_f * alpha + spec[..., :df_bins, :] * (1 - alpha)
    else:
        spec_out[..., :df_bins, :] = spec_f
    return spec_out


class DfOpCoefLoop(nn.Module):
    order: Final[int]
    lookahead: Final[int]
    n_df_bins: Final[int]

    def __init__(
        self, order: int, lookahead: int, n_df_bins: int, n_freq_bins: Optional[int] = None
    ):
        super().__init__()
        self.order = order
        self.lookahead = lookahead
        self.n_df_bins = n_df_bins
        self.n_freq_bins = n_freq_bins

    def forward(self, spec: Tensor, coefs: Tensor, alpha: Tensor):
        return df_real_loop(
            self.order,
            self.lookahead,
            self.n_df_bins,
            spec=spec,
            coefs=coefs,
            alpha=alpha,
        )


class DfOpStrided(nn.Module):
    order: Final[int]
    lookahead: Final[int]
    n_df_bins: Final[int]

    def __init__(
        self, order: int, lookahead: int, n_df_bins: int, n_freq_bins: Optional[int] = None
    ):
        super().__init__()
        self.order = order
        self.lookahead = lookahead
        self.n_df_bins = n_df_bins
        self.n_freq_bins = n_freq_bins

    def forward(self, spec: Tensor, coefs: Tensor, alpha: Tensor):
        return df_real_strided(
            self.order,
            self.lookahead,
            self.n_df_bins,
            spec=spec,
            coefs=coefs,
            alpha=alpha,
        )


class DfOpComplexStrided(nn.Module):
    order: Final[int]
    lookahead: Final[int]
    n_df_bins: Final[int]

    def __init__(
        self, order: int, lookahead: int, n_df_bins: int, n_freq_bins: Optional[int] = None
    ):
        super().__init__()
        self.order = order
        self.lookahead = lookahead
        self.n_df_bins = n_df_bins
        self.n_freq_bins = n_freq_bins

    def forward(self, spec: Tensor, coefs: Tensor, alpha: Tensor):
        return df_complex_strided(
            self.order,
            self.lookahead,
            self.n_df_bins,
            spec=spec,
            coefs=coefs,
            alpha=alpha,
        )


class DfOpUnfold(nn.Module):
    order: Final[int]
    lookahead: Final[int]
    n_df_bins: Final[int]

    def __init__(
        self, order: int, lookahead: int, n_df_bins: int, n_freq_bins: Optional[int] = None
    ):
        super().__init__()
        self.order = order
        self.lookahead = lookahead
        self.n_df_bins = n_df_bins
        self.n_freq_bins = n_freq_bins

    def forward(self, spec: Tensor, coefs: Tensor, alpha: Tensor):
        return df_real_unfold(
            self.order,
            self.lookahead,
            self.n_df_bins,
            spec=spec,
            coefs=coefs,
            alpha=alpha,
        )


class DfOpTimeLoop(nn.Module):
    order: Final[int]
    lookahead: Final[int]
    n_df_bins: Final[int]
    n_freq_bins: Final[int]

    def __init__(self, order: int, lookahead: int, n_df_bins: int, n_freq_bins: int):
        super().__init__()
        self.order = order
        self.lookahead = lookahead
        self.n_df_bins = n_df_bins
        self.n_freq_bins = n_freq_bins

    def forward(self, spec: Tensor, coefs: Tensor, alpha: Tensor):
        return df_real_hidden_state_loop(
            self.order,
            self.lookahead,
            self.n_df_bins,
            spec=spec,
            coefs=coefs,
            alpha=alpha,
            freq_bins=self.n_freq_bins,
        )


class DfDelaySpec(nn.Module):
    lookahead: Final[int]

    def __init__(self, lookahead: int):
        super().__init__()
        self.lookahead = lookahead

    def forward(self, spec: Tensor) -> Tensor:
        # Takes a spectrogram, delays it for `lookahead` steps and initializes the spec buffer.
        return df_delay_spec(spec, self.lookahead)


class DfOpInitSpecBuf(nn.Module):
    order: Final[int]
    lookahead: Final[int]
    n_df_bins: Final[int]
    n_freq_bins: Final[int]

    def __init__(self, order: int, lookahead: int, n_df_bins: int, n_freq_bins: int):
        super().__init__()
        self.order = order
        self.lookahead = lookahead
        self.n_df_bins = n_df_bins
        self.n_freq_bins = n_freq_bins

    def forward(self, spec: Tensor, spec_buf: Optional[Tensor] = None) -> Tensor:
        # Takes a spectrogram, delays it for `lookahead` steps and initializes the spec buffer.
        return df_hidden_state_init(
            self.order, self.lookahead, spec, self.n_freq_bins, spec_buf=spec_buf
        )


class DfOpTimeStep(nn.Module):
    order: Final[int]
    lookahead: Final[int]
    n_df_bins: Final[int]
    n_freq_bins: Final[int]

    def __init__(self, order: int, lookahead: int, n_df_bins: int, n_freq_bins: int):
        super().__init__()
        self.order = order
        self.lookahead = lookahead
        self.n_df_bins = n_df_bins
        self.n_freq_bins = n_freq_bins

    def forward(self, spec: Tensor, coefs: Tensor, alpha: Tensor, spec_buf: Tensor):
        return df_real_hidden_state_step(
            self.order,
            self.lookahead,
            self.n_df_bins,
            spec=spec,
            coefs=coefs,
            alpha=alpha,
            spec_buf=spec_buf,
        )


DF_OP_MAPPING = {
    "real_loop": DfOpCoefLoop,
    "real_strided": DfOpStrided,
    "real_unfold": DfOpUnfold,
    "forward_complex_strided": DfOpComplexStrided,
    "real_hidden_state_loop": DfOpTimeLoop,
    "real_hidden_state_step_init": DfOpInitSpecBuf,
}


def get_df_op(key: str):
    keys = list(DF_OP_MAPPING.keys())
    if key not in keys:
        raise ValueError(f"DfOp key must be one of `{keys}` but got `{key}`")
    return DF_OP_MAPPING[key]


@fx_wrap
def spec_pad(x: Tensor, window_size: int, lookahead: int, dim: int = 0) -> Tensor:
    pad = [0] * x.dim() * 2
    if dim >= 0:
        pad[(x.dim() - dim - 1) * 2] = window_size - lookahead - 1
        pad[(x.dim() - dim - 1) * 2 + 1] = lookahead
    else:
        pad[(-dim - 1) * 2] = window_size - lookahead - 1
        pad[(-dim - 1) * 2 + 1] = lookahead
    return F.pad(x, pad)


@fx_wrap
def as_strided(x: Tensor, window_size: int, lookahead: int, step: int = 1, dim: int = 0) -> Tensor:
    shape = list(x.shape)
    shape.insert(dim + 1, window_size)
    x = spec_pad(x, window_size, lookahead, dim=dim)
    # torch.fx workaround
    step = 1
    stride = [x.stride(0), x.stride(1), x.stride(2), x.stride(3)]
    stride.insert(dim, stride[dim] * step)
    return torch.as_strided(x, shape, stride)


def test_dfop():
    from df.config import config
    from df.model import ModelParams

    config.use_defaults()
    p = ModelParams()
    f = p.nb_df
    F = f * 2
    o = p.df_order
    d = 1  # lookahead of 1 step
    t = 10
    ic(f, F, o, d, t)
    spec = torch.randn(1, 1, t, F, 2)
    coefs = torch.randn(1, t, o, f, 2)
    alpha = torch.randn(1, t, 1)
    out1 = df_real_loop(o, d, p.nb_df, spec, coefs, alpha)
    out2 = df_real_strided(o, d, p.nb_df, spec, coefs, alpha)
    torch.testing.assert_close(out1, out2)
    out3 = df_real_unfold(o, d, p.nb_df, spec, coefs, alpha)
    torch.testing.assert_close(out1, out3)
    out4 = df_complex_strided(o, d, p.nb_df, spec, coefs, alpha)
    torch.testing.assert_close(out1, out4)
    # 5 only supports batch size of 1
    spec = spec[0].squeeze(0)
    coefs = coefs[0].squeeze(0)
    alpha = alpha[0]
    out5 = df_real_hidden_state_loop(o, d, p.nb_df, spec, coefs, alpha, freq_bins=F)
    torch.testing.assert_close(out1[0].squeeze(0), out5)
