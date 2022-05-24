from abc import ABC, abstractmethod
from typing import Dict, Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MultiFrameModule(nn.Module, ABC):
    """Multi-frame speech enhancement modules.

    Signal model and notation:
        Noisy: `x = s + n`
        Enhanced: `y = f(x)`
        Objective: `min ||s - y||`

        PSD: Power spectral density, notated eg. as `Rxx` for noisy PSD.
        IFC: Inter-frame correlation vector: PSD*u, u: selection vector. Notated as `rxx`
        RTF: Relative transfere function, also called steering vector.
    """

    num_freqs: Final[int]
    frame_size: Final[int]
    need_unfold: Final[bool]

    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        """Multi-Frame filtering module.

        Args:
            num_freqs (int): Number of frequency bins used for filtering.
            frame_size (int): Frame size in FD domain.
            lookahead (int): Lookahead, may be used to select the output time step. Note: This
                module does not add additional padding according to lookahead!
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.pad = nn.ConstantPad2d((0, 0, frame_size - 1, 0), 0.0)
        self.need_unfold = frame_size > 1
        self.lookahead = lookahead

    def spec_unfold(self, spec: Tensor):
        """Pads and unfolds the spectrogram according to frame_size.

        Args:
            spec (complex Tensor): Spectrogram of shape [B, C, T, F]
        Returns:
            spec (Tensor): Unfolded spectrogram of shape [B, C, T, F, N], where N: frame_size.
        """
        if self.need_unfold:
            return self.pad(spec).unfold(2, self.frame_size, 1)
        return spec.unsqueeze(-1)

    def forward(self, spec: Tensor, coefs: Tensor):
        """Pads and unfolds the spectrogram and forwards to impl.

        Args:
            spec (Tensor): Spectrogram of shape [B, C, T, F, 2]
            coefs (Tensor): Spectrogram of shape [B, C, T, F, 2]
        """
        spec_u = self.spec_unfold(torch.view_as_complex(spec))
        coefs = torch.view_as_complex(coefs)
        spec_f = spec_u.narrow(-2, 0, self.num_freqs)
        spec_f = self.forward_impl(spec_f, coefs)
        if self.training:
            spec = spec.clone()
        spec[..., : self.num_freqs, :] = torch.view_as_real(spec_f)
        return spec

    @staticmethod
    def solve(Rxx, rss, diag_eps: float = 1e-8, eps: float = 1e-7) -> Tensor:
        return torch.einsum(
            "...nm,...m->...n", torch.inverse(_tik_reg(Rxx, diag_eps, eps)), rss
        )  # [T, F, N]

    @staticmethod
    def apply_coefs(spec: Tensor, coefs: Tensor) -> Tensor:
        # spec: [B, C, T, F, N]
        # coefs: [B, C, T, F, N]
        return torch.einsum("...n,...n->...", spec, coefs)

    @abstractmethod
    def forward_impl(self, spec: Tensor, coefs: Tensor) -> Tensor:
        """Forward impl taking complex spectrogram and coefficients.

        Args:
            spec (complex Tensor): Spectrogram of shape [B, C1, T, F, N]
            coefs (complex Tensor): Coefficients [B, C2, T, F]

        Returns:
            spec (complex Tensor): Enhanced spectrogram of shape [B, C1, T, F]
        """
        ...

    @abstractmethod
    def num_channels(self) -> int:
        """Return the number of required channels.

        If multiple inputs are required, then all these should be combined in one Tensor containing
        the summed channels.
        """
        ...


def psd(x: Tensor, n: int) -> Tensor:
    """Compute the PSD correlation matrix Rxx for a spectrogram.

    That is, `X*conj(X)`, where `*` is the outer product.

    Args:
        x (complex Tensor): Spectrogram of shape [B, C, T, F]. Will be unfolded with `n` steps over
            the time axis.

    Returns:
        Rxx (complex Tensor): Correlation matrix of shape [B, C, T, F, N, N]
    """
    x = F.pad(x, (0, 0, n - 1, 0)).unfold(-2, n, 1)
    return torch.einsum("...n,...m->...mn", x, x.conj())


def df(spec: Tensor, coefs: Tensor) -> Tensor:
    """Deep filter implemenation using `torch.einsum`. Requires unfolded spectrogram.

    Args:
        spec (complex Tensor): Spectrogram of shape [B, C, T, F, N]
        coefs (complex Tensor): Spectrogram of shape [B, C, N, T, F]

    Returns:
        spec (complex Tensor): Spectrogram of shape [B, C, T, F]
    """
    return torch.einsum("...tfn,...ntf->...tf", spec, coefs)


class CRM(MultiFrameModule):
    """Complex ratio mask."""

    def __init__(self, num_freqs: int, frame_size: int = 1, lookahead: int = 0):
        assert frame_size == 1 and lookahead == 0, (frame_size, lookahead)
        super().__init__(num_freqs, 1)

    def forward_impl(self, spec: Tensor, coefs: Tensor):
        return spec.squeeze(-1).mul(coefs)

    def num_channels(self):
        return 2


class DF(MultiFrameModule):
    conj: Final[bool]
    """Deep Filtering."""

    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0, conj: bool = False):
        super().__init__(num_freqs, frame_size, lookahead)
        self.conj = conj

    def forward_impl(self, spec: Tensor, coefs: Tensor):
        coefs = coefs.view(coefs.shape[0], -1, self.frame_size, *coefs.shape[2:])
        if self.conj:
            coefs = coefs.conj()
        return df(spec, coefs)

    def num_channels(self):
        return self.frame_size * 2


class MfWf(MultiFrameModule):
    """Multi-frame Wiener filter base module."""

    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        """Multi-frame Wiener Filter.

        Several implementation methods are available resulting in different number of required input
        coefficient channels.

        Methods:
            psd_ifc: Predict PSD `Rxx` and IFC `rss`.
            df: Use deep filtering to predict speech and noisy spectrograms. These will be used for
                PSD calculation for Wiener filtering. Alias: `df_sx`
            c: Directly predict Wiener filter coefficients. Computation same as deep filtering.

        """
        super().__init__(num_freqs, frame_size, lookahead=0)
        self.idx = -lookahead

    def num_channels(self):
        return self.num_channels

    @abstractmethod
    def mfwf(self, spec: Tensor, coefs: Tensor) -> Tensor:
        """Multi-frame Wiener filter impl taking complex spectrogram and coefficients.

        Coefficients may be split into multiple parts w.g. for multiple DF coefs or PSDs.

        Args:
            spec (complex Tensor): Spectrogram of shape [B, C1, T, F, N]
            coefs (complex Tensor): Coefficients [B, C2, T, F]

        Returns:
            c (complex Tensor): MfWf coefs of shape [B, C1, T, F, N]
        """
        ...

    def forward_impl(self, spec: Tensor, coefs: Tensor) -> Tensor:
        coefs = self.mfwf(spec, coefs)
        return self.apply_coefs(spec, coefs)


class MfWfDf(MfWf):
    eps_diag: Final[float]

    def __init__(
        self,
        num_freqs: int,
        frame_size: int,
        lookahead: int = 0,
        eps_diag: float = 1e-7,
        eps: float = 1e-7,
    ):
        super().__init__(num_freqs, frame_size, lookahead)
        self.eps_diag = eps_diag
        self.eps = eps

    def num_channels(self):
        # frame_size/df_order * 2 (x/s) * 2 (re/im)
        return self.frame_size * 4

    def mfwf(self, spec: Tensor, coefs: Tensor) -> Tensor:
        coefs.chunk
        df_s, df_x = torch.chunk(coefs, 2, 1)  # [B, C, T, F, N]
        df_s = df_s.unflatten(1, (-1, self.frame_size))
        df_x = df_x.unflatten(1, (-1, self.frame_size))
        spec_s = df(spec, df_s)  # [B, C, T, F]
        spec_x = df(spec, df_x)
        Rss = psd(spec_s, self.frame_size)  # [B, C, T, F, N. N]
        Rxx = psd(spec_x, self.frame_size)
        rss = Rss[..., -1]  # TODO: use -1 or self.idx?
        c = self.solve(Rxx, rss, self.eps_diag, self.eps)  # [B, C, T, F, N]
        return c


class MfWfPsd(MfWf):
    """Multi-frame Wiener filter by predicting noisy PSD `Rxx` and speech IFC `rss`."""

    def num_channels(self):
        # (Rxx + rss) * 2 (re/im)
        return (self.frame_size**2 + self.frame_size) * 2

    def mfwf(self, spec: Tensor, coefs: Tensor) -> Tensor:  # type: ignore
        Rxx, rss = torch.split(coefs.movedim(1, -1), [self.frame_size**2, self.frame_size], -1)
        c = self.solve(Rxx.unflatten(-1, (self.frame_size, self.frame_size)), rss)
        return c


class MfWfC(MfWf):
    """Multi-frame Wiener filter by directly predicting the MfWf coefficients."""

    def num_channels(self):
        # mfwf coefs * 2 (re/im)
        return self.frame_size * 2

    def mfwf(self, spec: Tensor, coefs: Tensor) -> Tensor:  # type: ignore
        coefs = coefs.unflatten(1, (-1, self.frame_size)).permute(
            0, 1, 3, 4, 2
        )  # [B, C*N, T, F] -> [B, C, T, F, N]
        return coefs


class MfMvdr(MultiFrameModule):
    """Multi-frame minimum variance distortionless beamformer."""

    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        """Multi-frame minimum variance distortionless beamformer.

        Several implementation methods are available resulting in different number of required input
        coefficient channels. Mostly based on torchaudio's MVDR implementations.

        Methods:
            mvdr_evd: Estimate the steering vector (RTF) based on eigen value decomposition (evd) of Rss.
            mvdr_rtf_power: Estimate the RTF based using the power method using Rss and Rnn.
            mvdr_souden: Estimate coefs based on Rss and Rnn using the Souden method.
        """
        super().__init__(num_freqs, frame_size, lookahead=0)
        self.idx = -lookahead

    def num_channels(self):
        return self.num_channels

    @abstractmethod
    def mvdr(self, spec: Tensor, coefs: Tensor) -> Tensor:
        """Minimum variance distortionless beamformer impl taking complex spectrogram and coefficients.

        Coefficients may be split into multiple parts w.g. for multiple DF coefs or PSDs.

        Args:
            spec (complex Tensor): Spectrogram of shape [B, C1, T, F, N]
            coefs (complex Tensor): Coefficients [B, C2, T, F]

        Returns:
            c (complex Tensor): MfWf coefs of shape [B, C1, T, F, N]
        """
        ...

    def forward_impl(self, spec: Tensor, coefs: Tensor) -> Tensor:
        coefs = self.mvdr(spec, coefs)
        return self.apply_coefs(spec, coefs)


class MvdrSouden(MultiFrameModule):
    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        super().__init__(num_freqs, frame_size, lookahead)


class MvdrEvd(MultiFrameModule):
    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        super().__init__(num_freqs, frame_size, lookahead)


class MvdrRtfPower(MultiFrameModule):
    def __init__(self, num_freqs: int, frame_size: int, lookahead: int = 0):
        super().__init__(num_freqs, frame_size, lookahead)


MF_METHODS: Dict[str, MultiFrameModule] = {  # type: ignore
    "crm": CRM,
    "df": DF,
    "mfwf_df": MfWfDf,
    "mfwf_df_sx": MfWfDf,
    "mfwf_psd": MfWfPsd,
    "mfwf_psd_ifc": MfWfPsd,
    "mfwf_c": MfWfC,
}


# From torchaudio
def _compute_mat_trace(input: torch.Tensor, dim1: int = -1, dim2: int = -2) -> torch.Tensor:
    r"""Compute the trace of a Tensor along ``dim1`` and ``dim2`` dimensions.
    Args:
        input (torch.Tensor): Tensor of dimension `(..., channel, channel)`
        dim1 (int, optional): the first dimension of the diagonal matrix
            (Default: -1)
        dim2 (int, optional): the second dimension of the diagonal matrix
            (Default: -2)
    Returns:
        Tensor: trace of the input Tensor
    """
    assert input.ndim >= 2, "The dimension of the tensor must be at least 2."
    assert (
        input.shape[dim1] == input.shape[dim2]
    ), "The size of ``dim1`` and ``dim2`` must be the same."
    input = torch.diagonal(input, 0, dim1=dim1, dim2=dim2)
    return input.sum(dim=-1)


def _tik_reg(mat: torch.Tensor, reg: float = 1e-7, eps: float = 1e-8) -> torch.Tensor:
    """Perform Tikhonov regularization (only modifying real part).
    Args:
        mat (torch.Tensor): input matrix (..., channel, channel)
        reg (float, optional): regularization factor (Default: 1e-8)
        eps (float, optional): a value to avoid the correlation matrix is all-zero (Default: ``1e-8``)
    Returns:
        Tensor: regularized matrix (..., channel, channel)
    """
    # Add eps
    C = mat.size(-1)
    eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
    epsilon = _compute_mat_trace(mat).real[..., None, None] * reg
    # in case that correlation_matrix is all-zero
    epsilon = epsilon + eps
    mat = mat + epsilon * eye[..., :, :]
    return mat
