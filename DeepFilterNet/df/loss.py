import warnings
from collections import defaultdict
from typing import Dict, Final, Iterable, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from df.config import Csv, config
from df.model import ModelParams
from df.modules import LocalSnrTarget, erb_fb
from df.stoi import stoi
from df.utils import angle, as_complex, get_device
from libdf import DF


def wg(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    N = X - S
    SS = as_complex(S).abs().square()
    NN = as_complex(N).abs().square()
    return (SS / (SS + NN + eps)).clamp(0, 1)


def irm(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    N = X - S
    SS_mag = as_complex(S).abs()
    NN_mag = as_complex(N).abs()
    return (SS_mag / (SS_mag + NN_mag + eps)).clamp(0, 1)


def iam(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    SS_mag = as_complex(S).abs()
    XX_mag = as_complex(X).abs()
    return (SS_mag / (XX_mag + eps)).clamp(0, 1)


class Stft(nn.Module):
    def __init__(self, n_fft: int, hop: Optional[int] = None, window: Optional[Tensor] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        if window is not None:
            assert window.shape[0] == n_fft
        else:
            window = torch.hann_window(self.n_fft)
        self.w: torch.Tensor
        self.register_buffer("w", window)

    def forward(self, input: Tensor):
        # Time-domain input shape: [B, *, T]
        t = input.shape[-1]
        sh = input.shape[:-1]
        out = torch.stft(
            input.reshape(-1, t),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.w,
            normalized=True,
            return_complex=True,
        )
        out = out.view(*sh, *out.shape[-2:])
        return out


class Istft(nn.Module):
    def __init__(self, n_fft_inv: int, hop_inv: int, window_inv: Tensor):
        super().__init__()
        # Synthesis back to time domain
        self.n_fft_inv = n_fft_inv
        self.hop_inv = hop_inv
        self.w_inv: torch.Tensor
        assert window_inv.shape[0] == n_fft_inv
        self.register_buffer("w_inv", window_inv)

    def forward(self, input: Tensor):
        # Input shape: [B, * T, F, (2)]
        input = as_complex(input)
        t, f = input.shape[-2:]
        sh = input.shape[:-2]
        # Even though this is not the DF implementation, it numerical sufficiently close.
        # Pad one extra step at the end to get original signal length
        out = torch.istft(
            F.pad(input.reshape(-1, t, f).transpose(1, 2), (0, 1)),
            n_fft=self.n_fft_inv,
            hop_length=self.hop_inv,
            window=self.w_inv,
            normalized=True,
        )
        if input.ndim > 2:
            out = out.view(*sh, out.shape[-1])
        return out


class MultiResSpecLoss(nn.Module):
    gamma: Final[float]
    f: Final[float]
    f_complex: Final[Optional[List[float]]]

    def __init__(
        self,
        n_ffts: Iterable[int],
        gamma: float = 1,
        factor: float = 1,
        f_complex: Optional[Union[float, Iterable[float]]] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.f = factor
        self.stfts = nn.ModuleDict({str(n_fft): Stft(n_fft) for n_fft in n_ffts})
        if f_complex is None or f_complex == 0:
            self.f_complex = None
        elif isinstance(f_complex, Iterable):
            self.f_complex = list(f_complex)
        else:
            self.f_complex = [f_complex] * len(self.stfts)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.zeros((), device=input.device, dtype=input.dtype)
        for i, stft in enumerate(self.stfts.values()):
            Y = stft(input)
            S = stft(target)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(1e-12).pow(self.gamma)
                S_abs = S_abs.clamp_min(1e-12).pow(self.gamma)
            loss += F.mse_loss(Y_abs, S_abs) * self.f
            if self.f_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle.apply(Y))
                    S = S_abs * torch.exp(1j * angle.apply(S))
                loss += F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S)) * self.f_complex[i]
        return loss


class SpectralLoss(nn.Module):
    gamma: Final[float]
    f_m: Final[float]
    f_c: Final[float]
    f_u: Final[float]

    def __init__(
        self,
        gamma: float = 1,
        factor_magnitude: float = 1,
        factor_complex: float = 1,
        factor_under: float = 1,
    ):
        super().__init__()
        self.gamma = gamma
        self.f_m = factor_magnitude
        self.f_c = factor_complex
        self.f_u = factor_under

    def forward(self, input, target):
        input = as_complex(input)
        target = as_complex(target)
        input_abs = input.abs()
        target_abs = target.abs()
        if self.gamma != 1:
            input_abs = input_abs.clamp_min(1e-12).pow(self.gamma)
            target_abs = target_abs.clamp_min(1e-12).pow(self.gamma)
        tmp = (input_abs - target_abs).pow(2)
        if self.f_u != 1:
            # Weighting if predicted abs is too low
            tmp *= torch.where(input_abs < target_abs, self.f_u, 1.0)
        loss = torch.mean(tmp) * self.f_m
        if self.f_c > 0:
            if self.gamma != 1:
                input = input_abs * torch.exp(1j * angle.apply(input))
                target = target_abs * torch.exp(1j * angle.apply(target))
            loss_c = (
                F.mse_loss(torch.view_as_real(input), target=torch.view_as_real(target)) * self.f_c
            )
            loss = loss + loss_c
        return loss


class MaskLoss(nn.Module):
    def __init__(
        self,
        df_state: DF,
        mask: str = "iam",
        gamma: float = 0.6,
        powers: List[int] = [2],
        factors: List[float] = [1],
        f_under: float = 1,
        eps=1e-12,
        factor=1,
        gamma_pred: Optional[float] = None,
    ):
        super().__init__()
        if mask == "wg":
            self.mask_fn = wg
        elif mask == "irm":
            self.mask_fn = irm
        elif mask == "iam":
            self.mask_fn = iam
        else:
            raise ValueError("Unsupported mask function.")
        self.gamma = gamma
        self.gamma_pred = gamma if gamma_pred is None else gamma_pred
        self.powers = powers
        self.factors = factors
        self.f_under = f_under
        self.eps = eps
        self.factor = factor
        self.erb_fb: Tensor
        self.erb_inv_fb: Tensor
        self.register_buffer("erb_fb", erb_fb(df_state.erb_widths(), ModelParams().sr))
        self.register_buffer(
            "erb_inv_fb", erb_fb(df_state.erb_widths(), ModelParams().sr, inverse=True)
        )

    def __repr__(self):
        s = f"MaskLoss {self.mask_fn} (gamma: {self.gamma}"
        for p, f in zip(self.powers, self.factors):
            s += f", p: {p}, f: {f}"
        s += ")"
        return s

    @torch.jit.export
    def erb_mask_compr(self, clean: Tensor, noisy: Tensor, compressed: bool = True) -> Tensor:
        mask = self.mask_fn(clean, noisy)
        mask = torch.matmul(mask, self.erb_fb).clamp_min(self.eps)
        if compressed:
            mask = mask.pow(self.gamma)
        return mask

    @torch.jit.export
    def erb_inv(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.erb_inv_fb)

    def forward(
        self, input: Tensor, clean: Tensor, noisy: Tensor, max_bin: Optional[Tensor] = None
    ) -> Tensor:
        # Input mask shape: [B, C, T, F]
        b, _, _, f = input.shape
        if not torch.isfinite(input).all():
            raise ValueError("Input is NaN")
        assert input.min() >= 0
        g_t = self.erb_mask_compr(clean, noisy, compressed=True)
        g_p = input.clamp_min(self.eps).pow(self.gamma_pred)
        loss = torch.zeros((), device=input.device)
        tmp = g_t.sub(g_p).pow(2)
        if self.f_under != 1:
            # Weighting if gains are too low
            tmp *= torch.where(g_p < g_t, self.f_under, 1.0)
        if max_bin is not None:
            m = torch.ones((b, 1, 1, f), device=input.device)
            for i, mb in enumerate(max_bin):
                m[i, ..., mb:] = 0
            tmp = tmp * m
        for power, factor in zip(self.powers, self.factors):
            # Reduce the 2 from .pow(2) above
            loss += tmp.clamp_min(1e-13).pow(power // 2).mean().mul(factor) * self.factor
        return loss.mean()


class DfAlphaLoss(nn.Module):
    """Add a penalty to use DF for very noisy segments.

    Starting from lsnr_thresh, the penalty is increased and has its maximum at lsnr_min.
    """

    factor: Final[float]
    lsnr_thresh: Final[float]
    lsnr_min: Final[float]

    def __init__(self, factor: float = 1, lsnr_thresh: float = -7.5, lsnr_min: float = -10.0):
        super().__init__()
        self.factor = factor
        self.lsnr_thresh = lsnr_thresh
        self.lsnr_min = lsnr_min

    def forward(self, pred_alpha: Tensor, target_lsnr: Tensor):
        # pred_alpha: [B, T, 1]
        # target_lsnr: [B, T]

        # loss for lsnr < -5 -> penalize DF usage
        w = self.lsnr_mapping(target_lsnr, self.lsnr_thresh, self.lsnr_min).view_as(pred_alpha)
        # tmp = w[target_lsnr > -7.5]
        # torch.testing.assert_allclose(tmp, torch.zeros_like(tmp))
        # tmp = w[target_lsnr < -10]
        # torch.testing.assert_allclose(tmp, torch.ones_like(tmp))
        l_off = (pred_alpha * w).square().mean()

        # loss for lsnr > 0
        w = self.lsnr_mapping(target_lsnr, self.lsnr_thresh + 2.5, 0.0).view_as(pred_alpha)
        # tmp = w[target_lsnr > 0]
        # torch.testing.assert_allclose(tmp, torch.ones_like(tmp))
        # tmp = w[target_lsnr < -5]
        # torch.testing.assert_allclose(tmp, torch.zeros_like(tmp))
        l_on = 0.1 * ((1 - pred_alpha) * w).abs().mean()
        return l_off + l_on

    def lsnr_mapping(
        self, lsnr: Tensor, lsnr_thresh: float, lsnr_min: Optional[float] = None
    ) -> Tensor:
        """Map lsnr_min to 1 and lsnr_thresh to 0"""
        # s = a * lsnr + b
        lsnr_min = float(self.lsnr_min) if lsnr_min is None else lsnr_min
        a_ = 1 / (lsnr_thresh - lsnr_min)
        b_ = -a_ * lsnr_min
        return 1 - torch.clamp(a_ * lsnr + b_, 0.0, 1.0)


class SiSdr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor):
        # Input shape: [B, T]
        eps = torch.finfo(input.dtype).eps
        t = input.shape[-1]
        target = target.reshape(-1, t)
        input = input.reshape(-1, t)
        # Einsum for batch vector dot product
        Rss: Tensor = torch.einsum("bi,bi->b", target, target).unsqueeze(-1)
        a: Tensor = torch.einsum("bi,bi->b", target, input).add(eps).unsqueeze(-1) / Rss.add(eps)
        e_true = a * target
        e_res = input - e_true
        Sss = e_true.square()
        Snn = e_res.square()
        # Only reduce over each sample. Supposed to be used when used as a metric.
        Sss = Sss.sum(-1)
        Snn = Snn.sum(-1)
        return 10 * torch.log10(Sss.add(eps) / Snn.add(eps))


class SdrLoss(nn.Module):
    def __init__(self, factor=0.2):
        super().__init__()
        self.factor = factor
        self.sdr = SiSdr()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.factor == 0:
            return torch.zeros((), device=input.device)
        # Input shape: [B, T]
        return -self.sdr(input, target).mean() * self.factor


class SegSdrLoss(nn.Module):
    def __init__(self, window_sizes: List[int], factor: float = 0.2, overlap: float = 0):
        # Window size in samples
        super().__init__()
        self.window_sizes = window_sizes
        self.factor = factor
        self.hop = 1 - overlap
        self.sdr = SiSdr()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Input shape: [B, T]
        if self.factor == 0:
            return torch.zeros((), device=input.device)
        loss = torch.zeros((), device=input.device)
        for ws in self.window_sizes:
            if ws > input.size(-1):
                warnings.warn(
                    f"Input size {input.size(-1)} smaller than window size. Adjusting window size."
                )
                ws = input.size(1)
            loss += self.sdr(
                input=input.unfold(-1, ws, int(self.hop * ws)).reshape(-1, ws),
                target=target.unfold(-1, ws, int(self.hop * ws)).reshape(-1, ws),
            ).mean()
        return -loss * self.factor


class LocalSnrLoss(nn.Module):
    def __init__(self, factor: float = 1):
        super().__init__()
        self.factor = factor

    def forward(self, input: Tensor, target_lsnr: Tensor):
        # input (freq-domain): [B, T, 1]
        input = input.squeeze(-1)
        return F.mse_loss(input, target_lsnr) * self.factor


class Loss(nn.Module):
    ml_f: Final[float]
    cal_f: Final[float]
    sl_f: Final[float]
    mrsl_f: Final[float]

    def __init__(self, state: DF, istft: Optional[Istft] = None):
        super().__init__()
        p = ModelParams()
        self.lsnr = LocalSnrTarget(ws=20, target_snr_range=[p.lsnr_min - 5, p.lsnr_max + 5])
        self.istft = istft  # Could also be used for sdr loss
        self.sr = p.sr
        self.fft_size = p.fft_size
        self.nb_df = p.nb_df
        self.store_losses = False
        self.summaries: Dict[str, List[Tensor]] = self.reset_summaries()
        # Mask Loss
        self.ml_f = config("factor", 0, float, section="MaskLoss")  # e.g. 1
        self.ml_gamma = config("gamma", 0.6, float, section="MaskLoss")
        self.ml_gamma_pred = config("gamma_pred", 0.6, float, section="MaskLoss")
        self.ml_f_under = config("f_under", 2, float, section="MaskLoss")
        self.ml = MaskLoss(
            state,
            "iam",
            factor=self.ml_f,
            f_under=self.ml_f_under,
            gamma=self.ml_gamma,
            gamma_pred=self.ml_gamma_pred,
            factors=[1, 10],
            powers=[2, 4],
        )
        # SpectralLoss
        self.sl_fm = config("factor_magnitude", 0, float, section="SpectralLoss")  # e.g. 1e4
        self.sl_fc = config("factor_complex", 0, float, section="SpectralLoss")
        self.sl_fu = config("factor_under", 1, float, section="SpectralLoss")
        self.sl_gamma = config("gamma", 1, float, section="SpectralLoss")
        self.sl_f = self.sl_fm + self.sl_fc
        if self.sl_f > 0:
            self.sl = SpectralLoss(
                factor_magnitude=self.sl_fm,
                factor_complex=self.sl_fc,
                factor_under=self.sl_fu,
                gamma=self.sl_gamma,
            )
        else:
            self.sl = None
        # Multi Resolution Spectrogram Loss
        self.mrsl_f = config("factor", 0, float, section="MultiResSpecLoss")
        self.mrsl_fc = config("factor_complex", 0, float, section="MultiResSpecLoss")
        self.mrsl_gamma = config("gamma", 1, float, section="MultiResSpecLoss")
        self.mrsl_ffts: List[int] = config("fft_sizes", [512, 1024, 2048], Csv(int), section="MultiResSpecLoss")  # type: ignore
        if self.mrsl_f > 0:
            assert istft is not None
            self.mrsl = MultiResSpecLoss(self.mrsl_ffts, self.mrsl_gamma, self.mrsl_f, self.mrsl_fc)
        else:
            self.mrsl = None
        self.sdrl_f = config("factor", 0, float, section="SdrLoss")
        self.sdrl = None
        if self.sdrl_f > 0:
            sdr_sgemental_ws = config("segmental_ws", [], Csv(int), section="SdrLoss")
            if len(sdr_sgemental_ws) > 0 and any(ws > 0 for ws in sdr_sgemental_ws):
                self.sdrl = SegSdrLoss(sdr_sgemental_ws, factor=self.sdrl_f)
            else:
                self.sdrl = SdrLoss(self.sdrl_f)
        self.lsnr_f = config("factor", 0.0005, float, section="LocalSnrLoss")
        self.lsnrl = LocalSnrLoss(self.lsnr_f) if self.lsnr_f > 0 else None
        self.dev_str = get_device().type

    def forward(
        self,
        clean: Tensor,
        noisy: Tensor,
        enhanced: Tensor,
        mask: Tensor,
        lsnr: Tensor,
        snrs: Tensor,
        max_freq: Optional[Tensor] = None,
        multi_stage_specs: List[Tensor] = [],
    ):
        max_bin: Optional[Tensor] = None
        if max_freq is not None:
            max_bin = (
                max_freq.to(device=clean.device)
                .mul(self.fft_size)
                .div(self.sr, rounding_mode="trunc")
            ).long()
        enhanced_td = None
        clean_td = None
        multi_stage = None
        multi_stage_td = None
        if multi_stage_specs:
            # Stack spectrograms in a channel dimension
            multi_stage = as_complex(torch.stack(multi_stage_specs, dim=1))
        lsnr_gt = self.lsnr(clean, noise=noisy - clean)
        if self.istft is not None:
            if self.store_losses or self.mrsl is not None or self.sdrl is not None:
                enhanced_td = self.istft(enhanced)
                clean_td = self.istft(clean)
                if multi_stage is not None:
                    # leave out erb enhanced
                    multi_stage_td = self.istft(multi_stage)

        ml, sl, mrsl, cal, sdrl, lsnrl = [torch.zeros((), device=clean.device)] * 6
        if self.ml_f != 0 and self.ml is not None:
            ml = self.ml(input=mask, clean=clean, noisy=noisy, max_bin=max_bin)
        if self.sl_f != 0 and self.sl is not None:
            sl = torch.zeros((), device=clean.device)
            if multi_stage is not None:
                sl += self.sl(input=multi_stage, target=clean.expand_as(multi_stage))
            else:
                sl = self.sl(input=enhanced, target=clean)
        if self.mrsl_f > 0 and self.mrsl is not None:
            if multi_stage_td is not None:
                ms = multi_stage_td[:, 1:]
                mrsl = self.mrsl(ms, clean_td.expand_as(ms))
            else:
                mrsl = self.mrsl(enhanced_td, clean_td)
        if self.lsnr_f != 0:
            lsnrl = self.lsnrl(input=lsnr, target_lsnr=lsnr_gt)
        if self.sdrl_f != 0:
            if multi_stage_td is not None:
                ms = multi_stage_td[:, 1:]
                sdrl = self.sdrl(ms, clean_td.expand_as(ms))
            else:
                sdrl = self.sdrl(enhanced_td, clean_td)
        if self.store_losses and self.istft is not None:
            assert enhanced_td is not None
            assert clean_td is not None
            self.store_summaries(
                enhanced_td,
                clean_td,
                snrs,
                ml,
                sl,
                mrsl,
                sdrl,
                lsnrl,
                cal,
                multi_stage_td=multi_stage_td,
            )
        return ml + sl + mrsl + sdrl + lsnrl + cal

    def reset_summaries(self):
        self.summaries = defaultdict(list)
        return self.summaries

    @torch.jit.ignore  # type: ignore
    def get_summaries(self):
        return self.summaries.items()

    @torch.no_grad()
    @torch.jit.ignore  # type: ignore
    def store_summaries(
        self,
        enh_td: Tensor,
        clean_td: Tensor,
        snrs: Tensor,
        ml: Tensor,
        sl: Tensor,
        mrsl: Tensor,
        sdrl: Tensor,
        lsnrl: Tensor,
        cal: Tensor,
        multi_stage_td: Optional[Tensor] = None,
    ):
        if ml != 0:
            self.summaries["MaskLoss"].append(ml.detach())
        if sl != 0:
            self.summaries["SpectralLoss"].append(sl.detach())
        if mrsl != 0:
            self.summaries["MultiResSpecLoss"].append(mrsl.detach())
        if sdrl != 0:
            self.summaries["SdrLoss"].append(sdrl.detach())
        if cal != 0:
            self.summaries["DfAlphaLoss"].append(cal.detach())
        if lsnrl != 0:
            self.summaries["LocalSnrLoss"].append(lsnrl.detach())
        sdr = SiSdr()
        enh_td = enh_td.squeeze(1).detach()
        clean_td = clean_td.squeeze(1).detach()
        sdr_vals: Tensor = sdr(enh_td, target=clean_td)
        stoi_vals: Tensor = stoi(y=enh_td, x=clean_td, fs_source=self.sr)
        sdr_vals_ms, stoi_vals_ms = [], []
        if multi_stage_td is not None:
            for i in range(multi_stage_td.shape[1]):
                sdr_vals_ms.append(sdr(multi_stage_td[:, i].detach(), clean_td))
                stoi_vals_ms.append(
                    stoi(y=multi_stage_td[:, i].detach(), x=clean_td, fs_source=self.sr)
                )
        for snr in torch.unique(snrs, sorted=False):
            self.summaries[f"sdr_snr_{snr.item()}"].extend(
                sdr_vals.masked_select(snr == snrs).detach().split(1)
            )
            self.summaries[f"stoi_snr_{snr.item()}"].extend(
                stoi_vals.masked_select(snr == snrs).detach().split(1)
            )
            for i, (sdr_i, stoi_i) in enumerate(zip(sdr_vals_ms, stoi_vals_ms)):
                self.summaries[f"sdr_stage_{i}_snr_{snr.item()}"].extend(
                    sdr_i.masked_select(snr == snrs).detach().split(1)
                )
                self.summaries[f"stoi_stage_{i}_snr_{snr.item()}"].extend(
                    stoi_i.masked_select(snr == snrs).detach().split(1)
                )


def test_local_snr():
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    from df.modules import local_snr
    from libdf import DF

    sr, fft, hop, nb_bands = 48000, 960, 480, 32
    df = DF(sr, fft, hop, nb_bands)
    clean, sr = librosa.load("assets/clean_freesound_33711.wav", sr=sr)
    noise, sr = librosa.load("assets/noise_freesound_573577.wav", sr=sr)
    clean = clean[: 3 * sr]
    noise = noise[: 3 * sr]

    noisy = df.analysis((noise + clean).reshape(1, -1))
    clean = df.analysis(clean.reshape(1, -1))
    noise = df.analysis(noise.reshape(1, -1))
    _, ax = plt.subplots(2)
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(noisy.squeeze().T)),
        sr=sr,
        hop_length=hop,
        x_axis="time",
        y_axis="hz",
        ax=ax[0],
    )
    ax[0].set_ylim(0, 10000)
    lsnr, esp, ens = local_snr(
        torch.from_numpy(clean).unsqueeze(0), torch.from_numpy(noise).unsqueeze(0), 4, True, 8
    )
    t = librosa.times_like(lsnr, sr, hop, fft)
    ax[1].plot(t, lsnr.clamp_min(-20).squeeze().numpy(), label="lsnr")
    ax1_ = ax[1].twinx()
    ax1_.plot(t, esp.squeeze().numpy(), "g", label="speech")
    ax1_.plot(t, ens.squeeze().numpy(), "r", label="noise")
    ax[1].legend()
    ax1_.legend()
    plt.savefig("out/noisy.pdf")
    noisy = df.synthesis(noisy)
    sf.write("out/noisy.wav", noisy.T, sr)
