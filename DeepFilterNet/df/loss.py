import warnings
from collections import defaultdict
from typing import Dict, Final, Iterable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from df.config import Csv, config
from df.io import resample
from df.model import ModelParams
from df.modules import LocalSnrTarget, Mask, erb_fb
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
        factor: float = 1.0,
        gamma_pred: Optional[float] = None,
        f_max_idx: Optional[int] = None,  # Maximum frequency bin index
    ):
        super().__init__()
        if mask == "wg":
            self.mask_fn = wg
        elif mask == "irm":
            self.mask_fn = irm
        elif mask == "iam":
            self.mask_fn = iam
        elif mask == "spec":
            self.mask_fn = None
        else:
            raise ValueError(f"Unsupported mask function: {mask}.")
        self.gamma = gamma
        self.gamma_pred = gamma if gamma_pred is None else gamma_pred
        self.powers = powers
        self.factors = factors
        self.f_under = f_under
        self.eps = eps
        self.factor = factor
        self.f_max_idx = f_max_idx
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
        mask_fn = self.mask_fn or iam
        mask = mask_fn(clean, noisy)
        mask = self.erb(mask)
        if compressed:
            mask = mask.pow(self.gamma)
        return mask

    @torch.jit.export
    def erb(self, x: Tensor, clamp_min: Optional[float] = None) -> Tensor:
        x = torch.matmul(x, self.erb_fb)
        if clamp_min is not None:
            x = x.clamp_min(clamp_min)
        return x

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
        if self.mask_fn is not None:
            g_t = self.erb_mask_compr(clean, noisy, compressed=True)
            g_p = input.clamp_min(self.eps).pow(self.gamma_pred)
        else:
            g_t = self.erb(clean.abs()).pow(self.gamma)  # We use directly the clean spectrum
            g_p = (self.erb(noisy.abs()) * input).pow(self.gamma_pred)
        loss = torch.zeros((), device=input.device)
        if self.f_max_idx is not None:
            g_t = g_t[..., : self.f_max_idx]
            g_p = g_p[..., : self.f_max_idx]
        tmp = g_t.sub(g_p).pow(2)
        if self.f_under != 1:
            # Weighting if gains are too low
            tmp = tmp * torch.where(g_p < g_t, self.f_under, 1.0)
        if max_bin is not None:
            m = torch.ones((b, 1, 1, f), device=input.device)
            for i, mb in enumerate(max_bin):
                m[i, ..., mb:] = 0
            tmp = tmp * m
        for power, factor in zip(self.powers, self.factors):
            # Reduce the 2 from .pow(2) above
            loss += tmp.clamp_min(1e-13).pow(power // 2).mean().mul(factor) * self.factor
        return loss.mean()


class MaskSpecLoss(nn.Module):
    def __init__(
        self, df_state: DF, factor=1.0, gamma: float = 0.6, f_max_idx: Optional[int] = None
    ):
        super().__init__()
        self.f_max_idx = f_max_idx
        self.apply_mask = Mask(erb_fb(df_state.erb_widths(), ModelParams().sr, inverse=True))
        self.loss = SpectralLoss(factor_magnitude=factor, gamma=gamma)

    def forward(self, input: Tensor, clean: Tensor, noisy: Tensor) -> Tensor:
        enh = self.apply_mask(noisy, input)
        if self.f_max_idx is not None:
            enh = enh[..., : self.f_max_idx]
            clean = clean[..., : self.f_max_idx]
        return self.loss(enh, clean)


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


class ASRLoss(nn.Module):
    target_sr = 16000
    n_fft = 400
    hop = 160
    beam_size = 20
    lang = "en"
    task = "transcribe"
    max_ctx = 25

    def __init__(
        self,
        sr: int,
        factor: float = 1,
        factor_lm: float = 1,
        loss_lm: Literal["CTC", "CrossEntropy"] = "CrossEntropy",
        model: str = "base.en",
    ) -> None:
        super().__init__()
        import whisper

        self.sr = sr
        self.factor = factor
        self.factor_lm = factor_lm
        self.model = whisper.load_model(model)
        self.model.requires_grad_(False)
        self.options = whisper.DecodingOptions(
            task=self.task, language=self.lang, without_timestamps=True, sample_len=self.max_ctx
        )
        self.mel_filters: Tensor
        self.register_buffer(
            "mel_filters", torch.from_numpy(self.get_mel_filters(self.target_sr, 400, 80))
        )
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            self.model.is_multilingual, language=self.lang, task=self.options.task
        )
        self.decoder = whisper.decoding.GreedyDecoder(0.0, self.tokenizer.eot)
        # self.decoder = whisper.decoding.BeamSearchDecoder(self.beam_size, self.tokenizer.eot, , 1.)
        self.sot_sequence = self.tokenizer.sot_sequence_including_notimestamps
        self.n_ctx: int = self.model.dims.n_text_ctx
        self.initial_tokens = self._get_initial_tokens()
        self.sot_index: int = self.initial_tokens.index(self.tokenizer.sot)
        self.sample_begin: int = len(self.initial_tokens)
        self.sample_len: int = self.options.sample_len or self.model.dims.n_text_ctx // 2
        self.blank = self.tokenizer.encode(" ")[0]
        self.eot = self.tokenizer.eot
        self.loss_lm = loss_lm

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        features_i = self.model.embed_audio(self.preprocess(input))
        features_t = self.model.embed_audio(self.preprocess(target))
        # Loss based on the audio encoding:
        loss = 0
        if self.factor > 0:
            loss = F.mse_loss(features_i[0], features_t[0]) * self.factor
        if self.factor_lm > 0:
            _, tokens_t = self.decode_tokens(features_t)  # [N, S]
            logits_i, tokens_i = self.decode_tokens(features_i)  # [N, T, C]
            log_probs_i = F.log_softmax(logits_i, dim=-1)

            # Loss based on the logits:
            if self.factor_lm > 0:
                if self.loss_lm == "CTC":
                    input_lengths = torch.as_tensor(
                        [torch.argwhere(t == self.eot)[0] for t in tokens_i],
                        device=input.device,
                        dtype=torch.long,
                    )
                    target_lengths = torch.as_tensor(
                        [torch.argwhere(t == self.eot)[0] for t in tokens_t],
                        device=input.device,
                        dtype=torch.long,
                    )
                    ctc_loss = F.ctc_loss(
                        log_probs=log_probs_i[:, : input_lengths.max()].transpose(0, 1),
                        targets=tokens_t[:, : target_lengths.max()].to(torch.long),
                        input_lengths=input_lengths,
                        target_lengths=target_lengths,
                        blank=self.blank,
                        zero_infinity=True,
                    )
                    loss += ctc_loss * self.factor_lm
                else:
                    delta = log_probs_i.shape[1] - tokens_t.shape[1]
                    if delta > 0:
                        tokens_t = torch.cat(
                            (
                                tokens_t,
                                torch.full(
                                    (tokens_t.shape[0], delta),
                                    self.eot,
                                    device=tokens_t.device,
                                    dtype=tokens_t.dtype,
                                ),
                            ),
                            dim=1,
                        )
                    # if tokens_t.shape[1] != log_probs_i.shape[1]:
                    #     ic(tokens_t.shape, log_probs_i.shape)
                    #     for i in range(tokens_t.shape[0]):
                    #         ic(tokens_t[i])
                    #         ic(log_probs_i[i].argmax(dim=-1))
                    ce_loss = F.nll_loss(
                        log_probs_i.flatten(0, 1),
                        tokens_t[:, : tokens_i.shape[1]].flatten(0, 1),
                    )
                    loss += ce_loss * self.factor_lm
        return loss

    def decode_text(self, tokens: Tensor) -> List[str]:
        tokens = [t[: torch.argwhere(t == self.eot)[0]] for t in tokens]
        return [self.tokenizer.decode(t).strip() for t in tokens]

    def decode_tokens(
        self,
        features: Tensor,
        start_tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        n = features.shape[0]
        sum_logprobs: Tensor = torch.zeros(n, device=features.device)
        tokens: Tensor = start_tokens or torch.tensor(
            [self.initial_tokens], device=features.device
        ).repeat(n, 1)
        logits: List[Tensor] = []
        for i in range(self.sample_len):
            # we don't need no_speech_probs, only use last index (-1)
            logits.append(self.model.logits(tokens, features)[:, -1])
            tokens, completed = self.decoder.update(tokens, logits[-1], sum_logprobs)
            if completed or tokens.shape[-1] > self.n_ctx:
                break
        tokens, _ = self.decoder.finalize(tokens, sum_logprobs)
        return torch.stack(logits, dim=1), tokens[:, self.sample_begin : -1]

    def preprocess(self, audio: Tensor) -> Tensor:
        import whisper

        audio = resample(audio, self.sr, self.target_sr)
        audio = whisper.pad_or_trim(audio.squeeze(1))
        mel = self.log_mel_spectrogram(audio, self.mel_filters.to(audio.device))
        return mel

    def log_mel_spectrogram(self, audio: Tensor, mel_fb: Tensor):
        """From openai/whisper"""
        window = torch.hann_window(self.n_fft).to(audio.device)
        stft = torch.stft(audio, self.n_fft, self.hop, window=window, return_complex=True)
        assert stft.isfinite().all()
        magnitudes = stft[..., :-1].abs() ** 2
        assert magnitudes.isfinite().all()
        assert mel_fb.isfinite().all()

        mel_spec = mel_fb @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def get_mel_filters(self, sr, n_fft, n_mels=128, dtype=None):
        """From transformers/models/whisper/feature_extraction"""
        import numpy as np

        dtype = dtype or np.float32
        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = np.linspace(min_mel, max_mel, n_mels + 2)

        mels = np.asanyarray(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        mel_f = freqs

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

        return weights

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)
        prefix = self.options.prefix
        prompt = self.options.prompt

        if prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt
            )
            tokens = [self.tokenizer.sot_prev] + prompt_tokens[-(self.n_ctx // 2 - 1) :] + tokens

        return tuple(tokens)


class Loss(nn.Module):
    """Loss wrapper containing several different loss functions within this file.

    The configuration is done via the config file.
    """

    def __init__(self, state: DF, istft: Optional[Istft] = None):
        """Loss wrapper containing all methods for loss calculation.

        Args:
            state (DF): DF state needed for MaskLoss.
            istft (Callable/Module): Istft method needed for time domain losses.
        """
        super().__init__()
        p = ModelParams()
        self.lsnr = LocalSnrTarget(ws=20, target_snr_range=[p.lsnr_min - 1, p.lsnr_max + 1])
        self.istft = istft  # Could also be used for sdr loss
        self.sr = p.sr
        self.fft_size = p.fft_size
        self.nb_df = p.nb_df
        self.store_losses = False
        self.summaries: Dict[str, List[Tensor]] = self.reset_summaries()
        # Mask Loss
        self.ml_f = config("factor", 0, float, section="MaskLoss")  # e.g. 1
        self.ml_mask = config("mask", "iam", str, section="MaskLoss")  # e.g. 1
        self.ml_gamma = config("gamma", 0.6, float, section="MaskLoss")
        self.ml_gamma_pred = config("gamma_pred", 0.6, float, section="MaskLoss")
        self.ml_f_under = config("f_under", 2, float, section="MaskLoss")
        ml_max_freq = config("max_freq", 0, float, section="MaskLoss")
        if ml_max_freq == 0:
            self.ml_f_max_idx = None
        else:
            self.ml_f_max_idx = int(ml_max_freq / (p.sr / p.fft_size))
        if self.ml_mask == "spec":
            self.ml = MaskSpecLoss(state, self.ml_f, self.ml_gamma, f_max_idx=self.ml_f_max_idx)
        else:
            self.ml = MaskLoss(
                state,
                mask=self.ml_mask,
                factor=self.ml_f,
                f_under=self.ml_f_under,
                gamma=self.ml_gamma,
                gamma_pred=self.ml_gamma_pred,
                factors=[1, 10],
                powers=[2, 4],
                f_max_idx=self.ml_f_max_idx,
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

        self.asrl = None
        self.asrl_f = config("factor", 0, float, section="ASRLoss")
        self.asrl_f_lm = config("factor_lm", 0, float, section="ASRLoss")
        self.asrl_loss_lm = config("loss_lm", "CrossEntropy", str, section="ASRLoss")
        self.asrl_m = config("model", "base.en", str, section="ASRLoss")
        if self.asrl_f > 0 or self.asrl_f_lm > 0:
            self.asrl = ASRLoss(
                sr=self.sr,
                factor=self.asrl_f,
                factor_lm=self.asrl_f_lm,
                loss_lm=self.asrl_loss_lm,
                model=self.asrl_m,
            )

    def forward(
        self,
        clean: Tensor,
        noisy: Tensor,
        enhanced: Tensor,
        mask: Tensor,
        lsnr: Tensor,
        snrs: Tensor,
        max_freq: Optional[Tensor] = None,
    ):
        """Computes all losses.

        Args:
            clean (Tensor): Clean complex spectrum of shape [B, C, T, F].
            noisy (Tensor): Noisy complex spectrum of shape [B, C, T, F].
            enhanced (Tensor): Enhanced complex spectrum of shape [B, C, T, F].
            mask (Tensor): Mask (real-valued) estimate of shape [B, C, T, E], E: Number of ERB bins.
            lsnr (Tensor): Local SNR estimates of shape [B, T, 1].
            snrs (Tensor): Input SNRs of the noisy mixture of shape [B].
        """
        enhanced_td = None
        clean_td = None
        lsnr_gt = self.lsnr(clean, noise=noisy - clean)
        if self.istft is not None:
            if self.store_losses or self.mrsl is not None or self.sdrl is not None:
                enhanced_td = self.istft(enhanced)
                clean_td = self.istft(clean)

        ml, sl, mrsl, cal, sdrl, asrl, lsnrl = [torch.zeros((), device=clean.device)] * 7
        if self.ml_f != 0 and self.ml is not None:
            ml = self.ml(input=mask, clean=clean, noisy=noisy)
        if self.sl_f != 0 and self.sl is not None:
            sl = self.sl(input=enhanced, target=clean)
        if self.mrsl_f > 0 and self.mrsl is not None:
            mrsl = self.mrsl(enhanced_td, clean_td)
        if self.asrl_f > 0 or self.asrl_f_lm > 0:
            asrl = self.asrl(enhanced_td, clean_td)
        if self.lsnr_f != 0:
            lsnrl = self.lsnrl(input=lsnr, target_lsnr=lsnr_gt)
        if self.sdrl_f != 0:
            sdrl = self.sdrl(enhanced_td, clean_td)
        if self.store_losses and enhanced_td is not None:
            assert clean_td is not None
            self.store_summaries(
                enhanced_td,
                clean_td,
                snrs,
                ml,
                sl,
                mrsl,
                sdrl,
                asrl,
                lsnrl,
                cal,
            )
        return ml + sl + mrsl + sdrl + asrl + lsnrl + cal

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
        asrl: Tensor,
        lsnrl: Tensor,
        cal: Tensor,
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
        if asrl != 0:
            self.summaries["ASRLoss"].append(asrl.detach())
        if lsnrl != 0:
            self.summaries["LocalSnrLoss"].append(lsnrl.detach())
        sdr = SiSdr()
        enh_td = enh_td.squeeze(1).detach()
        clean_td = clean_td.squeeze(1).detach()
        sdr_vals: Tensor = sdr(enh_td, target=clean_td)
        stoi_vals: Tensor = stoi(y=enh_td, x=clean_td, fs_source=self.sr)
        sdr_vals_ms, stoi_vals_ms = [], []
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
