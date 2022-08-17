from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.nn import functional as F

from df.io import resample

EPS = np.finfo("float").eps


def as_windowed(x: Tensor, window_length: int, step: int = 1) -> Tensor:
    """Returns a tensor with chunks of overlapping windows of the first dim of x.

    Args:
        x (Tensor): Input of shape [N, B, H, W]
        window_length (int): Length of each window
        step (int): Step/hop of each window w.r.t. the original signal x

    Returns:
        windowed tensor (Tensor): Output tensor with shape
            [(N - window_length + step) // step, window_length, B, H, W]
    """
    shape = ((x.shape[0] - window_length + step) // step, window_length) + x.shape[1:]
    stride: List[int] = []
    for i in range(x.dim()):
        stride.append(x.stride(i))
    stride.insert(1, stride[0])
    stride[0] = stride[0] * step
    return x.as_strided(shape, stride)


def remove_silent_frames(
    x: Tensor, y: Tensor, dyn_range: int, framelen: int, hop: int, eps: float = EPS
) -> Tuple[List[Tensor], List[Tensor]]:
    """Remove the silent frames from each signal in the batch.

    Note:
        This implementation is based on https://github.com/mpariente/pystoi
        The overlap add code is based on https://github.com/pytorch/pytorchaudio

    Args:
        x (Tensor): Reference signal of shape [batch, samples].
        y (Tensor): Second signal of the same shape where the same frames are removed.
        dyn_range (int): Dynamic range / energy in [dB] to determin which frames to
            remove.
        framelen (int): Length for a frame that might be removed.
        hop (int): Hop between to subsequent frames.

    Returns:
        x (List[Tensor]): x without silent frames. Since each signal might have a
            different number of removed frames, the resulting signals will have
            different lengths. Thus a batch is return as list of tensors.
        y (List[Tensor]): y without silent frames.
    """
    pad = framelen - x.shape[1] % framelen
    pad_front, mod = divmod(pad, 2)
    pad_end = pad_front + mod
    x = F.pad(x, (pad_front, pad_end))
    y = F.pad(y, (pad_front, pad_end))
    B, _ = x.shape
    # [B, N] -> [N, B]
    x, y = x.t(), y.t()
    # Compute mask
    # Note: `framelen + 2` and `[1:-1]` is needed because otherwise the first sample is
    # 0 which results in NaN during overlapp add
    w = torch.hann_window(framelen + 2, periodic=False, device=x.device, dtype=x.dtype)
    w = w[1:-1].view(-1, 1)
    # Windowed signal length N_w = (N - L - H) // H; L=framelen, H=hop
    x_w = as_windowed(x, framelen, hop) * w  # [N_w, L, B]
    y_w = as_windowed(y, framelen, hop) * w  # [N_w, L, B]
    # Compute energies in dB
    # TODO: is `np.sqrt(win_size)` correct? Why not `w.sum().sqrt()` or torch.norm(w)?
    x_energies = 20 * torch.log10(x_w.norm(dim=1) / np.sqrt(framelen) + eps)  # [N_w, B]
    # Find boolean mask of energies lower than dynamic_range dB
    # with respect to maximum clean speech energy frame
    mask = (torch.max(x_energies, dim=0)[0] - dyn_range - x_energies).unsqueeze(1) < 0
    # Remove silent frames by masking for each sample in the batch
    x_w = [x_w[..., i].masked_select(mask[..., i]).view(-1, framelen) for i in range(B)]
    y_w = [y_w[..., i].masked_select(mask[..., i]).view(-1, framelen) for i in range(B)]
    n_no_sil_w = [x.shape[0] for x in x_w]
    # init zero arrays to hold x, y with silent frames removed
    n_no_sil = [(x.shape[0] - 1) * hop + framelen for x in x_w]
    x_no_sil = [torch.zeros((n_no_sil[i]), device=x.device) for i in range(B)]
    y_no_sil = [torch.zeros((n_no_sil[i]), device=x.device) for i in range(B)]
    # Overlapp add via transposed convolution
    x_w = [x.t().unsqueeze(0) for x in x_w]  # [N_w, L] -> [1, L, N_w]
    y_w = [y.t().unsqueeze(0) for y in y_w]  # [N_w, L] -> [1, L, N_w]
    eye = torch.eye(framelen, device=x.device, dtype=x.dtype).unsqueeze(1)
    # [1, L, N_w] -> [N]
    x_no_sil = [F.conv_transpose1d(x, eye, stride=hop).squeeze() for x in x_w]
    y_no_sil = [F.conv_transpose1d(y, eye, stride=hop).squeeze() for y in y_w]
    # Same for the window  [L, 1] -> [1, L, N_w]
    w = [w.repeat((1, n_no_sil_w[i])).unsqueeze(0) for i in range(B)]
    # [1, L, N_w] -> [N]
    w = [F.conv_transpose1d(w_, eye, stride=hop).squeeze() for w_ in w]
    x_no_sil = [x / w for x, w in zip(x_no_sil, w)]
    y_no_sil = [y / w for y, w in zip(y_no_sil, w)]
    # If the first frame is not masked out, we need to remove pad_front
    # Also maybe remove zero padding at the end
    for i in range(B):
        if mask[0, :, i]:
            x_no_sil[i] = x_no_sil[i][pad_front:]
            y_no_sil[i] = y_no_sil[i][pad_front:]
        if mask[-1, :, i]:
            x_no_sil[i] = x_no_sil[i][:-pad_end]
            y_no_sil[i] = y_no_sil[i][:-pad_end]
    return x_no_sil, y_no_sil


def thirdoct(fs, nfft, num_bands, min_freq):
    """Returns the 1/3 octave band matrix and its center frequencies
    # Arguments :
        fs : sampling rate
        nfft : FFT size
        num_bands : number of 1/3 octave bands
        min_freq : center frequency of the lowest 1/3 octave band
    # Returns :
        obm : Octave Band Matrix
        cf : center frequencies
    # Credit: https://github.com/mpariente/pystoi
    """
    f = np.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]
    k = np.array(range(num_bands)).astype(float)
    cf = np.power(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * np.power(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2.0, (2 * k + 1) / 6)
    obm = np.zeros((num_bands, len(f)))  # a verifier

    for i in range(len(cf)):
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        obm[i, fl_ii:fh_ii] = 1
    return obm, cf


def _stft(x, win_size, fft_size, hop_size, normalized=True, window=None):
    if window is None:
        ws = win_size + 2
        window = torch.hann_window(ws, periodic=False, device=x.device, dtype=x.dtype)[1:-1]
    # Pad the signal for (fft_size-win_size) / 2 at each side, because if win_size is <
    # fft_size, the first and last frames would not be considered for the non centered
    # (padded) version; this is inconsitent with scipy.signals stft
    missing_len = fft_size - win_size
    x = F.pad(x, (missing_len // 2, missing_len // 2))
    # To spectral domain
    spec = torch.stft(x, fft_size, hop_size, win_size, window, center=False, return_complex=False)
    # Normalize by default
    if normalized:
        spec /= window.sum().pow(2).sqrt()
    return spec


def stoi(x, y, fs_source):
    """Pytorch STOI implementation. Should only used for validation/developement, use pystoi for reporting test results.

    Arguments:
        x (Tensor): Target signal
        y (Tensor): Degraded signal
        fs_source (int): Sampling rate of input signals
    """
    assert x.shape == y.shape, "Inputs must have the same shape"
    assert x.dim() == 2, f"Expected input shape of [batch_size, samples], but got {x.shape}"
    fs = 10_000
    dyn_range = 40
    N_frame = 256
    N_fft = 512
    N_bands = 15
    min_freq = 150
    N = 30
    Beta = -15.0
    B = x.shape[0]  # batch size

    # Preallocate some stuff
    out = torch.empty(B, device=x.device, dtype=x.dtype)
    obm, _ = thirdoct(fs, N_fft, N_bands, min_freq)  # [N_fft//2-1, N_bands]
    obm = torch.from_numpy(obm).to(x)

    x = resample(x, fs_source, fs)
    y = resample(y, fs_source, fs)

    x_, y_ = remove_silent_frames(x, y, dyn_range, N_frame, N_frame // 2)

    for i in range(B):
        # To spectral domain
        if x_[i].numel() < N_fft:
            logger.warning("Could not calculate STOI (not enough frames left). Skipping.")
            continue
        x = _stft(x_[i], win_size=N_frame, fft_size=N_fft, hop_size=N_frame // 2)
        y = _stft(y_[i], win_size=N_frame, fft_size=N_fft, hop_size=N_frame // 2)
        # Power spectrogram
        x = x.pow(2).sum(-1)
        y = y.pow(2).sum(-1)
        # Reduce frequency res to 1/3 octave band
        x = torch.matmul(obm, x).sqrt()  # [N_bands, L]
        y = torch.matmul(obm, y).sqrt()
        if x.shape[-1] > N:
            x = x.unfold(-1, N, 1).permute(1, 2, 0)  # [L', N, N_bands]
            y = y.unfold(-1, N, 1).permute(1, 2, 0)
        else:
            # For short signals, we don't need the unfolding
            x = x.transpose(0, 1).unsqueeze(0)
            y = y.transpose(0, 1).unsqueeze(0)
        # Normalize per N-window
        norm = torch.norm(x, dim=1, keepdim=True) / (torch.norm(y, dim=1, keepdim=True) + EPS)
        y = y * norm
        # Clip
        c = 10 ** (-Beta / 20)
        y = torch.min(y, x * (1 + c))
        # Subtract mean vectors
        y = y - y.mean(dim=1, keepdim=True)
        x = x - x.mean(dim=1, keepdim=True)
        # Divide by norm
        x = x / (torch.norm(x, dim=1, keepdim=True) + EPS)
        y = y / (torch.norm(y, dim=1, keepdim=True) + EPS)
        corr = x * y
        # J, M as in eq. [6]
        J = x.shape[0]
        M = N_bands
        # Mean of all correlations
        out[i] = torch.sum(corr) / (M * J)
    return out
