import matplotlib.pyplot as plt
import numpy as np
import torch

from df.utils import as_complex


def spec_figure(
    spec: torch.Tensor,
    sr: int,
    figsize=(15, 5),
    colorbar=False,
    colorbar_format=None,
    from_audio=False,
    figure=None,
    return_im=False,
    labels=False,
    xlabels=False,
    ylabels=False,
    **kwargs,
) -> plt.Figure:
    spec = torch.as_tensor(spec).detach()
    if labels or xlabels:
        kwargs.setdefault("xlabel", "Time [s]")
    if labels or ylabels:
        if kwargs.get("kHz", False):
            kwargs.setdefault("ylabel", "Frequency [kHz]")
        else:
            kwargs.setdefault("ylabel", "Frequency [Hz]")
    if from_audio:
        n_fft = kwargs.setdefault("n_fft", 1024)
        hop = kwargs.setdefault("hop", 256)
        w = torch.hann_window(n_fft, device=spec.device)
        spec = torch.stft(spec, n_fft, hop, window=w, return_complex=False)
        spec = spec.div_(w.pow(2).sum().sqrt())
    if torch.is_complex(spec) or spec.shape[-1] == 2:
        spec = as_complex(spec).abs().add_(1e-12).log10_().mul_(20)
    if (
        kwargs.get("vmax", None) is not None
        and kwargs["vmax"] <= 0
        and (spec_max := spec.max()) > 0
    ):
        kwargs["vmax"] = spec_max
    kwargs.setdefault("vmax", max(0.0, spec.max().item()))

    if figure is None:
        figure = plt.figure(figsize=figsize)
        figure.set_tight_layout(True)
    if spec.dim() > 2:
        spec = spec.squeeze(0)
    im = specshow(spec, sr, **kwargs)
    if colorbar:
        ckwargs = {}
        if "ax" in kwargs:
            if colorbar_format is None:
                if kwargs.get("vmin", None) is not None or kwargs.get("vmax", None) is not None:
                    colorbar_format = "%+2.0f dB"
            ckwargs = {"ax": kwargs["ax"]}
        plt.colorbar(im, format=colorbar_format, **ckwargs)
    if return_im:
        return im
    return figure


@torch.no_grad()
def specshow(
    spec,
    sr,
    ax=None,
    title=None,
    xlabel=None,
    ylabel=None,
    n_fft=None,
    hop=None,
    t=None,
    f=None,
    vmin=-100,
    vmax=0,
    raw_in=False,
    xlim=None,
    ylim=None,
    kHz=False,
    ticks=False,
    cmap="inferno",
):
    """Plots a spectrogram of shape [F, T]"""
    if raw_in or spec.shape[-1] == 2:
        spec = as_complex(torch.as_tensor(spec)).abs().add_(1e-12).log10_().mul_(20)
    if spec.dim() > 2:
        spec = spec.squeeze()
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()

    if ax is not None:
        set_title = ax.set_title
        set_xlabel = ax.set_xlabel
        set_ylabel = ax.set_ylabel
        set_xlim = ax.set_xlim
        set_ylim = ax.set_ylim
    else:
        ax = plt
        set_title = plt.title
        set_xlabel = plt.xlabel
        set_ylabel = plt.ylabel
        set_xlim = plt.xlim
        set_ylim = plt.ylim
    n_fft = n_fft or (spec.shape[0] - 1) * 2
    hop = hop or n_fft // 4
    if t is None:
        t = np.arange(0, spec.shape[-1]) * hop / sr
    if f is None:
        f = np.arange(0, spec.shape[0]) * sr // 2 / (n_fft // 2)
        if kHz:
            f /= 1000
    im = ax.pcolormesh(t, f, spec, rasterized=True, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    if ticks is False:
        ax.axis("off")
    if title is not None:
        set_title(title)
    if xlabel is not None:
        set_xlabel(xlabel)
    if ylabel is not None:
        set_ylabel(ylabel)
    if xlim is not None:
        set_xlim(xlim)
    if ylim is not None:
        set_ylim(ylim)
    return im
