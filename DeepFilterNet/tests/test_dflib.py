import timeit
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from icecream import ic
from torch import Tensor

import libdf
from df.config import config
from df.io import get_test_sample, save_audio
from df.model import ModelParams
from df.utils import _calculate_norm_alpha
from df.visualization import spec_figure
from libdf import DF
from libdfdata import PytorchDataLoader as DataLoader

ic.includeContext = True


def rms(x) -> float:
    return torch.as_tensor(x).abs().pow(2).mean().sqrt().item()


def test_analysis_synthesis_stft(plot: Union[bool, str] = False):
    import matplotlib.pyplot as plt
    from icecream import ic, install

    install()

    sr = 24000
    n_fft = 96
    hop = n_fft // 4
    state = DF(sr=sr, fft_size=n_fft, hop_size=hop, nb_bands=32)
    sigin = get_test_sample(sr)
    ic(rms(sigin))

    # A note on normalization:
    # We have 3 options to normalize an FFT:
    #   - forward: Normalize by 1/N during forward path
    #   - backward: Normalize by 1/N during backward path
    #   - orthogonal: Normalize by sqrt(1/N) during both, forward and backward path
    #
    # DeepFilterNet does normalization during forward (i.e. stft) transform.
    # Unfortunately torch only offers two options:
    #   - normalized=True: Does orthogonal normalization
    #   - normalized=False: Does backward normalization
    #
    # To normalize an STFT, we also need to consider hop/overlap which slightly changes this
    # behaviour.
    #
    # To get the same normalization in torch, we will use the normalized = False, and manually
    # multiply by the normalization factor.
    # In the inverse transform, torch will apply the full normalization, thus we remove our
    # normalization again by dividing.
    stft_norm = 1 / (n_fft**2 / (2 * hop))
    w = torch.from_numpy(state.fft_window())
    freq_torch: Tensor = (
        torch.stft(
            sigin, n_fft=n_fft, hop_length=hop, window=w, return_complex=True, normalized=False
        ).transpose(1, 2)
        * stft_norm
    )
    sigout_torch: Tensor = torch.istft(
        freq_torch.transpose(1, 2) / stft_norm,
        n_fft=n_fft,
        hop_length=hop,
        window=w,
        normalized=False,
    )
    freq_rs = state.analysis(sigin.numpy())
    ic(rms(freq_rs))

    sigout_rs_torch: Tensor = torch.istft(
        torch.from_numpy(freq_rs).transpose(1, 2) / stft_norm,
        n_fft=n_fft,
        hop_length=hop,
        window=w,
        normalized=False,
    )
    ic(rms(sigout_torch))
    ic(rms(sigout_rs_torch))
    sigout_rs = state.synthesis(np.copy(freq_rs))
    ic(rms(sigout_rs))
    save_audio("out/fb_torch.wav", sigout_torch, sr)
    save_audio("out/fb_rs_torch.wav", sigout_rs_torch, sr)
    save_audio("out/fb_rs.wav", sigout_rs, sr)

    if isinstance(plot, str) or plot:
        target = freq_torch if plot == "torch" else torch.from_numpy(freq_rs)
        spec_figure(
            target.squeeze().T,
            sr,
            n_fft=n_fft,
            hop=hop,
            xlabel="time [s]",
            ylabel="freq",
            colorbar=True,
            ticks=True,
        )
        plt.show()


def plot_erb_spec():
    sr, fft, hop, nb_bands = 48000, 960, 480, 32
    df = DF(sr, fft, hop, nb_bands)
    x = get_test_sample(sr)
    x = df.analysis(x.reshape(1, -1).numpy())
    x = libdf.erb(x, df.erb_widths())
    spec_figure(x, sr)
    plt.ylim(0, 10000)
    plt.savefig("out_erb.pdf")


def test_timings(num=1000):
    import librosa  # noqa: F401
    import torch

    sr = 48000
    x = get_test_sample(sr).numpy()
    print("librosa: ", end="", flush=True)
    t = timeit.timeit(
        "librosa.stft(x, n_fft=960, hop_length=480, center=False)",
        globals=globals() | locals(),
        number=num,
    )
    print(f"{t:.3f}")

    sr, fft, hop, nb_bands = 48000, 960, 480, 32
    df = DF(sr, fft, hop, nb_bands)
    x = x.reshape(1, -1)
    print("dfrs: ", end="", flush=True)
    t = timeit.timeit("df.analysis(x)", globals=globals() | locals(), number=num)
    print(f"{t:.3f}")

    x = torch.from_numpy(x)
    w = torch.hann_window(960)
    print("Pytorch (with multi-threading): ", end="", flush=True)
    t = timeit.timeit(
        "torch.stft(x, n_fft=960, hop_length=480, window=w, center=False, return_complex=True)",
        globals=globals() | locals(),
        number=num,
    )
    print(f"{t:.3f}")


def test_dataloader(ds_dir):
    import os

    loader = DataLoader(
        ds_dir=os.path.expanduser(ds_dir),
        ds_config="../assets/dataset.cfg",
        sr=48_000,
        batch_size=8,
        split="train",
    )
    i = 0
    for batch in loader.iter_epoch(epoch=0):
        ic(i, batch)
        i += 1
        if i > 10:
            break
    ic("done")


def test_fft_dataloader(ds_dir):
    import os

    import librosa

    config.load("/tmp/dfrs1/config.ini")
    p = ModelParams()
    df = DF(p.sr, p.fft_size, p.hop_size, p.nb_erb)
    norm_alpha = _calculate_norm_alpha(p.sr, p.hop_size, tau=1)
    loader = DataLoader(
        ds_dir=os.path.expanduser(ds_dir),
        ds_config="assets/dataset.cfg",
        sr=p.sr,
        batch_size=1,
        fft_size=p.fft_size,
        hop_size=p.hop_size,
        num_workers=1,
        prefetch=1,
        nb_erb=p.nb_erb,
        nb_spec=p.nb_df,
        norm_alpha=norm_alpha,
    )
    for batch in loader.iter_epoch("train", seed=0):
        ic(batch.speech.shape)
        ic(batch.feat_spec.shape)  # [B, C, T, F]
        ic(batch.feat_erb.shape)
        ic(batch.atten)

        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(batch.speech[0].squeeze(0).T)),
            sr=p.sr,
            x_axis="time",
            y_axis="hz",
        )
        plt.savefig("out/spec.pdf")
        ic(
            batch.feat_spec.mean(),
            batch.feat_spec.var(),
            batch.feat_spec.abs().min(),
            batch.feat_spec.abs().max(),
        )
        librosa.display.specshow(
            batch.feat_erb[0].squeeze(0).numpy().T,
            sr=p.sr,
            x_axis="time",
            y_axis="log",
        )
        plt.colorbar()
        plt.savefig("out/erb.pdf")
        noisy_wav = df.synthesis(batch.noisy[0].numpy())
        clean_wav = df.synthesis(batch.speech[0].numpy())
        ic(noisy_wav.shape)
        save_audio("out/noisy.wav", noisy_wav.T, p.sr)
        save_audio("out/clean.wav", clean_wav.T, p.sr)
        break
    ic("done")
