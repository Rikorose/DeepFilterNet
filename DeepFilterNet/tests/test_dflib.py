import timeit

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from icecream import ic

from df import DF, erb
from df.config import config
from df.dataloader import DataLoader
from df.model import ModelParams
from df.utils import _calculate_norm_alpha

ic.includeContext = True


def test_analysis_synthesis():
    sr, fft, hop, nb_bands = 48000, 960, 480, 32
    df = DF(sr, fft, hop, nb_bands)
    x, sr = librosa.load(librosa.ex("trumpet"), sr=sr)
    sf.write("in.wav", x.T, sr)
    x = df.analysis(x.reshape(1, -1))
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(x[0].T)), sr=sr, x_axis="time", y_axis="hz"
    )
    plt.ylim(0, 10000)
    plt.savefig("out.pdf")
    x = df.synthesis(x)
    sf.write("out.wav", x.T, sr)


def plot_erb_spec():
    sr, fft, hop, nb_bands = 48000, 960, 480, 32
    df = DF(sr, fft, hop, nb_bands)
    x, sr = librosa.load(librosa.ex("trumpet"), sr=48000)
    x = df.analysis(x.reshape(1, -1))
    x = erb(x, nb_bands)
    librosa.display.specshow(librosa.amplitude_to_db(x[0].T), sr=sr, x_axis="time", y_axis="mel")
    plt.ylim(0, 10000)
    plt.savefig("out_erb.pdf")


def test_timings(num=1000):
    import torch

    x, sr = librosa.load(librosa.ex("trumpet"), sr=48000)
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

    import soundfile as sf

    config.load("/tmp/dfrs1/config.ini")
    p = ModelParams()
    df = DF(p.sr, p.fft_size, p.hop_size, p.nb_erb)
    norm_alpha = _calculate_norm_alpha(p.sr, p.hop_size, tau=1)
    loader = DataLoader(
        ds_dir=os.path.expanduser(ds_dir),
        ds_config="assets/dataset.cfg",
        sr=p.sr,
        batch_size=1,
        fft_dataloader=True,
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
        sf.write("out/noisy.wav", noisy_wav.T, p.sr)
        sf.write("out/clean.wav", clean_wav.T, p.sr)
        break
    ic("done")
