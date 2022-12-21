#!/usr/bin/env python3
import io
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio as ta
from torch import Tensor

from df.scripts.prepare_data import encode


def windowed_energy(x: Tensor, ws: int, hop) -> Tensor:
    x = x.to(torch.float32) / x.max().item()
    x = F.pad(x, (ws // 2, ws // 2)).unfold(-1, ws, hop)
    x = x.pow(2).add(1e-10)
    x = x.log10()
    x = x.mean(-1).mul(20)
    if x.dim() > 1:
        x = x.mean(0)
    return x


def load_encoded(buffer: np.ndarray, codec: str):
    # In some rare cases, torch audio failes to fully decode vorbis resulting in a way shorter signal
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format=codec.lower())
    return wav


def trim(audio: Tensor, sr: int) -> (Tensor, bool):
    ws = sr // 10
    hop = sr // 20
    e = windowed_energy(audio, ws, hop)
    # find first above -100dB
    start = 0
    for i in range(e.shape[-1]):
        if e[i] > -120 and i > 14:
            start = i - 15
            break
    # find last above -100dB
    end = -1
    for i in range(e.shape[-1]):
        if e[-i] > -100 and i > 10:
            end = -i + 10
            break
    if start - end >= e.shape[-1]:
        return torch.empty(()), True
    if end < -10:
        print(f"start: {start}, end: {end}", end="")
        return audio[..., start * hop : end * hop], True
    return audio, False


def main(path: str):
    assert os.path.isfile(path)
    group = sys.argv[2] if len(sys.argv) > 2 else "speech"
    with h5py.File(path, "r", libver="latest") as fr, h5py.File(
        path.replace(".hdf5", "_trimmed.hdf5"), "w", libver="latest"
    ) as fw:
        assert group in fr
        grp = fw.create_group(group)
        sr = int(fr.attrs["sr"])
        for attr in fr.attrs:
            fw.attrs[attr] = fr.attrs[attr]
        codec = fr.attrs.get("codec", "pcm")
        out_codec = sys.argv[3] if len(sys.argv) > 3 else codec
        fw.attrs["codec"] = out_codec
        comp_kwargs = {"compression": "gzip"} if codec == "pcm" else {}
        for n, sample in fr[group].items():  # type: ignore
            print(n, end=" ")
            if codec == "pcm":
                audio = torch.from_numpy(sample[...])
                if audio.dim() == 1:
                    audio.unsqueeze_(0)
            else:
                audio = load_encoded(sample, codec)
            audio, got_trimmed = trim(audio, sr)
            if audio.numel() == 0:
                continue  # Everything got trimmed
            if got_trimmed:
                if codec == "pcm" and out_codec == codec:
                    data = audio
                else:
                    data = encode(audio, sr, out_codec).squeeze()
            else:
                if codec == out_codec:
                    data = sample
                else:
                    data = encode(audio, sr, out_codec).squeeze()

            ds = grp.create_dataset(n, data=data, **comp_kwargs)
            ds.attrs["n_samples"] = audio.shape[-1]
            print()


if __name__ == "__main__":
    main(sys.argv[1])
