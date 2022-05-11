#!/usr/bin/env python3

import os
import sys

import h5py
import torch
import torch.nn.functional as F
from torch import Tensor


def windowed_energy(x: Tensor, ws: int, hop) -> Tensor:
    x = x.to(torch.float32) / x.max().item()
    x = F.pad(x, (ws // 2, ws // 2)).unfold(-1, ws, hop)
    x = x.pow(2).add(1e-10)
    x = x.log10()
    x = x.mean(-1).mul(20)
    if x.dim() > 1:
        x = x.mean(0)
    return x


def main(path: str):
    assert os.path.isfile(path)
    with h5py.File(path, "r", libver="latest") as fr, h5py.File(
        path.replace(".hdf5", "_trimmed.hdf5"), "w", libver="latest"
    ) as fw:
        assert "speech" in fr
        grp = fw.create_group("speech")
        sr = int(fr.attrs["sr"])
        for attr in fr.attrs:
            fw.attrs[attr] = fr.attrs[attr]
        assert fr.attrs.get("codec", "pcm") == "pcm"
        for n, sample in fr["speech"].items():  # type: ignore
            audio = torch.from_numpy(sample[:])
            ws = sr // 10
            hop = sr // 20
            e = windowed_energy(audio, ws, hop)
            # find first above -100dB
            start = 0
            for i in range(e.shape[-1]):
                if e[i] > -100 and i > 10:
                    start = i - 15
                    break
            # find last above -100dB
            end = -1
            for i in range(e.shape[-1]):
                if e[-i] > -100 and i > 10:
                    end = -i + 10
                    break
            print(n, e.min().item(), e.max().item(), e.shape[0], start, end)
            assert start - end < e.shape[-1]
            e = e[start:end]
            audio = audio[..., start * hop : end * hop]
            ds = grp.create_dataset(n, data=audio, compression="gzip")
            ds.attrs["n_samples"] = audio.shape[-1]


if __name__ == "__main__":
    main(sys.argv[1])
