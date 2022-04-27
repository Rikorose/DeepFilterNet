#!/usr/bin/env python3

import argparse
import io
import os
import sys

import h5py
import numpy as np
import torch
import torchaudio as ta


def load_vorbis(buffer: np.ndarray) -> torch.Tensor:
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format="vorbis")
    return wav


def load_flac(buffer: np.ndarray) -> torch.Tensor:
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format="flac")
    return wav


def main(args):
    with h5py.File(args.hdf5, "r+") as h5f:
        sr = h5f.attrs.get("sr", args.sr)
        if sr is None:
            print("sr not found.", file=sys.stderr)
            # exit(1)
        # h5f.attrs["sr"] = sr
        max_freq = h5f.attrs.get("max_freq", args.max_freq)
        if max_freq is None:
            print("max_freq not found.", file=sys.stderr)
            # max_freq = sr // 2
        # h5f.attrs["max_freq"] = max_freq
        codec = h5f.attrs.get("codec", "pcm")
        for group in h5f.values():
            for key, ds in group.items():
                n_samples = ds.attrs.get("n_samples", None)
                print(key, n_samples)
                if codec == "vorbis":
                    sample = load_vorbis(ds)
                elif codec == "flac":
                    sample = load_flac(ds)
                else:  # pcm
                    sample = torch.from_numpy(ds[...])
                if n_samples is not None:
                    assert n_samples == sample.shape[-1]
                    assert sample.numel() >= n_samples
                print(sample.shape)
                assert sample.dim() == 1 or sample.shape[0] <= 16  # Assume max 16 channels
                # exit()
                ds.attrs["n_samples"] = int(sample.shape[-1])
                ds.attrs["n_ch"] = 1 if sample.dim() == 1 else int(sample.shape[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=str)
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--max-freq", type=int, default=None)
    args = parser.parse_args()
    assert os.path.isfile(args.hdf5)
    assert args.hdf5.endswith(".hdf5")
    main(args)
