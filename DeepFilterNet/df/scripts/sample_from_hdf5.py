import argparse
import io
import os
import random
import sys
from typing import Optional

import h5py
import numpy as np
import torch
import torchaudio as ta
from icecream import ic


def load_encoded(buffer: np.ndarray, codec: str):
    # In some rare cases, torch audio failes to fully decode vorbis resulting in a way shorter signal
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format=codec.lower())
    return wav


def save_sample(group, key: str, codec: str, out_dir: str, sr: int, n_channels: Optional[int]):
    ds = group[key]
    if codec == "pcm":
        sample = torch.from_numpy(ds[...])
    else:
        ic(ds.shape)
        sample = load_encoded(ds, codec)
        ic(sample.shape)
    outname = f"{out_dir}/{key}"
    if not outname.endswith(".wav"):
        outname += ".wav"
    print(outname)
    if n_channels is not None and n_channels > 0:
        assert sample.dim() == 2
        sample = sample[:n_channels]
    ta.save(outname, sample, sample_rate=sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_file")
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--random", "-r", action="store_true")
    parser.add_argument("--out-dir", "-o", type=str, default="out")
    parser.add_argument("--n-samples", "-n", type=int, default=1)
    parser.add_argument("--n-channels", "-c", type=int, default=1)
    parser.add_argument("--keys", "-k", type=str, nargs="*")
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    with h5py.File(args.hdf5_file, "r", libver="latest", swmr=True) as h5f:
        sr = h5f.attrs.get("sr", args.sr)
        if sr is None:
            print("sr not found.", file=sys.stderr)
            exit(1)
        n_samples = args.n_samples
        i = 0
        codec = h5f.attrs.get("codec", "pcm")
        for group in h5f.values():
            if args.keys is not None:
                keys = args.keys
            else:
                keys = list(group.keys())
                if args.random:
                    keys = random.sample(keys, n_samples)
            for key in keys:
                save_sample(group, key, codec, args.out_dir, sr, args.n_channels)
                i += 1
                if n_samples > 0 and i >= n_samples:
                    break


if __name__ == "__main__":
    main()
