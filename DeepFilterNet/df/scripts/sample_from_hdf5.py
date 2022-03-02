import argparse
import io
import random
import sys

import h5py
import numpy as np
import torch
import torchaudio as ta


def load_vorbis(buffer: np.ndarray):
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format="vorbis")
    return wav


def load_flac(buffer: np.ndarray):
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format="flac")
    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_file")
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--random", "-r", action="store_true")
    parser.add_argument("--out-dir", "-o", type=str, default="out")
    parser.add_argument("--n-samples", "-n", type=int, default=1)
    parser.add_argument("--n-channels", "-c", type=int, default=1)
    args = parser.parse_args()

    with h5py.File(args.hdf5_file, "r", libver="latest", swmr=True) as h5f:
        sr = h5f.attrs.get("sr", args.sr)
        if sr is None:
            print("sr not found.", file=sys.stderr)
            exit(1)
        n_samples = args.n_samples
        i = 0
        codec = h5f.attrs.get("codec", "pcm")
        for group in h5f.values():
            keys = list(group.keys())
            if args.random:
                keys = random.sample(keys, n_samples)
            for key in keys:
                sample: np.ndarray = group[key][...]
                if codec == "vorbis":
                    sample = load_vorbis(sample)
                elif codec == "flac":
                    sample = load_flac(sample)
                outname = f"{args.out_dir}/{key}"
                if not outname.endswith(".wav"):
                    outname += ".wav"
                print(outname)
                if args.n_channels > 0:
                    sample = sample[: args.n_channels]
                ta.save(outname, torch.as_tensor(sample[:]), sample_rate=sr)
                i += 1
                if n_samples > 0 and i >= n_samples:
                    break


if __name__ == "__main__":
    main()
