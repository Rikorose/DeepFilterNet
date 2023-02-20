#!/usr/bin/env python3

import argparse
import io
import os
import sys

import h5py
import numpy as np
import torch
import torchaudio as ta
from torch import Tensor

import df.scripts.dnsmos as dnsmos
from df.io import resample
from df.scripts.prepare_data import encode


def load_encoded(buffer: np.ndarray, codec: str) -> Tensor:
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format=codec)
    return wav


def to_f32(audio: Tensor) -> Tensor:
    if audio.dtype != torch.float32:
        audio = audio.to(torch.float32) / (1 << 15)
    return audio


def to_int16(audio: Tensor) -> Tensor:
    if audio.dtype != torch.int16:
        audio = (audio * (1 << 15)).to(torch.int16)
    return audio


def main(args):
    onnx_sig, onnx_bak_ovr = dnsmos.download_onnx_models()
    if args.mos is not None:
        assert len(args.mos) == 3
        t = args.mos
    else:
        t = [4.2, 4.5, 4.0]

    with h5py.File(args.hdf5_input, "r") as h5_read, h5py.File(args.hdf5_filtered, "a") as h5_write:
        print(f"Opened datatset {args.hdf5_input}")

        # Copy attributes
        for k, v in h5_read.attrs.items():
            h5_write.attrs[k] = v

        sr: int = h5_read.attrs["sr"]
        if sr is None:
            print("sr not found.", file=sys.stderr)
            exit(1)
        codec = h5_read.attrs.get("codec", "pcm")
        codec_write = args.codec or codec
        h5_write.attrs["codec"] = codec_write
        for grp_name, group_read in h5_read.items():
            if grp_name not in h5_write:
                group_write = h5_write.create_group(grp_name)
            else:
                group_write = h5_write[grp_name]
            for key, ds in group_read.items():
                encoded = ds[...]
                if codec == "pcm":
                    audio = torch.from_numpy(encoded)
                else:
                    audio = load_encoded(encoded, codec=codec)
                print(f"{key} ...", end="", flush=True)
                assert audio.dim() <= 2
                # For now, only single channel is supported
                if audio.dim() == 1:
                    audio.unsqueeze_(0)
                audio = to_f32(audio)
                for ch in range(audio.shape[0]):
                    resampled = resample(audio[ch], sr, dnsmos.SR)
                    (ch_sig, ch_bak, ch_ovr) = dnsmos.dnsmos_local(
                        resampled, onnx_sig, onnx_bak_ovr
                    )
                sig, bak, ovr = np.mean(ch_sig), np.mean(ch_bak), np.mean(ch_ovr)
                print(f" sig: {sig:.2f}, bak: {bak:.2f}, ovr: {ovr:.2f} ... ", end="")
                if sig > t[0] and bak > t[1] and ovr > t[2]:
                    print("copying.")
                    if codec == codec_write:
                        data = encoded
                    else:
                        if codec == "pcm" and ds.dtype == "int16":
                            audio = to_int16(audio)
                        if audio.dim() == 1:
                            audio = audio.unsqueeze(0)
                        data = encode(audio, sr, codec_write, compression=8)
                    if key in group_write:
                        del group_write[key]
                    ds_write = group_write.create_dataset(
                        key, data=data, compression=None if codec_write != "pcm" else ds.compression
                    )
                    for k, v in ds.attrs.items():
                        ds_write.attrs[k] = v
                    ds_write.attrs["n_samples"] = audio.shape[-1]
                else:
                    print("skipping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_input", type=str)
    parser.add_argument(
        "--mos",
        type=float,
        default=None,
        nargs=3,
        help="Minimum values of SIG, BAK and OVR MOS values. Defaults to [4.2, 4.5, 4.0]",
    )
    parser.add_argument("--hdf5_filtered", type=str, default=None)
    parser.add_argument("--codec", type=str, default=None)
    args = parser.parse_args()
    assert os.path.isfile(args.hdf5_input), args.hdf5_input
    assert args.hdf5_input.endswith(".hdf5")
    if args.hdf5_filtered is None:
        args.hdf5_filtered = os.path.splitext(args.hdf5_input)[0] + "_filtered.hdf5"
    assert args.hdf5_filtered.endswith(".hdf5")
    main(args)
