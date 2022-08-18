#!/usr/bin/env python3

import argparse
import io
import os
import sys

import h5py
import numpy as np
import torch
import torchaudio as ta
from icecream import ic
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
    with h5py.File(args.hdf5_input, "r") as h5_read, h5py.File(args.hdf5_filtered, "a") as h5_write:
        # Copy attributes
        for (k, v) in h5_read.attrs.items():
            h5_write.attrs[k] = v

        sr: int = h5_read.attrs["sr"]
        if sr is None:
            print("sr not found.", file=sys.stderr)
            exit(1)
        codec = h5_read.attrs.get("codec", "pcm")
        codec_write = args.codec or codec
        for (grp_name, group_read) in h5_read.items():
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
                ic(ds.dtype, audio.min(), audio.max())
                print(f"{key}, ", end="", flush=True)
                assert audio.dim() <= 2
                # For now, only single channel is supported
                assert audio.dim() == 1 or audio.shape[0] == 1

                audio = to_f32(audio)
                resampled = resample(audio, sr, dnsmos.SR)
                # Check if we can trim up to first and last 3 seconds:
                trim_start, trim_end = 0, None
                for n in range(3, 0, -1):
                    if trim_start == 0:
                        (sig, _, ovr) = dnsmos.dnsmos_local(
                            resampled[: n * dnsmos.SR], onnx_sig, onnx_bak_ovr
                        )
                        if sig < 2.5 and ovr < 3.0:
                            trim_start = n * dnsmos.SR
                    if trim_end is None:
                        (sig, _, ovr) = dnsmos.dnsmos_local(
                            resampled[-n * dnsmos.SR :], onnx_sig, onnx_bak_ovr
                        )
                        if sig < 2.5 and ovr < 3.0:
                            trim_end = -1 * dnsmos.SR
                (sig, bak, ovr) = dnsmos.dnsmos_local(
                    resampled[..., trim_start:trim_end], onnx_sig, onnx_bak_ovr
                )
                print(f"got sig: {sig:.2f}, bak: {bak:.2f}, ovr: {ovr:.2f} ... ", end="")
                if sig > 4.2 and bak > 4.5 and ovr > 4.0:
                    print("copying.")
                    trim_start = trim_start // dnsmos.SR * sr
                    trim_end = trim_end // dnsmos.SR * sr if trim_end is not None else None
                    if trim_start == 0 and trim_end is None:
                        data = encoded
                    else:
                        if codec == "pcm" and ds.dtype == "int16":
                            audio = to_int16(audio)
                        if audio.dim() == 1:
                            audio = audio.unsqueeze(0)
                        data = encode(
                            audio[..., trim_start:trim_end], sr, codec_write, compression=8
                        )
                    if key in group_write:
                        del group_write[key]
                    ds_write = group_write.create_dataset(
                        key, data=data, compression=None if codec_write != "pcm" else ds.compression
                    )
                    for (k, v) in ds.attrs.items():
                        ds_write.attrs[k] = v
                    ds_write.attrs["n_samples"] = audio.shape[-1]
                else:
                    print("skipping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_input", type=str)
    parser.add_argument("--hdf5_filtered", type=str, default=None)
    parser.add_argument("--codec", type=str, default=None)
    args = parser.parse_args()
    assert os.path.isfile(args.hdf5_input), args.hdf5_input
    assert args.hdf5_input.endswith(".hdf5")
    if args.hdf5_filtered is None:
        args.hdf5_filtered = os.path.splitext(args.hdf5_input)[0] + "_filtered.hdf5"
        ic(args.hdf5_filtered)
    assert args.hdf5_filtered.endswith(".hdf5")
    main(args)
