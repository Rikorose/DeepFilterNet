#!/usr/bin/env python3

import argparse
import os
import time
import warnings
from multiprocessing import Pool
from tempfile import NamedTemporaryFile
from typing import List

import h5py as h5
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from df.logger import init_logger


def write_to_h5(
    file_name: str,
    data: dict,
    sr: int,
    max_freq: int = -1,
    dtype: str = "float32",
    codec: str = "pcm",
    mono: bool = False,
    compression: str = None,
    num_workers: int = 4,
):
    """Creates a HDF5 dataset based on the provided dict.

    Args:
        file_name: Full path of the HDF5 dataset file.
        data: Dictionary containing for each key (dataset group) a list of file names
            to pre-process and store.
    """
    if max_freq <= 0:
        max_freq = sr // 2
    with h5.File(file_name, "w", libver="latest") as f, torch.no_grad():
        for key, data_dict in data.items():
            grp = f.create_group(key)
            dataset = PreProcessingDataset(
                sr,
                file_names=data_dict["files"],
                dtype=dtype,
                codec=codec,
                mono=mono,
            )
            loader = DataLoader(dataset, num_workers=num_workers, batch_size=1, shuffle=False)
            # Computes the samples in several worker processes
            n_samples = len(dataset)
            for i, sample in enumerate(loader):
                # Sample is a dict containing a list
                fn = os.path.relpath(sample["file_name"][0], data_dict["working_dir"])
                audio: np.ndarray = sample["data"][0].numpy()
                if audio.shape[1] < sr / 100:  # Should be at least 100 ms
                    logger.warning(f"Audio {fn} too short: {audio.shape}. Skipping.")
                progress = i / n_samples * 100
                logger.info(f"{progress:2.0f}% | Writing file {fn} to the {key} dataset.")
                if sample["n_samples"] == 0:
                    continue
                ds = grp.create_dataset(fn.replace("/", "_"), data=audio, compression=compression)
                ds.attrs["n_samples"] = sample["n_samples"]
                del audio, sample
            logger.info("Added {} samples to the group {}.".format(n_samples, key))

        f.attrs["db_id"] = int(time.time())
        f.attrs["db_name"] = os.path.basename(file_name)
        f.attrs["max_freq"] = max_freq
        f.attrs["dtype"] = dtype
        f.attrs["sr"] = sr
        f.attrs["codec"] = codec


class PreProcessingDataset(Dataset):
    def __init__(
        self, sr: int, file_names: List[str] = None, dtype="float32", codec="pcm", mono=False
    ):
        self.file_names = file_names or []
        self.sr = sr
        if dtype == "float32":
            self.dtype = np.float32
        elif dtype == "int16":
            self.dtype = np.int16
        else:
            raise ValueError("Unkown dtype")
        self.codec = codec.lower()
        if self.codec == "vorbis":
            self.dtype = np.float32
        self.mono = mono

    def read(self, file: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Librosa does not support resampling with np.int16, thus do the type
            # conversation manually
            x, sr = torchaudio.load(file, normalize=False)
            if self.mono and x.shape[0] > 1:
                x = x.mean(1, keepdim=True)
            x = x.numpy()
            dtype = x.dtype if isinstance(x.dtype, np.dtype) else np.dtype(x.dtype)
            if sr != self.sr:
                if not np.issubdtype(dtype, np.floating):
                    # To float32 so we can resample
                    x = x.astype(np.float32) / (1 << 15)

                if x.shape[0] == 1:
                    x = x.squeeze(0)
                x = librosa.resample(x, sr, target_sr=self.sr, scale=True, res_type="kaiser_best")
                if x.ndim == 1:
                    x = x.reshape(1, -1)
            # Double check dtype, since librosa does not take care of that when it falls
            # back to audioread or it might be changed due to resampling.
            dtype = x.dtype if isinstance(x.dtype, np.dtype) else np.dtype(x.dtype)
            if dtype != self.dtype:
                if np.issubdtype(dtype, np.floating):
                    # Convert float to int16 pcm
                    x = (x * (1 << 15)).astype(self.dtype)
                elif np.issubdtype(dtype, np.integral):
                    # Convert int16 pcm to float
                    x = x.astype(self.dtype) / (1 << 15)
        return x

    def __getitem__(self, index):
        fn = self.file_names[index]
        logger.debug(f"Reading audio file {fn}")
        x = self.read(fn)
        if self.codec == "vorbis":
            with NamedTemporaryFile(suffix=".ogg") as tf:
                if x.shape[-1] > 80 * self.sr:
                    # Getting segfaults for larger samples with libsnd.
                    warnings.warn("Max sample length is something around 80s. Truncating.")
                    x = x[..., : 80 * self.sr]
                if len(x.shape) > 1 and x.shape[-1] > 16:
                    # Assume channels first
                    x = x.transpose()
                try:
                    sf.write(tf.name, x, self.sr, format="ogg", subtype="vorbis")
                except RuntimeError:
                    warnings.warn(f"Runtime error writing file {fn}, shape {x.shape}")
                    return {
                        "file_name": fn,
                        "data": np.zeros_like(x),
                        "n_samples": 0,
                    }
                # Return binary buffer as numpy array
                x = np.array(list(tf.read()), dtype=np.uint8)
        if "noise" in fn.lower() and "_P1" in fn and "_Ch1" in fn:
            x_p1_ch1 = x
            x_p1_ch2 = self.read(fn.replace("_Ch1", "_Ch2"))
            x_p2_ch1 = self.read(fn.replace("_P1", "_P2"))
            x_p2_ch2 = self.read(fn.replace("_P1", "_P2").replace("_Ch1", "_Ch2"))
            x = np.stack((x_p1_ch1, x_p1_ch2, x_p2_ch1, x_p2_ch2))
            fn = fn.replace("_P1_Ch1.wav", "")
        n_samples = len(x)
        return {"file_name": fn, "data": x, "n_samples": n_samples}

    def __len__(self):
        return len(self.file_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, help="Either 'speech', 'noise' or 'noisy'.")
    parser.add_argument(
        "audio_files",
        type=str,
        help="Text file containing speech or noise files separated by a new line.",
    )
    parser.add_argument(
        "hdf5_db", type=str, help="HDF5 file name where the data will be stored in."
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_freq",
        type=int,
        default=-1,
        help="Only frequencies below the specified frequency will be considered during loss computation.\n"
        "This is useful for upsampled signals, that contain no information in higher frequencies.",
    )
    parser.add_argument("--sr", type=int, default=48_000, help="Sample rate to resample to.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="int16",
        help="Dtype that will be used to store the audio files. Can be float32 or int16.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="pcm",
        help="codec that will be stored in the database. Can be PCM or Vorbis. Defaults to PCM.",
    )
    parser.add_argument("--mono", action="store_true")
    parser.add_argument(
        "--compression", type=str, default=None, help="HDF5 dataset compression (e.g. `gzip`)."
    )
    args = parser.parse_args()

    if not args.hdf5_db.endswith(".hdf5"):
        args.hdf5_db += ".hdf5"

    init_logger()
    valid_types = ("speech", "noise", "noisy", "rir")
    if args.type not in valid_types:
        raise ValueError(f"Dataset type must be one of {valid_types}, but got {args.type}")

    data = {
        args.type: {"working_dir": None, "files": []},
    }
    with open(args.audio_files) as f:
        working_dir = os.path.dirname(args.audio_files)
        data[args.type]["working_dir"] = working_dir
        logger.info(f"Using speech working directory {working_dir}")

        def _check_file(file: str):
            file = os.path.join(working_dir, file.strip())
            if not os.path.isfile(file):
                raise FileNotFoundError(f"file {file} not found")
            return file

        with Pool(max(args.num_workers, 1)) as p:
            res = p.imap(_check_file, f, 100)
            data[args.type]["files"] = list(res)
        logger.info("Checking all audio files complete")
    write_to_h5(
        file_name=args.hdf5_db,
        data=data,
        sr=args.sr,
        max_freq=args.max_freq,
        dtype=args.dtype,
        codec=args.codec.lower(),
        mono=args.mono,
        compression=args.compression,
        num_workers=args.num_workers,
    )
