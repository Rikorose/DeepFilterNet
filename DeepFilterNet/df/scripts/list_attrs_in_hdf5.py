import os
import sys

import h5py
import torch

assert len(sys.argv) in (2, 3)
assert os.path.exists(sys.argv[1]), sys.argv[1]

LIST_KEYS = len(sys.argv) == 3 and sys.argv[2] == "--keys"


def load_encoded(buffer, codec: str):
    import io

    import torchaudio as ta

    # In some rare cases, torch audio failes to fully decode vorbis resulting in a way shorter signal
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format=codec.lower())
    return wav


with h5py.File(sys.argv[1], "r", libver="latest", swmr=True) as h5f:
    for n, k in h5f.attrs.items():
        print(f"Found attr {n} '{k}' in {sys.argv[1]}")

    total_len = 0
    for group, samples in h5f.items():
        print(f"Found {len(samples)} samples in {group}")
        codec = h5f.attrs.get("codec", "pcm")
        sr = h5f.attrs["sr"]
        if LIST_KEYS:
            for n, sample in samples.items():
                print(n)
                if codec == "pcm":
                    audio = torch.from_numpy(sample[...])
                    if audio.dim() == 1:
                        audio.unsqueeze_(0)
                else:
                    audio = load_encoded(sample, codec)
                total_len += audio.shape[-1] / sr / 60 / 60
        print("Total len [h]:", total_len)
