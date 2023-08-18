import argparse
import os

import h5py
import torch

parser = argparse.ArgumentParser()
parser.add_argument("hdf5", nargs="+")
parser.add_argument("--keys", "-k", action="store_true")
parser.add_argument("--hours", action="store_true")
args = parser.parse_args()


def load_encoded(buffer, codec: str):
    import io

    import torchaudio as ta

    # In some rare cases, torch audio failes to fully decode vorbis resulting in a way shorter signal
    wav, _ = ta.load(io.BytesIO(buffer[...].tobytes()), format=codec.lower())
    return wav


total_hours = 0

for f in args.hdf5:
    assert os.path.isfile(f)
    with h5py.File(f, "r", libver="latest", swmr=True) as h5f:
        for n, k in h5f.attrs.items():
            print(f"Found attr {n} '{k}' in {f}")

        for group, samples in h5f.items():
            print(f"Found {len(samples)} samples in {group}")
            codec = h5f.attrs.get("codec", "pcm")
            sr = h5f.attrs["sr"]
            if args.keys or args.hours:
                for n, sample in samples.items():
                    if args.keys:
                        print(n)
                    if codec == "pcm":
                        audio = torch.from_numpy(sample[...])
                        if audio.dim() == 1:
                            audio.unsqueeze_(0)
                    else:
                        audio = load_encoded(sample, codec)
                    total_hours += audio.shape[-1] / sr / 60 / 60

if args.hours:
    print("Total len [h]:", total_hours)
