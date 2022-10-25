import os
import sys

import h5py

assert len(sys.argv) in (2, 3)
assert os.path.exists(sys.argv[1]), sys.argv[1]

LIST_KEYS = len(sys.argv) == 3 and sys.argv[2] == "--keys"

with h5py.File(sys.argv[1], "r", libver="latest", swmr=True) as h5f:
    for n, k in h5f.attrs.items():
        print(f"Found attr {n} '{k}' in {sys.argv[1]}")
    for group, samples in h5f.items():
        print(f"Found {len(samples)} samples in {group}")
        if LIST_KEYS:
            for sample in samples:
                print(sample)
