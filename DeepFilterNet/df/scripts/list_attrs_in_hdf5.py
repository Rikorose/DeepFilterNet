import sys

import h5py

with h5py.File(sys.argv[1], "r", libver="latest", swmr=True) as h5f:
    for n, k in h5f.attrs.items():
        print(f"Found attr {n} '{k}' in {sys.argv[1]}")
    for group, samples in h5f.items():
        print(f"Found {len(samples)} samples in {group}")
