#!/bin/env python3
import os
import sys

from icecream import ic
import numpy as np

ATOL = 1e-5
RTOL = 1e-7


def main():
    usage = f"{sys.argv[0]} <a.npz> <b.npz> <input-name> <concat-axis-pulse>"
    assert len(sys.argv) in (4, 5), usage
    a = sys.argv[1]
    b = sys.argv[2]
    name = sys.argv[3]
    assert os.path.isfile(a), usage
    assert os.path.isfile(b), usage
    npz_a = np.load(a)
    npz_b = np.load(b)
    # b is ref
    assert name in npz_b, f"{name} not found in {b} (avail: {npz_b.files})"
    ref = npz_b[name]
    ic(name, ref.shape)
    if name not in npz_a:
        names = [f for f in npz_a.files if name in f]
        actual = [npz_a[n] for n in names]
        actual = np.concatenate(actual, axis=int(sys.argv[4]))
        ic(actual.shape)
    else:
        actual = npz_a[name]
    np.testing.assert_allclose(actual, ref, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    main()
