#!/bin/env python3

import os
import sys

import numpy as np

ATOL=1e-4
RTOL=1e-5

def main():
    usage = f"{sys.argv[0]} <a.npz> <b.npz> <input-name>"
    assert len(sys.argv) == 4, usage
    a = sys.argv[1]
    b = sys.argv[2]
    name = sys.argv[3]
    assert os.path.isfile(a), usage
    assert os.path.isfile(b), usage
    a = np.load(a)
    b = np.load(b)
    np.testing.assert_allclose(a[name], b[name], atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    main()
