#!/bin/env python3

import os
import sys

import numpy as np


def main():
    usage = f"{sys.argv[0]} <input.npz> <output.npz> <input-name-1> <input-name-2> ..."
    assert len(sys.argv) > 3, usage
    input_name = sys.argv[1]
    output_name = sys.argv[2]
    assert os.path.isfile(input_name), usage
    input = np.load(input_name)
    out = {}
    for name in sys.argv[3:]:
        out[name] = input[name]
    if output_name != "/dev/null":
        np.savez_compressed(output_name, **out)


if __name__ == "__main__":
    main()
