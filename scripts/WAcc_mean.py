#!/bin/env/python

import os
import sys

import numpy as np
import pandas as pd

USAGE = f"Usage: python {sys.argv[0]} path-to-score-file.csv"
assert len(sys.argv) > 1, USAGE
assert os.path.isfile(sys.argv[1]), USAGE

df = pd.read_csv(sys.argv[1])
print(
    "Mean WAcc for the file {} is {:.2f} %".format(
        os.path.basename(sys.argv[1]), np.mean(df["wacc"]) * 100
    )
)
