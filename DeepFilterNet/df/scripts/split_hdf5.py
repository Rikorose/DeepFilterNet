#!/usr/bin/env python3

import os
import shutil
import sys

import h5py
import numpy as np
from icecream import ic

splits = (0.7, 0.15, 0.15)
assert np.sum(splits) == 1


def main(hdf5: str, force: bool = False):
    assert hdf5.endswith("TRAIN.hdf5")
    f_train = h5py.File(hdf5, "r+")
    if os.path.basename(hdf5).lower().startswith("vocalset"):
        # Correct wrong sampling rate
        f_train.attrs["sr"] = 16000
        f_train.attrs["max_freq"] = 8000
    hdf5_train_new = hdf5.replace("TRAIN", "TRAIN_NEW")
    hdf5_valid = hdf5.replace("TRAIN", "VALID")
    hdf5_test = hdf5.replace("TRAIN", "TEST")
    if (os.path.exists(hdf5_test) or os.path.exists(hdf5_valid)) and not force:
        raise FileExistsError(f"Dataset {hdf5_test} already exists.")
    f_valid = h5py.File(hdf5_valid, "w")
    f_test = h5py.File(hdf5_test, "w")
    f_train_new = h5py.File(hdf5_train_new, "w")
    for attr, v in f_train.attrs.items():
        ic(attr, v)
        f_train_new.attrs[attr] = v
        f_valid.attrs[attr] = v
        f_test.attrs[attr] = v
    for key in f_train:
        grp_train = f_train[key]
        grp_train_new = f_train_new.create_group(key)
        grp_valid = f_valid.create_group(key)
        grp_test = f_test.create_group(key)
        keys = list(grp_train.keys())
        n = len(keys)
        # sections is a list of sorted integers; I.e. the cumulated sum
        len_train, len_valid = (int(s * n) for s in splits[:2])
        sections = [len_train, len_train + len_valid]
        # We don't need a section for test, it will be the rest anyways
        len_test = n - sections[1]
        keys_train, keys_valid, keys_test = np.split(np.random.permutation(keys), sections)
        ic(len(keys_train), len(keys_valid), len(keys_test))
        for k in keys_train:
            print("train", k)
            grp_train_new.copy(grp_train[k], dest=k)
        assert len(grp_train_new) == len_train
        for k in keys_valid:
            print("valid", k)
            grp_valid.copy(grp_train[k], dest=k)
        assert len(grp_valid) == len_valid
        for k in keys_test:
            print("test", k)
            grp_test.copy(grp_train[k], dest=k)
        assert len(grp_test) == len_test
    ic(shutil.move(hdf5_train_new, hdf5))


if __name__ == "__main__":
    force = sys.argv[2] == "-f" if len(sys.argv) > 2 else False
    main(sys.argv[1], force=force)
