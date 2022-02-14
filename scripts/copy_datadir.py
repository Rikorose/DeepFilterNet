import argparse
import json
import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from time import sleep
from typing import DefaultDict, Dict, List, Optional, Tuple

import h5py


@dataclass
class DsConfig:
    sampling_rate: int
    sampling_factor: int
    max_freq: int


def du(path):
    """disk usage in human readable format (e.g. '2,1')"""
    return float(
        subprocess.check_output(["du", "-sh", "--block-size=1G", path]).split()[0].decode("utf-8")
    )


def cp(src, tgt):
    """Copy a file via rsync"""
    return subprocess.call(["rsync", "-aL", "--info=name,progress2", src, tgt])


def copy_datasets(
    src_dir: str, target_dir: str, cfg_path: str, max_gb: float, lock: Optional[str] = None
):
    lock_f = os.path.join(target_dir, ".lock")
    have_read_locks = False
    if lock is not None:
        # We can have a write lock allowing exclusive access as well as multiple parallel read locks
        if os.path.isfile(lock_f):
            tries = 1
            while any(
                line.strip().endswith(".write") and not line.strip().startswith(lock)
                for line in open(lock_f)
            ):
                # Waite until the current write lock is released
                warnings.warn(f"<copy_datadir.py>: Could not lock target_dir {target_dir}")
                sleep(tries)
                tries *= 2
                if tries >= 2**12:
                    break
            have_read_locks = sum(1 for _ in open(lock_f)) > 0
        # Lock the target dir for writing
        open(lock_f, "a+").write(f"\n{lock}.write")
    cfg = json.load(open(cfg_path))
    os.makedirs(target_dir, exist_ok=True)
    # Start with train since it will be accessed most of the time
    for split in ("train", "valid", "test"):
        # Get all Datasets and sort by dataset type and sampling factor
        # {DsType: (file_name, sampling_factor)}, DsType is one of (speech, noise, rir)
        datasets: Dict[str, List[Tuple[str, float]]] = DefaultDict(list)
        for entry in cfg[split]:
            fn = entry[0]
            dstype = list(h5py.File(os.path.join(src_dir, fn)).keys())[0]
            datasets[dstype].append((fn, float(entry[1])))
        for dstype in datasets.keys():
            datasets[dstype] = sorted(datasets[dstype], key=lambda e: e[1], reverse=True)
        # We will mix multiple noises with a single speech signal, thus start with noise datasets,
        # since they will be accessed more often.
        for dstype in ("noise", "speech", "rir"):
            for fn, _ in datasets[dstype]:
                fn_src = os.path.join(src_dir, fn)
                fn_tgt = os.path.join(target_dir, fn)
                if du(target_dir) + du(fn_src) > max_gb:  # If too large, link instead
                    if not os.path.exists(fn_tgt):
                        print("linking", fn_src)
                        subprocess.call(["ln", "-s", fn_src, fn_tgt])
                else:
                    if os.path.islink(fn_tgt) and not have_read_locks:
                        os.remove(fn_tgt)  # Only remove if no other process has a readlock
                    elif have_read_locks and os.path.isfile(fn_tgt):
                        continue
                    print("copying", fn_src)
                    cp(fn_src, fn_tgt)
    if lock is not None:
        remove_lock(target_dir, lock + ".write", lock + ".read")


def remove_lock(target_dir: str, lock: str, new_lock: Optional[str] = None):
    lock_f = os.path.join(target_dir, ".lock")
    with open(lock_f, "r+") as f:
        lines = []
        for line in f.readlines():
            line = line.strip()
            if line == "" or line == lock or line == new_lock:
                continue
            lines.append(line + "\n")
        if new_lock is not None:
            lines.append(new_lock + "\n")
        f.seek(0)
        f.writelines(lines)
        f.truncate()


def cleanup(target_dir: str):
    lock_f = os.path.join(target_dir, ".lock")
    lines = [line.strip() for line in open(lock_f).readlines()]
    if len(lines) > 0:
        print("Could not cleanup due to existing locks:", lines)
        return
    shutil.rmtree(target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")
    cp_parser = subparsers.add_parser("copy-datasets", aliases=["cp"])
    cp_parser.add_argument("src_dir")
    cp_parser.add_argument("target_dir")
    cp_parser.add_argument("data_cfg")
    cp_parser.add_argument("--max-gb", type=float, default=100)
    cp_parser.add_argument("--lock", type=str, default=None)
    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("target_dir")
    cleanup_parser.add_argument("--lock", type=str, default=None)
    args = parser.parse_args()
    if args.subparser_name in ("cp", "copy-datasets"):
        copy_datasets(args.src_dir, args.target_dir, args.data_cfg, args.max_gb, args.lock)
    else:
        if args.lock is not None:
            remove_lock(args.target_dir, args.lock + ".read")
            remove_lock(args.target_dir, args.lock + ".write")
        cleanup(args.target_dir)
