#!/usr/bin/python3

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from sys import stderr
from time import sleep
from typing import DefaultDict, Dict, List, Optional, Tuple

import h5py

TIMESTAMP_FORMAT = "%Y%m%d%H%M"
timestamp = datetime.now().strftime("%Y%m%d%H%M")

HOSTS = [f"lme{n}" for n in (50, 51, 52, 53, 170, 171, 221, 222, 223)] + ["fs"]
OWN = os.uname()[1]


@dataclass
class DsConfig:
    sampling_rate: int
    sampling_factor: int
    max_freq: int


def ln(src, tgt):
    """Link a file via ln"""
    return subprocess.check_call(["ln", "-s", src, tgt])


def du(path):
    """disk usage in human readable format (e.g. '2,1')"""
    return float(
        subprocess.check_output(["du", "-shD", "--block-size=1G", path]).split()[0].decode("utf-8")
    )


def _cp(src, tgt, *rsync_args) -> bool:
    try:
        subprocess.check_call(["rsync", "-aL", *rsync_args, src, tgt], stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        # print(f"Failed to copy file {src}: {e.returncode}", file=stderr)
        if os.path.exists(tgt):
            os.remove(tgt)
        return False


def _cp_or_link(src, tgt, *rsync_args):
    try:
        subprocess.check_call(["rsync", "-aL", *rsync_args, src, tgt])
    except subprocess.CalledProcessError as e:
        print(f"Failed to copy file {src}: {e.returncode}", file=stderr)
        if os.path.exists(tgt):
            os.remove(tgt)
        print("Linking instead", file=stderr, flush=True)
        ln(src, tgt)


def cp(src, tgt, try_other_hosts: bool = False, verbose=0):
    """Copy a file via rsync"""
    info = []
    if verbose == 1:
        info = ["--info=name,stats"]
    elif verbose > 1:
        info = ["--info=name,stats2"]
    if try_other_hosts:
        hosts = [h for h in HOSTS if h != OWN]
        random.shuffle(hosts)
        # Iterate through existing hosts and try to copy from here
        for h in hosts:
            if h == "fs":
                break
            if _cp(h + ":" + tgt, tgt, *info):
                return
    _cp_or_link(src, tgt, *info)


def has_locks(directory: str, lock: Optional[str] = None, wait_write_lock: bool = False):
    lock_f = os.path.join(directory, ".lock")
    have_read_locks = False
    have_write_locks = False
    # We can have a write lock allowing exclusive access as well as multiple parallel read locks
    if os.path.isfile(lock_f):
        tries = 1
        while True:
            have_write_locks = False
            for line in open(lock_f):
                line = line.strip()
                if lock is not None and line.endswith(".write") and not line.startswith(lock):
                    print("Directory currently locked:", line, file=stderr)
                    have_write_locks = True
                    break
            if not have_write_locks or not wait_write_lock:
                break
            # Wait until the current write lock is released
            print(
                f"Warning: Could not lock directory {directory}",
                file=stderr,
            )
            if tries > 2**8:  # ~ 5 minutes
                break
            print(f"Retrying in {tries} s ...", file=stderr, flush=True)
            sleep(tries)
            tries *= 2
        have_read_locks = False
        cur_timestamp = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        for line in open(lock_f):
            line = line.strip()
            if lock is not None and line.startswith(lock):
                continue  # Same lock should not be relevant
            try:
                lock_timestamp = line.split(".")[1]
                lock_timestamp = datetime.strptime(lock_timestamp, TIMESTAMP_FORMAT)
            except Exception as e:
                print(e, file=stderr)
                continue
            if cur_timestamp - lock_timestamp < timedelta(days=1):
                print("Found existing lock", line.strip(), file=stderr)
                have_read_locks = True
                break
    return have_read_locks, have_write_locks


def copy_datasets(
    src_dir: str,
    target_dir: str,
    cfg_path: str,
    max_gb: float,
    lock: Optional[str] = None,
    try_other_hosts: bool = False,
):
    print(f"Copying datasets from {src_dir} to {target_dir}", flush=True)
    copied = set()
    os.makedirs(target_dir, exist_ok=True)
    lock_f = os.path.join(target_dir, ".lock")
    have_read_locks, _ = has_locks(target_dir, lock, wait_write_lock=True)
    # Lock the target dir for writing
    open(lock_f, "a+").write(f"\n{lock}.{timestamp}.write")
    cfg = json.load(open(cfg_path))
    os.makedirs(target_dir, exist_ok=True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    futures = {}
    cur_gb = du(target_dir)
    print(f"Current target dir size: {cur_gb}")
    print(f"Copying max {max_gb} GB")
    # Start with train since it will be accessed most of the time
    for split in ("train", "valid", "test"):
        # Get all Datasets and sort by dataset type and sampling factor
        # {DsType: (file_name, sampling_factor)}, DsType is one of (speech, noise, rir)
        datasets: Dict[str, List[Tuple[str, float]]] = DefaultDict(list)
        for entry in cfg[split]:
            fn = entry[0]
            path = os.path.join(src_dir, fn)
            try:
                dstype = list(h5py.File(path).keys())[0]
            except OSError as e:
                print(f"Error: Could not open hdf5 dataset {fn}")
                print("Caused by:", e)
                dstype = "noise"  # just assume noise, is only used for copying anyways
            datasets[dstype].append((fn, float(entry[1])))
        for dstype in datasets.keys():
            datasets[dstype] = sorted(datasets[dstype], key=lambda e: e[1], reverse=True)
        # We will mix multiple noises with a single speech signal, thus start with noise datasets,
        # since they will be accessed more often.
        for dstype in ("noise", "speech", "rir"):
            for fn, _ in datasets[dstype]:
                if fn in copied:
                    continue
                copied.add(fn)
                fn_src = os.path.join(src_dir, fn)
                fn_tgt = os.path.join(target_dir, fn)
                new_gb = du(fn_src)
                if cur_gb + new_gb > max_gb or "test" in fn_src.lower():
                    # If too large or a test dataset, link instead
                    if not os.path.exists(fn_tgt):
                        print("linking", fn_src, flush=True)
                        ln(fn_src, fn_tgt)
                    elif (
                        not have_read_locks
                        and os.path.isfile(fn_tgt)
                        and not os.path.islink(fn_tgt)
                    ):
                        # Just run rsync again to make sure the file is not corrupt
                        print("checking", fn_tgt, flush=True)
                        futures[executor.submit(cp, fn_src, fn_tgt)] = fn_tgt
                else:
                    if os.path.islink(fn_tgt) and not have_read_locks:
                        os.remove(fn_tgt)  # Only remove if no other process has a readlock
                    elif have_read_locks and os.path.isfile(fn_tgt):
                        continue
                    print("copying", fn_src, flush=True)
                    cur_gb += new_gb
                    futures[executor.submit(cp, fn_src, fn_tgt, try_other_hosts)] = fn_tgt
    for future in concurrent.futures.as_completed(futures):
        print("Completed", futures[future], flush=True)

    if lock is not None:
        remove_lock(target_dir, lock, lock + "." + timestamp + ".read")


def remove_lock(target_dir: str, lock: str, new_lock: Optional[str] = None):
    lock_f = os.path.join(target_dir, ".lock")
    with open(lock_f, "r+") as f:
        lines = []
        for line in f.readlines():
            line = line.strip()
            if line == "" or line.startswith(lock):
                continue
            if new_lock is not None and line.startswith(new_lock):
                continue
            lines.append(line + "\n")
        if new_lock is not None:
            lines.append(new_lock + "\n")
        f.seek(0)
        f.writelines(lines)
        f.truncate()


def cleanup(target_dir: str, prev_lock: Optional[str] = None):
    have_read_locks, have_write_locks = has_locks(target_dir, prev_lock, wait_write_lock=True)
    if have_read_locks or have_write_locks:
        print("Could not cleanup due to existing locks.")
        return
    print(f"Cleaning directory {target_dir}")
    shutil.rmtree(target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser_name")
    cp_parser = subparsers.add_parser("copy-datasets", aliases=["cp"])
    cp_parser.add_argument("src_dir")
    cp_parser.add_argument("target_dir")
    cp_parser.add_argument("data_cfg")
    cp_parser.add_argument("--max-gb", type=float, default=100)
    cp_parser.add_argument("--lock", "-l", type=str, default=None)
    cp_parser.add_argument("--other-hosts", action="store_true")
    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("target_dir")
    cleanup_parser.add_argument("--lock", type=str, default=None)
    args = parser.parse_args()
    if args.subparser_name is None:
        parser.print_help()
        exit(1)
    if args.subparser_name in ("cp", "copy-datasets"):
        copy_datasets(
            args.src_dir, args.target_dir, args.data_cfg, args.max_gb, args.lock, args.other_hosts
        )
    else:
        if args.lock is not None:
            remove_lock(args.target_dir, args.lock + ".read")
            remove_lock(args.target_dir, args.lock + ".write")
        cleanup(args.target_dir, args.lock)
