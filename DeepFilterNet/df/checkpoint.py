import glob
import os
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch import nn

from df.config import Csv, config
from df.model import init_model
from df.utils import check_finite_module
from libdf import DF


def get_epoch(cp) -> int:
    return int(os.path.basename(cp).split(".")[0].split("_")[-1])


def load_model(
    cp_dir: Optional[str],
    df_state: DF,
    jit: bool = False,
    mask_only: bool = False,
    train_df_only: bool = False,
    extension: str = "ckpt",
    epoch: Union[str, int, None] = "latest",
) -> Tuple[nn.Module, int]:
    if mask_only and train_df_only:
        raise ValueError("Only one of `mask_only` `train_df_only` can be enabled")
    model = init_model(df_state, run_df=mask_only is False, train_mask=train_df_only is False)
    if jit:
        model = torch.jit.script(model)
    blacklist: List[str] = config("CP_BLACKLIST", [], Csv(), save=False, section="train")  # type: ignore
    if cp_dir is not None:
        epoch = read_cp(
            model, "model", cp_dir, blacklist=blacklist, extension=extension, epoch=epoch
        )
        epoch = 0 if epoch is None else epoch
    else:
        epoch = 0
    return model, epoch


def read_cp(
    obj: Union[torch.optim.Optimizer, nn.Module],
    name: str,
    dirname: str,
    epoch: Union[str, int, None] = "latest",
    extension="ckpt",
    blacklist=[],
    log: bool = True,
):
    checkpoints = []
    if isinstance(epoch, str):
        assert epoch in ("best", "latest")
    if epoch == "best":
        checkpoints = glob.glob(os.path.join(dirname, f"{name}*.{extension}.best"))
        if len(checkpoints) == 0:
            logger.warning("Could not find `best` checkpoint. Checking for default...")
    if len(checkpoints) == 0:
        checkpoints = glob.glob(os.path.join(dirname, f"{name}*.{extension}"))
        checkpoints += glob.glob(os.path.join(dirname, f"{name}*.{extension}.best"))
    if len(checkpoints) == 0:
        return None
    if isinstance(epoch, int):
        latest = next((x for x in checkpoints if get_epoch(x) == epoch), None)
        if latest is None:
            logger.error(f"Could not find checkpoint of epoch {epoch}")
            exit(1)
    else:
        latest = max(checkpoints, key=get_epoch)
        epoch = get_epoch(latest)
    if log:
        logger.info("Found checkpoint {} with epoch {}".format(latest, epoch))
    latest = torch.load(latest, map_location="cpu")
    latest = {k.replace("clc", "df"): v for k, v in latest.items()}
    if blacklist:
        reg = re.compile("".join(f"({b})|" for b in blacklist)[:-1])
        len_before = len(latest)
        latest = {k: v for k, v in latest.items() if reg.search(k) is None}
        if len(latest) < len_before:
            logger.info("Filtered checkpoint modules: {}".format(blacklist))
    if isinstance(obj, nn.Module):
        while True:
            try:
                missing, unexpected = obj.load_state_dict(latest, strict=False)
            except RuntimeError as e:
                e_str = str(e)
                logger.warning(e_str)
                if "size mismatch" in e_str:
                    latest = {k: v for k, v in latest.items() if k not in e_str}
                    continue
                raise e
            break
        for key in missing:
            logger.warning(f"Missing key: '{key}'")
        for key in unexpected:
            if key.endswith(".h0") or "erb_comp" in key:
                continue
            logger.warning(f"Unexpected key: {key}")
        return epoch
    obj.load_state_dict(latest)


def write_cp(
    obj: Union[torch.optim.Optimizer, nn.Module],
    name: str,
    dirname: str,
    epoch: int,
    extension="ckpt",
    metric: Optional[float] = None,
    cmp="min",
):
    check_finite_module(obj)
    n_keep = config("n_checkpoint_history", default=3, cast=int, section="train")
    n_keep_best = config("n_best_checkpoint_history", default=5, cast=int, section="train")
    if metric is not None:
        assert cmp in ("min", "max")
        metric = float(metric)  # Make sure it is not an integer
        # Each line contains a previous best with entries: (epoch, metric)
        with open(os.path.join(dirname, ".best"), "a+") as prev_best_f:
            prev_best_f.seek(0)  # "a+" creates a file in read/write mode without truncating
            lines = prev_best_f.readlines()
            if len(lines) == 0:
                prev_best = float("inf" if cmp == "min" else "-inf")
            else:
                prev_best = float(lines[-1].strip().split(" ")[1])
            cmp = "__lt__" if cmp == "min" else "__gt__"
            if getattr(metric, cmp)(prev_best):
                logger.info(f"Saving new best checkpoint at epoch {epoch} with metric: {metric}")
                prev_best_f.seek(0, os.SEEK_END)
                np.savetxt(prev_best_f, np.array([[float(epoch), metric]]))
                cp_name = os.path.join(dirname, f"{name}_{epoch}.{extension}.best")
                torch.save(obj.state_dict(), cp_name)
                cleanup(name, dirname, extension + ".best", nkeep=n_keep_best)
    cp_name = os.path.join(dirname, f"{name}_{epoch}.{extension}")
    logger.info(f"Writing checkpoint {cp_name} with epoch {epoch}")
    torch.save(obj.state_dict(), cp_name)
    cleanup(name, dirname, extension, nkeep=n_keep)


def cleanup(name: str, dirname: str, extension: str, nkeep=5):
    if nkeep < 0:
        return
    checkpoints = glob.glob(os.path.join(dirname, f"{name}*.{extension}"))
    if len(checkpoints) == 0:
        return
    checkpoints = sorted(checkpoints, key=get_epoch, reverse=True)
    for cp in checkpoints[nkeep:]:
        logger.debug("Removing old checkpoint: {}".format(cp))
        os.remove(cp)


def check_patience(
    dirname: str, max_patience: int, new_metric: float, cmp: str = "min", raise_: bool = True
):
    cmp = "__lt__" if cmp == "min" else "__gt__"
    new_metric = float(new_metric)  # Make sure it is not an integer
    prev_patience, prev_metric = read_patience(dirname)
    if prev_patience is None or getattr(new_metric, cmp)(prev_metric):
        # We have a better new_metric, reset patience
        write_patience(dirname, 0, new_metric)
    else:
        # We don't have a better metric, decrement patience
        new_patience = prev_patience + 1
        write_patience(dirname, new_patience, prev_metric)
        if new_patience >= max_patience:
            msg = f"No improvements on validation metric ({prev_metric:.3f} - {new_metric:.3f}) for {max_patience} epochs. Stopping."
            if raise_:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False
    return True


def read_patience(dirname: str) -> Tuple[Optional[int], float]:
    fn = os.path.join(dirname, ".patience")
    if not os.path.isfile(fn):
        return None, 0.0
    patience, metric = np.loadtxt(fn)
    return int(patience), float(metric)


def write_patience(dirname: str, new_patience: int, metric: float):
    return np.savetxt(os.path.join(dirname, ".patience"), [new_patience, metric])


def test_check_patience():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        check_patience(d, 3, 1.0)
        check_patience(d, 3, 1.0)
        check_patience(d, 3, 1.0)
        assert check_patience(d, 3, 1.0, raise_=False) is False

    with tempfile.TemporaryDirectory() as d:
        check_patience(d, 3, 1.0)
        check_patience(d, 3, 0.9)
        check_patience(d, 3, 1.0)
        check_patience(d, 3, 1.0)
        assert check_patience(d, 3, 1.0, raise_=False) is False

    with tempfile.TemporaryDirectory() as d:
        check_patience(d, 3, 1.0, cmp="max")
        check_patience(d, 3, 1.9, cmp="max")
        check_patience(d, 3, 1.0, cmp="max")
        check_patience(d, 3, 1.0, cmp="max")
        assert check_patience(d, 3, 1.0, cmp="max", raise_=False) is False
