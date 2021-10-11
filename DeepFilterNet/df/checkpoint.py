import glob
import os
import re
from typing import Union

import torch
from df.utils import check_finite_module
from loguru import logger
from torch import nn


def get_epoch(cp) -> int:
    return int(os.path.splitext(os.path.basename(cp))[0].split("_")[-1])


def read_cp(
    obj: Union[torch.optim.Optimizer, nn.Module],
    name: str,
    dirname: str,
    epoch="latest",
    extension="ckpt",
    blacklist=[],
):
    checkpoints = glob.glob(os.path.join(dirname, f"{name}*.{extension}"))
    if len(checkpoints) == 0:
        return None
    latest = max(checkpoints, key=get_epoch)
    epoch = get_epoch(latest)
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
            logger.warning(f"Unexpected key: {key}")
        return epoch
    obj.load_state_dict(latest)


def write_cp(
    obj: Union[torch.optim.Optimizer, nn.Module],
    name: str,
    dirname: str,
    epoch: int,
    extension="ckpt",
):
    check_finite_module(obj)
    cp_name = os.path.join(dirname, f"{name}_{epoch}.{extension}")
    logger.info(f"Writing checkpoint {cp_name} with epoch {epoch}")
    torch.save(obj.state_dict(), cp_name)
    cleanup(name, dirname, extension)


def cleanup(name: str, dirname: str, extension: str, nkeep=5):
    checkpoints = glob.glob(os.path.join(dirname, f"{name}*.{extension}"))
    if len(checkpoints) == 0:
        return
    checkpoints = sorted(checkpoints, key=get_epoch, reverse=True)
    for cp in checkpoints[nkeep:]:
        logger.debug("Removing old checkpoint: {}".format(cp))
        os.remove(cp)
