import os
import sys
from typing import Dict, Optional

import torch
from loguru import logger
from torch.types import Number

from df.utils import get_branch_name, get_commit_hash, get_host


def init_logger(file: Optional[str] = None, level: str = "INFO"):
    logger.remove()

    log_format = get_log_format(debug=level == "DEBUG")
    logger.add(sys.stdout, level=level, format=log_format)
    if file is not None:
        logger.add(file, level=level, format=log_format)

    logger.info(f"Running on torch {torch.__version__}")
    logger.info(f"Running on host {get_host()}")
    logger.info(f"Git commit: {get_commit_hash()}, branch: {get_branch_name()}")
    if (jobid := os.getenv("SLURM_JOB_ID")) is not None:
        logger.info(f"Slurm jobid: {jobid}")


def get_log_format(debug=False):
    if debug:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
            " | <level>{level: <8}</level>"
            " | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
            " | <level>{message}</level>"
        )
    else:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
            " | <level>{level: <8}</level>"
            " | <cyan>DF</cyan>"
            " | <level>{message}</level>"
        )


def log_metrics(prefix: str, metrics: Dict[str, Number]):
    msg = prefix
    for n, v in sorted(metrics.items()):
        msg += f" | {n}: {v:.5g}"
    logger.info(msg)


def log_model_summary(model: torch.nn.Module):
    import ptflops
    import torchinfo

    from df.model import ModelParams

    # Generate input of 1 second audio
    # Necessary inputs are:
    #   spec: [B, 1, T, F, 2], F: freq bin
    #   feat_erb: [B, 1, T, E], E: ERB bands
    #   feat_spec: [B, 2, T, C*2], C: Complex features
    #
    p = ModelParams()
    b = 1
    t = p.sr // p.hop_size
    spec = torch.randn([b, 1, t, p.fft_size // 2 + 1, 2])
    feat_erb = torch.randn([b, 1, t, p.nb_erb])
    feat_spec = torch.randn([b, 1, t, p.nb_df, 2])
    inputs = (spec, feat_erb, feat_spec)
    # s = torchinfo.summary(
    #    model,
    #    input_data=inputs,
    #    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    #    batch_dim=1,
    #    depth=12,
    #    verbose=0,
    # )
    # s.summary_list = [x for x in s.summary_list if "act" not in x.var_name.lower()]
    # ic(s)

    # macs, params = ptflops.get_model_complexity_info(
    #    model.enc,
    #    (t,),
    #    input_constructor=lambda _: {"feat_erb": feat_erb, "feat_spec": feat_spec},
    #    as_strings=True,
    #    print_per_layer_stat=True,
    #    verbose=True,
    # )
    # ic(macs, params)

    # model.run_df=False
    macs, params = ptflops.get_model_complexity_info(
        model,
        (t,),
        input_constructor=lambda _: {"spec": spec, "feat_erb": feat_erb, "feat_spec": feat_spec},
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )
    ic(macs, params)
