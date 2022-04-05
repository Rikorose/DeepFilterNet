import os
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.types import Number

from df.multistagenet import GroupedGRULayerMS, GroupedLinearMS, LocalLinearMS
from df.utils import get_branch_name, get_commit_hash, get_device, get_host

_logger_initialized = False
WARN_ONCE_NO = logger.level("WARNING").no + 1


def init_logger(file: Optional[str] = None, level: str = "INFO", model: Optional[str] = None):
    global _logger_initialized, _duplicate_filter
    if _logger_initialized:
        logger.debug("Logger already initialized.")
        return
    logger.remove()
    level = level.upper()
    if level.lower() != "none":
        log_format = Formatter(debug=level == "DEBUG").format
        logger.add(
            sys.stdout,
            level=level,
            format=log_format,
            filter=lambda r: r["level"].no != WARN_ONCE_NO,
        )
        if file is not None:
            logger.add(
                file, level=level, format=log_format, filter=lambda r: r["level"].no != WARN_ONCE_NO
            )

        if model is not None:
            logger.info("Loading model settings of {}", os.path.basename(model.rstrip("/")))
        logger.info(f"Running on torch {torch.__version__}")
        logger.info(f"Running on host {get_host()}")
        commit = get_commit_hash()
        if commit is not None:
            logger.info(f"Git commit: {commit}, branch: {get_branch_name()}")
        if (jobid := os.getenv("SLURM_JOB_ID")) is not None:
            logger.info(f"Slurm jobid: {jobid}")
        logger.level("WARNONCE", no=WARN_ONCE_NO, color="<yellow><bold>")
        logger.add(
            sys.stderr,
            level=max(logger.level(level).no, WARN_ONCE_NO),
            format=log_format,
            filter=lambda r: r["level"].no == WARN_ONCE_NO and _duplicate_filter(r),
        )
    _logger_initialized = True


def warn_once(message, *args, **kwargs):
    logger.log("WARNONCE", message, *args, **kwargs)


class Formatter:
    def __init__(self, debug=False):
        if debug:
            self.fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
                " | <level>{level: <8}</level>"
                " | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
                " | <level>{message}</level>"
            )
        else:
            self.fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
                " | <level>{level: <8}</level>"
                " | <cyan>DF</cyan>"
                " | <level>{message}</level>"
            )
        self.fmt += "\n{exception}"

    def format(self, record):
        if record["level"].no == WARN_ONCE_NO:
            return self.fmt.replace("{level: <8}", "WARNING ")
        return self.fmt


def _metrics_key(k_: Tuple[str, float]):
    k0 = k_[0]
    ks = k0.split("_")
    if len(ks) >= 2:
        return int(ks[-1])
    elif k0 == "loss":
        return -999
    elif "loss" in k0.lower():
        return -998
    elif k0 == "lr":
        return 998
    elif k0 == "wd":
        return 999
    else:
        return -101


def log_metrics(prefix: str, metrics: Dict[str, Number]):
    msg = ""
    stages = defaultdict(str)
    loss_msg = ""
    for n, v in sorted(metrics.items(), key=_metrics_key):
        if abs(v) > 1e-3:
            m = f" | {n}: {v:.5f}"
        else:
            m = f" | {n}: {v:.3E}"
        if "stage" in n:
            s = n.split("stage_")[1].split("_snr")[0]
            stages[s] += m.replace(f"stage_{s}_", "")
        elif ("valid" in prefix or "test" in prefix) and "loss" in n.lower():
            loss_msg += m
        else:
            msg += m
    for s, msg_s in stages.items():
        logger.info(f"{prefix} | stage {s}" + msg_s)
    if len(stages) == 0:
        logger.info(prefix + msg)
    if len(loss_msg) > 0:
        logger.info(prefix + loss_msg)


class DuplicateFilter:
    """
    Filters away duplicate log messages.
    Modified version of: https://stackoverflow.com/a/60462619
    """

    def __init__(self):
        self.msgs = set()

    def __call__(self, record) -> bool:
        rv = record["message"] not in self.msgs
        self.msgs.add(record["message"])
        return rv


_duplicate_filter = DuplicateFilter()


def log_model_summary(model: torch.nn.Module, verbose=False):
    import ptflops

    from df.model import ModelParams

    # Generate input of 1 second audio
    # Necessary inputs are:
    #   spec: [B, 1, T, F, 2], F: freq bin
    #   feat_erb: [B, 1, T, E], E: ERB bands
    #   feat_spec: [B, 2, T, C*2], C: Complex features
    p = ModelParams()
    b = 1
    t = p.sr // p.hop_size
    device = get_device()
    spec = torch.randn([b, 1, t, p.fft_size // 2 + 1, 2]).to(device)
    feat_erb = torch.randn([b, 1, t, p.nb_erb]).to(device)
    feat_spec = torch.randn([b, 1, t, p.nb_df, 2]).to(device)

    macs, params = ptflops.get_model_complexity_info(
        deepcopy(model),
        (t,),
        input_constructor=lambda _: {"spec": spec, "feat_erb": feat_erb, "feat_spec": feat_spec},
        as_strings=False,
        print_per_layer_stat=verbose,
        verbose=verbose,
        custom_modules_hooks={
            GroupedLinearMS: grouped_linear_flops_counter_hook,
            LocalLinearMS: local_linear_flops_counter_hook,
            GroupedGRULayerMS: grouped_gru_flops_counter_hook,
        },
    )
    logger.info(f"Model complexity: {params/1e6:.3f}M #Params, {macs/1e6:.1f}M MACS")


def grouped_linear_flops_counter_hook(module: GroupedLinearMS, input, output):
    # input: ([B, Ci, T, F],)
    # output: [B, Co, T, F]
    input = input[0]  # [B, C, T, F]
    output_last_dim = module.weight.shape[-1]
    input = input.permute(0, 2, 3, 1)
    input = input.unflatten(2, (module.n_groups, module.n_unfold))  # [B, T, G, F/G, Ci]
    bias_flops = np.prod(output.shape) if module.bias is not None else 0
    # GroupedLinear calculates "btfgi,fgio->btfgo"
    weight_flops = np.prod(input.shape) * output_last_dim
    module.__flops__ += int(weight_flops + bias_flops)  # type: ignore


def local_linear_flops_counter_hook(module: LocalLinearMS, input, output):
    # input: ([B, Ci, T, F],)
    # output: [B, Co, T, F]
    input = input[0]  # [B, Ci, T, F]
    output_last_dim = module.weight.shape[1]
    bias_flops = np.prod(output.shape) if module.bias is not None else 0
    # LocalLinear calculates "bitf,iof->botf"
    weight_flops = np.prod(input.shape) * output_last_dim
    module.__flops__ += int(weight_flops + bias_flops)  # type: ignore


def grouped_gru_flops_counter_hook(module: GroupedGRULayerMS, input, output):
    # input: ([B, Ci, T, F],)
    # output: ([B, Co, T, F],)
    input = input[0]  # [B, Ci, T, F]
    output = output[0]  # [B, Ci, T, F]
    input = input.permute(0, 2, 3, 1)  # [B, T, F, Ci]
    input = input.unflatten(2, (module.n_groups, module.g_freqs)).flatten(3)  # [B, T, F/G, G*Ci]
    input_shape = list(input.shape)
    # 2 for bias ih and hh, 3 for r,z,n
    bias_flops = 2 * 3 * np.prod(input_shape) if module.bias_hh_l is not None else 0
    # GroupedGRULayer input calculates "btgi,goi->btgo"
    weight_flops = np.prod(input_shape) * module.weight_hh_l.shape[-2]
    # GroupedGRULayer hidden calculates t*"bgo,gpo->bgp"
    input_shape[-1] = module.weight_ih_l.shape[-2]
    weight_flops += np.prod(input.shape) * module.weight_hh_l.shape[-2]
    module.__flops__ += int(weight_flops + bias_flops)  # type: ignore
