import os
import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.types import Number

from df.modules import GroupedLinearEinsum
from df.utils import get_branch_name, get_commit_hash, get_device, get_host

_logger_initialized = False
WARN_ONCE_NO = logger.level("WARNING").no + 1
DEPRECATED_NO = logger.level("WARNING").no + 2


def init_logger(file: Optional[str] = None, level: str = "INFO", model: Optional[str] = None):
    global _logger_initialized, _duplicate_filter
    if _logger_initialized:
        logger.debug("Logger already initialized.")
    else:
        logger.remove()
        level = level.upper()
        if level.lower() != "none":
            log_format = Formatter(debug=logger.level(level).no <= logger.level("DEBUG").no).format
            logger.add(
                sys.stdout,
                level=level,
                format=log_format,
                filter=lambda r: r["level"].no not in {WARN_ONCE_NO, DEPRECATED_NO},
            )
            if file is not None:
                logger.add(
                    file,
                    level=level,
                    format=log_format,
                    filter=lambda r: r["level"].no != WARN_ONCE_NO,
                )

            logger.info(f"Running on torch {torch.__version__}")
            logger.info(f"Running on host {get_host()}")
            commit = get_commit_hash()
            if commit is not None:
                logger.info(f"Git commit: {commit}, branch: {get_branch_name()}")
            jobid = os.getenv("SLURM_JOB_ID")
            if jobid is not None:
                logger.info(f"Slurm jobid: {jobid}")
            logger.level("WARNONCE", no=WARN_ONCE_NO, color="<yellow><bold>")
            logger.add(
                sys.stderr,
                level=max(logger.level(level).no, WARN_ONCE_NO),
                format=log_format,
                filter=lambda r: r["level"].no == WARN_ONCE_NO and _duplicate_filter(r),
            )
            logger.level("DEPRECATED", no=DEPRECATED_NO, color="<yellow><bold>")
            logger.add(
                sys.stderr,
                level=max(logger.level(level).no, DEPRECATED_NO),
                format=log_format,
                filter=lambda r: r["level"].no == DEPRECATED_NO and _duplicate_filter(r),
            )
    if model is not None:
        logger.info("Loading model settings of {}", os.path.basename(model.rstrip("/")))
    _logger_initialized = True


def warn_once(message, *args, **kwargs):
    try:
        logger.log("WARNONCE", message, *args, **kwargs)
    except ValueError:
        logger.warning(message, *args, **kwargs)


def log_deprecated(message, *args, **kwargs):
    try:
        logger.log("DEPRECATED", message, *args, **kwargs)
    except ValueError:
        logger.warning(message, *args, **kwargs)


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
    if len(ks) > 2:
        try:
            return int(ks[-1])
        except ValueError:
            return 1000
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


def log_metrics(prefix: str, metrics: Dict[str, Number], level="INFO"):
    msg = ""
    stages = defaultdict(str)
    loss_msg = ""
    for n, v in sorted(metrics.items(), key=_metrics_key):
        if abs(v) > 1e-3:
            m = f" | {n}: {v: #.5f}"
        else:
            m = f" | {n}: {v: #.3E}"
        if "stage" in n:
            s = n.split("stage_")[1].split("_snr")[0]
            stages[s] += m.replace(f"stage_{s}_", "")
        elif ("valid" in prefix or "test" in prefix) and "loss" in n.lower():
            loss_msg += m
        else:
            msg += m
    for s, msg_s in stages.items():
        logger.log(level, f"{prefix} | stage {s}" + msg_s)
    if len(stages) == 0:
        logger.log(level, prefix + msg)
    if len(loss_msg) > 0:
        logger.log(level, prefix + loss_msg)


class DuplicateFilter:
    """
    Filters away duplicate log messages.
    Modified version of: https://stackoverflow.com/a/60462619
    """

    def __init__(self):
        self.msgs = set()

    def __call__(self, record) -> bool:
        k = f"{record['level']}{record['message']}"
        if k in self.msgs:
            return False
        else:
            self.msgs.add(k)
            return True


_duplicate_filter = DuplicateFilter()


def log_model_summary(model: torch.nn.Module, verbose=False, force=False):
    try:
        import ptflops
    except ImportError as e:
        if not force:
            logger.debug("Failed to import ptflops. Cannot print model summary.")
            return
        else:
            raise e

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

    warnings.filterwarnings("ignore", "RNN module weights", category=UserWarning, module="torch")
    macs, params = ptflops.get_model_complexity_info(
        deepcopy(model),
        (t,),
        input_constructor=lambda _: {"spec": spec, "feat_erb": feat_erb, "feat_spec": feat_spec},
        as_strings=False,
        print_per_layer_stat=verbose,
        verbose=verbose,
        custom_modules_hooks={
            GroupedLinearEinsum: grouped_linear_flops_counter_hook,
        },
    )
    logger.info(f"Model complexity: {params/1e6:.3f}M #Params, {macs/1e6:.1f}M MACS")


def grouped_linear_flops_counter_hook(module: GroupedLinearEinsum, input, output):
    # input: ([B, T, I],)
    # output: [B, T, H]
    input = input[0]  # [B, T, I]
    output_last_dim = module.weight.shape[-1]
    input = input.unflatten(-1, (module.groups, module.ws))  # [B, T, G, I/G]
    # GroupedLinear calculates "...gi,...gih->...gh"
    weight_flops = np.prod(input.shape) * output_last_dim
    module.__flops__ += int(weight_flops)  # type: ignore
