import collections
import math
import os
import random
import subprocess
from socket import gethostname
from typing import Any, Set, Union

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch._six import string_classes
from torch.types import Number

from df.config import config
from df.model import ModelParams

try:
    from torchaudio.functional import resample as ta_resample
except ImportError:
    from torchaudio.compliance.kaldi import resample_waveform as ta_resample  # type: ignore


def resample(audio: Tensor, orig_sr: int, new_sr: int, method="sinc_fast"):
    params = {
        "sinc_fast": {"resampling_method": "sinc_interpolation", "lowpass_filter_width": 16},
        "sinc_best": {"resampling_method": "sinc_interpolation", "lowpass_filter_width": 64},
        "kaiser_fast": {
            "resampling_method": "kaiser_window",
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "beta": 8.555504641634386,
        },
        "kaiser_best": {
            "resampling_method": "kaiser_window",
            "lowpass_filter_width": 16,
            "rolloff": 0.9475937167399596,
            "beta": 14.769656459379492,
        },
    }
    assert method in params.keys(), f"method must be one of {list(params.keys())}"
    return ta_resample(audio, orig_sr, new_sr, **params[method])


def get_device():
    s = config("DEVICE", default="", section="train")
    if s == "":
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda:0")
        else:
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(s)
    return DEVICE


def as_complex(x: Tensor):
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Last dimension need to be of length 2 (re + im), but got {x.shape}")
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)


def as_real(x: Tensor):
    if torch.is_complex(x):
        return torch.view_as_real(x)
    return x


def check_finite_module(obj, name="Module", _raise=True) -> Set[str]:
    out: Set[str] = set()
    if isinstance(obj, torch.nn.Module):
        for name, child in obj.named_children():
            out = out | check_finite_module(child, name)
        for name, param in obj.named_parameters():
            out = out | check_finite_module(param, name)
        for name, buf in obj.named_buffers():
            out = out | check_finite_module(buf, name)
    if _raise and len(out) > 0:
        raise ValueError(f"{name} not finite during checkpoint writing including: {out}")
    return out


def make_np(x: Union[Tensor, np.ndarray, Number]) -> np.ndarray:
    """Transforms Tensor to numpy.
    Args:
      x: An instance of torch tensor or caffe blob name

    Returns:
        numpy.array: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    raise NotImplementedError(
        "Got {}, but numpy array, scalar, or torch tensor are expected.".format(type(x))
    )


def get_norm_alpha(log: bool = True) -> float:
    p = ModelParams()
    a_ = _calculate_norm_alpha(sr=p.sr, hop_size=p.hop_size, tau=p.norm_tau)
    precision = 3
    a = 1.0
    while a >= 1.0:
        a = round(a_, precision)
        precision += 1
    if log:
        logger.info(f"Running with normalization window alpha = '{a}'")
    return a


def _calculate_norm_alpha(sr: int, hop_size: int, tau: float):
    """Exponential decay factor alpha for a given tau (decay window size [s])."""
    dt = hop_size / sr
    return math.exp(-dt / tau)


def check_manual_seed(seed: int = None):
    """If manual seed is not specified, choose a random one and communicate it to the user."""
    seed = seed or random.randint(1, 10000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_git_root():
    git_local_dir = os.path.dirname(os.path.abspath(__file__))
    args = ["git", "-C", git_local_dir, "rev-parse", "--show-toplevel"]
    return subprocess.check_output(args).strip().decode()


def get_commit_hash():
    """Returns the current git commit."""
    try:
        git_dir = get_git_root()
        args = ["git", "-C", git_dir, "rev-parse", "--short", "--verify", "HEAD"]
        commit = subprocess.check_output(args).strip().decode()
    except subprocess.CalledProcessError:
        # probably not in git repo
        commit = None
    return commit


def get_host() -> str:
    return gethostname()


def get_branch_name():
    try:
        git_dir = os.path.dirname(os.path.abspath(__file__))
        args = ["git", "-C", git_dir, "rev-parse", "--abbrev-ref", "HEAD"]
        branch = subprocess.check_output(args).strip().decode()
    except subprocess.CalledProcessError:
        # probably not in git repo
        branch = None
    return branch


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0,
    warmup_steps: int = -1,
    initial_ep_per_cycle: float = -1,
    cycle_decay: float = 1,
    cycle_mul: float = 1,
):
    """Adopted from official ConvNeXt repo."""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters_after_warmup = epochs * niter_per_ep - warmup_iters
    if initial_ep_per_cycle == -1:
        initial_ep_per_cycle = iters_after_warmup
        num_cycles = 1
        cycle_lengths = [iters_after_warmup]
        assert cycle_decay == 1
        assert cycle_mul == 1
    else:
        initial_cycle_iter = int(round(initial_ep_per_cycle * niter_per_ep))
        if cycle_mul == 1:
            num_cycles = int(math.ceil(iters_after_warmup / (initial_ep_per_cycle * niter_per_ep)))
            cycle_lengths = [initial_cycle_iter] * num_cycles
        else:
            num_cycles = 0
            cycle_lengths = []
            i = 0
            while sum(cycle_lengths) < iters_after_warmup:
                num_cycles += 1
                cycle_lengths.append(initial_cycle_iter * cycle_mul**i)
                i += 1
    schedule_cycles = []
    for i in range(num_cycles):
        cycle_base_value = base_value * cycle_decay**i
        iters = np.arange(cycle_lengths[i])
        schedule = np.array(
            [
                final_value
                + 0.5
                * (cycle_base_value - final_value)
                * (1 + math.cos(math.pi * i / (len(iters))))
                for i in iters
            ]
        )
        schedule_cycles.append(schedule)

    schedule = np.concatenate((warmup_schedule, *schedule_cycles))
    schedule = schedule[: epochs * niter_per_ep]

    assert len(schedule) == epochs * niter_per_ep
    return schedule


# from pytorch/ignite:
def apply_to_tensor(input_, func):
    """Apply a function on a tensor or mapping, or sequence of tensors."""
    if isinstance(input_, torch.nn.Module):
        return [apply_to_tensor(c, func) for c in input_.children()]
    elif isinstance(input_, torch.nn.Parameter):
        return func(input_.data)
    elif isinstance(input_, Tensor):
        return func(input_)
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: apply_to_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Iterable):
        return [apply_to_tensor(sample, func) for sample in input_]
    elif input_ is None:
        return input_
    else:
        return input_


def detach_hidden(hidden: Any) -> Any:
    """Cut backpropagation graph.
    Auxillary function to cut the backpropagation graph by detaching the hidden
    vector.
    """
    return apply_to_tensor(hidden, Tensor.detach)
