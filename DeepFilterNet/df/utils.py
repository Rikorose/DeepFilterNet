import collections
import math
import os
import random
import subprocess
import warnings
from typing import Any, Set, Union

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch._six import string_classes
from torch.types import Number

from df.model import ModelParams


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
    git_dir = get_git_root()
    args = ["git", "-C", git_dir, "rev-parse", "--short", "--verify", "HEAD"]
    return subprocess.check_output(args).strip().decode()


def get_host() -> str:
    return os.uname().nodename


def get_branch_name():
    git_dir = os.path.dirname(os.path.abspath(__file__))
    args = ["git", "-C", git_dir, "rev-parse", "--abbrev-ref", "HEAD"]
    return subprocess.check_output(args).strip().decode()


def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    r"""Pytorch 1.9 backport: Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == torch._six.inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    if total_norm.isnan() or total_norm.isinf():
        if error_if_nonfinite:
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, "
                "set `error_if_nonfinite=False`"
            )
        else:
            warnings.warn(
                "Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway. "
                "Note that the default behavior will change in a future release to error out "
                "if a non-finite total norm is encountered. At that point, setting "
                "error_if_nonfinite=false will be required to retain the old behavior.",
                FutureWarning,
                stacklevel=2,
            )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm


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
