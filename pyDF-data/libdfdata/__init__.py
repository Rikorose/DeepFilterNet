from .libdfdata import _FdDataLoader, _TdDataLoader
from .libdfdata import __all__, __doc__

has_torch = False
try:
    import torch  # noqa

    has_torch = True
except ImportError:
    pass

if has_torch:
    from .torch_dataloader import PytorchDataLoader

__all__ += ["PytorchDataLoader"]
