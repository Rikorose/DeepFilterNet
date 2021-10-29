has_torch = False
try:
    import torch  # noqa

    has_torch = True
except ImportError:
    pass

__all__ = []
if has_torch:
    from .torch_dataloader import PytorchDataLoader

    __all__.append("PytorchDataLoader")
