has_torch = False
try:
    import torch  # noqa

    has_torch = True
except ImportError:
    pass

if has_torch:
    from .torch_dataloader import PytorchDataLoader

__all__ = ["PytorchDataLoader"]
