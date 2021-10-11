from importlib import import_module

from icecream import ic, install
from loguru import logger

from df.config import DfParams, config

install()
ic.includeContext = True


class ModelParams(DfParams):
    def __init__(self):
        self.__model = config("MODEL", default="deepfilternet", section="train")
        self.__params = getattr(import_module("df." + self.__model), "ModelParams")()

    def __getattr__(self, attr: str):
        return getattr(self.__params, attr)


def init_model(*args, **kwargs):
    """Initialize the model specified in the config."""
    model = config("MODEL", default="deepfilternet", section="train")
    logger.info(f"Initializing model `{model}`")
    return getattr(import_module("df." + model), "init_model")(*args, **kwargs)
