from importlib import import_module

import torch
from loguru import logger

from df.config import DfParams, config


class ModelParams(DfParams):
    def __init__(self):
        self.__model = config("MODEL", default="deepfilternet3", section="train")
        self.__params = getattr(import_module("df." + self.__model), "ModelParams")()

    def __getattr__(self, attr: str):
        return getattr(self.__params, attr)


def init_model(*args, **kwargs):
    """Initialize the model specified in the config."""
    model = config("MODEL", default="deepfilternet3", section="train")
    logger.info(f"Initializing model `{model}`")
    model = getattr(import_module("df." + model), "init_model")(*args, **kwargs)
    model.to(memory_format=torch.channels_last)
    return model
