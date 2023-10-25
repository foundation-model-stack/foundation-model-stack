from contextlib import nullcontext
import os
from pathlib import Path
import re
from typing import Callable, List
import torch
from torch import nn

__models = {}


def register_model(architecture, variant, factory):
    variants = {}
    if architecture in __models:
        variants = __models[architecture]
    variants[variant] = factory
    __models[architecture] = variants


def list_models():
    return list(__models.keys())


def list_variants(architecture: str):
    return list(__models[architecture].keys())


def get_model(
    architecture: str, variant: str, *, dtype=None, device=None, extra_args: dict = {}
) -> nn.Module:
    """
    Args:
    architecture: one of the architectures from list_models(). E.g. llama.
    variant: one of the variants from list_variants(architecture). E.g. '7b'
    extra_args: kwargs to be passed to the model factory.
    """
    model_factory = __models[architecture][variant]

    orig = torch.get_default_dtype()

    try:
        if dtype is not None:
            torch.set_default_dtype(dtype)
        with device if device is not None else nullcontext():
            return model_factory(**extra_args)
    finally:
        torch.set_default_dtype(orig)


from fms.models import llama
