from typing import Dict, Type, Union

import torch.nn as nn


class GELUTanh(nn.GELU):
    def __init__(self):
        super().__init__(approximate="tanh")


"""
Simple dict which given an activation string, return an activation function class
"""
__ACT_2_CLS: Dict[str, Type[nn.Module]] = {
    "gelu": nn.GELU,
    "gelu-tanh": GELUTanh,
    "gelu_pytorch_tanh": GELUTanh,
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}

"""
Simple dict which given an activation class, returns the activation string

Note: SiLU will always return swish when using this dict
"""
__CLS_2_ACT: Dict[Type[nn.Module], str] = {v: k for k, v in __ACT_2_CLS.items()}


def str_to_activation(activation_str: str) -> nn.Module:
    """Convert an activation string to an instantiated activation function

    Parameters
    ----------
    activation_str: str
        the activation key to convert

    Returns
    -------
    nn.Module
        one of nn.GELU, nn.Mish, nn.ReLU, nn.Sigmoid, nn.SiLU, nn.Tanh depending on the key given
    """
    activation_str = activation_str.lower()
    if activation_str not in __ACT_2_CLS.keys():
        raise ValueError(f"activation string must be one of {__ACT_2_CLS.keys()}")
    return __ACT_2_CLS[activation_str.lower()]()


def activation_to_str(activation: Union[Type[nn.Module], nn.Module]) -> str:
    """Convert an activation function or activation class to its string representation

    Parameters
    ----------
    activation: type(nn.Module) or nn.Module
        the activation key to convert

    Returns
    -------
    str
        one of "gelu", "mish", "relu", "sigmoid", "silu", "swish", "tanh" depending on the key given
    """
    if not isinstance(activation, type):
        activation = type(activation)

    if activation not in (a for a in __CLS_2_ACT.keys()):
        raise TypeError(
            f"activation module or activation module type must be one of {__CLS_2_ACT.keys()}"
        )
    return __CLS_2_ACT[activation]
