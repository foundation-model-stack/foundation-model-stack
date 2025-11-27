"""
Types and signature defs related to mapping between transformers
and FMS config params / ModelConfigs.
"""

from transformers import PretrainedConfig
from typing import Callable

# Define a type alias for a function that takes an int and returns a str
ParamBuilderFunc = Callable[[PretrainedConfig], dict]
RegistryMap = dict[str, tuple[str, ParamBuilderFunc]]
