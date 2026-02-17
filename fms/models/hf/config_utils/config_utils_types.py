"""
Types and signature defs related to mapping between transformers
and FMS config params / ModelConfigs.
"""

from transformers import PretrainedConfig
from typing import Callable

ParamBuilderFunc = Callable[[PretrainedConfig], dict]
RegistryMap = dict[str, tuple[str, ParamBuilderFunc]]
