from collections import ChainMap, OrderedDict
import os
from pathlib import Path
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Union
import torch

__adapters: MutableMapping[str, MutableMapping[str, Callable[[Mapping], Mapping]]] = {}


def register_adapter(
    architecture: str, source: str, adapter: Callable[[Mapping], Mapping]
):
    """
    Registers a state dict adapter to be available to the (de) serialization
    API.

    Args:
    architecture: The name of the model architecture, e.g. 'llama'
    source: A label representing the format of the weights to be converted.
            E.g. 'hf'
    adapter: the class of the adapter. The class must accept one constructor
                parameter, which will be a state dict (`OrderedDict`)
    """
    sources: MutableMapping[str, Callable[[Mapping], Mapping]] = {}
    if architecture in __adapters:
        sources = __adapters[architecture]
    if source in sources:
        raise KeyError(
            f"Variant {source} already registered for architecture {architecture}"
        )
    sources[source] = adapter
    __adapters[architecture] = sources


def list_sources(architecture: str):
    """
    Lists available sources (attribute formats) of a model architecture.
    E.g. `models.list_variants('llama')` -> ['meta', 'fms', 'hf']
    Args:
    architecture: one of the registered architectures returned by
                    `models.list_models()`.
    """
    if architecture not in __adapters:
        return []
    return list(__adapters[architecture].keys())


def _get_adapter(
    architecture: str, source: Optional[str]
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    if (
        source is None
        or architecture not in __adapters
        or source not in __adapters[architecture]
    ):
        # if no adapter is registered, assume the attributes are already in
        # fms format.
        # should we raise an error here instead?
        return lambda x: x
    else:
        return __adapters[architecture][source]


def get_adapted(
    architecture: str, source: Optional[str], state_dict: Mapping[str, Any]
) -> Mapping[str, Any]:
    """
    Convert a state dict to FMS format, using an adapter specified by name.

    Args:
    architecture: one of the architectures from `models.list_models()`.
                    E.g. llama.
    source: A reference to an attribute format
    state_dict: the model.state_dict() to be converted/adapted.
    """
    # sometimes we only load onto rank 0 so may not have a state_dict here.
    if not len(state_dict):
        return state_dict
    adapter = _get_adapter(architecture, source)
    adapted = adapter(state_dict)
    return adapted


# `models` imports each model class, causing models and adapters to be registered.
# down here to avoid circular dependencies.
from fms import models


def load_state_dict(
    model_path: Union[str, Path],
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    initial_device: torch.device = torch.device("cpu"),
    rank: int = 0,
    world_size: int = 1,
) -> Mapping[str, Any]:
    """
    Validates that the file(s) found at a checkpoint path are compatible with
    the intended (possibly distributed) use-case, and returns the state dict
    if needed.

    Args:
    model_path: the path to find the weights. If not set, return None.
    distributed_strategy: the kind of possibly-distributed model in which we
            intend to load these weights. E.g. tp, fsdp, None. Used for
            validation.
    checkpoint_sharding: the sharding format of the checkpoint.
            E.g. layer, tp, fsdp.
    initial_device: where the state dict will be loaded. if meta, return None.
    """
    if model_path is None or initial_device.type == "meta":
        return {}
    # TODO: Add support for tp-sharding a non-sharded state dict.
    if (
        distributed_strategy == "tp" or checkpoint_sharding == "tp"
    ) and distributed_strategy != checkpoint_sharding:
        raise ValueError(
            f"TP-sharded models are currently only compatible with TP-sharded checkpoints. Attempting to load {checkpoint_sharding} to {distributed_strategy}"
        )
    if checkpoint_sharding == "fsdp" and distributed_strategy not in ["fsdp", "hsdp"]:
        raise ValueError(f"FSDP checkpoints can only be loaded into an FSDP model")

    model_path = Path(os.path.expanduser(model_path))
    if not model_path.is_dir():
        checkpoints = [model_path]
    else:
        # TODO: add support for safetensors.
        checkpoints = sorted(model_path.glob("*.pth"))
        if not len(checkpoints):
            checkpoints = sorted(model_path.glob("*.bin"))

    if checkpoint_sharding is not None and checkpoint_sharding != "layer":
        assert world_size == len(
            checkpoints
        ), f"Loading a {checkpoint_sharding}-sharded checkpoint with len={len(checkpoints)} but world size is {world_size}"

        checkpoints = [checkpoints[rank]]

    checkpoint_sds = [
        torch.load(ckpt_path, map_location="cpu") for ckpt_path in checkpoints
    ]
    assert len(checkpoint_sds[0]), f"Unable to load checkpoint data at {model_path}"
    if len(checkpoint_sds) == 1:
        return checkpoint_sds[0]
    else:
        # layer-sharded checkpoints, e.g. as used in HF
        return ChainMap(*checkpoint_sds)
