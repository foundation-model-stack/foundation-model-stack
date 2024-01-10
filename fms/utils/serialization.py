import collections
import itertools
import os
from collections import ChainMap
from collections.abc import Iterable
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import torch

from fms.modules.tp import TPModule


__adapters: MutableMapping[str, MutableMapping[str, Callable[[Mapping], Mapping]]] = {}
__fusable_weight_groups: MutableMapping[
    str, MutableMapping[str, Callable[[List[str]], List[Union[str, List[str]]]]]
] = {}


def register_adapter(
    architecture: str,
    source: str,
    adapter: Callable[[Mapping], Mapping],
    fwg_adapter: Optional[Callable[[List[str]], List[Union[str, List[str]]]]] = None,
):
    """
    Registers a state dict adapter to be available to the (de) serialization
    API. Optionally registers which weights are to be fused together so the
    lazy loaders can properly work with the adapter

    Args:
    architecture: The name of the model architecture, e.g. 'llama'
    source: A label representing the format of the weights to be converted.
            E.g. 'hf'
    adapter: the class of the adapter. The class must accept one constructor
                parameter, which will be a state dict (`OrderedDict`)
    fwg_adapter: Optional[Callable[[List[str]], List[Union[str, List[str]]]]]
        Function that turns a single list of weights into groups that need to
        go together into the adapter function. For example, if a checkpoint has
        unfused q,k,v matrices, but the FMS model requires a fused qkv, adapter
        must be called with the three weights together always. This is needed
        so that lazy loaders know which weights to consider together.
    """
    sources: MutableMapping[str, Callable[[Mapping], Mapping]] = {}
    fusable_weight_groups: MutableMapping[
        str, Callable[[List[str]], List[Union[str, List[str]]]]
    ] = {}
    if architecture in __adapters:
        sources = __adapters[architecture]
    if architecture in __fusable_weight_groups:
        fusable_weight_groups = __fusable_weight_groups[architecture]

    if source in sources:
        raise KeyError(
            f"Variant {source} already registered for architecture {architecture}"
        )

    sources[source] = adapter
    if fwg_adapter is not None:
        fusable_weight_groups[source] = fwg_adapter
    __adapters[architecture] = sources
    __fusable_weight_groups[architecture] = fusable_weight_groups


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
) -> Tuple[
    Callable[[Mapping[str, Any]], Mapping[str, Any]],
    Optional[Callable[[List[str]], List[Union[str, List[str]]]]],
]:
    if (
        source is None
        or architecture not in __adapters
        or source not in __adapters[architecture]
    ):
        # if no adapter is registered, assume the attributes are already in
        # fms format.
        # should we raise an error here instead?
        return lambda x: x, None
    else:
        return __adapters[architecture][source], __fusable_weight_groups[
            architecture
        ].get(source, None)


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
    adapter, _ = _get_adapter(architecture, source)
    adapted = adapter(state_dict)
    return adapted


# `models` imports each model class, causing models and adapters to be registered.
# down here to avoid circular dependencies.
from fms import models


def get_ckp_format(model_path: Union[str, Path]) -> str:
    """
    Returns the checkpoint format of a model checkpoint. If format
    is not recognized, assumes all files are regular pytorch
    checkpoints and returns "pt"

    Args:
    model_path: the path to find the weights.
    """
    model_path = Path(os.path.expanduser(model_path))
    if (
        model_path.suffix == ".safetensors"
        or len(sorted(model_path.glob("*.safetensors"))) > 0
    ):
        return "st"
    if model_path.suffix == ".pth" or len(sorted(model_path.glob("*.pth"))) > 0:
        return "pt"
    if model_path.suffix == ".bin" or len(sorted(model_path.glob("*.bin"))) > 0:
        return "hf"
    return "pt_all"


def load_state_dict(
    model_path: Union[str, Path],
    model: torch.nn.Module,
    architecture: str,
    source: str,
    *,
    checkpoint_format: Optional[str] = None,
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    initial_device: torch.device = torch.device("cpu"),
    rank: int = 0,
    world_size: int = 1,
) -> torch.nn.Module:
    """
    Validates that the file(s) found at a checkpoint path are compatible with
    the intended (possibly distributed) use-case, and returns the model with
    the weights loaded on it. For memory efficiency, the model is modified
    in-place.

    If model_path is a directory, it'll try to load, in this order, pytorch
    models (.pth format), HF models (.bin format), and safetensors
    (.safetensors format).

    Args:
    model_path: the path to find the weights. If not set, return None.
    model: The pytorch model where the state dict will be loaded. Useful for
            sharded loading of the state dict
    architecture: The architecture of the model
    source: The source of the checkpoint
    checkpoint_format: how the checkpoint files are saved: None, 'pt',
            'hf', or 'st'. If None, guess based on files.
    distributed_strategy: the kind of possibly-distributed model in which we
            intend to load these weights. E.g. tp, fsdp, None. Used for
            validation.
    checkpoint_sharding: the sharding format of the checkpoint.
            E.g. layer, tp, fsdp.
    initial_device: where the state dict will be loaded. if meta, return None.
    """
    if model_path is None or initial_device.type == "meta":
        return model
    if checkpoint_sharding == "fsdp" and distributed_strategy not in ["fsdp", "hsdp"]:
        raise ValueError(f"FSDP checkpoints can only be loaded into an FSDP model")

    model_path = Path(os.path.expanduser(model_path))

    if checkpoint_format is None:
        # Try finding the checkpoint format internally
        checkpoint_format = get_ckp_format(model_path)

    if checkpoint_format == "pt":
        glob_pattern = "*.pth"
    elif checkpoint_format == "hf":
        glob_pattern = "*.bin"
    elif checkpoint_format == "st":
        glob_pattern = "*.safetensors"
    elif checkpoint_format == "pt_all":
        # Assume whatever file(s) are PT checkpoint(s)
        glob_pattern = "*"
    else:
        raise ValueError(f"Unsupported checkpoint format {checkpoint_format}")
    if model_path.is_file():
        checkpoints = [model_path]
        assert (
            model_path.suffix == glob_pattern[1:]
        ), f"Checkpoint {model_path} is not a {checkpoint_format} checkpoint"
    else:
        checkpoints = sorted(model_path.glob(glob_pattern))

    # Check if the requested file format matches the file format
    assert (
        len(checkpoints) > 0
    ), f"Can't find the requested checkpoint data at {model_path} for format {checkpoint_format}"

    if checkpoint_sharding is not None and checkpoint_sharding != "layer":
        assert world_size == len(
            checkpoints
        ), f"Loading a {checkpoint_sharding}-sharded checkpoint with len={len(checkpoints)} but world size is {world_size}"

        checkpoints = [checkpoints[rank]]
    sd_adapter, fwg_adapter = _get_adapter(architecture, source)

    tp_shard_ckp = distributed_strategy == "tp" and checkpoint_sharding != "tp"
    if checkpoint_format == "st":
        for ckp in checkpoints:
            load_safetensors_checkpoint(
                model,
                ckp,
                sd_adapter,
                fwg_adapter,
                tp_shard_ckp,
                initial_device,
                rank,
                world_size,
            )
    else:
        with torch.no_grad():
            [
                _load_partial_state_dict(
                    model,
                    sd_adapter(torch.load(ckpt_path, map_location=initial_device)),
                    tp_shard_ckp,
                    rank,
                    world_size,
                )
                for ckpt_path in checkpoints
            ]
    return model


def load_safetensors_checkpoint(
    model: torch.nn.Module,
    checkpoint: Path,
    adapter: Callable[[Mapping], Mapping],
    fwg_adapter: Optional[Callable[[List[str]], List[Union[str, List[str]]]]],
    tp_shard: bool,
    device,
    rank,
    world_size,
):
    """
    This function loads a safetensors unsharded (or layer-sharded) checkpoint into a model (possibly TP)
    with an arbitrary number of ranks.

    Args
    ====
    model: torch.nn.Module
        Model where the weights will be loaded
    state_dict: Mapping[str, any]
        Dictionary mapping FMS weight names to files and their weight names
    adapter: Callable[[Mapping], Mapping]
        Function that returns an FMS state_dict from a source state_dict.
        E.g. turns meta checkpoints into FMS checkpoints
    fwg_adapter: Optional[Callable[[List[str]], List[Union[str, List[str]]]]]
        Function that turns a single list of weights into groups that need to
        go together into the adapter function. For example, if a checkpoint has
        unfused q,k,v matrices, but the FMS model requires a fused qkv, adapter
        must be called with the three weights together always. This is needed
        so that lazy loaders know which weights to consider together.
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    """
    from safetensors import safe_open  # type: ignore[import-untyped]

    with torch.no_grad():
        with safe_open(
            checkpoint, framework="pt", device=str(device)
        ) as model_weights:  # type: ignore[attr-defined]
            sd_keys = list(model_weights.keys())
            if fwg_adapter is not None:
                key_groups = fwg_adapter(sd_keys)
            else:
                key_groups = sd_keys
            for key_group in key_groups:
                if not isinstance(key_group, list):
                    key_group = [key_group]
                keys_sd = {}
                for key in key_group:
                    keys_sd[key] = model_weights.get_tensor(key)
                fms_sd = adapter(keys_sd)
                _load_partial_state_dict(model, fms_sd, tp_shard, rank, world_size)


def _copy_colwise(param: torch.nn.Parameter, tensor_value, is_bias, rank, world_size):
    """
    This function copies the correct shard of the weights for a colwise-TP'd module
    according to the rank of the process and the world_size.

    Args
    ====
    param: torch.nn.Parameter
        Parameter that has had TP applied
    tensor_value: torch.Tensor
        tensor that needs sharding
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    """
    # Divide the weight matrix along the first dimension.
    output_size_per_partition = param.shape[0]
    if not is_bias:
        tensor = tensor_value[
            (rank * output_size_per_partition) : (
                (rank + 1) * output_size_per_partition
            ),
            :,
        ]
    else:
        tensor = tensor_value[
            (rank * output_size_per_partition) : (
                (rank + 1) * output_size_per_partition
            )
        ]
    param.copy_(tensor, non_blocking=True)


def _copy_rowwise(param: torch.nn.Parameter, tensor_value, is_bias, rank, world_size):
    """
    This function copies the correct shard of the weights for a rowwise-TP'd module
    according to the rank of the process and the world_size.

    Args
    ====
    param: torch.nn.Parameter
        Parameter that has had TP applied
    tensor_value: torch.Tensor
        tensor that needs sharding
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    """
    # Divide the weight matrix along the last dimension.
    if not is_bias:
        output_size_per_partition = param.shape[1]
        tensor = tensor_value[
            :,
            (rank * output_size_per_partition) : (
                (rank + 1) * output_size_per_partition
            ),
        ]
        param.copy_(tensor, non_blocking=True)
    else:
        if rank == 0:
            _copy_if_present(param, tensor_value)
        else:
            param.zero_()


def _copy_embedding(param: torch.nn.Parameter, tensor_value, rank, world_size):
    """
    This function copies the correct shard of the weights for a TP'd embedding module
    according to the rank of the process and the world_size.

    Args
    ====
    param: torch.nn.Parameter
        Parameter that has had TP applied
    tensor_value: torch.Tensor
        tensor that needs sharding
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    """
    # Divide the weight matrix along the last dimension.
    output_size_per_partition = param.shape[1]
    tensor = tensor_value[
        :,
        (rank * output_size_per_partition) : ((rank + 1) * output_size_per_partition),
    ]
    param.copy_(tensor, non_blocking=True)


def _copy_if_present(parameter, tensor_value):
    parameter.copy_(tensor_value, non_blocking=True)


def _load_partial_state_dict(
    model: torch.nn.Module,
    state_dict,
    tp_shard: bool,
    rank=0,
    world_size=1,
):
    unused_params = []
    for key, tensor_value in state_dict.items():
        target_module = model
        # Find where to put the weight and decide whether it needs TP'ing
        key_steps = key.split(".")
        prefix = ""
        key_step = 0
        tp_module = None
        while key_step < len(key_steps) - 1:
            try:
                target_module = getattr(target_module, key_steps[key_step])
                if key_step == 0:
                    prefix += key_steps[key_step]
                else:
                    prefix += "." + key_steps[key_step]
                key_step += 1
                if isinstance(target_module, Iterable):
                    target_module = target_module[int(key_steps[key_step])]  # type: ignore[index]
                    prefix += "." + key_steps[key_step]
                    key_step += 1
                if isinstance(target_module, TPModule):
                    tp_module = target_module
            except AttributeError:
                unused_params.append(key)
                break

        # Check if target_module has the Parameter/buffer
        try:
            param = getattr(target_module, key_steps[-1])

            if not tp_shard or tp_module is None:
                _copy_if_present(param, tensor_value)
            elif tp_module is not None:
                if key_steps[-2] in tp_module.list_colwise_weights():
                    _copy_colwise(
                        param,
                        tensor_value,
                        key_steps[-1] == "bias",
                        rank,
                        world_size,
                    )
                if key_steps[-2] in tp_module.list_rowwise_weights():
                    _copy_rowwise(
                        param,
                        tensor_value,
                        key_steps[-1] == "bias",
                        rank,
                        world_size,
                    )
                if key_steps[-2] in tp_module.list_embedding_weights():
                    _copy_embedding(
                        param,
                        tensor_value,
                        rank,
                        world_size,
                    )
        except AttributeError:
            unused_params.append(key)
