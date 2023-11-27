from collections import ChainMap
import itertools
import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)
import torch
from fms.modules.tp import TPModule

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


def get_ckp_format(model_path: Union[str, Path]) -> str:
    """
    Returns the checkpoint format of a model checkpoint. If format
    is not supported, returns "unk"

    Args:
    model_path: the path to find the weights.
    """
    model_path = Path(os.path.expanduser(model_path))
    if len(sorted(model_path.glob("*.pth"))) > 0:
        return "pt"
    if len(sorted(model_path.glob("*.bin"))) > 0:
        return "hf"
    if len(sorted(model_path.glob("*.safetensors"))) > 0:
        return "st"
    return "unk"


def load_state_dict(
    model_path: Union[str, Path],
    checkpoint_format: str,
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

    If model_path is a directory, it'll try to load, in this order, pytorch
    models (.pth format), HF models (.bin format), and safetensors
    (.safetensors format).

    Args:
    model_path: the path to find the weights. If not set, return None.
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
        return {}  # type: ignore
    # TODO: Add support for tp-sharding a non-sharded state dict for pt, hf formats.
    if checkpoint_sharding == "fsdp" and distributed_strategy not in ["fsdp", "hsdp"]:
        raise ValueError(f"FSDP checkpoints can only be loaded into an FSDP model")

    model_path = Path(os.path.expanduser(model_path))
    if checkpoint_format == "pt":
        glob_pattern = "*.pth"
    elif checkpoint_format == "hf":
        glob_pattern = "*.bin"
    elif checkpoint_format == "st":
        glob_pattern = "*.safetensors"
    else:
        raise ValueError(f"Unsupported checkpoint format {checkpoint_format}")
    checkpoints = sorted(model_path.glob(glob_pattern))

    # Check if the requested file format matches the file format
    assert (
        len(checkpoints) > 0
    ), f"Can't find the requested checkpoint data at {model_path} for format {checkpoint_format}"

    if (
        (distributed_strategy == "tp" or checkpoint_sharding == "tp")
        and distributed_strategy != checkpoint_sharding
        and checkpoint_format != "st"
    ):
        raise ValueError(
            f"TP-sharded models are currently only compatible with TP-sharded checkpoints unless you are using a safetensors checkpoint. Attempting to load {checkpoint_sharding} to {distributed_strategy}"
        )

    if checkpoint_sharding is not None and checkpoint_sharding != "layer":
        assert world_size == len(
            checkpoints
        ), f"Loading a {checkpoint_sharding}-sharded checkpoint with len={len(checkpoints)} but world size is {world_size}"

        checkpoints = [checkpoints[rank]]

    if checkpoint_format == "st":
        from safetensors import safe_open  # type: ignore

        # For safetensors, delay loading the weights until information about the model
        # is available for sharding
        checkpoint_sds = []
        for ckp in checkpoints:
            with safe_open(
                ckp, framework="pt", device=str(initial_device)
            ) as ckp_f:  # type: ignore[attr-defined]
                st_sd = {}
                for key in ckp_f.keys():
                    st_sd[key] = {"file": ckp, "orig_key": key}
                checkpoint_sds.append(st_sd)
    else:
        checkpoint_sds = [
            torch.load(ckpt_path, map_location="cpu") for ckpt_path in checkpoints
        ]
    assert len(checkpoint_sds[0]), f"Unable to load checkpoint data at {model_path}"
    if len(checkpoint_sds) == 1:
        return checkpoint_sds[0]
    else:
        # layer-sharded checkpoints, e.g. as used in HF
        return ChainMap(*checkpoint_sds)


def load_safetensors_checkpoint(
    model: torch.nn.Module, state_dict: Mapping[str, Any], device, rank, world_size
):
    """
    This function loads a safetensors unsharded checkpoint into a model (possibly TP)
    with an arbitrary number of ranks.

    Args
    ====
    model: torch.nn.Module
        Model where the weights will be loaded
    state_dict: Mapping[str, any]
        Dictionary mapping FMS weight names to files and their weight names
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    """
    from safetensors import safe_open

    # First redo state_dict into a better structure for loading
    weights_map: Dict[str, Dict[str, str]] = {}
    for weight_name, weight_info in state_dict.items():
        if not weight_info["file"] in weights_map:
            weights_map[weight_info["file"]] = {}
        weights_map[weight_info["file"]][weight_name] = weight_info["orig_key"]

    with torch.no_grad():
        for weights_file, weights_info in weights_map.items():
            with safe_open(
                weights_file, framework="pt", device=str(device)
            ) as model_weights:  # type: ignore[attr-defined]
                _load_safetensors_checkpoint_impl(
                    model, model_weights, weights_info, rank, world_size, ""
                )


def _copy_colwise_st(
    module: torch.nn.Module, weights, weight_name_map, rank, world_size, prefix
):
    """
    This function copies the correct shard of the weights for a colwise-TP'd module
    according to the rank of the process and the world_size.

    Args
    ====
    module: torch.nn.Module
        Module that has had TP applied
    weights:
        safetensors weights object
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    prefix: str
        Where to find the weight in the weights object
    """
    # Divide the weight matrix along the first dimension.
    output_size_per_partition = module.weight.shape[0]
    fms_weight_name = prefix + "weight"
    if fms_weight_name in weight_name_map:
        full_orig_name = weight_name_map[fms_weight_name]
        tensor_slice = weights.get_slice(full_orig_name)
        tensor = tensor_slice[
            (rank * output_size_per_partition) : (
                (rank + 1) * output_size_per_partition
            ),
            :,
        ]
        module.weight.copy_(tensor, non_blocking=True)
    if module.bias is not None:
        fms_bias_name = prefix + "bias"
        if fms_bias_name in weight_name_map:
            full_orig_name = weight_name_map[fms_bias_name]
            tensor_slice = weights.get_slice(full_orig_name)
            tensor = tensor_slice[
                (rank * output_size_per_partition) : (
                    (rank + 1) * output_size_per_partition
                )
            ]
            module.bias.copy_(tensor, non_blocking=True)


def _copy_rowwise_st(
    module: torch.nn.Module, weights, weight_name_map, rank, world_size, prefix
):
    """
    This function copies the correct shard of the weights for a rowwise-TP'd module
    according to the rank of the process and the world_size.

    Args
    ====
    module: torch.nn.Module
        Module that has had TP applied
    weights:
        safetensors weights object
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    prefix: str
        Where to find the weight in the weights object
    """
    # Divide the weight matrix along the last dimension.
    output_size_per_partition = module.weight.shape[1]
    fms_weight_name = prefix + "weight"
    if fms_weight_name in weight_name_map:
        full_orig_name = weight_name_map[fms_weight_name]
        tensor_slice = weights.get_slice(full_orig_name)
        tensor = tensor_slice[
            :,
            (rank * output_size_per_partition) : (
                (rank + 1) * output_size_per_partition
            ),
        ]
        module.weight.copy_(tensor, non_blocking=True)
    if module.bias is not None:
        if rank == 0:
            _copy_if_present_st(module.bias, weights, prefix + "bias", weight_name_map)
        else:
            module.bias.zero_()


def _copy_embedding_st(
    module: torch.nn.Module, weights, weight_name_map, rank, world_size, prefix
):
    """
    This function copies the correct shard of the weights for a TP'd embedding module
    according to the rank of the process and the world_size.

    Args
    ====
    module: torch.nn.Module
        Module that has had TP applied
    weights:
        safetensors weights object
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    prefix: str
        Where to find the weight in the weights object
    """
    # Divide the weight matrix along the last dimension.
    output_size_per_partition = module.weight.shape[1]
    fms_weight_name = prefix + "weight"
    if fms_weight_name in weight_name_map:
        full_orig_name = weight_name_map[fms_weight_name]
        tensor_slice = weights.get_slice(full_orig_name)
        tensor = tensor_slice[
            :,
            (rank * output_size_per_partition) : (
                (rank + 1) * output_size_per_partition
            ),
        ]
        module.weight.copy_(tensor, non_blocking=True)


def _copy_if_present_st(parameter, model_weights, fms_weight_name, weight_name_map):
    if fms_weight_name in weight_name_map:
        full_orig_name = weight_name_map[fms_weight_name]
        parameter.copy_(model_weights.get_tensor(full_orig_name), non_blocking=True)


def _load_safetensors_checkpoint_impl(
    layer: torch.nn.Module,
    model_weights,
    weight_name_map,
    rank=0,
    world_size=1,
    prefix="",
):
    # If we're on a leaf module and no TP has happened just copy the weights
    if len(list(layer.children())) == 0:
        for name, parameter in layer.named_parameters():
            full_fms_name = prefix + name
            _copy_if_present_st(
                parameter, model_weights, full_fms_name, weight_name_map
            )

    for name, child in layer.named_children():
        if isinstance(child, TPModule):
            for colwise_weight in child.list_colwise_weights():
                _copy_colwise_st(
                    getattr(child, colwise_weight),
                    model_weights,
                    weight_name_map,
                    rank,
                    world_size,
                    f"{prefix}{name}.{colwise_weight}.",
                )
            for rowwise_weight in child.list_rowwise_weights():
                _copy_rowwise_st(
                    getattr(child, rowwise_weight),
                    model_weights,
                    weight_name_map,
                    rank,
                    world_size,
                    f"{prefix}{name}.{rowwise_weight}.",
                )
            for embedding_weight in child.list_embedding_weights():
                _copy_embedding_st(
                    getattr(child, embedding_weight),
                    model_weights,
                    weight_name_map,
                    rank,
                    world_size,
                    f"{prefix}{name}.{embedding_weight}.",
                )
            tp_sharded_modules = list(
                itertools.chain(
                    child.list_colwise_weights(),
                    child.list_rowwise_weights(),
                    child.list_embedding_weights(),
                )
            )
            for mod_name, module in child.named_children():
                if not mod_name in tp_sharded_modules:
                    for param_name, param in module.named_parameters(recurse=False):
                        _copy_if_present_st(
                            param,
                            model_weights,
                            f"{prefix}{name}.{mod_name}.{param_name}",
                            weight_name_map,
                        )
        else:
            _load_safetensors_checkpoint_impl(
                child,
                model_weights,
                weight_name_map,
                rank,
                world_size,
                prefix + name + ".",
            )
