from collections import ChainMap, OrderedDict
import os
from pathlib import Path
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Tuple, Union
import torch
from fms.modules.attention import TPMultiHeadAttention
from fms.modules.embedding import TPWordEmbedding

from fms.modules.feedforward import TPFeedForwardBlock, TPGatedLinearUnit

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

def _format_from_file(path: Path) -> str:
    if path.suffix == ".pth":
        return "pt"
    elif path.suffix == ".bin":
        return "hf"
    elif path.suffix == ".safetensors":
        return "st"
    else:
        return "unk"

def load_state_dict(
    model_path: Union[str, Path],
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    initial_device: torch.device = torch.device("cpu"),
    checkpoint_format: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[Mapping[str, Any], str]:
    """
    Validates that the file(s) found at a checkpoint path are compatible with
    the intended (possibly distributed) use-case, and returns the state dict
    if needed.

    If model_path is a directory, it'll try to load, in this order, pytorch
    models (.pth format), HF models (.bin format), and safetensors 
    (.safetensors format).

    Args:
    model_path: the path to find the weights. If not set, return None.
    distributed_strategy: the kind of possibly-distributed model in which we
            intend to load these weights. E.g. tp, fsdp, None. Used for
            validation.
    checkpoint_sharding: the sharding format of the checkpoint.
            E.g. layer, tp, fsdp.
    checkpoint_format: how the checkpoint files are saved: None, 'pt',
            'hf', or 'st'. If None, guess based on files.
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
        checkpoints = sorted(model_path.glob("*.pth"))
        if not len(checkpoints):
            checkpoints = sorted(model_path.glob("*.bin"))
        if not len(checkpoints):
            checkpoints = sorted(model_path.glob("*.safetensors"))
    # Check if the requested file format matches the file format
    assert len(checkpoints) > 0, f"Can't find the requested checkpoint data at {model_path}"
    file_format = _format_from_file(checkpoints[0])
    if checkpoint_format != None and checkpoint_format != file_format:
        raise ValueError(f"Requested checkpoint format {checkpoint_format}, but you are loading {file_format} format.")

    if checkpoint_sharding is not None and checkpoint_sharding != "layer":
        assert world_size == len(
            checkpoints
        ), f"Loading a {checkpoint_sharding}-sharded checkpoint with len={len(checkpoints)} but world size is {world_size}"

        checkpoints = [checkpoints[rank]]

    if file_format == "st":
        from safetensors import safe_open

        # For safetensors, delay loading the weights until information about the model
        # is available for sharding
        checkpoint_sds = []
        for ckp in checkpoints:
            with safe_open(ckp, framework="pt", device=initial_device) as ckp_f:
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
        return checkpoint_sds[0], file_format
    else:
        # layer-sharded checkpoints, e.g. as used in HF
        return ChainMap(*checkpoint_sds), file_format


def load_safetensors_checkpoint(model: torch.nn.Module, state_dict: Mapping[str, any], device, rank, world_size):
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
    weights_map = {}
    for weight_name, weight_info in state_dict.items():
        if not weight_info["file"] in weights_map:
            weights_map[weight_info["file"]] = {}
        weights_map[weight_info["file"]][weight_name] = weight_info["orig_key"]

    with torch.no_grad():
        for weights_file, weights_info in weights_map.items():
            with safe_open(weights_file, framework="pt", device=str(device)) as model_weights:
                _load_safetensors_checkpoint_impl(model, model_weights, weights_info, rank, world_size, "")


def _copy_colwise_st(module: torch.nn.Module, weights, weight_name_map, rank, world_size, prefix):
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
        tensor = tensor_slice[(rank * output_size_per_partition) : ((rank + 1) * output_size_per_partition), :]
        module.weight.copy_(tensor, non_blocking=True)
    if module.bias is not None:
        fms_bias_name = prefix + "bias"
        if fms_bias_name in weight_name_map:
            full_orig_name = weight_name_map[fms_bias_name]
            tensor_slice = weights.get_slice(full_orig_name)
            tensor = tensor_slice[(rank * output_size_per_partition) : ((rank + 1) * output_size_per_partition)]
            module.bias.copy_(tensor, non_blocking=True)


def _copy_rowwise_st(module: torch.nn.Module, weights, weight_name_map, rank, world_size, prefix):
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
        tensor = tensor_slice[:, (rank * output_size_per_partition) : ((rank + 1) * output_size_per_partition)]
        module.weight.copy_(tensor, non_blocking=True)
    if module.bias is not None:
        if rank == 0:
            _copy_if_present_st(module.bias, weights, prefix + "bias", weight_name_map)
        else:
            module.bias.zero_()


def _copy_embedding_st(module: torch.nn.Module, weights, weight_name_map, rank, world_size, prefix):
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
        tensor = tensor_slice[:, (rank * output_size_per_partition) : ((rank + 1) * output_size_per_partition)]
        module.weight.copy_(tensor, non_blocking=True)


def _copy_if_present_st(parameter, model_weights, fms_weight_name, weight_name_map):
    if fms_weight_name in weight_name_map:
        full_orig_name = weight_name_map[fms_weight_name]
        parameter.copy_(model_weights.get_tensor(full_orig_name), non_blocking=True)

def _load_safetensors_checkpoint_impl(layer: torch.nn.Module, model_weights, weight_name_map, rank=0, world_size=1, prefix=""):
    # If we're on a leaf module and no TP has happened just copy the weights
    if len(list(layer.children())) == 0:
        for name, parameter in layer.named_parameters():
            full_fms_name = prefix + name
            _copy_if_present_st(parameter, model_weights, full_fms_name, weight_name_map)

    for name, child in layer.named_children():
        if isinstance(child, TPFeedForwardBlock):
            _copy_colwise_st(child.w1, model_weights, weight_name_map, rank, world_size, prefix + name + ".w1.")
            _copy_rowwise_st(child.w2, model_weights, weight_name_map, rank, world_size, prefix + name + ".w2.")
        elif isinstance(child, TPGatedLinearUnit):
            _copy_colwise_st(child.w1, model_weights, weight_name_map, rank, world_size, prefix + name + ".w1.")
            _copy_colwise_st(child.wg, model_weights, weight_name_map, rank, world_size, prefix + name + ".wg.")
            _copy_rowwise_st(child.w2, model_weights, weight_name_map, rank, world_size, prefix + name + ".w2.")
        elif isinstance(child, TPMultiHeadAttention):
            _copy_colwise_st(child.query, model_weights, weight_name_map, rank, world_size, prefix + name + ".query.")
            if child.kvheads == 1:
                _copy_if_present_st(child.key.weight, model_weights, prefix + name + ".key.weight", weight_name_map)
                _copy_if_present_st(child.value.weight, model_weights, prefix + name + ".value.weight", weight_name_map)
                if child.use_bias:
                    _copy_if_present_st(child.key.bias, model_weights, prefix + name + ".key.bias", weight_name_map)
                    _copy_if_present_st(child.value.bias, model_weights, prefix + name + ".value.bias", weight_name_map)
            else:
                _copy_colwise_st(child.key, model_weights, weight_name_map, rank, world_size, prefix + name + ".key.")
                _copy_colwise_st(child.value, model_weights, weight_name_map, rank, world_size, prefix + name + ".value.")
            _copy_rowwise_st(child.dense, model_weights, weight_name_map, rank, world_size, prefix + name + ".dense.")
        # TODO: Implement TPAlibi
        # elif isinstance(child, TPAlibi):
        #     # Divide the weight matrix along the last dimension.
        #     output_size_per_partition = child.nheads // world_size
        #     tensor_slice = model_weights.get_slice(prefix + name + ".scales")
        #     tensor = tensor_slice[:, (rank * output_size_per_partition) : ((rank + 1) * output_size_per_partition)]
        #     child.scales.copy_(tensor, non_blocking=True)
        elif isinstance(child, TPWordEmbedding):
            _copy_embedding_st(child.emb, model_weights, weight_name_map, rank, world_size, prefix + name + ".emb.")
            if child.abs_pos:
                _copy_embedding_st(child.pos_emb, model_weights, weight_name_map, rank, world_size, prefix + name + ".pos_emb.")
            if child.reversible and not child.tie_weights:
                _copy_colwise_st(child.head, model_weights, weight_name_map, rank, world_size, prefix + name + ".head.")
        else:
            _load_safetensors_checkpoint_impl(child, model_weights, weight_name_map, rank, world_size, prefix + name + ".")
