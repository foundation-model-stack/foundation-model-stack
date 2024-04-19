import collections
import os
import re
from collections import ChainMap
from collections.abc import Iterable
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch

from fms.modules.tp import TPModule


__adapters: MutableMapping[str, MutableMapping[str, Callable[[Mapping], Mapping]]] = {}


def register_adapter(
    architecture: str,
    source: str,
    adapter: Callable[[Mapping], Mapping],
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


def _legacy_attn_unfused_to_fused_weight_conversion(
    name: str, orig_sd: Mapping
) -> Optional[Tuple[str, torch.Tensor]]:
    """
    function which converts unfused fms weights to fused fms weights in the case the model was using the older unfused
    weights (version 0.0.3)

    Args:
        name: str
            current name to convert
        orig_sd: Mapping
            a mapping from a name to a param, in most cases will be a singleton, however when

    Returns:
    Optional[Tuple[str, torch.Tensor]]
        if query/key/value all exist in the given state dict, a tuple of the new fused name as well as weights will be
        returned from this function
        if one of query, key, or value exists in state dict (not all together), a FusableWeightsMissingError will be
        raised with the weights that are missing
        otherwise this function will return None signifying that no preprocessing needed to be done for this name param
    """
    # if we find query/key/value, this means the weights are unfused and therefore need to be fused
    if "attn.query" in name or "attn.key" in name or "attn.value" in name:
        weight_type = name.split(".")[-1]

        unfused_weights = [
            re.sub(
                rf"attn.(query|key|value).{weight_type}",
                f"attn.query.{weight_type}",
                name,
            ),
            re.sub(
                rf"attn.(query|key|value).{weight_type}",
                f"attn.key.{weight_type}",
                name,
            ),
            re.sub(
                rf"attn.(query|key|value).{weight_type}",
                f"attn.value.{weight_type}",
                name,
            ),
        ]

        result = (
            re.sub(
                rf"attn.(query|key|value).{weight_type}",
                f"attn.in_proj.qkv_fused.{weight_type}",
                name,
            ),
            torch.cat([orig_sd[w] for w in unfused_weights], dim=0),
        )

        return result
    return None


def simple_mapping_adapter(
    mappers: List[Callable[[str, Mapping], Optional[Tuple[str, torch.Tensor]]]],
) -> Callable[[Mapping], Mapping]:
    """
    Create a simple adapter mapping using a mapper function. This function will iterate through the items in a
    state_dict and execute the mapper on each key. This function could be used for versioning, where each version
    requires a new mapping from the previous version.

    Parameters
    ----------
    mappers: List[Callable[[str, Mapping], Optional[Tuple[str, torch.Tensor]]]]
        a list of function which given a param key and the original state dict, will optionally return the new name and param
        to be set in the new state dict. If None is returned, no change will be made in the current iteration

    Returns
    -------
    Callable[[Mapping], Mapping]
        the adapter function which gets the new state dict with the items re-mapped
    """

    def mapping_adapter_func(orig_sd):
        new_sd = {}
        for name, param in orig_sd.items():
            for mapper in mappers:
                mapper_out = mapper(name, orig_sd)

                if mapper_out is not None:
                    name, param = mapper_out
                    break

            new_sd[name] = param

        return new_sd

    return mapping_adapter_func


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


def _get_safetensors_item(key, file: Path, device: torch.device) -> torch.Tensor:
    from safetensors import safe_open  # type: ignore[import-untyped]

    with torch.no_grad():
        with safe_open(
            file, framework="pt", device=str(device)
        ) as model_weights:  # type: ignore[attr-defined]
            return model_weights.get_tensor(key)


class LazySafetensorsDict(collections.UserDict):
    def set_lazy_tensor(self, key, file, device):
        super().__setitem__(key, lambda: _get_safetensors_item(key, file, device))

    def __getitem__(self, key):
        lazy_tensor = super().__getitem__(key)
        if callable(lazy_tensor):
            lazy_tensor = lazy_tensor()
            super().__setitem__(key, lazy_tensor)
        return lazy_tensor


def load_state_dict(
    model_path: Union[str, Path],
    *,
    source: Optional[str] = None,
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    initial_device: torch.device = torch.device("cpu"),
    rank: int = 0,
    world_size: int = 1,
) -> MutableMapping[str, Any]:
    """
    Validates that the file(s) found at a checkpoint path are compatible with
    the intended (possibly distributed) use-case, and returns a lazy loading
    state dict if possible (some formats may not support that).

    If model_path is a directory, it'll try to load models based on the source
    (e.g. .bin for HF, .pth for Meta), and, if no source is specified or hasn't
    been registered, it'll try .safetensors, .pth, and .bin.

    Args:
    model_path: the path to find the weights. If not set, return None.
    source: If the weights in the state dict didn't come from an FMS model,
            `source` specifies which conversion function might be needed.
            See `serialization.list_sources(architecture)`
    distributed_strategy: the kind of possibly-distributed model in which we
            intend to load these weights. E.g. tp, fsdp, None. Used for
            validation.
    checkpoint_sharding: the sharding format of the checkpoint.
            E.g. layer, tp, fsdp.
    initial_device: where the state dict will be loaded if not lazy.
            If meta, return empty dict.
    """
    if model_path is None or initial_device.type == "meta":
        return {}
    if checkpoint_sharding == "fsdp" and distributed_strategy not in ["fsdp", "hsdp"]:
        raise ValueError(f"FSDP checkpoints can only be loaded into an FSDP model")
    if checkpoint_sharding == "tp" and distributed_strategy != "tp":
        raise ValueError("TP checkpoints can only be loaded into a TP model")

    # Before creating the Path object, check if model_path has a glob pattern
    if isinstance(model_path, str):
        model_path, sep, glob_pattern = model_path.partition("*")
    else:
        sep = ""
        glob_pattern = ""
    glob_pattern = sep + glob_pattern

    model_path = Path(os.path.expanduser(model_path))

    checkpoints = []

    if model_path.is_dir():
        if glob_pattern != "":
            glob_pattern_list = [glob_pattern]
        elif source == "meta":
            glob_pattern_list = ["*.pth", "*.safetensors"]
        elif source == "hf":
            glob_pattern_list = ["*.bin", "*.safetensors"]
        else:
            glob_pattern_list = ["*.safetensors", "*.pth", "*.bin"]
        for glob_pattern_possibility in glob_pattern_list:
            file_list = list(model_path.glob(glob_pattern_possibility))
            if len(file_list) > 0:
                checkpoints = sorted(file_list)
                break

    if model_path.is_file():
        checkpoints = [model_path]

    # Check if we found some files
    assert (
        len(checkpoints) > 0
    ), f"Can't find the requested checkpoint data at {model_path}"

    if checkpoint_sharding is not None and checkpoint_sharding != "layer":
        assert world_size == len(
            checkpoints
        ), f"Loading a {checkpoint_sharding}-sharded checkpoint with len={len(checkpoints)} but world size is {world_size}"

        checkpoints = [checkpoints[rank]]

    # if there's only one checkpoint for fsdp/hsdp, load it only into rank zero
    # and it will be distributed by the FSDP `sync_module_states` parameter
    if checkpoint_sharding is None and distributed_strategy in {"hsdp", "fsdp"}:
        if rank == 0:
            checkpoints = [checkpoints[0]]
        else:
            return {}

    checkpoint_sds = []
    if checkpoints[0].suffix == ".safetensors":
        for ckp in checkpoints:
            checkpoint_sds.append(
                _load_safetensors_state_dict(
                    ckp,
                    initial_device,
                )
            )
    else:
        with torch.no_grad():
            checkpoint_sds = [
                torch.load(str(ckpt_path), mmap=True) for ckpt_path in checkpoints
            ]
    return ChainMap(*checkpoint_sds)


def _load_safetensors_state_dict(
    checkpoint: Path,
    device: torch.device,
):
    sd = LazySafetensorsDict()

    from safetensors import safe_open

    with safe_open(checkpoint, framework="pt", device=str(device)) as model_weights:  # type: ignore[attr-defined]
        sd_keys = list(model_weights.keys())
        for key in sd_keys:
            sd.set_lazy_tensor(key, checkpoint, device)
    return sd


def _find_key_neighbors(key: str, sd_keys: Set[str]):
    # For loading most models that concern us, a good partition is the
    # one used for FSDP units: everything that is in a layer can
    # go together and memory usage will be keep in control
    key_steps = key.split(".")
    prefix = ""
    # Navigate the model tree to find a layer index. If not found,
    # grab that weight alone
    for idx, step in enumerate(key_steps):
        prefix = ".".join(key_steps[: idx + 1])
        if step.isnumeric():
            prefix += "."
            break
    prefix_neighbors = set()
    for key_in_sd in sd_keys:
        if prefix in key_in_sd:
            prefix_neighbors.add(key_in_sd)
    return list(prefix_neighbors)


def load_state_dict_into_model(
    model: torch.nn.Module,
    state_dict: MutableMapping[str, Any],
    architecture: str,
    source: str,
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    initial_device: torch.device = torch.device("cpu"),
    rank: int = 0,
    world_size: int = 0,
) -> None:
    """
    This function loads state_dict into model in the most efficient way possible,
    and it removes all weights that have been used in model from state_dict
    in order to conserve memory.

    Args:
    model: The model where the weights are being loaded.
    state_dict: The dictionary with all the weights. If it has been mmaped
            (for torch.load) or it is an instance of LazySafetensorsDict,
            the weights are loaded lazily from disk.
    architecture: the model architecture, e.g. llama. See `models.list_models()`.
    source: If the weights in the state dict didn't come from an FMS model,
            `source` specifies which conversion function might be needed.
            See `serialization.list_sources(architecture)`
    distributed_strategy: the kind of possibly-distributed model in which we
            intend to load these weights. E.g. tp, fsdp, None. Used for weight
            sharding.
    checkpoint_sharding: the sharding format of the checkpoint.
            E.g. layer, tp, fsdp. Used for weight sharding.
    initial_device: where the weights will be loaded from disk.
    """

    # 1. Get the adapter from checkpoint sd to fms sd
    adapter = _get_adapter(architecture, source)

    # 2. Decide if model needs sharding and how (for now only TP)
    needs_tp_sharding = checkpoint_sharding != "tp" and distributed_strategy == "tp"

    # 3. Iterate over the weights and load them into the model
    used_keys = set()
    sd_keys = set(state_dict.keys())

    with torch.no_grad():
        for key in sd_keys:
            if key in used_keys:
                continue
            used_keys.add(key)

            partial_sd = {key: state_dict[key]}
            # Find neighbors to the key. If the adapter requires a neighbor and
            # this function doesn't find it, it will crash.
            remaining_keys = sd_keys.difference(used_keys)
            neighbors = _find_key_neighbors(key, remaining_keys)
            for neighbor in neighbors:
                partial_sd[neighbor] = state_dict[neighbor]
                used_keys.add(neighbor)
            for psd_key in partial_sd.keys():
                if partial_sd[psd_key].device != initial_device:
                    partial_sd[psd_key] = partial_sd[psd_key].to(device=initial_device)
            fms_partial_sd = adapter(partial_sd)
            _load_partial_state_dict(
                model, fms_partial_sd, needs_tp_sharding, rank, world_size
            )
            # Be aggressive in removing weights to save as much memory as possible
            for p_key in partial_sd.keys():
                if isinstance(state_dict, ChainMap):
                    for child_sd in state_dict.maps:
                        child_sd.pop(p_key, None)
                else:
                    state_dict.pop(p_key)
            del partial_sd
            del fms_partial_sd


def _copy_colwise(
    param: torch.nn.Parameter,
    tensor_value,
    is_bias,
    max_partition_sizes,
    rank,
    world_size,
):
    """
    This function copies the correct shard of the weights for a colwise-TP'd module
    according to the rank of the process and the world_size.

    Args
    ====
    param: torch.nn.Parameter
        Parameter that has had TP applied
    tensor_value: torch.Tensor
        tensor that needs sharding
    max_partition_sizes: List[int]
        for each number in the list, if world_size is smaller than or equal to that number, the tensor will get
        partitioned in worldsize parts, else if world size is larger than the number then you will get world size parts
        replicated in worldsize / number

        world_size = 4, max_partition_sizes = [8], tensor = [0 1 2 3 4 5 6 7]
        [0 1] [2 3] [4 5] [6 7]

        world_size = 8, max_partition_sizes = [4], tensor = [0 1 2 3 4 5 6 7]
        [0 1] [0 1] [2 3] [2 3] [4 5] [4 5] [6 7] [6 7]

        If there are multiple numbers in the max_partition_sizes list, then the param gets filled with non-contiguous
        slices of the tensor_value. This is useful for fused weight cases (qkv, mlp, moe, etc.)

        world_size = 4, max_partition_sizes = [4, 4], tensor = [0 1 2 3 4 5 6 7]
        [0 4] [1 5] [2 6] [3 7]

        world_size = 4, max_partition_sizes = [4, 1], tensor = [0 1 2 3 4 5 6 7 8 9]
        [0 1 8 9] [2 3 8 9] [4 5 8 9] [6 7 8 9]

    is_bias: bool
        if True is bias, else is weight
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    """
    # Divide the weight matrix along the first dimension.
    cusum_max_partition_sizes = [0]
    min_partition_size = min(max_partition_sizes)
    for m in max_partition_sizes:
        cusum_max_partition_sizes.append(
            cusum_max_partition_sizes[-1] + (m // min_partition_size)
        )

    output_size_per_partition = param.shape[0] // (
        sum(max_partition_sizes) // min(max_partition_sizes)
    )
    if not is_bias:
        tensor = torch.cat(
            [
                tensor_value[
                    (rank // max(1, world_size // replication))
                    * (output_size_per_partition * (max_partition_sizes[i] // min_partition_size))
                    + (
                        cusum_max_partition_sizes[i]
                        * tensor_value.shape[0]
                        // cusum_max_partition_sizes[-1]
                    ) : (
                        ((rank // max(1, world_size // replication)) + 1)
                        * (output_size_per_partition * (max_partition_sizes[i] // min_partition_size))
                        + (
                            cusum_max_partition_sizes[i]
                            * tensor_value.shape[0]
                            // cusum_max_partition_sizes[-1]
                        )
                    ),
                    :,
                ]
                for i, replication in enumerate(max_partition_sizes)
            ],
            dim=0,
        )
    else:
        tensor = torch.cat(
            [
                tensor_value[
                    (rank // max(1, world_size // replication))
                    * (output_size_per_partition * (max_partition_sizes[i] // min_partition_size))
                    + (
                        cusum_max_partition_sizes[i]
                        * tensor_value.shape[0]
                        // cusum_max_partition_sizes[-1]
                    ) : (
                        ((rank // max(1, world_size // replication)) + 1)
                        * (output_size_per_partition * (max_partition_sizes[i] // min_partition_size))
                        + (
                            cusum_max_partition_sizes[i]
                            * tensor_value.shape[0]
                            // cusum_max_partition_sizes[-1]
                        )
                    )
                ]
                for i, replication in enumerate(max_partition_sizes)
            ],
            dim=0,
        )
    param.copy_(tensor, non_blocking=True)


def _copy_rowwise(
    param: torch.nn.Parameter,
    tensor_value,
    is_bias,
    max_partition_sizes,
    rank,
    world_size,
):
    """
    This function copies the correct shard of the weights for a rowwise-TP'd module
    according to the rank of the process and the world_size.

    Args
    ====
    param: torch.nn.Parameter
        Parameter that has had TP applied
    tensor_value: torch.Tensor
        tensor that needs sharding
    max_partition_sizes: List[int]
        for each number in the list, if world_size is smaller than or equal to that number, the tensor will get
        partitioned in worldsize parts, else if world size is larger than the number then you will get world size parts
        replicated in worldsize / number

        world_size = 4, max_partition_sizes = [8], tensor = [0 1 2 3 4 5 6 7]
        [0 1] [2 3] [4 5] [6 7]

        world_size = 8, max_partition_sizes = [4], tensor = [0 1 2 3 4 5 6 7]
        [0 1] [0 1] [2 3] [2 3] [4 5] [4 5] [6 7] [6 7]

        If there are multiple numbers in the max_partition_sizes list, then the param gets filled with non-contiguous
        slices of the tensor_value. This is useful for fused weight cases (qkv, mlp, moe, etc.)

        world_size = 4, max_partition_sizes = [4, 4], tensor = [0 1 2 3 4 5 6 7]
        [0 4] [1 5] [2 6] [3 7]

        world_size = 4, max_partition_sizes = [4, 1], tensor = [0 1 2 3 4 5 6 7 8 9]
        [0 1 8 9] [2 3 8 9] [4 5 8 9] [6 7 8 9]

    is_bias: bool
        if True is bias, else is weight
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    """
    # Divide the weight matrix along the first dimension.
    cusum_max_partition_sizes = [0]
    min_partition_size = min(max_partition_sizes)
    for m in max_partition_sizes:
        cusum_max_partition_sizes.append(
            cusum_max_partition_sizes[-1] + (m // min_partition_size)
        )

    output_size_per_partition = param.shape[1] // (
        sum(max_partition_sizes) // min(max_partition_sizes)
    )
    if not is_bias:
        tensor = torch.cat(
            [
                tensor_value[
                    :,
                    (rank // max(1, world_size // replication))
                    * (output_size_per_partition * (max_partition_sizes[i] // min_partition_size))
                    + (
                        cusum_max_partition_sizes[i]
                        * tensor_value.shape[0]
                        // cusum_max_partition_sizes[-1]
                    ) : (
                        ((rank // max(1, world_size // replication)) + 1)
                        * (output_size_per_partition * (max_partition_sizes[i] // min_partition_size))
                        + (
                            cusum_max_partition_sizes[i]
                            * tensor_value.shape[0]
                            // cusum_max_partition_sizes[-1]
                        )
                    ),
                ]
                for i, replication in enumerate(max_partition_sizes)
            ],
            dim=1,
        )
        param.copy_(tensor, non_blocking=True)
    else:
        if rank == 0:
            _copy_if_present(param, tensor_value)
        else:
            param.zero_()


def _copy_embedding(
    param: torch.nn.Parameter, tensor_value, max_partition_sizes, rank, world_size
):
    """
    This function copies the correct shard of the weights for a TP'd embedding module
    according to the rank of the process and the world_size.

    Args
    ====
    param: torch.nn.Parameter
        Parameter that has had TP applied
    tensor_value: torch.Tensor
        tensor that needs sharding
    max_partition_sizes: List[int]
        for each number in the list, if world_size is smaller than or equal to that number, the tensor will get
        partitioned in worldsize parts, else if world size is larger than the number then you will get world size parts
        replicated in worldsize / number

        world_size = 4, max_partition_sizes = [8], tensor = [0 1 2 3 4 5 6 7]
        [0 1] [2 3] [4 5] [6 7]

        world_size = 8, max_partition_sizes = [4], tensor = [0 1 2 3 4 5 6 7]
        [0 1] [0 1] [2 3] [2 3] [4 5] [4 5] [6 7] [6 7]

        If there are multiple numbers in the max_partition_sizes list, then the param gets filled with non-contiguous
        slices of the tensor_value. This is useful for fused weight cases (qkv, mlp, moe, etc.)

        world_size = 4, max_partition_sizes = [4, 4], tensor = [0 1 2 3 4 5 6 7]
        [0 4] [1 5] [2 6] [3 7]

        world_size = 4, max_partition_sizes = [4, 1], tensor = [0 1 2 3 4 5 6 7 8 9]
        [0 1 8 9] [2 3 8 9] [4 5 8 9] [6 7 8 9]

    is_bias: bool
        if True is bias, else is weight
    rank: int
        Rank of the current process
    world_size: int
        Total number of TP processes
    """
    # Divide the weight matrix along the first dimension.
    cusum_max_partition_sizes = [0]
    min_partition_size = min(max_partition_sizes)
    for m in max_partition_sizes:
        cusum_max_partition_sizes.append(
            cusum_max_partition_sizes[-1] + (m // min_partition_size)
        )

    output_size_per_partition = param.shape[1] // (
        sum(max_partition_sizes) // min(max_partition_sizes)
    )
    tensor = torch.cat(
        [
            tensor_value[
                :,
                (rank // max(1, world_size // replication)) * output_size_per_partition
                + (
                    cusum_max_partition_sizes[i]
                    * tensor_value.shape[0]
                    // cusum_max_partition_sizes[-1]
                ) : (
                    ((rank // max(1, world_size // replication)) + 1)
                    * (output_size_per_partition * (max_partition_sizes[i] // min_partition_size))
                    + (
                        cusum_max_partition_sizes[i]
                        * tensor_value.shape[0]
                        // cusum_max_partition_sizes[-1]
                    )
                ),
            ]
            for i, replication in enumerate(max_partition_sizes)
        ],
        dim=1,
    )
    param.copy_(tensor, non_blocking=True)


def _copy_if_present(parameter, tensor_value):
    parameter.copy_(tensor_value, non_blocking=True)


def _load_partial_state_dict(
    model: torch.nn.Module,
    state_dict,
    needs_tp_sharding: bool,
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
        # Navigate the model tree to find the module where the parameter is
        # located and whether there is a TPModule in the way in case the
        # parameter requires sharding
        while key_step < len(key_steps) - 1:
            try:
                target_module = getattr(target_module, key_steps[key_step])
                if key_step > 0:
                    prefix += "."
                prefix += key_steps[key_step]
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

            # If TP sharding is not needed, copy the parameter
            # into the model
            if not needs_tp_sharding or tp_module is None:
                _copy_if_present(param, tensor_value)
            elif tp_module is not None:
                # Handle TP sharding
                if key_steps[-2] in tp_module.colwise_params():
                    _copy_colwise(
                        param,
                        tensor_value,
                        key_steps[-1] == "bias",
                        tp_module.colwise_params()[key_steps[-2]],
                        rank,
                        world_size,
                    )
                if key_steps[-2] in tp_module.rowwise_params():
                    _copy_rowwise(
                        param,
                        tensor_value,
                        key_steps[-1] == "bias",
                        tp_module.rowwise_params()[key_steps[-2]],
                        rank,
                        world_size,
                    )
                if key_steps[-2] in tp_module.embedding_params():
                    _copy_embedding(
                        param,
                        tensor_value,
                        tp_module.embedding_params()[key_steps[-2]],
                        rank,
                        world_size,
                    )
        except AttributeError:
            unused_params.append(key)
