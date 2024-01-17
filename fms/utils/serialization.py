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


def get_safetensors_item(key, file: Path, device: torch.device) -> torch.Tensor:
    from safetensors import safe_open  # type: ignore[import-untyped]

    with torch.no_grad():
        with safe_open(
            file, framework="pt", device=str(device)
        ) as model_weights:  # type: ignore[attr-defined]
            return model_weights.get_tensor(key)


class LazySafetensorsDict(collections.UserDict):
    def set_lazy_tensor(self, key, file, device):
        super().__setitem__(key, lambda: get_safetensors_item(key, file, device))

    def __getitem__(self, key):
        lazy_tensor = super().__getitem__(key)
        if callable(lazy_tensor):
            lazy_tensor = lazy_tensor()
            super().__setitem__(key, lazy_tensor)
        return lazy_tensor


def load_state_dict(
    model_path: Union[str, Path],
    *,
    checkpoint_format: Optional[str] = None,
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    initial_device: torch.device = torch.device("cpu"),
    rank: int = 0,
    world_size: int = 1,
) -> Mapping[str, Any]:
    """
    Validates that the file(s) found at a checkpoint path are compatible with
    the intended (possibly distributed) use-case, and returns a lazy loading
    state dict if possible (some formats may not support that).

    If model_path is a directory, it'll try to load, in this order, pytorch
    models (.pth format), HF models (.bin format), and safetensors
    (.safetensors format), unless checkpoint_format is specified.

    Args:
    model_path: the path to find the weights. If not set, return None.
    checkpoint_format: how the checkpoint files are saved: None, 'pt',
            'hf', or 'st'. If None, guess based on files.
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
    if checkpoint_sharding == "tp" and distributed_strategy not in ["tp"]:
        raise ValueError("TP checkpoints can only be loaded into a TP model")

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

    checkpoint_sds = []
    if checkpoint_format == "st":
        for ckp in checkpoints:
            checkpoint_sds.append(
                load_safetensors_state_dict(
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


def load_safetensors_state_dict(
    checkpoint: Path,
    device: torch.device,
):
    sd = LazySafetensorsDict()

    from safetensors import safe_open  # type: ignore[import-untyped]

    with safe_open(
        checkpoint, framework="pt", device=str(device)
    ) as model_weights:  # type: ignore[attr-defined]
        sd_keys = list(model_weights.keys())
        for key in sd_keys:
            sd.set_lazy_tensor(key, checkpoint, device)
    return sd


class FusableWeightsMissingError(Exception):
    missing_weights: List[str] = []

    def __init__(self, missing_weights):
        self.missing_weights = missing_weights
        super().__init__()


def load_state_dict_into_model(
    model: torch.nn.Module,
    state_dict: Mapping[str, Any],
    architecture: str,
    source: str,
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    initial_device: torch.device = torch.device("cpu"),
    rank: int = 0,
    world_size: int = 0,
) -> None:
    # 1. Get the adapter from checkpoint sd to fms sd
    adapter = _get_adapter(architecture, source)

    # 2. Decide if model needs sharding and how (for now only TP)
    needs_tp_sharding = checkpoint_sharding != "tp" and distributed_strategy == "tp"

    # 3. Iterate over the weights and load them into the model
    used_keys = set()
    sd_keys = state_dict.keys()
    with torch.no_grad():
        for key in sd_keys:
            if key in used_keys:
                continue
            used_keys.add(key)
            try:
                partial_sd = {key: state_dict[key]}
                if partial_sd[key].device != initial_device:
                    partial_sd[key] = partial_sd[key].to(device=initial_device)
                fms_partial_sd = adapter(partial_sd)
            except FusableWeightsMissingError as e:
                for weight in e.missing_weights:
                    partial_sd[weight] = state_dict[weight]
                    if partial_sd[weight].device != initial_device:
                        partial_sd[weight] = partial_sd[weight].to(
                            device=initial_device
                        )
                fms_partial_sd = adapter(partial_sd)
            _load_partial_state_dict(
                model, fms_partial_sd, needs_tp_sharding, rank, world_size
            )
            del partial_sd
            del fms_partial_sd
            del state_dict[key]


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
                if key_steps[-2] in tp_module.colwise_param_names():
                    _copy_colwise(
                        param,
                        tensor_value,
                        key_steps[-1] == "bias",
                        rank,
                        world_size,
                    )
                if key_steps[-2] in tp_module.rowwise_param_names():
                    _copy_rowwise(
                        param,
                        tensor_value,
                        key_steps[-1] == "bias",
                        rank,
                        world_size,
                    )
                if key_steps[-2] in tp_module.embedding_param_names():
                    _copy_embedding(
                        param,
                        tensor_value,
                        rank,
                        world_size,
                    )
        except AttributeError:
            unused_params.append(key)
