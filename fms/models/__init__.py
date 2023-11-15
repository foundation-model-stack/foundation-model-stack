from contextlib import nullcontext
from typing import Callable, Optional
import torch
from torch import nn
from fms import distributed
from fms.distributed.strategy import (
    TensorParallelStrategy,
    UniformModelParallelStrategy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.distributed_c10d import ProcessGroup
from fms.utils import serialization

__models = {}


def register_model(architecture: str, variant: str, factory: Callable):
    """
    Registers a model variant to be made available in the registration API.
    Args:
    architecture: The name of the model architecture, e.g. 'llama'
    variant: A reference for a particular configuration of the architecture,
        e.g. '7b'
    factory: A callable that constructs an instance of the model variant.
    """
    variants = {}
    if architecture in __models:
        variants = __models[architecture]
    if variant in variants:
        raise KeyError(
            f"Variant {variant} already registered for architecture {architecture}"
        )
    variants[variant] = factory
    __models[architecture] = variants


def list_models():
    """
    Lists registered model architectures.
    """
    return list(__models.keys())


def list_variants(architecture: str):
    """
    Lists available variants (configurations) of a model architecture.
    E.g. `models.list_variants('llama')` -> ['micro', '7b', '13b', '70b']
    Args:
    architecture: one of the registered architectures returned by `list_models()`.
    """
    if architecture not in __models:
        raise KeyError(
            f"{architecture} is not registered. See `models.list_models()` for available architectures"
        )
    return list(__models[architecture].keys())


def _get_model_instance(
    architecture: str, variant: str, *, dtype=None, device=None, extra_args: dict = {}
) -> nn.Module:
    """
    Gets a model by name and variant, e.g. `models.get_model('llama', '7b')`
    Does not load weights.
    See public API `models.get_model()`
    Args:
    architecture: one of the architectures from list_models(). E.g. llama.
    variant: one of the variants from list_variants(architecture). E.g. '7b'
    extra_args: kwargs to be passed to the model factory.
    """
    if architecture not in __models:
        raise KeyError(
            f"{architecture} is not registered. See `models.list_models()` for available architectures"
        )
    if variant not in __models[architecture]:
        raise KeyError(
            f'{variant} is not a registered variant of {architecture}. See `models.list_variants("{architecture}")` for available variants.'
        )

    model_factory = __models[architecture][variant]

    orig = torch.get_default_dtype()

    try:
        if dtype is not None:
            torch.set_default_dtype(dtype)
        with device if device is not None else nullcontext():
            return model_factory(**extra_args)
    finally:
        torch.set_default_dtype(orig)


def _guess_num_layers(state_dict):
    """
    This function attempts to guess the number of "layers" in a state_dict by
    looking for lists of sub modules. This can be used to setup model-parallel
    when we don't yet have a model instance.
    """
    if state_dict is None or len(state_dict) == 0:
        raise ValueError(
            "Use model parallel with pre-trained models that have a state dict"
        )

    layers = set()
    import re

    for key in state_dict.keys():
        # when there's a list of layers, layers have numeric IDs in the key
        layerid = re.sub("[^.]*\.([0-9]+)\..*", "\\1", key)
        if layerid != key:
            layers.add(layerid)
    return len(layers)


# TODO: FSDP configuration isn't tested, more of a placeholder to make sure
# it fits the rest of the model-loading paradigm being used here. Will be
# updated along with in-progress tuning script.
def _fsdp_wrap(model: nn.Module, distributed_strategy, local_rank, sync_module_states):
    # initializes parameters that are on meta devices
    def init_fn(x):
        return x.to_empty()

    if distributed_strategy == "fsdp":
        dp_strategy = ShardingStrategy.FULL_SHARD
    elif distributed_strategy == "hsdp":
        dp_strategy = ShardingStrategy.HYBRID_SHARD
    elif distributed_strategy == "ddp":
        dp_strategy = ShardingStrategy.NO_SHARD
    else:
        raise KeyError("distributed strategy was supposed to be one of fsdp or hsdp")

    model = FSDP(
        model,
        param_init_fn=init_fn,
        sync_module_states=sync_module_states,
        device_id=local_rank,
        limit_all_gathers=True,
        use_orig_params=True,
        # TODO: add wrap and mixed precision policies.
        # auto_wrap_policy=wrapping_policy,
        # mixed_precision=mp_policy,
        sharding_strategy=dp_strategy,
    )
    # TODO: add activation checkpointing.
    return model


def _is_dp(distributed_strategy):
    return distributed_strategy in {"fsdp", "hsdp", "ddp"}


def get_model(
    architecture: str,
    variant: str,
    model_path: Optional[str] = None,
    source: Optional[str] = None,
    device_type: str = "cpu",
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    group: Optional[ProcessGroup] = None,
    **kwargs,
):
    """
    Load an instance of a model with weights.

    Args:
    architecture: the model architecture, e.g. llama. See
                `models.list_models()`.
    variant: the configuration of the model, e.g. 7b. See
                `models.list_variants(architecture)`
    model_path: the path to the state_dict of weights. If None, don't load.
    device_type: where to load the model
    distributed_strategy: None, 'fsdp', 'hsdp', 'tp', or 'mp'.
    checkpoint_sharding: how the checkpoint files are sharded: None, 'tp',
                'fsdp', or 'layer'. If None, guess based on files.
    source: If the weights in the state dict didn't come from an FMS model,
                `source` specifies which conversion function might be needed.
                See `serialization.list_sources(architecture)`
    group: ProcessGroup The PG to use for any model distribution
    """
    local_rank, world_size = distributed.rank_and_world(group)

    if distributed_strategy is None or distributed_strategy == "":
        if world_size > 1:
            distributed_strategy = "tp"

    device = torch.device(device_type, local_rank)

    if (
        _is_dp(distributed_strategy)
        and local_rank != 0
        and checkpoint_sharding != "fsdp"
    ):
        initial_device = torch.device("meta")
    elif distributed_strategy == "mp":
        initial_device = torch.device("cpu")
    else:
        initial_device = device

    if model_path is not None:
        fms_sd = serialization.load_state_dict(
            model_path,
            distributed_strategy,
            checkpoint_sharding,
            initial_device,
            local_rank,
            world_size,
        )
        fms_sd = serialization.get_adapted(architecture, source, fms_sd)
    else:
        fms_sd = {}

    extra_args = kwargs
    if "distributed_strategy" not in extra_args:
        if distributed_strategy == "tp":
            print("using tensor parallel")
            extra_args["distributed_strategy"] = TensorParallelStrategy()
        elif distributed_strategy == "mp":
            print("using model parallel")
            devices = [i for i in range(torch.cuda.device_count())]
            extra_args["distributed_strategy"] = UniformModelParallelStrategy(
                devices, _guess_num_layers(fms_sd)
            )

    fms_model = _get_model_instance(
        architecture, variant, device=initial_device, extra_args=extra_args
    )

    # In some cases we'd need to load the sd before distributing, in some cases
    # after.
    # e.g. a checkpoint can be loaded onto rank0 and synced across ranks by
    # FSDP, but in most cases we need to load the rank-specific checkpoint
    # onto the current rank.
    pre_load = (
        distributed_strategy in ["fsdp", "hsdp"] and checkpoint_sharding != "fsdp"
    )

    if pre_load and local_rank == 0 and fms_sd is not None:
        fms_model.load_state_dict(fms_sd, strict=False)

    # post-init distribution
    if _is_dp(distributed_strategy):
        fms_model = _fsdp_wrap(fms_model, distributed_strategy, local_rank, pre_load)

    if not pre_load and len(fms_sd):
        fms_model.load_state_dict(fms_sd, strict=False)

    return fms_model


from fms.models import llama
from fms.models import roberta
