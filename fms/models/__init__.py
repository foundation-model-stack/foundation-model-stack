import logging
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from fms import distributed
from fms.distributed.strategy import (
    TensorParallelStrategy,
    UniformModelParallelStrategy,
)
from fms.utils import fusion, serialization


logger = logging.getLogger(__name__)

__models: MutableMapping[str, MutableMapping[str, Callable[[], nn.Module]]] = {}


def register_model(architecture: str, variant: str, factory: Callable[[], nn.Module]):
    """
    Registers a model variant to be made available in the registration API.
    Args:
    architecture: The name of the model architecture, e.g. 'llama'
    variant: A reference for a particular configuration of the architecture,
        e.g. '7b'
    factory: A callable that constructs an instance of the model variant.
    """
    variants: MutableMapping[str, Callable[[], nn.Module]] = {}
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


def __maybe_infer_model_variant(
    architecture: str,
    variant: str,
    model_path: Optional[str],
    source: Optional[str],
    **kwargs,
) -> Tuple[str, str, Optional[str], Optional[str], Dict[str, Any]]:
    """Infer the model variant configuration from different sources, currently only supported sources are hf"""
    extra_kwargs = kwargs

    if architecture in ("hf_pretrained", "hf_configured"):
        from fms.models.hf.utils import _infer_model_configuration  # type: ignore

        logger.info(f"inferring model configuration from {variant}")
        is_hf_pretrained = architecture == "hf_pretrained"
        if is_hf_pretrained:
            if model_path is not None or source is not None:
                raise ValueError(
                    """architecture="hf_pretrained" implies model weights will be downloaded and extracted from hf cache and loaded into the model, therefore model_path and source should not be set"""
                )
            if len(kwargs) > 0:
                logger.warning(
                    f"ignoring the following parameters as a pretrained model with an inferred configuration is being loaded: {list(kwargs.keys())}"
                )

        extra_kwargs = _infer_model_configuration(
            variant, download_weights=is_hf_pretrained
        )
        architecture = extra_kwargs.pop("architecture")
        variant = extra_kwargs.pop("variant")

        if is_hf_pretrained:
            model_path = extra_kwargs.pop("model_path")
            source = "hf"
        else:
            extra_kwargs = {**extra_kwargs, **kwargs}

    return architecture, variant, model_path, source, extra_kwargs


def _get_model_instance(
    architecture: str,
    variant: str,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    extra_args: dict = {},
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
        device_ctx: Union[torch.device, nullcontext] = (
            device if device is not None else nullcontext()
        )
        with device_ctx:
            model = model_factory(**extra_args)
        torch.set_default_dtype(orig)
        return model
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
        layerid = re.sub("[^.]*\\.([0-9]+)\\..*", "\\1", key)
        if layerid != key:
            layers.add(layerid)
    return len(layers)


def _class_hierarchy(clz):
    if clz == object:
        return {clz}
    bases = clz.__bases__
    all = [_class_hierarchy(c) for c in bases]
    result = {clz}
    for classes in all:
        result = result | classes
    return result


def _fsdp_autowrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int):
    if recurse:
        return True
    classes = _class_hierarchy(module.__class__)
    for clz in classes:
        name = str(clz).lower()
        if ("layer" in name or "block" in name) and "layernorm" not in name:
            return True
    return False


def _activation_checkpoint_check_fn(layer):
    for name in layer.__class__.__bases__:
        name = str(name).lower()
        if "block" in name or "layer" in name:
            return True
    return False


def _fsdp_wrap(
    model: nn.Module,
    distributed_strategy: Optional[str],
    device: torch.device,
    rank0: bool,
) -> nn.Module:
    # initializes parameters that are on meta devices
    def init_fn(x: nn.Module):
        if not rank0:
            return x.to_empty(device=device, recurse=False)
        else:
            return x

    # TODO: enable other policies
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    if distributed_strategy == "fsdp":
        dp_strategy = ShardingStrategy.FULL_SHARD
    elif distributed_strategy == "hsdp":
        dp_strategy = ShardingStrategy.HYBRID_SHARD
    elif distributed_strategy == "ddp":
        dp_strategy = ShardingStrategy.NO_SHARD
    else:
        raise KeyError("distributed strategy should be one of fsdp, dpp, or hsdp")

    model = FSDP(
        model,
        param_init_fn=init_fn,
        sync_module_states=True,
        device_id=device.index,
        limit_all_gathers=True,
        auto_wrap_policy=_fsdp_autowrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=dp_strategy,
    )

    wrapper_fn = partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper_fn,
        check_fn=_activation_checkpoint_check_fn,
    )

    return model


def _is_dp(distributed_strategy):
    return distributed_strategy in {"fsdp", "hsdp", "ddp"}


def _validate_unfuse_strategy(extra_args, rank: int = 0):
    """Input checkpoint and output model may be fused or unfused, thus
    support is needed for all 4 possible combinations of fusion.

    For FP16 models, the checkpoint is always fused (converting from
    unfused if needed), then its parameters are copied into a fused model.
    If an unfused output model is desired, `unfuse_strategy` can be set
    to `post` such that the fused model will be unfused as the last
    step of processing.

    For GPTQ, handling an unfused checkpoint requires instantiation of
    an unfused model. This is obtained by setting `unfuse_strategy` to
    `pre`.

    ckpt       target_model   unfuse_strategy
                              FP16     GPTQ
    -----------------------------------------
    fused      fused          None     None
    fused      unfused        post     n/a
    unfused    fused          None     n/a
    unfused    unfused        post     pre
    """

    unfuse = extra_args["unfuse_strategy"]
    if unfuse is not None and unfuse not in ["post", "pre"]:
        raise ValueError(
            f"Unsupported unfuse strategy `{unfuse}`. Choose between [None, `post`, `pre`]"
        )
    if rank == 0:
        model_str = "fused" if unfuse is None else "unfused"
        print(
            f"Output model will use {model_str} projections "
            f"(unfuse_strategy = {unfuse}). "
            "Select a different unfuse_strategy to change this behavior."
        )


def get_model(
    architecture: str,
    variant: str,
    model_path: Optional[str] = None,
    source: Optional[str] = None,
    device_type: str = "cpu",
    data_type: Optional[Union[str, torch.dtype]] = None,
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    group: Optional[ProcessGroup] = None,
    **kwargs,
):
    """
    Load an instance of a model with weights.

    Args:
    architecture: the model architecture, e.g. llama. See
                `models.list_models()`. If hf_pretrained is given, the model architecture will be inferred by the
                model_config associated with the hf model_id_or_path and subsequently the weights will be downloaded and
                loaded into the model. If hf_configured is given, only the model architecture configuration will be and
                no weights will explicitly be loaded unless following the normal model_path logic. Note, if
                hf_pretrained is given and model_path or source are set, an exception will be raised as model loading
                will occur through the hf cache.
    variant: the configuration of the model, e.g. 7b. See
                `models.list_variants(architecture)`. If architecture is given as "hf_pretrained" or "hf_configured",
                the variant will refer to the hf model_id_or_path.
    model_path: the path to the state_dict of weights. If None, don't load.
    device_type: where to load the model
    distributed_strategy: None, 'fsdp', 'hsdp', 'tp', or 'mp'.
    checkpoint_sharding: how the checkpoint files are sharded: None, 'tp',
                'fsdp', or 'layer'. If None, guess based on files.
    source: If the weights in the state dict didn't come from an FMS model,
                `source` specifies which conversion function might be needed.
                See `serialization.list_sources(architecture)`.
    group: ProcessGroup The PG to use for any model distribution
    """

    rank, world_size = distributed.rank_and_world(group)
    local_rank = distributed.local_rank()

    if distributed_strategy is None or distributed_strategy == "":
        if world_size > 1:
            distributed_strategy = "tp"

    if device_type == "cuda":
        device = torch.device(device_type, local_rank)
    else:
        device = torch.device(device_type)

    extra_args = kwargs
    # TODO: streamline this logic
    data_type_parsed: Optional[torch.dtype] = None
    if isinstance(data_type, str):  # convert str to torch.dtype
        try:
            data_type_parsed = getattr(torch, data_type)
        except:
            raise ValueError(f"Data type `{data_type}` is not a supported torch dtype")
        if extra_args.get("linear_config", None) and "gptq" in extra_args[
            "linear_config"
        ].get("linear_type", None):
            # TODO: introduce logger with different log levels?
            print(
                f"[WARNING] data_type {data_type} provided, but GPTQ does not support "
                "casting to custom data type. Will use checkpoint data type instead."
            )
            data_type_parsed = None
    else:
        data_type_parsed = data_type

    is_gptq = (
        extra_args.get("linear_config", None)
        and extra_args["linear_config"].get("linear_type", None) == "gptq"
    )

    hsdp = distributed_strategy == "hsdp"
    fsdp = distributed_strategy == "fsdp"
    ddp = distributed_strategy == "ddp"
    if hsdp or fsdp or ddp:
        if (hsdp and local_rank != 0) or ((fsdp or ddp) and rank != 0):
            initial_device = torch.device("meta")
        else:
            initial_device = torch.device("cpu")
    elif distributed_strategy == "mp":
        initial_device = torch.device("cpu")
    else:
        initial_device = device

    # infer the model architecture and variant if they do not exist yet
    architecture, variant, model_path, source, extra_args = __maybe_infer_model_variant(
        architecture,
        variant,
        model_path,
        source,
        **kwargs,
    )

    lazy_sd: MutableMapping[str, Any] = {}
    if model_path is not None:
        lazy_sd = serialization.load_state_dict(
            model_path,
            source=source,
            distributed_strategy=distributed_strategy,
            checkpoint_sharding=checkpoint_sharding,
            initial_device=initial_device,
            rank=rank,
            world_size=world_size,
        )

    if "distributed_strategy" not in extra_args:
        if distributed_strategy == "tp":
            print("using tensor parallel")
            extra_args["distributed_strategy"] = TensorParallelStrategy(group)
        elif distributed_strategy == "mp":
            print("using model parallel")
            devices = [i for i in range(torch.cuda.device_count())]
            extra_args["distributed_strategy"] = UniformModelParallelStrategy(
                devices, _guess_num_layers(lazy_sd)
            )

    if extra_args.get("unfuse_strategy", None):
        _validate_unfuse_strategy(extra_args, rank)

        # change source for "gptq + pre" (= unfused gptq ckpt into unfused model)
        if is_gptq and extra_args.get("unfuse_strategy") == "pre":
            if source != "hf":  # GPTQ ckpt is always "hf" style
                raise ValueError(
                    f"Expected GPTQ checkpoint of type `hf` but source is `{source}` instead"
                )
            source = "gptq_" + source + "_unfused"

    # Create the model on meta device to allocate weights lazily
    fms_model = _get_model_instance(
        architecture,
        variant,
        dtype=data_type_parsed,
        device=torch.device("meta"),
        extra_args=extra_args,
    )

    # Choose when to wrap and load the model weights based on the combination
    # distribution strategy and checkpoint sharding
    pre_load = (
        distributed_strategy in ["fsdp", "hsdp"] and checkpoint_sharding != "fsdp"
    )

    def model_wrap(model):
        if _is_dp(distributed_strategy):
            return _fsdp_wrap(model, distributed_strategy, device, rank == 0)
        return model

    if not pre_load:
        fms_model = model_wrap(fms_model)

    if len(lazy_sd):
        serialization.load_state_dict_into_model(
            model=fms_model,
            state_dict=lazy_sd,
            architecture=architecture,
            source=source if source is not None else "fms",
            dtype=data_type_parsed,
            distributed_strategy=distributed_strategy,
            checkpoint_sharding=checkpoint_sharding,
            initial_device=initial_device,
            rank=rank,
        )
    else:
        # move from meta device to real device
        if initial_device != torch.device("meta"):
            fms_model.to_empty(device=initial_device)
        # randomly initialize the model (non-gptq models only)
        if hasattr(fms_model, "reset_parameters") and not is_gptq:
            fms_model.reset_parameters()

    if pre_load:
        fms_model = model_wrap(fms_model)

    # Call post-init to take care of post-wrapping/device-mapping initialization
    # Examples include tying weights, init Rope embeddings
    if getattr(fms_model, "post_init", None):
        fms_model.post_init()

    # Make sure any uninitialized tensors are at least moved to device
    # TODO: should we raise a warning? are uninitialized tensors ever acceptable?
    if initial_device != torch.device("meta"):
        fms_model._apply(
            lambda t: torch.empty_like(t, device=initial_device)
            if t.device == torch.device("meta")
            else t
        )

    if extra_args.get("unfuse_strategy", None) == "post":
        fms_model = fusion.apply_unfuse_weights(fms_model)

    return fms_model


from fms.models import gpt_bigcode, llama, mixtral, roberta
