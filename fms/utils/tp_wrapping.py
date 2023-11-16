from typing import Tuple
from torch import nn
from torch.distributed._device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import Replicate

from fms.modules.attention import MultiHeadAttention, TPMultiHeadAttention
from fms.modules.embedding import TPWordEmbedding, WordEmbedding
from fms.modules.feedforward import (
    FeedForwardBlock,
    GatedLinearUnit,
    TPFeedForwardBlock,
    TPGatedLinearUnit,
)
from fms.modules.positions import Alibi


# this probably belongs somewhere else but can't go in fms.distribtued b/c
# circular dependency.
def _tp_wrapped(module: nn.Module, group: ProcessGroup):
    if isinstance(module, FeedForwardBlock):
        return TPFeedForwardBlock.import_module(module, group)
    elif isinstance(module, GatedLinearUnit):
        return TPGatedLinearUnit.import_module(module, group)
    elif isinstance(module, MultiHeadAttention):
        return TPMultiHeadAttention.import_module(module, group)
    elif isinstance(module, Alibi):
        raise NotImplementedError("TODO: implement TP for Alibi")
        # tp_layer = TPAlibi.import_module(layer, world_size, rank, dtype)
        # setattr(model, name, tp_layer)
    elif isinstance(module, WordEmbedding):
        return TPWordEmbedding.import_module(module, group)
    else:
        return module
    
def _pt_tp_wrapped(module: nn.Module, mesh: DeviceMesh) -> Tuple[nn.Module, bool]:
    if isinstance(module, FeedForwardBlock):
        module.w1 = parallelize_module(module.w1, mesh, ColwiseParallel())
        module.w2 = parallelize_module(module.w2, mesh, RowwiseParallel())
        return module, True
    elif isinstance(module, GatedLinearUnit):
        module.w1 = parallelize_module(module.w1, mesh, ColwiseParallel())
        module.wg = parallelize_module(module.wg, mesh, ColwiseParallel())
        module.w2 = parallelize_module(module.w2, mesh, RowwiseParallel())
        return module, True
    elif isinstance(module, MultiHeadAttention):
        module.query = parallelize_module(module.query, mesh, ColwiseParallel())
        if module.kvheads != 1:
            module.key = parallelize_module(module.key, mesh, ColwiseParallel())
            module.value = parallelize_module(module.value, mesh, ColwiseParallel())
        module.dense = parallelize_module(module.dense, mesh, RowwiseParallel())
        return module, True
    elif isinstance(module, Alibi):
        raise NotImplementedError("TODO: implement TP for Alibi")
        # tp_layer = TPAlibi.import_module(layer, world_size, rank, dtype)
        # setattr(model, name, tp_layer)
    elif isinstance(module, WordEmbedding):
        module.emb = parallelize_module(module.emb, mesh, ColwiseParallel(output_layouts=Replicate()))
        if module.abs_pos:
            module.pos_emb = parallelize_module(module.pos_emb, mesh, ColwiseParallel(output_layouts=Replicate()))
        if module.reversible and not module.tie_weights:
            module.head = parallelize_module(module.head, mesh, ColwiseParallel(output_layouts=Replicate()))
        return module, True
    else:
        return module, False


def apply_tp(model: nn.Module, group: ProcessGroup):
    wrapped = _tp_wrapped(model, group)
    if wrapped is not model:
        return wrapped

    for name, layer in model.named_children():
        tp_layer = apply_tp(layer, group)
        setattr(model, name, tp_layer)
    return model


def pt_apply_tp(model: nn.Module, mesh: DeviceMesh):
    model, is_wrapped = _pt_tp_wrapped(model, mesh)
    if is_wrapped:
        return model

    for name, layer in model.named_children():
        pt_apply_tp(layer, mesh)
    return model
