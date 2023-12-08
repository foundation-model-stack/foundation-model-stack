# mypy: disable-error-code="method-assign,misc"
from typing import Any, List, Tuple

import torch
import torch._dynamo as dynamo
import torch._inductor.codegen.wrapper as inductor_wrapper
import torch._inductor.ir as inductor_ir
import torch._inductor.lowering as inductor_lowering
import torch.distributed._functional_collectives as distfunc
from torch import nn


def apply_colwise_tp(par_mod: nn.Linear, mod: nn.Linear, world_size, rank):
    # Divide the weight matrix along the last dimension.
    output_size_per_partition = mod.out_features // world_size
    with torch.no_grad():
        par_mod.weight.copy_(
            torch.split(mod.weight, output_size_per_partition, dim=0)[rank]
        )
        if par_mod.bias is not None:
            par_mod.bias.copy_(torch.split(mod.bias, output_size_per_partition)[rank])
    # print(f"For rank {rank}, we have the following weights: Base weight {mod.weight} bias {mod.bias}; Par weight {par_mod.weight}, bias {par_mod.bias}")


def apply_rowwise_tp(par_mod: nn.Linear, mod: nn.Linear, world_size, rank):
    # Divide the weight matrix along the last dimension.
    output_size_per_partition = mod.in_features // world_size
    with torch.no_grad():
        par_mod.weight.copy_(
            torch.split(mod.weight, output_size_per_partition, dim=1)[rank]
        )
        if par_mod.bias is not None:
            if rank == 0:
                par_mod.bias.copy_(mod.bias)
            else:
                par_mod.bias.zero_()
    # print(f"For rank {rank}, we have the following weights: Base weight {mod.weight}, bias {mod.bias}; Par weight {par_mod.weight}, bias {par_mod.bias}")


def apply_embedding_tp(par_mod: nn.Embedding, mod: nn.Embedding, world_size, rank):
    # Divide the weight matrix along the last dimension.
    output_size_per_partition = mod.embedding_dim // world_size
    with torch.no_grad():
        par_mod.weight.copy_(
            torch.split(mod.weight, output_size_per_partition, dim=1)[rank]
        )
    # print(f"For rank {rank}, we have the following weights: Base weight {mod.weight} bias {mod.bias}; Par weight {par_mod.weight}, bias {par_mod.bias}")


def _all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    world_size = torch.distributed.get_world_size()

    if world_size == 1:
        return input_

    return distfunc.all_reduce(input_, "sum", list(range(world_size)))


# Fix #1 is porting the code changes in https://github.com/pytorch/pytorch/pull/108811
@classmethod
def wait_create(cls, collective_op: "inductor_ir.TensorBox"):
    collective_op.decide_layout()
    return inductor_ir.Wait(
        layout=inductor_ir.AliasedLayout(collective_op),  # type: ignore[arg-type]
        inputs=[collective_op],
    )


inductor_ir.Wait.create = wait_create  # type: ignore[assignment]

inductor_ir.AllReduce.get_mutation_names = lambda self: [self.inputs[0].get_name()]  # type: ignore[attr-defined]


@classmethod
def all_reduce_create(
    cls,
    x: "inductor_ir.TensorBox",
    reduce_op: str,
    tag: str,
    ranks: List[int],
    group_size: int,
):
    inplace_inputs = cls.wrap_inputs_as_inplace([x])
    layout = inductor_ir.MutationLayout(inplace_inputs[0])

    _ = inductor_ir.AllReduce(
        layout=layout,
        inputs=inplace_inputs,
        constant_args=[tag, ranks, group_size],
        reduce_op=reduce_op,
    )
    return inplace_inputs[0]


inductor_ir.AllReduce.create = all_reduce_create  # type: ignore[assignment]


def wcg_codegen_free(self, buffer):
    name = buffer.get_name()

    # can be freed but not reused
    # TODO: Port this one-line fix to PyTorch
    if isinstance(buffer, (inductor_ir.InputBuffer, inductor_ir.OutputBuffer)):
        self.writeline(self.make_buffer_free(buffer))
        return

    if not self.can_reuse(buffer):
        return
    self.freed.add(name)

    layout = buffer.get_layout()
    if isinstance(layout, (inductor_ir.AliasedLayout, inductor_ir.MultiOutputLayout)):
        self.writeline(self.make_buffer_free(buffer))
        return

    self.writeline(inductor_wrapper.FreeIfNotReusedLine(self, buffer))


inductor_wrapper.WrapperCodeGen.codegen_free = wcg_codegen_free
# End of fix #1


# Fix #2: Asserts + dynamic shapes create graph breaks
# This function is redefined from torch.distributed._functional_collectives.all_gather_tensor
# to remove an assert that creates an extra graph break
def _all_gather_tensor(
    self: torch.Tensor,
    gather_dim: int,
    group: distfunc.RANK_TYPES,
    tag: str = "",
):
    tag, rankset, group_size = distfunc._expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_gather_into_tensor(self, tag, rankset, group_size)  # type: ignore[attr-defined]
    res = distfunc._maybe_wrap_tensor(tensor)
    # TODO this should be done inside AsyncCollectiveTensor to delay the wait() call
    if gather_dim != 0:
        res = torch.cat(torch.chunk(res, group_size, dim=0), dim=gather_dim)
    return res


# Fix #3: Avoid recompiles on batch size for embedding + TP (until https://github.com/pytorch/pytorch/pull/109561 lands)
for overload in torch.ops.c10d_functional.all_gather_into_tensor.overloads():
    other_fn = getattr(torch.ops.c10d_functional.all_gather_into_tensor, overload)
    if other_fn in inductor_lowering.lowerings:
        del inductor_lowering.lowerings[other_fn]


@inductor_lowering.register_lowering(torch.ops.c10d_functional.all_gather_into_tensor)
def all_gather_into_tensor(shard, tag, ranks, group_size):
    return inductor_ir.TensorBox.create(
        inductor_ir.AllGatherIntoTensor.create(
            inductor_ir.ExternKernel.require_contiguous(shard), tag, ranks, group_size
        )
    )


def _all_gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather the input tensor across model parallel group."""
    world_size = torch.distributed.get_world_size()

    if world_size == 1:
        return input_

    # The transposes here are to avoid excessive recompilation due to split()
    # specializing the dimension where the all_gather is happening
    last_dim = input_.dim() - 1
    return (
        _all_gather_tensor(
            input_.transpose(0, last_dim).contiguous(),
            0,
            list(range(world_size)),
        )
        .transpose(0, last_dim)
        .contiguous()
    )


def _split(input_: torch.Tensor, rank, world_size) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    if world_size == 1:
        return input_

    # Split along last dimension.
    # Get the size and dimension.
    last_dim = input_.dim() - 1
    last_dim_size = input_.size()[last_dim] // world_size
    # Split.
    input_list = torch.split(input_, last_dim_size, dim=last_dim)

    # Note: torch.split does not create contiguous tensors by default.
    output = input_list[rank].contiguous()

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _all_reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _all_reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _AllGatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _all_gather(input_)

    @staticmethod
    def forward(ctx, input_, rank, world_size):
        ctx.rank = rank
        ctx.world_size = world_size
        return _all_gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.rank, ctx.world_size)


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def all_gather_from_tensor_model_parallel_region(input_, rank, world_size):
    return _AllGatherFromModelParallelRegion.apply(input_, rank, world_size)
