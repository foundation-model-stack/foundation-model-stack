# mypy: disable-error-code="method-assign,misc"

import torch
import torch._inductor.ir as ir
import torch._inductor.lowering as lowering
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
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


## Fixes for PT 2.2 collectives until PT 2.3 is released


# Fix 1: https://github.com/pytorch/pytorch/issues/121311
def get_volatile_reads_fixed(self):
    inp = self.inputs[0]
    if isinstance(inp, ir._CollectiveKernel):
        # Out-of-place single-output
        return [inp.inputs[0]]
    elif isinstance(inp, ir.MultiOutput):
        # Out-of-place multi-output
        coll = inp.inputs[0]
        if isinstance(coll, ir._CollectiveKernel):
            _, idx = inp.indices[0]
            return [coll.inputs[idx]]
        return []  # e.g. regular FallbackKernel
    else:
        # In-place requires no additional deps handling for volatile
        # reads since the inputs are mutated.
        return []


ir._WaitKernel.get_volatile_reads = get_volatile_reads_fixed

# Fix 2: These are fixed already in nightlies and will be in 2.3
for overload in torch.ops._c10d_functional.all_reduce.overloads():
    other_fn = getattr(torch.ops._c10d_functional.all_reduce, overload)
    if other_fn in lowering.lowerings:
        del lowering.lowerings[other_fn]


@lowering.register_lowering(torch.ops._c10d_functional.all_reduce)
def _all_reduce_fixed(inp, reduce_op, group_name):
    inp = torch.clone(inp)
    ir._CollectiveKernel.create_inplace(
        torch.ops._c10d_functional.all_reduce_.default,
        ir.ExternKernel.require_contiguous(inp),
        reduce_op,
        group_name,
    )
    return inp


for overload in torch.ops._c10d_functional.all_gather_into_tensor.overloads():
    other_fn = getattr(torch.ops._c10d_functional.all_gather_into_tensor, overload)
    if other_fn in lowering.lowerings:
        del lowering.lowerings[other_fn]


@lowering.register_lowering(torch.ops._c10d_functional.all_gather_into_tensor)
def _all_gather_into_tensor(inp, group_size, group_name):
    return ir.TensorBox.create(
        ir._CollectiveKernel.create_out_of_place(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            ir.ExternKernel.require_contiguous(inp),
            group_size,
            group_name,
        )
    )


def _all_gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather the input tensor across model parallel group."""
    world_size = dist.get_world_size()

    if world_size == 1:
        return input_

    # The transposes here are to avoid excessive recompilation due to split()
    # specializing the dimension where the all_gather is happening
    last_dim = input_.dim() - 1
    # Starting PT 2.3, we can go back to funcol.all_gather_tensor
    return (
        torch.ops._c10d_functional.wait_tensor(
            torch.ops._c10d_functional.all_gather_into_tensor(
                input_.transpose(0, last_dim).contiguous(), world_size, "default"
            )
        )
        .transpose(0, last_dim)
        .contiguous()
    )


def _all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    world_size = dist.get_world_size()

    if world_size == 1:
        return input_

    # Starting PT 2.3, we can go back to funcol.all_reduce
    return torch.ops._c10d_functional.wait_tensor(
        torch.ops._c10d_functional.all_reduce(input_, "sum", "default")
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
