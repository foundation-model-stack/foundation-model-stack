from typing import Any, Tuple

import torch
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


def _all_reduce(input_: torch.Tensor, is_host_dist=False) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    world_size = torch.distributed.get_world_size()

    if world_size == 1:
        return input_

    if is_host_dist:
        orig_device = input_.device
        cpu_inp = input_.to("cpu")
    else:
        cpu_inp = input_
    cpu_inp = distfunc.all_reduce(cpu_inp, "sum", list(range(world_size)))
    if is_host_dist:
        new_inp = cpu_inp.to(orig_device)
    else:
        new_inp = cpu_inp

    return new_inp

def all_gather_tensor(
    self: torch.Tensor,
    gather_dim: int,
    group,
    tag: str = "",
):
    self.contiguous()
    tag, rankset, group_size = distfunc._expand_group(group, tag)
    tensor = torch.ops.c10d_functional.all_gather_into_tensor(self, tag, rankset, group_size)  # type: ignore[attr-defined]
    res = distfunc._maybe_wrap_tensor(tensor)
    distfunc.wait_tensor(res)
    # TODO this should be done inside AsyncCollectiveTensor to delay the wait() call
    if gather_dim != 0:
        res = torch.cat(torch.chunk(res, group_size, dim=0), dim=gather_dim)
    return res


def _all_gather(input_: torch.Tensor, is_host_dist=False) -> torch.Tensor:
    """Gather the input tensor across model parallel group."""
    world_size = torch.distributed.get_world_size()

    if world_size == 1:
        return input_

    if is_host_dist:
        orig_device = input_.device
        cpu_inp = input_.to("cpu")
    else:
        cpu_inp = input_

    last_dim = cpu_inp.dim() - 1
    cpu_inp = all_gather_tensor(cpu_inp, last_dim, list(range(world_size)))

    if is_host_dist:
        new_inp = cpu_inp.to(orig_device)
    else:
        new_inp = cpu_inp
    return new_inp


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
