# mypy: disable-error-code="method-assign,misc"

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol


def _all_gather(input_: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    """Gather the input tensor across model parallel group."""

    if pg.size() == 1:
        return input_

    # The transposes here are to avoid excessive recompilation due to split()
    # specializing the dimension where the all_gather is happening
    last_dim = input_.dim() - 1
    return (
        funcol.all_gather_tensor(input_.transpose(0, last_dim).contiguous(), 0, pg)
        .transpose(0, last_dim)
        .contiguous()
    )


def _all_reduce(input_: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""

    if pg.size() == 1:
        return input_

    return funcol.all_reduce(input_, "sum", pg)


def _split(input_: torch.Tensor, rank: int, pg: dist.ProcessGroup) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    if pg.size() == 1:
        return input_

    # Split along last dimension.
    # Get the size and dimension.
    last_dim = input_.dim() - 1
    last_dim_size = input_.size()[last_dim] // pg.size()
    # Split.
    input_list = torch.split(input_, last_dim_size, dim=last_dim)

    # Note: torch.split does not create contiguous tensors by default.
    output = input_list[rank].contiguous()

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, pg):
        return input_

    @staticmethod
    def forward(ctx, input_, pg):
        ctx.pg = pg
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce(grad_output, ctx.pg)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, pg):
        return _all_reduce(input_, pg)

    @staticmethod
    def forward(ctx, input_, pg):
        return _all_reduce(input_, pg)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _AllGatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, pg):
        return _all_gather(input_, pg)

    @staticmethod
    def forward(ctx, input_, rank, pg):
        ctx.rank = rank
        ctx.pg = pg
        return _all_gather(input_, pg)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.rank, ctx.pg)


def copy_to_tensor_model_parallel_region(input_, pg: dist.ProcessGroup):
    return _CopyToModelParallelRegion.apply(input_, pg)


def reduce_from_tensor_model_parallel_region(
    input_: torch.Tensor, pg: dist.ProcessGroup
):
    return _ReduceFromModelParallelRegion.apply(input_, pg)


def all_gather_from_tensor_model_parallel_region(
    input_: torch.Tensor, rank: int, pg: dist.ProcessGroup
):
    return _AllGatherFromModelParallelRegion.apply(input_, rank, pg)
