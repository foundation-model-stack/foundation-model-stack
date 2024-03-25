from typing import List, Optional

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.distributed_c10d import ProcessGroup

from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.tp import TPModule
from fms.triton import moe_kernel


class FeedForwardBlock(nn.Module):
    """
    A two-layer, symmetric, fully-connected MLP structure.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of input and output vectors.
    hidden_grow_factor : float
        Sets dimensionality of inner latent space (emb_dim * hidden_grow_factor)
    multiple_of : Optional[int]
        Ensure inner latent space is a multiple of this parameter if defined (useful for
        TensorParallel as well as GPU kernel speed)
    activation_fn : nn.Module
        An activation function over torch.FloatTensors applied to inner latent space.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor=4.0,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
    ):
        super(FeedForwardBlock, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        self.w1 = nn.Linear(emb_dim, self.hidden_dim, bias=use_bias)
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = nn.Linear(self.hidden_dim, emb_dim, bias=use_bias)
        self.use_bias = use_bias

    def reset_parameters(self):
        for layer in ["w1", "w2"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=0.02,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def forward(self, x):
        out = self.a(self.w1(x))
        if self.p_dropout:
            out = self.d(out)
        out = self.w2(out)
        return out


class TPFeedForwardBlock(FeedForwardBlock, TPModule):
    """
    A two-layer, symmetric, fully-connected MLP structure with Tensor Parallel support.

    Args
    ----
    Check FeedForwardBlock for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor: float = 4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        rank, world_size = distributed.rank_and_world(group)
        assert (
            hidden_dim % world_size == 0
        ), "Hidden dim must be divisible by world size"
        FeedForwardBlock.__init__(
            self,
            emb_dim,
            hidden_grow_factor / world_size,
            multiple_of,
            activation_fn,
            p_dropout,
            use_bias,
        )
        self.setup_tp(rank, world_size)

    def colwise_param_names(self) -> List[str]:
        return ["w1"]

    def rowwise_param_names(self) -> List[str]:
        return ["w2"]

    @staticmethod
    def import_module(
        ffb: FeedForwardBlock, group: ProcessGroup
    ) -> "TPFeedForwardBlock":
        tp_ffb = TPFeedForwardBlock(
            emb_dim=ffb.w1.in_features,
            hidden_grow_factor=ffb.hidden_dim / ffb.w1.in_features,
            multiple_of=None,
            activation_fn=ffb.a,
            p_dropout=ffb.p_dropout,
            use_bias=ffb.use_bias,
            group=group,
        )
        return tp_ffb

    def forward(self, x):
        x_par = copy_to_tensor_model_parallel_region(x)
        out_par = FeedForwardBlock.forward(self, x_par)
        return reduce_from_tensor_model_parallel_region(out_par)


class GatedLinearUnit(nn.Module):
    """
    A two-point-five-layer, fully-connected gated linear MLP structure (GLU).
    Contains 50% extra params compared to FeedForwardBlock, adjust accordingly.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of input and output vectors.
    hidden_grow_factor : float
        Sets dimensionality of inner latent space (emb_dim * hidden_grow_factor)
    multiple_of : Optional[int]
        Ensure inner latent space is a multiple of this parameter if defined (useful for
        TensorParallel as well as GPU kernel speed)
    activation_fn : nn.Module
        An activation function over torch.FloatTensors applied to inner latent gates.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor: float = 4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
    ):
        super(GatedLinearUnit, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        self.w1 = nn.Linear(emb_dim, self.hidden_dim, bias=use_bias)
        self.wg = nn.Linear(emb_dim, self.hidden_dim, bias=use_bias)
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = nn.Linear(self.hidden_dim, emb_dim, bias=use_bias)
        self.use_bias = use_bias
        self.width = emb_dim
        self.grow_factor = hidden_grow_factor

    def reset_parameters(self):
        for layer in ["w1", "w2", "wg"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=0.02,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def forward(self, x):
        out = self.a(self.wg(x)) * self.w1(x)
        if self.p_dropout:
            out = self.d(out)
        return self.w2(out)


class TPGatedLinearUnit(GatedLinearUnit, TPModule):
    """
    A two-point-five-layer, fully-connected gated linear MLP structure (GLU).
    Contains 50% extra params compared to FeedForwardBlock, adjust accordingly.
    This subclass adds Tensor Parallel support.

    Args
    ----
    Check GatedLinearUnit for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor: float = 4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)

        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert (
            hidden_dim % world_size == 0
        ), "Hidden dim must be divisible by world size"
        GatedLinearUnit.__init__(
            self,
            emb_dim,
            hidden_grow_factor / world_size,
            multiple_of,
            activation_fn,
            p_dropout,
            use_bias,
        )
        self.setup_tp(rank, world_size)

    def colwise_param_names(self) -> List[str]:
        return ["w1", "wg"]

    def rowwise_param_names(self) -> List[str]:
        return ["w2"]

    @staticmethod
    def import_module(glu: GatedLinearUnit, group: ProcessGroup) -> "TPGatedLinearUnit":
        tp_glu = TPGatedLinearUnit(
            emb_dim=glu.width,
            hidden_grow_factor=glu.hidden_dim / glu.width,
            multiple_of=None,
            activation_fn=glu.a,
            p_dropout=glu.p_dropout,
            use_bias=glu.use_bias,
            group=group,
        )

        return tp_glu

    def forward(self, x):
        x_par = copy_to_tensor_model_parallel_region(x)
        out_par = GatedLinearUnit.forward(self, x_par)
        return reduce_from_tensor_model_parallel_region(out_par)


class ConditionalFeedForward(nn.Module):
    """
    This class represents the expert feed forward networks of an MoE FF layer.

    For more information, see the review paper in https://arxiv.org/pdf/2209.01667.pdf

    Args
    ----
    num_experts : int
        The number of expert feed forward networks.
    dim : int
        The embedding dimension for the transformer model.
    intermediate_size : int
        The intermediate size for the expert networks.
    """

    def __init__(self, num_experts: int, dim: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, intermediate_size))
        self.moe_impl = "fms"

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        # if x.shape[0] > 4:
        if self.moe_impl == "fms":
            ## Triton path
            # Check constraints.
            assert x.shape[1] == self.w13.shape[2], "Hidden size mismatch"
            assert x.is_contiguous(), "Hidden_states must be contiguous"
            assert self.w13.is_contiguous(), "Expert weights 1 must be contiguous"
            assert self.w2.is_contiguous(), "Expert weights 2 must be contiguous"

            M, _ = x.shape
            E, N, _ = self.w13.shape

            if expert_indices.numel() <= E:
                padding_size = 16
            else:
                padding_size = 32

            (
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
            ) = torch.ops.moe.align_vllm(expert_indices, padding_size, E)

            x1, x3 = (
                torch.ops.moe.moe_mm(
                    x,
                    self.w13,
                    expert_indices,
                    padded_token_ids_per_block,
                    expert_block_mapping,
                    total_padded_tokens,
                    expert_indices.shape[1],
                    padding_size,
                )
                .view(-1, N)
                .chunk(2, dim=1)
            )
            return torch.ops.moe.moe_mm(
                F.silu(x1) * x3,
                self.w2,
                expert_indices,
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
                1,
                padding_size,
            )
        elif self.moe_impl == "vllm":
            # Check constraints.
            assert x.shape[1] == self.w13.shape[2], "Hidden size mismatch"
            assert x.is_contiguous(), "Hidden_states must be contiguous"
            assert self.w13.is_contiguous(), "Expert weights 1 must be contiguous"
            assert self.w2.is_contiguous(), "Expert weights 2 must be contiguous"

            M, _ = x.shape
            E, N, _ = self.w13.shape

            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            }

            if M <= E:
                config = {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 1,
                }

            (
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
            ) = torch.ops.moe.align_vllm(expert_indices, config["BLOCK_SIZE_M"], E)

            x1, x3 = (
                torch.ops.moe.moe_mm_vllm(
                    x,
                    self.w13,
                    expert_indices,
                    padded_token_ids_per_block,
                    expert_block_mapping,
                    total_padded_tokens,
                    expert_indices.shape[1],
                )
                .view(-1, N)
                .chunk(2, dim=1)
            )

            return torch.ops.moe.moe_mm_vllm(
                F.silu(x1) * x3,
                self.w2,
                expert_indices,
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
                1,
            )

        elif self.moe_impl == "gpt-fast":
            ## Pure Pytorch path
            # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
            w13_weights = self.w13[expert_indices].transpose(-1, -2)  # [T, A, D, D]
            # w3_weights = self.w13[:, self.intermediate_size:][expert_indices].transpose(-1, -2)  # [T, A, D, D]
            w2_weights = self.w2[expert_indices]  # [T, A, D, D]
            x1, x3 = torch.einsum("ti, taio -> tao", x, w13_weights).chunk(2, dim=2)
            # x3 = torch.einsum("ti, taio -> tao", x, w3_weights)
            expert_outs = torch.einsum(
                "tao, taio -> tai", (F.silu(x1) * x3), w2_weights
            )
            return expert_outs
        return x  # Should not hit this


class TPConditionalFeedForward(ConditionalFeedForward, TPModule):
    """
    This class represents the expert feed forward networks of an MoE FF layer.
    This subclass adds TP support.

    Args
    ----
    num_experts : int
        The number of expert feed forward networks.
    dim : int
        The embedding dimension for the transformer model.
    intermediate_size : int
        The intermediate size for the expert networks.
    """

    def __init__(
        self,
        num_experts: int,
        dim: int,
        intermediate_size: int,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)

        assert (
            intermediate_size % world_size == 0
        ), "Intermediate size must be divisible by world size"
        ConditionalFeedForward.__init__(
            self,
            num_experts,
            dim,
            intermediate_size // world_size,
        )
        self.setup_tp(rank, world_size)

    def moe_param_names(self) -> List[str]:
        return ["w13", "w2"]

    @staticmethod
    def import_module(
        cff: ConditionalFeedForward, group: ProcessGroup
    ) -> "TPConditionalFeedForward":
        tp_cff = TPConditionalFeedForward(
            num_experts=cff.num_experts,
            dim=cff.dim,
            intermediate_size=cff.intermediate_size,
            group=group,
        )

        return tp_cff

    def forward(self, x, expert_indices):
        x_par = copy_to_tensor_model_parallel_region(x)
        out_par = ConditionalFeedForward.forward(self, x_par, expert_indices)
        return reduce_from_tensor_model_parallel_region(out_par)


class MOEFeedForward(nn.Module):
    """
    A Sparse Mixture Of Experts (MoE) Feed Forward layer. The output of this layer for a
    given input is determined by the weighted sum of the outputs of a subset of size
    `num_activated_experts` of the `num_experts` expert networks. The weights are given
    by the gating network, then passed through a topK and a softmax filter to make it _sparse_.

    For more information, see the review paper in https://arxiv.org/pdf/2209.01667.pdf

    Args
    ----
    num_experts : int
        The number of expert feed forward networks.
    num_activated_experts : int
        How many experts can be activated at any single time.
    dim : int
        The embedding dimension for the transformer model.
    intermediate_size : int
        The intermediate size for the expert networks.
    """

    def __init__(
        self,
        num_experts: int,
        num_activated_experts: int,
        dim: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(num_experts, dim, intermediate_size)
        self.dim = dim
        self.num_activated_experts = num_activated_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape[:2]
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # x: [T, D]
        scores = self.gate(x)  # [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(
            expert_weights, self.num_activated_experts, dim=-1
        )  # [T, A], [T, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)  # [T, A]
        # Given the balloning memory requirements, only process at most 10 tokens at a time
        # if x.shape[0] > 10:
        #     split_x = x.chunk(x.shape[0] // 10 + 1)
        #     split_ei = expert_indices.chunk(expert_indices.shape[0] // 10 + 1)
        #     expert_outs = torch.cat([self.cond_ffn(x_i, ei_i) for x_i, ei_i in zip(split_x, split_ei)], dim=0)
        # else:
        expert_outs = self.cond_ffn(x, expert_indices)
        # print(expert_outs.shape)
        int_v1 = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        int_v2 = int_v1.view(B, S, self.dim)
        return int_v2
