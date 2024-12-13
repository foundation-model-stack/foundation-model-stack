from typing import Any, Mapping, Optional, Set

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.distributed_c10d import ProcessGroup

import fms.triton.pytorch_ops as triton_ops  # registers the PT custom ops
from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.linear import (
    LinearModuleShardingInfo,
    get_all_linear_type_to_sharding_maps,
    get_linear,
    get_linear_type,
)


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
        linear_config: Optional[Mapping[str, Any]] = None,
    ):
        super(FeedForwardBlock, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        self.w1 = get_linear(
            emb_dim,
            self.hidden_dim,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = get_linear(
            self.hidden_dim,
            emb_dim,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.use_bias = use_bias
        self.linear_config = linear_config
        self.linear_type = get_linear_type(linear_config)

    def reset_parameters(self):
        for layer in ["w1", "w2"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=0.02,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def to_tp(self, group: ProcessGroup) -> "TPFeedForwardBlock":
        return TPFeedForwardBlock.import_module(self, group)

    def forward(self, x):
        out = self.a(self.w1(x))
        if self.p_dropout:
            out = self.d(out)
        out = self.w2(out)
        return out


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
        fused: bool = True,
        linear_config: Optional[Mapping[str, Any]] = None,
    ):
        super(GatedLinearUnit, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        self.fused = fused
        self.multiple_of = multiple_of
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        if self.fused:
            self.wg1_fused = get_linear(
                emb_dim,
                2 * self.hidden_dim,
                bias=use_bias,
                linear_config=linear_config,
            )
        else:
            self.w1 = get_linear(
                emb_dim,
                self.hidden_dim,
                bias=use_bias,
                linear_config=linear_config,
            )
            self.wg = get_linear(
                emb_dim,
                self.hidden_dim,
                bias=use_bias,
                linear_config=linear_config,
            )
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = get_linear(
            self.hidden_dim,
            emb_dim,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.use_bias = use_bias
        self.width = emb_dim
        self.grow_factor = hidden_grow_factor
        self.linear_config = linear_config
        self.linear_type = get_linear_type(linear_config)

    def reset_parameters(self):
        layers = ["w2"]
        if self.fused:
            layers.append("wg1_fused")
        else:
            layers.extend(["w1", "wg"])

        for layer in layers:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=0.02,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def to_tp(self, group: ProcessGroup) -> "TPGatedLinearUnit":
        return TPGatedLinearUnit.import_module(self, group)

    def forward(self, x):
        if self.fused:
            out_fused = self.wg1_fused(x)
            world_size = torch.distributed.get_world_size()
            wg, w1 = torch.split(out_fused, [self.hidden_dim//world_size, self.hidden_dim//world_size], dim=2)
            out = self.a(wg) * w1
        else:
            out = self.a(self.wg(x)) * self.w1(x)
        if self.p_dropout:
            out = self.d(out)
        return self.w2(out)

    def _initialize_empty_module(self):
        with torch.device("meta"):
            return GatedLinearUnit(
                self.width,
                self.grow_factor,
                self.multiple_of,
                self.a,
                self.p_dropout,
                self.use_bias,
                fused=False,
            )

    def unfuse_weights(self):
        result = self._initialize_empty_module()
        wg, w1 = torch.split(
            self.wg1_fused.weight, [self.hidden_dim, self.hidden_dim], dim=0
        )
        result.wg.weight = torch.nn.Parameter(wg)
        result.w1.weight = torch.nn.Parameter(w1)
        result.w2.weight = torch.nn.Parameter(self.w2.weight)
        if self.use_bias:
            wg_bias, w1_bias = torch.split(
                self.wg1_fused.bias, [self.hidden_dim, self.hidden_dim], dim=0
            )
            result.wg.bias = torch.nn.Parameter(wg_bias)
            result.w1.bias = torch.nn.Parameter(w1_bias)
            result.w2.bias = torch.nn.Parameter(self.w2.bias)
        return result



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

    def reset_parameters(self):
        for param in ["w13", "w2"]:
            nn.init.trunc_normal_(
                getattr(self, param),
                mean=0.0,
                std=0.02,
            )

    def to_tp(self, group: ProcessGroup) -> "TPConditionalFeedForward":
        return TPConditionalFeedForward.import_module(self, group)

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        # Check constraints.
        assert x.shape[1] == self.w13.shape[2], "Hidden size mismatch"
        assert x.is_contiguous(), "Hidden_states must be contiguous"
        assert self.w13.is_contiguous(), "Expert weights 1 must be contiguous"
        assert self.w2.is_contiguous(), "Expert weights 2 must be contiguous"

        M, D = x.shape
        E, N, _ = self.w13.shape
        _, A = expert_indices.shape

        #  if x.device.type == "cuda":
        ## Triton path

        if expert_indices.numel() <= E:
            padding_size = 16
        else:
            padding_size = 64

        (
            padded_token_ids_per_block,
            expert_block_mapping,
            total_padded_tokens,
        ) = triton_ops.moe_align_block_size(expert_indices, padding_size, E)

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

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.gate.weight,
            mean=0.0,
            std=0.02,
        )

        self.cond_ffn.reset_parameters()

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
        expert_outs = self.cond_ffn(x, expert_indices)
        int_v1 = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        int_v2 = int_v1.view(B, S, self.dim)
        return int_v2
